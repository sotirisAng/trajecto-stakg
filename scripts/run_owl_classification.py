"""
OWL 2 RL profile classification using `reasonable` (Rust/DataFrog engine).

Pipeline:
  1. Load ontology modules into rdflib.Graph (OWL/XML → rdflib parse)
  2. SPARQL CONSTRUCT lean ABox from Fuseki (~10K triples)
  3. Merge TBox + ABox into single graph
  4. reasonable.PyReasoner.from_graph() → infer ~50K triples in <1s
  5. Extract inferred rdf:type for profiles:* classes
  6. INSERT DATA to Fuseki named graph <http://w3id.org/trajecto/inferred>

Handles 105 of 117 defined classes natively (OWL 2 RL).
The 12 minQualifiedCardinality classes (focused trajectories, user frequency,
discovery-driven arc) require SPARQL INSERT WHERE post-processing.
"""

import sys
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
from collections import defaultdict

import rdflib
import reasonable

# ── Config ───────────────────────────────────────────────────────────────────

_HELPERS = Path(__file__).resolve().parent.parent / "notebooks"
if str(_HELPERS) not in sys.path:
    sys.path.insert(0, str(_HELPERS))
from helpers.fuseki_config import query_endpoint as FUSEKI_QUERY, update_endpoint as FUSEKI_UPDATE
ONTOLOGY_DIR = Path(__file__).parent.parent / "ontology" / "modules"

ONTOLOGY_FILES = [
    ONTOLOGY_DIR / "poi.ttl",
    ONTOLOGY_DIR / "context.ttl",
    ONTOLOGY_DIR / "patterns.ttl",
    ONTOLOGY_DIR / "purpose.ttl",
    ONTOLOGY_DIR / "spaces.ttl",
    ONTOLOGY_DIR / "spatial.ttl",
    ONTOLOGY_DIR / "user.ttl",
    ONTOLOGY_DIR / "events.ttl",
    ONTOLOGY_DIR / "profiles.ttl",
]

PROPERTY_SHIM_TTL = """
@prefix owl:      <http://www.w3.org/2002/07/owl#> .
@prefix rdfs:     <http://www.w3.org/2000/01/rdf-schema#> .
@prefix trajecto: <http://w3id.org/trajecto#> .
@prefix poi:      <http://w3id.org/trajecto/poi#> .
@prefix datacron: <http://www.datacron-project.eu/datAcron#> .

trajecto:includesInteraction a owl:ObjectProperty ;
    rdfs:domain datacron:Trajectory ;
    rdfs:range  poi:POIInteraction .

trajecto:hasEngagementLevel a owl:ObjectProperty ;
    rdfs:domain datacron:Trajectory .

trajecto:hasPurpose a owl:ObjectProperty ;
    rdfs:domain datacron:Trajectory .

trajecto:hasEvent a owl:ObjectProperty ;
    rdfs:domain datacron:Trajectory ;
    rdfs:range  <http://w3id.org/trajecto/events#TrajectoryEvent> .
"""

INFERRED_GRAPH = "http://w3id.org/trajecto/inferred"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
PROFILES_NS = "http://w3id.org/trajecto/profiles#"
DATACRON_TRAJECTORY = "http://www.datacron-project.eu/datAcron#Trajectory"

CONSTRUCT_QUERY = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX datacron: <http://www.datacron-project.eu/datAcron#>
PREFIX trajecto: <http://w3id.org/trajecto#>
PREFIX patterns: <http://w3id.org/trajecto/patterns#>
PREFIX poi: <http://w3id.org/trajecto/poi#>
PREFIX context: <http://w3id.org/trajecto/context#>
PREFIX user: <http://w3id.org/trajecto/user#>
PREFIX events: <http://w3id.org/trajecto/events#>

CONSTRUCT {
    ?traj rdf:type datacron:Trajectory .
    ?traj trajecto:hasPurpose ?purpose .
    ?traj patterns:exhibitsPattern ?pattern .
    ?traj trajecto:hasTemporalContext ?ctx .
    ?traj trajecto:hasEngagementLevel ?eng .
    ?traj trajecto:includesInteraction ?int .
    ?int rdf:type ?intType .
    ?int poi:interactsWith ?p .
    ?p rdf:type ?poiType .
    ?profile rdf:type user:UserProfile .
    ?profile user:hasTrajectory ?traj .
    ?traj trajecto:hasEvent ?evt .
    ?evt rdf:type ?evtType .
}
WHERE {
    ?traj a datacron:Trajectory .
    OPTIONAL { ?traj trajecto:hasPurpose ?purpose }
    OPTIONAL { ?traj patterns:exhibitsPattern ?pattern }
    OPTIONAL { ?traj trajecto:hasTemporalContext ?ctx }
    OPTIONAL { ?traj trajecto:hasEngagementLevel ?eng }
    OPTIONAL {
        ?traj trajecto:includesInteraction ?int .
        ?int a ?intType .
        FILTER(?intType IN (
            poi:MeaningfulInteraction,
            poi:PassingInteraction,
            poi:ImmersiveInteraction,
            poi:DominantInteraction,
            poi:ExtendedInteraction,
            poi:StartInteraction,
            poi:DestinationInteraction
        ))
        OPTIONAL {
            ?int poi:interactsWith ?p .
            ?p a ?poiType .
            FILTER(STRSTARTS(STR(?poiType), "http://w3id.org/trajecto/poi#"))
        }
    }
    OPTIONAL {
        ?traj trajecto:hasEvent ?evt .
        ?evt a ?evtType .
        FILTER(STRSTARTS(STR(?evtType), "http://w3id.org/trajecto/events#"))
    }
    OPTIONAL {
        ?profile a user:UserProfile .
        ?profile user:hasTrajectory ?traj .
    }
}
"""


# ── SPARQL helpers ───────────────────────────────────────────────────────────

def sparql_construct_ntriples(query: str) -> str:
    data = urllib.parse.urlencode({"query": query}).encode()
    req = urllib.request.Request(
        FUSEKI_QUERY, data=data,
        headers={"Accept": "application/n-triples"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode()


def sparql_select(query: str) -> list:
    data = urllib.parse.urlencode({"query": query}).encode()
    req = urllib.request.Request(
        FUSEKI_QUERY, data=data,
        headers={"Accept": "application/sparql-results+json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    return result["results"]["bindings"]


def sparql_update(update: str):
    data = urllib.parse.urlencode({"update": update}).encode()
    req = urllib.request.Request(FUSEKI_UPDATE, data=data)
    with urllib.request.urlopen(req, timeout=60) as resp:
        resp.read()


# ── Core pipeline ────────────────────────────────────────────────────────────

def load_tbox() -> rdflib.Graph:
    """Load all ontology modules + property shim into rdflib Graph."""
    g = rdflib.Graph()
    for f in ONTOLOGY_FILES:
        if not f.exists():
            print(f"  SKIP (not found): {f.name}")
            continue
        try:
            g.parse(str(f), format="turtle")
            print(f"  OK: {f.name}")
        except Exception as e:
            print(f"  WARN: {f.name} — {e}")
    g.parse(data=PROPERTY_SHIM_TTL, format="turtle")
    print("  Property shim loaded")
    return g


def fetch_abox() -> rdflib.Graph:
    """Fetch lean trajectory-level triples from Fuseki."""
    nt_data = sparql_construct_ntriples(CONSTRUCT_QUERY)
    g = rdflib.Graph()
    g.parse(data=nt_data, format="nt")
    return g


def run_reasoner(tbox: rdflib.Graph, abox: rdflib.Graph) -> list:
    """Run reasonable OWL 2 RL reasoner on merged TBox + ABox."""
    combined = rdflib.Graph()
    for triple in tbox:
        combined.add(triple)
    for triple in abox:
        combined.add(triple)

    r = reasonable.PyReasoner()
    r.from_graph(combined)
    return list(r.reason())


def extract_classifications(inferred: list) -> dict[str, list[str]]:
    """Extract trajectory → [profile classes] from inferred triples."""
    classifications = defaultdict(list)
    for s, p, o in inferred:
        if str(p) == RDF_TYPE and str(o).startswith(PROFILES_NS):
            classifications[str(s)].append(str(o))
    return dict(classifications)


# ── SPARQL fallback for minQualifiedCardinality classes ──────────────────────
# OWL 2 RL does not support minQualifiedCardinality. These 13 classes are
# computed via SPARQL COUNT + HAVING and inserted into the inferred graph.

# 6 focused trajectory classes: ≥3 meaningful interactions with POIs in a group
FOCUSED_CLASSES = {
    "CulturalFocusedTrajectory": [
        "poi:Artwork", "poi:Exhibit", "poi:Gallery", "poi:Museum", "poi:Cultural",
    ],
    "HistoricalFocusedTrajectory": [
        "poi:Historical", "poi:HistoricalLandmark", "poi:HistoricalSite", "poi:Monument",
    ],
    "DiningFocusedTrajectory": [
        "poi:Cafe", "poi:FastFood", "poi:FoodCourt", "poi:Restaurant",
    ],
    "ShoppingFocusedTrajectory": [
        "poi:CommercialStreet", "poi:MarketArea", "poi:RetailStore",
        "poi:ShoppingMall", "poi:Commercial",
    ],
    "EntertainmentFocusedTrajectory": [
        "poi:NightlifeFacility", "poi:Bar", "poi:Nightclub",
        "poi:PerformanceVenue", "poi:Cinema", "poi:Theater", "poi:Entertainment",
    ],
    "RecreationFocusedTrajectory": [
        "poi:LeisureFacility", "poi:SportsFacility", "poi:Beach",
        "poi:Park", "poi:Recreational",
    ],
}

# 6 user frequency classes: user with ≥3 trajectories of a visitor class
FREQUENCY_CLASSES = {
    "FrequentCulturalVisitor": "CulturalVisitor",
    "FrequentHistoricalVisitor": "HistoricalVisitor",
    "FrequentDiningVisitor": "RestaurantVisitor",
    "FrequentShoppingVisitor": "RetailStoreVisitor",
    "FrequentEntertainmentVisitor": "EntertainmentVisitor",
    "FrequentRecreationVisitor": "ParkVisitor",
}


def classify_cardinality_classes() -> dict[str, list[str]]:
    """Compute minQualifiedCardinality classes via SPARQL."""
    extra = defaultdict(list)

    # Focused trajectory classes: trajectory with ≥3 meaningful interactions
    # where the POI type is in the category group
    for class_name, poi_types in FOCUSED_CLASSES.items():
        values = " ".join(poi_types)
        query = f"""
        PREFIX trajecto: <http://w3id.org/trajecto#>
        PREFIX poi: <http://w3id.org/trajecto/poi#>
        PREFIX datacron: <http://www.datacron-project.eu/datAcron#>
        SELECT ?traj WHERE {{
            ?traj a datacron:Trajectory .
            {{
                SELECT ?traj (COUNT(DISTINCT ?int) AS ?cnt) WHERE {{
                    ?traj trajecto:includesInteraction ?int .
                    ?int a poi:MeaningfulInteraction .
                    ?int poi:interactsWith ?p .
                    ?p a ?poiType .
                    VALUES ?poiType {{ {values} }}
                }}
                GROUP BY ?traj
                HAVING (COUNT(DISTINCT ?int) >= 3)
            }}
        }}
        """
        bindings = sparql_select(query)
        for b in bindings:
            uri = b["traj"]["value"]
            extra[uri].append(f"{PROFILES_NS}{class_name}")

    # User frequency classes: user profile with ≥3 trajectories classified
    # as the corresponding visitor class (reads from inferred graph)
    for class_name, visitor_class in FREQUENCY_CLASSES.items():
        query = f"""
        PREFIX user: <http://w3id.org/trajecto/user#>
        PREFIX profiles: <http://w3id.org/trajecto/profiles#>
        SELECT ?profile WHERE {{
            ?profile a user:UserProfile .
            {{
                SELECT ?profile (COUNT(DISTINCT ?traj) AS ?cnt) WHERE {{
                    ?profile user:hasTrajectory ?traj .
                    GRAPH <{INFERRED_GRAPH}> {{
                        ?traj a profiles:{visitor_class} .
                    }}
                }}
                GROUP BY ?profile
                HAVING (COUNT(DISTINCT ?traj) >= 3)
            }}
        }}
        """
        bindings = sparql_select(query)
        for b in bindings:
            uri = b["profile"]["value"]
            extra[uri].append(f"{PROFILES_NS}{class_name}")

    return dict(extra)


def write_to_fuseki(classifications: dict[str, list[str]], batch_size: int = 500,
                    clear_first: bool = True):
    """Write inferred classifications to Fuseki named graph."""
    if clear_first:
        sparql_update(f"DROP SILENT GRAPH <{INFERRED_GRAPH}>")

    triples = []
    for subject, classes in classifications.items():
        for cls in classes:
            triples.append(f"<{subject}> a <{cls}> .")

    # Batch INSERT DATA
    for i in range(0, len(triples), batch_size):
        batch = triples[i:i + batch_size]
        insert = f"""
        INSERT DATA {{
            GRAPH <{INFERRED_GRAPH}> {{
                {chr(10).join(batch)}
            }}
        }}
        """
        sparql_update(insert)
        print(f"  Inserted batch {i // batch_size + 1} ({len(batch)} triples)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    dry_run = "--dry-run" in sys.argv
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    print("=" * 60)
    print("OWL 2 RL Profile Classification (reasonable)")
    print("=" * 60)

    # Step 1: Load TBox
    print("\n[1] Loading ontology modules...")
    t0 = time.time()
    tbox = load_tbox()
    print(f"  TBox: {len(tbox)} triples ({time.time() - t0:.1f}s)")

    # Step 2: Fetch ABox
    print("\n[2] Fetching trajectory-level triples from Fuseki...")
    t0 = time.time()
    abox = fetch_abox()
    traj_count = sum(1 for _, p, o in abox
                     if str(p) == RDF_TYPE and str(o) == DATACRON_TRAJECTORY)
    print(f"  ABox: {len(abox)} triples, {traj_count} trajectories ({time.time() - t0:.1f}s)")

    if traj_count == 0:
        print("  ERROR: No trajectories found. Is Fuseki running?")
        sys.exit(1)

    # Step 3: Run reasoner
    print("\n[3] Running OWL 2 RL reasoner...")
    t0 = time.time()
    inferred = run_reasoner(tbox, abox)
    dt = time.time() - t0
    print(f"  Inferred {len(inferred)} triples in {dt:.1f}s")

    # Step 4: Extract classifications
    print("\n[4] Extracting profile classifications...")
    classifications = extract_classifications(inferred)

    total_classifications = sum(len(v) for v in classifications.values())
    classified_trajectories = len(classifications)

    # Summarize by class
    class_counts = defaultdict(int)
    for classes in classifications.values():
        for cls in classes:
            class_counts[cls.replace(PROFILES_NS, "")] += 1

    print(f"  Classified trajectories: {classified_trajectories}/{traj_count}")
    print(f"  Total classifications: {total_classifications}")
    print(f"  Unique profile classes: {len(class_counts)}")

    if verbose or dry_run:
        print("\n  Class distribution:")
        for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"    {cls}: {count}")

    # Step 5: Write to Fuseki (reasoner results first)
    if dry_run:
        print(f"\n[5] DRY RUN — skipping Fuseki write ({total_classifications} triples)")
    else:
        print(f"\n[5] Writing {total_classifications} triples to <{INFERRED_GRAPH}>...")
        t0 = time.time()
        write_to_fuseki(classifications)
        print(f"  Done ({time.time() - t0:.1f}s)")

    # Step 6: SPARQL fallback for minQualifiedCardinality classes
    print("\n[6] Computing cardinality classes (SPARQL fallback)...")
    t0 = time.time()
    cardinality_classes = classify_cardinality_classes()
    card_total = sum(len(v) for v in cardinality_classes.values())
    card_by_class = defaultdict(int)
    for classes in cardinality_classes.values():
        for cls in classes:
            card_by_class[cls.replace(PROFILES_NS, "")] += 1
    print(f"  {card_total} classifications across {len(card_by_class)} classes ({time.time() - t0:.1f}s)")
    for cls, count in sorted(card_by_class.items(), key=lambda x: -x[1]):
        print(f"    {cls}: {count}")

    if card_total > 0 and not dry_run:
        print(f"  Writing {card_total} cardinality triples to <{INFERRED_GRAPH}>...")
        write_to_fuseki(cardinality_classes, clear_first=False)

    # Merge counts for summary
    for cls, count in card_by_class.items():
        class_counts[cls] = class_counts.get(cls, 0) + count
    total_classifications += card_total

    print("\n" + "=" * 60)
    print(f"Classification complete: {len(class_counts)} classes, "
          f"{total_classifications} type assertions")


if __name__ == "__main__":
    main()
