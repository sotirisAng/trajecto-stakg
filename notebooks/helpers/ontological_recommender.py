"""
Ontological Recommender — Three-layer trajectory similarity and POI recommendations.

Layer 1: OWL profile classification → shared archetype classes (SPARQL on inferred graph)
Layer 2: Vector fine-ranking within archetype groups (67-dim cosine from semantic_features)
Layer 3: POI extraction + scoring from similar trajectories

Uses Information Content (IC) weighting: rare classes count more than common ones.
"""

import math
import numpy as np
from collections import defaultdict

try:
    from notebooks.helpers.fuseki_config import query_endpoint, update_endpoint
    from notebooks.helpers.fuseki_queries import execute_sparql_query
except ModuleNotFoundError:
    from helpers.fuseki_config import query_endpoint
    from helpers.fuseki_queries import execute_sparql_query

INFERRED_GRAPH = "http://w3id.org/trajecto/inferred"
PROFILES_NS = "http://w3id.org/trajecto/profiles#"

PREFIXES = """
PREFIX rdf:      <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs:     <http://www.w3.org/2000/01/rdf-schema#>
PREFIX trajecto: <http://w3id.org/trajecto#>
PREFIX poi:      <http://w3id.org/trajecto/poi#>
PREFIX datacron: <http://www.datacron-project.eu/datAcron#>
PREFIX profiles: <http://w3id.org/trajecto/profiles#>
PREFIX patterns: <http://w3id.org/trajecto/patterns#>
PREFIX context:  <http://w3id.org/trajecto/context#>
PREFIX user:     <http://w3id.org/trajecto/user#>
PREFIX purpose:  <http://w3id.org/trajecto/purpose#>
"""

# POI types excluded from recommendations (transportation, utility)
EXCLUDED_POI_TYPES = {
    "BusStation", "TrainStation", "MetroStation", "TramStop",
    "ParkingFacility", "GasStation", "Taxi", "Airport",
    "PublicToilet", "ATM", "PostOffice", "PoliceStation",
    "TransportationPOI", "ServicePOI", "BikeRentalStation",
    "Entrance", "PointOfInterest", "SignificantPOI",
    "DiningFacility", "AccommodationPOI", "ViewingPoint",
    "InformationPOI",
}

# Classes too common to be useful for similarity (IC < 0.2)
# These are near-universal and matching on them is noise
LOW_IC_THRESHOLD = 0.2

# Behavioral/event classes — describe trajectory dynamics, not user interests.
# Their IC is dampened in similarity so visitor/domain classes dominate.
_BEHAVIORAL_CLASSES = {
    "CasualToImmersiveArc", "DiscoveryDrivenArc", "EngagementFocusedArc",
    "HasCategoryDiscovery", "HasCategoryTransition",
    "HasDestinationInteraction", "HasDominantEncounter", "HasDominantInteraction",
    "HasEngagementDrop", "HasEngagementEscalation",
    "HasImmersiveInteraction", "HasStartInteraction",
    "HighEngagementTrajectory", "LowEngagementTrajectory",
    "MediumEngagementTrajectory", "ImmersiveDominantTrajectory",
}

# Temporal classes — context, not preference signal.
_TEMPORAL_CLASSES = {
    "AfternoonTrajectory", "EveningTrajectory", "MorningTrajectory",
    "NightTrajectory", "WeekdayTrajectory", "WeekendTrajectory",
    "FallTrajectory", "SpringTrajectory", "SummerTrajectory",
    "WinterTrajectory", "HolidayPeriodTrajectory",
}

# IC dampening factors: behavioral/temporal contribute less to similarity
_CLASS_WEIGHT = 1.0
_BEHAVIORAL_WEIGHT = 0.25
_TEMPORAL_WEIGHT = 0.15


def _effective_ic(cls: str, raw_ic: float) -> float:
    """Apply class-type dampening to an IC weight."""
    if cls in _BEHAVIORAL_CLASSES:
        return raw_ic * _BEHAVIORAL_WEIGHT
    if cls in _TEMPORAL_CLASSES:
        return raw_ic * _TEMPORAL_WEIGHT
    return raw_ic


class OntologicalRecommender:
    """
    Three-layer trajectory similarity and POI recommendation engine.

    Usage:
        rec = OntologicalRecommender()
        rec.load_classifications()

        # Similarity
        similar = rec.find_similar("http://w3id.org/trajecto#trajectory_123", top_k=10)

        # Recommendations
        recs = rec.recommend_pois("http://w3id.org/trajecto#trajectory_123", top_k=5)
    """

    def __init__(self, endpoint: str = None, semantic_features: dict = None):
        self.endpoint = endpoint or query_endpoint
        self.classifications = {}   # uri → set of class names
        self.class_counts = {}      # class name → count
        self.ic_weights = {}        # class name → IC weight
        self.total_trajectories = 0
        self.semantic_features = semantic_features  # uri → np.array (67-dim)

    # ── Data Loading ─────────────────────────────────────────────────────

    def load_classifications(self):
        """Load all trajectory→profile class mappings from the inferred graph."""
        query = f"""{PREFIXES}
        SELECT ?traj ?class
        FROM <{INFERRED_GRAPH}>
        WHERE {{
            ?traj a ?class .
            FILTER(STRSTARTS(STR(?class), STR(profiles:)))
        }}
        """
        result = execute_sparql_query(self.endpoint, query)
        if not result:
            print("ERROR: Failed to load classifications from inferred graph.")
            return

        self.classifications = defaultdict(set)
        self.class_counts = defaultdict(int)

        for binding in result["results"]["bindings"]:
            traj = binding["traj"]["value"]
            cls = binding["class"]["value"].replace(PROFILES_NS, "")
            self.classifications[traj].add(cls)
            self.class_counts[cls] += 1

        self.classifications = dict(self.classifications)
        self.class_counts = dict(self.class_counts)
        self.total_trajectories = len(self.classifications)

        # Compute IC weights: IC(c) = log(N / count(c))
        # Classes with fewer members carry more information
        self.ic_weights = {}
        for cls, count in self.class_counts.items():
            self.ic_weights[cls] = math.log(self.total_trajectories / count)

        return self

    def set_semantic_features(self, features: dict):
        """Set precomputed semantic feature vectors (uri → np.array)."""
        self.semantic_features = features

    # ── Similarity Computation ───────────────────────────────────────────

    def compute_similarity(self, traj_a: str, traj_b: str) -> dict:
        """
        Compute ontological similarity between two trajectories.

        Returns dict with:
          - score: IC-weighted shared class count (normalized by max possible)
          - shared_classes: list of shared class names
          - shared_weights: {class: IC weight} for shared classes
          - class_a: classes of traj_a
          - class_b: classes of traj_b
        """
        classes_a = self.classifications.get(traj_a, set())
        classes_b = self.classifications.get(traj_b, set())

        # Filter out low-IC classes (near-universal, add noise)
        informative_a = {c for c in classes_a if self.ic_weights.get(c, 0) >= LOW_IC_THRESHOLD}
        informative_b = {c for c in classes_b if self.ic_weights.get(c, 0) >= LOW_IC_THRESHOLD}

        shared = informative_a & informative_b
        if not shared:
            return {"score": 0.0, "shared_classes": [], "shared_weights": {},
                    "class_a": classes_a, "class_b": classes_b}

        shared_weights = {c: _effective_ic(c, self.ic_weights.get(c, 1.0)) for c in shared}
        raw_score = sum(shared_weights.values())

        # Normalize: max possible is sum of effective IC for union of informative classes
        all_classes = informative_a | informative_b
        max_score = sum(_effective_ic(c, self.ic_weights.get(c, 1.0)) for c in all_classes)
        normalized = raw_score / max_score if max_score > 0 else 0.0

        return {
            "score": normalized,
            "raw_score": raw_score,
            "shared_classes": sorted(shared, key=lambda c: -shared_weights[c]),
            "shared_weights": shared_weights,
            "class_a": classes_a,
            "class_b": classes_b,
        }

    def find_similar(self, target_uri: str, top_k: int = 10,
                     min_shared: int = 1, vector_rerank: bool = True) -> list[dict]:
        """
        Find most similar trajectories to target using ontological similarity.

        Visitor/domain classes carry full IC weight. Behavioral and temporal
        classes are dampened so similarity is driven by what users visit,
        not how trajectories behave dynamically.

        Args:
            target_uri: URI of the target trajectory
            top_k: Number of results to return
            min_shared: Minimum number of shared classes to include
            vector_rerank: If True and semantic features available, rerank within
                          ontological tiers using 67-dim cosine similarity

        Returns:
            List of dicts with: uri, score, shared_classes, explanation
        """
        target_classes = self.classifications.get(target_uri, set())
        if not target_classes:
            return []

        # Filter to informative classes only
        target_informative = {c for c in target_classes
                              if self.ic_weights.get(c, 0) >= LOW_IC_THRESHOLD}
        if not target_informative:
            return []

        candidates = []
        for uri, classes in self.classifications.items():
            if uri == target_uri:
                continue
            other_informative = {c for c in classes
                                 if self.ic_weights.get(c, 0) >= LOW_IC_THRESHOLD}
            shared = target_informative & other_informative
            if len(shared) < min_shared:
                continue

            shared_weights = {c: _effective_ic(c, self.ic_weights.get(c, 1.0)) for c in shared}
            raw_score = sum(shared_weights.values())
            all_classes = target_informative | other_informative
            max_score = sum(_effective_ic(c, self.ic_weights.get(c, 1.0)) for c in all_classes)
            normalized = raw_score / max_score if max_score > 0 else 0.0

            candidates.append({
                "uri": uri,
                "score": normalized,
                "raw_score": raw_score,
                "shared_classes": sorted(shared, key=lambda c: -shared_weights[c]),
                "shared_weights": shared_weights,
                "n_shared": len(shared),
            })

        # Sort by IC-weighted score
        candidates.sort(key=lambda x: -x["score"])

        # Vector reranking within ontological tiers
        if vector_rerank and self.semantic_features and target_uri in self.semantic_features:
            candidates = self._vector_rerank(target_uri, candidates, top_k * 3)

        results = candidates[:top_k]

        # Add explanations
        for r in results:
            r["explanation"] = self._build_explanation(target_uri, r)

        return results

    def _vector_rerank(self, target_uri: str, candidates: list, pool_size: int) -> list:
        """Rerank candidates within ontological tiers using vector similarity."""
        target_vec = self.semantic_features.get(target_uri)
        if target_vec is None:
            return candidates

        pool = candidates[:pool_size]
        for c in pool:
            other_vec = self.semantic_features.get(c["uri"])
            if other_vec is not None:
                dot = np.dot(target_vec, other_vec)
                norm_t = np.linalg.norm(target_vec)
                norm_o = np.linalg.norm(other_vec)
                cosine_sim = dot / (norm_t * norm_o) if norm_t > 0 and norm_o > 0 else 0.0
                # Combined: 60% ontological + 40% vector
                c["vector_sim"] = float(cosine_sim)
                c["combined_score"] = 0.6 * c["score"] + 0.4 * cosine_sim
            else:
                c["vector_sim"] = None
                c["combined_score"] = c["score"]

        pool.sort(key=lambda x: -x["combined_score"])
        return pool

    def _build_explanation(self, target_uri: str, result: dict) -> str:
        """Build human-readable explanation for a similarity result."""
        shared = result["shared_classes"]
        top_classes = shared[:3]
        parts = []
        for cls in top_classes:
            weight = result["shared_weights"].get(cls, 0)
            parts.append(f"{cls} (IC={weight:.2f})")
        explanation = f"Shared {len(shared)} archetype classes: {', '.join(parts)}"
        if len(shared) > 3:
            explanation += f" + {len(shared) - 3} more"
        return explanation

    # ── POI Recommendations ──────────────────────────────────────────────

    def recommend_pois(self, target_uri: str, top_k: int = 5,
                       sim_pool: int = 20) -> list[dict]:
        """
        Recommend POIs based on ontological similarity + profile alignment.

        Scoring:
          evidence  = similarity-weighted frequency / total similarity budget
          alignment = profile class → POI type preference match (0–0.5 boost)
          score     = evidence × (1 + alignment), capped at 1.0

        Returns:
            List of dicts: poi_uri, poi_type, score, frequency,
                          explanation (dict), source_trajectories
        """
        # Step 1: Find similar trajectories
        similar = self.find_similar(target_uri, top_k=sim_pool)
        if not similar:
            return []

        sim_uris = [s["uri"] for s in similar]
        sim_lookup = {s["uri"]: s for s in similar}
        sim_scores = {s["uri"]: s.get("combined_score", s["score"]) for s in similar}
        total_sim = sum(sim_scores.values())

        # Profile alignment: infer preferred POI types from archetype classes
        target_classes = self.classifications.get(target_uri, set())
        preferred = _infer_preferred_types(target_classes, self.ic_weights)

        # Step 2: Get POIs the target actually engaged with (Meaningful+)
        # PassingInteraction POIs are valid recommendations — user saw but didn't explore
        target_pois = self._get_trajectory_pois(target_uri)
        target_poi_uris = {p[0] for p in target_pois}

        # Step 3: Get POIs from similar trajectories
        poi_sources = defaultdict(list)
        for sim_uri in sim_uris:
            pois = self._get_trajectory_pois(sim_uri)
            for poi_uri, poi_type, int_type in pois:
                if poi_uri not in target_poi_uris:
                    short_type = poi_type.split("#")[-1] if "#" in poi_type else poi_type
                    if short_type not in EXCLUDED_POI_TYPES:
                        poi_sources[poi_uri].append((sim_uri, poi_type, int_type))

        if not poi_sources:
            return []

        # Step 4: Score POIs
        scored = []
        for poi_uri, sources in poi_sources.items():
            # Deduplicate: best interaction per source trajectory
            traj_best_int = {}
            for sim_uri, poi_type, int_type in sources:
                if (sim_uri not in traj_best_int or
                        INTERACTION_RANK.get(int_type, 0) >
                        INTERACTION_RANK.get(traj_best_int[sim_uri], 0)):
                    traj_best_int[sim_uri] = int_type

            frequency = len(traj_best_int)
            poi_type = sources[0][1].split("#")[-1] if "#" in sources[0][1] else sources[0][1]

            # Evidence: similarity-weighted frequency, normalized
            weighted_freq = sum(sim_scores.get(t, 0) for t in traj_best_int)
            evidence = min(1.0, weighted_freq / total_sim) if total_sim > 0 else 0

            # Profile alignment boost (0–0.5)
            alignment = preferred.get(poi_type, 0.0)

            score = min(1.0, evidence * (1.0 + alignment))

            # Collect shared archetype classes from source trajectories
            driving_classes = defaultdict(int)
            for t_uri in traj_best_int:
                for cls in sim_lookup[t_uri].get("shared_classes", []):
                    driving_classes[cls] += 1
            top_archetypes = sorted(driving_classes, key=lambda c: -driving_classes[c])[:3]

            # Build structured explanation
            matching_profile = [c for c in target_classes
                                if _class_matches_poi_type(c, poi_type)]

            explanation = {
                "archetype_link": top_archetypes,
                "profile_fit": matching_profile[:3] if matching_profile else None,
                "alignment": round(alignment, 2),
                "evidence": round(evidence, 3),
                "confidence": (
                    "strong" if frequency >= 4 and evidence >= 0.3 else
                    "moderate" if frequency >= 2 or evidence >= 0.15 else
                    "weak"
                ),
            }

            scored.append({
                "poi_uri": poi_uri,
                "poi_type": poi_type,
                "score": round(score, 3),
                "frequency": frequency,
                "explanation": explanation,
                "source_trajectories": list(traj_best_int.keys())[:5],
            })

        scored.sort(key=lambda x: -x["score"])
        return scored[:top_k]

    def _get_trajectory_pois(self, traj_uri: str) -> list[tuple]:
        """Get meaningfully-engaged POIs: [(poi_uri, poi_type, interaction_type)].

        Excludes PassingInteraction — only POIs with actual engagement
        are used as recommendation signals from source trajectories.
        """
        query = f"""{PREFIXES}
        SELECT ?poi ?poiType ?intType WHERE {{
            <{traj_uri}> trajecto:includesInteraction ?int .
            ?int a ?intType .
            ?int poi:interactsWith ?poi .
            ?poi a ?poiType .
            FILTER(?intType IN (
                poi:MeaningfulInteraction,
                poi:ImmersiveInteraction, poi:DominantInteraction
            ))
            FILTER(STRSTARTS(STR(?poiType), "http://w3id.org/trajecto/poi#"))
        }}
        """
        result = execute_sparql_query(self.endpoint, query)
        if not result:
            return []

        pois = []
        for b in result["results"]["bindings"]:
            pois.append((
                b["poi"]["value"],
                b["poiType"]["value"],
                b["intType"]["value"].split("#")[-1],
            ))
        return pois

    # ── Batch Similarity ─────────────────────────────────────────────────

    def pairwise_similarity_matrix(self, uris: list[str] = None) -> dict:
        """
        Compute pairwise IC-weighted Jaccard similarity for all trajectories.

        Returns dict with:
          - matrix: np.array (N×N)
          - uris: list of URIs (row/col order)
        """
        uris = uris or sorted(self.classifications.keys())
        n = len(uris)
        matrix = np.zeros((n, n))

        # Only use informative classes (IC >= threshold)
        all_classes = sorted(c for c, w in self.ic_weights.items() if w >= LOW_IC_THRESHOLD)
        class_idx = {c: i for i, c in enumerate(all_classes)}
        ic_vec = np.array([_effective_ic(c, self.ic_weights[c]) for c in all_classes])

        # Build binary membership matrix (informative classes only)
        membership = np.zeros((n, len(all_classes)))
        for i, uri in enumerate(uris):
            for cls in self.classifications.get(uri, set()):
                if cls in class_idx:
                    membership[i, class_idx[cls]] = 1.0

        # Weighted Jaccard via vectorized operations
        for i in range(n):
            for j in range(i + 1, n):
                intersection = membership[i] * membership[j]
                union = np.maximum(membership[i], membership[j])
                num = np.dot(intersection, ic_vec)
                den = np.dot(union, ic_vec)
                sim = num / den if den > 0 else 0.0
                matrix[i, j] = sim
                matrix[j, i] = sim
            matrix[i, i] = 1.0

        return {"matrix": matrix, "uris": uris}

    # ── Diagnostics ──────────────────────────────────────────────────────

    def class_distribution(self) -> list[tuple]:
        """Return (class_name, count, IC_weight) sorted by count desc."""
        return sorted(
            [(c, n, self.ic_weights.get(c, 0))
             for c, n in self.class_counts.items()],
            key=lambda x: -x[1]
        )

    def unclassified_trajectories(self) -> list[str]:
        """Return trajectory URIs with 0 profile classifications."""
        all_trajs = self._get_all_trajectory_uris()
        classified = set(self.classifications.keys())
        return sorted(set(all_trajs) - classified)

    def _get_all_trajectory_uris(self) -> list[str]:
        query = f"""{PREFIXES}
        SELECT ?traj WHERE {{ ?traj a datacron:Trajectory }}
        """
        result = execute_sparql_query(self.endpoint, query)
        if not result:
            return []
        return [b["traj"]["value"] for b in result["results"]["bindings"]]


# ── Utilities ────────────────────────────────────────────────────────────────

INTERACTION_RANK = {
    "ImmersiveInteraction": 4,
    "DominantInteraction": 3,
    "MeaningfulInteraction": 2,
    "PassingInteraction": 1,
}

# ── Profile → POI Type Alignment ────────────────────────────────────────────

# Domain keywords in profile class names → POI types they imply preference for
_DOMAIN_POI_MAP = {
    "Cultural": {"Museum", "Gallery", "Artwork", "Exhibit", "Cultural"},
    "Historical": {"Monument", "HistoricalLandmark", "HistoricalSite", "Historical"},
    "Dining": {"Restaurant", "Cafe", "FastFood", "FoodCourt"},
    "Shopping": {"RetailStore", "ShoppingMall", "MarketArea", "CommercialStreet"},
    "Commercial": {"RetailStore", "ShoppingMall", "MarketArea", "CommercialStreet"},
    "Recreation": {"Park", "Beach", "LeisureFacility", "SportsFacility", "Recreational"},
    "Leisure": {"Park", "Beach", "LeisureFacility", "SportsFacility"},
    "Entertainment": {"Cinema", "Theater", "PerformanceVenue", "NightlifeFacility", "Bar"},
}

# Known POI types extractable from <Type>Visitor class names
_VISITOR_POI_TYPES = {
    "Bar", "Cafe", "Restaurant", "FastFood", "FoodCourt", "Museum", "Gallery",
    "Artwork", "Exhibit", "Monument", "HistoricalLandmark", "HistoricalSite",
    "Park", "Beach", "LeisureFacility", "SportsFacility", "RetailStore",
    "ShoppingMall", "MarketArea", "CommercialStreet", "Cinema", "Theater",
    "PerformanceVenue", "NightlifeFacility", "Nightclub", "Library", "School",
    "University", "Landmark", "Religious", "Accommodation",
}


def _infer_preferred_types(profile_classes: set, ic_weights: dict) -> dict:
    """
    Extract POI type preferences from profile class names.

    Returns {poi_type: alignment_weight} where weight is 0.0–0.5.
    Higher weight = stronger profile alignment signal.
    """
    prefs = defaultdict(float)

    for cls in profile_classes:
        ic = ic_weights.get(cls, 0)
        if ic < LOW_IC_THRESHOLD:
            continue

        # Direct visitor: BarVisitor → Bar, ImmersiveBarVisitor → Bar
        for poi_type in _VISITOR_POI_TYPES:
            if cls == f"{poi_type}Visitor":
                prefs[poi_type] = max(prefs[poi_type], 0.35)
            elif cls == f"Immersive{poi_type}Visitor":
                prefs[poi_type] = max(prefs[poi_type], 0.5)
            elif cls == f"Frequent{poi_type}Visitor":
                prefs[poi_type] = max(prefs[poi_type], 0.4)

        # Domain patterns: CulturalExplorer, DiningFocusedTrajectory, etc.
        for domain, poi_types in _DOMAIN_POI_MAP.items():
            if domain in cls:
                strength = 0.4 if ("Immersive" in cls or "HighEngagement" in cls) else 0.25
                for pt in poi_types:
                    prefs[pt] = max(prefs[pt], strength)

    return dict(prefs)


def _class_matches_poi_type(class_name: str, poi_type: str) -> bool:
    """Check if a profile class name implies interest in a POI type."""
    if poi_type in class_name:
        return True
    for domain, types in _DOMAIN_POI_MAP.items():
        if domain in class_name and poi_type in types:
            return True
    return False
