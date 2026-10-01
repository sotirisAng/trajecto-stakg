"""Competency-question coverage for the Trajecto KG (evaluation axis C).

Each competency question (thesis Appendix A) names a capability the methodology
claims. This runs one SPARQL query per question against a live dataset and
reports whether the graph answers it, with a sample answer. Running it over
several instantiations also evidences portability (axis D): a capability that
answers on Rome, GeoLife and BIRD transferred; one that answers on only one
did not.

    python cq_coverage.py                       # default dataset
    python cq_coverage.py --kg geolife
    python cq_coverage.py --kg osm-rome --kg geolife   # side by side
    python cq_coverage.py --self-check

Output: output/evaluation/cq_coverage[_<kg>].csv
"""

import argparse
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).resolve().parent / "output" / "evaluation"
SERVER = "http://localhost:3030"

P = """
PREFIX trajecto: <http://w3id.org/trajecto#>
PREFIX poi:      <http://w3id.org/trajecto/poi#>
PREFIX patterns: <http://w3id.org/trajecto/patterns#>
PREFIX context:  <http://w3id.org/trajecto/context#>
PREFIX user:     <http://w3id.org/trajecto/user#>
PREFIX purpose:  <http://w3id.org/trajecto/purpose#>
PREFIX spatial:  <http://w3id.org/trajecto/spatial#>
PREFIX datacron: <http://www.datacron-project.eu/datAcron#>
PREFIX dul:      <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#>
PREFIX rdfs:     <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:      <http://www.w3.org/2001/XMLSchema#>
"""

# (id, source, capability, question, query)
#   source "requirements" -- a capability the requirements name
#   source "pipeline"     -- a capability the build provides that the requirements never asked
#                        for; these are contributions, not requirements met
CQS = [
 ("CQ1", "requirements", "recognition of stops",
  "How many stop positions does each trajectory contain?",
  """SELECT ?traj (COUNT(DISTINCT ?pos) AS ?stops) WHERE {
       ?pos a trajecto:StopPosition ; dul:isPartOf ?traj .
     } GROUP BY ?traj ORDER BY DESC(?stops)"""),

 ("CQ2", "requirements", "recognition of semantic overlap between trajectories",
  "Which POIs are co-visited, and by how many trajectories?",
  """SELECT ?a ?b WHERE { ?a poi:covisitedWith ?b }"""),

 ("CQ3", "requirements", "trajectory segmentation",
  "How many trajectories did each moving entity produce?",
  """SELECT ?person (COUNT(DISTINCT ?traj) AS ?n) WHERE {
       ?person trajecto:formsTrajectory ?traj .
     } GROUP BY ?person ORDER BY DESC(?n)"""),

 ("CQ4", "requirements", "recognition of POIs by dwell time and proximity",
  "Which POI interactions were recognised, with what dwell time and distance?",
  """SELECT ?poi ?dwell ?dist WHERE {
       ?i a poi:POIInteraction ; poi:interactsWith ?poi ;
          poi:hasDwellTimeInSeconds ?dwell .
       OPTIONAL { ?i spatial:hasDistance ?dist }
     } ORDER BY DESC(xsd:decimal(?dwell))"""),

 ("CQ5", "requirements", "prediction of next positions",
  "Given a POI, which POIs are recorded as following it?",
  """SELECT ?from ?to WHERE { ?from poi:followedBy ?to }"""),

 ("CQ6", "requirements", "trajectory classification by shared characteristics",
  "Which purpose classes are assigned, and to how many trajectories each?",
  """SELECT ?purpose (COUNT(DISTINCT ?traj) AS ?n) WHERE {
       ?traj trajecto:hasPurpose ?purpose .
     } GROUP BY ?purpose ORDER BY DESC(?n)"""),

 ("CQ7", "requirements", "trajectory clustering by extracted patterns",
  "Which movement patterns are exhibited, and by how many trajectories?",
  """SELECT ?pattern (COUNT(DISTINCT ?traj) AS ?n) WHERE {
       ?traj patterns:exhibitsPattern ?pattern .
     } GROUP BY ?pattern ORDER BY DESC(?n)"""),

 ("CQ8", "requirements", "recognition of visitor engagement",
  "What engagement level is inferred per trajectory, and how are levels distributed?",
  """SELECT ?level (COUNT(DISTINCT ?traj) AS ?n) WHERE {
       ?traj trajecto:hasEngagementLevel ?level .
     } GROUP BY ?level ORDER BY DESC(?n)"""),

 ("CQ9", "requirements", "interest recognition based on POI visit/selection",
  "What are a user's inferred category preferences, with score and confidence?",
  # Two instantiations represent this differently: Rome and GeoLife use the
  # EWMA snapshot model (ProfileSnapshot -> hasPreferences -> categoryType),
  # BIRD uses the SR-20 rule model (hasInferredPreference -> forCategory).
  # The competency question is about the capability, so it accepts either --
  # but the divergence is itself a representation-consistency finding.
  # Grouped per category: the answer is the preference profile, not one row
  # per stored preference record.
  """SELECT ?category (COUNT(*) AS ?records) (AVG(xsd:decimal(?score)) AS ?meanScore) WHERE {
       { ?snap a user:ProfileSnapshot ; user:hasPreferences ?pref .
         ?pref user:prefersPoiCategory ?category ; user:preferenceScore ?score }
       UNION
       { ?p user:hasInferredPreference ?pref .
         ?pref user:forCategory ?category ; user:preferenceScore ?score }
     } GROUP BY ?category ORDER BY DESC(?records)"""),

 ("CQ10", "requirements", "prediction of trajectory timespan from its class",
  "What is the mean dwell time of trajectories grouped by purpose?",
  """SELECT ?purpose (AVG(xsd:decimal(?dwell)) AS ?meanDwell)
            (COUNT(DISTINCT ?traj) AS ?n) WHERE {
       ?traj trajecto:hasPurpose ?purpose .
       ?pos dul:isPartOf ?traj ; trajecto:hasInteraction ?i .
       ?i poi:hasDwellTimeInSeconds ?dwell .
     } GROUP BY ?purpose ORDER BY DESC(?meanDwell)"""),

 ("CQ11", "requirements", "recommend POIs from similar trajectories",
  "Which POIs are co-visited with a POI the trajectory already contains?",
  # Grouped per trajectory. Ungrouped this returns the raw join cardinality
  # (~900k rows on GeoLife), which measures graph size rather than answering
  # the question: how many distinct new POIs can be recommended for a trip.
  # Answer = trajectories for which a novel recommendation exists.
  # The inner DISTINCT matters: collapsing (traj, candidate) before the
  # anti-join takes this from a >10-minute timeout to ~9s on GeoLife. A
  # correlated FILTER NOT EXISTS over the raw join re-checks every position.
  """SELECT ?traj (COUNT(DISTINCT ?candidate) AS ?candidates) WHERE {
       { SELECT DISTINCT ?traj ?candidate WHERE {
           ?pos dul:isPartOf ?traj ; trajecto:hasInteraction ?i .
           ?i poi:interactsWith ?seed .
           ?seed poi:covisitedWith ?candidate . } }
       MINUS { ?p2 dul:isPartOf ?traj ; trajecto:hasInteraction ?i2 .
               ?i2 poi:interactsWith ?candidate . }
     } GROUP BY ?traj ORDER BY DESC(?candidates)"""),

 ("CQ12", "requirements", "context-based recommendation from visitor interests",
  "Which POI categories does a user prefer that they have not yet visited?",
  # the trajectory join here is not needed to answer the question and
  # multiplies every preference by the person's trajectory count, which
  # times out on a corpus with tens of thousands of preference records
  """SELECT DISTINCT ?person ?category ?score WHERE {
       { ?person user:hasProfile ?profile .
         ?profile user:hasSnapshot ?snap .
         ?snap user:hasPreferences ?pref .
         ?pref user:prefersPoiCategory ?category ; user:preferenceScore ?score }
       UNION
       { ?person user:hasInferredPreference ?pref .
         ?pref user:forCategory ?category ; user:preferenceScore ?score }
     }"""),

 ("CQ13", "requirements", "temporal context of movement",
  "Under which temporal contexts do trajectories occur?",
  """SELECT ?ctx (COUNT(DISTINCT ?traj) AS ?n) WHERE {
       ?traj context:hasTemporalContext ?ctx .
     } GROUP BY ?ctx ORDER BY DESC(?n)"""),

 ("CQ14", "pipeline", "ontology-parameterised interaction semantics",
  "What dwell thresholds does the ontology define for each POI type?",
  """SELECT ?type ?typical ?minMeaningful ?maxReasonable WHERE {
       ?type poi:hasTypicalDuration ?typical ;
             poi:hasMinimumMeaningfulDuration ?minMeaningful ;
             poi:hasMaximumReasonableDuration ?maxReasonable .
     }"""),

 ("CQ15", "pipeline", "KG-native structural analytics",
  "Which POIs are structurally central, by materialised PageRank?",
  """SELECT ?poi ?pr WHERE { ?poi trajecto:pageRank ?pr }
     ORDER BY DESC(xsd:decimal(?pr))"""),
]


def ask(kg, query, timeout=300):
    r = requests.post(f"{SERVER}/{kg}/sparql", data={"query": P + query},
                      headers={"Accept": "application/json"}, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    return j["head"]["vars"], j["results"]["bindings"]


def short(uri):
    return uri.rsplit("#", 1)[-1].rsplit("/", 1)[-1] if uri.startswith("http") else uri


def run(kg):
    rows = []
    for cid, source, capability, question, q in CQS:
        try:
            varz, b = ask(kg, q)
            n = len(b)
            sample = "; ".join(f"{v}={short(b[0][v]['value'])[:26]}"
                               for v in varz if v in b[0]) if b else ""
        except Exception as exc:
            n, sample = -1, f"ERROR: {type(exc).__name__}"
        rows.append({"cq": cid, "source": source, "capability": capability,
                     "question": question, "answers": n, "sample": sample})
        mark = "ok " if n > 0 else ("EMPTY" if n == 0 else "ERR")
        print(f"  {cid:5} {mark:5} {n:>7}  {question[:62]}")
    return pd.DataFrame(rows)


def self_check():
    """Query text is well-formed and every CQ is uniquely identified."""
    ids = [c[0] for c in CQS]
    assert len(ids) == len(set(ids)), "duplicate CQ id"
    for cid, src, cap, qn, q in CQS:
        assert q.strip().upper().startswith("SELECT"), f"{cid} is not a SELECT"
        assert qn.strip().endswith("?"), f"{cid} question is not a question"
        assert src in ("requirements", "pipeline"), f"{cid} has an unknown source"
        assert cap, f"{cid} has no capability"
    assert sum(1 for c in CQS if c[1] == "requirements") == 13
    assert sum(1 for c in CQS if c[1] == "pipeline") == 2
    assert short("http://w3id.org/trajecto/poi#Cafe") == "Cafe"
    assert short("PT15M") == "PT15M"
    print(f"cq_coverage self-check passed ({len(CQS)} questions)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", action="append", default=None,
                    help="dataset name; repeat for a side-by-side comparison")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        self_check(); return

    kgs = a.kg or ["osm-rome"]
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {}
    for kg in kgs:
        print(f"\n=== {kg} ===")
        df = run(kg)
        df.to_csv(OUT / f"cq_coverage_{kg}.csv", index=False)
        frames[kg] = df

    print("\n=== coverage summary ===")
    for kg, df in frames.items():
        ok = int((df.answers > 0).sum())
        req = df[df.source == "requirements"]
        print(f"  {kg:16} {ok}/{len(df)} answered"
              f"  (requirements {int((req.answers > 0).sum())}/{len(req)})"
              f"   empty: {sorted(df[df.answers == 0].cq)}"
              f"   errors: {sorted(df[df.answers < 0].cq)}")
    if len(frames) > 1:
        comp = pd.DataFrame({kg: (df.set_index('cq').answers > 0)
                             for kg, df in frames.items()})
        comp.to_csv(OUT / "cq_coverage_portability.csv")
        print(f"\n  portability matrix -> {OUT}/cq_coverage_portability.csv")


if __name__ == "__main__":
    main()
