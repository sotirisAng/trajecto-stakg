#!/usr/bin/env python3
"""Tag-layer reliability across instantiations (evaluation axis E).

The tag layer is the pipeline's primary deliverable -- session transfer showed
cross-session POI identity is not learnable, so the tags, not the recommender,
are what the KG contributes. It has been measured on one corpus. This runs the
perturbation harness on each instantiation and reports them side by side, so
reliability and portability are evidenced by the same instrument.

    python run_tag_reliability.py                                  # default KG
    python run_tag_reliability.py --kg osm-rome --kg bird
    python run_tag_reliability.py --self-check

Output: output/evaluation/tag_reliability_<kg>.csv and a combined table.
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CFG = HERE / "helpers" / "fuseki_config.py"
OUT = HERE / "output" / "evaluation"


def run_for(kg, min_pois=3):
    """Run the harness against one dataset, reloading helpers so the endpoint
    switch takes effect (fuseki_config binds graph_name at import time)."""
    for mod in [m for m in list(sys.modules) if "helpers" in m]:
        del sys.modules[mod]
    from helpers import tag_stability as ts
    return ts.run_harness(min_pois=min_pois, verbose=False)


def corpus_notes(kg):
    """Facts that qualify how the numbers should be read for this corpus."""
    import requests
    from helpers.fuseki_config import query_endpoint

    def n(q):
        r = requests.post(query_endpoint, data={"query": q},
                          headers={"Accept": "application/json"}, timeout=180)
        r.raise_for_status()
        return int(r.json()["results"]["bindings"][0]["n"]["value"])

    dist = n("""PREFIX poi:<http://w3id.org/trajecto/poi#>
                PREFIX sp:<http://w3id.org/trajecto/spatial#>
                SELECT (COUNT(*) AS ?n) WHERE {?i a poi:POIInteraction ; sp:hasDistance ?d}""")
    inter = n("""PREFIX poi:<http://w3id.org/trajecto/poi#>
                 SELECT (COUNT(*) AS ?n) WHERE {?i a poi:POIInteraction}""")
    notes = []
    if inter and dist == 0:
        # stage-5 scores 0.8*log-norm dwell + 0.2*inv-norm distance; with no
        # distance the second term is uniformly zero, so dominance ranks on
        # dwell alone. The argmax is unaffected, the score scale is not.
        notes.append("no spatial:hasDistance -- dominant POI is dwell-only")
    return "; ".join(notes) or "-"


def self_check():
    """The harness rules, exercised without a KG."""
    for mod in [m for m in list(sys.modules) if "helpers" in m]:
        del sys.modules[mod]
    from helpers.tag_stability import infer_tags

    def rec(poi, dwell, dist=None, imm=False, mean=False, cat="Museum"):
        return {"poi": poi, "category": cat, "dwell": dwell, "dist": dist,
                "imm": imm, "meaningful": mean, "ts": "2020-01-01T00:00:00"}

    # dominance follows dwell
    t = infer_tags([rec("a", 10), rec("b", 600), rec("c", 30)])
    assert t["dominant_poi"] == "b", t

    # with no distances at all the 0.2 term is uniform, so the argmax is
    # unchanged -- this is what makes the harness valid on BIRD
    with_d = infer_tags([rec("a", 10, dist=5), rec("b", 600, dist=900)])
    without_d = infer_tags([rec("a", 10), rec("b", 600)])
    assert with_d["dominant_poi"] == without_d["dominant_poi"] == "b"

    # SR-10 engagement thresholds
    assert infer_tags([rec("a", 1, imm=True) for _ in range(3)])["engagement"] == "HighEngagement"
    assert infer_tags([rec("a", 1, mean=True) for _ in range(5)])["engagement"] == "MediumEngagement"
    assert infer_tags([rec("a", 1)])["engagement"] == "LowEngagement"

    # interaction-count navigation is knife-edge by construction
    assert infer_tags([rec(str(i), 1) for i in range(2)])["nav_interaction"] == "TargetedSegment"
    assert infer_tags([rec(str(i), 1) for i in range(11)])["nav_interaction"] == "ExploratoryMovement"
    assert infer_tags([rec(str(i), 1) for i in range(5)])["nav_interaction"] is None
    print("tag_reliability self-check passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", action="append", default=None)
    ap.add_argument("--min-pois", type=int, default=3)
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        self_check(); return

    original = CFG.read_text()
    kgs = a.kg or [re.search(r'^graph_name = "([^"]*)"', original, re.M).group(1)]
    OUT.mkdir(parents=True, exist_ok=True)
    frames, notes = {}, {}

    try:
        for kg in kgs:
            CFG.write_text(re.sub(r'^graph_name = "[^"]*"', f'graph_name = "{kg}"',
                                  original, count=1, flags=re.M))
            print(f"\n=== {kg} ===")
            df = run_for(kg, a.min_pois)
            notes[kg] = corpus_notes(kg)
            df.to_csv(OUT / f"tag_reliability_{kg}.csv", index=False)
            frames[kg] = df.set_index("tag")
            print(df.to_string(index=False))
            print(f"  note: {notes[kg]}")
    finally:
        CFG.write_text(original)

    if len(frames) > 1:
        print("\n=== stability / predictivity across instantiations ===")
        tags = list(next(iter(frames.values())).index)
        print(f"{'tag':32} " + " ".join(f"{k:>22}" for k in frames))
        for t in tags:
            cells = []
            for k, f in frames.items():
                if t not in f.index or pd.isna(f.loc[t, "stability"]):
                    cells.append(f"{'-':>22}")
                else:
                    cells.append(f"{f.loc[t,'stability']:>10.3f} /{f.loc[t,'predictivity']:>10.3f}")
            print(f"{t:32} " + " ".join(cells))
        comb = pd.concat({k: f for k, f in frames.items()}, names=["corpus"])
        comb.to_csv(OUT / "tag_reliability_all.csv")
        print(f"\nwrote {OUT}/tag_reliability_all.csv")
        for k, v in notes.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
