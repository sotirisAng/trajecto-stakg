#!/usr/bin/env python3
"""Re-test the ladder comparisons with per-user sign flipping.

The reported p-values treat trajectories as independent. They are not: 87 of
Rome's 107 eligible trajectories belong to 18 users. Drawing one sign per user
rather than per trajectory respects that, and is the honest test to report.

    python run_clustered_significance.py --kg osm-rome
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CFG = HERE / "helpers" / "fuseki_config.py"
OUT = HERE / "output" / "evaluation"

PAIRS = [("unified_route_heavy", "popularity"),
         ("unified_route_heavy", "markov"),
         ("unified_route_heavy", "markov2"),
         ("lr_fusion", "unified_route_heavy"),
         ("rrf_uni+markov2+prox", "unified_route_heavy")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default=None)
    ap.add_argument("--summaries", default=None)
    a = ap.parse_args()

    original = CFG.read_text()
    kg = a.kg or re.search(r'^graph_name = "([^"]*)"', original, re.M).group(1)
    prefix = "geolife" if "geolife" in kg else "osm_rome"
    summaries = a.summaries or str(OUT / f"{prefix}_summaries.json")

    try:
        CFG.write_text(re.sub(r'^graph_name = "[^"]*"', f'graph_name = "{kg}"',
                              original, count=1, flags=re.M))
        for m in [m for m in list(sys.modules) if "helpers" in m]:
            del sys.modules[m]
        from helpers.unified_recommender import UnifiedRecommender
        from helpers.eval_protocols import paired_delta

        rec = UnifiedRecommender()
        # traj_to_user is populated by load_all_data, not __init__ -- without
        # this every trajectory becomes its own cluster and the clustered test
        # silently reduces to the unclustered one
        feats = ('./output/semantic_features/geolife_features.json'
                 if 'geolife' in kg else
                 './output/semantic_features/trajectory_features.json')
        rec.load_all_data(features_path=feats)
        t2u = dict(rec.traj_to_user)
        if not t2u:
            raise RuntimeError('traj_to_user is empty; clustering would be a no-op')
        res = json.load(open(summaries))["fullpool"]

        rows = []
        print(f"\n=== {kg} — {len(set(t2u.values()))} users over {len(t2u)} trajectories ===")
        print(f"{'comparison':44} {'delta':>8} {'p flat':>8} {'p clust':>8}  verdict")
        for x, y in PAIRS:
            if x not in res or y not in res:
                continue
            for metric in ("recall@10", "mrr"):
                flat = paired_delta(res[x], res[y], metric=metric)
                clus = paired_delta(res[x], res[y], metric=metric, traj_to_user=t2u)
                keep = "holds" if clus["p_value"] < 0.05 else (
                       "LOST" if flat["p_value"] < 0.05 else "-")
                print(f"{x[:20]+' vs '+y[:14]+' '+metric:44} "
                      f"{flat['delta']:+8.3f} {flat['p_value']:8.3f} "
                      f"{clus['p_value']:8.3f}  {keep}")
                rows.append({"kg": kg, "a": x, "b": y, "metric": metric,
                             "delta": flat["delta"], "p_flat": flat["p_value"],
                             "p_clustered": clus["p_value"],
                             "n_trajectories": flat["n_common"],
                             "n_users": clus.get("n_clusters")})
        pd.DataFrame(rows).to_csv(OUT / f"{prefix}_clustered_significance.csv", index=False)
        print(f"\nwrote {OUT}/{prefix}_clustered_significance.csv")
    finally:
        CFG.write_text(original)


if __name__ == "__main__":
    main()
