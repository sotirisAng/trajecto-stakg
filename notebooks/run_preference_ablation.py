#!/usr/bin/env python3
"""Which user-preference model should the recommender read?

The recommender used to compute preferences once, from all of a user's
interactions, so the categories of the held-out part of the evaluated
trajectory already raised that user's scores. This compares that leaky model
with leak-free ones, swapped into the same loaded recommender on the full-pool
ladder. Everything else is held fixed.

  runtime (leaky)   0.4*immersive_frac + 0.5*meaningful_frac + 0.1*visit_norm
                    over all the user's trajectories (the old behaviour)
  none              signal ablated
  snapshot-before   the Stage 9 EWMA profile as it stood before the trajectory:
                    the user's latest ProfileSnapshot whose updateTimestamp
                    (the end of the trajectory that produced it) precedes the
                    trajectory's start
  runtime-loo       the runtime formula over the user's other trajectories
                    (what the recommender now uses)
  runtime-past      the runtime formula over trajectories that ended before
                    this one started

    python run_preference_ablation.py --kg geolife
"""

import argparse
import re
import sys
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CFG = HERE / "helpers" / "fuseki_config.py"
OUT = HERE / "output" / "evaluation"


def snapshots(endpoint):
    """user -> time-sorted [(timestamp, [{category, score}, ...]), ...]."""
    q = """
    PREFIX user: <http://w3id.org/trajecto/user#>
    SELECT ?person ?snap ?ts ?category ?score WHERE {
      ?person user:hasProfile ?prof . ?prof user:hasSnapshot ?snap .
      ?snap user:updateTimestamp ?ts ; user:hasPreferences ?pref .
      ?pref user:prefersPoiCategory ?category ; user:preferenceScore ?score .
    }"""
    r = requests.post(endpoint, data={"query": q},
                      headers={"Accept": "application/json"}, timeout=600)
    r.raise_for_status()
    snaps = defaultdict(lambda: defaultdict(list))
    stamp = {}
    for b in r.json()["results"]["bindings"]:
        u, s = b["person"]["value"], b["snap"]["value"]
        cat = b["category"]["value"].rsplit("#", 1)[-1].rsplit("/", 1)[-1]
        snaps[u][s].append({"category": cat, "score": float(b["score"]["value"])})
        stamp[s] = b["ts"]["value"][:19]
    out = {}
    for u, by_snap in snaps.items():
        out[u] = sorted(((stamp[s], sorted(p, key=lambda x: -x["score"]))
                         for s, p in by_snap.items()), key=lambda x: x[0])
    return out


def runtime_prefs(rec, trajs):
    """The recommender's formula, over the given trajectories only."""
    from helpers.context_recommender import EXCLUDED_CATEGORIES
    stats = defaultdict(lambda: {"total": 0, "immersive": 0, "meaningful": 0})
    for t in trajs:
        for it in rec.interaction_data.get(t, []):
            cat = it.get("category", "Unknown")
            if cat in EXCLUDED_CATEGORIES or cat in ("PointOfInterest", "SignificantPOI"):
                continue
            stats[cat]["total"] += 1
            stats[cat]["immersive"] += it["int_type"] == "ImmersiveInteraction"
            stats[cat]["meaningful"] += it["int_type"] in (
                "MeaningfulInteraction", "ImmersiveInteraction")
    if not stats:
        return []
    top = max(s["total"] for s in stats.values())
    return sorted(({"category": c, "score": 0.4 * s["immersive"] / s["total"]
                    + 0.5 * s["meaningful"] / s["total"] + 0.1 * s["total"] / top}
                   for c, s in stats.items()), key=lambda x: -x["score"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default=None)
    a = ap.parse_args()
    original = CFG.read_text()
    kg = a.kg or re.search(r'^graph_name = "([^"]*)"', original, re.M).group(1)
    feats = ('./output/semantic_features/geolife_features.json' if 'geolife' in kg
             else './output/semantic_features/trajectory_features.json')

    try:
        CFG.write_text(re.sub(r'^graph_name = "[^"]*"', f'graph_name = "{kg}"',
                              original, count=1, flags=re.M))
        for m in [m for m in list(sys.modules) if "helpers" in m]:
            del sys.modules[m]
        from helpers.unified_recommender import UnifiedRecommender, UNIFIED_WEIGHTS
        from helpers.baseline_evaluation import evaluate_method
        from helpers.context_recommender import EXCLUDED_CATEGORIES
        from helpers.fuseki_config import query_endpoint

        rec = UnifiedRecommender()
        rec.load_all_data(features_path=feats)
        leaky = {u: [dict(p) for p in v] for u, v in rec.user_preferences.items()}
        window = rec._traj_time_window
        start = lambda t: window[t][0].isoformat()[:19] if t in window else None
        end = lambda t: window[t][1].isoformat()[:19] if t in window else None
        user_trajs = defaultdict(list)
        for t, u in rec.traj_to_user.items():
            user_trajs[u].append(t)
        snaps = snapshots(query_endpoint)

        def snapshot_before(t, u):
            seq, s0 = snaps.get(u, []), start(t)
            if not seq or s0 is None:
                return []
            i = bisect_left([ts for ts, _ in seq], s0)
            return seq[i - 1][1] if i else []

        def loo(t, u):
            return runtime_prefs(rec, [x for x in user_trajs[u] if x != t])

        def past(t, u):
            s0 = start(t)
            return runtime_prefs(rec, [x for x in user_trajs[u]
                                       if x != t and s0 and end(x) and end(x) < s0])

        models = {
            "runtime (leaky)": lambda t, u: leaky.get(u, []),
            "none": lambda t, u: [],
            "snapshot-before": snapshot_before,
            "runtime-loo": loo,
            "runtime-past": past,
        }
        ROUTE_HEAVY = {'archetype_sim': 0.10, 'semantic_sim': 0.10,
                       'route_sim': 0.30, 'flow_prob': 0.20, 'dominant_ctx': 0.08,
                       'user_pref': 0.08, 'significant': 0.05, 'engagement': 0.04,
                       'rule_coverage': 0.05}

        rows = []
        for name, model in models.items():
            print(f"\n--- {name}")
            for label, w in (("default", UNIFIED_WEIGHTS), ("route-heavy", ROUTE_HEAVY)):
                def ranker(r, t, seq, k, _w=w, _m=model):
                    r._preferences = lambda u, ex, _t=t: _m(_t, u) if u else []
                    return r.recommend(t, observed_sequence=seq, top_k=k,
                                       signal_weights=_w, exclude_traj=t)
                s = evaluate_method(rec, ranker, method_name=f"{name}/{label}",
                                    k_values=(10,), min_seq_length=3, verbose=False,
                                    holdout_exclude_categories=set(EXCLUDED_CATEGORIES))
                rows.append({"kg": kg, "preferences": name, "weights": label,
                             "n": s["n_evaluated"], "recall@10": s["recall@10"],
                             "R/ceil": s["normalized_recall@10"], "mrr": s["mrr"]})
                print(f"      {label:12} R@10={s['recall@10']:.4f} "
                      f"R/ceil={s['normalized_recall@10']:.4f} MRR={s['mrr']:.4f}")

        df = pd.DataFrame(rows)
        OUT.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT / f"preference_ablation_{kg}.csv", index=False)
        print(f"\n{df.to_string(index=False, float_format='%.4f')}")
        print(f"\nwrote {OUT}/preference_ablation_{kg}.csv")
    finally:
        CFG.write_text(original)


if __name__ == "__main__":
    main()
