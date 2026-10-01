"""
Cold-start evaluation.

Splits protocol-B outcomes by (i) held-out-POI evidence level (leak-free
popularity: 0 = unrecoverable by corpus evidence, 1-2 = tail, >=3 = head)
and (ii) user history size (# other trajectories of the same user). The
tail x low-history cells are the cold-start regime; the question is which
signals still work there.

Methods: global popularity (the incumbent cold prior), markov2, unified
route-heavy, the RRF fusion, and the two new cold levers — tag-conditioned
popularity and KG centrality (reads the PageRank that notebook 21 materializes,
trajecto:pageRank) — plus their cold blend.

Usage:  cd notebooks && python run_coldstart.py
Output: output/evaluation/coldstart_slices.csv
"""

import os
import sys
import time
from functools import partial
from pathlib import Path

import pandas as pd

sys.path.insert(0, '.')

from helpers.unified_recommender import UnifiedRecommender
from helpers.context_recommender import EXCLUDED_CATEGORIES
from helpers.baselines import popularity_rec, proximity_rec, _popularity
from helpers.baseline_evaluation import _split
from helpers.rec_levers import (
    markov2_rec, rrf, tagpop_rec, centrality_rec,
)

OUTPUT_DIR = Path('./output/evaluation')
MIN_SEQ = 3
HOLD_EXCL = set(EXCLUDED_CATEGORIES)

ROUTE_HEAVY_WEIGHTS = {
    'archetype_sim': 0.10, 'semantic_sim': 0.10, 'route_sim': 0.30,
    'flow_prob': 0.20, 'dominant_ctx': 0.08, 'user_pref': 0.08,
    'significant': 0.05, 'engagement': 0.04, 'rule_coverage': 0.05,
}


def unified_rec(rec_engine, traj_uri, input_seq, top_k,
                weights=None, exclude_traj=None):
    return rec_engine.recommend(
        traj_uri, observed_sequence=input_seq, top_k=top_k,
        signal_weights=weights, exclude_traj=exclude_traj or traj_uri,
    )


def main():
    t0 = time.time()
    rec = UnifiedRecommender()
    rec.load_all_data(features_path=os.environ.get(
        'TRAJECTO_FEATURES', './output/semantic_features/trajectory_features.json'))
    print(f'Loaded in {time.time()-t0:.0f}s')

    uni_rh = partial(unified_rec, weights=ROUTE_HEAVY_WEIGHTS)
    methods = {
        'popularity': popularity_rec,
        'markov2': markov2_rec,
        'unified_route_heavy': uni_rh,
        'rrf_uni+markov2+prox': rrf((uni_rh, markov2_rec, proximity_rec)),
        'tagpop (1.7b)': tagpop_rec,
        'centrality (1.7c)': centrality_rec,
        'cold_blend (1.7d)': rrf((tagpop_rec, centrality_rec,
                                  popularity_rec)),
    }

    eligible = [t for t in rec.sequences
                if len(rec.sequences[t]) >= MIN_SEQ]

    rows = []
    for traj in eligible:
        input_seq, hold = _split(rec, traj, 'trajectory_split', 0.7, 1)
        hold_set = {h for h in set(hold)
                    if rec.poi_metadata.get(h, {}).get('category', '')
                    not in HOLD_EXCL}
        if not hold_set:
            continue
        user = rec.traj_to_user.get(traj)
        hist = len([t for t in rec.user_to_trajs.get(user, [])
                    if t != traj and rec.sequences.get(t)])
        tops = {}
        for name, fn in methods.items():
            try:
                recs = fn(rec, traj, input_seq, 10, exclude_traj=traj)
            except TypeError:
                recs = fn(rec, traj, input_seq, 10)
            tops[name] = {r['poi'] for r in recs}
        for h in hold_set:
            pop = _popularity(rec, h, exclude_traj=traj)
            evidence = ('0_unrecoverable' if pop == 0
                        else '1-2_tail' if pop <= 2 else '3+_head')
            for name in methods:
                rows.append({'method': name, 'traj': traj, 'poi': h,
                             'evidence': evidence,
                             'user_history': '2+' if hist >= 2 else '0-1',
                             'hit': int(h in tops[name])})

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / 'coldstart_raw.csv', index=False)

    print('\n=== Hit-rate@10 by held-out-POI evidence level ===')
    ev = df.pivot_table(index='method', columns='evidence', values='hit',
                        aggfunc='mean').round(3)
    counts = df[df.method == list(methods)[0]]['evidence'] \
        .value_counts().to_dict()
    print(f'   (holdout POIs per slice: {counts})')
    print(ev.to_string())

    print('\n=== Hit-rate@10 by user history (recoverable POIs only) ===')
    rec_df = df[df.evidence != '0_unrecoverable']
    uh = rec_df.pivot_table(index='method', columns='user_history',
                            values='hit', aggfunc='mean').round(3)
    counts = rec_df[rec_df.method == list(methods)[0]]['user_history'] \
        .value_counts().to_dict()
    print(f'   (holdout POIs per slice: {counts})')
    print(uh.to_string())

    print('\n=== Cold cell: tail evidence x low history ===')
    cold = df[(df.evidence == '1-2_tail') & (df.user_history == '0-1')]
    if len(cold):
        print(f'   ({len(cold)//len(methods)} holdout POIs)')
        print(cold.groupby('method')['hit'].mean().round(3)
              .sort_values(ascending=False).to_string())

    ev.join(uh, lsuffix='_ev').to_csv(OUTPUT_DIR / 'coldstart_slices.csv')
    print(f'\nSaved. Total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
