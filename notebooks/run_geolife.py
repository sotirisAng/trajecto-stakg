"""
Recommender evaluation on GeoLife (Section 7.10 of the thesis): the protocols
and methods of run_osm_rome.py, with outputs prefixed geolife_.

Candidates are all POIs of the KG except the categories the recommender never
suggests; held-out targets are restricted to the same recommendable categories.

Protocols:
  0  full-pool trajectory_split 0.7
  1  within-radius 500 m / 1000 m (shared candidate set)
  2  user session transfer — novel-POI holdout AND repeats-allowed holdout

Method set: baseline ladder + markov2 + unified presets + the balanced RRF
ensemble + LR fusion (refit on the tune half, seed 11 — its test-half
numbers are the honest ones; full-corpus numbers include its training
trajectories and are marked accordingly).

Usage:  cd notebooks && TRAJECTO_KG=geolife python run_geolife.py
Output: output/evaluation/geolife_protocol_{fullpool,radius500,radius1000,
        transfer,transfer_repeats}.csv, geolife_summaries.json,
        geolife_significance.csv
"""

import json
import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, '.')

from helpers.unified_recommender import UnifiedRecommender, UNIFIED_WEIGHTS
from helpers.context_recommender import EXCLUDED_CATEGORIES
from helpers.baselines import (
    popularity_rec, category_popularity_rec, proximity_rec, markov_rec,
)
from helpers.baseline_evaluation import evaluate_method, _split
from helpers.eval_protocols import (
    evaluate_within_radius, evaluate_user_transfer, paired_delta,
)
from helpers.rec_levers import markov2_rec, rrf, repeat_rec, LRFusion

OUTPUT_DIR = Path('./output/evaluation')
MIN_SEQ = 3
SEED = 11
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


def random_ranker(rec_engine, traj_uri, input_seq, top_k,
                  _rng=np.random.default_rng(42)):
    visited = set(input_seq)
    # sorted() before shuffle: all_poi_uris is a set, so the pre-shuffle order
    # varies between processes and the seed alone does not make this
    # reproducible. The sanity floor has to be as stable as the methods it
    # bounds.
    candidates = sorted(p for p in rec_engine.all_poi_uris - visited
                        if rec_engine.poi_metadata.get(p, {}).get('category', '')
                        not in EXCLUDED_CATEGORIES)
    _rng.shuffle(candidates)
    out = []
    for p in candidates[:top_k]:
        m = rec_engine.poi_metadata.get(p, {})
        out.append({'poi': p, 'label': m.get('label', ''),
                    'category': m.get('category', ''),
                    'score': 0.0, 'signals': {}})
    return out


def _excl(fn):
    def wrapped(re_, t, i, k, exclude_traj=None):
        return fn(re_, t, i, k, exclude_traj=exclude_traj or t)
    return wrapped


def main():
    t0 = time.time()
    rec = UnifiedRecommender()
    rec.load_all_data(
        features_path=os.environ.get('TRAJECTO_FEATURES', './output/semantic_features/geolife_features.json'))
    print(f'Loaded in {time.time()-t0:.0f}s')

    eligible = sorted(t for t in rec.sequences
                      if len(rec.sequences[t]) >= MIN_SEQ)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(eligible))
    tune_set = {eligible[i] for i in perm[:len(eligible) // 2]}

    uni_rh = partial(unified_rec, weights=ROUTE_HEAVY_WEIGHTS)

    print('Fitting LR fusion on tune half (seed 11)...')
    lrf = LRFusion(rec, uni_rh, signal_names=list(UNIFIED_WEIGHTS))
    lrf.fit(sorted(tune_set),
            lambda t: _split(rec, t, 'trajectory_split', 0.7, 1))

    methods = {
        'random': random_ranker,
        'popularity': _excl(popularity_rec),
        'category_popularity': _excl(category_popularity_rec),
        'proximity': proximity_rec,
        'markov': markov_rec,
        'markov2': markov2_rec,
        'unified_default': partial(unified_rec, weights=UNIFIED_WEIGHTS),
        'unified_route_heavy': uni_rh,
        'rrf_uni+markov2+prox': rrf((uni_rh, markov2_rec, proximity_rec)),
        'lr_fusion': lrf.ranker(),
    }

    all_results = {}

    def dump_csv(results, fname, extra=()):
        cols = ['method', 'n_evaluated', 'recall@10',
                'normalized_recall@10', 'cat_recall@10', 'mrr',
                'ceiling_recall'] + list(extra)
        rows = []
        for name, s in results.items():
            row = {c: s.get(c) for c in cols}
            row['method'] = name
            rows.append(row)
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / fname, index=False)

    # ── Protocol 0: full-pool ─────────────────────────────────────────
    print(f'\n=== Full-pool (protocol B, min_seq={MIN_SEQ}) ===')
    fullpool = {}
    for name, fn in methods.items():
        s = evaluate_method(rec, fn, method_name=name, k_values=(5, 10),
                            protocol='trajectory_split', split_ratio=0.7,
                            min_seq_length=MIN_SEQ, verbose=False,
                            holdout_exclude_categories=HOLD_EXCL)
        # test-half means (the honest slice for lr_fusion)
        per = s['per_trajectory']['recall@10']
        test_vals = [v for t, v in per.items() if t not in tune_set]
        s['test_half_recall@10'] = float(np.mean(test_vals)) if test_vals else 0.0
        per_m = s['per_trajectory']['mrr']
        test_m = [v for t, v in per_m.items() if t not in tune_set]
        s['test_half_mrr'] = float(np.mean(test_m)) if test_m else 0.0
        fullpool[name] = s
        print(f"  {name:<22} N={s['n_evaluated']:>3}  "
              f"R@10={s['recall@10']:.3f}  "
              f"R/ceil={s['normalized_recall@10']:.2f}  "
              f"MRR={s['mrr']:.3f}  "
              f"[test-half R={s['test_half_recall@10']:.3f} "
              f"MRR={s['test_half_mrr']:.3f}]")
    all_results['fullpool'] = fullpool
    dump_csv(fullpool, 'geolife_protocol_fullpool.csv',
             extra=('test_half_recall@10', 'test_half_mrr'))

    # ── Protocol 1: within-radius ─────────────────────────────────────
    for radius in (500.0, 1000.0):
        print(f'\n=== Within-radius {radius:.0f} m ===')
        res = {}
        for name, fn in methods.items():
            s = evaluate_within_radius(
                rec, fn, method_name=name, radius_m=radius,
                k_values=(5, 10), min_seq_length=MIN_SEQ)
            res[name] = s
            print(f"  {name:<22} N={s['n_evaluated']:>3}  "
                  f"R@10={s['recall@10']:.3f}  MRR={s['mrr']:.3f}")
        any_r = next(iter(res.values()))
        print(f"  candidates≈{any_r['mean_candidates']:.0f} | "
              f"holdout outside: {any_r['holdout_out_of_radius']:.1%} | "
              f"ceiling {any_r['ceiling_recall']:.3f}")
        all_results[f'radius{int(radius)}'] = res
        dump_csv(res, f'geolife_protocol_radius{int(radius)}.csv',
                 extra=('mean_candidates',))

    # ── Protocol 2: session transfer (novel / repeats) ────────────────
    transfer_methods = dict(methods)
    transfer_methods['repeat'] = repeat_rec
    transfer_methods['rrf_repeat+uni+markov2'] = rrf(
        (repeat_rec, uni_rh, markov2_rec))

    for label, repeats in (('transfer', False), ('transfer_repeats', True)):
        print(f'\n=== Session transfer ({label}) ===')
        res = {}
        for name, fn in transfer_methods.items():
            if not repeats and name in ('repeat',
                                        'rrf_repeat+uni+markov2'):
                continue
            s = evaluate_user_transfer(
                rec, fn, method_name=name, k_values=(5, 10, 20),
                min_seq_length=MIN_SEQ,
                holdout_exclude_categories=HOLD_EXCL,
                include_repeats=repeats)
            res[name] = s
            print(f"  {name:<24} N={s['n_evaluated']:>3}  "
                  f"R@10={s['recall@10']:.3f}  MRR={s['mrr']:.3f}")
        any_r = next(iter(res.values()))
        print(f"  ceiling {any_r['ceiling_recall']:.3f} | repeat fraction "
              f"{any_r['mean_repeat_fraction']:.1%}")
        all_results[label] = res
        dump_csv(res, f'geolife_protocol_{label}.csv')

    # ── Significance ──────────────────────────────────────────────────
    print('\n=== Significance (paired sign-flip) ===')
    sig_rows = []
    pairs = [
        ('fullpool', 'unified_route_heavy', 'popularity'),
        ('fullpool', 'unified_route_heavy', 'markov'),
        ('fullpool', 'unified_route_heavy', 'markov2'),
        ('fullpool', 'lr_fusion', 'unified_route_heavy'),
        ('fullpool', 'rrf_uni+markov2+prox', 'unified_route_heavy'),
        ('radius500', 'unified_route_heavy', 'markov'),
        ('radius1000', 'unified_route_heavy', 'markov'),
        ('radius1000', 'lr_fusion', 'markov'),
        ('transfer_repeats', 'rrf_repeat+uni+markov2', 'repeat'),
        ('transfer_repeats', 'rrf_repeat+uni+markov2',
         'unified_route_heavy'),
    ]
    for proto, a, b in pairs:
        res = all_results[proto]
        if a not in res or b not in res:
            continue
        for metric in ('recall@10', 'mrr'):
            d = paired_delta(res[a], res[b], metric=metric)
            sig_rows.append({'protocol': proto, 'a': a, 'b': b,
                             'metric': metric, **d})
            print(f"  [{proto}] {a} vs {b} ({metric}): "
                  f"Δ={d['delta']:+.3f} p={d['p_value']:.3f} "
                  f"n={d['n_common']}")
    # lr_fusion vs route_heavy on the TEST half only (honest for LR)
    fp = all_results['fullpool']
    pair = {}
    for label, res in (('a', fp['lr_fusion']),
                       ('b', fp['unified_route_heavy'])):
        pair[label] = {'per_trajectory': {
            m: {t: v for t, v in res['per_trajectory'][m].items()
                if t not in tune_set}
            for m in ('recall@10', 'mrr')}}
    for metric in ('recall@10', 'mrr'):
        d = paired_delta(pair['a'], pair['b'], metric=metric)
        sig_rows.append({'protocol': 'fullpool_testhalf', 'a': 'lr_fusion',
                         'b': 'unified_route_heavy', 'metric': metric, **d})
        print(f"  [fullpool TEST-HALF] lr_fusion vs unified_route_heavy "
              f"({metric}): Δ={d['delta']:+.3f} p={d['p_value']:.3f} "
              f"n={d['n_common']}")

    pd.DataFrame(sig_rows).to_csv(OUTPUT_DIR / 'geolife_significance.csv',
                                  index=False)
    with open(OUTPUT_DIR / 'geolife_summaries.json', 'w') as fh:
        json.dump(all_results, fh, indent=1, default=str)
    print(f'\nSaved geolife_* to {OUTPUT_DIR}/. Total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
