#!/usr/bin/env python3
"""Does the Stage 9 EWMA decay earn its place over a plain average?

Every model reads only trajectories that ended before the evaluated one
started, and scores each category from per-trajectory observations
o_c in {1.0, 0.75, 0.5} (strongest Dominant / Immersive / Meaningful
interaction with that category, as in profile_learning.INTERACTION_SCORES).
Scores are scaled to a per-user maximum of 1 so only the decay differs.

  none              signal ablated
  snapshot          read from the graph: the user's latest Stage 9
                    ProfileSnapshot that precedes the trajectory
  mean              no decay: mean of a category's observations (EWMA, alpha=1/n)
  ewma-0.15         the thesis EWMA (decay per observation of the category)
  ewma-0.3, -0.5    faster per-observation decay
  sum               no decay, frequency-weighted: sum of observations
  time-30d, -180d   sum of observations discounted by elapsed time,
                    2^(-days/half_life), so unvisited categories fade
  *-pass            as mean / ewma, but a category the trajectory passed
                    without a meaningful interaction is observed as 0

    TRAJECTO_KG=geolife python run_decay_ablation.py
"""

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "output" / "evaluation"

from helpers.unified_recommender import UnifiedRecommender, UNIFIED_WEIGHTS
from helpers.baseline_evaluation import evaluate_method
from helpers.context_recommender import EXCLUDED_CATEGORIES
from helpers.eval_protocols import paired_delta
from helpers.fuseki_config import graph_name as KG
from helpers.profile_learning import INTERACTION_SCORES
from helpers.fuseki_config import query_endpoint
from run_preference_ablation import snapshots
from bisect import bisect_left

ROUTE_HEAVY = {'archetype_sim': 0.10, 'semantic_sim': 0.10,
               'route_sim': 0.30, 'flow_prob': 0.20, 'dominant_ctx': 0.08,
               'user_pref': 0.08, 'significant': 0.05, 'engagement': 0.04,
               'rule_coverage': 0.05}
SKIP = set(EXCLUDED_CATEGORIES) | {'PointOfInterest', 'SignificantPOI', 'Unknown', ''}


def observations(interactions, passing=False):
    """category -> strongest interaction score in one trajectory. With
    passing, a category the trajectory only passed is observed as 0."""
    obs = {}
    for it in interactions:
        c = it['category']
        if c in SKIP:
            continue
        s = INTERACTION_SCORES.get(it['int_type'])
        if s:
            obs[c] = max(obs.get(c, 0), s)
        elif passing and it['int_type'] == 'PassingInteraction':
            obs.setdefault(c, 0.0)
    return obs


def ewma(history, alpha):
    p = {}
    for _, obs in history:
        for c, o in obs.items():
            p[c] = o if c not in p else (1 - alpha) * p[c] + alpha * o
    return p


def mean(history):
    tot, n = defaultdict(float), defaultdict(int)
    for _, obs in history:
        for c, o in obs.items():
            tot[c] += o; n[c] += 1
    return {c: tot[c] / n[c] for c in tot}


def discounted_sum(history, now, half_life_days=None):
    p = defaultdict(float)
    for end, obs in history:
        w = 1.0 if half_life_days is None else \
            2 ** (-(now - end).total_seconds() / 86400 / half_life_days)
        for c, o in obs.items():
            p[c] += w * o
    return p


def as_prefs(scores):
    top = max(scores.values(), default=0)
    if top <= 0:
        return []
    return sorted(({'category': c, 'score': s / top} for c, s in scores.items() if s > 0),
                  key=lambda x: -x['score'])


def demo():
    from datetime import datetime, timedelta
    t0 = datetime(2020, 1, 1)
    h = [(t0, {'Museum': 1.0}), (t0 + timedelta(days=30), {'Museum': 0.5, 'Cafe': 0.5})]
    e = ewma(h, 0.15)
    assert abs(e['Museum'] - 0.925) < 1e-9 and e['Cafe'] == 0.5
    assert mean(h) == {'Museum': 0.75, 'Cafe': 0.5}
    d = discounted_sum(h, t0 + timedelta(days=30), 30)
    assert abs(d['Museum'] - 1.0) < 1e-9 and d['Cafe'] == 0.5
    o = observations([{'category': 'Museum', 'int_type': 'PassingInteraction'},
                      {'category': 'Cafe', 'int_type': 'PassingInteraction'},
                      {'category': 'Cafe', 'int_type': 'MeaningfulInteraction'}], passing=True)
    assert o == {'Museum': 0.0, 'Cafe': 0.5}
    assert ewma([(t0, {'Museum': 1.0}), (t0, {'Museum': 0.0})], 0.15)['Museum'] == 0.85


def main():
    demo()
    feats = ('./output/semantic_features/geolife_features.json' if 'geolife' in KG
             else './output/semantic_features/trajectory_features.json')
    rec = UnifiedRecommender()
    rec.load_all_data(features_path=feats)
    win = rec._traj_time_window
    user_trajs = defaultdict(list)
    for t, u in rec.traj_to_user.items():
        if t in win:
            user_trajs[u].append(t)
    obs = {t: observations(rec.interaction_data.get(t, [])) for t in rec.traj_to_user}
    obs_pass = {t: observations(rec.interaction_data.get(t, []), passing=True)
                for t in rec.traj_to_user}

    def history(t, u, _obs=obs):
        if t not in win:
            return []
        s0 = win[t][0]
        return sorted(((win[x][1], _obs[x]) for x in user_trajs[u]
                       if x != t and win[x][1] < s0), key=lambda h: h[0])

    snaps = snapshots(query_endpoint)

    def snapshot(t, u):
        seq = snaps.get(u, [])
        if not seq or t not in win:
            return {}
        i = bisect_left([ts for ts, _ in seq], win[t][0].isoformat()[:19])
        return {p['category']: p['score'] for p in seq[i - 1][1]} if i else {}

    models = {
        'none': lambda t, u: {},
        'snapshot': snapshot,
        'mean': lambda t, u: mean(history(t, u)),
        'ewma-0.15': lambda t, u: ewma(history(t, u), 0.15),
        'ewma-0.3': lambda t, u: ewma(history(t, u), 0.3),
        'ewma-0.5': lambda t, u: ewma(history(t, u), 0.5),
        'sum': lambda t, u: discounted_sum(history(t, u), None),
        'time-30d': lambda t, u: discounted_sum(history(t, u), win[t][0], 30) if t in win else {},
        'time-180d': lambda t, u: discounted_sum(history(t, u), win[t][0], 180) if t in win else {},
        'mean-pass': lambda t, u: mean(history(t, u, obs_pass)),
        'ewma-pass-0.15': lambda t, u: ewma(history(t, u, obs_pass), 0.15),
        'ewma-pass-0.3': lambda t, u: ewma(history(t, u, obs_pass), 0.3),
    }

    rows, summ = [], {}
    for label, w in (('default', UNIFIED_WEIGHTS), ('route-heavy', ROUTE_HEAVY)):
        for name, model in models.items():
            def ranker(r, t, seq, k, _w=w, _m=model):
                r._preferences = lambda u, ex, _t=t: as_prefs(_m(_t, u)) if u else []
                return r.recommend(t, observed_sequence=seq, top_k=k,
                                   signal_weights=_w, exclude_traj=t)
            s = evaluate_method(rec, ranker, method_name=f'{name}/{label}',
                                k_values=(10,), min_seq_length=3, verbose=False,
                                holdout_exclude_categories=set(EXCLUDED_CATEGORIES))
            summ[(name, label)] = s
            rows.append({'kg': KG, 'model': name, 'weights': label, 'n': s['n_evaluated'],
                         'recall@10': s['recall@10'], 'mrr': s['mrr']})
            print(f"{label:12} {name:10} R@10={s['recall@10']:.4f} MRR={s['mrr']:.4f}", flush=True)
        for name in models:
            if name == 'ewma-0.15':
                continue
            for m in ('recall@10', 'mrr'):
                d = paired_delta(summ[(name, label)], summ[('ewma-0.15', label)], metric=m,
                                 traj_to_user=rec.traj_to_user)
                row = next(r for r in rows if r['model'] == name and r['weights'] == label)
                row[f'd_{m}_vs_ewma'] = d['delta']; row[f'p_{m}'] = d['p_value']

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f'decay_ablation_{KG}.csv', index=False)
    print(df.to_string(index=False, float_format='%.4f'))


if __name__ == '__main__':
    main()
