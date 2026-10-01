"""Within-radius protocol on a shared candidate set: the POIs within the
radius are passed into each ranker, rather than filtering each ranker's output,
so every method ranks the same candidates."""
import os
import re
import sys
import argparse
import atexit
import pathlib
sys.path.insert(0, '.')
from functools import partial
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--kg', default='osm-rome')
ap.add_argument('--tag', default='osm_rome')
a = ap.parse_args()

# The dataset is selected by rewriting graph_name in fuseki_config, the same
# way run_clustered_significance.py does it. An env var does nothing: that
# module hardcodes the name. Restored on exit so a crash cannot leave the
# config pointing at the wrong corpus.
CFG = pathlib.Path('helpers/fuseki_config.py')
_original = CFG.read_text()
atexit.register(lambda: CFG.write_text(_original))
CFG.write_text(re.sub(r'^graph_name = "[^"]*"', f'graph_name = "{a.kg}"',
                      _original, count=1, flags=re.M))
for _m in [m for m in list(sys.modules) if 'helpers' in m]:
    del sys.modules[_m]

from helpers.unified_recommender import UnifiedRecommender, UNIFIED_WEIGHTS
from helpers.baselines import (popularity_rec, category_popularity_rec,
                               proximity_rec, markov_rec)
from helpers.rec_levers import markov2_rec, rrf
from helpers.eval_protocols import evaluate_within_radius
import numpy as np
from helpers.context_recommender import EXCLUDED_CATEGORIES

ROUTE_HEAVY = {'archetype_sim': 0.10, 'semantic_sim': 0.10, 'route_sim': 0.30,
               'flow_prob': 0.20, 'dominant_ctx': 0.08, 'user_pref': 0.08,
               'significant': 0.05, 'engagement': 0.04, 'rule_coverage': 0.05}

def unified_rec(rec_engine, traj_uri, input_seq, top_k,
                weights=None, exclude_traj=None, restrict_to=None):
    return rec_engine.recommend(traj_uri, observed_sequence=input_seq,
        top_k=top_k, signal_weights=weights,
        exclude_traj=exclude_traj or traj_uri, restrict_to=restrict_to)

def random_ranker(rec_engine, traj_uri, input_seq, top_k,
                  restrict_to=None, _rng=np.random.default_rng(42)):
    pool = rec_engine.all_poi_uris if restrict_to is None else set(restrict_to)
    cands = sorted(p for p in pool - set(input_seq)
                   if rec_engine.poi_metadata.get(p, {}).get('category','')
                   not in EXCLUDED_CATEGORIES)
    _rng.shuffle(cands)
    return [{'poi': p, 'label': '', 'category': '', 'score': 0.0, 'signals': {}}
            for p in cands[:top_k]]

def _excl(fn):
    def w(re_, t, i, k, exclude_traj=None, restrict_to=None):
        return fn(re_, t, i, k, exclude_traj=exclude_traj or t,
                  restrict_to=restrict_to)
    return w

feats = os.environ.get('TRAJECTO_FEATURES') or (
        'output/semantic_features/trajectory_features.json'
         if a.kg == 'osm-rome'
         else 'output/semantic_features-geolife/trajectory_features.json')
rec = UnifiedRecommender()
if a.kg not in rec.endpoint:
    raise RuntimeError(f'endpoint {rec.endpoint} does not match --kg {a.kg}')
rec.load_all_data(features_path=feats)
print(f'CORPUS CHECK: endpoint={rec.endpoint}, '
      f'trajectories={len(rec.sequences)}, POIs={len(rec.all_poi_uris)}')

uni_rh = partial(unified_rec, weights=ROUTE_HEAVY)
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
}

for radius in (500.0, 1000.0):
    rows = []
    print(f'\n=== within {int(radius)} m ===', flush=True)
    for name, fn in methods.items():
        s = evaluate_within_radius(rec, fn, method_name=name,
                                   radius_m=radius, k_values=(10,))
        rows.append({'method': name, 'n_evaluated': s['n_evaluated'],
                     'recall@10': s['recall@10'],
                     'normalized_recall@10': s.get('normalized_recall@10'),
                     'mrr': s['mrr'], 'ceiling_recall': s.get('ceiling_recall')})
        print(f"  {name:<24} R@10={s['recall@10']:.3f}  MRR={s['mrr']:.3f}  N={s['n_evaluated']}", flush=True)
    df = pd.DataFrame(rows)
    out = (os.environ.get('TRAJECTO_EVAL_DIR', 'output/evaluation')
           + f'/{a.tag}_protocol_radius{int(radius)}_fixed.csv')
    df.to_csv(out, index=False)
    print(f'  -> {out}')
