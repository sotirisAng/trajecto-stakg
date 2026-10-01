"""
Unified evaluation runner for baselines and the unified recommender.

A single loop that:
  1. iterates eligible trajectories (filtered by min_seq_length),
  2. applies a hold-out protocol (trajectory_split or leave_k_out),
  3. calls a supplied ranker function,
  4. computes recall/precision/MRR/MAP against the held-out set,
  5. computes proxy metrics (coverage, novelty, diversity, plausibility,
     spatial_cohesion, tag_match) on the top-k list,
  6. computes the holdout recoverability ceiling so R@k can be normalized.

The ranker contract:
    ranker(rec_engine, traj_uri, input_seq, top_k) -> List[{'poi': str, ...}]

Examples live in notebooks/17-baseline-comparison.ipynb.
"""

import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .proxy_metrics import compute_proxies, PROXY_METRICS


def _split(rec_engine, traj_uri: str, protocol: str,
           split_ratio: float, leave_k: int
           ) -> Tuple[List[str], List[str]]:
    seq = rec_engine.sequences.get(traj_uri, [])
    if protocol == 'trajectory_split':
        if len(seq) < 3:
            return seq, []
        idx = max(2, int(len(seq) * split_ratio))
        return seq[:idx], seq[idx:]
    # leave_k_out
    if len(seq) <= leave_k:
        return seq[:1], seq[1:]
    return seq[:-leave_k], seq[-leave_k:]


def category_recall_at_k(rec_engine, rec_pois: List[str],
                         hold_set: set, k: int) -> float:
    """Fraction of held-out categories represented in the top-k list."""
    hold_cats = {rec_engine.poi_metadata.get(h, {}).get('category', '')
                 for h in hold_set}
    hold_cats.discard('')
    if not hold_cats:
        return 0.0
    top_cats = {rec_engine.poi_metadata.get(p, {}).get('category', '')
                for p in rec_pois[:k]}
    return len(hold_cats & top_cats) / len(hold_cats)


def holdout_ceiling(rec_engine, eligible: List[str], protocol: str,
                    split_ratio: float, leave_k: int,
                    holdout_exclude_categories: Optional[set] = None
                    ) -> float:
    """Fraction of held-out POIs that appear in at least one other
    trajectory. No ranker can exceed this recall ceiling."""
    total, recoverable = 0, 0
    for t in eligible:
        _, hold = _split(rec_engine, t, protocol, split_ratio, leave_k)
        if not hold:
            continue
        other_pois = set()
        for ot, pois in rec_engine.traj_poi_sets.items():
            if ot != t:
                other_pois |= pois
        hold_set = set(hold)
        if holdout_exclude_categories:
            hold_set = {h for h in hold_set
                        if rec_engine.poi_metadata.get(h, {})
                        .get('category', '')
                        not in holdout_exclude_categories}
        total += len(hold_set)
        recoverable += len(hold_set & other_pois)
    return recoverable / total if total else 0.0


def evaluate_method(rec_engine,
                    ranker: Callable,
                    *,
                    method_name: str = 'method',
                    k_values: Tuple[int, ...] = (5, 10, 20),
                    protocol: str = 'trajectory_split',
                    split_ratio: float = 0.7,
                    leave_k: int = 1,
                    min_seq_length: int = 3,
                    proxy_k: int = 10,
                    compute_ceiling: bool = True,
                    verbose: bool = True,
                    holdout_exclude_categories: Optional[set] = None,
                    ) -> Dict:
    """Run one evaluation for a given ranker.

    Returns a dict with recall@k, precision@k, hit_rate@k, mrr, map, n,
    proxy means + CIs, and (optionally) ceiling & normalized_recall@k.

    `holdout_exclude_categories`: drop held-out POIs of these categories
    from the target set. Use with the candidate exclusion list so the
    protocol only scores POIs the system is allowed to recommend;
    otherwise excluded-category holdouts are guaranteed misses for every
    method. The number of dropped targets is reported as
    `holdout_excluded`.
    """
    eligible = [t for t in rec_engine.sequences
                if len(rec_engine.sequences[t]) >= min_seq_length]
    holdout_excluded = 0
    ranker_failures = 0

    per_traj = {
        **{f'recall@{k}': [] for k in k_values},
        **{f'precision@{k}': [] for k in k_values},
        **{f'hit_rate@{k}': [] for k in k_values},
        **{f'cat_recall@{k}': [] for k in k_values},
        'mrr': [], 'map': [],
    }
    # traj_uri-keyed copies for paired significance tests (eval_protocols)
    per_traj_keyed = {f'recall@{k}': {} for k in k_values}
    per_traj_keyed['mrr'] = {}
    proxy_accum = {f'{m}@{proxy_k}': [] for m in PROXY_METRICS}

    t0 = time.time()
    for idx, traj_uri in enumerate(eligible):
        input_seq, hold = _split(
            rec_engine, traj_uri, protocol, split_ratio, leave_k
        )
        if not hold:
            continue

        hold_set = set(hold)
        if holdout_exclude_categories:
            kept = {h for h in hold_set
                    if rec_engine.poi_metadata.get(h, {}).get('category', '')
                    not in holdout_exclude_categories}
            holdout_excluded += len(hold_set) - len(kept)
            hold_set = kept
            if not hold_set:
                continue
        try:
            recs = ranker(rec_engine, traj_uri, input_seq, max(k_values))
        except Exception as exc:
            # Reported, not hidden: a failing trajectory drops out of this
            # method's sample only, which makes methods incomparable.
            ranker_failures += 1
            print(f'    WARNING {method_name}: ranker raised on {traj_uri}: {exc}')
            continue

        rec_pois = [r['poi'] for r in recs]

        for k in k_values:
            top = set(rec_pois[:k])
            hits = len(hold_set & top)
            per_traj[f'recall@{k}'].append(hits / len(hold_set))
            per_traj_keyed[f'recall@{k}'][traj_uri] = hits / len(hold_set)
            per_traj[f'precision@{k}'].append(hits / k)
            per_traj[f'hit_rate@{k}'].append(1.0 if hits > 0 else 0.0)
            # "did we suggest the right kind of place" -- tolerant to the POI
            # identity noise the recoverability ceiling reflects
            per_traj[f'cat_recall@{k}'].append(
                category_recall_at_k(rec_engine, rec_pois, hold_set, k))

        rr = 0.0
        for rank, poi in enumerate(rec_pois, 1):
            if poi in hold_set:
                rr = 1.0 / rank
                break
        per_traj['mrr'].append(rr)
        per_traj_keyed['mrr'][traj_uri] = rr

        n_rel, psum = 0, 0.0
        for rank, poi in enumerate(rec_pois, 1):
            if poi in hold_set:
                n_rel += 1
                psum += n_rel / rank
        per_traj['map'].append(psum / len(hold_set))

        prox = compute_proxies(recs, traj_uri, input_seq, rec_engine, k=proxy_k)
        for key, val in prox.items():
            proxy_accum[key].append(val)

        if verbose and idx > 0 and idx % 25 == 0:
            elapsed = time.time() - t0
            eta = elapsed / idx * (len(eligible) - idx)
            print(f'    {method_name}: {idx}/{len(eligible)} '
                  f'({elapsed:.0f}s, ~{eta:.0f}s remaining)')

    summary = {
        'method': method_name,
        'protocol': protocol,
        'split_ratio': split_ratio if protocol == 'trajectory_split' else None,
        'leave_k': leave_k if protocol == 'leave_k_out' else None,
        'min_seq_length': min_seq_length,
        'n_evaluated': len(per_traj['mrr']),
        'n_ranker_failures': ranker_failures,
        'elapsed_seconds': round(time.time() - t0, 2),
        'per_trajectory': per_traj_keyed,
        'holdout_excluded': holdout_excluded,
    }
    for k in k_values:
        vals = per_traj[f'cat_recall@{k}']
        summary[f'cat_recall@{k}'] = float(np.mean(vals)) if vals else 0.0
    for k in k_values:
        for m in ('recall', 'precision', 'hit_rate'):
            vals = per_traj[f'{m}@{k}']
            summary[f'{m}@{k}'] = float(np.mean(vals)) if vals else 0.0
            summary[f'{m}@{k}_ci'] = rec_engine.bootstrap_ci(vals) if vals else (0.0, 0.0)
    summary['mrr'] = float(np.mean(per_traj['mrr'])) if per_traj['mrr'] else 0.0
    summary['mrr_ci'] = rec_engine.bootstrap_ci(per_traj['mrr']) if per_traj['mrr'] else (0.0, 0.0)
    summary['map'] = float(np.mean(per_traj['map'])) if per_traj['map'] else 0.0
    summary['map_ci'] = rec_engine.bootstrap_ci(per_traj['map']) if per_traj['map'] else (0.0, 0.0)

    for key, vals in proxy_accum.items():
        summary[key] = float(np.mean(vals)) if vals else 0.0

    if compute_ceiling:
        ceiling = holdout_ceiling(
            rec_engine, eligible, protocol, split_ratio, leave_k,
            holdout_exclude_categories=holdout_exclude_categories)
        summary['ceiling_recall'] = ceiling
        for k in k_values:
            r = summary.get(f'recall@{k}', 0.0)
            summary[f'normalized_recall@{k}'] = (r / ceiling) if ceiling > 0 else 0.0

    return summary


def lift_table(results: Dict[str, Dict],
               baseline_keys: Tuple[str, ...] = ('popularity', 'category_popularity'),
               k: int = 10) -> List[Dict]:
    """Convert per-method results into a table with lifts vs each baseline.

    Expects `results` keyed by method name. Returns a list of row dicts."""
    rows = []
    for method, summary in results.items():
        row = {
            'method': method,
            f'R@{k}': summary.get(f'recall@{k}', 0.0),
            f'R@{k}/ceil': summary.get(f'normalized_recall@{k}', 0.0),
            'MRR': summary.get('mrr', 0.0),
            f'Cov@{k}': summary.get(f'coverage@{k}', 0.0),
            f'Nov@{k}': summary.get(f'novelty@{k}', 0.0),
            f'Div@{k}': summary.get(f'diversity@{k}', 0.0),
            f'Plaus@{k}': summary.get(f'plausibility@{k}', 0.0),
            f'Cohes@{k}': summary.get(f'spatial_cohesion@{k}', 0.0),
            f'TagMatch@{k}': summary.get(f'tag_match@{k}', 0.0),
            'N': summary.get('n_evaluated', 0),
        }
        for bk in baseline_keys:
            baseline = results.get(bk)
            if baseline is not None:
                row[f'Δ vs {bk}'] = (
                    summary.get(f'recall@{k}', 0.0)
                    - baseline.get(f'recall@{k}', 0.0)
                )
        rows.append(row)
    return rows
