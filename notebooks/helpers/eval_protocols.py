"""
Use-case-aligned evaluation protocols.

The baseline ladder in notebook 17 compares rankers that each generate
their own candidates over the full POI pool. Under that design geography
is a *competitor*, and every semantic claim has to be argued against the
proximity baseline. These protocols
restructure the evaluation so the numbers carry the claim directly:

1. Geo-constrained re-ranking (`evaluate_within_radius`)
   Proximity stops being a competitor and becomes shared candidate
   generation: every method ranks the SAME radius-bounded candidate set
   around the observed prefix. Measured differences are pure ordering
   quality — semantics, not geography. This matches the deployment
   question ("the visitor is here; which nearby POI?") instead of the
   retrieval question ("which POI anywhere in the city?").

2. Session transfer (`evaluate_user_transfer`)
   Hold out an ENTIRE trajectory of a user; recommend from their other
   trajectories. Matches the "returning visitor" use case and removes a
   leak the split protocol has: trajectory tags (purpose, pattern,
   engagement) are inferred from the full trajectory including held-out
   POIs, so within-trajectory splits let every tag-consuming signal peek
   at the answer. Session transfer scores against tags/preferences built
   from *other* sessions only.

3. Category-level recall (both protocols + `category_recall_at_k`)
   A hit when the top-k contains a POI of the same category as a
   held-out POI. Tolerant to Overpass identity noise and to the
   POI-identity ceiling; measures the semantic claim — the
   system suggests the right *kind* of place — at the granularity the
   tag layer actually produces.

4. Paired significance (`paired_delta`)
   Sign-flip permutation test on per-trajectory paired differences.
   Ladder gaps get p-values instead of narrative.

Ranker contract is unchanged from baseline_evaluation:
    ranker(rec_engine, traj_uri, input_seq, top_k) -> List[{'poi', ...}]
Rankers MAY accept an extra keyword `exclude_traj`; the session-transfer
loop passes it when supported so flow/popularity structures can drop the
held-out trajectory's contribution.
"""

import inspect
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .baselines import _haversine
from .baseline_evaluation import _split, category_recall_at_k
from .context_recommender import EXCLUDED_CATEGORIES
from .proxy_metrics import compute_proxies, PROXY_METRICS


# ── Shared geometry helpers ────────────────────────────────────────────

def _coords(rec_engine, poi: str) -> Optional[Tuple[float, float]]:
    m = rec_engine.poi_metadata.get(poi, {})
    lat, lon = m.get('lat'), m.get('lon')
    if lat is None or lon is None:
        return None
    return (lat, lon)


def prefix_centroid(rec_engine, input_seq: List[str]
                    ) -> Optional[Tuple[float, float]]:
    pts = [c for c in (_coords(rec_engine, p) for p in input_seq) if c]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts))


def radius_candidates(rec_engine, center: Tuple[float, float],
                      radius_m: float, visited: set) -> set:
    """Unvisited, non-excluded pool POIs within radius of center."""
    out = set()
    for p in rec_engine.all_poi_uris:
        if p in visited:
            continue
        if rec_engine.poi_metadata.get(p, {}).get('category', '') \
                in EXCLUDED_CATEGORIES:
            continue
        c = _coords(rec_engine, p)
        if c and _haversine(center[0], center[1], c[0], c[1]) <= radius_m:
            out.add(p)
    return out


# ── Ranker invocation ──────────────────────────────────────────────────

def _accepts_exclude(ranker: Callable) -> bool:
    try:
        sig = inspect.signature(ranker)
    except (TypeError, ValueError):
        return False
    return 'exclude_traj' in sig.parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values())


def _accepts(ranker: Callable, name: str) -> bool:
    try:
        sig = inspect.signature(ranker)
    except (TypeError, ValueError):
        return False
    return name in sig.parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values())


def _call_ranker(ranker, rec_engine, traj_uri, input_seq, top_k,
                 exclude_traj=None, restrict_to=None):
    kw = {}
    if exclude_traj is not None and _accepts_exclude(ranker):
        kw['exclude_traj'] = exclude_traj
    if restrict_to is not None and _accepts(ranker, 'restrict_to'):
        kw['restrict_to'] = restrict_to
    return ranker(rec_engine, traj_uri, input_seq, top_k, **kw)


# ── Protocol 1: geo-constrained re-ranking ─────────────────────────────

def evaluate_within_radius(rec_engine,
                           ranker: Callable,
                           *,
                           method_name: str = 'method',
                           radius_m: float = 500.0,
                           k_values: Tuple[int, ...] = (5, 10),
                           protocol: str = 'trajectory_split',
                           split_ratio: float = 0.7,
                           leave_k: int = 1,
                           min_seq_length: int = 3,
                           proxy_k: int = 10,
                           verbose: bool = False,
                           ) -> Dict:
    """Rank the shared radius-bounded candidate set.

    The candidate set is passed INTO each ranker (`restrict_to`), so every
    method ranks exactly the same nearby POIs. Rankers that cannot accept
    the restriction fall back to output intersection.

    Output intersection alone is only equivalent to ranking the restricted
    set for score-monotonic rankers, and the unified stack is not one: a
    score floor, a diversity penalty computed over the global list, and a
    purpose gate all reorder or truncate it. Under the old output-filtering
    path the stack returned a median of 20 POIs corpus-wide, of which a
    median of 2 fell inside a 500 m radius, so it was scored on recall@10
    while able to fill two slots. That measured list length, not ordering.

    Held-out POIs outside the radius are dropped from the target set
    (the question is "which nearby POI", not "which POI anywhere");
    the dropped fraction is reported as `holdout_out_of_radius`.
    """
    eligible = [t for t in rec_engine.sequences
                if len(rec_engine.sequences[t]) >= min_seq_length]
    full_k = len(rec_engine.all_poi_uris)

    per_traj = defaultdict(dict)   # metric -> {traj: value}
    proxy_accum = {f'{m}@{proxy_k}': [] for m in PROXY_METRICS}
    cand_sizes, hold_total, hold_in = [], 0, 0
    ceiling_num, ceiling_den = 0, 0

    for traj_uri in eligible:
        input_seq, hold = _split(rec_engine, traj_uri, protocol,
                                 split_ratio, leave_k)
        if not hold:
            continue
        center = prefix_centroid(rec_engine, input_seq)
        if center is None:
            continue
        visited = set(input_seq)
        cand = radius_candidates(rec_engine, center, radius_m, visited)
        if not cand:
            continue

        hold_set_all = set(hold)
        hold_set = hold_set_all & cand
        hold_total += len(hold_set_all)
        hold_in += len(hold_set)
        if not hold_set:
            continue
        cand_sizes.append(len(cand))

        # In-radius retrieval ceiling: held-out POI appears in >=1 other
        # trajectory (co-visit / popularity / flow methods can only ever
        # surface such POIs).
        for h in hold_set:
            ceiling_den += 1
            for ot, pois in rec_engine.traj_poi_sets.items():
                if ot != traj_uri and h in pois:
                    ceiling_num += 1
                    break

        try:
            # exclude_traj=traj_uri: rankers that support it drop the
            # target trajectory's own contribution to popularity/flow
            # counts (self-count leak).
            recs = _call_ranker(ranker, rec_engine, traj_uri, input_seq,
                                full_k, exclude_traj=traj_uri,
                                restrict_to=cand)
        except Exception as exc:
            if verbose:
                print(f'    {method_name}: ranker raised on '
                      f'{traj_uri}: {exc}')
            continue
        ranked = [r['poi'] for r in recs if r['poi'] in cand]

        for k in k_values:
            top = set(ranked[:k])
            hits = len(hold_set & top)
            per_traj[f'recall@{k}'][traj_uri] = hits / len(hold_set)
            per_traj[f'hit_rate@{k}'][traj_uri] = 1.0 if hits else 0.0
            per_traj[f'cat_recall@{k}'][traj_uri] = category_recall_at_k(
                rec_engine, ranked, hold_set, k)

        rr = 0.0
        for rank, poi in enumerate(ranked, 1):
            if poi in hold_set:
                rr = 1.0 / rank
                break
        per_traj['mrr'][traj_uri] = rr

        recs_in = [r for r in recs if r['poi'] in cand]
        prox = compute_proxies(recs_in, traj_uri, input_seq, rec_engine,
                               k=proxy_k)
        for key, val in prox.items():
            proxy_accum[key].append(val)

    summary = {
        'method': method_name,
        'protocol': f'within_radius_{int(radius_m)}m',
        'radius_m': radius_m,
        'min_seq_length': min_seq_length,
        'n_evaluated': len(per_traj.get('mrr', {})),
        'mean_candidates': float(np.mean(cand_sizes)) if cand_sizes else 0.0,
        'holdout_out_of_radius':
            1.0 - (hold_in / hold_total) if hold_total else 0.0,
        'ceiling_recall':
            ceiling_num / ceiling_den if ceiling_den else 0.0,
        'per_trajectory': {m: dict(v) for m, v in per_traj.items()},
    }
    for metric, values in per_traj.items():
        vals = list(values.values())
        summary[metric] = float(np.mean(vals)) if vals else 0.0
    for k in k_values:
        c = summary['ceiling_recall']
        summary[f'normalized_recall@{k}'] = (
            summary.get(f'recall@{k}', 0.0) / c if c else 0.0)
    for key, vals in proxy_accum.items():
        summary[key] = float(np.mean(vals)) if vals else 0.0
    return summary


# ── Protocol 2: session transfer ───────────────────────────────────────

def eligible_transfer_users(rec_engine, min_seq_length: int = 3
                            ) -> Dict[str, List[str]]:
    """user -> list of eligible trajectories, for users with >= 2."""
    out = {}
    for user, trajs in rec_engine.user_to_trajs.items():
        elig = [t for t in trajs
                if len(rec_engine.sequences.get(t, [])) >= min_seq_length]
        if len(elig) >= 2:
            out[user] = elig
    return out


def evaluate_user_transfer(rec_engine,
                           ranker: Callable,
                           *,
                           method_name: str = 'method',
                           k_values: Tuple[int, ...] = (5, 10, 20),
                           min_seq_length: int = 3,
                           verbose: bool = False,
                           holdout_exclude_categories: Optional[set] = None,
                           include_repeats: bool = False,
                           ) -> Dict:
    """Leave-one-trajectory-out per user.

    For each user with >= 2 eligible trajectories, hold out each
    trajectory in turn. The ranker sees the user's OTHER trajectories
    (concatenated as the observed sequence, with the longest one as the
    reference trajectory for tag lookups) and must recover the held-out
    trajectory's novel POIs — POIs the user visited in the target
    session but in none of the source sessions.

    Tags consumed by tag-aware rankers belong to source trajectories
    only, so the tag-leak in within-trajectory splits is absent here.
    `exclude_traj` (the held-out trajectory) is forwarded to rankers
    that accept it so flow/popularity edges drop its contribution.

    `include_repeats=True` keeps the target session's ALL POIs as the
    holdout (repeats allowed) instead of only the novel ones — the
    protocol for revisit-aware recommendation. Standard rankers exclude
    visited POIs from their candidates, so pair this with repeat-aware
    rankers (rec_levers.repeat_rec or a fusion with it).
    `holdout_exclude_categories` drops non-recommendable target POIs,
    mirroring evaluate_method's protocol-B rule.
    """
    users = eligible_transfer_users(rec_engine, min_seq_length)
    per_traj = defaultdict(dict)
    repeat_fracs = []
    ceiling_num, ceiling_den = 0, 0

    for user, trajs in users.items():
        for target in trajs:
            sources = [t for t in trajs if t != target]
            input_seq = []
            for s in sorted(sources,
                            key=lambda t: -len(rec_engine.sequences[t])):
                input_seq.extend(rec_engine.sequences[s])
            ref_traj = max(sources,
                           key=lambda t: len(rec_engine.sequences[t]))

            target_pois = set(rec_engine.sequences[target])
            novel = target_pois - set(input_seq)
            if target_pois:
                repeat_fracs.append(
                    1.0 - len(novel) / len(target_pois))
            hold_set = target_pois if include_repeats else novel
            if holdout_exclude_categories:
                hold_set = {
                    h for h in hold_set
                    if rec_engine.poi_metadata.get(h, {})
                    .get('category', '') not in holdout_exclude_categories}
            if not hold_set:
                continue

            for h in hold_set:
                ceiling_den += 1
                for ot, pois in rec_engine.traj_poi_sets.items():
                    if ot != target and h in pois:
                        ceiling_num += 1
                        break

            try:
                recs = _call_ranker(ranker, rec_engine, ref_traj,
                                    input_seq, max(k_values),
                                    exclude_traj=target)
            except Exception as exc:
                if verbose:
                    print(f'    {method_name}: ranker raised on '
                          f'{target}: {exc}')
                continue
            ranked = [r['poi'] for r in recs]

            for k in k_values:
                top = set(ranked[:k])
                hits = len(hold_set & top)
                per_traj[f'recall@{k}'][target] = hits / len(hold_set)
                per_traj[f'hit_rate@{k}'][target] = 1.0 if hits else 0.0
                per_traj[f'cat_recall@{k}'][target] = category_recall_at_k(
                    rec_engine, ranked, hold_set, k)

            rr = 0.0
            for rank, poi in enumerate(ranked, 1):
                if poi in hold_set:
                    rr = 1.0 / rank
                    break
            per_traj['mrr'][target] = rr

    summary = {
        'method': method_name,
        'protocol': ('user_transfer_repeats' if include_repeats
                     else 'user_transfer'),
        'min_seq_length': min_seq_length,
        'n_users': len(users),
        'n_evaluated': len(per_traj.get('mrr', {})),
        'mean_repeat_fraction':
            float(np.mean(repeat_fracs)) if repeat_fracs else 0.0,
        'ceiling_recall':
            ceiling_num / ceiling_den if ceiling_den else 0.0,
        'per_trajectory': {m: dict(v) for m, v in per_traj.items()},
    }
    for metric, values in per_traj.items():
        vals = list(values.values())
        summary[metric] = float(np.mean(vals)) if vals else 0.0
    for k in k_values:
        c = summary['ceiling_recall']
        summary[f'normalized_recall@{k}'] = (
            summary.get(f'recall@{k}', 0.0) / c if c else 0.0)
    return summary


# ── Paired significance ────────────────────────────────────────────────

def paired_delta(summary_a: Dict, summary_b: Dict,
                 metric: str = 'recall@10',
                 n_perm: int = 10000, seed: int = 7,
                 traj_to_user: Optional[Dict[str, str]] = None) -> Dict:
    """Sign-flip permutation test on per-trajectory paired differences.

    Both summaries must come from the SAME protocol run (same splits) and
    carry `per_trajectory`. Returns the mean difference (a − b), a 95%
    bootstrap CI, the two-sided permutation p-value, and n_common.

    `traj_to_user` switches the test from per-trajectory to per-USER sign
    flipping. The unclustered test treats every trajectory as independent,
    which these corpora are not: 87 of Rome's 107 eligible trajectories
    belong to 18 users, so its effective sample is nearer 38 than 107 and its
    p-values are optimistic. Drawing one sign per user and applying it to all
    that user's differences respects the clustering. Pass
    `rec_engine.traj_to_user`.
    """
    pa = summary_a.get('per_trajectory', {}).get(metric, {})
    pb = summary_b.get('per_trajectory', {}).get(metric, {})
    common = sorted(set(pa) & set(pb))
    if not common:
        return {'delta': 0.0, 'ci': (0.0, 0.0), 'p_value': 1.0,
                'n_common': 0}
    diffs = np.array([pa[t] - pb[t] for t in common])
    rng = np.random.default_rng(seed)

    observed = float(diffs.mean())
    if traj_to_user:
        # one sign per cluster, broadcast to that cluster's trajectories
        users = [traj_to_user.get(t, t) for t in common]
        uniq = sorted(set(users))
        idx = np.array([uniq.index(u) for u in users])
        flips = rng.choice([-1.0, 1.0], size=(n_perm, len(uniq)))[:, idx]
    else:
        flips = rng.choice([-1.0, 1.0], size=(n_perm, len(diffs)))
    perm_means = (flips * diffs).mean(axis=1)
    p = float((np.abs(perm_means) >= abs(observed)).mean())

    boots = np.array([
        diffs[rng.integers(0, len(diffs), len(diffs))].mean()
        for _ in range(2000)
    ])
    ci = (float(np.percentile(boots, 2.5)),
          float(np.percentile(boots, 97.5)))
    out = {'delta': observed, 'ci': ci, 'p_value': p,
           'n_common': len(common)}
    if traj_to_user:
        out['n_clusters'] = len(set(traj_to_user.get(t, t) for t in common))
    return out


def significance_matrix(results: Dict[str, Dict],
                        metric: str = 'recall@10') -> List[Dict]:
    """Pairwise paired_delta over all methods in a results dict."""
    rows = []
    names = list(results)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = paired_delta(results[a], results[b], metric=metric)
            rows.append({'a': a, 'b': b, 'metric': metric, **d})
    return rows
