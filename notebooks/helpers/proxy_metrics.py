"""
Proxy metrics for complementary recommendation.

When the held-out POI is not available (or is the wrong evaluation target),
we measure *properties* of the recommendation list:

    coverage@k        — category alignment with inferred trajectory intent
    novelty@k         — degree of long-tail-ness (1 - mean_popularity_norm)
    diversity@k       — entropy of category distribution inside top-k
    plausibility@k    — fraction matching any trajectory of the same purpose
    spatial_cohesion  — 1 - mean_pairwise_distance / city_scale
    tag_match@k       — fraction whose category aligns with purpose ontology

Each is deterministic, cheap, and does not require labeling.  Together they
catch obvious failure modes (mode-collapse, head-item sprinkling, spatially
incoherent suggestions, off-purpose drift) without requiring a ground-truth
held-out POI.

The single function `compute_proxies(recs, traj_uri, rec, k)` returns a dict
of metric → scalar.  It is intended to be called inside the evaluation loop
alongside recall/MRR.
"""

import math
from collections import Counter
from typing import Dict, List

from .baselines import (
    _haversine,
    dominant_category,
    purpose_categories,
)


CITY_SCALE_METERS = 5000.0  # Rome center diameter order of magnitude


def _top_k_pois(recs: List[Dict], k: int) -> List[str]:
    return [r['poi'] for r in recs[:k]]


def _categories(rec_engine, pois: List[str]) -> List[str]:
    out = []
    for p in pois:
        c = rec_engine.poi_metadata.get(p, {}).get('category', '')
        out.append(c)
    return out


def coverage_at_k(rec_engine, traj_uri: str, input_seq: List[str],
                  pois: List[str]) -> float:
    """Fraction of recs whose category matches the dominant observed category
    or any category in the trajectory's inferred purpose."""
    if not pois:
        return 0.0
    dom = dominant_category(rec_engine, traj_uri, input_seq)
    allowed = {dom} if dom else set()
    allowed |= purpose_categories(rec_engine, traj_uri)
    if not allowed:
        return 0.0
    cats = _categories(rec_engine, pois)
    return sum(1 for c in cats if c in allowed) / len(pois)


def novelty_at_k(rec_engine, pois: List[str]) -> float:
    """1 - mean(poi_popularity / max_popularity).

    High = suggests long-tail. Low = suggests head items. Measures tendency,
    not correctness."""
    if not pois or not rec_engine._poi_popularity:
        return 0.0
    max_pop = max(rec_engine._poi_popularity.values()) if rec_engine._poi_popularity else 1
    if max_pop <= 0:
        return 0.0
    s = 0.0
    for p in pois:
        s += rec_engine._poi_popularity.get(p, 0) / max_pop
    return 1.0 - (s / len(pois))


def diversity_at_k(rec_engine, pois: List[str]) -> float:
    """Normalized Shannon entropy of the category distribution in top-k.

    1.0 = perfectly uniform over k categories; 0.0 = single category."""
    if not pois:
        return 0.0
    cats = _categories(rec_engine, pois)
    counts = Counter(c for c in cats if c)
    if not counts:
        return 0.0
    total = sum(counts.values())
    h = -sum((c / total) * math.log(c / total) for c in counts.values())
    max_h = math.log(min(len(counts), len(pois))) if len(counts) > 1 else 1.0
    return h / max_h if max_h > 0 else 0.0


def plausibility_at_k(rec_engine, traj_uri: str,
                      pois: List[str]) -> float:
    """Fraction of recs that appear in any trajectory sharing this
    trajectory's purpose set.  Category-conditioned popularity sanity
    check."""
    if not pois:
        return 0.0
    target_purposes = rec_engine.traj_metadata.get(traj_uri, {}).get('purposes', set())
    if not target_purposes:
        return 0.0
    matching_pois = set()
    for other_traj, meta in rec_engine.traj_metadata.items():
        if other_traj == traj_uri:
            continue
        if meta.get('purposes', set()) & target_purposes:
            matching_pois |= rec_engine.traj_poi_sets.get(other_traj, set())
    return sum(1 for p in pois if p in matching_pois) / len(pois)


def spatial_cohesion_at_k(rec_engine, pois: List[str]) -> float:
    """1 - mean_pairwise_distance / CITY_SCALE. Capped at [0, 1].

    High = recs are spatially clustered. Low = recs are scattered."""
    coords = []
    for p in pois:
        m = rec_engine.poi_metadata.get(p, {})
        lat, lon = m.get('lat'), m.get('lon')
        if lat is not None and lon is not None:
            coords.append((lat, lon))
    if len(coords) < 2:
        return 0.0
    total = 0.0
    n = 0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            total += _haversine(*coords[i], *coords[j])
            n += 1
    mean_d = total / n if n else 0.0
    return max(0.0, min(1.0, 1.0 - mean_d / CITY_SCALE_METERS))


def tag_match_at_k(rec_engine, traj_uri: str, pois: List[str]) -> float:
    """Fraction of recs whose category is in the purpose→category map for
    any of the trajectory's purposes.  Ontology-grounded version of
    coverage@k (purposes only, not observed-dominant)."""
    if not pois:
        return 0.0
    allowed = purpose_categories(rec_engine, traj_uri)
    if not allowed:
        return 0.0
    cats = _categories(rec_engine, pois)
    return sum(1 for c in cats if c in allowed) / len(pois)


def compute_proxies(recs: List[Dict], traj_uri: str,
                    input_seq: List[str], rec_engine,
                    k: int = 10) -> Dict[str, float]:
    """Return the full proxy vector at k."""
    pois = _top_k_pois(recs, k)
    return {
        f'coverage@{k}':        coverage_at_k(rec_engine, traj_uri, input_seq, pois),
        f'novelty@{k}':         novelty_at_k(rec_engine, pois),
        f'diversity@{k}':       diversity_at_k(rec_engine, pois),
        f'plausibility@{k}':    plausibility_at_k(rec_engine, traj_uri, pois),
        f'spatial_cohesion@{k}': spatial_cohesion_at_k(rec_engine, pois),
        f'tag_match@{k}':       tag_match_at_k(rec_engine, traj_uri, pois),
    }


PROXY_METRICS = [
    'coverage', 'novelty', 'diversity',
    'plausibility', 'spatial_cohesion', 'tag_match',
]
