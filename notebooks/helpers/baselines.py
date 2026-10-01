"""
Baseline rankers for recommender evaluation.

Each baseline is a free function that takes a loaded recommender (providing
the shared data: sequences, traj_poi_sets, poi_metadata, flow_graph,
traj_metadata, user_preferences, _poi_popularity, all_poi_uris) and returns
a ranked list of dicts compatible with the evaluation pipeline:
    [{'poi', 'label', 'category', 'score', 'signals'}, ...]

Baselines intentionally avoid any logic from the unified stack. They exist
to answer: "what lift does the nine-signal model provide over trivial priors?"
"""

import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from .context_recommender import EXCLUDED_CATEGORIES

# Purpose → category mapping mirrors unified_recommender._PURPOSE_CATEGORY_MAP
# Duplicated here to avoid a dependency cycle with unified_recommender.
# poi_metadata categories are ontology local names (CamelCase), so each
# multi-word display name needs its concatenated variant or it never matches.
_PURPOSE_CATEGORY_MAP = {
    'CulturalPurpose': {'Museum', 'Gallery', 'Artwork', 'Historical Landmark',
                        'Historical Site', 'Monument', 'Cultural', 'Religious Site',
                        'HistoricalLandmark', 'HistoricalSite', 'ReligiousSite',
                        'ArtworkPOI', 'HeritageBuilding', 'CulturalPOI'},
    'CommercialPurpose': {'Retail Store', 'Shopping Mall', 'Market Area',
                          'RetailStore', 'ShoppingMall', 'MarketArea',
                          'CommercialPOI'},
    'RecreationalPurpose': {'Park', 'Beach', 'Leisure Facility',
                            'Sports Facility', 'Natural Landmark',
                            'LeisureFacility', 'SportsFacility',
                            'NaturalLandmark', 'RecreationalPOI'},
    'EducationalPurpose': {'Library', 'University', 'Educational', 'Museum',
                           'EducationalPOI'},
    'AccommodationPurpose': {'Hotel', 'Accommodation', 'AccommodationPOI'},
    'CafeVisit': {'Cafe', 'Bar', 'Food Court', 'FoodCourt'},
    'NightlifeActivity': {'Nightlife Facility', 'Bar', 'Cafe',
                          'NightlifeFacility'},
    'HealthcarePurpose': {'Healthcare Facility', 'Medical Facility',
                          'HealthcareFacility', 'MedicalFacility'},
}


# ── Helpers ────────────────────────────────────────────────────────────

def _candidate_filter(rec, visited: set, restrict_to: set = None) -> set:
    """Unvisited POIs with a non-excluded category.

    `restrict_to` bounds the pool to a caller-supplied candidate set, so
    that geo-constrained protocols can hand every ranker the SAME
    candidates instead of filtering each ranker's output afterwards.
    Output filtering is only equivalent for score-monotonic rankers, and
    the unified stack is not one (score floor, diversity penalty,
    purpose gate), so it must receive the restriction up front."""
    pool = rec.all_poi_uris if restrict_to is None else set(restrict_to)
    return {p for p in pool - visited
            if rec.poi_metadata.get(p, {}).get('category', '')
            not in EXCLUDED_CATEGORIES}


def _meta(rec, poi: str) -> Dict:
    return rec.poi_metadata.get(poi, {})


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _pack(rec, poi: str, score: float, signal: str) -> Dict:
    m = _meta(rec, poi)
    return {
        'poi': poi,
        'label': m.get('label', poi.split('/')[-1]),
        'category': m.get('category', ''),
        'score': float(score),
        'signals': {signal: float(score)},
    }


def dominant_category(rec, traj_uri: str, input_seq: List[str]) -> Optional[str]:
    """Most frequent non-excluded category among observed POIs."""
    counts = Counter()
    for p in input_seq:
        c = _meta(rec, p).get('category', '')
        if c and c not in EXCLUDED_CATEGORIES:
            counts[c] += 1
    if counts:
        return counts.most_common(1)[0][0]
    # Fallback: first category implied by trajectory purpose
    purposes = rec.traj_metadata.get(traj_uri, {}).get('purposes', set())
    for p in purposes:
        cats = _PURPOSE_CATEGORY_MAP.get(p, set())
        if cats:
            return sorted(cats)[0]
    return None


def purpose_categories(rec, traj_uri: str) -> set:
    """Union of categories implied by a trajectory's purposes."""
    purposes = rec.traj_metadata.get(traj_uri, {}).get('purposes', set())
    out = set()
    for p in purposes:
        out |= _PURPOSE_CATEGORY_MAP.get(p, set())
    return out


# ── Baselines ──────────────────────────────────────────────────────────

def _popularity(rec, poi: str, exclude_traj: Optional[str] = None) -> int:
    """Trajectory-count popularity, minus the excluded trajectory's visit."""
    pop = rec._poi_popularity.get(poi, 0)
    if exclude_traj and poi in rec.traj_poi_sets.get(exclude_traj, set()):
        pop -= 1
    return max(pop, 0)


def popularity_rec(rec, traj_uri: str, input_seq: List[str],
                   top_k: int, exclude_traj: Optional[str] = None,
                   restrict_to: set = None) -> List[Dict]:
    """Global popularity: top-k POIs by total visit count, excluding visited.

    `exclude_traj` removes that trajectory's contribution to the counts
    (used by hold-out protocols where the target trajectory must not
    feed its own popularity)."""
    visited = set(input_seq)
    candidates = _candidate_filter(rec, visited, restrict_to)
    ranked = sorted(
        ((p, _popularity(rec, p, exclude_traj)) for p in candidates),
        key=lambda x: (-x[1], x[0]),
    )
    max_pop = ranked[0][1] if ranked else 1
    return [_pack(rec, p, c / max(max_pop, 1), 'popularity')
            for p, c in ranked[:top_k]]


def category_popularity_rec(rec, traj_uri: str, input_seq: List[str],
                            top_k: int, exclude_traj: Optional[str] = None,
                            restrict_to: set = None) -> List[Dict]:
    """Popularity restricted to the trajectory's dominant category.

    Fallback order if no candidates remain:
        dominant_category → any category in purpose_categories → global.
    """
    visited = set(input_seq)
    candidates = _candidate_filter(rec, visited, restrict_to)

    dom = dominant_category(rec, traj_uri, input_seq)
    allowed = {dom} if dom else set()
    allowed |= purpose_categories(rec, traj_uri)

    if allowed:
        filtered = {p for p in candidates
                    if _meta(rec, p).get('category', '') in allowed}
        if filtered:
            candidates = filtered

    ranked = sorted(
        ((p, _popularity(rec, p, exclude_traj)) for p in candidates),
        key=lambda x: (-x[1], x[0]),
    )
    max_pop = ranked[0][1] if ranked else 1
    return [_pack(rec, p, c / max(max_pop, 1), 'category_popularity')
            for p, c in ranked[:top_k]]


def proximity_rec(rec, traj_uri: str, input_seq: List[str],
                  top_k: int, restrict_to: set = None) -> List[Dict]:
    """Geographic proximity to observed-so-far centroid.

    If the input sequence has no georeferenced POIs, returns nothing.
    """
    visited = set(input_seq)
    candidates = _candidate_filter(rec, visited, restrict_to)

    coords = []
    for p in input_seq:
        m = _meta(rec, p)
        if m.get('lat') is not None and m.get('lon') is not None:
            coords.append((m['lat'], m['lon']))
    if not coords:
        return []

    clat = sum(c[0] for c in coords) / len(coords)
    clon = sum(c[1] for c in coords) / len(coords)

    scored = []
    for p in candidates:
        m = _meta(rec, p)
        if m.get('lat') is None or m.get('lon') is None:
            continue
        d = _haversine(clat, clon, m['lat'], m['lon'])
        scored.append((p, d))
    scored.sort(key=lambda x: (x[1], x[0]))

    max_d = scored[-1][1] if scored else 1.0
    return [_pack(rec, p, 1.0 - d / max(max_d, 1.0), 'proximity')
            for p, d in scored[:top_k]]


def markov_rec(rec, traj_uri: str, input_seq: List[str],
               top_k: int, exclude_traj: Optional[str] = None,
               restrict_to: set = None) -> List[Dict]:
    """1-step transition from last observed POI using the flow graph.

    Uses `_flow_weight(src, dst, exclude_traj=...)` so the held-out
    trajectory's own contribution to the flow edges is removed before
    ranking (prevents evaluation leakage on small corpora). The excluded
    trajectory defaults to `traj_uri`; hold-out protocols that evaluate
    against a different trajectory pass it explicitly.

    Backs off to popularity when the flow graph has no outgoing edges
    from the last POI with remaining weight after exclusion.
    """
    exclude = exclude_traj or traj_uri
    visited = set(input_seq)
    candidates = _candidate_filter(rec, visited, restrict_to)

    if not input_seq or rec.flow_graph is None:
        return popularity_rec(rec, traj_uri, input_seq, top_k,
                              exclude_traj=exclude)

    last = input_seq[-1]
    if not rec.flow_graph.has_node(last):
        return popularity_rec(rec, traj_uri, input_seq, top_k,
                              exclude_traj=exclude)

    weights = defaultdict(float)
    for _, dst in rec.flow_graph.out_edges(last):
        if dst not in candidates:
            continue
        w = rec._flow_weight(last, dst, exclude_traj=exclude)
        if w > 0:
            weights[dst] += float(w)

    if not weights:
        return popularity_rec(rec, traj_uri, input_seq, top_k,
                              exclude_traj=exclude)

    total = sum(weights.values())
    ranked = sorted(weights.items(), key=lambda x: (-x[1], x[0]))
    return [_pack(rec, p, w / total, 'markov')
            for p, w in ranked[:top_k]]


BASELINE_REGISTRY = {
    'random': None,  # handled by caller — kept for naming symmetry
    'popularity': popularity_rec,
    'category_popularity': category_popularity_rec,
    'proximity': proximity_rec,
    'markov': markov_rec,
}
