"""
Unified Recommender — nine-signal POI recommendation with 4-layer explainability.

Extends ContextAwareRecommender with:
  - archetype_sim: OWL profile class similarity (IC-weighted Jaccard with class-type dampening)
  - rule_coverage: SR-30..34 independent rule endorsement count

Produces scores in [0,1] with structured explanations grounded in the knowledge graph.
"""

import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

from .context_recommender import (
    ContextAwareRecommender,
    EXCLUDED_CATEGORIES,
)
from .ontological_recommender import (
    OntologicalRecommender,
    LOW_IC_THRESHOLD,
    _effective_ic, _infer_preferred_types, _class_matches_poi_type,
)
from .semantic_reasoning import SemanticReasoningEngine
from .sequence_patterns import REPEAT_EXCEPTION_CATEGORIES
from .fuseki_queries import execute_sparql_query

# Floor of the ontology dwell gate: a POI type whose declared minimum
# meaningful duration exceeds the trajectory's observed budget is demoted,
# not removed, since the budget is an estimate from a short prefix.
DWELL_GATE_FLOOR = 0.25

# The gate is OFF by default, and the reason is a result rather than a
# preference. The POI module's dwell thresholds are sound for CLASSIFYING an
# interaction that happened (Stage 3 uses them exactly so, and the engagement
# tag they produce is stable at 0.93 and prefix-predictive at 0.96). They are
# not sound for PREDICTING whether a future interaction could happen, because
# the only budget available at prediction time is the dwell observed in the
# prefix, and a GPS stop duration does not estimate a person's willingness to
# dwell. On OSM Rome the median prefix budget is 289s while Restaurant
# declares a 900s minimum, so the gate demoted 36% of the true held-out
# targets, RetailStore 23/36 and Restaurant 17/31 among them, and cost 0.064
# MRR (p=0.016). Set _dwell_gate_on=True to reproduce that measurement.
DWELL_GATE_DEFAULT = False

# Second-hop co-visitation neighbours are real evidence but weaker; the
# discount matches the subgraph scorer the project already documents.
COVISIT_2HOP_DISCOUNT = 0.3

_ISO_DUR = re.compile(r'^P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$')


def _iso_duration_seconds(text: str) -> Optional[float]:
    """xsd:duration -> seconds. Returns None for anything unparseable."""
    m = _ISO_DUR.match((text or '').strip())
    if not m or not any(m.groups()):
        return None
    d, h, mi, sec = (float(g) if g else 0.0 for g in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + sec


# ── Signal Configuration ────────────────────────────────────────────────────

UNIFIED_WEIGHTS = {
    'archetype_sim': 0.14,
    'semantic_sim': 0.16,
    'route_sim': 0.20,
    'flow_prob': 0.05,
    'dominant_ctx': 0.11,
    'user_pref': 0.13,
    'significant': 0.04,
    'engagement': 0.09,
    'event_match': 0.08,
}

# Weight set for the ontology-grounded configuration. Co-visitation enters as
# a signal in its own right; its weight is taken from the five signals whose
# fitted coefficients were indistinguishable from zero, rather than from a new
# search. The route-heavy proportions are otherwise preserved.
SEMANTIC_WEIGHTS = {
    'archetype_sim': 0.06, 'semantic_sim': 0.06, 'route_sim': 0.30,
    'flow_prob': 0.20, 'dominant_ctx': 0.08, 'user_pref': 0.08,
    'significant': 0.02, 'engagement': 0.02, 'covisit': 0.18,
}

UNIFIED_KG_PROVENANCE = {
    'archetype_sim': 'profiles:* (IC-weighted Jaccard on OWL inferred graph)',
    'semantic_sim': 'feature cosine (category + temporal + interaction slices)',
    'route_sim': 'Jaccard POI overlap + category sequence LCS',
    'flow_prob': 'poi:followedBy (Markov-2: last two POIs + 2-hop chains)',
    'proximity': 'geo:lat/geo:long (1 / (1 + km from the observed-prefix centroid))',
    'rule_coverage': 'SR-30..34 rule endorsement (in-memory)',
    'dominant_ctx': 'poi:covisitedWith (SR-05), from the dominant POI',
    'covisit': 'poi:covisitedWith (SR-05), whole prefix, 2 hops',
    'user_pref': 'user:ProfileSnapshot / user:preferenceScore (Stage 9 EWMA)',
    'significant': 'poi:SignificantPOI (rdf:type)',
    'engagement': 'trajecto:hasEngagementLevel (SR-10)',
    'event_match': 'scheduled:ScheduledEvent (hostedAt + startTime/endTime overlap)',
}

# Rule name → human-readable label
_RULE_LABELS = {
    'SR-30': 'collaborative',
    'SR-31': 'preference gap',
    'SR-32': 'sequence continuation',
    'SR-33': 'covisitation',
    'SR-34': 'purpose alignment',
}

_RULE_EXPLANATIONS = {
    'SR-30': 'Visited by trajectories with similar semantic profile',
    'SR-31': 'Matches inferred user preferences (unvisited category)',
    'SR-32': 'Next in common visit sequence',
    'SR-33': 'Co-visited with dominant POI in other trajectories',
    'SR-34': 'Aligns with trajectory purpose',
}

# SR-34 purpose → POI category mapping (mirrors SPARQL VALUES block)
_PURPOSE_CATEGORY_MAP = {
    'CulturalPurpose': {'Museum', 'Gallery', 'Artwork', 'Historical Landmark',
                        'Historical Site', 'Monument', 'Cultural', 'Religious Site',
                        'ArtworkPOI', 'HistoricalLandmark', 'HistoricalSite',
                        'HeritageBuilding', 'ReligiousSite', 'CulturalPOI'},
    'CommercialPurpose': {'Retail Store', 'Shopping Mall', 'Market Area',
                          'RetailStore', 'ShoppingMall', 'MarketArea', 'CommercialPOI'},
    'RecreationalPurpose': {'Park', 'Beach', 'Leisure Facility',
                            'Sports Facility', 'Natural Landmark',
                            'LeisureFacility', 'SportsFacility', 'NaturalLandmark',
                            'RecreationalPOI'},
    'EducationalPurpose': {'Library', 'University', 'Educational', 'Museum',
                           'EducationalPOI'},
    'AccommodationPurpose': {'Hotel', 'Accommodation', 'AccommodationPOI'},
    'CafeVisit': {'Cafe', 'Bar', 'Food Court', 'FoodCourt'},
    'NightlifeActivity': {'Nightlife Facility', 'Bar', 'Cafe', 'NightlifeFacility'},
    # HealthcarePurpose intentionally omitted — healthcare POIs are in
    # EXCLUDED_CATEGORIES; a healthcare trajectory should fall back to
    # route/semantic/covisit signals instead of gating to an empty set.
}

def _fmt_event_when(iso_ts: str) -> str:
    """Return a short "on YYYY-MM-DD at HH:MM" string for an event time."""
    if not iso_ts:
        return ''
    try:
        dt = datetime.fromisoformat(iso_ts)
    except ValueError:
        return ''
    return f"on {dt.strftime('%Y-%m-%d')} at {dt.strftime('%H:%M')}"


def _parse_dt(s: str) -> datetime:
    """Parse an ISO-8601 timestamp, tolerating missing timezone / 'Z' suffix.

    Strips timezone info so all comparisons are naive — trajectory timestamps
    are naive local time in the current pipeline, events are written the same
    way by the ingester.
    """
    s = s.replace('Z', '')
    # Drop explicit timezone offset if present (+HH:MM or -HH:MM at the end).
    if len(s) >= 6 and s[-6] in ('+', '-') and s[-3] == ':':
        s = s[:-6]
    return datetime.fromisoformat(s)


MIN_ARCHETYPE_PEERS = 3            # zero archetype_sim if fewer peers than this
FLOW_ESTIMATOR = 'markov2'         # or 'first_order' (last three POIs, 0.5/0.3/0.2)
# Proximity is added to every weight set that does not name it: the other
# weights are scaled by (1 - PROXIMITY_WEIGHT). Pass 'proximity': 0 to disable.
PROXIMITY_WEIGHT = 0.05

# POI categories for which an overlapping ScheduledEvent is the only
# justification to recommend (hard gate; no event → excluded).
EVENT_GATED_CATEGORIES = {
    'ConferenceCenter', 'ConventionCenter', 'ExhibitionHall',
}

# POI categories where an overlapping event is a positive signal but
# absence of an event is fine (soft boost).
EVENT_BOOST_CATEGORIES = {
    'Museum', 'Gallery', 'Theater', 'ConcertHall', 'MusicVenue',
    'OperaHouse', 'PerformanceVenue', 'Stadium', 'SportsFacility',
    'Park', 'ReligiousSite', 'CulturalPOI', 'HistoricalSite',
    'Landmark', 'ArtworkPOI',
}

# How many hours before trajectory_start and after trajectory_end still
# count as "overlapping" the trip for event matching.
EVENT_MATCH_WINDOW_HOURS = 6


class UnifiedRecommender(ContextAwareRecommender):
    """
    Nine-signal POI recommendation with 4-layer explainability.

    Inherits signals from ContextAwareRecommender, adds:
      - archetype_sim: OWL profile class similarity
      - route_sim: Jaccard POI overlap + category sequence LCS
      - flow_prob: Recency-weighted transition probability from flow graph

    Rule coverage (SR-30..34) is computed for Layer 3 explanations but
    excluded from scoring weights — ablation showed it degrades ranking
    by flooding candidates with low-quality endorsements.

    Usage:
        rec = UnifiedRecommender()
        rec.load_all_data(features_path=...)
        recs = rec.recommend(traj_uri, top_k=10)
    """

    def __init__(self, endpoint: str = None, upd_endpoint: str = None):
        super().__init__(endpoint, upd_endpoint)

        # Ontological data
        self.classifications = {}     # uri → set of class names
        self.class_counts = {}
        self.ic_weights = {}
        self.total_classified = 0
        self._archetype_sim_cache = {}  # traj_uri → [(uri, score, shared_classes)]

        # Route similarity data
        self._route_sim_cache = {}    # traj_uri → [{'uri', 'score', 'shared_poi_count', 'lcs_ratio'}]
        self._cat_sequences = {}      # traj_uri → [category_name, ...]

        # Event matching data
        # poi_uri → [{'label','category','start','end','source'}]
        self._events_by_poi: Dict[str, List[dict]] = {}
        # traj_uri → (start_dt, end_dt) or None
        self._traj_time_window: Dict[str, Optional[Tuple[datetime, datetime]]] = {}

        # SR engine (available for direct SPARQL execution if needed)
        self._sr_engine = None

    # ── Data Loading ─────────────────────────────────────────────────────

    def load_all_data(self, features_path: str = None) -> None:
        """Load all data: context-aware base + ontological + SR engine."""
        super().load_all_data(features_path)
        self._load_ontological_data()
        self._load_sr_engine()
        self._precompute_archetype_similarity()
        self._precompute_route_similarity()
        self._precompute_indexes()
        self._load_events()
        self._load_trajectory_time_windows()
        self._load_dwell_semantics()

    def _preferences(self, user, exclude_traj=None):
        """User category preferences, read from the Stage 9 ProfileSnapshot
        chain in the KG. For an evaluated trajectory of the user, the latest
        snapshot stamped before that trajectory starts, so neither it nor any
        later trajectory has shaped the profile; otherwise the latest one."""
        if not user:
            return []
        snaps = self._profile_snapshots().get(user, [])
        if exclude_traj and self.traj_to_user.get(exclude_traj) == user:
            window = self._traj_time_window.get(exclude_traj)
            if not window:
                return []
            start = window[0].isoformat()[:19]
            snaps = [s for s in snaps if s[0] < start]
        return snaps[-1][1] if snaps else []

    def _profile_snapshots(self):
        """user -> time-sorted [(timestamp, [{category, score}, ...]), ...]."""
        if getattr(self, '_snapshot_cache', None) is None:
            res = execute_sparql_query(self.endpoint, """
            PREFIX user: <http://w3id.org/trajecto/user#>
            SELECT ?user ?snap ?ts ?category ?score WHERE {
              ?user user:hasProfile/user:hasSnapshot ?snap .
              ?snap user:updateTimestamp ?ts ; user:hasPreferences ?pref .
              ?pref user:prefersPoiCategory ?category ; user:preferenceScore ?score .
            }""")
            by_snap = defaultdict(list)
            for b in (res or {}).get('results', {}).get('bindings', []):
                key = (b['user']['value'], b['snap']['value'], b['ts']['value'][:19])
                by_snap[key].append({'category': b['category']['value'].split('#')[-1],
                                     'score': float(b['score']['value'])})
            cache = defaultdict(list)
            for (user, _, ts), prefs in by_snap.items():
                cache[user].append((ts, sorted(prefs, key=lambda x: -x['score'])))
            for seq in cache.values():
                seq.sort(key=lambda x: x[0])
            self._snapshot_cache = dict(cache)
        return self._snapshot_cache

    # ── Ontology-parameterised dwell semantics ─────────────────────────
    #
    # The POI module declares, per POI type, the dwell thresholds that make
    # an interaction meaningful (poi:hasMinimumMeaningfulDuration and its two
    # companions), carried by 91 POITypeConfiguration individuals linked
    # through poi:hasTypeConfiguration. Stage 3 of the pipeline already uses
    # them to CLASSIFY an interaction as Passing / Meaningful / Immersive.
    # They are used here to gate a RECOMMENDATION by the same semantics: a
    # trajectory whose own stops are short cannot meaningfully engage with a
    # POI type the ontology says needs a long one. The threshold is read from
    # the ontology, never fitted, so the same gate transfers to any
    # instantiation that populates the POI module.

    def _load_dwell_semantics(self):
        """Load ontology dwell thresholds and per-trajectory dwell evidence."""
        cfg_query = """
        PREFIX poi: <http://w3id.org/trajecto/poi#>
        SELECT ?cat ?minM WHERE {
          ?cat poi:hasTypeConfiguration ?cfg .
          ?cfg poi:hasMinimumMeaningfulDuration ?minM .
        }
        """
        self.dwell_min_meaningful = {}
        res = execute_sparql_query(self.endpoint, cfg_query)
        for b in (res or {}).get('results', {}).get('bindings', []):
            secs = _iso_duration_seconds(b['minM']['value'])
            if secs is not None:
                self.dwell_min_meaningful[
                    b['cat']['value'].split('#')[-1]] = secs
        self._dwell_default = self.dwell_min_meaningful.get(
            'PointOfInterest', 15.0)

        # Per (trajectory, POI) dwell, so a budget can be computed from the
        # OBSERVED prefix alone. A trajectory-level aggregate would draw on
        # the held-out POIs and leak under every hold-out protocol.
        dwell_query = """
        PREFIX poi:      <http://w3id.org/trajecto/poi#>
        PREFIX trajecto: <http://w3id.org/trajecto#>
        PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#>
        SELECT ?traj ?poi (MAX(?d) AS ?dwell) WHERE {
          ?pos dul:isPartOf ?traj ; trajecto:hasInteraction ?i .
          ?i poi:interactsWith ?poi ; poi:hasDwellTimeInSeconds ?d .
        } GROUP BY ?traj ?poi
        """
        self.traj_poi_dwell = {}
        res = execute_sparql_query(self.endpoint, dwell_query)
        for b in (res or {}).get('results', {}).get('bindings', []):
            try:
                self.traj_poi_dwell[(b['traj']['value'], b['poi']['value'])] = \
                    float(b['dwell']['value'])
            except (ValueError, KeyError):
                continue
        print(f"  Dwell semantics: {len(self.dwell_min_meaningful)} typed "
              f"thresholds, {len(self.traj_poi_dwell)} interaction dwells")

    def _dwell_budget(self, traj_uri: str, observed: List[str]
                      ) -> Optional[float]:
        """Median dwell the trajectory actually spends, from its prefix."""
        vals = sorted(d for d in
                      (self.traj_poi_dwell.get((traj_uri, p))
                       for p in observed) if d is not None and d > 0)
        if not vals:
            return None
        mid = len(vals) // 2
        return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2

    def _dwell_feasibility(self, poi: str, budget: Optional[float]) -> float:
        """How meaningfully this trajectory could engage with this POI type.

        1.0 when the observed budget meets the ontology's minimum meaningful
        duration for the POI's type, falling off linearly below it and
        floored at DWELL_GATE_FLOOR so a long-dwell type is demoted rather
        than eliminated. Returns 1.0 for types the ontology does not
        parameterise, so an incomplete POI module never penalises.
        """
        if budget is None:
            return 1.0
        cat = self.poi_metadata.get(poi, {}).get('category', '')
        need = self.dwell_min_meaningful.get(cat)
        if need is None or need <= 0:
            return 1.0
        return max(DWELL_GATE_FLOOR, min(1.0, budget / need))

    def _load_ontological_data(self):
        """Load OWL classifications and IC weights from the inferred graph."""
        print("  Loading ontological classifications...")
        rec = OntologicalRecommender(endpoint=self.endpoint)
        rec.load_classifications()

        self.classifications = rec.classifications
        self.class_counts = rec.class_counts
        self.ic_weights = rec.ic_weights
        self.total_classified = rec.total_trajectories
        print(f"  Ontological: {self.total_classified} classified, "
              f"{len(self.class_counts)} profile classes")

    def _load_sr_engine(self):
        """Initialize semantic reasoning engine for rule execution."""
        self._sr_engine = SemanticReasoningEngine(
            endpoint=self.endpoint,
            upd_endpoint=self.update_endpoint,
        )
        print("  SR engine: ready (5 recommendation rules)")

    # ── Archetype Similarity ─────────────────────────────────────────────

    def _precompute_archetype_similarity(self):
        """Precompute top-15 archetype similar trajectories for all classified.

        Avoids O(N^2) per-call during evaluation (N calls × N comparisons).
        """
        print("  Precomputing archetype similarity matrix...")

        # Precompute informative sets and effective IC weights
        informative_sets = {}
        effective_weights = {}
        for uri, classes in self.classifications.items():
            info = set()
            for c in classes:
                ic = self.ic_weights.get(c, 0)
                if ic >= LOW_IC_THRESHOLD:
                    info.add(c)
                    if c not in effective_weights:
                        effective_weights[c] = _effective_ic(c, ic)
            if info:
                informative_sets[uri] = info

        uris = list(informative_sets.keys())
        n = len(uris)

        # Precompute denominator weight per trajectory
        uri_total_weight = {}
        for uri in uris:
            uri_total_weight[uri] = sum(
                effective_weights[c] for c in informative_sets[uri])

        self._archetype_sim_cache = {}
        for i, uri_a in enumerate(uris):
            set_a = informative_sets[uri_a]
            weight_a = uri_total_weight[uri_a]
            candidates = []

            for j, uri_b in enumerate(uris):
                if i == j:
                    continue
                set_b = informative_sets[uri_b]
                shared = set_a & set_b
                if not shared:
                    continue

                shared_w = sum(effective_weights[c] for c in shared)
                union_w = weight_a + uri_total_weight[uri_b] - shared_w
                score = shared_w / union_w if union_w > 0 else 0.0

                shared_sorted = sorted(
                    shared, key=lambda c: (-effective_weights[c], c))
                candidates.append({
                    'uri': uri_b,
                    'score': score,
                    'shared_classes': shared_sorted,
                })

            candidates.sort(key=lambda x: (-x['score'], x['uri']))
            self._archetype_sim_cache[uri_a] = candidates[:15]

        print(f"  Archetype similarity: {n} trajectories, "
              f"{sum(len(v) for v in self._archetype_sim_cache.values())} pairs cached")

    def _find_archetype_similar(self, traj_uri: str,
                                top_k: int = 15) -> List[dict]:
        """Return precomputed archetypally similar trajectories."""
        return self._archetype_sim_cache.get(traj_uri, [])[:top_k]

    # ── Route Similarity ────────────────────────────────────────────────

    def _precompute_route_similarity(self):
        """Precompute top-15 route-similar trajectories per trajectory.

        Route similarity = 0.5 * Jaccard POI overlap + 0.5 * normalized
        category sequence LCS.  Captures literal spatial overlap and
        visit-order patterns missed by abstract feature-space similarities.
        """
        print("  Precomputing route similarity matrix...")

        # Build ordered category sequences
        self._cat_sequences = {}
        for traj, seq in self.sequences.items():
            cat_seq = []
            for poi in seq:
                cat = self.poi_metadata.get(poi, {}).get('category', '')
                if cat and cat not in EXCLUDED_CATEGORIES:
                    cat_seq.append(cat)
            if cat_seq:
                self._cat_sequences[traj] = cat_seq

        eligible = [t for t in self.traj_poi_sets
                    if t in self._cat_sequences
                    and len(self.traj_poi_sets[t]) >= 2]

        self._route_sim_cache = {}
        for i, traj_a in enumerate(eligible):
            pois_a = self.traj_poi_sets[traj_a]
            cats_a = self._cat_sequences[traj_a]
            candidates = []

            for j, traj_b in enumerate(eligible):
                if i == j:
                    continue
                pois_b = self.traj_poi_sets[traj_b]

                intersection = len(pois_a & pois_b)
                if intersection == 0:
                    continue

                union = len(pois_a | pois_b)
                jaccard = intersection / union

                cats_b = self._cat_sequences[traj_b]
                lcs_len = self._lcs_length(cats_a, cats_b)
                max_len = max(len(cats_a), len(cats_b))
                lcs_sim = lcs_len / max_len if max_len > 0 else 0.0

                score = 0.5 * jaccard + 0.5 * lcs_sim
                candidates.append({
                    'uri': traj_b,
                    'score': score,
                    'shared_poi_count': intersection,
                    'lcs_ratio': lcs_sim,
                })

            candidates.sort(key=lambda x: (-x['score'], x['uri']))
            self._route_sim_cache[traj_a] = candidates[:15]

        n = len(eligible)
        pairs = sum(len(v) for v in self._route_sim_cache.values())
        print(f"  Route similarity: {n} trajectories, {pairs} pairs cached")

    def _find_route_similar(self, traj_uri: str, top_k: int = 15,
                            observed: List[str] = None) -> List[dict]:
        """Route-similar trajectories. With an observed prefix, similarity is
        computed from that prefix only: the precomputed cache compares full
        trajectories, which for an evaluated trajectory includes the held-out
        POIs and selects exactly the peers that visited them."""
        if observed is None:
            return self._route_sim_cache.get(traj_uri, [])[:top_k]
        pois_a = set(observed)
        cats_a = [c for c in (self.poi_metadata.get(p, {}).get('category', '')
                              for p in observed)
                  if c and c not in EXCLUDED_CATEGORIES]
        if not pois_a or not cats_a:
            return []
        out = []
        for other, pois_b in self.traj_poi_sets.items():
            if other == traj_uri or other not in self._cat_sequences:
                continue
            inter = len(pois_a & pois_b)
            if not inter:
                continue
            cats_b = self._cat_sequences[other]
            lcs = self._lcs_length(cats_a, cats_b) / max(len(cats_a), len(cats_b))
            out.append({'uri': other, 'shared_poi_count': inter, 'lcs_ratio': lcs,
                        'score': 0.5 * inter / len(pois_a | pois_b) + 0.5 * lcs})
        out.sort(key=lambda x: (-x['score'], x['uri']))
        return out[:top_k]

    @staticmethod
    def _lcs_length(seq_a: list, seq_b: list) -> int:
        """Longest Common Subsequence length (space-optimized DP)."""
        m, n = len(seq_a), len(seq_b)
        prev = [0] * (n + 1)
        for i in range(1, m + 1):
            curr = [0] * (n + 1)
            for j in range(1, n + 1):
                if seq_a[i - 1] == seq_b[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev = curr
        return prev[n]

    def _precompute_indexes(self):
        """Build indexes for fast signal computation during evaluation."""
        # Engagement level → set of POIs from matching trajectories
        self._engagement_pois = defaultdict(lambda: defaultdict(int))
        for traj, meta in self.traj_metadata.items():
            eng = meta.get('engagement')
            if eng:
                for poi in self.traj_poi_sets.get(traj, set()):
                    self._engagement_pois[eng][poi] += 1

        # Category → set of POI URIs (for SR-31, SR-34)
        self._category_index = defaultdict(set)
        for poi, pmeta in self.poi_metadata.items():
            cat = pmeta.get('category', '')
            if cat:
                self._category_index[cat].add(poi)

        # User → all visited POIs (for rule exclusion)
        self._user_visited_cache = {}
        for user, trajs in self.user_to_trajs.items():
            visited = set()
            for t in trajs:
                visited |= self.traj_poi_sets.get(t, set())
            self._user_visited_cache[user] = visited

        print(f"  Indexes: {len(self._engagement_pois)} engagement levels, "
              f"{len(self._category_index)} category groups, "
              f"{len(self._user_visited_cache)} user visit sets")

    # ── Events Loading ──────────────────────────────────────────────────

    def _load_events(self):
        """Load scheduled events from Fuseki, indexed by hostedAt POI."""
        query = """
        PREFIX scheduled: <http://w3id.org/trajecto/scheduled#>
        PREFIX rdfs:      <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?event ?poi ?label ?cat ?start ?end ?source WHERE {
          ?event a scheduled:ScheduledEvent ;
                 scheduled:hostedAt ?poi ;
                 scheduled:startTime ?start ;
                 scheduled:endTime ?end ;
                 scheduled:eventCategory ?cat ;
                 scheduled:externalSource ?source .
          OPTIONAL { ?event rdfs:label ?label }
        }
        """
        res = execute_sparql_query(self.endpoint, query)
        self._events_by_poi = defaultdict(list)
        if not res:
            print("  Events: none indexed (no Fuseki response)")
            return
        count = 0
        for b in res.get('results', {}).get('bindings', []):
            try:
                start_dt = _parse_dt(b['start']['value'])
                end_dt = _parse_dt(b['end']['value'])
            except ValueError:
                continue
            self._events_by_poi[b['poi']['value']].append({
                'uri': b['event']['value'],
                'label': b.get('label', {}).get('value', ''),
                'category': b['cat']['value'],
                'start': start_dt,
                'end': end_dt,
                'source': b['source']['value'],
            })
            count += 1
        print(f"  Events: {count} scheduled events at "
              f"{len(self._events_by_poi)} POIs")

    def _load_trajectory_time_windows(self):
        """Fetch (min_ts, max_ts) per trajectory from RawPosition timestamps."""
        query = """
        PREFIX trajecto: <http://w3id.org/trajecto#>
        PREFIX dul:      <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#>
        SELECT ?traj (MIN(?ts) AS ?start) (MAX(?ts) AS ?end) WHERE {
          ?pos dul:isPartOf ?traj ;
               trajecto:timestamp ?ts .
        } GROUP BY ?traj
        """
        res = execute_sparql_query(self.endpoint, query)
        self._traj_time_window = {}
        if not res:
            print("  Trajectory windows: none loaded")
            return
        for b in res.get('results', {}).get('bindings', []):
            try:
                start_dt = _parse_dt(b['start']['value'])
                end_dt = _parse_dt(b['end']['value'])
            except ValueError:
                continue
            self._traj_time_window[b['traj']['value']] = (start_dt, end_dt)
        print(f"  Trajectory windows: {len(self._traj_time_window)} cached")

    def _events_overlapping(self, poi_uri: str,
                            window: Tuple[datetime, datetime]) -> List[dict]:
        """Return events at poi_uri whose time span overlaps window ± tolerance."""
        events = self._events_by_poi.get(poi_uri, [])
        if not events:
            return []
        tol = timedelta(hours=EVENT_MATCH_WINDOW_HOURS)
        win_start, win_end = window
        expanded_start = win_start - tol
        expanded_end = win_end + tol
        return [e for e in events
                if e['end'] >= expanded_start and e['start'] <= expanded_end]

    # ── In-Memory Rule Computation ─────────────────────────────────────

    def _compute_rules_in_memory(self, traj_uri: str, observed: List[str] = None,
                                 exclude_traj: str = None) -> Dict[str, set]:
        """Compute SR-30..34 rule results from in-memory data.

        Equivalent to SPARQL execution but avoids 5 Fuseki round-trips
        with expensive FILTER NOT EXISTS clauses. With an observed prefix,
        the trajectory's visited POIs are the prefix and the user's history
        is their other trajectories, so held-out POIs neither seed SR-32 nor
        get removed as already visited.
        """
        meta = self.traj_metadata.get(traj_uri, {})
        purposes = meta.get('purposes', set())
        nav_styles = meta.get('nav_styles', set())
        user = self.traj_to_user.get(traj_uri)

        # All POIs this user has visited (other trajectories + prefix when evaluating)
        if observed is None:
            user_visited = self._user_visited_cache.get(user, set()) if user else set()
        else:
            user_visited = set(observed).union(*(
                self.traj_poi_sets.get(t, set()) for t, u in self.traj_to_user.items()
                if u == user and t != traj_uri)) if user else set(observed)

        # SR-30: Profile collaborative — same purpose + shared pattern/temporal
        sr30 = set()
        if purposes:
            for other_uri, other_meta in self.traj_metadata.items():
                if other_uri == traj_uri:
                    continue
                shared_purpose = purposes & other_meta.get('purposes', set())
                if not shared_purpose:
                    continue
                shared_nav = nav_styles & other_meta.get('nav_styles', set())
                if not shared_nav:
                    continue
                sr30 |= self.traj_poi_sets.get(other_uri, set())
            sr30 -= user_visited

        # SR-31: Preference gap — POIs in categories user has affinity for
        sr31 = set()
        if user:
            prefs = self._preferences(user, exclude_traj)
            pref_cats = {p['category'] for p in prefs if p.get('score', 0) > 0}
            for cat in pref_cats:
                sr31 |= self._category_index.get(cat, set())
            sr31 -= user_visited

        # SR-32: Sequence continuation — follow flow graph from visited POIs
        sr32 = set()
        visited_pois = (set(observed) if observed is not None
                        else self.traj_poi_sets.get(traj_uri, set()))
        if self.flow_graph:
            for poi in visited_pois:
                if self.flow_graph.has_node(poi):
                    for neighbor in self.flow_graph.successors(poi):
                        # leave-one-out: an edge only this trajectory walked
                        # would point at its own held-out POIs
                        if (neighbor not in user_visited
                                and self._flow_weight(poi, neighbor, exclude_traj) > 0):
                            sr32.add(neighbor)

        # SR-33: Covisitation — co-visited with dominant POI
        sr33 = set()
        dominant = self._get_dominant_poi(traj_uri)
        if dominant and self.covisit_graph and self.covisit_graph.has_node(dominant):
            for neighbor in self.covisit_graph.neighbors(dominant):
                if (neighbor not in user_visited
                        and self._covisit_weight(dominant, neighbor, exclude_traj) > 0):
                    sr33.add(neighbor)

        # SR-34: Purpose alignment — POIs whose category matches purpose
        sr34 = set()
        if purposes:
            aligned_cats = set()
            for p in purposes:
                aligned_cats |= _PURPOSE_CATEGORY_MAP.get(p, set())
            for cat in aligned_cats:
                sr34 |= self._category_index.get(cat, set())
            sr34 -= user_visited

        # Filter excluded categories
        result = {}
        for rule_name, poi_set in [('SR-30', sr30), ('SR-31', sr31),
                                    ('SR-32', sr32), ('SR-33', sr33),
                                    ('SR-34', sr34)]:
            result[rule_name] = {p for p in poi_set
                                 if self.poi_metadata.get(p, {}).get(
                                     'category', '') not in EXCLUDED_CATEGORIES}
        return result

    # ── Unified Recommendation ───────────────────────────────────────────

    def recommend(self, traj_uri: str,
                  observed_sequence: List[str] = None,
                  top_k: int = 20,
                  min_score: float = 0.15,
                  signal_weights: Dict[str, float] = None,
                  exclude_traj: str = None,
                  restrict_to: set = None) -> List[Dict]:
        """
        Nine-signal POI recommendation with 4-layer explainability.

        Rule coverage is computed for explanation Layer 3 but does not
        affect ranking (weight = 0).

        Returns list of dicts with:
            poi, label, category, category_group, score, confidence,
            latitude, longitude, popularity_rank, dwell_time_label,
            signals, archetype, route_match, rules, provenance, explanation
        """
        weights = dict(signal_weights or UNIFIED_WEIGHTS)
        if 'proximity' not in weights:
            weights = {k: v * (1 - PROXIMITY_WEIGHT) for k, v in weights.items()}
            weights['proximity'] = PROXIMITY_WEIGHT

        visited = set(observed_sequence or self.sequences.get(traj_uri, []))
        meta = self.traj_metadata.get(traj_uri, {})

        # All unvisited POIs as candidates (no forward reachability filter —
        # that constraint suits closed environments like museums, not open
        # city-scale GPS trajectories where the POI space is large and sparse)
        #
        # `restrict_to` bounds the pool up front for geo-constrained
        # protocols. Filtering this method's OUTPUT instead is wrong: the
        # score floor, the diversity penalty and the purpose gate below all
        # break score-monotonicity, so an output-filtered list is both
        # shorter and differently ordered than one ranked over the
        # restricted pool. When the pool is supplied the floor is dropped,
        # since the caller wants a full ranking of exactly these candidates.
        pool = self.all_poi_uris if restrict_to is None else set(restrict_to)
        if restrict_to is not None:
            min_score = 0.0
        candidates = pool - visited
        candidates = {c for c in candidates
                      if self.poi_metadata.get(c, {}).get('category', '')
                      not in EXCLUDED_CATEGORIES}

        # No purpose gate: purpose acts through the scoring signals (archetype,
        # rules, semantic), not as a veto. A hard purpose->category filter put
        # two thirds of the POIs visited next out of reach, and the purpose tag
        # is inferred from the whole trajectory, held-out part included.

        # Archetype peer fallback: if too few classified peers, the signal
        # is noise — zero its weight and redistribute proportionally.
        n_archetype_peers = len(self._find_archetype_similar(traj_uri, top_k=15))
        if n_archetype_peers < MIN_ARCHETYPE_PEERS and weights.get('archetype_sim', 0) > 0:
            dropped = weights.pop('archetype_sim')
            remaining_total = sum(weights.values())
            if remaining_total > 0:
                for k in weights:
                    weights[k] += dropped * weights[k] / remaining_total
            weights['archetype_sim'] = 0.0

        # Event-gated category hard filter: ConferenceCenter-class POIs only
        # surface when a scheduled event overlaps the trajectory's time window.
        # If the trajectory has no known time window, treat as no-match
        # (conservative: such categories stay excluded).
        traj_window = self._traj_time_window.get(traj_uri)
        if traj_window:
            candidates = {
                c for c in candidates
                if self.poi_metadata.get(c, {}).get('category', '')
                   not in EVENT_GATED_CATEGORIES
                or self._events_overlapping(c, traj_window)
            }
        else:
            candidates = {
                c for c in candidates
                if self.poi_metadata.get(c, {}).get('category', '')
                   not in EVENT_GATED_CATEGORIES
            }

        # If no events are indexed at all, drop event_match weight and
        # redistribute so the signal doesn't silently dilute the others.
        if not self._events_by_poi and weights.get('event_match', 0) > 0:
            dropped = weights.pop('event_match')
            remaining_total = sum(weights.values())
            if remaining_total > 0:
                for k in weights:
                    weights[k] += dropped * weights[k] / remaining_total
            weights['event_match'] = 0.0

        scores = defaultdict(lambda: defaultdict(float))
        evidence = defaultdict(lambda: defaultdict(dict))

        # ── Signal 1: Archetype similarity (NEW) ────────────────────────
        target_classes = self.classifications.get(traj_uri, set())
        preferred = _infer_preferred_types(target_classes, self.ic_weights)
        archetype_similar = self._find_archetype_similar(traj_uri, top_k=15)

        for sim in archetype_similar:
            sim_pois = self.traj_poi_sets.get(sim['uri'], set())
            for poi in (sim_pois & candidates):
                poi_cat = self.poi_metadata.get(poi, {}).get('category', '')
                alignment = preferred.get(poi_cat, 0.0)
                contribution = sim['score'] * (1.0 + alignment)
                scores[poi]['archetype_sim'] += contribution

                existing = evidence[poi]['archetype_sim']
                existing.setdefault('sources', []).append({
                    'traj': sim['uri'],
                    'score': sim['score'],
                    'shared': sim['shared_classes'][:3],
                })
                existing['alignment'] = max(
                    existing.get('alignment', 0), alignment)

        # Collect archetype info for explanations
        archetype_info = {}
        for poi in scores:
            ev = evidence[poi].get('archetype_sim', {})
            sources = ev.get('sources', [])
            if not sources:
                continue

            # Aggregate shared classes across sources
            class_freq = defaultdict(int)
            for src in sources:
                for cls in src['shared']:
                    class_freq[cls] += 1
            top_shared = sorted(class_freq, key=lambda c: -class_freq[c])[:3]

            # Profile fit: which target classes match this POI type
            poi_cat = self.poi_metadata.get(poi, {}).get('category', '')
            profile_fit = [c for c in target_classes
                           if _class_matches_poi_type(c, poi_cat)
                           ] if poi_cat else []

            archetype_info[poi] = {
                'shared_classes': top_shared,
                'profile_fit': profile_fit[:3] if profile_fit else None,
                'alignment': ev.get('alignment', 0.0),
                'source_count': len(sources),
            }

        # ── Signal 2: Semantic similarity (inherited logic) ─────────────
        if traj_uri in self.features:
            similar = self._find_similar_63dim(traj_uri, top_k=10)
            for sim_traj, sim_score in similar:
                sim_pois = self.traj_poi_sets.get(sim_traj, set())
                for poi in (sim_pois & candidates):
                    scores[poi]['semantic_sim'] += sim_score
                    contribs = evidence[poi]['semantic_sim'].setdefault(
                        'contributors', [])
                    contribs.append({
                        'traj': sim_traj, 'similarity': sim_score})

        # ── Signal 3: Route similarity (Jaccard + LCS) ────────────────────
        route_similar = self._find_route_similar(
            traj_uri, top_k=15,
            observed=list(observed_sequence) if observed_sequence else None)
        route_info = {}

        for sim in route_similar:
            sim_pois = self.traj_poi_sets.get(sim['uri'], set())
            for poi in (sim_pois & candidates):
                scores[poi]['route_sim'] += sim['score']
                existing = evidence[poi]['route_sim']
                existing.setdefault('sources', []).append({
                    'traj': sim['uri'],
                    'score': sim['score'],
                    'shared_poi_count': sim['shared_poi_count'],
                })

        for poi in scores:
            ev = evidence[poi].get('route_sim', {})
            sources = ev.get('sources', [])
            if sources:
                best = max(sources, key=lambda s: s['shared_poi_count'])
                route_info[poi] = {
                    'source_count': len(sources),
                    'max_shared_pois': best['shared_poi_count'],
                    'top_score': best['score'],
                }

        # ── Signal 4: Rule coverage ────────────────────────────────────────
        rule_results = {}
        per_rule = self._compute_rules_in_memory(
            traj_uri, observed=list(observed_sequence) if observed_sequence else None,
            exclude_traj=exclude_traj)
        for rule_name, poi_set in per_rule.items():
            rule_results[rule_name] = poi_set
            for poi in (poi_set & candidates):
                scores[poi]['rule_coverage'] += 1.0
                evidence[poi]['rule_coverage'].setdefault(
                    'rules', []).append(rule_name)

        # ── Signal 5: Flow probability (recency-weighted) ─────────────────
        input_seq = list(observed_sequence or self.sequences.get(traj_uri, []))
        if self.flow_graph and input_seq and FLOW_ESTIMATOR == 'markov2':
            from .rec_levers import markov2_scores
            m2 = markov2_scores(self, input_seq, exclude_traj or traj_uri, candidates)
            for dst, prob in m2.items():
                scores[dst]['flow_prob'] += prob
                evidence[dst]['flow_prob'].setdefault(
                    'source_pois', list(input_seq[-2:]))
        elif self.flow_graph and input_seq:
            last_pois = input_seq[-3:] if len(input_seq) >= 3 else input_seq
            recency_weights = [0.5, 0.3, 0.2]

            # Build probability distribution per source POI
            flow_dists = {}
            for poi in last_pois:
                if self.flow_graph.has_node(poi):
                    adj_weights = {}
                    for _, dst in self.flow_graph.out_edges(poi):
                        fw = self._flow_weight(poi, dst, exclude_traj)
                        if fw > 0:
                            adj_weights[dst] = fw
                    total_w = sum(adj_weights.values())
                    if total_w > 0:
                        flow_dists[poi] = {
                            dst: w / total_w for dst, w in adj_weights.items()}

            for i, poi in enumerate(reversed(last_pois)):
                w = recency_weights[i] if i < len(recency_weights) else 0.1
                if poi in flow_dists:
                    for dst, prob in flow_dists[poi].items():
                        if dst in candidates:
                            scores[dst]['flow_prob'] += w * prob
                            ev = evidence[dst]['flow_prob']
                            ev.setdefault('source_pois', [])
                            if poi not in ev['source_pois']:
                                ev['source_pois'].append(poi)

        # ── Signal 5b: Proximity (optional; only when given a weight) ────
        # 1 / (1 + km from the centroid of the observed POIs).
        if weights.get('proximity', 0) > 0:
            from .baselines import _haversine
            coords = [(m['lat'], m['lon']) for m in
                      (self.poi_metadata.get(p, {}) for p in input_seq)
                      if m.get('lat') is not None and m.get('lon') is not None]
            if coords:
                clat = sum(c[0] for c in coords) / len(coords)
                clon = sum(c[1] for c in coords) / len(coords)
                for poi in candidates:
                    m = self.poi_metadata.get(poi, {})
                    if m.get('lat') is None or m.get('lon') is None:
                        continue
                    km = _haversine(clat, clon, m['lat'], m['lon']) / 1000.0
                    scores[poi]['proximity'] += 1.0 / (1.0 + km)
                    evidence[poi]['proximity'] = {'distance_km': round(km, 3)}

        # ── Signal 6: Dominant context (inherited) ──────────────────────
        dominant = self._get_dominant_poi(traj_uri)
        if dominant and self.covisit_graph and self.covisit_graph.has_node(dominant):
            dominant_label = self.poi_metadata.get(
                dominant, {}).get('label', '?')
            for neighbor in self.covisit_graph.neighbors(dominant):
                if neighbor in candidates:
                    edge_w = self._covisit_weight(
                        dominant, neighbor, exclude_traj)
                    if edge_w > 0:
                        scores[neighbor]['dominant_ctx'] += edge_w
                        evidence[neighbor]['dominant_ctx'] = {
                            'dominant_label': dominant_label,
                            'covisit_weight': edge_w,
                        }

        # ── Signal 6b: Co-visitation over the whole observed prefix ─────
        # poi:covisitedWith is materialised by rule SR-05 over the KG. Signal
        # 6 above reads it from the dominant POI only, which uses one node of
        # a graph that carries the corpus's entire "these places go together"
        # structure. This reads it from every observed POI, and one hop
        # further out at a discount, because the cold-start slice shows
        # co-visitation is the only evidence that reaches POIs with one or
        # two corpus visits, where every count-based prior is zero by
        # arithmetic.
        if self.covisit_graph and weights.get('covisit', 0) > 0:
            for src in visited:
                if not self.covisit_graph.has_node(src):
                    continue
                for nb in self.covisit_graph.neighbors(src):
                    w1 = self._covisit_weight(src, nb, exclude_traj)
                    if w1 <= 0:
                        continue
                    if nb in candidates:
                        scores[nb]['covisit'] += w1
                        ev = evidence[nb].setdefault('covisit', {'hops': 1,
                                                                 'via': []})
                        if src not in ev['via']:
                            ev['via'].append(src)
                    for nb2 in self.covisit_graph.neighbors(nb):
                        if nb2 in candidates and nb2 not in visited:
                            w2 = self._covisit_weight(nb, nb2, exclude_traj)
                            if w2 > 0:
                                scores[nb2]['covisit'] += (
                                    COVISIT_2HOP_DISCOUNT * w1 * w2)
                                evidence[nb2].setdefault(
                                    'covisit', {'hops': 2, 'via': []})

        # ── Signal 7: User preference ───────────────────────────────────
        user = self.traj_to_user.get(traj_uri)
        prefs = self._preferences(user, exclude_traj)
        pref_map = {p['category']: p['score'] for p in prefs}
        for poi in candidates:
            poi_meta = self.poi_metadata.get(poi, {})
            cat = poi_meta.get('category', '')
            cat_score = pref_map.get(cat, 0)
            if cat_score > 0:
                scores[poi]['user_pref'] += cat_score
                evidence[poi]['user_pref'] = {
                    'category': cat, 'cat_score': cat_score,
                }

        # ── Signal 8: Significant POIs (inherited) ──────────────────────
        sig_pois = self.significant_pois or self._get_significant_pois()
        for poi in (sig_pois & candidates):
            scores[poi]['significant'] += 1.0
            evidence[poi]['significant'] = {
                'total_significant': len(sig_pois),
            }

        # ── Signal 9: Event match (ScheduledEvent overlap) ──────────────
        # Score = 1.0 for candidates with ≥1 overlapping event if the POI
        # is event-eligible (gated or boost category). Evidence includes
        # event label + category + time for explanations.
        if traj_window and self._events_by_poi:
            for poi in candidates:
                cat = self.poi_metadata.get(poi, {}).get('category', '')
                if cat not in EVENT_GATED_CATEGORIES \
                        and cat not in EVENT_BOOST_CATEGORIES:
                    continue
                overlapping = self._events_overlapping(poi, traj_window)
                if not overlapping:
                    continue
                # Pick best event: prefer exact-day overlap over tolerance edge.
                traj_start, traj_end = traj_window
                best = max(overlapping, key=lambda e: (
                    min(e['end'], traj_end) - max(e['start'], traj_start)
                ).total_seconds())
                scores[poi]['event_match'] += 1.0
                evidence[poi]['event_match'] = {
                    'event_label': best['label'],
                    'event_category': best['category'],
                    'event_start': best['start'].isoformat(),
                    'event_end': best['end'].isoformat(),
                    'source': best['source'],
                    'n_events': len(overlapping),
                    'gated_category': cat in EVENT_GATED_CATEGORIES,
                }

        # ── Signal 10: Engagement level (precomputed index) ────────────────
        engagement = meta.get('engagement')
        if engagement and engagement in self._engagement_pois:
            eng_pois = self._engagement_pois[engagement]
            # Subtract this trajectory's own POIs from counts
            own_pois = self.traj_poi_sets.get(traj_uri, set())
            for poi in candidates:
                count = eng_pois.get(poi, 0)
                if poi in own_pois and count > 0:
                    count -= 1
                if count > 0:
                    scores[poi]['engagement'] += float(count)
                    evidence[poi]['engagement'] = {
                        'engagement_level': engagement,
                        'match_count': count,
                    }

        # ── Normalize + Aggregate ────────────────────────────────────────
        for signal in weights:
            signal_vals = [scores[poi][signal]
                           for poi in scores if scores[poi][signal] > 0]
            if signal_vals:
                max_val = max(signal_vals)
                if max_val > 0:
                    for poi in scores:
                        if scores[poi][signal] > 0:
                            scores[poi][signal] /= max_val

        max_score = 0.0
        ranked = []
        for poi, signals in scores.items():
            total = sum(weights.get(sig, 0) * val
                        for sig, val in signals.items())
            if total > max_score:
                max_score = total
            ranked.append((poi, signals, total))

        # ── Category Saturation Penalty ────────────────────────────────────
        # Penalize POIs whose category is already well-represented in the
        # visited sequence (e.g., don't suggest another restaurant if two
        # were already visited).  Exception categories (Cultural, Museum, …)
        # tolerate more repeats.
        visited_cat_counts = Counter()
        for v_poi in visited:
            c = self.poi_metadata.get(v_poi, {}).get('category', '')
            if c and c != 'Unknown':
                visited_cat_counts[c] += 1

        # Ontology dwell gate: demote candidates whose declared minimum
        # meaningful duration this trajectory's own stops cannot support.
        # Multiplicative rather than a tenth weighted signal, because it is a
        # semantic constraint read from the POI module, not evidence to be
        # traded off against other evidence.
        budget = (self._dwell_budget(traj_uri, list(visited))
                  if getattr(self, '_dwell_gate_on', DWELL_GATE_DEFAULT)
                  else None)

        penalised = []
        for poi, signals, total in ranked:
            total *= self._dwell_feasibility(poi, budget)
            cat = self.poi_metadata.get(poi, {}).get('category', '')
            count = visited_cat_counts.get(cat, 0)
            if cat in REPEAT_EXCEPTION_CATEGORIES:
                penalty = 0.0 if count <= 2 else 0.5
            else:
                if count == 0:
                    penalty = 0.0
                elif count == 1:
                    penalty = 0.6
                elif count == 2:
                    penalty = 0.85
                else:
                    penalty = 1.0
            penalised.append((poi, signals, total * (1.0 - penalty)))

        penalised.sort(key=lambda x: (-x[2], x[0]))   # URI tie-break: reproducibility
        ranked = [(p, s, t) for p, s, t in penalised if t >= min_score][:top_k]

        # Visited categories for context
        visited_cat_names = set(visited_cat_counts.keys())

        # ── Build Output with 4-Layer Explanations ───────────────────────
        results = []
        for poi, signals, total in ranked:
            poi_meta = self.poi_metadata.get(poi, {})
            category = poi_meta.get('category', 'Unknown')
            category_group = self._get_category_group(category)
            lat = poi_meta.get('lat')
            lon = poi_meta.get('lon')
            popularity_count = self._poi_popularity.get(poi, 0)

            # Confidence: score strength + evidence breadth
            active_signals = sum(1 for v in signals.values() if v > 0)
            norm_score = total / max_score if max_score > 0 else 0
            confidence_val = (0.5 * norm_score
                              + 0.5 * min(active_signals / 6, 1.0))
            if confidence_val >= 0.6:
                confidence = 'strong'
            elif confidence_val >= 0.35:
                confidence = 'moderate'
            else:
                confidence = 'weak'

            avg_dwell = self._get_avg_dwell(poi)
            dwell_label = self._get_dwell_time_label(avg_dwell)
            popularity_rank = self._get_popularity_rank(popularity_count)

            # Layer 1: Signal decomposition
            signal_dict = dict(signals)

            # Layer 2: Archetype reasoning
            archetype = archetype_info.get(poi)

            # Layer 3: Rule endorsement
            rule_ev = evidence[poi].get('rule_coverage', {})
            matched_rules = rule_ev.get('rules', [])
            rules_layer = None
            if matched_rules:
                rules_layer = {
                    'count': len(matched_rules),
                    'matched': matched_rules,
                    'explanations': {
                        r: _RULE_EXPLANATIONS.get(r, r)
                        for r in matched_rules
                    },
                }

            # Layer 4: KG provenance (predicate + actual values)
            provenance = []
            for sig, val in sorted(signals.items(),
                                   key=lambda x: -x[1]):
                w = weights.get(sig, 0)
                contribution = w * val
                if contribution <= 0:
                    continue
                ev = evidence.get(poi, {}).get(sig, {})
                prov_entry = {
                    'signal': sig,
                    'contribution': round(contribution, 4),
                    'predicate': UNIFIED_KG_PROVENANCE.get(sig, ''),
                    'raw_score': round(val, 4),
                }
                # Attach signal-specific values
                if sig == 'archetype_sim':
                    sources = ev.get('sources', [])
                    if sources:
                        prov_entry['top_similarity'] = round(
                            sources[0]['score'], 3)
                        prov_entry['n_sources'] = len(sources)
                elif sig == 'semantic_sim':
                    contribs = ev.get('contributors', [])
                    if contribs:
                        prov_entry['top_cosine'] = round(
                            contribs[0]['similarity'], 3)
                        prov_entry['n_contributors'] = len(contribs)
                elif sig == 'rule_coverage':
                    prov_entry['rules'] = ev.get('rules', [])
                elif sig == 'dominant_ctx':
                    prov_entry['dominant_poi'] = ev.get(
                        'dominant_label', '?')
                    prov_entry['covisit_weight'] = ev.get(
                        'covisit_weight', 0)
                elif sig == 'route_sim':
                    sources = ev.get('sources', [])
                    if sources:
                        best = max(sources,
                                   key=lambda s: s['shared_poi_count'])
                        prov_entry['max_shared_pois'] = best[
                            'shared_poi_count']
                        prov_entry['n_sources'] = len(sources)
                elif sig == 'flow_prob':
                    prov_entry['source_pois'] = ev.get('source_pois', [])
                elif sig == 'user_pref':
                    prov_entry['category'] = ev.get('category', '')
                    prov_entry['pref_score'] = ev.get('cat_score', 0)
                elif sig == 'event_match':
                    prov_entry['event_label'] = ev.get('event_label', '')
                    prov_entry['event_category'] = ev.get('event_category', '')
                    prov_entry['event_start'] = ev.get('event_start', '')
                    prov_entry['event_end'] = ev.get('event_end', '')
                    prov_entry['source'] = ev.get('source', '')
                    prov_entry['n_events'] = ev.get('n_events', 0)
                    prov_entry['gated'] = ev.get('gated_category', False)
                provenance.append(prov_entry)

            # Route match layer
            route_match = route_info.get(poi)

            # Explanation in the format the map visualizer reads
            poi_evidence = dict(evidence.get(poi, {}))
            explanation = self._build_unified_explanation(
                poi, category, category_group, signals, weights,
                archetype, route_match, rules_layer, visited_cat_names,
                popularity_count, poi_evidence
            )

            results.append({
                'poi': poi,
                'label': poi_meta.get('label', poi.split('/')[-1]),
                'category': category,
                'category_group': category_group,
                'score': round(total, 4),
                'confidence': confidence,
                'latitude': float(lat) if lat is not None else None,
                'longitude': float(lon) if lon is not None else None,
                'popularity_rank': popularity_rank,
                'dwell_time_label': dwell_label,
                'signals': signal_dict,
                'archetype': archetype,
                'route_match': route_match,
                'rules': rules_layer,
                'provenance': provenance,
                'explanation': explanation,
            })

        return results

    # ── Explanation Builders ─────────────────────────────────────────────

    def _build_unified_explanation(
        self, poi, category, category_group, signals, weights,
        archetype, route_match, rules_layer, visited_cat_names,
        popularity_count, poi_evidence: Dict = None
    ) -> Dict[str, Any]:
        """Build the map visualizer's explanation from the 4-layer data."""
        poi_evidence = poi_evidence or {}

        # Dominant context label
        dom_ev = poi_evidence.get('dominant_ctx', {})
        dom_label = dom_ev.get('dominant_label', '')
        dom_weight = dom_ev.get('covisit_weight', 0)
        dom_str = (f'Co-visited with {dom_label} in {int(dom_weight)} trajectories'
                   if dom_label and dom_weight
                   else 'Co-visited with the primary POI of this trajectory')

        # Primary reason from strongest signal
        weighted = {sig: weights.get(sig, 0) * val
                    for sig, val in signals.items()}
        top_sig = max(weighted, key=weighted.get, default='')
        top_val = weighted.get(top_sig, 0)

        if top_sig == 'archetype_sim' and archetype:
            classes = ', '.join(archetype['shared_classes'][:2])
            primary = (f"{classes} profile match"
                       + (f" (alignment={archetype['alignment']:.2f})"
                          if archetype['alignment'] > 0 else ""))
        elif top_sig == 'semantic_sim' and top_val > 0.05:
            cat_label = category_group.lower() if category_group != 'Other' else category.lower()
            primary = (f'Behaviorally similar trajectories '
                       f'visited this {cat_label}')
        elif top_sig == 'route_sim' and route_match:
            primary = (f"{route_match['source_count']} trajectories sharing "
                       f"{route_match['max_shared_pois']} POIs visited this "
                       f"{category.lower()}")
        elif top_sig == 'flow_prob':
            primary = 'Natural next step based on transition patterns from recent POIs'
        elif top_sig == 'rule_coverage' and rules_layer:
            labels = ', '.join(
                _RULE_LABELS.get(r, r) for r in rules_layer['matched'][:3])
            primary = (f"{rules_layer['count']}/5 SR rules endorse: "
                       f"{labels}")
        elif top_sig == 'dominant_ctx':
            primary = dom_str
        elif top_sig == 'user_pref':
            pref_label = category_group if category_group != 'Other' else category
            primary = f'Aligns with inferred {pref_label} preference'
        elif top_sig == 'significant':
            primary = 'Significant destination included by most visitors'
        elif top_sig == 'engagement':
            primary = 'Frequented by trajectories with matching engagement level'
        elif top_sig == 'event_match':
            event_ev = poi_evidence.get('event_match', {}) or {}
            label = event_ev.get('event_label') or 'a scheduled event'
            cat = event_ev.get('event_category', '').lower()
            when = _fmt_event_when(event_ev.get('event_start', ''))
            primary = (f"Scheduled {cat} \"{label}\" {when} "
                       f"overlapping the trajectory's time window"
                       if cat else
                       f"Scheduled event \"{label}\" {when} "
                       f"overlapping the trajectory's time window")
        else:
            primary = 'Converging evidence from multiple recommendation signals'

        # Supporting reasons from other active signals
        supporting = []
        if archetype and top_sig != 'archetype_sim':
            classes = ', '.join(archetype['shared_classes'][:2])
            supporting.append(
                f"{archetype['source_count']} archetypally similar "
                f"trajectories ({classes})")
        if route_match and top_sig != 'route_sim':
            supporting.append(
                f"{route_match['source_count']} shared-route trajectories "
                f"({route_match['max_shared_pois']} common POIs)")
        if signals.get('flow_prob', 0) > 0 and top_sig != 'flow_prob':
            supporting.append(
                f'Flow probability {signals["flow_prob"]:.0%} from recent POIs')
        if rules_layer and top_sig != 'rule_coverage':
            labels = ', '.join(
                _RULE_LABELS.get(r, r) for r in rules_layer['matched'][:3])
            supporting.append(
                f"{rules_layer['count']}/5 SR rules: {labels}")
        if signals.get('dominant_ctx', 0) > 0 and top_sig != 'dominant_ctx':
            supporting.append(dom_str)
        if signals.get('user_pref', 0) > 0 and top_sig != 'user_pref':
            pref_label = category_group if category_group != 'Other' else category
            supporting.append(f'Aligns with inferred {pref_label} preference')
        if signals.get('event_match', 0) > 0 and top_sig != 'event_match':
            event_ev = poi_evidence.get('event_match', {}) or {}
            label = event_ev.get('event_label') or 'a scheduled event'
            when = _fmt_event_when(event_ev.get('event_start', ''))
            supporting.append(f'Scheduled event "{label}" {when}')

        # Summary
        active = sum(1 for sig, v in signals.items()
                     if v > 0 and sig in weights)
        n_signals = sum(1 for w in weights.values() if w > 0)
        summary = (f"{category} recommended via {active}/{n_signals} signals"
                   + (f", {rules_layer['count']}/5 rules"
                      if rules_layer else ""))

        return {
            'primary_reason': primary,
            'supporting_reasons': supporting,
            'summary': summary,
        }

    def evaluate_recommendations(self, method: str = 'unified',
                                 **kwargs) -> Dict[str, Any]:
        """Dispatch 'unified' to self.recommend(), else delegate to parent."""
        if method == 'unified':
            # Temporarily patch context_aware_recommend to route through unified
            original = self.context_aware_recommend
            self.context_aware_recommend = lambda traj_uri, input_seq, top_k=20, \
                signal_weights=None, exclude_traj=None: \
                self.recommend(traj_uri, observed_sequence=input_seq, top_k=top_k,
                               signal_weights=signal_weights, exclude_traj=exclude_traj)
            try:
                return super().evaluate_recommendations(
                    method='context_aware', **kwargs)
            finally:
                self.context_aware_recommend = original
        return super().evaluate_recommendations(method=method, **kwargs)

    def format_evidence(self, signal: str, evidence: dict,
                        score: float) -> str:
        """Human-readable evidence string for a signal."""
        if signal == 'archetype_sim':
            sources = evidence.get('sources', [])
            if sources:
                top = sources[0]
                classes = ', '.join(top['shared'][:2])
                return (f"Archetype match: {classes} "
                        f"(from {len(sources)} similar trajectories)")
            return f"Archetype similarity score {score:.3f}"

        if signal == 'route_sim':
            sources = evidence.get('sources', [])
            if sources:
                best = max(sources, key=lambda s: s['shared_poi_count'])
                return (f"Route match: {len(sources)} trajectories "
                        f"sharing {best['shared_poi_count']} POIs")
            return f"Route similarity score {score:.3f}"

        if signal == 'flow_prob':
            source_pois = evidence.get('source_pois', [])
            return (f"Flow probability {score:.1%} from "
                    f"{len(source_pois)} recent POIs")

        if signal == 'rule_coverage':
            rules = evidence.get('rules', [])
            labels = ', '.join(_RULE_LABELS.get(r, r) for r in rules)
            return f"{len(rules)}/5 rules endorse: {labels}"

        if signal == 'event_match':
            label = evidence.get('event_label') or 'scheduled event'
            when = _fmt_event_when(evidence.get('event_start', ''))
            return f"\"{label}\" {when}"

        # Delegate inherited signals to parent
        return self._format_evidence(signal, evidence, score)
