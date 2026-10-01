"""
Context-Aware POI Recommender for Trajecto

Ported from bird/ pipeline's MuseumRecommender, adapted for GPS trajectory data.
Provides context-aware multi-signal recommendations, 4-layer next-POI prediction,
hybrid merging, and comprehensive evaluation.


Capabilities:
- 6-signal context-aware POI recommendations with KG predicate traceability
- 4-layer next-POI prediction (flow, context blend, covisit smoothing, category fallback)
- Hybrid merge of context-aware + flow prediction
- Evaluation: trajectory-split, leave-k-out, rank-based (NDCG, AUC)
- Baselines: popularity, random, existing subgraph recommender, SPARQL rules
- Bootstrapped confidence intervals, per-navigation-style breakdowns
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
from pathlib import Path
import json

from .fuseki_config import query_endpoint, update_endpoint
from .fuseki_queries import execute_sparql_query, format_results
from .poi_categories import resolve_category, CATEGORY_GROUPS

PREFIXES = """
PREFIX trajecto: <http://w3id.org/trajecto#>
PREFIX poi:      <http://w3id.org/trajecto/poi#>
PREFIX patterns: <http://w3id.org/trajecto/patterns#>
PREFIX context:  <http://w3id.org/trajecto/context#>
PREFIX user:     <http://w3id.org/trajecto/user#>
PREFIX purpose:  <http://w3id.org/trajecto/purpose#>
PREFIX dul:      <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#>
PREFIX datacron: <http://www.datacron-project.eu/datAcron#>
PREFIX rdf:      <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs:     <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:      <http://www.w3.org/2001/XMLSchema#>
"""

# Categories excluded from recommendations (transportation / utility / non-leisure)
EXCLUDED_CATEGORIES = {
    # Street furniture / utility
    'SeatingArea', 'PublicService', 'VendingMachine', 'WaterFountain',
    'Embassy',
    'ATM', 'Bank', 'PostOffice',
    'BusStation', 'TrainStation', 'Airport', 'Transportation', 'TransportationPOI',
    'TransitPOI', 'TransitStop',
    'ParkingArea',
    'Elevator', 'Escalator', 'Staircase',
    'Entrance', 'Exit', 'Intersection', 'Junction',
    'Transition', 'TransitionPOI', 'Waypoint', 'NavigationLandmark',
    'Restroom', 'RestRoom', 'ServicePOI',
    # Healthcare (utility, not a leisure/tourist rec)
    'Pharmacy', 'Hospital', 'Clinic', 'HealthcareFacility', 'MedicalFacility',
    'HealthcarePOI', 'DoctorsOffice', 'DentalOffice',
    # Business (not a consumer-leisure rec). ConferenceCenter intentionally
    # NOT listed — it is recommendable only when a scheduled event overlaps
    # the trajectory's time window (gated via event-match logic).
    'OfficeBuilding', 'GovernmentBuilding',
    'nan', 'None', 'Unknown', 'Uncategorized',
}

# Signal weights for context-aware recommendation
SIGNAL_WEIGHTS = {
    'semantic_sim': 0.30,
    'nav_match': 0.20,
    'significant': 0.15,
    'dominant_ctx': 0.15,
    'user_pref': 0.10,
    'engagement': 0.10,
}

# KG predicate provenance for each signal
SIGNAL_KG_PREDICATES = {
    'semantic_sim': 'Cosine similarity over SIM_SLICES feature groups (category + temporal + interaction — excludes navigation & engagement)',
    'nav_match': 'patterns:exhibitsPattern',
    'significant': 'poi:SignificantPOI (rdf:type)',
    'dominant_ctx': 'poi:covisitedWith (edge weight from covisit graph)',
    'user_pref': 'user:hasInferredPreference (category-based preference score)',
    'engagement': 'trajecto:hasEngagementLevel',
    'flow_prob': 'poi:followedBy (directed transition counts)',
}

# Pipeline step provenance for each signal
SIGNAL_PIPELINE = {
    'semantic_sim': 'Notebook 17 (Semantic Feature Extraction)',
    'nav_match': 'Notebook 6 (Navigation Style Classification)',
    'significant': 'Notebook 5 (Dominant & Significant POIs)',
    'dominant_ctx': 'Notebook 5 + SR-05 (Co-visitation Materialization)',
    'user_pref': 'Notebook 8-9 (User Profile & Preference Learning)',
    'engagement': 'Notebook 3 (Interaction Classification)',
    'flow_prob': 'Notebook 3 (Position Sequences) + SR-06 (poi:followedBy)',
}

# Similarity slice: category + temporal + interaction groups.
# Excludes navigation and engagement, which have dedicated signals.
# Ranges are derived from the extractor so they track the real vector layout.
from .semantic_features import feature_group_ranges as _fgr

_RANGES = _fgr()
SIM_SLICES = [_RANGES['category'], _RANGES['temporal'], _RANGES['interaction']]
SIM_DIM = sum(end - start for start, end in SIM_SLICES)


def category_preferences(interaction_lists) -> List[Dict]:
    """Category score = 0.4 * immersive_frac + 0.5 * meaningful_frac + 0.1 * visit_norm,
    over the given trajectories' interactions."""
    stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {'total': 0, 'immersive': 0, 'meaningful': 0})
    for interactions in interaction_lists:
        for item in interactions:
            cat = item.get('category', 'Unknown')
            if cat in EXCLUDED_CATEGORIES or cat in ('PointOfInterest', 'SignificantPOI'):
                continue
            stats[cat]['total'] += 1
            if item['int_type'] == 'ImmersiveInteraction':
                stats[cat]['immersive'] += 1
            if item['int_type'] in ('MeaningfulInteraction', 'ImmersiveInteraction'):
                stats[cat]['meaningful'] += 1
    max_visits = max((st['total'] for st in stats.values()), default=1)
    prefs = [{'category': cat,
              'score': 0.4 * st['immersive'] / st['total']
                       + 0.5 * st['meaningful'] / st['total']
                       + 0.1 * st['total'] / max_visits}
             for cat, st in stats.items() if st['total']]
    prefs.sort(key=lambda x: -x['score'])
    return prefs


class ContextAwareRecommender:
    """
    Context-aware POI recommender with KG predicate traceability.

    6 signals for recommendation, 4-layer next-POI prediction, hybrid merge,
    and comprehensive evaluation framework.

    Usage:
        rec = ContextAwareRecommender()
        rec.load_all_data()
        recs = rec.context_aware_recommend(traj_uri, explain=True)
        preds = rec.predict_next_poi(sequence, context)
    """

    def __init__(self, endpoint: str = None, upd_endpoint: str = None):
        self.endpoint = endpoint or query_endpoint
        self.update_endpoint = upd_endpoint or update_endpoint

        # Data stores (populated by load_all_data)
        self.features = {}              # traj_uri -> 67-dim np.ndarray
        self.sequences = {}             # traj_uri -> [poi_uri, ...]
        self.traj_metadata = {}         # traj_uri -> {user, purposes, nav_styles, engagement}
        self.poi_metadata = {}          # poi_uri -> {label, category, popularity, lat, lon}
        self.interaction_data = {}      # traj_uri -> [{poi, dwell, int_type, category}, ...]
        self.user_preferences = {}      # user_uri -> [{category, score}, ...]
        self.traj_to_user = {}
        self.user_to_trajs = {}         # user_uri -> [traj_uri, ...]

        # Graphs
        self.flow_graph = None          # nx.DiGraph
        self.covisit_graph = None       # nx.Graph

        # Similarity engine
        self.sim_engine = None

        # Derived
        self.all_poi_uris = set()
        self.traj_poi_sets = {}         # traj_uri -> set of poi_uris
        self.category_transitions = {}  # src_cat -> {dst_cat: prob}
        self.significant_pois = set()

        # Precomputed SIM_DIM normalized matrix for fast similarity
        self._sim_matrix = None
        self._sim_uris = []

        self._loaded = False

    def _query(self, sparql: str) -> pd.DataFrame:
        result = execute_sparql_query(self.endpoint, PREFIXES + sparql)
        return format_results(result)

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------

    def load_all_data(self, features_path: str = None) -> None:
        """Load all data from Fuseki + cached features."""
        print("Loading context-aware recommender data...")

        self._load_features(features_path)
        self._load_interactions()
        self._load_trajectory_metadata()
        self._load_poi_metadata()

        eng_count = self._compute_engagement_levels()
        print(f"  Engagement: {eng_count} trajectories classified")

        self._load_user_preferences()
        self._load_graphs()
        self._build_derived()

        self.significant_pois = self._get_significant_pois()
        print(f"  Significant POIs: {len(self.significant_pois)}")

        self._loaded = True
        print(f"Loaded: {len(self.features)} trajectories, "
              f"{len(self.all_poi_uris)} POIs, "
              f"{len(self.user_preferences)} users with preferences")

    def _load_features(self, features_path: str = None) -> None:
        if features_path and Path(features_path).exists():
            with open(features_path) as f:
                raw = json.load(f)
            self.features = {uri: np.array(vec, dtype=np.float32)
                             for uri, vec in raw.items()}
            print(f"  Features: loaded {len(self.features)} from file")
        else:
            from .semantic_features import SemanticFeatureExtractor
            extractor = SemanticFeatureExtractor(self.endpoint)
            self.features = extractor.batch_extract_features()
            print(f"  Features: extracted {len(self.features)} from Fuseki")

    def _extract_category_name(self, category: str) -> str:
        """Extract local name from category URI and resolve via aliases."""
        if category is None:
            return 'Unknown'
        cat_str = str(category)
        if '#' in cat_str:
            raw = cat_str.split('#')[-1]
        elif '/' in cat_str:
            raw = cat_str.split('/')[-1]
        else:
            raw = cat_str
        resolved = resolve_category(raw)
        return resolved if resolved else raw

    def _get_category_group(self, category: str) -> str:
        """Map category name to its parent group."""
        cat_name = self._extract_category_name(category)
        for group, members in CATEGORY_GROUPS.items():
            if cat_name in members:
                return group
        return 'Other'

    def _load_interactions(self) -> None:
        df = self._query("""
        SELECT ?traj ?pos ?poi ?dwell ?intType ?poiType ?timestamp
        WHERE {
            ?pos dul:isPartOf ?traj .
            ?pos trajecto:hasInteraction ?interaction .
            ?interaction poi:interactsWith ?poi .
            ?interaction poi:hasDwellTimeInSeconds ?dwell .
            ?interaction a ?intType .
            FILTER(?intType IN (poi:PassingInteraction, poi:MeaningfulInteraction,
                                poi:ImmersiveInteraction, poi:DominantInteraction,
                                poi:StartInteraction, poi:DestinationInteraction))
            OPTIONAL {
                ?poi a ?poiType .
                FILTER(?poiType != poi:SignificantPOI && ?poiType != trajecto:PointOfInterest)
            }
            OPTIONAL { ?pos trajecto:timestamp ?timestamp . }
        }
        ORDER BY ?traj ?timestamp
        """)

        if df.empty:
            print("  Interactions: no data found")
            return

        if 'dwell' in df.columns:
            df['dwell'] = pd.to_numeric(df['dwell'], errors='coerce')

        for traj in df['traj'].unique():
            traj_df = df[df['traj'] == traj].copy()
            if 'timestamp' in traj_df.columns:
                traj_df = traj_df.sort_values('timestamp')

            interactions = []
            sequence = []
            seen_in_seq = None

            for _, row in traj_df.iterrows():
                int_type = row['intType'].split('#')[-1] if pd.notna(row.get('intType')) else ''
                cat = self._extract_category_name(row.get('poiType', '')) if pd.notna(row.get('poiType')) and row.get('poiType') else 'Unknown'

                interactions.append({
                    'poi': row['poi'],
                    'dwell': float(row['dwell']) if pd.notna(row.get('dwell')) else 0.0,
                    'int_type': int_type,
                    'category': cat,
                })
                self.all_poi_uris.add(row['poi'])

                # Deduplicated sequence (meaningful+ interactions)
                if int_type in ('MeaningfulInteraction', 'ImmersiveInteraction',
                                'DominantInteraction'):
                    if row['poi'] != seen_in_seq:
                        sequence.append(row['poi'])
                        seen_in_seq = row['poi']

            self.interaction_data[traj] = interactions
            self.sequences[traj] = sequence
            self.traj_poi_sets[traj] = {i['poi'] for i in interactions}

        print(f"  Interactions: {len(self.interaction_data)} trajectories, "
              f"{len(self.all_poi_uris)} unique POIs")

    def _load_trajectory_metadata(self) -> None:
        df = self._query("""
        SELECT ?traj ?user ?purpose ?navStyle ?engagement
        WHERE {
            ?user trajecto:formsTrajectory ?traj .
            OPTIONAL { ?traj trajecto:hasPurpose ?purpose . }
            OPTIONAL { ?traj patterns:exhibitsPattern ?navStyle . }
            OPTIONAL { ?traj trajecto:hasEngagementLevel ?engagement . }
        }
        """)

        if df.empty:
            return

        for _, row in df.iterrows():
            traj = row['traj']
            user = row['user']
            self.traj_to_user[traj] = user
            if user not in self.user_to_trajs:
                self.user_to_trajs[user] = []
            if traj not in [t for t in self.user_to_trajs[user]]:
                self.user_to_trajs[user].append(traj)

            if traj not in self.traj_metadata:
                self.traj_metadata[traj] = {
                    'user': user,
                    'purposes': set(),
                    'nav_styles': set(),
                    'engagement': None,
                }

            if pd.notna(row.get('purpose')) and row['purpose']:
                self.traj_metadata[traj]['purposes'].add(row['purpose'].split('#')[-1])
            if pd.notna(row.get('navStyle')) and row['navStyle']:
                self.traj_metadata[traj]['nav_styles'].add(row['navStyle'].split('#')[-1])
            if pd.notna(row.get('engagement')) and row['engagement']:
                self.traj_metadata[traj]['engagement'] = row['engagement'].split('#')[-1]

        print(f"  Metadata: {len(self.traj_metadata)} trajectories")

    def _load_poi_metadata(self) -> None:
        df = self._query("""
        SELECT ?poi ?label ?poiType ?lat ?lon
        WHERE {
            ?poi a trajecto:PointOfInterest .
            OPTIONAL { ?poi rdfs:label ?label . }
            OPTIONAL {
                ?poi a ?poiType .
                FILTER(?poiType != poi:SignificantPOI && ?poiType != trajecto:PointOfInterest)
            }
            OPTIONAL { ?poi datacron:hasLatitude ?lat . }
            OPTIONAL { ?poi datacron:hasLongitude ?lon . }
        }
        """)

        if df.empty:
            return

        # Compute popularity: fraction of trajectories visiting each POI
        poi_visit_counts = Counter()
        for arts in self.traj_poi_sets.values():
            for a in arts:
                poi_visit_counts[a] += 1
        n_trajs = max(len(self.traj_poi_sets), 1)

        for _, row in df.iterrows():
            poi = row['poi']
            cat = self._extract_category_name(row.get('poiType', '')) if pd.notna(row.get('poiType')) and row.get('poiType') else 'Unknown'

            lat = float(row['lat']) if pd.notna(row.get('lat')) else None
            lon = float(row['lon']) if pd.notna(row.get('lon')) else None

            self.poi_metadata[poi] = {
                'label': row.get('label', poi.split('/')[-1]) if pd.notna(row.get('label')) else poi.split('/')[-1],
                'category': cat,
                'popularity': poi_visit_counts.get(poi, 0) / n_trajs,
                'lat': lat,
                'lon': lon,
            }

        print(f"  POI metadata: {len(self.poi_metadata)} POIs")

    def _compute_engagement_levels(self) -> int:
        """Compute engagement from interaction data (main pipeline SR-10 thresholds).

        High: >= 3 immersive interactions
        Medium: >= 5 meaningful AND < 3 immersive
        Low: remainder
        """
        updated = 0
        for traj, interactions in self.interaction_data.items():
            if traj not in self.traj_metadata:
                continue
            immersive_count = sum(
                1 for i in interactions if i['int_type'] == 'ImmersiveInteraction')
            meaningful_count = sum(
                1 for i in interactions if i['int_type'] in
                ('MeaningfulInteraction', 'ImmersiveInteraction'))

            if immersive_count >= 3:
                level = 'HighEngagement'
            elif meaningful_count >= 5:
                level = 'MediumEngagement'
            else:
                level = 'LowEngagement'

            self.traj_metadata[traj]['engagement'] = level
            updated += 1
        return updated

    def _load_user_preferences(self) -> None:
        """Compute user preferences from interaction data (see category_preferences)."""
        if not self.interaction_data or not self.traj_to_user:
            print("  Preferences: none (no interaction data)")
            return
        by_user = defaultdict(list)
        for traj, interactions in self.interaction_data.items():
            user = self.traj_to_user.get(traj)
            if user:
                by_user[user].append(interactions)
        for user, lists in by_user.items():
            self.user_preferences[user] = category_preferences(lists)

        print(f"  Preferences: {len(self.user_preferences)} users")

    def _load_graphs(self) -> None:
        """Build flow and covisit graphs from in-memory interaction data.

        Also stores per-trajectory edge contributions for leave-one-out
        evaluation (subtract a trajectory's edges to avoid data leakage).
        """
        import networkx as nx

        # Flow graph: consecutive POI transitions from sequences
        self.flow_graph = nx.DiGraph()
        transition_counts: Dict[tuple, int] = {}
        self._traj_flow_edges: Dict[str, Dict[tuple, int]] = {}
        for traj, seq in self.sequences.items():
            traj_edges: Dict[tuple, int] = {}
            for i in range(len(seq) - 1):
                if seq[i] != seq[i + 1]:
                    edge = (seq[i], seq[i + 1])
                    transition_counts[edge] = transition_counts.get(edge, 0) + 1
                    traj_edges[edge] = traj_edges.get(edge, 0) + 1
            self._traj_flow_edges[traj] = traj_edges
        for (p1, p2), count in transition_counts.items():
            self.flow_graph.add_edge(p1, p2, weight=count, transitions=count)

        # Covisit graph: POIs co-visited in same trajectory (meaningful+)
        self.covisit_graph = nx.Graph()
        covisit_counts: Dict[tuple, int] = {}
        self._traj_covisit_edges: Dict[str, Dict[tuple, int]] = {}
        for traj, interactions in self.interaction_data.items():
            meaningful_pois = set()
            for item in interactions:
                if item['int_type'] in ('MeaningfulInteraction',
                                        'ImmersiveInteraction',
                                        'DominantInteraction'):
                    meaningful_pois.add(item['poi'])
            pois = sorted(meaningful_pois)
            traj_covisits: Dict[tuple, int] = {}
            for i in range(len(pois)):
                for j in range(i + 1, len(pois)):
                    edge = (pois[i], pois[j])
                    covisit_counts[edge] = covisit_counts.get(edge, 0) + 1
                    traj_covisits[edge] = traj_covisits.get(edge, 0) + 1
            self._traj_covisit_edges[traj] = traj_covisits
        for (p1, p2), count in covisit_counts.items():
            self.covisit_graph.add_edge(p1, p2, weight=count, covisits=count)

        print(f"  Graphs: flow={self.flow_graph.number_of_edges()} edges, "
              f"covisit={self.covisit_graph.number_of_edges()} edges")

    def _flow_weight(self, src: str, dst: str, exclude_traj: str = None) -> int:
        """Get flow edge weight, subtracting excluded trajectory's contribution."""
        if not self.flow_graph or not self.flow_graph.has_edge(src, dst):
            return 0
        w = self.flow_graph[src][dst].get('weight', 0)
        if exclude_traj:
            w -= self._traj_flow_edges.get(exclude_traj, {}).get((src, dst), 0)
        return max(w, 0)

    def _covisit_weight(self, p1: str, p2: str, exclude_traj: str = None) -> int:
        """Get covisit edge weight, subtracting excluded trajectory's contribution."""
        if not self.covisit_graph or not self.covisit_graph.has_edge(p1, p2):
            return 0
        w = self.covisit_graph[p1][p2].get('weight', 0)
        if exclude_traj:
            edge = (min(p1, p2), max(p1, p2))
            w -= self._traj_covisit_edges.get(exclude_traj, {}).get(edge, 0)
        return max(w, 0)

    def _build_derived(self) -> None:
        """Build similarity engine, category transitions, and precomputed similarity matrix."""
        from .semantic_similarity import TrajectorySimilarityEngine
        self.sim_engine = TrajectorySimilarityEngine(
            semantic_features=dict(self.features)
        )
        self.sim_engine._semantic_uris = list(self.features.keys())
        self.sim_engine._semantic_matrix = np.array([
            self.features[uri] for uri in self.sim_engine._semantic_uris
        ])

        # Category transitions from sequences
        cat_counts = defaultdict(Counter)
        for traj, seq in self.sequences.items():
            for i in range(len(seq) - 1):
                cat_src = self.poi_metadata.get(seq[i], {}).get('category')
                cat_dst = self.poi_metadata.get(seq[i + 1], {}).get('category')
                if cat_src and cat_dst:
                    cat_counts[cat_src][cat_dst] += 1

        for src, targets in cat_counts.items():
            total = sum(targets.values())
            self.category_transitions[src] = {
                dst: count / total for dst, count in targets.items()
            }

        # Precompute POI popularity: count of trajectories visiting each POI
        self._poi_popularity = defaultdict(int)
        for poi_set in self.traj_poi_sets.values():
            for poi in poi_set:
                self._poi_popularity[poi] += 1

        # Precompute nav style → POI counts (across all trajectories)
        self._nav_style_pois = defaultdict(lambda: defaultdict(int))
        for traj, meta in self.traj_metadata.items():
            for style in meta.get('nav_styles', set()):
                for poi in self.traj_poi_sets.get(traj, set()):
                    self._nav_style_pois[style][poi] += 1

        # Precompute engagement level → POI counts
        self._engagement_level_pois = defaultdict(lambda: defaultdict(int))
        for traj, meta in self.traj_metadata.items():
            eng = meta.get('engagement')
            if eng:
                for poi in self.traj_poi_sets.get(traj, set()):
                    self._engagement_level_pois[eng][poi] += 1

        # Precompute nav-style-conditioned transitions for predict_next_poi
        self._nav_transitions = {}  # style → {(src, dst): count}
        for traj, meta in self.traj_metadata.items():
            seq = self.sequences.get(traj, [])
            for style in meta.get('nav_styles', set()):
                if style not in self._nav_transitions:
                    self._nav_transitions[style] = defaultdict(int)
                for j in range(len(seq) - 1):
                    if seq[j] != seq[j + 1]:
                        self._nav_transitions[style][(seq[j], seq[j + 1])] += 1

        # Per-trajectory contribution to nav transitions (for leave-one-out)
        self._traj_nav_transitions = {}  # traj → {style: {(src, dst): count}}
        for traj, meta in self.traj_metadata.items():
            seq = self.sequences.get(traj, [])
            if len(seq) < 2:
                continue
            traj_contribs = {}
            for style in meta.get('nav_styles', set()):
                contribs = defaultdict(int)
                for j in range(len(seq) - 1):
                    if seq[j] != seq[j + 1]:
                        contribs[(seq[j], seq[j + 1])] += 1
                traj_contribs[style] = dict(contribs)
            if traj_contribs:
                self._traj_nav_transitions[traj] = traj_contribs

        # Precompute avg dwell time per POI (avoids O(total_interactions) per lookup)
        dwell_accum = defaultdict(list)
        for interactions in self.interaction_data.values():
            for item in interactions:
                if item['dwell'] > 0:
                    dwell_accum[item['poi']].append(item['dwell'])
        self._avg_dwell_cache = {poi: np.mean(dwells)
                                 for poi, dwells in dwell_accum.items()}

        # Precompute SIM_DIM normalized matrix for fast similarity
        self._build_sim_matrix()

    def _build_sim_matrix(self) -> None:
        """Precompute normalized SIM_DIM matrix for vectorized similarity."""
        if not self.features:
            return

        self._sim_uris = list(self.features.keys())
        raw = np.array([self.features[uri] for uri in self._sim_uris], dtype=np.float32)

        # Extract SIM_SLICES groups: category + temporal + interaction
        slices = [raw[:, s:e] for s, e in SIM_SLICES]
        matrix = np.hstack(slices)

        # L2 normalize for cosine similarity via dot product
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._sim_matrix = matrix / norms

    def _get_significant_pois(self) -> set:
        df = self._query("""
        SELECT DISTINCT ?poi WHERE {
            ?poi a poi:SignificantPOI .
        }
        """)
        if df.empty:
            return set()
        return set(df['poi'].tolist())

    def _get_dominant_poi(self, traj_uri: str) -> Optional[str]:
        interactions = self.interaction_data.get(traj_uri, [])
        for item in interactions:
            if item['int_type'] == 'DominantInteraction':
                return item['poi']
        return None

    # ------------------------------------------------------------------
    # Forward-Reachable Candidate Set
    # ------------------------------------------------------------------

    def _get_forward_candidates(self, sequence: List[str],
                                visited: Set[str],
                                nav_style: str = None,
                                exclude_traj: str = None) -> Set[str]:
        """Return POIs reachable via directed flow edges from last 3 POIs."""
        forward = set()
        if not sequence:
            return forward

        last_pois = sequence[-3:] if len(sequence) >= 3 else list(sequence)

        # Layer 1: Direct flow graph outgoing edges (leave-one-out)
        if self.flow_graph:
            for poi in last_pois:
                if self.flow_graph.has_node(poi):
                    for _, dst in self.flow_graph.out_edges(poi):
                        if dst not in visited:
                            if self._flow_weight(poi, dst, exclude_traj) > 0:
                                forward.add(dst)

        # Layer 2: Nav-style-conditioned transitions (precomputed)
        if nav_style and nav_style in self._nav_transitions:
            trans = self._nav_transitions[nav_style]
            # Subtract excluded trajectory's contributions if needed
            excl_contribs = {}
            if exclude_traj and exclude_traj in self._traj_nav_transitions:
                excl_contribs = self._traj_nav_transitions[exclude_traj].get(
                    nav_style, {})
            seed_pois = set(last_pois[-2:])
            for (src, dst), count in trans.items():
                if src in seed_pois and dst not in visited:
                    adj_count = count - excl_contribs.get((src, dst), 0)
                    if adj_count > 0:
                        forward.add(dst)

        return forward

    # ------------------------------------------------------------------
    # Context-Aware Recommendation
    # ------------------------------------------------------------------

    def context_aware_recommend(self, traj_uri: str,
                                observed_sequence: List[str] = None,
                                top_k: int = 20,
                                signal_weights: Dict[str, float] = None,
                                explain: bool = False,
                                exclude_traj: str = None) -> List[Dict]:
        """
        Multi-signal POI recommendation.

        Signals:
        - semantic_sim (0.30): POIs from semantically similar trajectories (SIM_DIM)
        - nav_match (0.20): POIs from trajectories sharing navigation style
        - significant (0.15): Boost significant POIs visitor missed
        - dominant_ctx (0.15): Co-visited with visitor's dominant POI
        - user_pref (0.10): Preference score alignment
        - engagement (0.10): Same engagement level trajectories

        When explain=True, each recommendation includes an 'explanations' list
        with signal provenance tracing to KG predicates and pipeline steps.
        """
        weights = signal_weights or SIGNAL_WEIGHTS
        visited = set(observed_sequence or self.sequences.get(traj_uri, []))
        meta = self.traj_metadata.get(traj_uri, {})

        # Forward-reachable candidates
        input_seq = list(observed_sequence or self.sequences.get(traj_uri, []))
        nav_styles = meta.get('nav_styles', set())
        nav_style = list(nav_styles)[0] if nav_styles else None
        forward_set = self._get_forward_candidates(input_seq, visited, nav_style, exclude_traj)
        candidates = forward_set if forward_set else (self.all_poi_uris - visited)

        # Filter excluded categories
        candidates = {c for c in candidates
                      if self.poi_metadata.get(c, {}).get('category', '') not in EXCLUDED_CATEGORIES}

        scores = defaultdict(lambda: defaultdict(float))
        evidence = defaultdict(lambda: defaultdict(dict)) if explain else None

        # 1. Semantic similarity signal (SIM_DIM, no nav/engagement overlap)
        if traj_uri in self.features:
            similar = self._find_similar_63dim(traj_uri, top_k=10)
            for sim_traj, sim_score in similar:
                sim_pois = self.traj_poi_sets.get(sim_traj, set())
                for poi in (sim_pois & candidates):
                    scores[poi]['semantic_sim'] += sim_score
                    if explain:
                        contribs = evidence[poi]['semantic_sim'].setdefault('contributors', [])
                        contribs.append({'traj': sim_traj, 'similarity': sim_score})

        # 2. Navigation style match signal (precomputed index)
        own_pois = self.traj_poi_sets.get(traj_uri, set())
        if nav_styles:
            for style in nav_styles:
                style_pois = self._nav_style_pois.get(style, {})
                for poi in candidates:
                    count = style_pois.get(poi, 0)
                    # Subtract own trajectory's contribution
                    if poi in own_pois and count > 0:
                        count -= 1
                    if count > 0:
                        scores[poi]['nav_match'] += float(count)
                        if explain:
                            evidence[poi]['nav_match'] = {
                                'nav_style': nav_style or '?',
                                'match_count': count,
                            }

        # 3. Significant POIs signal
        sig_pois = self.significant_pois or self._get_significant_pois()
        for poi in (sig_pois & candidates):
            scores[poi]['significant'] += 1.0
            if explain:
                evidence[poi]['significant'] = {
                    'total_significant': len(sig_pois),
                }

        # 4. Dominant context signal — co-visited with visitor's dominant POI
        dominant = self._get_dominant_poi(traj_uri)
        if dominant and self.covisit_graph and self.covisit_graph.has_node(dominant):
            dominant_label = self.poi_metadata.get(dominant, {}).get('label', '?')
            dominant_dwell = 0
            if explain:
                for item in self.interaction_data.get(traj_uri, []):
                    if item['poi'] == dominant:
                        dominant_dwell = item['dwell']
                        break
            for neighbor in self.covisit_graph.neighbors(dominant):
                if neighbor in candidates:
                    edge_w = self._covisit_weight(dominant, neighbor, exclude_traj)
                    if edge_w > 0:
                        scores[neighbor]['dominant_ctx'] += edge_w
                        if explain:
                            evidence[neighbor]['dominant_ctx'] = {
                                'dominant_label': dominant_label,
                                'dominant_dwell': dominant_dwell,
                                'covisit_weight': edge_w,
                            }

        # 5. User preference alignment signal (category only)
        user = self.traj_to_user.get(traj_uri)
        prefs = self.user_preferences.get(user, []) if user else []
        pref_map = {p['category']: p['score'] for p in prefs}
        for poi in candidates:
            poi_meta = self.poi_metadata.get(poi, {})
            cat = poi_meta.get('category', '')
            cat_score = pref_map.get(cat, 0)
            if cat_score > 0:
                scores[poi]['user_pref'] += cat_score
                if explain:
                    evidence[poi]['user_pref'] = {
                        'category': cat, 'cat_score': cat_score,
                    }

        # 6. Engagement level signal (precomputed index)
        engagement = meta.get('engagement')
        if engagement and engagement in self._engagement_level_pois:
            eng_pois = self._engagement_level_pois[engagement]
            for poi in candidates:
                count = eng_pois.get(poi, 0)
                if poi in own_pois and count > 0:
                    count -= 1
                if count > 0:
                    scores[poi]['engagement'] += float(count)
                    if explain:
                        evidence[poi]['engagement'] = {
                            'engagement_level': engagement,
                            'match_count': count,
                        }

        # Normalize each signal to [0, 1]
        for signal in weights:
            signal_vals = [scores[poi][signal] for poi in scores if scores[poi][signal] > 0]
            if signal_vals:
                max_val = max(signal_vals)
                if max_val > 0:
                    for poi in scores:
                        if scores[poi][signal] > 0:
                            scores[poi][signal] /= max_val

        # Compute weighted aggregate
        max_score = 0.0
        ranked = []
        for poi, signals in scores.items():
            total = sum(weights.get(sig, 0) * val for sig, val in signals.items())
            if total > max_score:
                max_score = total
            ranked.append((poi, signals, total))

        ranked.sort(key=lambda x: (-x[2], x[0]))   # URI tie-break: reproducibility
        ranked = ranked[:top_k]

        # Visited categories for context
        visited_cat_names = set()
        for v_poi in visited:
            c = self.poi_metadata.get(v_poi, {}).get('category', '')
            if c and c != 'Unknown':
                visited_cat_names.add(c)

        # Build enriched output
        results = []
        for poi, signals, total in ranked:
            poi_meta = self.poi_metadata.get(poi, {})
            category = poi_meta.get('category', 'Unknown')
            category_group = self._get_category_group(category)
            lat = poi_meta.get('lat')
            lon = poi_meta.get('lon')
            popularity_count = self._poi_popularity.get(poi, 0)

            # Confidence based on signal coverage + score strength
            active_signals = sum(1 for v in signals.values() if v > 0)
            norm_score = total / max_score if max_score > 0 else 0
            confidence_val = 0.5 * norm_score + 0.5 * min(active_signals / 4, 1.0)
            if confidence_val >= 0.6:
                confidence = 'High'
            elif confidence_val >= 0.35:
                confidence = 'Medium'
            else:
                confidence = 'Low'

            # Dwell time stats
            avg_dwell = self._get_avg_dwell(poi)
            dwell_label = self._get_dwell_time_label(avg_dwell)

            # Popularity rank
            popularity_rank = self._get_popularity_rank(popularity_count)

            rec_dict = {
                'poi': poi,
                'label': poi_meta.get('label', poi.split('/')[-1]),
                'category': category,
                'category_group': category_group,
                'score': total,
                'confidence': confidence,
                'latitude': float(lat) if lat is not None else None,
                'longitude': float(lon) if lon is not None else None,
                'popularity_rank': popularity_rank,
                'dwell_time_label': dwell_label,
                'signals': dict(signals),
            }

            if explain:
                explanations = []
                for sig, val in signals.items():
                    w = weights.get(sig, 0)
                    contribution = w * val
                    if contribution <= 0:
                        continue
                    ev = evidence.get(poi, {}).get(sig, {})
                    explanations.append({
                        'signal': sig,
                        'contribution': contribution,
                        'normalized_score': val,
                        'weight': w,
                        'evidence': self._format_evidence(sig, ev, val),
                        'kg_predicate': SIGNAL_KG_PREDICATES.get(sig, ''),
                        'pipeline_step': SIGNAL_PIPELINE.get(sig, ''),
                    })
                explanations.sort(key=lambda x: -x['contribution'])
                rec_dict['explanations'] = explanations

            # Explanation dict in the format the map visualizer reads
            rec_dict['explanation'] = self._generate_explanation(
                rec_dict, category_group, visited_cat_names, popularity_count
            )

            results.append(rec_dict)

        return results

    def rank_unvisited(self, traj_uri: str,
                       observed_sequence: List[str] = None,
                       exclude_traj: str = None) -> List[Dict]:
        """Rank all unvisited POIs by score (for rank-based evaluation)."""
        return self.context_aware_recommend(
            traj_uri, observed_sequence, top_k=len(self.all_poi_uris),
            exclude_traj=exclude_traj
        )

    # ------------------------------------------------------------------
    # Hybrid Merge
    # ------------------------------------------------------------------

    def hybrid_merge_recommend(self, traj_uri: str,
                               observed_sequence: List[str] = None,
                               top_k: int = 20,
                               ctx_weight: float = 0.5,
                               flow_weight: float = 0.5,
                               explain: bool = False,
                               exclude_traj: str = None) -> List[Dict]:
        """Merge context-aware and flow prediction ranked lists.

        Normalizes both to [0,1], combines with given weights, returns top_k.
        Args:
            exclude_traj: Passed to predict_next_poi and context_aware_recommend
                for leave-one-out evaluation.
        """
        input_seq = observed_sequence or self.sequences.get(traj_uri, [])

        ctx_recs = self.context_aware_recommend(
            traj_uri, input_seq, top_k=top_k * 2, explain=explain,
            exclude_traj=exclude_traj
        )

        context = {}
        nav_styles = self.traj_metadata.get(traj_uri, {}).get('nav_styles', set())
        if nav_styles:
            context['nav_style'] = list(nav_styles)[0]
        flow_recs = self.predict_next_poi(input_seq, context=context, top_k=top_k * 2,
                                          exclude_traj=exclude_traj)

        ctx_scores = {r['poi']: r['score'] for r in ctx_recs}
        flow_scores = {r['poi']: r['raw_score'] for r in flow_recs}
        ctx_max = max(ctx_scores.values()) if ctx_scores else 1.0
        flow_max = max(flow_scores.values()) if flow_scores else 1.0

        ctx_explanations = {r['poi']: r.get('explanations', []) for r in ctx_recs} if explain else {}
        flow_sources = {r['poi']: r.get('source', '') for r in flow_recs}

        all_pois = set(ctx_scores) | set(flow_scores)
        merged = {}
        for poi in all_pois:
            c = ctx_scores.get(poi, 0) / ctx_max if ctx_max > 0 else 0
            f = flow_scores.get(poi, 0) / flow_max if flow_max > 0 else 0
            merged[poi] = ctx_weight * c + flow_weight * f

        # Build lookup from context-aware recs (carry enrichment forward)
        ctx_lookup = {r['poi']: r for r in ctx_recs}

        ranked = sorted(merged.items(), key=lambda x: (-x[1], x[0]))[:top_k]
        results = []
        for p, s in ranked:
            poi_meta = self.poi_metadata.get(p, {})
            category = poi_meta.get('category', 'Unknown')

            # Inherit enrichment from context-aware rec if available
            ctx_rec = ctx_lookup.get(p, {})

            rec_dict = {
                'poi': p,
                'label': poi_meta.get('label', p.split('/')[-1]),
                'category': category,
                'category_group': ctx_rec.get('category_group', self._get_category_group(category)),
                'score': s,
                'confidence': ctx_rec.get('confidence', 'Medium'),
                'latitude': ctx_rec.get('latitude', float(poi_meta['lat']) if poi_meta.get('lat') is not None else None),
                'longitude': ctx_rec.get('longitude', float(poi_meta['lon']) if poi_meta.get('lon') is not None else None),
                'popularity_rank': ctx_rec.get('popularity_rank', ''),
                'dwell_time_label': ctx_rec.get('dwell_time_label', ''),
                'explanation': ctx_rec.get('explanation', {
                    'primary_reason': 'Predicted by hybrid context + flow model',
                    'supporting_reasons': [],
                    'summary': 'Matches route patterns and contextual signals.',
                }),
                'source': 'hybrid_merge',
                'flow_source': flow_sources.get(p, ''),
                'signals': ctx_rec.get('signals', {}),
            }
            if explain:
                rec_dict['explanations'] = ctx_explanations.get(p, [])
            results.append(rec_dict)
        return results

    # ------------------------------------------------------------------
    # Sliced similarity (local, no modification to semantic_similarity.py)
    # ------------------------------------------------------------------

    def _find_similar_63dim(self, traj_uri: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Fast 63-dim cosine similarity via precomputed normalized matrix."""
        if traj_uri not in self.features or self._sim_matrix is None:
            return []

        try:
            idx = self._sim_uris.index(traj_uri)
        except ValueError:
            return []

        query_vec = self._sim_matrix[idx]
        sims = self._sim_matrix @ query_vec  # cosine sim via dot product on L2-normalized
        sims[idx] = -1.0  # exclude self

        top_indices = np.argpartition(sims, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

        return [(self._sim_uris[i], float(sims[i])) for i in top_indices if sims[i] > 0]

    # ------------------------------------------------------------------
    # Next-POI Prediction (4-layer)
    # ------------------------------------------------------------------

    def predict_next_poi(self, current_sequence: List[str],
                         context: Dict = None,
                         top_k: int = 10,
                         exclude_traj: str = None) -> List[Dict]:
        """
        4-layer next-POI prediction.

        Layers:
        1. Base flow: Global transitions from last 3 POIs, recency-weighted [0.5, 0.3, 0.2]
        2. Context blend: Nav-style-conditioned flow, 0.6/0.4 blend
        3. Covisit smoothing: 0.05x for unobserved transitions via covisit graph
        4. Category fallback: Top 3 category-level transitions, 0.01x prior

        Args:
            exclude_traj: If set, subtract this trajectory's contributions from
                flow/covisit graphs (leave-one-out to prevent data leakage
                during evaluation).
        """
        if not current_sequence:
            return []

        candidates = defaultdict(float)
        candidate_sources = defaultdict(str)
        visited_set = set(current_sequence)

        # Layer 1: Base flow transitions (leave-one-out adjusted)
        last_pois = current_sequence[-3:] if len(current_sequence) >= 3 else current_sequence
        recency_weights = [0.5, 0.3, 0.2]

        for i, poi in enumerate(reversed(last_pois)):
            w = recency_weights[i] if i < len(recency_weights) else 0.1
            if self.flow_graph and self.flow_graph.has_node(poi):
                out_edges = list(self.flow_graph.out_edges(poi, data=True))
                # Compute adjusted weights
                adj_weights = []
                for _, dst, data in out_edges:
                    aw = self._flow_weight(poi, dst, exclude_traj)
                    adj_weights.append((dst, aw))
                total_w = sum(aw for _, aw in adj_weights)
                if total_w > 0:
                    for dst, aw in adj_weights:
                        if dst not in visited_set and aw > 0:
                            prob = w * (aw / total_w)
                            candidates[dst] += prob
                            if dst not in candidate_sources:
                                candidate_sources[dst] = 'flow'

        # Layer 2: Context blend (nav-style-conditioned, precomputed)
        if context:
            nav_style = context.get('nav_style')
            if nav_style and nav_style in self._nav_transitions:
                ctx_transitions = dict(self._nav_transitions[nav_style])
                # Leave-one-out: subtract excluded trajectory's contributions
                if exclude_traj and exclude_traj in self._traj_nav_transitions:
                    for (s, d), c in self._traj_nav_transitions[exclude_traj].get(
                            nav_style, {}).items():
                        if (s, d) in ctx_transitions:
                            ctx_transitions[(s, d)] -= c
                            if ctx_transitions[(s, d)] <= 0:
                                del ctx_transitions[(s, d)]

                if ctx_transitions:
                    for poi in last_pois[-2:]:
                        src_total = sum(c for (s, d), c in ctx_transitions.items() if s == poi)
                        if src_total > 0:
                            for (src, dst), count in ctx_transitions.items():
                                if src == poi and dst not in visited_set:
                                    ctx_prob = count / src_total
                                    if dst in candidates:
                                        candidates[dst] = 0.6 * candidates[dst] + 0.4 * ctx_prob
                                    else:
                                        candidates[dst] = 0.4 * ctx_prob
                                        candidate_sources[dst] = 'context'

        # Layer 3: Covisit smoothing (leave-one-out adjusted)
        if self.covisit_graph and len(candidates) < top_k * 2:
            last_poi = current_sequence[-1]
            if self.covisit_graph.has_node(last_poi):
                neighbors = list(self.covisit_graph.neighbors(last_poi))
                adj_neighbor_weights = [
                    (n, self._covisit_weight(last_poi, n, exclude_traj))
                    for n in neighbors
                ]
                max_w = max((aw for _, aw in adj_neighbor_weights), default=1)
                for neighbor, edge_w in adj_neighbor_weights:
                    if neighbor not in visited_set and neighbor not in candidates and edge_w > 0:
                        smoothed = 0.05 * (edge_w / max_w) if max_w > 0 else 0.01
                        candidates[neighbor] = smoothed
                        candidate_sources[neighbor] = 'covisit'

        # Layer 4: Category fallback
        if len(candidates) < top_k:
            last_cat = self.poi_metadata.get(current_sequence[-1], {}).get('category')
            if last_cat and last_cat in self.category_transitions:
                cat_trans = self.category_transitions[last_cat]
                for next_cat, cat_prob in sorted(cat_trans.items(),
                                                 key=lambda x: x[1], reverse=True)[:3]:
                    for poi, meta in self.poi_metadata.items():
                        if (meta.get('category') == next_cat
                                and poi not in visited_set
                                and poi not in candidates):
                            candidates[poi] = 0.01 * cat_prob
                            candidate_sources[poi] = 'category'

        # Sort and return
        sorted_cands = sorted(candidates.items(), key=lambda x: (-x[1], x[0]))
        total_prob = sum(s for _, s in sorted_cands[:top_k])

        results = []
        for poi, score in sorted_cands[:top_k]:
            norm_prob = score / total_prob if total_prob > 0 else 0.0
            results.append({
                'poi': poi,
                'label': self.poi_metadata.get(poi, {}).get('label', poi.split('/')[-1]),
                'category': self.poi_metadata.get(poi, {}).get('category', 'Unknown'),
                'probability': norm_prob,
                'raw_score': score,
                'source': candidate_sources.get(poi, 'unknown'),
            })

        return results

    # ------------------------------------------------------------------
    # Trajectory Splitting
    # ------------------------------------------------------------------

    def split_trajectory(self, traj_uri: str, ratio: float = 0.7
                         ) -> Tuple[List[str], List[str]]:
        seq = self.sequences.get(traj_uri, [])
        if len(seq) < 3:
            return seq, []
        split_idx = max(2, int(len(seq) * ratio))
        return seq[:split_idx], seq[split_idx:]

    def leave_k_out(self, traj_uri: str, k: int = 5
                    ) -> Tuple[List[str], List[str]]:
        seq = self.sequences.get(traj_uri, [])
        if len(seq) <= k:
            return seq[:1], seq[1:]
        return seq[:-k], seq[-k:]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_recommendations(self, method: str = 'context_aware',
                                 k_values: List[int] = None,
                                 protocol: str = 'trajectory_split',
                                 split_ratio: float = 0.7,
                                 leave_k: int = 5,
                                 signal_weights: Dict[str, float] = None,
                                 min_seq_length: int = 5
                                 ) -> Dict[str, Any]:
        """
        Evaluate recommendations using specified protocol.

        Methods: context_aware, hybrid_merge, popularity, random,
                 sparql_rules, ontological, ontological_reranked
        Protocols: trajectory_split, leave_k_out
        Metrics: recall@k, precision@k, hit_rate@k, coverage@k, MRR, MAP (with 95% CI)
        """
        if k_values is None:
            k_values = [5, 10, 20]

        results = {f'recall@{k}': [] for k in k_values}
        results.update({f'precision@{k}': [] for k in k_values})
        results.update({f'hit_rate@{k}': [] for k in k_values})
        results['mrr'] = []
        results['map'] = []
        results['nav_style_breakdown'] = defaultdict(lambda: {f'recall@{k}': [] for k in k_values})
        all_recommended = {k: set() for k in k_values}

        eligible_trajs = [t for t in self.sequences if len(self.sequences[t]) >= min_seq_length]
        n_total = len(eligible_trajs)
        t_start = time.time()

        for idx, traj_uri in enumerate(eligible_trajs):
            if idx > 0 and idx % 10 == 0:
                elapsed = time.time() - t_start
                rate = elapsed / idx
                eta = rate * (n_total - idx)
                print(f"    {method}/{protocol}: {idx}/{n_total} "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

            if protocol == 'trajectory_split':
                input_seq, holdout = self.split_trajectory(traj_uri, split_ratio)
            elif protocol == 'leave_k_out':
                input_seq, holdout = self.leave_k_out(traj_uri, leave_k)
            else:
                input_seq = self.sequences[traj_uri]
                holdout = self._get_similar_visitor_pois(traj_uri)

            if not holdout:
                continue

            holdout_set = set(holdout)

            if method == 'context_aware':
                recs = self.context_aware_recommend(
                    traj_uri, input_seq, top_k=max(k_values),
                    signal_weights=signal_weights,
                    exclude_traj=traj_uri
                )
            elif method == 'hybrid_merge':
                recs = self.hybrid_merge_recommend(
                    traj_uri, input_seq, top_k=max(k_values),
                    exclude_traj=traj_uri
                )
            elif method == 'popularity':
                recs = self._popularity_baseline(set(input_seq), max(k_values))
            elif method == 'random':
                recs = self._random_baseline(set(input_seq), max(k_values))
            elif method == 'sparql_rules':
                recs = self._sparql_baseline(traj_uri, max(k_values))
            elif method == 'ontological':
                recs = self._ontological_baseline(traj_uri, max(k_values))
            elif method == 'ontological_reranked':
                recs = self._ontological_baseline(traj_uri, max(k_values), vector_rerank=True)
            else:
                recs = self.context_aware_recommend(
                    traj_uri, input_seq, top_k=max(k_values),
                    signal_weights=signal_weights
                )

            rec_pois = [r['poi'] for r in recs]

            for k in k_values:
                top_k_set = set(rec_pois[:k])
                hits = len(holdout_set & top_k_set)
                recall = hits / len(holdout_set)
                precision = hits / k
                results[f'recall@{k}'].append(recall)
                results[f'precision@{k}'].append(precision)
                results[f'hit_rate@{k}'].append(1.0 if hits > 0 else 0.0)
                all_recommended[k].update(top_k_set)

                nav = self._get_nav_label(traj_uri)
                results['nav_style_breakdown'][nav][f'recall@{k}'].append(recall)

            # MRR
            rr = 0.0
            for rank, poi in enumerate(rec_pois, 1):
                if poi in holdout_set:
                    rr = 1.0 / rank
                    break
            results['mrr'].append(rr)

            # MAP
            n_relevant = 0
            precision_sum = 0.0
            for rank, poi in enumerate(rec_pois, 1):
                if poi in holdout_set:
                    n_relevant += 1
                    precision_sum += n_relevant / rank
            ap = precision_sum / len(holdout_set) if holdout_set else 0
            results['map'].append(ap)

        # Aggregate
        summary = {}
        for k in k_values:
            for metric in ['recall', 'precision', 'hit_rate']:
                key = f'{metric}@{k}'
                vals = results[key]
                summary[key] = np.mean(vals) if vals else 0.0
                summary[f'{key}_ci'] = self.bootstrap_ci(vals) if vals else (0.0, 0.0)
            summary[f'coverage@{k}'] = (len(all_recommended[k]) / len(self.all_poi_uris)
                                        if self.all_poi_uris else 0)
        summary['mrr'] = np.mean(results['mrr']) if results['mrr'] else 0.0
        summary['mrr_ci'] = self.bootstrap_ci(results['mrr']) if results['mrr'] else (0.0, 0.0)
        summary['map'] = np.mean(results['map']) if results['map'] else 0.0
        summary['map_ci'] = self.bootstrap_ci(results['map']) if results['map'] else (0.0, 0.0)
        summary['n_evaluated'] = len(results['mrr'])

        # Nav style breakdown
        nav_summary = {}
        for nav, nav_results in results['nav_style_breakdown'].items():
            nav_summary[nav] = {}
            for k in k_values:
                key = f'recall@{k}'
                vals = nav_results[key]
                nav_summary[nav][key] = np.mean(vals) if vals else 0.0
                nav_summary[nav]['n'] = len(vals)
        summary['nav_style_breakdown'] = nav_summary

        return summary

    def evaluate_recommendations_rank_based(self, k_values: List[int] = None,
                                            signal_weights: Dict[str, float] = None
                                            ) -> Dict[str, Any]:
        """Rank-based evaluation: NDCG@k and AUC against similar visitors' POIs."""
        if k_values is None:
            k_values = [5, 10, 20]

        ndcg_scores = []
        auc_scores = []
        eligible_trajs = [t for t in self.sequences if len(self.sequences[t]) >= 5]

        for traj_uri in eligible_trajs:
            ranked = self.rank_unvisited(traj_uri, exclude_traj=traj_uri)
            if not ranked:
                continue

            relevant = self._get_similar_visitor_pois(traj_uri)
            if not relevant:
                continue
            relevant_set = set(relevant)
            ranked_pois = [r['poi'] for r in ranked]

            for k in k_values:
                ndcg = self._compute_ndcg(ranked_pois[:k], relevant_set, k)
                ndcg_scores.append({'k': k, 'traj': traj_uri, 'ndcg': ndcg})

            auc = self._compute_auc(ranked_pois, relevant_set)
            auc_scores.append(auc)

        summary = {}
        for k in k_values:
            vals = [s['ndcg'] for s in ndcg_scores if s['k'] == k]
            summary[f'ndcg@{k}'] = np.mean(vals) if vals else 0.0
            summary[f'ndcg@{k}_ci'] = self.bootstrap_ci(vals) if vals else (0.0, 0.0)

        summary['auc'] = np.mean(auc_scores) if auc_scores else 0.0
        summary['auc_ci'] = self.bootstrap_ci(auc_scores) if auc_scores else (0.0, 0.0)
        summary['n_evaluated'] = len(auc_scores)

        return summary

    def evaluate_next_prediction(self, protocol: str = 'trajectory_split',
                                 split_ratio: float = 0.7,
                                 leave_k: int = 5,
                                 k_values: List[int] = None
                                 ) -> Dict[str, Any]:
        """Evaluate next-POI prediction. Returns Hit@K, MRR with CI."""
        if k_values is None:
            k_values = [1, 5, 10]

        results = {f'hit@{k}': [] for k in k_values}
        results['mrr'] = []
        results['nav_breakdown'] = defaultdict(lambda: {f'hit@{k}': [] for k in k_values})

        eligible_trajs = [t for t in self.sequences if len(self.sequences[t]) >= 5]

        for traj_uri in eligible_trajs:
            seq = self.sequences[traj_uri]
            meta = self.traj_metadata.get(traj_uri, {})
            nav_styles = meta.get('nav_styles', set())
            context = {}
            if nav_styles:
                context['nav_style'] = list(nav_styles)[0]

            if protocol == 'trajectory_split':
                split_idx = max(2, int(len(seq) * split_ratio))
                input_seq = seq[:split_idx]
                holdout_seq = seq[split_idx:]
            else:
                input_seq = seq[:-leave_k]
                holdout_seq = seq[-leave_k:]

            if not holdout_seq:
                continue

            current = list(input_seq)
            for actual_next in holdout_seq:
                preds = self.predict_next_poi(current, context=context, top_k=max(k_values),
                                              exclude_traj=traj_uri)
                pred_pois = [p['poi'] for p in preds]

                for k in k_values:
                    hit = 1.0 if actual_next in pred_pois[:k] else 0.0
                    results[f'hit@{k}'].append(hit)

                    nav = self._get_nav_label(traj_uri)
                    results['nav_breakdown'][nav][f'hit@{k}'].append(hit)

                rr = 0.0
                for rank, poi in enumerate(pred_pois, 1):
                    if poi == actual_next:
                        rr = 1.0 / rank
                        break
                results['mrr'].append(rr)

                current.append(actual_next)

        summary = {}
        for k in k_values:
            key = f'hit@{k}'
            vals = results[key]
            summary[key] = np.mean(vals) if vals else 0.0
            summary[f'{key}_ci'] = self.bootstrap_ci(vals) if vals else (0.0, 0.0)
        summary['mrr'] = np.mean(results['mrr']) if results['mrr'] else 0.0
        summary['mrr_ci'] = self.bootstrap_ci(results['mrr']) if results['mrr'] else (0.0, 0.0)
        summary['n_steps'] = len(results['mrr'])

        nav_summary = {}
        for nav, nav_results in results['nav_breakdown'].items():
            nav_summary[nav] = {}
            for k in k_values:
                key = f'hit@{k}'
                vals = nav_results[key]
                nav_summary[nav][key] = np.mean(vals) if vals else 0.0
                nav_summary[nav]['n'] = len(vals)
        summary['nav_breakdown'] = nav_summary

        return summary

    def sweep_split_ratios(self, ratios: List[float] = None,
                           configs: Dict[str, Dict[str, float]] = None,
                           k_values: List[int] = None) -> Dict[float, Dict[str, Dict]]:
        """Sweep split ratios across configs. Returns {ratio: {config: metrics}}."""
        if ratios is None:
            ratios = [0.5, 0.6, 0.7, 0.8, 0.9]
        if k_values is None:
            k_values = [5, 10, 20]
        if configs is None:
            configs = {
                'context_aware': {'method': 'context_aware'},
                'hybrid_merge': {'method': 'hybrid_merge'},
                'popularity': {'method': 'popularity'},
            }

        results = {}
        for ratio in ratios:
            print(f"  Split ratio {ratio:.2f}...")
            ratio_results = {}
            for config_name, config in configs.items():
                ratio_results[config_name] = self.evaluate_recommendations(
                    method=config.get('method', 'context_aware'),
                    k_values=k_values,
                    protocol='trajectory_split',
                    split_ratio=ratio,
                    signal_weights=config.get('signal_weights'),
                )
            results[ratio] = ratio_results

        return results

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------

    def _popularity_baseline(self, visited: set, top_k: int) -> List[Dict]:
        # Use precomputed popularity counts from _build_derived()
        sorted_pois = sorted(self._poi_popularity.items(),
                             key=lambda x: -x[1])

        ranked = []
        for poi, count in sorted_pois:
            if poi not in visited:
                cat = self.poi_metadata.get(poi, {}).get('category', '')
                if cat not in EXCLUDED_CATEGORIES:
                    ranked.append({
                        'poi': poi,
                        'label': self.poi_metadata.get(poi, {}).get('label', ''),
                        'category': cat,
                        'score': count / max(len(self.traj_poi_sets), 1),
                        'signals': {'popularity': count},
                    })
                    if len(ranked) >= top_k:
                        break
        return ranked

    def _random_baseline(self, visited: set, top_k: int) -> List[Dict]:
        unvisited = [p for p in self.all_poi_uris - visited
                     if self.poi_metadata.get(p, {}).get('category', '') not in EXCLUDED_CATEGORIES]
        np.random.shuffle(unvisited)
        return [{'poi': poi,
                 'label': self.poi_metadata.get(poi, {}).get('label', ''),
                 'category': self.poi_metadata.get(poi, {}).get('category', ''),
                 'score': 0.0, 'signals': {}}
                for poi in unvisited[:top_k]]

    def _sparql_baseline(self, traj_uri: str, top_k: int) -> List[Dict]:
        """Recommendations from SPARQL SR-30..34 rules."""
        try:
            from .semantic_reasoning import SemanticReasoningEngine
            engine = SemanticReasoningEngine(self.endpoint, self.update_endpoint)
            recs = engine.recommend(traj_uri, top_k=top_k)
            return [{'poi': r['poi'],
                     'label': r.get('poiLabel', ''),
                     'category': r['categories'][0] if r.get('categories') else '',
                     'score': r['score'],
                     'signals': {'sparql_rules': r.get('ruleCoverage', 0)}}
                    for r in recs]
        except Exception as e:
            print(f"  sparql_rules baseline failed: {e}")
            return []

    def _ontological_baseline(self, traj_uri: str, top_k: int,
                              vector_rerank: bool = False) -> List[Dict]:
        """Recommendations from OWL profile classification similarity."""
        try:
            from .ontological_recommender import OntologicalRecommender
            if not hasattr(self, '_onto_rec'):
                self._onto_rec = OntologicalRecommender()
                self._onto_rec.load_classifications()
                # Load semantic features for vector reranking if available
                features_path = Path('output/semantic_features/trajectory_features.json')
                if features_path.exists():
                    import json
                    with open(features_path) as f:
                        raw = json.load(f)
                    self._onto_rec.set_semantic_features(
                        {uri: np.array(vec) for uri, vec in raw.items()}
                    )

            # Get similar trajectories, then extract POIs
            similar = self._onto_rec.find_similar(
                traj_uri, top_k=20, vector_rerank=vector_rerank
            )
            if not similar:
                return []

            # Collect POIs from similar trajectories
            visited = self.traj_poi_sets.get(traj_uri, set())
            poi_scores = defaultdict(float)
            poi_meta = {}
            sim_scores = {s['uri']: s.get('combined_score', s['score']) for s in similar}

            for s in similar:
                other_pois = self.traj_poi_sets.get(s['uri'], set())
                for poi in other_pois:
                    if poi not in visited:
                        cat = self.poi_metadata.get(poi, {}).get('category', '')
                        if cat not in EXCLUDED_CATEGORIES:
                            poi_scores[poi] += sim_scores.get(s['uri'], 0)
                            poi_meta[poi] = {
                                'label': self.poi_metadata.get(poi, {}).get('label', ''),
                                'category': cat,
                            }

            ranked = sorted(poi_scores.items(), key=lambda x: (-x[1], x[0]))
            return [{'poi': poi,
                     'label': poi_meta.get(poi, {}).get('label', ''),
                     'category': poi_meta.get(poi, {}).get('category', ''),
                     'score': score / len(similar),
                     'signals': {'ontological': score}}
                    for poi, score in ranked[:top_k]]
        except Exception as e:
            print(f"  ontological baseline failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_similar_visitor_pois(self, traj_uri: str) -> List[str]:
        """Get POIs visited by similar visitors (for rank-based eval)."""
        if traj_uri not in self.features:
            return []

        similar = self._find_similar_63dim(traj_uri, top_k=5)
        visited = self.traj_poi_sets.get(traj_uri, set())
        relevant = set()
        for sim_traj, _ in similar:
            sim_pois = self.traj_poi_sets.get(sim_traj, set())
            relevant.update(sim_pois - visited)
        return list(relevant)

    def _get_nav_label(self, traj_uri: str) -> str:
        meta = self.traj_metadata.get(traj_uri, {})
        styles = meta.get('nav_styles', set())
        if not styles:
            return 'NoStyle'
        return list(styles)[0]

    def _format_evidence(self, signal: str, evidence: dict, score: float) -> str:
        if signal == 'semantic_sim':
            contribs = evidence.get('contributors', [])
            if contribs:
                parts = [f"{c['traj'].split('/')[-1]} (sim={c['similarity']:.2f})"
                         for c in contribs[:3]]
                return f"Similar to trajectories {', '.join(parts)} whose visitors went here"
            return f"Aggregated similarity score {score:.3f}"

        elif signal == 'nav_match':
            style = evidence.get('nav_style', '?')
            count = evidence.get('match_count', 0)
            return f"Visited by {count} trajectories with {style} navigation style"

        elif signal == 'significant':
            total = evidence.get('total_significant', 0)
            return f"Globally significant POI (one of {total} in dataset)"

        elif signal == 'dominant_ctx':
            dom_label = evidence.get('dominant_label', '?')
            dom_dwell = evidence.get('dominant_dwell', 0)
            covisit_w = evidence.get('covisit_weight', 0)
            return (f"Co-visited with '{dom_label}' (dominant POI, "
                    f"dwell {dom_dwell:.0f}s) by {covisit_w} other trajectories")

        elif signal == 'user_pref':
            cat = evidence.get('category')
            cat_score = evidence.get('cat_score', 0)
            if cat and cat_score > 0:
                return f"Matches preference for {cat} (score {cat_score:.2f})"
            return f"Preference alignment score {score:.3f}"

        elif signal == 'engagement':
            level = evidence.get('engagement_level', '?')
            count = evidence.get('match_count', 0)
            return (f"Visited by {count} other "
                    f"{level.replace('Engagement', '-engagement').lower()} trajectories")

        return f"Score: {score:.3f}"

    def _generate_explanation(self, rec_dict: Dict, category_group: str,
                              visited_cat_names: set, popularity: int) -> Dict[str, Any]:
        """Generate an explanation with primary_reason + supporting_reasons, as the map visualizer reads it."""
        signals = rec_dict.get('signals', {})
        category = rec_dict.get('category', 'Unknown')
        label = rec_dict.get('label', 'this place')

        is_new_category = category not in visited_cat_names
        supporting_reasons = []

        # Primary reason: pick strongest signal
        top_signal = max(signals.items(), key=lambda x: x[1], default=('', 0))

        if top_signal[0] == 'semantic_sim' and top_signal[1] > 0.5:
            primary_reason = f'Travelers with similar behavior patterns visited {label}'
        elif top_signal[0] == 'dominant_ctx' and top_signal[1] > 0.3:
            primary_reason = 'Frequently visited alongside your most important stop'
        elif top_signal[0] == 'nav_match' and top_signal[1] > 0.3:
            primary_reason = 'Popular with visitors who share your navigation style'
        elif top_signal[0] == 'user_pref':
            primary_reason = f'Matches your preference for {category_group} places'
        elif top_signal[0] == 'significant':
            primary_reason = 'Significant destination that most visitors include'
        elif top_signal[0] == 'engagement':
            primary_reason = 'Visited by travelers with similar engagement levels'
        elif category_group != 'Other':
            primary_reason = f'Recommended {category_group.lower()} spot based on your journey'
        else:
            primary_reason = 'Matches your route patterns and travel interests'

        # Supporting reasons from other active signals
        if signals.get('dominant_ctx', 0) > 0 and 'alongside' not in primary_reason:
            supporting_reasons.append('Co-visited with places on your route')
        if signals.get('nav_match', 0) > 0 and 'navigation' not in primary_reason:
            supporting_reasons.append('Common stop for your navigation style')
        if signals.get('user_pref', 0) > 0 and 'preference' not in primary_reason.lower():
            supporting_reasons.append(f'Aligns with your {category_group} preferences')
        if signals.get('engagement', 0) > 0 and 'engagement' not in primary_reason:
            supporting_reasons.append('Visited by travelers with similar engagement')
        if signals.get('significant', 0) > 0 and 'Significant' not in primary_reason:
            supporting_reasons.append('Globally significant destination')

        # Popularity and discovery
        popularity_rank = self._get_popularity_rank(popularity)
        if popularity >= 10 and 'visitor' not in primary_reason:
            supporting_reasons.append(f'{popularity_rank} destination ({popularity} visits)')
        if is_new_category and category_group != 'Other':
            supporting_reasons.append(f'Discover {category_group}: new category for you')
        elif not is_new_category and category_group != 'Other':
            supporting_reasons.append(f'Continues your interest in {category_group}')

        supporting_reasons = supporting_reasons[:3]

        summary_parts = [primary_reason]
        if popularity >= 20:
            summary_parts.append('Popular choice')
        summary = '. '.join(summary_parts) + '.'

        return {
            'primary_reason': primary_reason,
            'supporting_reasons': supporting_reasons,
            'summary': summary,
        }

    def _get_avg_dwell(self, poi_uri: str) -> float:
        """Average dwell time across all interactions with this POI."""
        return self._avg_dwell_cache.get(poi_uri, 0.0)

    def _get_dwell_time_label(self, avg_dwell: float) -> str:
        if avg_dwell < 300:
            return 'Quick visit (~5 min)'
        elif avg_dwell < 900:
            return 'Brief stop (~10-15 min)'
        elif avg_dwell < 1800:
            return 'Medium visit (~20-30 min)'
        elif avg_dwell < 3600:
            return 'Extended visit (~45-60 min)'
        else:
            return f'Long visit (~{avg_dwell / 3600:.1f} hours)'

    def _get_popularity_rank(self, popularity: int) -> str:
        if popularity >= 50:
            return 'Very Popular'
        elif popularity >= 20:
            return 'Popular'
        elif popularity >= 5:
            return 'Moderate'
        else:
            return 'Niche'

    def _compute_ndcg(self, ranked_pois: List[str], relevant_set: set, k: int) -> float:
        dcg = 0.0
        for i, poi in enumerate(ranked_pois[:k]):
            if poi in relevant_set:
                dcg += 1.0 / np.log2(i + 2)
        n_relevant = min(len(relevant_set), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))
        return dcg / idcg if idcg > 0 else 0.0

    def _compute_auc(self, ranked_pois: List[str], relevant_set: set) -> float:
        n_relevant = 0
        n_irrelevant = 0
        score = 0.0
        for poi in ranked_pois:
            if poi in relevant_set:
                n_relevant += 1
            else:
                n_irrelevant += 1
                score += n_relevant
        total = n_relevant * n_irrelevant
        return score / total if total > 0 else 0.5

    @staticmethod
    def bootstrap_ci(values: List[float], n_resamples: int = 1000,
                     ci: float = 0.95) -> Tuple[float, float]:
        if not values or len(values) < 2:
            m = np.mean(values) if values else 0.0
            return (m, m)

        values = np.array(values)
        rng = np.random.default_rng(42)
        boot_means = []
        for _ in range(n_resamples):
            sample = rng.choice(values, size=len(values), replace=True)
            boot_means.append(np.mean(sample))

        alpha = (1 - ci) / 2
        lower = np.percentile(boot_means, alpha * 100)
        upper = np.percentile(boot_means, (1 - alpha) * 100)
        return (round(float(lower), 4), round(float(upper), 4))

    # ------------------------------------------------------------------
    # Navigation Style & Engagement Distribution (for notebooks)
    # ------------------------------------------------------------------

    def get_nav_style_distribution(self) -> Dict[str, int]:
        counts = Counter()
        for meta in self.traj_metadata.values():
            for style in meta.get('nav_styles', set()):
                counts[style] += 1
        return dict(counts)

    def get_engagement_distribution(self) -> Dict[str, int]:
        counts = Counter()
        for meta in self.traj_metadata.values():
            eng = meta.get('engagement')
            if eng:
                counts[eng] += 1
        return dict(counts)

    def get_trajectory_profile(self, traj_uri: str) -> Dict:
        """Get full profile for a trajectory (for display in notebooks)."""
        meta = self.traj_metadata.get(traj_uri, {})
        seq = self.sequences.get(traj_uri, [])
        dominant = self._get_dominant_poi(traj_uri)

        user = self.traj_to_user.get(traj_uri)
        prefs = self.user_preferences.get(user, []) if user else []

        return {
            'uri': traj_uri,
            'user': meta.get('user', ''),
            'nav_styles': list(meta.get('nav_styles', set())),
            'engagement': meta.get('engagement', ''),
            'purposes': list(meta.get('purposes', set())),
            'sequence_length': len(seq),
            'dominant_poi': dominant,
            'dominant_label': self.poi_metadata.get(dominant, {}).get('label', '') if dominant else '',
            'top_preferences': prefs[:5],
        }
