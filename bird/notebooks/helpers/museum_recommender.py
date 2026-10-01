"""
Museum Recommender — 7-signal unified recommendation with 4-layer explainability,
prediction, and evaluation engine for the BIRD Nancy Museum dataset.

Follows main pipeline's UnifiedRecommender architecture:
  semantic_sim, route_sim, flow_prob, dominant_ctx, user_pref, significant, engagement.
  Rule coverage (SR-30..34) computed in-memory for explanations, not scoring.

Capabilities:
- Unified multi-signal artwork recommendations with confidence + provenance
- Route similarity (Jaccard artwork overlap + category sequence LCS)
- In-memory SR-30..34 rule computation for explanation Layer 3
- Dwell time prediction (k-NN on visitor features)
- Next artwork prediction (4-layer flow model)
- Three evaluation protocols: trajectory-split, leave-k-out, rank-based
- Bootstrapped confidence intervals
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

try:
    from .artwork_categories import ARTWORK_CATEGORIES, resolve_category, resolve_era
    from .fuseki_config import query_endpoint, update_endpoint
    from .fuseki_queries import execute_sparql_query, format_results
except ImportError:
    from artwork_categories import ARTWORK_CATEGORIES, resolve_category, resolve_era
    from fuseki_config import query_endpoint, update_endpoint
    from fuseki_queries import execute_sparql_query, format_results

PREFIXES = """
PREFIX trajecto: <http://w3id.org/trajecto#>
PREFIX poi:      <http://w3id.org/trajecto/poi#>
PREFIX patterns: <http://w3id.org/trajecto/patterns#>
PREFIX user:     <http://w3id.org/trajecto/user#>
PREFIX purpose:  <http://w3id.org/trajecto/purpose#>
PREFIX dul:      <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#>
PREFIX datacron: <http://www.datacron-project.eu/datAcron#>
PREFIX rdf:      <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs:     <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:      <http://www.w3.org/2001/XMLSchema#>
"""

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
    'semantic_sim': 'Cosine similarity over 59-dim feature vectors (category, floor, room, era, interaction — excludes navigation & engagement)',
    'nav_match': 'patterns:exhibitsPattern',
    'significant': 'poi:SignificantPOI (rdf:type)',
    'dominant_ctx': 'poi:covisitedWith (edge weight from covisit graph)',
    'user_pref': 'user:hasInferredPreference + poi:hasRoom',
    'engagement': 'trajecto:hasEngagementLevel',
    'flow_prob': 'poi:followedBy (directed transition counts)',
}

# Pipeline step provenance for each signal
SIGNAL_PIPELINE = {
    'semantic_sim': 'BF3 (Semantic Feature Extraction)',
    'nav_match': 'B4 (Navigation Style Classification)',
    'significant': 'B3 (Dominant & Significant Artworks)',
    'dominant_ctx': 'B3 + SR-05 (Co-visitation Materialization)',
    'user_pref': 'B6 (User Profile & Preference Learning)',
    'engagement': 'B2 (Interaction Classification)',
    'flow_prob': 'B2 (Position Sequences) + SR-06 (poi:followedBy)',
}

# ── Unified signal configuration ─────────────────────────────────────────
# Follows main pipeline's UnifiedRecommender pattern.
# archetype_sim omitted (requires ontological_recommender, not yet ported).

UNIFIED_WEIGHTS = {
    'semantic_sim': 0.20,
    'route_sim': 0.26,
    'flow_prob': 0.06,
    'dominant_ctx': 0.14,
    'user_pref': 0.16,
    'significant': 0.06,
    'engagement': 0.12,
}

UNIFIED_KG_PROVENANCE = {
    'semantic_sim': '59-dim feature cosine (category + floor + room + era + interaction)',
    'route_sim': 'Jaccard artwork overlap + category sequence LCS',
    'flow_prob': 'poi:followedBy (recency-weighted transition probability)',
    'rule_coverage': 'SR-30..34 rule endorsement (in-memory)',
    'dominant_ctx': 'poi:covisitedWith (SR-05)',
    'user_pref': 'user:hasInferredPreference + poi:hasRoom (SR-20)',
    'significant': 'poi:SignificantPOI (rdf:type)',
    'engagement': 'trajecto:hasEngagementLevel (SR-10)',
}

_RULE_LABELS = {
    'SR-30': 'collaborative',
    'SR-31': 'preference gap',
    'SR-32': 'sequence continuation',
    'SR-33': 'covisitation',
    'SR-34': 'purpose alignment',
}

_RULE_EXPLANATIONS = {
    'SR-30': 'Visited by visitors with similar navigation and purpose profile',
    'SR-31': 'Matches inferred visitor preferences (unvisited category)',
    'SR-32': 'Next in common visit sequence',
    'SR-33': 'Co-visited with dominant artwork in other trajectories',
    'SR-34': 'Aligns with trajectory purpose',
}

# Purpose → artwork category mapping for SR-34 in-memory computation
_PURPOSE_CATEGORY_MAP = {
    'ClassicalArtsPurpose': {'Painting', 'Sculpture'},
    'ModernArtsPurpose': {'Painting', 'Sculpture', 'Print'},
    'ContemporaryArtsPurpose': {'Photography', 'MixedMedia', 'Installation', 'Mirror'},
    'CulturalPurpose': {'Painting', 'Sculpture', 'Print', 'Photography'},
}


class MuseumRecommender:
    """
    Consolidated recommendation/prediction engine for museum trajectories.

    Usage:
        rec = MuseumRecommender()
        rec.load_all_data()
        recs = rec.recommend(traj_uri, top_k=10)
    """

    def __init__(self, endpoint: Optional[str] = None, upd_endpoint: Optional[str] = None):
        self.endpoint = endpoint or query_endpoint
        self.update_endpoint = upd_endpoint or update_endpoint

        # Data stores (populated by load_all_data)
        self.features = {}              # traj_uri -> 72-dim np.ndarray (59 used for similarity)
        self.sequences = {}             # traj_uri -> [poi_uri, ...]
        self.traj_metadata = {}         # traj_uri -> {user, purpose, nav_styles, engagement}
        self.poi_metadata = {}          # poi_uri -> {label, category, creator, floor, room, era}
        self.interaction_data = {}      # traj_uri -> [{poi, dwell, int_type, floor}, ...]
        self.user_preferences = {}      # user_uri -> [{category, score}, ...]
        self.user_room_preferences = {} # user_uri -> {room: score}
        self.traj_to_user = {}          # traj_uri -> user_uri
        self.user_to_traj = {}          # user_uri -> traj_uri

        # Graphs
        self.flow_graph = None          # nx.DiGraph
        self.covisit_graph = None       # nx.Graph

        # Similarity engine
        self.sim_engine = None

        # Spatial coordinates
        self.position_coords = {}       # traj_uri -> [{poi, x, y, floor, timestamp}, ...]
        self.artwork_coords = {}        # poi_uri -> {x, y, floor} (median position)

        # Derived
        self.all_artwork_uris = set()
        self.traj_artwork_sets = {}     # traj_uri -> set of poi_uris
        self.category_transitions = {}  # src_cat -> {dst_cat: prob}
        self.significant_artworks = set()  # cached at load time

        # Route similarity
        self._route_sim_cache = {}    # traj_uri -> [{'uri', 'score', 'shared_poi_count', 'lcs_ratio'}]
        self._cat_sequences = {}      # traj_uri -> [category_name, ...]

        # Precomputed indexes for fast signal computation
        self._engagement_pois = {}    # engagement_level -> {poi: count}
        self._category_index = {}     # category -> set of poi URIs
        self._user_visited_cache = {} # user_uri -> set of all visited poi URIs

        # Signal score cache for ablation: {traj_uri: {art_uri: {signal: raw_score}}}
        self._signal_cache = {}

        self._loaded = False

    def _query(self, sparql: str) -> pd.DataFrame:
        result = execute_sparql_query(self.endpoint, PREFIXES + sparql)
        return format_results(result)

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------

    def load_all_data(self, features_path: Optional[str] = None) -> None:
        """
        Single load from Fuseki + cached features. Builds all internal stores.

        Args:
            features_path: Path to trajectory_features.json. If None, extracts live.
        """
        print("Loading museum data...")

        # 1. Features
        self._load_features(features_path)

        # 2. Interaction data + sequences
        self._load_interactions()

        # 3. Trajectory metadata
        self._load_trajectory_metadata()

        # 4. POI metadata
        self._load_poi_metadata()

        # 4b. Engagement levels (computed from interaction data, matches SR-10)
        eng_count = self._compute_engagement_levels()
        print(f"  Engagement: {eng_count} trajectories classified")

        # 5. User preferences (computed from interaction data, matches SR-20)
        self._load_user_preferences()

        # 6. Spatial coordinates
        self._load_coordinates()

        # 7. Graphs
        self._load_graphs()

        # 8. Derived structures
        self._build_derived()

        # 9. Cache significant artworks
        self.significant_artworks = self._get_significant_artworks()
        print(f"  Significant artworks: {len(self.significant_artworks)}")

        # 10. Route similarity precompute
        self._precompute_route_similarity()

        # 11. Indexes for fast unified scoring
        self._precompute_indexes()

        self._loaded = True
        print(f"Loaded: {len(self.features)} trajectories, "
              f"{len(self.all_artwork_uris)} artworks, "
              f"{len(self.user_preferences)} users with preferences")

    def _load_features(self, features_path: Optional[str] = None) -> None:
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

    def _load_interactions(self) -> None:
        """Load interaction data and build sequences."""
        df = self._query("""
        SELECT ?traj ?pos ?poi ?dwell ?intType ?artworkType ?floor ?timestamp
        WHERE {
            ?pos dul:isPartOf ?traj .
            ?pos trajecto:hasInteraction ?interaction .
            ?interaction poi:interactsWith ?poi .
            ?interaction poi:hasDwellTimeInSeconds ?dwell .
            ?interaction a ?intType .
            FILTER(?intType IN (poi:PassingInteraction, poi:MeaningfulInteraction,
                                poi:ImmersiveInteraction, poi:DominantInteraction,
                                poi:StartInteraction, poi:DestinationInteraction))
            OPTIONAL { ?poi poi:hasType ?artworkType . }
            OPTIONAL {
                ?pos datacron:hasGeometry ?point .
                ?point trajecto:floor ?floor .
            }
            OPTIONAL { ?pos trajecto:timestamp ?timestamp . }
        }
        ORDER BY ?traj ?timestamp
        """)

        if df.empty:
            print("  Interactions: no data found")
            return

        for col in ['dwell', 'floor']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Build interaction data and sequences per trajectory
        for traj in df['traj'].unique():
            traj_df = df[df['traj'] == traj].copy()
            if 'timestamp' in traj_df.columns:
                traj_df = traj_df.sort_values('timestamp')

            interactions = []
            sequence = []
            seen_in_seq = None

            for _, row in traj_df.iterrows():
                int_type = row['intType'].split('#')[-1] if pd.notna(row.get('intType')) else ''
                raw_type = row.get('artworkType', '')
                if pd.notna(raw_type) and raw_type:
                    raw_type = raw_type.split('#')[-1] if '#' in raw_type else raw_type
                    cat = resolve_category(raw_type)
                else:
                    cat = 'Unknown'

                interactions.append({
                    'poi': row['poi'],
                    'dwell': float(row['dwell']) if pd.notna(row.get('dwell')) else 0.0,
                    'int_type': int_type,
                    'category': cat,
                    'floor': int(row['floor']) if pd.notna(row.get('floor')) else -1,
                })
                self.all_artwork_uris.add(row['poi'])

                # Build deduplicated sequence (MeaningfulInteraction includes Immersive)
                if int_type in ('MeaningfulInteraction', 'ImmersiveInteraction',
                                'DominantInteraction') and row['poi'] != seen_in_seq:
                    sequence.append(row['poi'])
                    seen_in_seq = row['poi']

            self.interaction_data[traj] = interactions
            self.sequences[traj] = sequence
            self.traj_artwork_sets[traj] = {i['poi'] for i in interactions}

        print(f"  Interactions: {len(self.interaction_data)} trajectories, "
              f"{len(self.all_artwork_uris)} unique artworks")

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
            self.user_to_traj[user] = traj

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
        SELECT ?poi ?label ?artworkType ?creator ?room ?period
        WHERE {
            ?poi a trajecto:PointOfInterest .
            OPTIONAL { ?poi rdfs:label ?label . }
            OPTIONAL { ?poi poi:hasType ?artworkType . }
            OPTIONAL { ?poi poi:creator ?creator . }
            OPTIONAL { ?poi poi:hasRoom ?roomUri .
                       BIND(REPLACE(STR(?roomUri), "^.*#", "") AS ?room) }
            OPTIONAL { ?poi poi:creationPeriod ?period . }
        }
        """)

        if df.empty:
            return

        # Also get floor info
        floor_df = self._query("""
        SELECT ?poi (SAMPLE(?floor) AS ?poiFloor)
        WHERE {
            ?pos trajecto:hasInteraction ?int .
            ?int poi:interactsWith ?poi .
            ?pos datacron:hasGeometry ?point .
            ?point trajecto:floor ?floor .
        }
        GROUP BY ?poi
        """)
        poi_floors = {}
        if not floor_df.empty:
            for _, row in floor_df.iterrows():
                if pd.notna(row.get('poiFloor')):
                    poi_floors[row['poi']] = int(float(row['poiFloor']))

        for _, row in df.iterrows():
            poi = row['poi']
            raw_type = row.get('artworkType', '')
            if pd.notna(raw_type) and raw_type:
                raw_type = raw_type.split('#')[-1] if '#' in raw_type else raw_type
                cat = resolve_category(raw_type)
            else:
                cat = 'Unknown'

            room = row.get('room', '') if pd.notna(row.get('room')) else ''
            period = row.get('period', '') if pd.notna(row.get('period')) else ''
            era = resolve_era(period) if period else 'Unknown'

            self.poi_metadata[poi] = {
                'label': row.get('label', poi.split('/')[-1]) if pd.notna(row.get('label')) else poi.split('/')[-1],
                'category': cat,
                'creator': row.get('creator', '') if pd.notna(row.get('creator')) else '',
                'floor': poi_floors.get(poi, -1),
                'room': room,
                'era': era,
            }

        print(f"  POI metadata: {len(self.poi_metadata)} artworks")

    def _load_user_preferences(self) -> None:
        """Compute user preferences from in-memory interaction data.

        Category score = 0.4 * immersive_frac + 0.5 * meaningful_frac + 0.1 * visit_norm
        Room preferences = dwell-weighted room distribution per user (normalized).
        """
        if not self.interaction_data or not self.traj_to_user:
            print("  Preferences: none (no interaction data yet)")
            return

        # Aggregate per user per category
        user_cat_stats: dict[str, dict[str, dict]] = defaultdict(
            lambda: defaultdict(lambda: {'total': 0, 'immersive': 0, 'meaningful': 0})
        )
        # Aggregate per user per room (dwell-weighted)
        user_room_dwell: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        for traj, interactions in self.interaction_data.items():
            user = self.traj_to_user.get(traj)
            if not user:
                continue
            for item in interactions:
                cat = item.get('category', 'Unknown')
                if cat not in ('PointOfInterest', 'SignificantPOI', 'Unknown'):
                    user_cat_stats[user][cat]['total'] += 1
                    if item['int_type'] == 'ImmersiveInteraction':
                        user_cat_stats[user][cat]['immersive'] += 1
                    if item['int_type'] in ('MeaningfulInteraction', 'ImmersiveInteraction'):
                        user_cat_stats[user][cat]['meaningful'] += 1

                # Room preferences from dwell time
                poi = item.get('poi', '')
                room = self.poi_metadata.get(poi, {}).get('room', '')
                if room:
                    user_room_dwell[user][room] += item.get('dwell', 0)

        for user, cats in user_cat_stats.items():
            max_visits = max((s['total'] for s in cats.values()), default=1)
            prefs = []
            for cat, stats in cats.items():
                total = stats['total']
                if total == 0:
                    continue
                imm_frac = stats['immersive'] / total
                mean_frac = stats['meaningful'] / total
                visit_norm = total / max_visits if max_visits > 0 else 0
                score = 0.4 * imm_frac + 0.5 * mean_frac + 0.1 * visit_norm
                prefs.append({'category': cat, 'score': score})
            prefs.sort(key=lambda x: -x['score'])
            self.user_preferences[user] = prefs

        # Normalize room preferences per user
        for user, rooms in user_room_dwell.items():
            total_dwell = sum(rooms.values())
            if total_dwell > 0:
                self.user_room_preferences[user] = {
                    room: dwell / total_dwell for room, dwell in rooms.items()
                }

        print(f"  Preferences: {len(self.user_preferences)} users "
              f"({len(self.user_room_preferences)} with room prefs)")

    def _load_coordinates(self) -> None:
        """Load spatial coordinates (easting/northing/floor) for trajectory positions."""
        df = self._query("""
        SELECT ?traj ?poi ?x ?y ?floor ?timestamp
        WHERE {
            ?pos dul:isPartOf ?traj .
            ?pos trajecto:hasInteraction ?interaction .
            ?interaction poi:interactsWith ?poi .
            ?interaction a poi:MeaningfulInteraction .
            ?pos datacron:hasGeometry ?point .
            ?point trajecto:hasEasting ?x .
            ?point trajecto:hasNorthing ?y .
            ?point trajecto:floor ?floor .
            OPTIONAL { ?pos trajecto:timestamp ?timestamp . }
        }
        ORDER BY ?traj ?timestamp
        """)

        if df.empty:
            print("  Coordinates: no spatial data found")
            return

        for col in ['x', 'y', 'floor']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Per-trajectory coordinate sequences
        poi_coords_agg = defaultdict(lambda: {'xs': [], 'ys': [], 'floors': []})

        for traj in df['traj'].unique():
            traj_df = df[df['traj'] == traj]
            if 'timestamp' in traj_df.columns:
                traj_df = traj_df.sort_values('timestamp')

            coords = []
            for _, row in traj_df.iterrows():
                if pd.notna(row['x']) and pd.notna(row['y']):
                    coords.append({
                        'poi': row['poi'],
                        'x': float(row['x']),
                        'y': float(row['y']),
                        'floor': int(row['floor']) if pd.notna(row['floor']) else 0,
                    })
                    poi_coords_agg[row['poi']]['xs'].append(float(row['x']))
                    poi_coords_agg[row['poi']]['ys'].append(float(row['y']))
                    poi_coords_agg[row['poi']]['floors'].append(
                        int(row['floor']) if pd.notna(row['floor']) else 0)
            self.position_coords[traj] = coords

        # Artwork median coordinates
        for poi, data in poi_coords_agg.items():
            self.artwork_coords[poi] = {
                'x': float(np.median(data['xs'])),
                'y': float(np.median(data['ys'])),
                'floor': int(np.median(data['floors'])),
            }

        print(f"  Coordinates: {len(self.position_coords)} trajectories, "
              f"{len(self.artwork_coords)} artwork positions")

    def _load_graphs(self) -> None:
        """Build flow and covisit graphs from in-memory data (no SPARQL)."""
        import networkx as nx

        # Flow graph: consecutive POI transitions from sequences
        self.flow_graph = nx.DiGraph()
        transition_counts: dict[tuple, int] = {}
        for _, seq in self.sequences.items():
            for i in range(len(seq) - 1):
                if seq[i] != seq[i + 1]:
                    edge = (seq[i], seq[i + 1])
                    transition_counts[edge] = transition_counts.get(edge, 0) + 1
        for (p1, p2), count in transition_counts.items():
            self.flow_graph.add_edge(p1, p2, weight=count, transitions=count)

        # Covisit graph: POIs visited in same trajectory (meaningful+ only)
        self.covisit_graph = nx.Graph()
        covisit_counts: dict[tuple, int] = {}
        for _, interactions in self.interaction_data.items():
            meaningful_pois = set()
            for item in interactions:
                if item['int_type'] in ('MeaningfulInteraction',
                                        'ImmersiveInteraction',
                                        'DominantInteraction'):
                    meaningful_pois.add(item['poi'])
            pois = sorted(meaningful_pois)
            for i in range(len(pois)):
                for j in range(i + 1, len(pois)):
                    edge = (pois[i], pois[j])
                    covisit_counts[edge] = covisit_counts.get(edge, 0) + 1
        for (p1, p2), count in covisit_counts.items():
            self.covisit_graph.add_edge(p1, p2, weight=count, covisits=count)

        print(f"  Graphs: flow={self.flow_graph.number_of_edges()} edges, "
              f"covisit={self.covisit_graph.number_of_edges()} edges")

    def _build_derived(self) -> None:
        """Build similarity engine and category transitions."""
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
        for _, seq in self.sequences.items():
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

    def _compute_engagement_levels(self) -> int:
        """Compute engagement levels from in-memory interaction data (matches SR-10 logic).

        High: >= 5 immersive interactions (>90s dwell)
        Medium: >= 3 immersive AND < 5
        Low: remainder with any interactions
        """
        updated = 0
        for traj, interactions in self.interaction_data.items():
            if traj not in self.traj_metadata:
                continue
            immersive_count = sum(
                1 for i in interactions if i['int_type'] == 'ImmersiveInteraction')

            if immersive_count >= 5:
                level = 'HighEngagement'
            elif immersive_count >= 3:
                level = 'MediumEngagement'
            else:
                level = 'LowEngagement'

            self.traj_metadata[traj]['engagement'] = level
            updated += 1
        return updated

    # ------------------------------------------------------------------
    # Route Similarity (unified signal)
    # ------------------------------------------------------------------

    def _precompute_route_similarity(self):
        """Precompute top-15 route-similar trajectories per trajectory.

        Route similarity = 0.5 * Jaccard artwork overlap + 0.5 * normalized
        category sequence LCS.
        """
        print("  Precomputing route similarity matrix...")

        self._cat_sequences = {}
        for traj, seq in self.sequences.items():
            cat_seq = []
            for poi in seq:
                cat = self.poi_metadata.get(poi, {}).get('category', '')
                if cat and cat != 'Unknown':
                    cat_seq.append(cat)
            if cat_seq:
                self._cat_sequences[traj] = cat_seq

        eligible = [t for t in self.traj_artwork_sets
                    if t in self._cat_sequences
                    and len(self.traj_artwork_sets[t]) >= 2]

        self._route_sim_cache = {}
        for i, traj_a in enumerate(eligible):
            pois_a = self.traj_artwork_sets[traj_a]
            cats_a = self._cat_sequences[traj_a]
            candidates = []

            for j, traj_b in enumerate(eligible):
                if i == j:
                    continue
                pois_b = self.traj_artwork_sets[traj_b]

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

            candidates.sort(key=lambda x: -x['score'])
            self._route_sim_cache[traj_a] = candidates[:15]

        n = len(eligible)
        pairs = sum(len(v) for v in self._route_sim_cache.values())
        print(f"  Route similarity: {n} trajectories, {pairs} pairs cached")

    def _find_route_similar(self, traj_uri: str,
                            top_k: int = 15) -> list[dict]:
        """Return precomputed route-similar trajectories."""
        return self._route_sim_cache.get(traj_uri, [])[:top_k]

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

    # ------------------------------------------------------------------
    # Precomputed Indexes
    # ------------------------------------------------------------------

    def _precompute_indexes(self):
        """Build indexes for fast signal computation during unified scoring."""
        # Engagement level → POI frequency across matching trajectories
        self._engagement_pois = defaultdict(lambda: defaultdict(int))
        for traj, meta in self.traj_metadata.items():
            eng = meta.get('engagement')
            if eng:
                for poi in self.traj_artwork_sets.get(traj, set()):
                    self._engagement_pois[eng][poi] += 1

        # Category → set of POI URIs (for SR-31, SR-34)
        self._category_index = defaultdict(set)
        for poi, pmeta in self.poi_metadata.items():
            cat = pmeta.get('category', '')
            if cat:
                self._category_index[cat].add(poi)

        # User → all visited POIs (for rule exclusion)
        self._user_visited_cache = {}
        for user in set(self.traj_to_user.values()):
            visited = set()
            for traj, u in self.traj_to_user.items():
                if u == user:
                    visited |= self.traj_artwork_sets.get(traj, set())
            self._user_visited_cache[user] = visited

        print(f"  Indexes: {len(self._engagement_pois)} engagement levels, "
              f"{len(self._category_index)} category groups, "
              f"{len(self._user_visited_cache)} user visit sets")

    # ------------------------------------------------------------------
    # In-Memory Rule Computation (SR-30..34)
    # ------------------------------------------------------------------

    def _compute_rules_in_memory(self, traj_uri: str) -> dict[str, set]:
        """Compute SR-30..34 rule results from in-memory data.

        Equivalent to SPARQL execution but avoids 5 Fuseki round-trips.
        """
        meta = self.traj_metadata.get(traj_uri, {})
        purposes = meta.get('purposes', set())
        nav_styles = meta.get('nav_styles', set())
        user = self.traj_to_user.get(traj_uri)

        user_visited = self._user_visited_cache.get(user, set()) if user else set()

        # SR-30: Profile collaborative — same purpose + shared navigation pattern
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
                sr30 |= self.traj_artwork_sets.get(other_uri, set())
            sr30 -= user_visited

        # SR-31: Preference gap — artworks in categories user has affinity for
        sr31 = set()
        if user:
            prefs = self.user_preferences.get(user, [])
            pref_cats = {p['category'] for p in prefs if p.get('score', 0) > 0}
            for cat in pref_cats:
                sr31 |= self._category_index.get(cat, set())
            sr31 -= user_visited

        # SR-32: Sequence continuation — follow flow graph from visited artworks
        sr32 = set()
        visited_pois = self.traj_artwork_sets.get(traj_uri, set())
        if self.flow_graph:
            for poi in visited_pois:
                if self.flow_graph.has_node(poi):
                    for neighbor in self.flow_graph.successors(poi):
                        if neighbor not in user_visited:
                            sr32.add(neighbor)

        # SR-33: Covisitation — co-visited with dominant artwork
        sr33 = set()
        dominant = self._get_dominant_artwork(traj_uri)
        if dominant and self.covisit_graph and self.covisit_graph.has_node(dominant):
            for neighbor in self.covisit_graph.neighbors(dominant):
                if neighbor not in user_visited:
                    sr33.add(neighbor)

        # SR-34: Purpose alignment — artworks whose category matches purpose
        sr34 = set()
        if purposes:
            aligned_cats = set()
            for p in purposes:
                aligned_cats |= _PURPOSE_CATEGORY_MAP.get(p, set())
            for cat in aligned_cats:
                sr34 |= self._category_index.get(cat, set())
            sr34 -= user_visited

        return {
            'SR-30': sr30, 'SR-31': sr31, 'SR-32': sr32,
            'SR-33': sr33, 'SR-34': sr34,
        }

    # ------------------------------------------------------------------
    # Trajectory Splitting
    # ------------------------------------------------------------------

    def split_trajectory(self, traj_uri: str, ratio: float = 0.7
                         ) -> tuple[list[str], list[str]]:
        """Split trajectory sequence into input (first ratio) and holdout (rest)."""
        seq = self.sequences.get(traj_uri, [])
        if len(seq) < 3:
            return seq, []
        split_idx = max(2, int(len(seq) * ratio))
        return seq[:split_idx], seq[split_idx:]

    def leave_k_out(self, traj_uri: str, k: int = 5
                    ) -> tuple[list[str], list[str]]:
        """Hide last k interactions, return input sequence + holdout set."""
        seq = self.sequences.get(traj_uri, [])
        if len(seq) <= k:
            return seq[:1], seq[1:]
        return seq[:-k], seq[-k:]

    # ------------------------------------------------------------------
    # Context-Aware Recommendation
    # ------------------------------------------------------------------

    def recommend(self, traj_uri: str,
                  observed_sequence: Optional[list[str]] = None,
                  top_k: int = 20,
                  min_score: float = 0.05,
                  signal_weights: Optional[dict[str, float]] = None,
                  ) -> list[dict]:
        """
        7-signal artwork recommendation with 4-layer explainability.

        Signals (UNIFIED_WEIGHTS):
          semantic_sim, route_sim, flow_prob, dominant_ctx,
          user_pref, significant, engagement.

        Rule coverage (SR-30..34) is computed for Layer 3 explanations
        but excluded from scoring weights.

        Returns list of dicts with:
            poi, label, category, floor, score, confidence,
            signals, route_match, rules, provenance, explanation
        """
        weights = signal_weights or UNIFIED_WEIGHTS

        visited = set(observed_sequence or self.sequences.get(traj_uri, []))
        meta = self.traj_metadata.get(traj_uri, {})
        input_seq = list(observed_sequence or self.sequences.get(traj_uri, []))
        nav_styles = meta.get('nav_styles', set())
        nav_style = next(iter(nav_styles)) if nav_styles else None

        # Forward-reachable candidates (directed flow edges only)
        forward_set = self._get_forward_candidates(input_seq, visited, nav_style)
        candidates = forward_set if forward_set else (self.all_artwork_uris - visited)

        scores = defaultdict(lambda: defaultdict(float))
        evidence = defaultdict(lambda: defaultdict(dict))

        # ── Signal 1: Semantic similarity ──────────────────────────────
        if traj_uri in self.features:
            similar = self.sim_engine.find_similar_semantic(
                traj_uri, top_k=10, metric='cosine'
            )
            for sim_traj, sim_score in similar:
                sim_artworks = self.traj_artwork_sets.get(sim_traj, set())
                for art in (sim_artworks & candidates):
                    scores[art]['semantic_sim'] += sim_score
                    contribs = evidence[art]['semantic_sim'].setdefault(
                        'contributors', [])
                    contribs.append({
                        'traj': sim_traj, 'similarity': sim_score})

        # ── Signal 2: Route similarity (Jaccard + LCS) ─────────────────
        route_similar = self._find_route_similar(traj_uri, top_k=15)
        route_info = {}

        for sim in route_similar:
            sim_artworks = self.traj_artwork_sets.get(sim['uri'], set())
            for art in (sim_artworks & candidates):
                scores[art]['route_sim'] += sim['score']
                existing = evidence[art]['route_sim']
                existing.setdefault('sources', []).append({
                    'traj': sim['uri'],
                    'score': sim['score'],
                    'shared_poi_count': sim['shared_poi_count'],
                })

        for art in scores:
            ev = evidence[art].get('route_sim', {})
            sources = ev.get('sources', [])
            if sources:
                best = max(sources, key=lambda s: s['shared_poi_count'])
                route_info[art] = {
                    'source_count': len(sources),
                    'max_shared_pois': best['shared_poi_count'],
                    'top_score': best['score'],
                }

        # ── Signal 3: Rule coverage (SR-30..34, explanation only) ──────
        rule_results = {}
        per_rule = self._compute_rules_in_memory(traj_uri)
        for rule_name, poi_set in per_rule.items():
            rule_results[rule_name] = poi_set
            for art in (poi_set & candidates):
                scores[art]['rule_coverage'] += 1.0
                evidence[art]['rule_coverage'].setdefault(
                    'rules', []).append(rule_name)

        # ── Signal 4: Flow probability (recency-weighted) ──────────────
        if self.flow_graph and input_seq:
            last_pois = input_seq[-3:] if len(input_seq) >= 3 else input_seq
            recency_weights = [0.5, 0.3, 0.2]

            flow_dists = {}
            for poi in last_pois:
                if self.flow_graph.has_node(poi):
                    adj_weights = {}
                    for _, dst in self.flow_graph.out_edges(poi):
                        fw = self.flow_graph[poi][dst].get('weight', 0)
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

        # ── Signal 5: Dominant context ─────────────────────────────────
        dominant = self._get_dominant_artwork(traj_uri)
        if dominant and self.covisit_graph and self.covisit_graph.has_node(dominant):
            dominant_label = self.poi_metadata.get(
                dominant, {}).get('label', '?')
            for neighbor in self.covisit_graph.neighbors(dominant):
                if neighbor in candidates:
                    edge_w = self.covisit_graph[dominant][neighbor].get(
                        'weight', 1)
                    scores[neighbor]['dominant_ctx'] += edge_w
                    evidence[neighbor]['dominant_ctx'] = {
                        'dominant_label': dominant_label,
                        'covisit_weight': edge_w,
                    }

        # ── Signal 6: User preference (category + room) ────────────────
        user = self.traj_to_user.get(traj_uri)
        prefs = self.user_preferences.get(user, []) if user else []
        pref_map = {p['category']: p['score'] for p in prefs}
        room_prefs = self.user_room_preferences.get(user, {}) if user else {}
        for art in candidates:
            art_meta = self.poi_metadata.get(art, {})
            cat = art_meta.get('category', '')
            room = art_meta.get('room', '')
            cat_score = pref_map.get(cat, 0)
            room_score = room_prefs.get(room, 0)
            if cat_score > 0 or room_score > 0:
                scores[art]['user_pref'] += 0.4 * cat_score + 0.6 * room_score
                evidence[art]['user_pref'] = {
                    'category': cat, 'cat_score': cat_score,
                    'room': room, 'room_score': room_score,
                }

        # ── Signal 7: Significant artworks ─────────────────────────────
        sig_artworks = self.significant_artworks or self._get_significant_artworks()
        for art in (sig_artworks & candidates):
            scores[art]['significant'] += 1.0
            evidence[art]['significant'] = {
                'total_significant': len(sig_artworks),
            }

        # ── Signal 8: Engagement level (precomputed index) ─────────────
        engagement = meta.get('engagement')
        if engagement and engagement in self._engagement_pois:
            eng_pois = self._engagement_pois[engagement]
            own_pois = self.traj_artwork_sets.get(traj_uri, set())
            for art in candidates:
                count = eng_pois.get(art, 0)
                if art in own_pois and count > 0:
                    count -= 1
                if count > 0:
                    scores[art]['engagement'] += float(count)
                    evidence[art]['engagement'] = {
                        'engagement_level': engagement,
                        'match_count': count,
                    }

        # ── Normalize + Aggregate ──────────────────────────────────────
        for signal in weights:
            signal_vals = [scores[art][signal]
                           for art in scores if scores[art][signal] > 0]
            if signal_vals:
                max_val = max(signal_vals)
                if max_val > 0:
                    for art in scores:
                        if scores[art][signal] > 0:
                            scores[art][signal] /= max_val

        max_score = 0.0
        ranked = []
        for art, signals in scores.items():
            total = sum(weights.get(sig, 0) * val
                        for sig, val in signals.items())
            if total > max_score:
                max_score = total
            ranked.append((art, signals, total))

        ranked.sort(key=lambda x: -x[2])
        ranked = [(a, s, t) for a, s, t in ranked if t >= min_score][:top_k]

        # ── Build Output with 4-Layer Explanations ─────────────────────
        results = []
        for art, signals, total in ranked:
            art_meta = self.poi_metadata.get(art, {})
            category = art_meta.get('category', 'Unknown')

            # Confidence: score strength + evidence breadth
            active_signals = sum(1 for v in signals.values() if v > 0)
            norm_score = total / max_score if max_score > 0 else 0
            confidence_val = (0.5 * norm_score
                              + 0.5 * min(active_signals / 7, 1.0))
            if confidence_val >= 0.6:
                confidence = 'strong'
            elif confidence_val >= 0.35:
                confidence = 'moderate'
            else:
                confidence = 'weak'

            # Layer 1: Signal decomposition
            signal_dict = dict(signals)

            # Layer 2: Route match
            route_match = route_info.get(art)

            # Layer 3: Rule endorsement
            rule_ev = evidence[art].get('rule_coverage', {})
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

            # Layer 4: KG provenance
            provenance = []
            for sig, val in sorted(signals.items(),
                                   key=lambda x: -x[1]):
                w = weights.get(sig, 0)
                contribution = w * val
                if contribution <= 0:
                    continue
                ev = evidence.get(art, {}).get(sig, {})
                prov_entry = {
                    'signal': sig,
                    'contribution': round(contribution, 4),
                    'predicate': UNIFIED_KG_PROVENANCE.get(sig, ''),
                    'raw_score': round(val, 4),
                }
                if sig == 'semantic_sim':
                    contribs = ev.get('contributors', [])
                    if contribs:
                        prov_entry['top_cosine'] = round(
                            contribs[0]['similarity'], 3)
                        prov_entry['n_contributors'] = len(contribs)
                elif sig == 'route_sim':
                    sources = ev.get('sources', [])
                    if sources:
                        best = max(sources,
                                   key=lambda s: s['shared_poi_count'])
                        prov_entry['max_shared_pois'] = best[
                            'shared_poi_count']
                        prov_entry['n_sources'] = len(sources)
                elif sig == 'rule_coverage':
                    prov_entry['rules'] = ev.get('rules', [])
                elif sig == 'dominant_ctx':
                    prov_entry['dominant_artwork'] = ev.get(
                        'dominant_label', '?')
                    prov_entry['covisit_weight'] = ev.get(
                        'covisit_weight', 0)
                elif sig == 'flow_prob':
                    prov_entry['source_pois'] = ev.get('source_pois', [])
                elif sig == 'user_pref':
                    prov_entry['category'] = ev.get('category', '')
                    prov_entry['pref_score'] = ev.get('cat_score', 0)
                    prov_entry['room'] = ev.get('room', '')
                    prov_entry['room_score'] = ev.get('room_score', 0)
                provenance.append(prov_entry)

            # Explanation summary
            explanation = self._build_unified_explanation(
                art, category, signals, weights,
                route_match, rules_layer, evidence.get(art, {}),
            )

            results.append({
                'poi': art,
                'label': art_meta.get('label', art.split('/')[-1]),
                'category': category,
                'floor': art_meta.get('floor', -1),
                'score': round(total, 4),
                'confidence': confidence,
                'signals': signal_dict,
                'route_match': route_match,
                'rules': rules_layer,
                'provenance': provenance,
                'explanation': explanation,
            })

        return results

    def context_aware_recommend(self, traj_uri: str,
                                observed_sequence: Optional[list[str]] = None,
                                top_k: int = 20,
                                signal_weights: Optional[dict[str, float]] = None,
                                explain: bool = False,
                                ) -> list[dict]:
        """Backward-compatible wrapper around recommend().

        Maps old 6-signal weights to UNIFIED_WEIGHTS if custom weights are
        passed; otherwise uses UNIFIED_WEIGHTS directly.
        """
        return self.recommend(
            traj_uri, observed_sequence=observed_sequence,
            top_k=top_k, signal_weights=signal_weights,
        )

    def rank_unvisited(self, traj_uri: str,
                       observed_sequence: Optional[list[str]] = None) -> list[dict]:
        """Rank all unvisited artworks by score (for rank-based evaluation)."""
        return self.recommend(
            traj_uri, observed_sequence, top_k=len(self.all_artwork_uris),
            min_score=0.0
        )

    def hybrid_merge_recommend(self, traj_uri: str,
                               observed_sequence: Optional[list[str]] = None,
                               top_k: int = 20,
                               ctx_weight: float = 0.5,
                               flow_weight: float = 0.5,
                               explain: bool = False
                               ) -> list[dict]:
        """Merge context-aware and flow prediction ranked lists (Strategy B).

        Normalizes both to [0,1], combines with given weights, returns top_k.
        When explain=True, carries forward explanations from context-aware component.
        """
        input_seq = observed_sequence or self.sequences.get(traj_uri, [])

        # Unified top candidates
        ctx_recs = self.recommend(
            traj_uri, input_seq, top_k=top_k * 2,
        )

        # Flow predictions
        context = {}
        nav_styles = self.traj_metadata.get(traj_uri, {}).get('nav_styles', set())
        if nav_styles:
            context['nav_style'] = next(iter(nav_styles))
        flow_recs = self.predict_next_artwork(input_seq, context=context, top_k=top_k * 2)

        # Normalize both to [0, 1]
        ctx_scores = {r['poi']: r['score'] for r in ctx_recs}
        flow_scores = {r['poi']: r['raw_score'] for r in flow_recs}
        ctx_max = max(ctx_scores.values()) if ctx_scores else 1.0
        flow_max = max(flow_scores.values()) if flow_scores else 1.0

        # Lookup for explanations from context-aware component
        ctx_explanations = {r['poi']: r.get('explanations', []) for r in ctx_recs} if explain else {}

        # Union with weighted combination
        all_pois = set(ctx_scores) | set(flow_scores)
        merged = {}
        for poi in all_pois:
            c = ctx_scores.get(poi, 0) / ctx_max if ctx_max > 0 else 0
            f = flow_scores.get(poi, 0) / flow_max if flow_max > 0 else 0
            merged[poi] = ctx_weight * c + flow_weight * f

        ranked = sorted(merged.items(), key=lambda x: -x[1])[:top_k]
        results = []
        for p, s in ranked:
            rec_dict = {
                'poi': p,
                'label': self.poi_metadata.get(p, {}).get('label', p.split('/')[-1]),
                'category': self.poi_metadata.get(p, {}).get('category', 'Unknown'),
                'floor': self.poi_metadata.get(p, {}).get('floor', -1),
                'score': s,
                'source': 'hybrid_merge',
            }
            if explain:
                rec_dict['explanations'] = ctx_explanations.get(p, [])
            results.append(rec_dict)
        return results

    def _get_significant_artworks(self) -> set:
        """Get artworks marked as significant POIs."""
        df = self._query("""
        SELECT DISTINCT ?poi WHERE {
            ?poi a poi:SignificantPOI .
        }
        """)
        if df.empty:
            return set()
        return set(df['poi'].tolist())

    def _get_dominant_artwork(self, traj_uri: str) -> Optional[str]:
        """Get the dominant artwork for a trajectory."""
        interactions = self.interaction_data.get(traj_uri, [])
        for item in interactions:
            if item['int_type'] == 'DominantInteraction':
                return item['poi']
        return None

    def _build_unified_explanation(
        self, art, category, signals, weights,
        route_match, rules_layer, art_evidence: dict = None,
    ) -> dict[str, Any]:
        """Build 4-layer explanation from unified signal data."""
        art_evidence = art_evidence or {}

        # Dominant context label
        dom_ev = art_evidence.get('dominant_ctx', {})
        dom_label = dom_ev.get('dominant_label', '')
        dom_weight = dom_ev.get('covisit_weight', 0)
        dom_str = (f'Co-visited with {dom_label} by {int(dom_weight)} visitors'
                   if dom_label and dom_weight
                   else 'Co-visited with the dominant artwork of this trajectory')

        # Primary reason from strongest signal
        weighted = {sig: weights.get(sig, 0) * val
                    for sig, val in signals.items()}
        top_sig = max(weighted, key=weighted.get, default='')
        top_val = weighted.get(top_sig, 0)

        if top_sig == 'semantic_sim' and top_val > 0.05:
            primary = (f'Behaviorally similar visitors '
                       f'visited this {category.lower()}')
        elif top_sig == 'route_sim' and route_match:
            primary = (f"{route_match['source_count']} visitors sharing "
                       f"{route_match['max_shared_pois']} artworks visited this "
                       f"{category.lower()}")
        elif top_sig == 'flow_prob':
            primary = 'Natural next step based on transition patterns from recent artworks'
        elif top_sig == 'rule_coverage' and rules_layer:
            labels = ', '.join(
                _RULE_LABELS.get(r, r) for r in rules_layer['matched'][:3])
            primary = (f"{rules_layer['count']}/5 SR rules endorse: "
                       f"{labels}")
        elif top_sig == 'dominant_ctx':
            primary = dom_str
        elif top_sig == 'user_pref':
            primary = f'Aligns with inferred {category} preference'
        elif top_sig == 'significant':
            primary = 'Significant artwork included by most visitors'
        elif top_sig == 'engagement':
            primary = 'Frequented by visitors with matching engagement level'
        else:
            primary = 'Converging evidence from multiple recommendation signals'

        # Supporting reasons from other active signals
        supporting = []
        if route_match and top_sig != 'route_sim':
            supporting.append(
                f"{route_match['source_count']} shared-route visitors "
                f"({route_match['max_shared_pois']} common artworks)")
        if signals.get('flow_prob', 0) > 0 and top_sig != 'flow_prob':
            supporting.append(
                f'Flow probability {signals["flow_prob"]:.0%} from recent artworks')
        if rules_layer and top_sig != 'rule_coverage':
            labels = ', '.join(
                _RULE_LABELS.get(r, r) for r in rules_layer['matched'][:3])
            supporting.append(
                f"{rules_layer['count']}/5 SR rules: {labels}")
        if signals.get('dominant_ctx', 0) > 0 and top_sig != 'dominant_ctx':
            supporting.append(dom_str)
        if signals.get('user_pref', 0) > 0 and top_sig != 'user_pref':
            supporting.append(f'Aligns with inferred {category} preference')

        active = sum(1 for sig, v in signals.items()
                     if v > 0 and sig in weights)
        summary = (f"{category} recommended via {active}/7 signals"
                   + (f", {rules_layer['count']}/5 rules"
                      if rules_layer else ""))

        return {
            'primary_reason': primary,
            'supporting_reasons': supporting,
            'summary': summary,
        }

    def format_evidence(self, signal: str, evidence: dict,
                        score: float) -> str:
        """Human-readable evidence string for a signal."""
        if signal == 'semantic_sim':
            contribs = evidence.get('contributors', [])
            if contribs:
                parts = [f"{c['traj'].split('/')[-1]} (sim={c['similarity']:.2f})"
                         for c in contribs[:3]]
                return f"Similar to visitors {', '.join(parts)} who visited this"
            return f"Aggregated similarity score {score:.3f}"

        if signal == 'route_sim':
            sources = evidence.get('sources', [])
            if sources:
                best = max(sources, key=lambda s: s['shared_poi_count'])
                return (f"Route match: {len(sources)} visitors "
                        f"sharing {best['shared_poi_count']} artworks")
            return f"Route similarity score {score:.3f}"

        if signal == 'flow_prob':
            source_pois = evidence.get('source_pois', [])
            return (f"Flow probability {score:.1%} from "
                    f"{len(source_pois)} recent artworks")

        if signal == 'rule_coverage':
            rules = evidence.get('rules', [])
            labels = ', '.join(_RULE_LABELS.get(r, r) for r in rules)
            return f"{len(rules)}/5 rules endorse: {labels}"

        if signal == 'significant':
            total = evidence.get('total_significant', 0)
            return f"Globally significant artwork (one of {total} in museum)"

        if signal == 'dominant_ctx':
            dom_label = evidence.get('dominant_label', '?')
            covisit_w = evidence.get('covisit_weight', 0)
            return (f"Co-visited with '{dom_label}' (dominant artwork) "
                    f"by {covisit_w} other visitors")

        if signal == 'user_pref':
            parts = []
            cat = evidence.get('category')
            cat_score = evidence.get('cat_score', 0)
            room = evidence.get('room')
            room_score = evidence.get('room_score', 0)
            if cat and cat_score > 0:
                parts.append(f"Matches preference for {cat} (score {cat_score:.2f})")
            if room and room_score > 0:
                parts.append(f"In preferred room {room} (score {room_score:.2f})")
            return '; '.join(parts) if parts else f"Preference alignment score {score:.3f}"

        if signal == 'engagement':
            level = evidence.get('engagement_level', '?')
            count = evidence.get('match_count', 0)
            return f"Visited by {count} other {level.replace('Engagement', '-engagement').lower()} visitors"

        return f"Score: {score:.3f}"

    # ------------------------------------------------------------------
    # Dwell Time Prediction
    # ------------------------------------------------------------------

    def predict_dwell(self, traj_uri: str, artwork_uri: str, k: int = 5
                      ) -> dict:
        """
        k-NN regression on 72-dim visitor features to predict dwell time.

        Finds k most similar visitors who interacted with the artwork,
        returns weighted average of their dwell times.
        """
        if traj_uri not in self.features:
            return {'predicted_dwell': 0.0, 'neighbors': [], 'method': 'no_features'}

        query_vec = self.features[traj_uri]
        candidates = []

        for other_traj, other_feats in self.features.items():
            if other_traj == traj_uri:
                continue
            # Check if other trajectory visited this artwork
            other_interactions = self.interaction_data.get(other_traj, [])
            dwells = [i['dwell'] for i in other_interactions if i['poi'] == artwork_uri and i['dwell'] > 0]
            if not dwells:
                continue

            sim = 1 - cosine(query_vec, other_feats) if np.linalg.norm(query_vec) > 0 and np.linalg.norm(other_feats) > 0 else 0.0
            avg_dwell = np.mean(dwells)
            candidates.append((other_traj, sim, avg_dwell))

        if not candidates:
            # Fallback: global average dwell for this artwork
            all_dwells = []
            for interactions in self.interaction_data.values():
                for i in interactions:
                    if i['poi'] == artwork_uri and i['dwell'] > 0:
                        all_dwells.append(i['dwell'])
            global_avg = np.mean(all_dwells) if all_dwells else 30.0
            return {'predicted_dwell': global_avg, 'neighbors': [], 'method': 'global_avg'}

        candidates.sort(key=lambda x: -x[1])
        top_k = candidates[:k]

        total_sim = sum(s for _, s, _ in top_k)
        if total_sim > 0:
            predicted = sum(s * d / total_sim for _, s, d in top_k)
        else:
            predicted = np.mean([d for _, _, d in top_k])

        neighbors = [{'traj': t, 'sim': round(s, 3), 'dwell': round(d, 1)}
                     for t, s, d in top_k]

        return {'predicted_dwell': round(predicted, 1), 'neighbors': neighbors, 'method': 'knn'}

    def predict_dwell_combined(self, traj_uri: str, artwork_uri: str, k: int = 5
                               ) -> dict:
        """
        k-NN regression on 41-dim joint space (28 visitor + 13 artwork features).
        Artwork features: category 8-dim, floor 3-dim, significance, popularity.
        """
        if traj_uri not in self.features:
            return self.predict_dwell(traj_uri, artwork_uri, k)

        art_vec = self._build_artwork_feature_vector(artwork_uri)
        query_joint = np.concatenate([self.features[traj_uri], art_vec])
        candidates = []

        for other_traj, other_feats in self.features.items():
            if other_traj == traj_uri:
                continue
            other_interactions = self.interaction_data.get(other_traj, [])
            for item in other_interactions:
                if item['poi'] == artwork_uri and item['dwell'] > 0:
                    other_art_vec = art_vec  # Same artwork
                    other_joint = np.concatenate([other_feats, other_art_vec])
                    norm_q = np.linalg.norm(query_joint)
                    norm_o = np.linalg.norm(other_joint)
                    sim = 1 - cosine(query_joint, other_joint) if norm_q > 0 and norm_o > 0 else 0.0
                    candidates.append((other_traj, sim, item['dwell']))
                    break

        if not candidates:
            return self.predict_dwell(traj_uri, artwork_uri, k)

        candidates.sort(key=lambda x: -x[1])
        top_k = candidates[:k]
        total_sim = sum(s for _, s, _ in top_k)
        if total_sim > 0:
            predicted = sum(s * d / total_sim for _, s, d in top_k)
        else:
            predicted = np.mean([d for _, _, d in top_k])

        return {'predicted_dwell': round(predicted, 1),
                'neighbors': [{'traj': t, 'sim': round(s, 3), 'dwell': round(d, 1)}
                              for t, s, d in top_k],
                'method': 'combined_knn'}

    def _build_artwork_feature_vector(self, artwork_uri: str) -> np.ndarray:
        """Build 13-dim artwork feature vector: category(8) + floor(3) + significance(1) + popularity(1)."""
        meta = self.poi_metadata.get(artwork_uri, {})

        # Category one-hot (8 dims)
        cat_vec = np.zeros(len(ARTWORK_CATEGORIES), dtype=np.float32)
        cat = meta.get('category', '')
        if cat in ARTWORK_CATEGORIES:
            cat_vec[ARTWORK_CATEGORIES.index(cat)] = 1.0

        # Floor one-hot (3 dims)
        floor_vec = np.zeros(3, dtype=np.float32)
        floor = meta.get('floor', -1)
        if 0 <= floor <= 2:
            floor_vec[floor] = 1.0

        # Significance (1 dim)
        sig_artworks = self._get_significant_artworks()
        significance = 1.0 if artwork_uri in sig_artworks else 0.0

        # Popularity: fraction of trajectories that visited this artwork (1 dim)
        visit_count = sum(1 for arts in self.traj_artwork_sets.values() if artwork_uri in arts)
        popularity = visit_count / max(len(self.traj_artwork_sets), 1)

        return np.concatenate([cat_vec, floor_vec, [significance], [popularity]])

    # ------------------------------------------------------------------
    # Forward-Reachable Candidate Set (directional gate)
    # ------------------------------------------------------------------

    def _get_forward_candidates(self, sequence: list[str],
                                visited: set[str],
                                nav_style: Optional[str] = None) -> set[str]:
        """Return artworks reachable via directed flow edges (layers 1+2 only).

        Uses flow graph outgoing edges from last 3 artworks + nav-style-
        conditioned transitions.  Excludes covisit smoothing and category
        fallback which are undirected and destroy directionality.
        """
        forward = set()
        if not sequence:
            return forward

        last_pois = sequence[-3:] if len(sequence) >= 3 else list(sequence)

        # Layer 1: Direct flow graph outgoing edges
        if self.flow_graph:
            for poi in last_pois:
                if self.flow_graph.has_node(poi):
                    for _, dst in self.flow_graph.out_edges(poi):
                        if dst not in visited:
                            forward.add(dst)

        # Layer 2: Nav-style-conditioned transitions
        if nav_style:
            for other_traj, other_meta in self.traj_metadata.items():
                if nav_style in other_meta.get('nav_styles', set()):
                    seq = self.sequences.get(other_traj, [])
                    for j in range(len(seq) - 1):
                        if seq[j] in set(last_pois[-2:]) and seq[j + 1] not in visited:
                            forward.add(seq[j + 1])

        return forward

    # ------------------------------------------------------------------
    # Next Artwork Prediction (4-layer)
    # ------------------------------------------------------------------

    def predict_next_artwork(self, current_sequence: list[str],
                             context: Optional[dict] = None,
                             top_k: int = 10) -> list[dict]:
        """
        4-layer next-artwork prediction adapted from Rome's trajectory_predictor.

        Layers:
        1. Base flow: Global transitions from last 3 artworks, recency-weighted
        2. Context blend: Nav-style-conditioned flow + floor affinity
        3. Covisit smoothing: Co-visitation neighbors for unobserved transitions
        4. Category fallback: Category-level transition as weak prior

        Floor penalty: 0.7x for different-floor candidates.
        """
        if not current_sequence:
            return []

        candidates = defaultdict(float)
        candidate_sources = defaultdict(str)
        visited_set = set(current_sequence)

        # Current floor (from last artwork)
        current_floor = self.poi_metadata.get(current_sequence[-1], {}).get('floor', -1)

        # Layer 1: Base flow transitions
        last_pois = current_sequence[-3:] if len(current_sequence) >= 3 else current_sequence
        recency_weights = [0.5, 0.3, 0.2]

        for i, poi in enumerate(reversed(last_pois)):
            w = recency_weights[i] if i < len(recency_weights) else 0.1
            if self.flow_graph and self.flow_graph.has_node(poi):
                out_edges = list(self.flow_graph.out_edges(poi, data=True))
                total_w = sum(d.get('weight', 1) for _, _, d in out_edges)
                if total_w > 0:
                    for _, dst, data in out_edges:
                        if dst not in visited_set:
                            prob = w * (data.get('weight', 1) / total_w)
                            candidates[dst] += prob
                            if dst not in candidate_sources:
                                candidate_sources[dst] = 'flow'

        # Layer 2: Context blend (nav-style-conditioned)
        if context:
            nav_style = context.get('nav_style')
            if nav_style:
                # Find trajectories with same nav style, build context flow
                ctx_transitions = defaultdict(int)
                for other_traj, other_meta in self.traj_metadata.items():
                    if nav_style in other_meta.get('nav_styles', set()):
                        seq = self.sequences.get(other_traj, [])
                        for j in range(len(seq) - 1):
                            if seq[j] != seq[j + 1]:
                                ctx_transitions[(seq[j], seq[j + 1])] += 1

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

        # Layer 3: Covisit smoothing
        if self.covisit_graph and len(candidates) < top_k * 2:
            last_poi = current_sequence[-1]
            if self.covisit_graph.has_node(last_poi):
                max_w = max((d.get('weight', 1) for _, _, d
                             in self.covisit_graph.edges(last_poi, data=True)),
                            default=1)
                for neighbor in self.covisit_graph.neighbors(last_poi):
                    if neighbor not in visited_set and neighbor not in candidates:
                        edge_w = self.covisit_graph[last_poi][neighbor].get('weight', 1)
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

        # Floor penalty: 0.7x for different floor
        if current_floor >= 0:
            for poi in list(candidates.keys()):
                poi_floor = self.poi_metadata.get(poi, {}).get('floor', -1)
                if poi_floor >= 0 and poi_floor != current_floor:
                    candidates[poi] *= 0.7

        # Sort and return
        sorted_cands = sorted(candidates.items(), key=lambda x: -x[1])
        total_prob = sum(s for _, s in sorted_cands[:top_k])

        results = []
        for poi, score in sorted_cands[:top_k]:
            norm_prob = score / total_prob if total_prob > 0 else 0.0
            results.append({
                'poi': poi,
                'label': self.poi_metadata.get(poi, {}).get('label', poi.split('/')[-1]),
                'category': self.poi_metadata.get(poi, {}).get('category', 'Unknown'),
                'floor': self.poi_metadata.get(poi, {}).get('floor', -1),
                'probability': norm_prob,
                'raw_score': score,
                'source': candidate_sources.get(poi, 'unknown'),
            })

        return results

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_recommendations(self, method: str = 'context_aware',
                                 k_values: Optional[list[int]] = None,
                                 protocol: str = 'trajectory_split',
                                 split_ratio: float = 0.7,
                                 leave_k: int = 5,
                                 signal_weights: Optional[dict[str, float]] = None
                                 ) -> dict[str, Any]:
        """
        Evaluate recommendations using specified protocol.

        Protocols:
        - trajectory_split: First split_ratio as input, rest as holdout
        - leave_k_out: Hide last k interactions
        - rank_based: Rank all unvisited, check if similar visitors' artworks rank higher

        Returns dict with metrics: recall@k, mrr, ndcg, per-nav-style breakdown.
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

        eligible_trajs = [t for t in self.sequences if len(self.sequences[t]) >= 5]

        for traj_uri in eligible_trajs:
            # Get holdout
            if protocol == 'trajectory_split':
                input_seq, holdout = self.split_trajectory(traj_uri, split_ratio)
            elif protocol == 'leave_k_out':
                input_seq, holdout = self.leave_k_out(traj_uri, leave_k)
            else:
                # rank_based: use full sequence, holdout = artworks visited by similar visitors
                input_seq = self.sequences[traj_uri]
                holdout = self._get_similar_visitor_artworks(traj_uri)

            if not holdout:
                continue

            holdout_set = set(holdout)

            # Get recommendations
            if method in ('context_aware', 'unified'):
                recs = self.recommend(
                    traj_uri, input_seq, top_k=max(k_values),
                    signal_weights=signal_weights,
                )
            elif method == 'popularity':
                recs = self._popularity_baseline(set(input_seq), max(k_values))
            elif method == 'random':
                recs = self._random_baseline(set(input_seq), max(k_values))
            elif method == 'same_floor':
                recs = self._same_floor_baseline(traj_uri, set(input_seq), max(k_values))
            elif method == 'sparql_rules':
                recs = self._sparql_baseline(traj_uri, max(k_values))
            elif method == 'hybrid_merge':
                recs = self.hybrid_merge_recommend(traj_uri, input_seq, top_k=max(k_values))
            else:
                recs = self.recommend(
                    traj_uri, input_seq, top_k=max(k_values),
                    signal_weights=signal_weights,
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

                # Per-nav-style
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
            summary[f'coverage@{k}'] = len(all_recommended[k]) / len(self.all_artwork_uris) if self.all_artwork_uris else 0
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

    def evaluate_recommendations_rank_based(self, k_values: Optional[list[int]] = None,
                                            signal_weights: Optional[dict[str, float]] = None
                                            ) -> dict[str, Any]:
        """
        Rank-based evaluation: rank all unvisited artworks, measure quality
        via NDCG and AUC against artworks visited by similar visitors.
        """
        if k_values is None:
            k_values = [5, 10, 20]

        ndcg_scores = []
        auc_scores = []
        eligible_trajs = [t for t in self.sequences if len(self.sequences[t]) >= 5]

        for traj_uri in eligible_trajs:
            ranked = self.rank_unvisited(traj_uri)
            if not ranked:
                continue

            # Relevance: artworks visited by similar visitors (not in this trajectory)
            relevant = self._get_similar_visitor_artworks(traj_uri)
            if not relevant:
                continue
            relevant_set = set(relevant)

            ranked_pois = [r['poi'] for r in ranked]

            # NDCG
            for k in k_values:
                ndcg = self._compute_ndcg(ranked_pois[:k], relevant_set, k)
                ndcg_scores.append({'k': k, 'traj': traj_uri, 'ndcg': ndcg})

            # AUC
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

    def evaluate_dwell_prediction(self, k_range: Optional[list[int]] = None,
                                  protocol: str = 'trajectory_split',
                                  split_ratio: float = 0.7,
                                  leave_k: int = 5
                                  ) -> dict[str, Any]:
        """
        Evaluate dwell time prediction.

        Returns MAE, RMSE, Spearman correlation for k-NN, per-artwork-mean,
        and global-mean baselines.
        """
        if k_range is None:
            k_range = [3, 5, 7, 10, 15]

        results_by_k = {}
        baseline_results = {'per_artwork': [], 'global_mean': [], 'per_visitor': []}

        # Global average dwell
        all_dwells = []
        for interactions in self.interaction_data.values():
            for item in interactions:
                if item['dwell'] > 0:
                    all_dwells.append(item['dwell'])
        global_mean_dwell = np.mean(all_dwells) if all_dwells else 30.0

        # Per-artwork average dwell
        artwork_avg_dwell = defaultdict(list)
        for interactions in self.interaction_data.values():
            for item in interactions:
                if item['dwell'] > 0:
                    artwork_avg_dwell[item['poi']].append(item['dwell'])
        artwork_mean = {poi: np.mean(dwells) for poi, dwells in artwork_avg_dwell.items()}

        eligible_trajs = [t for t in self.interaction_data if len(self.interaction_data[t]) >= 5]

        for k in k_range:
            results_by_k[k] = []

        for traj_uri in eligible_trajs:
            interactions = self.interaction_data[traj_uri]
            meaningful = [i for i in interactions
                          if i['int_type'] in ('MeaningfulInteraction', 'ImmersiveInteraction',
                                               'DominantInteraction')
                          and i['dwell'] > 0]

            if len(meaningful) < 5:
                continue

            if protocol == 'trajectory_split':
                split_idx = max(2, int(len(meaningful) * split_ratio))
                holdout = meaningful[split_idx:]
            else:
                holdout = meaningful[-leave_k:]

            for item in holdout:
                actual = item['dwell']

                # k-NN predictions
                for k in k_range:
                    pred = self.predict_dwell(traj_uri, item['poi'], k=k)
                    results_by_k[k].append((actual, pred['predicted_dwell']))

                # Baselines
                baseline_results['per_artwork'].append(
                    (actual, artwork_mean.get(item['poi'], global_mean_dwell))
                )
                baseline_results['global_mean'].append((actual, global_mean_dwell))

                # Per-visitor mean
                visitor_dwells = [i['dwell'] for i in meaningful if i['dwell'] > 0]
                visitor_mean = np.mean(visitor_dwells) if visitor_dwells else global_mean_dwell
                baseline_results['per_visitor'].append((actual, visitor_mean))

        summary = {}

        for k in k_range:
            pairs = results_by_k[k]
            if pairs:
                summary[f'knn_k{k}'] = self._dwell_metrics(pairs)
            else:
                summary[f'knn_k{k}'] = {'mae': 0, 'rmse': 0, 'spearman': 0, 'n': 0}

        for baseline_name, pairs in baseline_results.items():
            if pairs:
                summary[baseline_name] = self._dwell_metrics(pairs)

        return summary

    def evaluate_next_prediction(self, protocol: str = 'trajectory_split',
                                 split_ratio: float = 0.7,
                                 leave_k: int = 5,
                                 k_values: Optional[list[int]] = None
                                 ) -> dict[str, Any]:
        """
        Evaluate next-artwork prediction.

        Returns Hit@K, MRR, per-nav-style breakdown.
        """
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
                context['nav_style'] = next(iter(nav_styles))

            if protocol == 'trajectory_split':
                split_idx = max(2, int(len(seq) * split_ratio))
                input_seq = seq[:split_idx]
                holdout_seq = seq[split_idx:]
            else:
                input_seq = seq[:-leave_k]
                holdout_seq = seq[-leave_k:]

            if not holdout_seq:
                continue

            # Predict each step
            current = list(input_seq)
            for actual_next in holdout_seq:
                preds = self.predict_next_artwork(current, context=context, top_k=max(k_values))
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

        # Nav breakdown
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

    def evaluate_next_learning_curve(self, positions: Optional[list[int]] = None
                                     ) -> dict[int, dict[str, float]]:
        """
        Accuracy vs trajectory position: after seeing N artworks, how well
        can we predict the next one?
        """
        if positions is None:
            positions = [3, 5, 10, 20]

        results = {pos: {'hit@1': [], 'hit@5': [], 'mrr': []} for pos in positions}

        for traj_uri, seq in self.sequences.items():
            meta = self.traj_metadata.get(traj_uri, {})
            context = {}
            nav_styles = meta.get('nav_styles', set())
            if nav_styles:
                context['nav_style'] = next(iter(nav_styles))

            for pos in positions:
                if len(seq) <= pos:
                    continue
                input_seq = seq[:pos]
                actual_next = seq[pos]

                preds = self.predict_next_artwork(input_seq, context=context, top_k=10)
                pred_pois = [p['poi'] for p in preds]

                results[pos]['hit@1'].append(1.0 if actual_next in pred_pois[:1] else 0.0)
                results[pos]['hit@5'].append(1.0 if actual_next in pred_pois[:5] else 0.0)

                rr = 0.0
                for rank, poi in enumerate(pred_pois, 1):
                    if poi == actual_next:
                        rr = 1.0 / rank
                        break
                results[pos]['mrr'].append(rr)

        summary = {}
        for pos in positions:
            summary[pos] = {
                'hit@1': np.mean(results[pos]['hit@1']) if results[pos]['hit@1'] else 0.0,
                'hit@5': np.mean(results[pos]['hit@5']) if results[pos]['hit@5'] else 0.0,
                'mrr': np.mean(results[pos]['mrr']) if results[pos]['mrr'] else 0.0,
                'n': len(results[pos]['mrr']),
            }
        return summary

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------

    def _popularity_baseline(self, visited: set, top_k: int) -> list[dict]:
        """Most-visited unvisited artworks."""
        art_counts = Counter()
        for arts in self.traj_artwork_sets.values():
            for a in arts:
                art_counts[a] += 1

        ranked = []
        for poi, count in art_counts.most_common():
            if poi not in visited:
                ranked.append({
                    'poi': poi,
                    'label': self.poi_metadata.get(poi, {}).get('label', ''),
                    'category': self.poi_metadata.get(poi, {}).get('category', ''),
                    'floor': self.poi_metadata.get(poi, {}).get('floor', -1),
                    'score': count / max(len(self.traj_artwork_sets), 1),
                    'signals': {'popularity': count},
                })
                if len(ranked) >= top_k:
                    break
        return ranked

    def _random_baseline(self, visited: set, top_k: int) -> list[dict]:
        """Random unvisited artworks."""
        unvisited = list(self.all_artwork_uris - visited)
        np.random.shuffle(unvisited)
        return [{'poi': poi,
                 'label': self.poi_metadata.get(poi, {}).get('label', ''),
                 'category': self.poi_metadata.get(poi, {}).get('category', ''),
                 'floor': self.poi_metadata.get(poi, {}).get('floor', -1),
                 'score': 0.0, 'signals': {}}
                for poi in unvisited[:top_k]]

    def _same_floor_baseline(self, traj_uri: str, visited: set, top_k: int) -> list[dict]:
        """Most popular unvisited artworks on the same floor."""
        interactions = self.interaction_data.get(traj_uri, [])
        floor_counts = Counter(i['floor'] for i in interactions if i['floor'] >= 0)
        primary_floor = floor_counts.most_common(1)[0][0] if floor_counts else 0

        art_counts = Counter()
        for arts in self.traj_artwork_sets.values():
            for a in arts:
                art_counts[a] += 1

        ranked = []
        for poi, count in art_counts.most_common():
            if poi not in visited:
                poi_floor = self.poi_metadata.get(poi, {}).get('floor', -1)
                if poi_floor == primary_floor:
                    ranked.append({
                        'poi': poi,
                        'label': self.poi_metadata.get(poi, {}).get('label', ''),
                        'category': self.poi_metadata.get(poi, {}).get('category', ''),
                        'floor': poi_floor,
                        'score': count / max(len(self.traj_artwork_sets), 1),
                        'signals': {'same_floor_popularity': count},
                    })
                    if len(ranked) >= top_k:
                        break
        return ranked

    def _sparql_baseline(self, traj_uri: str, top_k: int) -> list[dict]:
        """Recommendations from SPARQL SR-30..34 rules."""
        try:
            from .semantic_reasoning import SemanticReasoningEngine
            engine = SemanticReasoningEngine(self.endpoint, self.update_endpoint)
            recs = engine.recommend(traj_uri, top_k=top_k)
            return [{'poi': r['poi'],
                     'label': r.get('poiLabel', ''),
                     'category': r['categories'][0] if r['categories'] else '',
                     'floor': self.poi_metadata.get(r['poi'], {}).get('floor', -1),
                     'score': r['score'],
                     'signals': {'sparql_rules': r['ruleCoverage']}}
                    for r in recs]
        except Exception:
            return []

    def _popularity_next_baseline(self, visited_set: set, top_k: int) -> list[dict]:
        """Popularity baseline for next-artwork prediction."""
        return self._popularity_baseline(visited_set, top_k)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_similar_visitor_artworks(self, traj_uri: str) -> list[str]:
        """Get artworks visited by similar visitors (for rank-based eval)."""
        if traj_uri not in self.features:
            return []

        similar = self.sim_engine.find_similar_semantic(traj_uri, top_k=5)
        visited = self.traj_artwork_sets.get(traj_uri, set())
        relevant = set()
        for sim_traj, _ in similar:
            sim_arts = self.traj_artwork_sets.get(sim_traj, set())
            relevant.update(sim_arts - visited)
        return list(relevant)

    def _get_nav_label(self, traj_uri: str) -> str:
        meta = self.traj_metadata.get(traj_uri, {})
        styles = meta.get('nav_styles', set())
        if not styles:
            return 'NoStyle'
        return next(iter(styles))

    def _dwell_metrics(self, pairs: list[tuple[float, float]]) -> dict[str, float]:
        """Compute MAE, RMSE, Spearman from (actual, predicted) pairs."""
        from scipy.stats import spearmanr
        actuals = np.array([a for a, _ in pairs])
        preds = np.array([p for _, p in pairs])
        mae = np.mean(np.abs(actuals - preds))
        rmse = np.sqrt(np.mean((actuals - preds) ** 2))
        corr = 0.0
        if len(actuals) > 2:
            corr, _ = spearmanr(actuals, preds)
            if np.isnan(corr):
                corr = 0.0
        return {'mae': round(mae, 2), 'rmse': round(rmse, 2),
                'spearman': round(corr, 3), 'n': len(pairs)}

    def _compute_ndcg(self, ranked_pois: list[str], relevant_set: set, k: int) -> float:
        """Compute NDCG@k."""
        dcg = 0.0
        for i, poi in enumerate(ranked_pois[:k]):
            if poi in relevant_set:
                dcg += 1.0 / np.log2(i + 2)

        # Ideal DCG
        n_relevant = min(len(relevant_set), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))

        return dcg / idcg if idcg > 0 else 0.0

    def _compute_auc(self, ranked_pois: list[str], relevant_set: set) -> float:
        """Compute AUC from ranked list."""
        n_relevant = 0
        n_irrelevant = 0
        score = 0.0

        for poi in ranked_pois:
            if poi in relevant_set:
                n_relevant += 1
            else:
                n_irrelevant += 1
                score += n_relevant  # Count relevant items ranked above this irrelevant one

        total = n_relevant * n_irrelevant
        return score / total if total > 0 else 0.5

    @staticmethod
    def bootstrap_ci(values: list[float], n_resamples: int = 1000,
                     ci: float = 0.95) -> tuple[float, float]:
        """Bootstrapped confidence interval."""
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
    # Spatial Visualization
    # ------------------------------------------------------------------

    def get_trajectory_coords(self, traj_uri: str,
                              poi_list: Optional[list[str]] = None) -> list[dict]:
        """
        Get ordered (x, y, floor) coordinates for a trajectory's artworks.
        If poi_list given, returns coords for those POIs in that order.
        Otherwise uses the stored sequence.
        """
        if poi_list is None:
            poi_list = self.sequences.get(traj_uri, [])

        coords = []
        for poi in poi_list:
            if poi in self.artwork_coords:
                c = self.artwork_coords[poi]
                coords.append({
                    'poi': poi,
                    'label': self.poi_metadata.get(poi, {}).get('label', poi.split('/')[-1]),
                    'category': self.poi_metadata.get(poi, {}).get('category', 'Unknown'),
                    'x': c['x'],
                    'y': c['y'],
                    'floor': c['floor'],
                })
        return coords

    def plot_trajectory_on_floor(self, traj_uri: str,
                                 predicted_pois: Optional[list[str]] = None,
                                 holdout_pois: Optional[list[str]] = None,
                                 title: Optional[str] = None,
                                 floor_filter: Optional[int] = None,
                                 ax=None):
        """
        Plot a trajectory on the museum floor plan (scatter + arrows).

        Shows:
        - Visited artworks as connected scatter points (blue)
        - Holdout artworks in green (actual next artworks the visitor went to)
        - Predicted artworks in red (model predictions)
        - Arrows showing visit order
        - All other artworks as faint gray dots for spatial context

        Args:
            traj_uri: Trajectory to plot
            predicted_pois: POI URIs predicted by the model
            holdout_pois: Actual held-out POI URIs (ground truth)
            title: Plot title
            floor_filter: If set, only show artworks on this floor
            ax: matplotlib axes (creates new figure if None)
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 9))
        else:
            fig = ax.get_figure()

        # Background: all artworks as faint dots
        for _, c in self.artwork_coords.items():
            if floor_filter is not None and c['floor'] != floor_filter:
                continue
            ax.scatter(c['x'], c['y'], c=f'C{c["floor"]}', s=8, alpha=0.08, zorder=1)

        # Visited sequence
        visited_coords = self.get_trajectory_coords(traj_uri)
        if floor_filter is not None:
            visited_coords = [c for c in visited_coords if c['floor'] == floor_filter]

        if visited_coords:
            xs = [c['x'] for c in visited_coords]
            ys = [c['y'] for c in visited_coords]

            # Draw path arrows
            for i in range(len(xs) - 1):
                ax.annotate('', xy=(xs[i+1], ys[i+1]), xytext=(xs[i], ys[i]),
                            arrowprops=dict(arrowstyle='->', color='#1976D2',
                                            lw=1.2, alpha=0.5),
                            zorder=2)

            # Visited points
            ax.scatter(xs, ys, c='#1976D2', s=50, zorder=4, edgecolors='white',
                       linewidths=0.5, label='Visited')

            # Number the visit order
            for i, c in enumerate(visited_coords):
                ax.annotate(str(i+1), (c['x'], c['y']),
                            fontsize=6, ha='center', va='center',
                            color='white', fontweight='bold', zorder=5)

        # Holdout artworks (ground truth)
        if holdout_pois:
            holdout_coords = self.get_trajectory_coords(traj_uri, holdout_pois)
            if floor_filter is not None:
                holdout_coords = [c for c in holdout_coords if c['floor'] == floor_filter]
            if holdout_coords:
                hx = [c['x'] for c in holdout_coords]
                hy = [c['y'] for c in holdout_coords]
                ax.scatter(hx, hy, c='#4CAF50', s=80, marker='D', zorder=4,
                           edgecolors='white', linewidths=0.8, label='Actual next')
                for c in holdout_coords:
                    ax.annotate(c['label'][:18], (c['x'], c['y']),
                                fontsize=5, ha='left', va='bottom',
                                xytext=(4, 4), textcoords='offset points',
                                color='#2E7D32', zorder=5)

                # Arrow from last visited to first holdout
                if visited_coords and holdout_coords:
                    ax.annotate('', xy=(holdout_coords[0]['x'], holdout_coords[0]['y']),
                                xytext=(visited_coords[-1]['x'], visited_coords[-1]['y']),
                                arrowprops=dict(arrowstyle='->', color='#4CAF50',
                                                lw=1.5, linestyle='--', alpha=0.7),
                                zorder=3)

        # Predicted artworks
        if predicted_pois:
            pred_coords = self.get_trajectory_coords(traj_uri, predicted_pois)
            if floor_filter is not None:
                pred_coords = [c for c in pred_coords if c['floor'] == floor_filter]
            if pred_coords:
                px = [c['x'] for c in pred_coords]
                py = [c['y'] for c in pred_coords]
                ax.scatter(px, py, c='#F44336', s=80, marker='*', zorder=4,
                           edgecolors='white', linewidths=0.5, label='Predicted')
                for c in pred_coords:
                    ax.annotate(c['label'][:18], (c['x'], c['y']),
                                fontsize=5, ha='left', va='bottom',
                                xytext=(4, -8), textcoords='offset points',
                                color='#C62828', zorder=5)

        # Floor color legend
        floor_colors = {0: 'C0', 1: 'C1', 2: 'C2'}
        for fl, col in floor_colors.items():
            if floor_filter is not None and fl != floor_filter:
                continue
            ax.scatter([], [], c=col, s=20, alpha=0.3, label=f'Floor {fl} artworks')

        meta = self.traj_metadata.get(traj_uri, {})
        nav = meta.get('nav_styles', set())
        eng = meta.get('engagement', '?')
        default_title = (f"{traj_uri.split('/')[-1]}  |  nav={nav or 'none'}  |  eng={eng}"
                         + (f"  |  floor {floor_filter}" if floor_filter is not None else ""))
        ax.set_title(title or default_title, fontsize=10)
        ax.set_xlabel('Easting (px)')
        ax.set_ylabel('Northing (px)')
        ax.legend(loc='upper right', fontsize=7, framealpha=0.8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.15)

        return fig, ax

    def plot_actual_vs_predicted_map(self, traj_uri: str,
                                     split_ratio: float = 0.7,
                                     top_k: int = 5,
                                     floor_filter: Optional[int] = None):
        """
        Visualization: split trajectory, show input path + actual holdout + predicted next.
        Creates a single figure with the trajectory on the museum floor plan.
        """
        import matplotlib.pyplot as plt

        input_seq, holdout = self.split_trajectory(traj_uri, split_ratio)
        meta = self.traj_metadata.get(traj_uri, {})
        context = {}
        nav_styles = meta.get('nav_styles', set())
        if nav_styles:
            context['nav_style'] = next(iter(nav_styles))

        preds = self.predict_next_artwork(input_seq, context=context, top_k=top_k)
        pred_pois = [p['poi'] for p in preds]

        fig, ax = self.plot_trajectory_on_floor(
            traj_uri,
            predicted_pois=pred_pois,
            holdout_pois=holdout,
            floor_filter=floor_filter,
        )

        # Add text annotation with prediction details
        hits = set(pred_pois) & set(holdout)
        info = (f"Input: {len(input_seq)} artworks  |  "
                f"Holdout: {len(holdout)}  |  "
                f"Predicted: {top_k}  |  "
                f"Hits: {len(hits)}")
        ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=7,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='bottom')

        plt.tight_layout()
        return fig

    def plot_recommendation_map(self, traj_uri: str, top_k: int = 10,
                                floor_filter: Optional[int] = None):
        """
        Visualization: visited trajectory + top-K recommended artworks on the floor plan.
        """
        import matplotlib.pyplot as plt

        recs = self.recommend(traj_uri, top_k=top_k)
        rec_pois = [r['poi'] for r in recs]

        fig, ax = self.plot_trajectory_on_floor(
            traj_uri,
            predicted_pois=rec_pois,
            floor_filter=floor_filter,
        )

        # Annotate top 3 with score
        for i, r in enumerate(recs[:3]):
            if r['poi'] in self.artwork_coords:
                c = self.artwork_coords[r['poi']]
                if floor_filter is not None and c['floor'] != floor_filter:
                    continue
                ax.annotate(f"#{i+1} ({r['score']:.2f})",
                            (c['x'], c['y']),
                            fontsize=6, ha='right', va='top',
                            xytext=(-4, -4), textcoords='offset points',
                            color='#B71C1C', fontweight='bold', zorder=6)

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Ablation Support
    # ------------------------------------------------------------------

    def _precompute_signals(self, protocol: str = 'trajectory_split',
                            split_ratio: float = 0.7, leave_k: int = 5
                            ) -> dict[str, dict[str, dict[str, float]]]:
        """Precompute raw signal scores per (trajectory, artwork) once.

        Returns {traj_uri: {art_uri: {signal: normalized_score}}}.
        """
        cache = {}
        eligible = [t for t in self.sequences if len(self.sequences[t]) >= 5]
        sig_artworks = self.significant_artworks or self._get_significant_artworks()

        for traj_uri in eligible:
            if protocol == 'trajectory_split':
                input_seq, holdout = self.split_trajectory(traj_uri, split_ratio)
            elif protocol == 'leave_k_out':
                input_seq, holdout = self.leave_k_out(traj_uri, leave_k)
            else:
                input_seq = self.sequences[traj_uri]
                holdout = self._get_similar_visitor_artworks(traj_uri)

            if not holdout:
                continue

            visited = set(input_seq)
            meta = self.traj_metadata.get(traj_uri, {})
            scores = defaultdict(lambda: defaultdict(float))

            # Forward-reachable candidates (directed flow edges only)
            nav_styles = meta.get('nav_styles', set())
            nav_style = next(iter(nav_styles)) if nav_styles else None
            forward_set = self._get_forward_candidates(input_seq, visited, nav_style)
            candidates = forward_set if forward_set else (self.all_artwork_uris - visited)

            # 1. Semantic similarity
            if traj_uri in self.features:
                similar = self.sim_engine.find_similar_semantic(
                    traj_uri, top_k=10, metric='cosine')
                for sim_traj, sim_score in similar:
                    for art in (self.traj_artwork_sets.get(sim_traj, set()) & candidates):
                        scores[art]['semantic_sim'] += sim_score

            # 2. Route similarity
            route_similar = self._find_route_similar(traj_uri, top_k=15)
            for sim in route_similar:
                for art in (self.traj_artwork_sets.get(sim['uri'], set()) & candidates):
                    scores[art]['route_sim'] += sim['score']

            # 3. Flow probability (recency-weighted)
            if self.flow_graph and input_seq:
                last_pois = input_seq[-3:] if len(input_seq) >= 3 else input_seq
                recency_weights = [0.5, 0.3, 0.2]
                for i, poi in enumerate(reversed(last_pois)):
                    w = recency_weights[i] if i < len(recency_weights) else 0.1
                    if self.flow_graph.has_node(poi):
                        adj = {}
                        for _, dst in self.flow_graph.out_edges(poi):
                            fw = self.flow_graph[poi][dst].get('weight', 0)
                            if fw > 0:
                                adj[dst] = fw
                        total_w = sum(adj.values())
                        if total_w > 0:
                            for dst, fw in adj.items():
                                if dst in candidates:
                                    scores[dst]['flow_prob'] += w * (fw / total_w)

            # 4. Dominant context
            dominant = self._get_dominant_artwork(traj_uri)
            if dominant and self.covisit_graph and self.covisit_graph.has_node(dominant):
                for neighbor in self.covisit_graph.neighbors(dominant):
                    if neighbor in candidates:
                        scores[neighbor]['dominant_ctx'] += self.covisit_graph[dominant][neighbor].get('weight', 1)

            # 5. User preference (category + room)
            user = self.traj_to_user.get(traj_uri)
            prefs = self.user_preferences.get(user, []) if user else []
            pref_map = {p['category']: p['score'] for p in prefs}
            room_prefs = self.user_room_preferences.get(user, {}) if user else {}
            for art in candidates:
                art_meta = self.poi_metadata.get(art, {})
                cat_score = pref_map.get(art_meta.get('category', ''), 0)
                room_score = room_prefs.get(art_meta.get('room', ''), 0)
                if cat_score > 0 or room_score > 0:
                    scores[art]['user_pref'] += 0.4 * cat_score + 0.6 * room_score

            # 6. Significant artworks
            for art in (sig_artworks & candidates):
                scores[art]['significant'] += 1.0

            # 7. Engagement (precomputed index)
            engagement = meta.get('engagement')
            if engagement and engagement in self._engagement_pois:
                eng_pois = self._engagement_pois[engagement]
                own_pois = self.traj_artwork_sets.get(traj_uri, set())
                for art in candidates:
                    count = eng_pois.get(art, 0)
                    if art in own_pois and count > 0:
                        count -= 1
                    if count > 0:
                        scores[art]['engagement'] += float(count)

            # Normalize each signal to [0, 1]
            all_signals = list(UNIFIED_WEIGHTS.keys())
            for signal in all_signals:
                vals = [scores[a][signal] for a in scores if scores[a][signal] > 0]
                if vals:
                    max_val = max(vals)
                    if max_val > 0:
                        for a in scores:
                            if scores[a][signal] > 0:
                                scores[a][signal] /= max_val

            cache[traj_uri] = {'scores': dict(scores), 'holdout': holdout}

        return cache

    def _evaluate_from_cache(self, cache: dict, weights: dict[str, float],
                             k_values: list[int]) -> dict[str, Any]:
        """Evaluate recommendations from precomputed signal cache."""
        results = {f'recall@{k}': [] for k in k_values}
        results.update({f'precision@{k}': [] for k in k_values})
        results.update({f'hit_rate@{k}': [] for k in k_values})
        results['mrr'] = []
        results['map'] = []
        results['nav_style_breakdown'] = defaultdict(lambda: {f'recall@{k}': [] for k in k_values})
        all_recommended = {k: set() for k in k_values}
        candidate_sizes = []

        for traj_uri, data in cache.items():
            holdout_set = set(data['holdout'])
            scores = data['scores']
            candidate_sizes.append(len(scores))

            # Re-weight and rank
            ranked_arts = []
            for art, signals in scores.items():
                total = sum(weights.get(sig, 0) * val for sig, val in signals.items())
                ranked_arts.append((art, total))
            ranked_arts.sort(key=lambda x: -x[1])

            rec_list = [art for art, _ in ranked_arts]

            for k in k_values:
                top_k = set(rec_list[:k])
                hits = len(top_k & holdout_set)
                recall = hits / len(holdout_set) if holdout_set else 0
                precision = hits / k
                results[f'recall@{k}'].append(recall)
                results[f'precision@{k}'].append(precision)
                results[f'hit_rate@{k}'].append(1.0 if hits > 0 else 0.0)
                all_recommended[k].update(top_k)

            # MRR
            rr = 0.0
            for rank, art in enumerate(rec_list, 1):
                if art in holdout_set:
                    rr = 1.0 / rank
                    break
            results['mrr'].append(rr)

            # MAP — average precision across all positions
            n_relevant = 0
            precision_sum = 0.0
            for rank, art in enumerate(rec_list, 1):
                if art in holdout_set:
                    n_relevant += 1
                    precision_sum += n_relevant / rank
            ap = precision_sum / len(holdout_set) if holdout_set else 0
            results['map'].append(ap)

            # Per-nav-style
            meta = self.traj_metadata.get(traj_uri, {})
            nav_styles = meta.get('nav_styles', set()) or {'NoStyle'}
            for style in nav_styles:
                for k in k_values:
                    top_k = set(rec_list[:k])
                    recall = len(top_k & holdout_set) / len(holdout_set) if holdout_set else 0
                    results['nav_style_breakdown'][style][f'recall@{k}'].append(recall)

        # Aggregate
        final = {'n_evaluated': len(cache)}
        for k in k_values:
            for metric in ['recall', 'precision', 'hit_rate']:
                key = f'{metric}@{k}'
                vals = results[key]
                final[key] = np.mean(vals) if vals else 0
                final[f'{key}_ci'] = self.bootstrap_ci(vals)
            final[f'coverage@{k}'] = len(all_recommended[k]) / len(self.all_artwork_uris) if self.all_artwork_uris else 0

        mrr_vals = results['mrr']
        final['mrr'] = np.mean(mrr_vals) if mrr_vals else 0
        final['mrr_ci'] = self.bootstrap_ci(mrr_vals)

        map_vals = results['map']
        final['map'] = np.mean(map_vals) if map_vals else 0
        final['map_ci'] = self.bootstrap_ci(map_vals)

        final['avg_candidates'] = np.mean(candidate_sizes) if candidate_sizes else 0
        final['min_candidates'] = min(candidate_sizes) if candidate_sizes else 0
        final['max_candidates'] = max(candidate_sizes) if candidate_sizes else 0

        final['nav_style_breakdown'] = {}
        for style, metrics in results['nav_style_breakdown'].items():
            style_result = {'n': len(metrics.get(f'recall@{k_values[0]}', []))}
            for k in k_values:
                vals = metrics[f'recall@{k}']
                style_result[f'recall@{k}'] = np.mean(vals) if vals else 0
            final['nav_style_breakdown'][style] = style_result

        return final

    def ablation_study(self, k_values: Optional[list[int]] = None,
                       protocol: str = 'trajectory_split'
                       ) -> dict[str, dict]:
        """
        Run evaluation with each signal turned off, plus baselines.
        Precomputes signals once, then re-weights for each config.
        Returns {config_name: evaluation_results}.
        """
        if k_values is None:
            k_values = [5, 10, 20]

        # Precompute all signal scores once
        print("  Precomputing signal scores...")
        cache = self._precompute_signals(protocol=protocol)
        print(f"  Cached {len(cache)} trajectories")

        configs = {'full': dict(UNIFIED_WEIGHTS)}
        for signal in UNIFIED_WEIGHTS:
            ablated = dict(UNIFIED_WEIGHTS)
            ablated[signal] = 0.0
            total = sum(ablated.values())
            if total > 0:
                ablated = {k: v / total for k, v in ablated.items()}
            configs[f'-{signal}'] = ablated

        # Flow-only baseline
        configs['flow_only'] = {s: 0.0 for s in UNIFIED_WEIGHTS}
        configs['flow_only']['flow_prob'] = 1.0

        # Weight sweep: flow at 10%, 20%, 30%, 40%, 50%
        for fw in [0.10, 0.20, 0.30, 0.40, 0.50]:
            remaining = 1.0 - fw
            swept = {s: v * remaining for s, v in UNIFIED_WEIGHTS.items()}
            swept['flow_prob'] = fw
            configs[f'flow_{int(fw * 100)}pct'] = swept

        results = {}

        # Multi-signal configs — fast re-weight from cache
        for config_name, weights in configs.items():
            print(f"  Evaluating {config_name}...")
            results[config_name] = self._evaluate_from_cache(cache, weights, k_values)

        # Baselines — these are cheap, no caching needed
        for baseline in ['popularity', 'random', 'same_floor']:
            print(f"  Evaluating {baseline}...")
            results[baseline] = self.evaluate_recommendations(
                method=baseline, k_values=k_values, protocol=protocol
            )

        return results

    def sweep_split_ratios(self, ratios: Optional[list[float]] = None,
                           configs: Optional[dict[str, dict[str, float]]] = None,
                           k_values: Optional[list[int]] = None) -> dict[float, dict[str, dict]]:
        """Sweep split ratios to find optimal operating point.

        For each ratio, precomputes signals once and evaluates multiple configs.
        Returns {ratio: {config_name: metrics_dict}}.
        """
        if ratios is None:
            ratios = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
        if k_values is None:
            k_values = [5, 10, 20]
        if configs is None:
            configs = {
                'full': dict(UNIFIED_WEIGHTS),
                'flow_only': {**{s: 0.0 for s in UNIFIED_WEIGHTS}, 'flow_prob': 1.0},
            }

        results = {}
        for ratio in ratios:
            print(f"  Split ratio {ratio:.2f}...")
            cache = self._precompute_signals(split_ratio=ratio)
            ratio_results = {}
            for config_name, weights in configs.items():
                ratio_results[config_name] = self._evaluate_from_cache(
                    cache, weights, k_values
                )
            # Popularity baseline (not cache-based)
            ratio_results['popularity'] = self.evaluate_recommendations(
                method='popularity', k_values=k_values, split_ratio=ratio
            )
            results[ratio] = ratio_results

        return results
