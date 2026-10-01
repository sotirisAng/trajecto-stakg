"""
Semantic Feature Extractor for BIRD Museum Trajectory Similarity

Adapted from Trajecto's semantic_features.py for indoor museum context.
No temporal context (BIRD has elapsed timestamps, not wall-clock).
Uses poi:hasType for artwork categories instead of rdf:type class hierarchy.

Feature Groups (72 dimensions total):
- Artwork Type Distribution (8 dims): Category distribution weighted by dwell time
- Floor Distribution (3 dims): Proportion of time on floor 0, 1, 2
- Room Distribution (37 dims): Dwell-weighted proportion per room
- Era Distribution (6 dims): Dwell-weighted proportion per creation era
- Navigation Style (5 dims): CasualMovement, ExploratoryMovement, ImmersiveMovement,
                              TargetedSegment, SteadyMovement
- Engagement Metrics (8 dims): artwork_count, category_diversity, avg_dwell, std_dwell,
                                min_dwell, max_dwell, revisit_ratio, transition_count
- Interaction Types (5 dims): Passing, Meaningful, Immersive, Start, Destination
"""

import json
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
from scipy.stats import entropy

try:
    from .artwork_categories import ARTWORK_CATEGORIES, ERA_BUCKETS, ROOM_LIST, resolve_category, resolve_era
    from .fuseki_config import query_endpoint
except ImportError:
    from artwork_categories import ARTWORK_CATEGORIES, ERA_BUCKETS, ROOM_LIST, resolve_category, resolve_era
    from fuseki_config import query_endpoint

# SPARQL Prefixes
PREFIXES = """
PREFIX trajecto: <http://w3id.org/trajecto#>
PREFIX poi: <http://w3id.org/trajecto/poi#>
PREFIX patterns: <http://w3id.org/trajecto/patterns#>
PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""

# Feature group dimensions
FEATURE_GROUPS = {
    'artwork_type': 8,   # Artwork categories from artwork_categories.py
    'floor': 3,          # Floor 0, 1, 2
    'room': 37,          # 37 museum rooms
    'era': 6,            # 6 creation era buckets
    'navigation': 5,     # CasualMovement, ExploratoryMovement, ImmersiveMovement, TargetedSegment, SteadyMovement
    'engagement': 8,     # artwork_count, diversity, dwell stats, etc.
    'interaction': 5     # Passing, Meaningful, Immersive, Start, Destination
}

TOTAL_DIMENSIONS = sum(FEATURE_GROUPS.values())  # 72

# Navigation styles (must match patterns:exhibitsPattern values in Fuseki)
NAVIGATION_STYLES = ['CasualMovement', 'ExploratoryMovement', 'ImmersiveMovement',
                     'TargetedSegment', 'SteadyMovement']

# Floor levels in Nancy Museum
FLOOR_LEVELS = [0, 1, 2]

# Interaction types
INTERACTION_TYPES = ['PassingInteraction', 'MeaningfulInteraction', 'ImmersiveInteraction',
                     'StartInteraction', 'DestinationInteraction']


class SemanticFeatureExtractor:
    """
    Extracts semantic features from museum trajectories for similarity computation.

    Usage:
        extractor = SemanticFeatureExtractor()
        features = extractor.extract_trajectory_features(trajectory_uri)
    """

    def __init__(self, endpoint: Optional[str] = None):
        self.endpoint = endpoint or query_endpoint
        self.artwork_categories = ARTWORK_CATEGORIES
        self.feature_names = self._build_feature_names()

    def _execute_query(self, query: str) -> pd.DataFrame:
        """Execute SPARQL query and return results as DataFrame."""
        full_query = PREFIXES + query
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json'
        }
        data = urlencode({'query': full_query})

        try:
            response = requests.post(self.endpoint, data=data, headers=headers)
            response.raise_for_status()
            results = response.json()

            if not results or 'results' not in results:
                return pd.DataFrame()

            bindings = results['results']['bindings']
            if not bindings:
                return pd.DataFrame()

            rows = []
            for binding in bindings:
                row = {}
                for var, value in binding.items():
                    row[var] = value['value']
                rows.append(row)

            return pd.DataFrame(rows)

        except requests.exceptions.RequestException as e:
            print(f"Query Error: {e}")
            return pd.DataFrame()

    def _build_feature_names(self) -> list[str]:
        """Build list of feature names for each dimension."""
        names = []

        # Artwork type features (8)
        for cat in self.artwork_categories:
            names.append(f'art_{cat}')

        # Floor features (3)
        for floor in FLOOR_LEVELS:
            names.append(f'floor_{floor}')

        # Room features (37)
        for room in ROOM_LIST:
            names.append(f'room_{room}')

        # Era features (6)
        for era in ERA_BUCKETS:
            names.append(f'era_{era}')

        # Navigation style (5)
        for style in NAVIGATION_STYLES:
            names.append(f'nav_{style}')

        # Engagement metrics (8)
        names.extend([
            'artwork_count', 'category_diversity', 'avg_dwell', 'std_dwell',
            'min_dwell', 'max_dwell', 'revisit_ratio', 'transition_count'
        ])

        # Interaction types (5)
        for it in INTERACTION_TYPES:
            names.append(f'int_{it}')

        return names

    def get_all_trajectory_uris(self) -> list[str]:
        """Get all trajectory URIs from the knowledge graph."""
        query = """
        SELECT DISTINCT ?traj WHERE {
            ?user trajecto:formsTrajectory ?traj .
        }
        """
        df = self._execute_query(query)
        if df.empty:
            return []
        return df['traj'].tolist()

    def extract_trajectory_features(self, trajectory_uri: str) -> np.ndarray:
        """
        Extract full semantic feature vector for a trajectory.

        Returns:
            numpy array of shape (72,) with all semantic features
        """
        features = []

        # 1. Artwork type distribution (8 dims)
        cat_features = self._extract_category_features(trajectory_uri)
        features.extend(cat_features)

        # 2. Floor distribution (3 dims)
        floor_features = self._extract_floor_features(trajectory_uri)
        features.extend(floor_features)

        # 3. Room distribution (37 dims)
        room_features = self._extract_room_features(trajectory_uri)
        features.extend(room_features)

        # 4. Era distribution (6 dims)
        era_features = self._extract_era_features(trajectory_uri)
        features.extend(era_features)

        # 5. Navigation style (5 dims)
        nav_features = self._extract_navigation_features(trajectory_uri)
        features.extend(nav_features)

        # 6. Engagement metrics (8 dims)
        engagement_features = self._extract_engagement_features(trajectory_uri)
        features.extend(engagement_features)

        # 7. Interaction types (5 dims)
        interaction_features = self._extract_interaction_features(trajectory_uri)
        features.extend(interaction_features)

        return np.array(features, dtype=np.float32)

    def _extract_category_features(self, trajectory_uri: str) -> list[float]:
        """Extract artwork category distribution weighted by dwell time."""
        query = f"""
        SELECT ?artworkType (SUM(?dwellTime) AS ?totalDwell)
        WHERE {{
            ?pos dul:isPartOf <{trajectory_uri}> .
            ?pos trajecto:hasInteraction ?interaction .
            ?interaction poi:interactsWith ?poi .
            ?interaction poi:hasDwellTimeInSeconds ?dwellTime .
            ?poi poi:hasType ?artworkType .
        }}
        GROUP BY ?artworkType
        """
        df = self._execute_query(query)

        category_dwell = {cat: 0.0 for cat in self.artwork_categories}

        if not df.empty:
            for _, row in df.iterrows():
                raw_type = row['artworkType']
                cat_name = resolve_category(raw_type)
                if cat_name and cat_name in category_dwell:
                    category_dwell[cat_name] += float(row['totalDwell'])

        # Normalize to probability distribution
        total = sum(category_dwell.values())
        if total > 0:
            features = [category_dwell[cat] / total for cat in self.artwork_categories]
        else:
            features = [0.0] * len(self.artwork_categories)

        return features

    def _extract_floor_features(self, trajectory_uri: str) -> list[float]:
        """Extract floor distribution (proportion of time on each floor)."""
        query = f"""
        SELECT ?floor (COUNT(?pos) AS ?posCount)
        WHERE {{
            ?pos dul:isPartOf <{trajectory_uri}> .
            ?pos datacron:hasGeometry ?point .
            ?point trajecto:floor ?floor .
        }}
        GROUP BY ?floor
        """
        # Add datacron prefix
        full_query = """
        PREFIX datacron: <http://www.datacron-project.eu/datAcron#>
        """ + PREFIXES + query

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json'
        }
        data = urlencode({'query': full_query})

        floor_counts = {f: 0 for f in FLOOR_LEVELS}

        try:
            response = requests.post(self.endpoint, data=data, headers=headers)
            response.raise_for_status()
            results = response.json()

            if results and 'results' in results:
                for binding in results['results']['bindings']:
                    floor = int(binding['floor']['value'])
                    count = int(binding['posCount']['value'])
                    if floor in floor_counts:
                        floor_counts[floor] = count
        except requests.exceptions.RequestException:
            pass

        total = sum(floor_counts.values())
        if total > 0:
            features = [floor_counts[f] / total for f in FLOOR_LEVELS]
        else:
            features = [0.0] * len(FLOOR_LEVELS)

        return features

    def _extract_room_features(self, trajectory_uri: str) -> list[float]:
        """Extract room distribution weighted by dwell time (37 dims)."""
        query = f"""
        SELECT ?room (SUM(?dwellTime) AS ?totalDwell)
        WHERE {{
            ?pos dul:isPartOf <{trajectory_uri}> .
            ?pos trajecto:hasInteraction ?interaction .
            ?interaction poi:interactsWith ?poi .
            ?interaction poi:hasDwellTimeInSeconds ?dwellTime .
            ?poi poi:hasRoom ?roomUri .
            BIND(REPLACE(STR(?roomUri), "^.*#", "") AS ?room)
        }}
        GROUP BY ?room
        """
        df = self._execute_query(query)

        room_dwell = {room: 0.0 for room in ROOM_LIST}

        if not df.empty:
            for _, row in df.iterrows():
                room_name = row['room']
                if room_name in room_dwell:
                    room_dwell[room_name] += float(row['totalDwell'])

        total = sum(room_dwell.values())
        if total > 0:
            features = [room_dwell[room] / total for room in ROOM_LIST]
        else:
            features = [0.0] * len(ROOM_LIST)

        return features

    def _extract_era_features(self, trajectory_uri: str) -> list[float]:
        """Extract era distribution weighted by dwell time (6 dims)."""
        query = f"""
        SELECT ?period (SUM(?dwellTime) AS ?totalDwell)
        WHERE {{
            ?pos dul:isPartOf <{trajectory_uri}> .
            ?pos trajecto:hasInteraction ?interaction .
            ?interaction poi:interactsWith ?poi .
            ?interaction poi:hasDwellTimeInSeconds ?dwellTime .
            ?poi poi:creationPeriod ?period .
        }}
        GROUP BY ?period
        """
        df = self._execute_query(query)

        era_dwell = {era: 0.0 for era in ERA_BUCKETS}

        if not df.empty:
            for _, row in df.iterrows():
                era = resolve_era(row['period'])
                if era in era_dwell:
                    era_dwell[era] += float(row['totalDwell'])

        total = sum(era_dwell.values())
        if total > 0:
            features = [era_dwell[era] / total for era in ERA_BUCKETS]
        else:
            features = [0.0] * len(ERA_BUCKETS)

        return features

    def _extract_navigation_features(self, trajectory_uri: str) -> list[float]:
        """Extract navigation style features from patterns:exhibitsPattern triples."""
        query = f"""
        SELECT ?navStyle
        WHERE {{
            <{trajectory_uri}> patterns:exhibitsPattern ?navStyle .
            FILTER(STRSTARTS(STR(?navStyle), "http://w3id.org/trajecto/patterns#"))
        }}
        """
        df = self._execute_query(query)

        styles = {style: 0.0 for style in NAVIGATION_STYLES}

        if not df.empty:
            for _, row in df.iterrows():
                style_name = row['navStyle'].split('#')[-1]
                if style_name in styles:
                    styles[style_name] = 1.0

        if sum(styles.values()) == 0:
            for style in NAVIGATION_STYLES:
                styles[style] = 0.25

        return [styles[style] for style in NAVIGATION_STYLES]

    def _extract_engagement_features(self, trajectory_uri: str) -> list[float]:
        """Extract engagement metrics."""
        query = f"""
        SELECT ?poi ?dwellTime
        WHERE {{
            ?pos dul:isPartOf <{trajectory_uri}> .
            ?pos trajecto:hasInteraction ?interaction .
            ?interaction poi:interactsWith ?poi .
            ?interaction poi:hasDwellTimeInSeconds ?dwellTime .
            ?interaction a poi:MeaningfulInteraction .
        }}
        """
        df = self._execute_query(query)

        if df.empty:
            return [0.0] * 8

        # Artwork count (normalized, assume max 50 for museum)
        unique_pois = df['poi'].nunique()
        artwork_count = min(unique_pois / 50, 1.0)

        # Category diversity (entropy of category distribution)
        cat_features = self._extract_category_features(trajectory_uri)
        cat_dist = [f for f in cat_features if f > 0]
        if len(cat_dist) > 1:
            category_diversity = entropy(cat_dist) / np.log(len(self.artwork_categories))
        else:
            category_diversity = 0.0

        # Dwell time statistics (museum scale: seconds, not hours)
        dwell_times = df['dwellTime'].astype(float).values
        avg_dwell = min(np.mean(dwell_times) / 120, 1.0)      # Normalize to 2 min
        std_dwell = min(np.std(dwell_times) / 60, 1.0) if len(dwell_times) > 1 else 0.0
        min_dwell = min(np.min(dwell_times) / 60, 1.0)         # Normalize to 1 min
        max_dwell = min(np.max(dwell_times) / 300, 1.0)        # Normalize to 5 min

        # Revisit ratio
        total_visits = len(df)
        revisit_ratio = (total_visits - unique_pois) / total_visits if total_visits > 0 else 0.0

        # Transition count (normalized)
        transition_count = min((unique_pois - 1) / 30, 1.0) if unique_pois > 0 else 0.0

        return [artwork_count, category_diversity, avg_dwell, std_dwell,
                min_dwell, max_dwell, revisit_ratio, transition_count]

    def _extract_interaction_features(self, trajectory_uri: str) -> list[float]:
        """Extract interaction type distribution."""
        query = f"""
        SELECT ?interactionType (COUNT(?interaction) AS ?count)
        WHERE {{
            ?pos dul:isPartOf <{trajectory_uri}> .
            ?pos trajecto:hasInteraction ?interaction .
            ?interaction a ?interactionType .
            FILTER(STRSTARTS(STR(?interactionType), "http://w3id.org/trajecto/poi#"))
        }}
        GROUP BY ?interactionType
        """
        df = self._execute_query(query)

        interactions = {it: 0 for it in INTERACTION_TYPES}

        if not df.empty:
            for _, row in df.iterrows():
                int_name = row['interactionType'].split('#')[-1]
                if int_name in interactions:
                    interactions[int_name] = int(row['count'])

        total = sum(interactions.values())
        if total > 0:
            features = [interactions[it] / total for it in INTERACTION_TYPES]
        else:
            features = [0.2] * len(INTERACTION_TYPES)

        return features

    def extract_user_profile(self, user_uri: str, aggregation: str = 'mean') -> np.ndarray:
        """
        Extract aggregated semantic profile for a user.
        In BIRD each visitor has exactly one trajectory, so this returns that trajectory's features.
        """
        query = f"""
        SELECT ?traj
        WHERE {{
            <{user_uri}> trajecto:formsTrajectory ?traj .
        }}
        """
        df = self._execute_query(query)

        if df.empty:
            return np.zeros(len(self.feature_names))

        trajectory_features = []
        for _, row in df.iterrows():
            features = self.extract_trajectory_features(row['traj'])
            trajectory_features.append(features)

        trajectory_features = np.array(trajectory_features)

        if aggregation == 'weighted' and len(trajectory_features) > 1:
            n = len(trajectory_features)
            weights = np.exp(-0.3 * np.arange(n))
            weights = weights / weights.sum()
            return np.average(trajectory_features, axis=0, weights=weights)
        else:
            return np.mean(trajectory_features, axis=0)

    def batch_extract_features(self, trajectory_uris: list[str] = None,
                               progress_callback=None) -> dict[str, np.ndarray]:
        """Extract features for multiple trajectories."""
        if trajectory_uris is None:
            trajectory_uris = self.get_all_trajectory_uris()

        total = len(trajectory_uris)
        print(f"Extracting semantic features for {total} trajectories...")

        features = {}
        for i, uri in enumerate(trajectory_uris):
            features[uri] = self.extract_trajectory_features(uri)
            if progress_callback:
                progress_callback(i + 1, total)
            elif (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{total} trajectories")

        print(f"Extracted features for {len(features)} trajectories")
        return features

    def extract_partial_features(self, trajectory_uri: str,
                                  available_groups: list[str] = None
                                  ) -> tuple[np.ndarray, list[int]]:
        """
        Extract features for an in-progress trajectory, returning only
        the dimensions that are computable from available data.
        """
        groups = self.get_feature_groups()
        total_dims = sum(end - start for start, end in groups.values())
        features = np.zeros(total_dims, dtype=np.float32)
        observed = []

        if available_groups is None:
            available_groups = list(groups.keys())

        extractors = {
            'artwork_type': self._extract_category_features,
            'floor': self._extract_floor_features,
            'room': self._extract_room_features,
            'era': self._extract_era_features,
            'navigation': self._extract_navigation_features,
            'engagement': self._extract_engagement_features,
            'interaction': self._extract_interaction_features,
        }

        for group_name in available_groups:
            if group_name not in groups or group_name not in extractors:
                continue

            start, end = groups[group_name]
            try:
                group_features = extractors[group_name](trajectory_uri)
                if any(v != 0 for v in group_features):
                    features[start:end] = group_features[:end - start]
                    observed.extend(range(start, end))
            except Exception:
                pass

        return features, observed

    def get_feature_names(self) -> list[str]:
        """Get list of feature names."""
        return self.feature_names

    def get_feature_groups(self) -> dict[str, tuple[int, int]]:
        """Get feature group boundaries (start_idx, end_idx)."""
        groups = {}
        offset = 0
        for group, dim in FEATURE_GROUPS.items():
            groups[group] = (offset, offset + dim)
            offset += dim
        return groups

    def save_features(self, features: dict[str, np.ndarray], output_path: str):
        """Save extracted features to JSON file."""
        features_json = {uri: vec.tolist() for uri, vec in features.items()}

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(features_json, f)

        print(f"Saved features for {len(features)} trajectories to {output_path}")

    def load_features(self, input_path: str) -> dict[str, np.ndarray]:
        """Load features from JSON file."""
        with open(input_path) as f:
            features_json = json.load(f)

        features = {uri: np.array(vec, dtype=np.float32)
                    for uri, vec in features_json.items()}

        print(f"Loaded features for {len(features)} trajectories from {input_path}")
        return features

    def save_metadata(self, output_path: str):
        """Save feature metadata (names, groups, dimensions)."""
        metadata = {
            'total_dimensions': len(self.feature_names),
            'feature_names': self.feature_names,
            'feature_groups': FEATURE_GROUPS,
            'artwork_categories': self.artwork_categories,
            'navigation_styles': NAVIGATION_STYLES,
            'floor_levels': FLOOR_LEVELS,
            'interaction_types': INTERACTION_TYPES
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved feature metadata to {output_path}")


def normalize_features(features: dict[str, np.ndarray],
                       feature_groups: dict[str, tuple[int, int]]) -> dict[str, np.ndarray]:
    """
    Z-normalize features per group so each group contributes equally.
    """
    if not features:
        return features

    uris = list(features.keys())
    X = np.array([features[uri] for uri in uris], dtype=np.float32)

    for _group_name, (start, end) in feature_groups.items():
        group_data = X[:, start:end]
        mean = group_data.mean(axis=0)
        std = group_data.std(axis=0)
        std[std == 0] = 1.0
        X[:, start:end] = (group_data - mean) / std

    return {uri: X[i] for i, uri in enumerate(uris)}


def extract_and_save_all_features(output_dir: str = 'output/semantic_features') -> dict[str, np.ndarray]:
    """Convenience function to extract all trajectory features and save them."""
    extractor = SemanticFeatureExtractor()

    features = extractor.batch_extract_features()

    output_path = Path(output_dir)
    extractor.save_features(features, output_path / 'trajectory_features.json')
    extractor.save_metadata(output_path / 'feature_metadata.json')

    trajectory_ids = list(features.keys())
    with open(output_path / 'trajectory_ids.json', 'w') as f:
        json.dump(trajectory_ids, f, indent=2)

    print("\nFeature extraction complete!")
    print(f"  Features: {output_path / 'trajectory_features.json'}")
    print(f"  Metadata: {output_path / 'feature_metadata.json'}")
    print(f"  IDs: {output_path / 'trajectory_ids.json'}")

    return features
