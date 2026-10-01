"""
Semantic Feature Extractor for Trajecto Trajectory Similarity

This module extracts location-agnostic semantic features from trajectories
to enable cross-region similarity comparison. Features are organized into
groups that capture behavioral patterns rather than specific POI identities.

Feature Groups (67 dimensions total; the category count is derived from
poi_categories.POI_CATEGORIES and must never be hardcoded downstream):
- POI Category Distribution (38 dims): Normalized visit proportions weighted by dwell time
- Temporal Context (12 dims): TimeOfDay, Weekpart, Season, duration metrics
- Navigation Style (4 dims): CasualMovement, ExploratoryMovement, ImmersiveMovement, TargetedSegment
- Engagement Metrics (8 dims): POI count, diversity, dwell stats, etc.
- Interaction Types (5 dims): Passing, Meaningful, Immersive, Start, Destination

Slice consumers (semantic_similarity.FEATURE_GROUPS, context_recommender
SIM_SLICES) must derive index ranges via feature_group_ranges().
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import requests
from urllib.parse import urlencode
import json
from pathlib import Path
from scipy.stats import entropy

from .fuseki_config import query_endpoint
from .poi_categories import POI_CATEGORIES, resolve_category

# SPARQL Prefixes
PREFIXES = """
PREFIX trajecto: <http://w3id.org/trajecto#>
PREFIX poi: <http://w3id.org/trajecto/poi#>
PREFIX patterns: <http://w3id.org/trajecto/patterns#>
PREFIX context: <http://w3id.org/trajecto/context#>
PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""

# Feature group dimensions. The category block tracks POI_CATEGORIES —
# hardcoding it caused a 75-vs-67 desync that silently misaligned every
# downstream slice consumer.
FEATURE_GROUPS = {
    'category': len(POI_CATEGORIES),  # POI categories from poi_categories.py
    'temporal': 12,      # TimeOfDay (4) + Weekpart (2) + Season (4) + duration (2)
    'navigation': 4,     # CasualMovement, ExploratoryMovement, ImmersiveMovement, TargetedSegment
    'engagement': 8,     # POI count, diversity, dwell stats, etc.
    'interaction': 5     # Passing, Meaningful, Immersive, Start, Destination
}

TOTAL_DIMENSIONS = sum(FEATURE_GROUPS.values())  # 67 with the 38-category list


def feature_group_ranges():
    """(start, end) index range per feature group, in vector order."""
    ranges, start = {}, 0
    for group in ('category', 'temporal', 'navigation', 'engagement',
                  'interaction'):
        ranges[group] = (start, start + FEATURE_GROUPS[group])
        start += FEATURE_GROUPS[group]
    return ranges

# Navigation styles (must match patterns:exhibitsPattern values in Fuseki)
NAVIGATION_STYLES = ['CasualMovement', 'ExploratoryMovement', 'ImmersiveMovement', 'TargetedSegment']

# Temporal features
TIME_OF_DAY = ['Morning', 'Afternoon', 'Evening', 'Night']
WEEKPART = ['Weekday', 'Weekend']
SEASONS = ['Spring', 'Summer', 'Fall', 'Winter']

# Interaction types
INTERACTION_TYPES = ['PassingInteraction', 'MeaningfulInteraction', 'ImmersiveInteraction',
                     'StartInteraction', 'DestinationInteraction']


class SemanticFeatureExtractor:
    """
    Extracts semantic features from trajectories for location-agnostic similarity.

    Features abstract away from specific POI identities to capture behavioral patterns
    that can be compared across different geographic regions.

    Usage:
        extractor = SemanticFeatureExtractor()
        features = extractor.extract_trajectory_features(trajectory_uri)
        user_profile = extractor.extract_user_profile(user_uri)
    """

    def __init__(self, endpoint: str = None):
        """
        Initialize the feature extractor.

        Args:
            endpoint: SPARQL endpoint URL. Defaults to fuseki_config.query_endpoint
        """
        self.endpoint = endpoint or query_endpoint
        self.poi_categories = POI_CATEGORIES
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

    def _build_feature_names(self) -> List[str]:
        """Build list of feature names for each dimension."""
        names = []

        # Category features (46)
        for cat in self.poi_categories:
            names.append(f'cat_{cat}')

        # Temporal features (12)
        for tod in TIME_OF_DAY:
            names.append(f'time_{tod}')
        for wp in WEEKPART:
            names.append(f'weekpart_{wp}')
        for season in SEASONS:
            names.append(f'season_{season}')
        names.extend(['duration_normalized', 'activity_density'])

        # Navigation style (4)
        for style in NAVIGATION_STYLES:
            names.append(f'nav_{style}')

        # Engagement metrics (8)
        names.extend([
            'poi_count', 'category_diversity', 'avg_dwell', 'std_dwell',
            'min_dwell', 'max_dwell', 'revisit_ratio', 'transition_count'
        ])

        # Interaction types (5)
        for it in INTERACTION_TYPES:
            names.append(f'int_{it}')

        return names

    def get_all_trajectory_uris(self) -> List[str]:
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

        Args:
            trajectory_uri: URI of the trajectory

        Returns:
            numpy array of shape (TOTAL_DIMENSIONS,) with all semantic features
        """
        features = []

        # 1. Category distribution (len(POI_CATEGORIES) dims)
        cat_features = self._extract_category_features(trajectory_uri)
        features.extend(cat_features)

        # 2. Temporal features (12 dims)
        temporal_features = self._extract_temporal_features(trajectory_uri)
        features.extend(temporal_features)

        # 3. Navigation style (4 dims)
        nav_features = self._extract_navigation_features(trajectory_uri)
        features.extend(nav_features)

        # 4. Engagement metrics (8 dims)
        engagement_features = self._extract_engagement_features(trajectory_uri)
        features.extend(engagement_features)

        # 5. Interaction types (5 dims)
        interaction_features = self._extract_interaction_features(trajectory_uri)
        features.extend(interaction_features)

        return np.array(features, dtype=np.float32)

    def _extract_category_features(self, trajectory_uri: str) -> List[float]:
        """Extract POI category distribution weighted by dwell time."""
        query = f"""
        SELECT ?category (SUM(?dwellTime) AS ?totalDwell)
        WHERE {{
            ?pos dul:isPartOf <{trajectory_uri}> .
            ?pos trajecto:hasInteraction ?interaction .
            ?interaction poi:interactsWith ?poi .
            ?interaction poi:hasDwellTimeInSeconds ?dwellTime .
            ?poi a ?category .
            FILTER(?category != trajecto:PointOfInterest)
            FILTER(?category != poi:SignificantPOI)
        }}
        GROUP BY ?category
        """
        df = self._execute_query(query)

        # Initialize category counts
        category_dwell = {cat: 0.0 for cat in self.poi_categories}

        if not df.empty:
            for _, row in df.iterrows():
                raw_name = row['category'].split('#')[-1]
                cat_name = resolve_category(raw_name)
                if cat_name and cat_name in category_dwell:
                    category_dwell[cat_name] += float(row['totalDwell'])

        # Normalize to probability distribution
        total = sum(category_dwell.values())
        if total > 0:
            features = [category_dwell[cat] / total for cat in self.poi_categories]
        else:
            features = [0.0] * len(self.poi_categories)

        return features

    def _extract_temporal_features(self, trajectory_uri: str) -> List[float]:
        """Extract temporal context features."""
        # Get interaction timestamps
        query = f"""
        SELECT ?timestamp ?dwellTime
        WHERE {{
            ?pos dul:isPartOf <{trajectory_uri}> .
            ?pos trajecto:timestamp ?timestamp .
            ?pos trajecto:hasInteraction ?interaction .
            ?interaction poi:hasDwellTimeInSeconds ?dwellTime .
        }}
        ORDER BY ?timestamp
        """
        df = self._execute_query(query)

        # Initialize time of day distribution
        time_of_day = {tod: 0.0 for tod in TIME_OF_DAY}
        weekpart = {wp: 0.0 for wp in WEEKPART}
        season = {s: 0.0 for s in SEASONS}

        total_duration = 0.0
        activity_count = 0

        if not df.empty:
            for _, row in df.iterrows():
                try:
                    timestamp = pd.to_datetime(row['timestamp'])
                    dwell = float(row['dwellTime']) if 'dwellTime' in row else 1.0

                    # Time of day
                    hour = timestamp.hour
                    if 6 <= hour < 12:
                        time_of_day['Morning'] += dwell
                    elif 12 <= hour < 17:
                        time_of_day['Afternoon'] += dwell
                    elif 17 <= hour < 21:
                        time_of_day['Evening'] += dwell
                    else:
                        time_of_day['Night'] += dwell

                    # Weekpart
                    if timestamp.weekday() < 5:
                        weekpart['Weekday'] += dwell
                    else:
                        weekpart['Weekend'] += dwell

                    # Season
                    month = timestamp.month
                    if month in [3, 4, 5]:
                        season['Spring'] += dwell
                    elif month in [6, 7, 8]:
                        season['Summer'] += dwell
                    elif month in [9, 10, 11]:
                        season['Fall'] += dwell
                    else:
                        season['Winter'] += dwell

                    total_duration += dwell
                    activity_count += 1
                except Exception:
                    pass

        # Normalize distributions
        features = []

        # Time of day (4)
        tod_total = sum(time_of_day.values())
        for tod in TIME_OF_DAY:
            features.append(time_of_day[tod] / tod_total if tod_total > 0 else 0.0)

        # Weekpart (2)
        wp_total = sum(weekpart.values())
        for wp in WEEKPART:
            features.append(weekpart[wp] / wp_total if wp_total > 0 else 0.0)

        # Season (4)
        season_total = sum(season.values())
        for s in SEASONS:
            features.append(season[s] / season_total if season_total > 0 else 0.0)

        # Duration metrics (2)
        # Normalize duration to [0, 1] (assuming max 8 hours = 28800 seconds)
        duration_normalized = min(total_duration / 28800, 1.0) if total_duration > 0 else 0.0
        # Activity density = activities per hour
        activity_density = (activity_count / (total_duration / 3600)) if total_duration > 0 else 0.0
        activity_density = min(activity_density / 10, 1.0)  # Normalize

        features.extend([duration_normalized, activity_density])

        return features

    def _extract_navigation_features(self, trajectory_uri: str) -> List[float]:
        """Extract navigation style features from patterns:exhibitsPattern triples."""
        query = f"""
        SELECT ?navStyle
        WHERE {{
            <{trajectory_uri}> patterns:exhibitsPattern ?navStyle .
            FILTER(STRSTARTS(STR(?navStyle), "http://w3id.org/trajecto/patterns#"))
        }}
        """
        df = self._execute_query(query)

        # Initialize style distribution
        styles = {style: 0.0 for style in NAVIGATION_STYLES}

        if not df.empty:
            for _, row in df.iterrows():
                style_name = row['navStyle'].split('#')[-1]
                if style_name in styles:
                    styles[style_name] = 1.0

        # Only default to uniform if no navigation patterns found at all
        if sum(styles.values()) == 0:
            for style in NAVIGATION_STYLES:
                styles[style] = 0.25

        return [styles[style] for style in NAVIGATION_STYLES]

    def _extract_engagement_features(self, trajectory_uri: str) -> List[float]:
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

        # POI count
        unique_pois = df['poi'].nunique()
        poi_count = min(unique_pois / 20, 1.0)  # Normalize (assume max 20 POIs)

        # Category diversity (entropy of category distribution)
        cat_features = self._extract_category_features(trajectory_uri)
        cat_dist = [f for f in cat_features if f > 0]
        if len(cat_dist) > 1:
            category_diversity = entropy(cat_dist) / np.log(len(self.poi_categories))
        else:
            category_diversity = 0.0

        # Dwell time statistics
        dwell_times = df['dwellTime'].astype(float).values
        avg_dwell = min(np.mean(dwell_times) / 3600, 1.0)  # Normalize to hours
        std_dwell = min(np.std(dwell_times) / 1800, 1.0) if len(dwell_times) > 1 else 0.0
        min_dwell = min(np.min(dwell_times) / 600, 1.0)  # Normalize to 10 min
        max_dwell = min(np.max(dwell_times) / 7200, 1.0)  # Normalize to 2 hours

        # Revisit ratio
        total_visits = len(df)
        revisit_ratio = (total_visits - unique_pois) / total_visits if total_visits > 0 else 0.0

        # Transition count (normalized)
        transition_count = min((unique_pois - 1) / 15, 1.0) if unique_pois > 0 else 0.0

        return [poi_count, category_diversity, avg_dwell, std_dwell,
                min_dwell, max_dwell, revisit_ratio, transition_count]

    def _extract_interaction_features(self, trajectory_uri: str) -> List[float]:
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

        # Initialize interaction counts
        interactions = {it: 0 for it in INTERACTION_TYPES}

        if not df.empty:
            for _, row in df.iterrows():
                int_name = row['interactionType'].split('#')[-1]
                if int_name in interactions:
                    interactions[int_name] = int(row['count'])

        # Normalize to probability distribution
        total = sum(interactions.values())
        if total > 0:
            features = [interactions[it] / total for it in INTERACTION_TYPES]
        else:
            features = [0.2] * len(INTERACTION_TYPES)  # Uniform if no data

        return features

    def extract_user_profile(self, user_uri: str, aggregation: str = 'mean') -> np.ndarray:
        """
        Extract aggregated semantic profile for a user.

        Args:
            user_uri: URI of the user
            aggregation: 'mean' or 'weighted' (by recency)

        Returns:
            numpy array with user's semantic profile
        """
        # Get user's trajectories
        query = f"""
        SELECT ?traj ?startTime
        WHERE {{
            <{user_uri}> trajecto:formsTrajectory ?traj .
            OPTIONAL {{ ?traj trajecto:hasStartTime ?startTime }}
        }}
        ORDER BY DESC(?startTime)
        """
        df = self._execute_query(query)

        if df.empty:
            return np.zeros(len(self.feature_names))

        # Extract features for each trajectory
        trajectory_features = []
        for _, row in df.iterrows():
            features = self.extract_trajectory_features(row['traj'])
            trajectory_features.append(features)

        trajectory_features = np.array(trajectory_features)

        if aggregation == 'weighted':
            # Exponential decay weights (recent trajectories weighted higher)
            n = len(trajectory_features)
            weights = np.exp(-0.3 * np.arange(n))
            weights = weights / weights.sum()
            return np.average(trajectory_features, axis=0, weights=weights)
        else:
            return np.mean(trajectory_features, axis=0)

    def batch_extract_features(self, trajectory_uris: List[str] = None,
                               progress_callback=None) -> Dict[str, np.ndarray]:
        """
        Extract features for multiple trajectories.

        Args:
            trajectory_uris: List of trajectory URIs. If None, extracts for all.
            progress_callback: Optional callback(current, total) for progress

        Returns:
            Dictionary mapping trajectory URI to feature vector
        """
        if trajectory_uris is None:
            trajectory_uris = self.get_all_trajectory_uris()

        total = len(trajectory_uris)
        print(f"Extracting semantic features for {total} trajectories...")

        features = {}
        for i, uri in enumerate(trajectory_uris):
            features[uri] = self.extract_trajectory_features(uri)
            if progress_callback:
                progress_callback(i + 1, total)
            elif (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{total} trajectories")

        print(f"Extracted features for {len(features)} trajectories")
        return features

    def extract_partial_features(self, trajectory_uri: str,
                                  available_groups: List[str] = None
                                  ) -> Tuple[np.ndarray, List[int]]:
        """
        Extract features for an in-progress trajectory, returning only
        the dimensions that are computable from available data.

        Args:
            trajectory_uri: URI of the trajectory
            available_groups: List of feature group names to extract.
                If None, extracts all groups that have data.

        Returns:
            Tuple of (feature_vector, observed_indices) where:
            - feature_vector is full-length with zeros for unobserved dims
            - observed_indices lists the indices that were actually computed
        """
        groups = self.get_feature_groups()
        total_dims = sum(end - start for start, end in groups.values())
        features = np.zeros(total_dims, dtype=np.float32)
        observed = []

        # Default: try all groups
        if available_groups is None:
            available_groups = list(groups.keys())

        extractors = {
            'category': self._extract_category_features,
            'temporal': self._extract_temporal_features,
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
                # Check if any non-zero values were extracted
                if any(v != 0 for v in group_features):
                    features[start:end] = group_features[:end - start]
                    observed.extend(range(start, end))
            except Exception:
                pass  # Skip groups that fail (e.g., missing data)

        return features, observed

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names

    def get_feature_groups(self) -> Dict[str, Tuple[int, int]]:
        """Get feature group boundaries (start_idx, end_idx) based on actual dimensions."""
        actual_dims = {
            'category': len(self.poi_categories),
            'temporal': 12,
            'navigation': 4,
            'engagement': 8,
            'interaction': 5
        }

        groups = {}
        offset = 0
        for group, dim in actual_dims.items():
            groups[group] = (offset, offset + dim)
            offset += dim
        return groups

    def save_features(self, features: Dict[str, np.ndarray], output_path: str):
        """Save extracted features to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        features_json = {uri: vec.tolist() for uri, vec in features.items()}

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(features_json, f)

        print(f"Saved features for {len(features)} trajectories to {output_path}")

    def load_features(self, input_path: str) -> Dict[str, np.ndarray]:
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
            'poi_categories': self.poi_categories,
            'navigation_styles': NAVIGATION_STYLES,
            'time_of_day': TIME_OF_DAY,
            'weekpart': WEEKPART,
            'seasons': SEASONS,
            'interaction_types': INTERACTION_TYPES
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved feature metadata to {output_path}")


def normalize_features(features: Dict[str, np.ndarray],
                       feature_groups: Dict[str, Tuple[int, int]]) -> Dict[str, np.ndarray]:
    """
    Z-normalize features per group so each group contributes equally.

    Without normalization, the 46 POI category dims (each ~0.02, low variance)
    dominate by count but add little discrimination, while engagement and temporal
    dims (fewer but more varied) get overshadowed.

    Per-group z-normalization ensures each feature group has zero mean and unit
    variance, so rare-but-distinctive categories get amplified and all groups
    contribute proportionally to distance-based methods (clustering, t-SNE).

    Args:
        features: Dict mapping trajectory URI -> raw feature vector
        feature_groups: Dict mapping group name -> (start_idx, end_idx)

    Returns:
        Dict mapping trajectory URI -> normalized feature vector
    """
    if not features:
        return features

    uris = list(features.keys())
    X = np.array([features[uri] for uri in uris], dtype=np.float32)

    for group_name, (start, end) in feature_groups.items():
        group_data = X[:, start:end]
        mean = group_data.mean(axis=0)
        std = group_data.std(axis=0)
        std[std == 0] = 1.0  # Avoid division by zero for constant features
        X[:, start:end] = (group_data - mean) / std

    return {uri: X[i] for i, uri in enumerate(uris)}


def extract_and_save_all_features(output_dir: str = 'output/semantic_features') -> Dict[str, np.ndarray]:
    """
    Convenience function to extract all trajectory features and save them.

    Args:
        output_dir: Directory to save output files

    Returns:
        Dictionary of trajectory URI -> feature vector
    """
    extractor = SemanticFeatureExtractor()

    # Extract features
    features = extractor.batch_extract_features()

    # Save features
    output_path = Path(output_dir)
    extractor.save_features(features, output_path / 'trajectory_features.json')

    # Save metadata
    extractor.save_metadata(output_path / 'feature_metadata.json')

    # Save trajectory IDs
    trajectory_ids = list(features.keys())
    with open(output_path / 'trajectory_ids.json', 'w') as f:
        json.dump(trajectory_ids, f, indent=2)

    print("\nFeature extraction complete!")
    print(f"  Features: {output_path / 'trajectory_features.json'}")
    print(f"  Metadata: {output_path / 'feature_metadata.json'}")
    print(f"  IDs: {output_path / 'trajectory_ids.json'}")

    return features
