"""
Semantic Similarity Module for Trajecto Trajectory Analysis

This module provides similarity computation functions for both:
- Semantic feature vectors (location-agnostic similarity)
- Subgraph embeddings (structure-aware similarity)

It supports multiple similarity metrics and hybrid approaches that combine
both semantic and structural similarity for comprehensive trajectory matching.

Similarity Modes:
- Semantic: Based on 67-dimensional behavioral features (cross-region capable)
- Subgraph: Based on graph embeddings (same-region structure matching)
- Hybrid: Weighted combination of both approaches
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy.spatial.distance import cosine, euclidean, cityblock
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import json
from pathlib import Path

# Feature group weights for weighted similarity
DEFAULT_FEATURE_WEIGHTS = {
    'category': 1.0,      # POI category distribution
    'temporal': 0.8,      # Temporal patterns
    'navigation': 1.0,    # Navigation style
    'engagement': 0.7,    # Engagement metrics
    'interaction': 0.8    # Interaction types
}

# Feature group boundaries derived from semantic_features so index ranges
# can never drift from the extractor's actual vector layout.
from .semantic_features import TOTAL_DIMENSIONS, feature_group_ranges

FEATURE_GROUPS = feature_group_ranges()


class TrajectorySimilarityEngine:
    """
    Computes similarity between trajectories using multiple approaches.

    Supports semantic (location-agnostic) and subgraph (structure-aware)
    similarity with various metrics and combination strategies.

    Usage:
        engine = TrajectorySimilarityEngine()
        engine.load_semantic_features('features.json')
        engine.load_subgraph_embeddings('embeddings.json')
        similar = engine.find_similar(query_uri, mode='hybrid', top_k=5)
    """

    def __init__(self,
                 semantic_features: Dict[str, np.ndarray] = None,
                 subgraph_embeddings: Dict[str, np.ndarray] = None,
                 feature_weights: Dict[str, float] = None):
        """
        Initialize the similarity engine.

        Args:
            semantic_features: Dict of URI -> TOTAL_DIMENSIONS feature vector
            subgraph_embeddings: Dict of URI -> embedding vector
            feature_weights: Weights for feature groups in similarity computation
        """
        self.semantic_features = semantic_features or {}
        self.subgraph_embeddings = subgraph_embeddings or {}
        self.feature_weights = feature_weights or DEFAULT_FEATURE_WEIGHTS

        # Precomputed matrices for fast similarity lookup
        self._semantic_matrix = None
        self._semantic_uris = []
        self._subgraph_matrix = None
        self._subgraph_uris = []

        # Scalers for normalization
        self._semantic_scaler = None
        self._subgraph_scaler = None

    def load_semantic_features(self, input_path: str, normalize: bool = True):
        """
        Load semantic features from JSON file.

        Args:
            input_path: Path to features JSON
            normalize: Whether to normalize features
        """
        with open(input_path) as f:
            features_json = json.load(f)

        self.semantic_features = {
            uri: np.array(vec, dtype=np.float32)
            for uri, vec in features_json.items()
        }

        if normalize:
            self._normalize_semantic_features()

        # Precompute matrix
        self._semantic_uris = list(self.semantic_features.keys())
        self._semantic_matrix = np.array([
            self.semantic_features[uri] for uri in self._semantic_uris
        ])

        print(f"Loaded {len(self.semantic_features)} semantic feature vectors")

    def load_subgraph_embeddings(self, input_path: str, normalize: bool = True):
        """
        Load subgraph embeddings from JSON file.

        Args:
            input_path: Path to embeddings JSON
            normalize: Whether to normalize embeddings
        """
        with open(input_path) as f:
            data = json.load(f)

        # Handle both direct format and nested format
        if 'embeddings' in data:
            embeddings_json = data['embeddings']
        else:
            embeddings_json = data

        self.subgraph_embeddings = {
            uri: np.array(vec, dtype=np.float32)
            for uri, vec in embeddings_json.items()
        }

        if normalize:
            self._normalize_subgraph_embeddings()

        # Precompute matrix
        self._subgraph_uris = list(self.subgraph_embeddings.keys())
        self._subgraph_matrix = np.array([
            self.subgraph_embeddings[uri] for uri in self._subgraph_uris
        ])

        print(f"Loaded {len(self.subgraph_embeddings)} subgraph embeddings")

    def _normalize_semantic_features(self):
        """Apply L2 normalization to semantic features."""
        for uri in self.semantic_features:
            vec = self.semantic_features[uri]
            norm = np.linalg.norm(vec)
            if norm > 0:
                self.semantic_features[uri] = vec / norm

    def _normalize_subgraph_embeddings(self):
        """Apply L2 normalization to subgraph embeddings."""
        for uri in self.subgraph_embeddings:
            vec = self.subgraph_embeddings[uri]
            norm = np.linalg.norm(vec)
            if norm > 0:
                self.subgraph_embeddings[uri] = vec / norm

    # =========================================================================
    # Semantic Similarity Methods
    # =========================================================================

    def semantic_similarity(self, uri1: str, uri2: str,
                           metric: str = 'cosine',
                           weighted: bool = False) -> float:
        """
        Compute semantic similarity between two trajectories.

        Args:
            uri1: First trajectory URI
            uri2: Second trajectory URI
            metric: Similarity metric ('cosine', 'euclidean', 'manhattan', 'pearson')
            weighted: Whether to use feature group weights

        Returns:
            Similarity score in [0, 1] for most metrics
        """
        if uri1 not in self.semantic_features or uri2 not in self.semantic_features:
            return 0.0

        vec1 = self.semantic_features[uri1]
        vec2 = self.semantic_features[uri2]

        if weighted:
            return self._weighted_similarity(vec1, vec2, metric)
        else:
            return self._compute_similarity(vec1, vec2, metric)

    def _compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray,
                           metric: str = 'cosine') -> float:
        """Compute similarity between two vectors."""
        if metric == 'cosine':
            # Cosine similarity (1 - cosine distance)
            sim = 1 - cosine(vec1, vec2)
            return max(0.0, sim)  # Handle numerical errors

        elif metric == 'euclidean':
            # Convert distance to similarity
            dist = euclidean(vec1, vec2)
            return 1.0 / (1.0 + dist)

        elif metric == 'manhattan':
            dist = cityblock(vec1, vec2)
            return 1.0 / (1.0 + dist)

        elif metric == 'pearson':
            corr, _ = pearsonr(vec1, vec2)
            return (corr + 1) / 2  # Normalize to [0, 1]

        elif metric == 'spearman':
            corr, _ = spearmanr(vec1, vec2)
            return (corr + 1) / 2

        elif metric == 'dot':
            return np.dot(vec1, vec2)

        else:
            raise ValueError(f"Unknown metric: {metric}")

    def _weighted_similarity(self, vec1: np.ndarray, vec2: np.ndarray,
                            metric: str = 'cosine') -> float:
        """
        Compute weighted similarity using feature group weights.

        Different feature groups can have different importance based on
        the application (e.g., category matching vs temporal patterns).
        """
        weighted_sims = []
        total_weight = 0

        for group, (start, end) in FEATURE_GROUPS.items():
            weight = self.feature_weights.get(group, 1.0)
            group_vec1 = vec1[start:end]
            group_vec2 = vec2[start:end]

            # Compute similarity for this group
            sim = self._compute_similarity(group_vec1, group_vec2, metric)
            weighted_sims.append(sim * weight)
            total_weight += weight

        return sum(weighted_sims) / total_weight if total_weight > 0 else 0.0

    def find_similar_semantic(self, query_uri: str, top_k: int = 10,
                              metric: str = 'cosine',
                              weighted: bool = False,
                              exclude_self: bool = True) -> List[Tuple[str, float]]:
        """
        Find most similar trajectories using semantic features.

        Args:
            query_uri: Query trajectory URI
            top_k: Number of results
            metric: Similarity metric
            weighted: Use feature group weights
            exclude_self: Exclude query from results

        Returns:
            List of (URI, similarity) tuples sorted by similarity
        """
        if query_uri not in self.semantic_features:
            return []

        similarities = []
        for uri in self.semantic_features:
            if exclude_self and uri == query_uri:
                continue
            sim = self.semantic_similarity(query_uri, uri, metric, weighted)
            similarities.append((uri, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    # =========================================================================
    # Subgraph Similarity Methods
    # =========================================================================

    def subgraph_similarity(self, uri1: str, uri2: str,
                            metric: str = 'cosine') -> float:
        """
        Compute subgraph embedding similarity between two trajectories.

        Args:
            uri1: First trajectory URI
            uri2: Second trajectory URI
            metric: Similarity metric

        Returns:
            Similarity score
        """
        if uri1 not in self.subgraph_embeddings or uri2 not in self.subgraph_embeddings:
            return 0.0

        vec1 = self.subgraph_embeddings[uri1]
        vec2 = self.subgraph_embeddings[uri2]

        return self._compute_similarity(vec1, vec2, metric)

    def find_similar_subgraph(self, query_uri: str, top_k: int = 10,
                              metric: str = 'cosine',
                              exclude_self: bool = True) -> List[Tuple[str, float]]:
        """
        Find most similar trajectories using subgraph embeddings.

        Args:
            query_uri: Query trajectory URI
            top_k: Number of results
            metric: Similarity metric
            exclude_self: Exclude query from results

        Returns:
            List of (URI, similarity) tuples
        """
        if query_uri not in self.subgraph_embeddings:
            return []

        similarities = []
        for uri in self.subgraph_embeddings:
            if exclude_self and uri == query_uri:
                continue
            sim = self.subgraph_similarity(query_uri, uri, metric)
            similarities.append((uri, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    # =========================================================================
    # Hybrid Similarity Methods
    # =========================================================================

    def hybrid_similarity(self, uri1: str, uri2: str,
                          semantic_weight: float = 0.5,
                          metric: str = 'cosine') -> float:
        """
        Compute hybrid similarity combining semantic and subgraph approaches.

        Args:
            uri1: First trajectory URI
            uri2: Second trajectory URI
            semantic_weight: Weight for semantic similarity (0-1)
            metric: Similarity metric

        Returns:
            Combined similarity score
        """
        subgraph_weight = 1.0 - semantic_weight

        # Compute both similarities
        sem_sim = self.semantic_similarity(uri1, uri2, metric)
        sub_sim = self.subgraph_similarity(uri1, uri2, metric)

        # Handle cases where one type is missing
        if uri1 not in self.subgraph_embeddings or uri2 not in self.subgraph_embeddings:
            return sem_sim
        if uri1 not in self.semantic_features or uri2 not in self.semantic_features:
            return sub_sim

        return semantic_weight * sem_sim + subgraph_weight * sub_sim

    def find_similar(self, query_uri: str, top_k: int = 10,
                     mode: str = 'hybrid',
                     semantic_weight: float = 0.5,
                     metric: str = 'cosine',
                     exclude_self: bool = True) -> List[Tuple[str, float]]:
        """
        Find most similar trajectories using specified mode.

        Args:
            query_uri: Query trajectory URI
            top_k: Number of results
            mode: 'semantic', 'subgraph', or 'hybrid'
            semantic_weight: Weight for semantic in hybrid mode
            metric: Similarity metric
            exclude_self: Exclude query from results

        Returns:
            List of (URI, similarity) tuples
        """
        if mode == 'semantic':
            return self.find_similar_semantic(query_uri, top_k, metric,
                                              exclude_self=exclude_self)
        elif mode == 'subgraph':
            return self.find_similar_subgraph(query_uri, top_k, metric,
                                              exclude_self=exclude_self)
        elif mode == 'hybrid':
            # Get all URIs that have both representations
            all_uris = set(self.semantic_features.keys()) & set(self.subgraph_embeddings.keys())

            similarities = []
            for uri in all_uris:
                if exclude_self and uri == query_uri:
                    continue
                sim = self.hybrid_similarity(query_uri, uri, semantic_weight, metric)
                similarities.append((uri, sim))

            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # =========================================================================
    # Batch Similarity Computation
    # =========================================================================

    def compute_semantic_similarity_matrix(self, metric: str = 'cosine') -> Tuple[np.ndarray, List[str]]:
        """
        Compute pairwise semantic similarity matrix.

        Returns:
            Tuple of (similarity matrix, list of URIs)
        """
        if self._semantic_matrix is None or len(self._semantic_matrix) == 0:
            return np.array([]), []

        n = len(self._semantic_uris)

        if metric == 'cosine':
            sim_matrix = cosine_similarity(self._semantic_matrix)
        else:
            sim_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    sim = self._compute_similarity(
                        self._semantic_matrix[i],
                        self._semantic_matrix[j],
                        metric
                    )
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim

        return sim_matrix, self._semantic_uris

    def compute_subgraph_similarity_matrix(self, metric: str = 'cosine') -> Tuple[np.ndarray, List[str]]:
        """
        Compute pairwise subgraph similarity matrix.

        Returns:
            Tuple of (similarity matrix, list of URIs)
        """
        if self._subgraph_matrix is None or len(self._subgraph_matrix) == 0:
            return np.array([]), []

        n = len(self._subgraph_uris)

        if metric == 'cosine':
            sim_matrix = cosine_similarity(self._subgraph_matrix)
        else:
            sim_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    sim = self._compute_similarity(
                        self._subgraph_matrix[i],
                        self._subgraph_matrix[j],
                        metric
                    )
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim

        return sim_matrix, self._subgraph_uris

    # =========================================================================
    # User Profile Similarity
    # =========================================================================

    def user_similarity(self, user1_trajectories: List[str],
                        user2_trajectories: List[str],
                        mode: str = 'semantic',
                        aggregation: str = 'mean') -> float:
        """
        Compute similarity between two users based on their trajectories.

        Args:
            user1_trajectories: List of trajectory URIs for user 1
            user2_trajectories: List of trajectory URIs for user 2
            mode: Similarity mode ('semantic', 'subgraph', 'hybrid')
            aggregation: 'mean' averages all pairwise, 'max' takes max similarity

        Returns:
            User similarity score
        """
        if not user1_trajectories or not user2_trajectories:
            return 0.0

        # Compute all pairwise similarities
        similarities = []
        for t1 in user1_trajectories:
            for t2 in user2_trajectories:
                if mode == 'semantic':
                    sim = self.semantic_similarity(t1, t2)
                elif mode == 'subgraph':
                    sim = self.subgraph_similarity(t1, t2)
                else:
                    sim = self.hybrid_similarity(t1, t2)
                similarities.append(sim)

        if not similarities:
            return 0.0

        if aggregation == 'mean':
            return np.mean(similarities)
        elif aggregation == 'max':
            return np.max(similarities)
        elif aggregation == 'min':
            return np.min(similarities)
        else:
            return np.mean(similarities)

    def aggregate_user_features(self, trajectory_uris: List[str],
                                aggregation: str = 'mean') -> np.ndarray:
        """
        Aggregate trajectory features into a user profile.

        Args:
            trajectory_uris: List of user's trajectory URIs
            aggregation: 'mean', 'weighted' (by recency), or 'max'

        Returns:
            Aggregated feature vector
        """
        features = []
        for uri in trajectory_uris:
            if uri in self.semantic_features:
                features.append(self.semantic_features[uri])

        if not features:
            return np.zeros(TOTAL_DIMENSIONS)

        features = np.array(features)

        if aggregation == 'mean':
            return np.mean(features, axis=0)
        elif aggregation == 'weighted':
            # Exponential decay (recent trajectories weighted higher)
            n = len(features)
            weights = np.exp(-0.3 * np.arange(n))
            weights = weights / weights.sum()
            return np.average(features, axis=0, weights=weights)
        elif aggregation == 'max':
            return np.max(features, axis=0)
        else:
            return np.mean(features, axis=0)

    # =========================================================================
    # Feature Group Analysis
    # =========================================================================

    def get_similarity_breakdown(self, uri1: str, uri2: str,
                                 metric: str = 'cosine') -> Dict[str, float]:
        """
        Get similarity breakdown by feature group.

        Args:
            uri1: First trajectory URI
            uri2: Second trajectory URI
            metric: Similarity metric

        Returns:
            Dictionary of feature group -> similarity score
        """
        if uri1 not in self.semantic_features or uri2 not in self.semantic_features:
            return {}

        vec1 = self.semantic_features[uri1]
        vec2 = self.semantic_features[uri2]

        breakdown = {}
        for group, (start, end) in FEATURE_GROUPS.items():
            group_vec1 = vec1[start:end]
            group_vec2 = vec2[start:end]
            breakdown[group] = self._compute_similarity(group_vec1, group_vec2, metric)

        return breakdown

    def find_similar_by_aspect(self, query_uri: str, aspect: str,
                               top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find similar trajectories focusing on a specific aspect.

        Args:
            query_uri: Query trajectory URI
            aspect: Feature group name (category, temporal, navigation, etc.)
            top_k: Number of results

        Returns:
            List of (URI, similarity) tuples
        """
        if aspect not in FEATURE_GROUPS:
            raise ValueError(f"Unknown aspect: {aspect}")

        if query_uri not in self.semantic_features:
            return []

        start, end = FEATURE_GROUPS[aspect]
        query_vec = self.semantic_features[query_uri][start:end]

        similarities = []
        for uri, features in self.semantic_features.items():
            if uri == query_uri:
                continue
            target_vec = features[start:end]
            sim = self._compute_similarity(query_vec, target_vec, 'cosine')
            similarities.append((uri, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    # =========================================================================
    # Cross-Region Matching
    # =========================================================================

    def find_cross_region_matches(self, query_uri: str,
                                  target_region_uris: List[str],
                                  top_k: int = 5,
                                  min_similarity: float = 0.5) -> List[Tuple[str, float]]:
        """
        Find similar trajectories in a different region.

        Uses semantic features only (location-agnostic).

        Args:
            query_uri: Query trajectory URI
            target_region_uris: Trajectory URIs in target region
            top_k: Number of results
            min_similarity: Minimum similarity threshold

        Returns:
            List of (URI, similarity) tuples
        """
        if query_uri not in self.semantic_features:
            return []

        matches = []
        for uri in target_region_uris:
            if uri not in self.semantic_features:
                continue
            sim = self.semantic_similarity(query_uri, uri, weighted=True)
            if sim >= min_similarity:
                matches.append((uri, sim))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_k]

    # =========================================================================
    # Persistence
    # =========================================================================

    def save_similarity_data(self, output_dir: str):
        """Save all similarity data to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save semantic similarity matrix
        if self._semantic_matrix is not None and len(self._semantic_matrix) > 0:
            sem_sim, sem_uris = self.compute_semantic_similarity_matrix()
            np.save(output_path / 'semantic_similarity_matrix.npy', sem_sim)
            with open(output_path / 'semantic_similarity_uris.json', 'w') as f:
                json.dump(sem_uris, f)

        # Save subgraph similarity matrix
        if self._subgraph_matrix is not None and len(self._subgraph_matrix) > 0:
            sub_sim, sub_uris = self.compute_subgraph_similarity_matrix()
            np.save(output_path / 'subgraph_similarity_matrix.npy', sub_sim)
            with open(output_path / 'subgraph_similarity_uris.json', 'w') as f:
                json.dump(sub_uris, f)

        print(f"Saved similarity data to {output_path}")


def compute_and_save_similarities(semantic_features_path: str,
                                   subgraph_embeddings_path: str = None,
                                   output_dir: str = 'output/similarity') -> TrajectorySimilarityEngine:
    """
    Convenience function to compute and save all similarity data.

    Args:
        semantic_features_path: Path to semantic features JSON
        subgraph_embeddings_path: Path to subgraph embeddings JSON (optional)
        output_dir: Directory for output files

    Returns:
        Configured TrajectorySimilarityEngine
    """
    engine = TrajectorySimilarityEngine()

    # Load features
    engine.load_semantic_features(semantic_features_path)

    if subgraph_embeddings_path and Path(subgraph_embeddings_path).exists():
        engine.load_subgraph_embeddings(subgraph_embeddings_path)

    # Save similarity data
    engine.save_similarity_data(output_dir)

    print("\nSimilarity computation complete!")
    return engine
