"""
Semantic Similarity Module for BIRD Museum Trajectory Analysis

Adapted from Trajecto's semantic_similarity.py for indoor museum context.
No temporal similarity (BIRD has elapsed timestamps, not wall-clock).

Similarity Modes:
- Semantic: Based on 59-dim behavioral features (72 extracted, navigation & engagement excluded to avoid signal overlap)
- Subgraph: Based on graph embeddings (if available)
- Hybrid: Weighted combination of both approaches
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.distance import cityblock, cosine, euclidean
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity

# Feature group weights for weighted similarity
# Navigation and engagement excluded (handled by dedicated recommendation signals)
DEFAULT_FEATURE_WEIGHTS = {
    'artwork_type': 1.0,   # Artwork category distribution
    'floor': 0.6,          # Floor distribution
    'room': 0.9,           # Room distribution (high entropy, discriminative)
    'era': 0.8,            # Creation era distribution
    'interaction': 0.8     # Interaction types
}

# Feature group boundaries (from semantic_features.py)
# 72 total dims: artwork_type(8) + floor(3) + room(37) + era(6) + navigation(5) + engagement(8) + interaction(5)
FEATURE_GROUPS = {
    'artwork_type': (0, 8),
    'floor': (8, 11),
    'room': (11, 48),
    'era': (48, 54),
    'navigation': (54, 59),
    'engagement': (59, 67),
    'interaction': (67, 72)
}

# Indices used for similarity computation — excludes navigation(54:59) and
# engagement(59:67) which have dedicated recommendation signals (nav_match,
# engagement) to avoid double-counting.
_SIM_INDICES = np.concatenate([
    np.arange(0, 54),    # artwork_type + floor + room + era
    np.arange(67, 72),   # interaction
])  # 59 dims


class TrajectorySimilarityEngine:
    """
    Computes similarity between museum trajectories using multiple approaches.

    Usage:
        engine = TrajectorySimilarityEngine()
        engine.load_semantic_features('features.json')
        similar = engine.find_similar(query_uri, mode='semantic', top_k=5)
    """

    def __init__(self,
                 semantic_features: Optional[dict[str, np.ndarray]] = None,
                 subgraph_embeddings: Optional[dict[str, np.ndarray]] = None,
                 feature_weights: Optional[dict[str, float]] = None):
        self.semantic_features = semantic_features or {}
        self.subgraph_embeddings = subgraph_embeddings or {}
        self.feature_weights = feature_weights or DEFAULT_FEATURE_WEIGHTS

        self._semantic_matrix = None
        self._semantic_uris = []
        self._subgraph_matrix = None
        self._subgraph_uris = []

    def load_semantic_features(self, input_path: str, normalize: bool = True):
        """Load semantic features from JSON file."""
        with open(input_path) as f:
            features_json = json.load(f)

        self.semantic_features = {
            uri: np.array(vec, dtype=np.float32)
            for uri, vec in features_json.items()
        }

        if normalize:
            self._normalize_semantic_features()

        self._semantic_uris = list(self.semantic_features.keys())
        self._semantic_matrix = np.array([
            self.semantic_features[uri] for uri in self._semantic_uris
        ])

        print(f"Loaded {len(self.semantic_features)} semantic feature vectors")

    def load_subgraph_embeddings(self, input_path: str, normalize: bool = True):
        """Load subgraph embeddings from JSON file."""
        with open(input_path) as f:
            data = json.load(f)

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
        """Compute semantic similarity between two trajectories.

        Uses 59-dim slice (excludes navigation & engagement dims) to avoid
        double-counting with dedicated nav_match and engagement signals.
        Weighted mode uses per-group computation on full vectors (skipping
        navigation and engagement groups).
        """
        if uri1 not in self.semantic_features or uri2 not in self.semantic_features:
            return 0.0

        if weighted:
            # Per-group weighting needs full vectors with original indices
            return self._weighted_similarity(
                self.semantic_features[uri1],
                self.semantic_features[uri2], metric)
        else:
            vec1 = self.semantic_features[uri1][_SIM_INDICES]
            vec2 = self.semantic_features[uri2][_SIM_INDICES]
            return self._compute_similarity(vec1, vec2, metric)

    def _compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray,
                           metric: str = 'cosine') -> float:
        """Compute similarity between two vectors."""
        if metric == 'cosine':
            sim = 1 - cosine(vec1, vec2)
            return max(0.0, sim)
        elif metric == 'euclidean':
            dist = euclidean(vec1, vec2)
            return 1.0 / (1.0 + dist)
        elif metric == 'manhattan':
            dist = cityblock(vec1, vec2)
            return 1.0 / (1.0 + dist)
        elif metric == 'pearson':
            corr, _ = pearsonr(vec1, vec2)
            return (corr + 1) / 2
        elif metric == 'spearman':
            corr, _ = spearmanr(vec1, vec2)
            return (corr + 1) / 2
        elif metric == 'dot':
            return np.dot(vec1, vec2)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def _weighted_similarity(self, vec1: np.ndarray, vec2: np.ndarray,
                            metric: str = 'cosine') -> float:
        """Compute weighted similarity using feature group weights.

        Skips navigation and engagement groups (handled by dedicated signals).
        """
        weighted_sims = []
        total_weight = 0

        for group, (start, end) in FEATURE_GROUPS.items():
            if group in ('navigation', 'engagement'):
                continue
            weight = self.feature_weights.get(group, 1.0)
            group_vec1 = vec1[start:end]
            group_vec2 = vec2[start:end]

            sim = self._compute_similarity(group_vec1, group_vec2, metric)
            weighted_sims.append(sim * weight)
            total_weight += weight

        return sum(weighted_sims) / total_weight if total_weight > 0 else 0.0

    def find_similar_semantic(self, query_uri: str, top_k: int = 10,
                              metric: str = 'cosine',
                              weighted: bool = False,
                              exclude_self: bool = True) -> list[tuple[str, float]]:
        """Find most similar trajectories using semantic features."""
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
        """Compute subgraph embedding similarity between two trajectories."""
        if uri1 not in self.subgraph_embeddings or uri2 not in self.subgraph_embeddings:
            return 0.0

        vec1 = self.subgraph_embeddings[uri1]
        vec2 = self.subgraph_embeddings[uri2]

        return self._compute_similarity(vec1, vec2, metric)

    def find_similar_subgraph(self, query_uri: str, top_k: int = 10,
                              metric: str = 'cosine',
                              exclude_self: bool = True) -> list[tuple[str, float]]:
        """Find most similar trajectories using subgraph embeddings."""
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
        """Compute hybrid similarity combining semantic and subgraph approaches."""
        subgraph_weight = 1.0 - semantic_weight

        sem_sim = self.semantic_similarity(uri1, uri2, metric)
        sub_sim = self.subgraph_similarity(uri1, uri2, metric)

        if uri1 not in self.subgraph_embeddings or uri2 not in self.subgraph_embeddings:
            return sem_sim
        if uri1 not in self.semantic_features or uri2 not in self.semantic_features:
            return sub_sim

        return semantic_weight * sem_sim + subgraph_weight * sub_sim

    def find_similar(self, query_uri: str, top_k: int = 10,
                     mode: str = 'semantic',
                     semantic_weight: float = 0.5,
                     metric: str = 'cosine',
                     exclude_self: bool = True) -> list[tuple[str, float]]:
        """Find most similar trajectories using specified mode."""
        if mode == 'semantic':
            return self.find_similar_semantic(query_uri, top_k, metric,
                                              exclude_self=exclude_self)
        elif mode == 'subgraph':
            return self.find_similar_subgraph(query_uri, top_k, metric,
                                              exclude_self=exclude_self)
        elif mode == 'hybrid':
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

    def compute_semantic_similarity_matrix(self, metric: str = 'cosine') -> tuple[np.ndarray, list[str]]:
        """Compute pairwise semantic similarity matrix."""
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

    # =========================================================================
    # Feature Group Analysis
    # =========================================================================

    def get_similarity_breakdown(self, uri1: str, uri2: str,
                                 metric: str = 'cosine') -> dict[str, float]:
        """Get similarity breakdown by feature group."""
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
                               top_k: int = 10) -> list[tuple[str, float]]:
        """Find similar trajectories focusing on a specific aspect."""
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
    # User Profile Similarity
    # =========================================================================

    def user_similarity(self, user1_trajectories: list[str],
                        user2_trajectories: list[str],
                        mode: str = 'semantic',
                        aggregation: str = 'mean') -> float:
        """Compute similarity between two users based on their trajectories."""
        if not user1_trajectories or not user2_trajectories:
            return 0.0

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

    def aggregate_user_features(self, trajectory_uris: list[str],
                                aggregation: str = 'mean') -> np.ndarray:
        """Aggregate trajectory features into a user profile."""
        features = []
        for uri in trajectory_uris:
            if uri in self.semantic_features:
                features.append(self.semantic_features[uri])

        if not features:
            return np.zeros(FEATURE_GROUPS['interaction'][1])

        features = np.array(features)

        if aggregation == 'mean':
            return np.mean(features, axis=0)
        elif aggregation == 'weighted':
            n = len(features)
            weights = np.exp(-0.3 * np.arange(n))
            weights = weights / weights.sum()
            return np.average(features, axis=0, weights=weights)
        elif aggregation == 'max':
            return np.max(features, axis=0)
        else:
            return np.mean(features, axis=0)

    # =========================================================================
    # Persistence
    # =========================================================================

    def save_similarity_data(self, output_dir: str):
        """Save all similarity data to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self._semantic_matrix is not None and len(self._semantic_matrix) > 0:
            sem_sim, sem_uris = self.compute_semantic_similarity_matrix()
            np.save(output_path / 'semantic_similarity_matrix.npy', sem_sim)
            with open(output_path / 'semantic_similarity_uris.json', 'w') as f:
                json.dump(sem_uris, f)

        print(f"Saved similarity data to {output_path}")


def compute_and_save_similarities(semantic_features_path: str,
                                   subgraph_embeddings_path: Optional[str] = None,
                                   output_dir: str = 'output/similarity') -> TrajectorySimilarityEngine:
    """Convenience function to compute and save all similarity data."""
    engine = TrajectorySimilarityEngine()

    engine.load_semantic_features(semantic_features_path)

    if subgraph_embeddings_path and Path(subgraph_embeddings_path).exists():
        engine.load_subgraph_embeddings(subgraph_embeddings_path)

    engine.save_similarity_data(output_dir)

    print("\nSimilarity computation complete!")
    return engine
