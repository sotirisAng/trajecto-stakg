"""
Category Sequence Pattern Analyzer for Trajecto

Mines POI category transition patterns and n-grams from trajectory sequences.
Provides category saturation scoring to penalize redundant recommendations
and transition scoring to predict natural next categories.

Used by unified_recommender.py.
"""

import json
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
from pathlib import Path

# Categories where repeat visits are natural (multi-stop by nature)
REPEAT_EXCEPTION_CATEGORIES = {'Cultural', 'Historical', 'Museum', 'Monument',
                                'HistoricalLandmark', 'HistoricalSite',
                                'Artwork', 'Exhibit', 'Gallery'}


class CategorySequenceAnalyzer:
    """
    Analyzes POI category sequences across trajectories to extract
    transition patterns, n-gram frequencies, and category saturation scores.

    Usage:
        analyzer = CategorySequenceAnalyzer()
        analyzer.load_data(graphs_dir='output/graphs')
        analyzer.mine_category_transitions()
        analyzer.mine_ngram_patterns()
        score = analyzer.compute_sequence_score('Cafe', ['Cultural', 'Cultural'], {'Cultural': 2})
    """

    def __init__(self):
        self.poi_sequences = {}       # traj_uri -> [poi_uri, ...]
        self.poi_categories = {}      # poi_uri -> category_name
        self.category_sequences = {}  # traj_uri -> [category_name, ...]
        self.transition_matrix = {}   # category -> {next_category: probability}
        self.ngram_patterns = []      # [(ngram_tuple, support), ...]
        self._loaded = False

    def load_data(self, graphs_dir: str = None, poi_sequences: dict = None) -> None:
        """
        Load POI sequences and build category sequences.

        Uses pre-built sequences if provided, otherwise queries SPARQL directly.

        Args:
            graphs_dir: Unused, kept for backward compatibility
            poi_sequences: Optional pre-built dict of traj_uri -> [poi_uri, ...]
        """
        # Load POI categories via SPARQL
        self._load_poi_categories()

        # Use pre-built sequences if provided, otherwise query SPARQL
        if poi_sequences:
            self.poi_sequences = poi_sequences
            print(f"Using pre-built POI sequences for {len(self.poi_sequences)} trajectories")
        else:
            self._load_sequences_from_sparql()

        # Build category sequences from POI sequences
        for traj_uri, poi_seq in self.poi_sequences.items():
            cat_seq = []
            for poi in poi_seq:
                cat = self.poi_categories.get(poi)
                if cat:
                    cat_seq.append(cat)
            if cat_seq:
                self.category_sequences[traj_uri] = cat_seq

        print(f"Built category sequences for {len(self.category_sequences)} trajectories")
        self._loaded = True

    def _load_sequences_from_sparql(self) -> None:
        """Load POI visit sequences directly from SPARQL."""
        from .fuseki_config import query_endpoint
        from .fuseki_queries import execute_sparql_query, format_results

        query = """
        PREFIX trajecto: <http://w3id.org/trajecto#>
        PREFIX poi: <http://w3id.org/trajecto/poi#>
        PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#>

        SELECT ?traj ?poi ?timestamp
        WHERE {
            ?pos dul:isPartOf ?traj .
            ?pos trajecto:hasInteraction ?interaction .
            ?interaction poi:interactsWith ?poi .
            ?interaction a poi:MeaningfulInteraction .
            ?pos trajecto:timestamp ?timestamp .
        }
        ORDER BY ?traj ?timestamp
        """
        results = execute_sparql_query(query_endpoint, query)
        df = format_results(results)

        if df.empty:
            print("Warning: no POI sequences found via SPARQL")
            return

        for traj in df['traj'].unique():
            traj_df = df[df['traj'] == traj].sort_values('timestamp')
            pois = traj_df['poi'].tolist()

            # Remove consecutive duplicates
            unique_pois = [pois[0]] if pois else []
            for poi in pois[1:]:
                if poi != unique_pois[-1]:
                    unique_pois.append(poi)

            self.poi_sequences[traj] = unique_pois

        print(f"Loaded POI sequences from SPARQL for {len(self.poi_sequences)} trajectories")

    def _load_poi_categories(self) -> None:
        """Load POI -> category mapping from SPARQL."""
        from .fuseki_config import query_endpoint
        from .fuseki_queries import execute_sparql_query, format_results

        query = """
        PREFIX trajecto: <http://w3id.org/trajecto#>
        PREFIX poi: <http://w3id.org/trajecto/poi#>

        SELECT ?poi ?category
        WHERE {
            ?poi a trajecto:PointOfInterest .
            ?poi a ?category .
            FILTER(?category != trajecto:PointOfInterest)
            FILTER(?category != poi:SignificantPOI)
        }
        """
        results = execute_sparql_query(query_endpoint, query)
        df = format_results(results)

        if not df.empty:
            for _, row in df.iterrows():
                poi = row['poi']
                cat = row['category'].split('#')[-1]
                self.poi_categories[poi] = cat
            print(f"Loaded categories for {len(self.poi_categories)} POIs")

    def mine_category_transitions(self) -> Dict[str, Dict[str, float]]:
        """
        Build category-level transition probability matrix from all trajectories.

        Returns:
            Dict mapping source_category -> {target_category: probability}
        """
        transition_counts = defaultdict(Counter)

        for cat_seq in self.category_sequences.values():
            for i in range(len(cat_seq) - 1):
                src = cat_seq[i]
                dst = cat_seq[i + 1]
                transition_counts[src][dst] += 1

        # Normalize to probabilities
        self.transition_matrix = {}
        for src, targets in transition_counts.items():
            total = sum(targets.values())
            self.transition_matrix[src] = {
                dst: count / total for dst, count in targets.items()
            }

        print(f"Built transition matrix: {len(self.transition_matrix)} source categories")
        return self.transition_matrix

    def mine_ngram_patterns(self, n: int = 3,
                             min_support: float = 0.1) -> List[Tuple[Tuple[str, ...], float]]:
        """
        Extract frequent n-gram category patterns with support scores.

        Args:
            n: N-gram size (default 3)
            min_support: Minimum fraction of trajectories containing the pattern

        Returns:
            List of (ngram_tuple, support) sorted by support descending
        """
        total_trajs = len(self.category_sequences)
        if total_trajs == 0:
            return []

        ngram_counts = Counter()
        for cat_seq in self.category_sequences.values():
            # Extract all n-grams from this sequence
            seen_in_traj = set()
            for i in range(len(cat_seq) - n + 1):
                ngram = tuple(cat_seq[i:i + n])
                seen_in_traj.add(ngram)
            # Count each ngram once per trajectory
            for ngram in seen_in_traj:
                ngram_counts[ngram] += 1

        # Filter by minimum support
        self.ngram_patterns = [
            (ngram, count / total_trajs)
            for ngram, count in ngram_counts.most_common()
            if count / total_trajs >= min_support
        ]

        print(f"Found {len(self.ngram_patterns)} n-gram patterns (n={n}, min_support={min_support})")
        return self.ngram_patterns

    def get_next_category_scores(self,
                                  current_category_sequence: List[str]) -> Dict[str, float]:
        """
        Score each category for "what comes next" given the current sequence.

        Uses both the transition matrix (from last category) and n-gram pattern
        matching (from last n-1 categories).

        Args:
            current_category_sequence: Categories visited so far in this trajectory

        Returns:
            Dict mapping category_name -> score (0.0 to 1.0)
        """
        if not current_category_sequence:
            return {}

        scores = defaultdict(float)

        # Signal 1: Transition probability from last category (weight 0.6)
        last_cat = current_category_sequence[-1]
        transitions = self.transition_matrix.get(last_cat, {})
        for cat, prob in transitions.items():
            scores[cat] += 0.6 * prob

        # Signal 2: N-gram pattern matching (weight 0.4)
        if len(current_category_sequence) >= 2 and self.ngram_patterns:
            # Check which n-grams match the tail of current sequence
            for ngram, support in self.ngram_patterns:
                prefix_len = len(ngram) - 1
                if prefix_len <= len(current_category_sequence):
                    tail = tuple(current_category_sequence[-prefix_len:])
                    ngram_prefix = ngram[:prefix_len]
                    if tail == ngram_prefix:
                        next_cat = ngram[-1]
                        scores[next_cat] += 0.4 * support

        # Normalize to [0, 1]
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {cat: s / max_score for cat, s in scores.items()}

        return dict(scores)

    def get_category_saturation(self,
                                 visited_categories_in_trajectory: List[str]) -> Dict[str, float]:
        """
        Compute per-category saturation score based on visit counts in current trajectory.

        Saturation rules:
        - 0 visits -> 0.0 (not saturated)
        - 1 visit -> 0.0 (first visit, no penalty)
        - 2 visits -> 0.5 (moderate penalty), unless exception category
        - 3+ visits -> 0.8 (strong penalty), unless exception category
        - Exception categories (Cultural, Historical): 0.0 for 1-2 visits, 0.3 for 3+

        Args:
            visited_categories_in_trajectory: List of category names visited so far

        Returns:
            Dict mapping category_name -> saturation score (0.0 to 1.0)
        """
        counts = Counter(visited_categories_in_trajectory)
        saturation = {}

        for cat, count in counts.items():
            if cat in REPEAT_EXCEPTION_CATEGORIES:
                # Exception categories: allow more repeats
                if count <= 2:
                    saturation[cat] = 0.0
                else:
                    saturation[cat] = 0.3
            else:
                if count <= 1:
                    saturation[cat] = 0.0
                elif count == 2:
                    saturation[cat] = 0.5
                else:
                    saturation[cat] = 0.8

        return saturation

    def compute_sequence_score(self, candidate_category: str,
                                current_sequence: List[str],
                                visited_counts: Dict[str, int]) -> float:
        """
        Combined score: transition probability * (1 - saturation_penalty).

        A Cafe after Cultural->Cultural scores high; a second Restaurant
        in the same trajectory scores low.

        Args:
            candidate_category: Category of the candidate POI
            current_sequence: Category sequence so far in this trajectory
            visited_counts: Dict of category -> visit count in current trajectory

        Returns:
            Score between 0.0 and 1.0
        """
        # Get transition score
        next_scores = self.get_next_category_scores(current_sequence)
        transition_score = next_scores.get(candidate_category, 0.1)  # Small default

        # Get saturation penalty
        visited_list = []
        for cat, count in visited_counts.items():
            visited_list.extend([cat] * count)
        saturation = self.get_category_saturation(visited_list)
        sat_penalty = saturation.get(candidate_category, 0.0)

        return transition_score * (1.0 - sat_penalty)

    def save_patterns(self, output_dir: str = 'output/patterns') -> None:
        """Save mined patterns to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save transition matrix
        with open(output_path / 'category_transitions.json', 'w') as f:
            json.dump(self.transition_matrix, f, indent=2)

        # Save n-gram patterns (convert tuples to lists for JSON)
        ngram_data = [
            {'pattern': list(ngram), 'support': support}
            for ngram, support in self.ngram_patterns
        ]
        with open(output_path / 'ngram_patterns.json', 'w') as f:
            json.dump(ngram_data, f, indent=2)

        print(f"Saved patterns to {output_path}")

    def load_patterns(self, output_dir: str = 'output/patterns') -> None:
        """Load previously mined patterns from disk."""
        output_path = Path(output_dir)

        # Load transition matrix
        trans_path = output_path / 'category_transitions.json'
        if trans_path.exists():
            with open(trans_path) as f:
                self.transition_matrix = json.load(f)
            print(f"Loaded transition matrix: {len(self.transition_matrix)} source categories")

        # Load n-gram patterns
        ngram_path = output_path / 'ngram_patterns.json'
        if ngram_path.exists():
            with open(ngram_path) as f:
                ngram_data = json.load(f)
            self.ngram_patterns = [
                (tuple(item['pattern']), item['support'])
                for item in ngram_data
            ]
            print(f"Loaded {len(self.ngram_patterns)} n-gram patterns")
