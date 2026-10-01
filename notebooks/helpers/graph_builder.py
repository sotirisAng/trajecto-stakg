"""
Graph Builder Module for Trajecto Knowledge Graph Analytics

This module provides functions to extract subgraphs from the Trajecto RDF
knowledge graph stored in Apache Fuseki and construct NetworkX graphs
for various analytics tasks.

Graph Types:
- User-POI bipartite graph (weighted by visits/dwell time)
- POI co-visitation graph (POIs visited in same trajectory)
- User similarity graph (users with shared POI visits)
- Trajectory-POI bipartite graph
- Trajectory similarity graph (shared POIs)
- POI flow graph (directed transitions between POIs)
- Trajectory sequence graph (POI visit order)
"""

import networkx as nx
import pandas as pd
from typing import Dict, List, Any
from .fuseki_queries import execute_sparql_query, format_results
from .fuseki_config import query_endpoint

# SPARQL Prefixes used across queries
PREFIXES = """
PREFIX trajecto: <http://w3id.org/trajecto#>
PREFIX poi: <http://w3id.org/trajecto/poi#>
PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#>
PREFIX patterns: <http://w3id.org/trajecto/patterns#>
PREFIX purpose: <http://w3id.org/trajecto/purpose#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""


class TrajectoNXGraphBuilder:
    """
    Builds NetworkX graphs from the Trajecto RDF knowledge graph.

    Usage:
        builder = TrajectoNXGraphBuilder()
        user_poi_graph = builder.build_user_poi_graph()
        poi_covisit_graph = builder.build_poi_covisit_graph()
    """

    def __init__(self, endpoint: str = None):
        """
        Initialize the graph builder.

        Args:
            endpoint: SPARQL endpoint URL. Defaults to fuseki_config.query_endpoint
        """
        self.endpoint = endpoint or query_endpoint

    def _execute_query(self, query: str) -> pd.DataFrame:
        """Execute SPARQL query and return results as DataFrame."""
        full_query = PREFIXES + query
        results = execute_sparql_query(self.endpoint, full_query)
        return format_results(results)

    # ========== User & POI Graphs ==========

    def build_user_poi_graph(self, meaningful_only: bool = True) -> nx.Graph:
        """
        Build bipartite User-POI graph.

        Nodes:
            - Users (type='user')
            - POIs (type='poi')

        Edges:
            - User -> POI with weight = visit count, total dwell time

        Args:
            meaningful_only: If True, only include MeaningfulInteractions

        Returns:
            NetworkX Graph with user and POI nodes
        """
        interaction_filter = "?interaction a poi:MeaningfulInteraction ." if meaningful_only else ""

        query = f"""
        SELECT ?user ?poi (COUNT(?interaction) AS ?visits) (SUM(?dwell) AS ?totalDwell)
        WHERE {{
            ?user a trajecto:Person .
            ?user trajecto:formsTrajectory ?traj .
            ?pos dul:isPartOf ?traj .
            ?pos trajecto:hasInteraction ?interaction .
            ?interaction poi:interactsWith ?poi .
            ?interaction poi:hasDwellTimeInSeconds ?dwell .
            {interaction_filter}
        }}
        GROUP BY ?user ?poi
        """

        df = self._execute_query(query)

        G = nx.Graph()

        if df.empty:
            return G

        # Add nodes with types
        users = df['user'].unique()
        pois = df['poi'].unique()

        for user in users:
            G.add_node(user, type='user', bipartite=0)
        for poi in pois:
            G.add_node(poi, type='poi', bipartite=1)

        # Add edges
        for _, row in df.iterrows():
            visits = int(row['visits']) if pd.notna(row['visits']) else 1
            dwell = float(row['totalDwell']) if pd.notna(row['totalDwell']) else 0
            G.add_edge(row['user'], row['poi'],
                      visits=visits,
                      total_dwell=dwell,
                      weight=visits)

        return G

    def build_poi_covisit_graph(self, meaningful_only: bool = True) -> nx.Graph:
        """
        Build POI co-visitation graph.

        POIs are connected if visited in the same trajectory.
        Edge weight = number of trajectories where both POIs were visited.

        Args:
            meaningful_only: If True, only include MeaningfulInteractions

        Returns:
            NetworkX Graph with POI nodes and co-visitation edges
        """
        interaction_filter = """
            ?interaction1 a poi:MeaningfulInteraction .
            ?interaction2 a poi:MeaningfulInteraction .
        """ if meaningful_only else ""

        query = f"""
        SELECT ?poi1 ?poi2 (COUNT(DISTINCT ?traj) AS ?covisits)
        WHERE {{
            ?pos1 dul:isPartOf ?traj .
            ?pos2 dul:isPartOf ?traj .
            ?pos1 trajecto:hasInteraction ?interaction1 .
            ?pos2 trajecto:hasInteraction ?interaction2 .
            ?interaction1 poi:interactsWith ?poi1 .
            ?interaction2 poi:interactsWith ?poi2 .
            {interaction_filter}
            FILTER(STR(?poi1) < STR(?poi2))
        }}
        GROUP BY ?poi1 ?poi2
        """

        df = self._execute_query(query)

        G = nx.Graph()

        if df.empty:
            return G

        for _, row in df.iterrows():
            covisits = int(row['covisits']) if pd.notna(row['covisits']) else 1
            G.add_edge(row['poi1'], row['poi2'],
                      weight=covisits,
                      covisits=covisits)

        # Add node type
        for node in G.nodes():
            G.nodes[node]['type'] = 'poi'

        return G

    def build_user_similarity_graph(self, min_shared: int = 1) -> nx.Graph:
        """
        Build user similarity graph based on shared POI visits.

        Users are connected if they visited at least min_shared common POIs.
        Edge weight = number of shared POIs.

        Fetches user-POI pairs with a simple query and computes
        similarity in Python

        Args:
            min_shared: Minimum number of shared POIs to create an edge

        Returns:
            NetworkX Graph with user nodes
        """
        query = """
        SELECT DISTINCT ?user ?poi
        WHERE {
            ?user trajecto:formsTrajectory ?traj .
            ?pos dul:isPartOf ?traj .
            ?pos trajecto:hasInteraction ?int .
            ?int poi:interactsWith ?poi .
        }
        """

        df = self._execute_query(query)

        G = nx.Graph()

        if df.empty:
            return G

        # Build user -> set of POIs mapping
        user_pois = df.groupby('user')['poi'].apply(set).to_dict()

        # Compute pairwise similarity in Python
        users = list(user_pois.keys())
        for i, user1 in enumerate(users):
            for user2 in users[i + 1:]:
                shared = len(user_pois[user1] & user_pois[user2])
                if shared >= min_shared:
                    G.add_edge(user1, user2, weight=shared, shared_pois=shared)

        # Add node type
        for node in G.nodes():
            G.nodes[node]['type'] = 'user'

        return G

    # ========== Trajectory Graphs ==========

    def build_trajectory_poi_graph(self, meaningful_only: bool = True) -> nx.Graph:
        """
        Build bipartite Trajectory-POI graph.

        Nodes:
            - Trajectories (type='trajectory')
            - POIs (type='poi')

        Edges:
            - Trajectory -> POI with weight = visit count, total dwell time

        Returns:
            NetworkX Graph with trajectory and POI nodes
        """
        interaction_filter = "?interaction a poi:MeaningfulInteraction ." if meaningful_only else ""

        query = f"""
        SELECT ?traj ?poi (COUNT(?interaction) AS ?visits) (SUM(?dwell) AS ?totalDwell)
        WHERE {{
            ?pos dul:isPartOf ?traj .
            ?pos trajecto:hasInteraction ?interaction .
            ?interaction poi:interactsWith ?poi .
            ?interaction poi:hasDwellTimeInSeconds ?dwell .
            {interaction_filter}
        }}
        GROUP BY ?traj ?poi
        """

        df = self._execute_query(query)

        G = nx.Graph()

        if df.empty:
            return G

        # Add nodes with types
        trajectories = df['traj'].unique()
        pois = df['poi'].unique()

        for traj in trajectories:
            G.add_node(traj, type='trajectory', bipartite=0)
        for poi in pois:
            G.add_node(poi, type='poi', bipartite=1)

        # Add edges
        for _, row in df.iterrows():
            visits = int(row['visits']) if pd.notna(row['visits']) else 1
            dwell = float(row['totalDwell']) if pd.notna(row['totalDwell']) else 0
            G.add_edge(row['traj'], row['poi'],
                      visits=visits,
                      total_dwell=dwell,
                      weight=visits)

        return G

    def build_trajectory_similarity_graph(self, min_shared: int = 1) -> nx.Graph:
        """
        Build trajectory similarity graph based on shared POI visits.

        Trajectories are connected if they share at least min_shared POIs.
        Edge weight = number of shared POIs.

        Args:
            min_shared: Minimum number of shared POIs to create an edge

        Returns:
            NetworkX Graph with trajectory nodes
        """
        # Get all trajectory-POI pairs (much faster than self-join in SPARQL)
        query = """
        SELECT DISTINCT ?traj ?poi
        WHERE {
            ?pos dul:isPartOf ?traj .
            ?pos trajecto:hasInteraction ?interaction .
            ?interaction poi:interactsWith ?poi .
            ?interaction a poi:MeaningfulInteraction .
        }
        """

        df = self._execute_query(query)

        G = nx.Graph()

        if df.empty:
            return G

        # Build trajectory -> set of POIs mapping
        traj_pois = df.groupby('traj')['poi'].apply(set).to_dict()

        # Compute pairwise similarity in Python (set intersection is fast)
        trajectories = list(traj_pois.keys())
        for i, traj1 in enumerate(trajectories):
            for traj2 in trajectories[i + 1:]:
                shared = len(traj_pois[traj1] & traj_pois[traj2])
                if shared >= min_shared:
                    G.add_edge(traj1, traj2, weight=shared, shared_pois=shared)

        # Add node type
        for node in G.nodes():
            G.nodes[node]['type'] = 'trajectory'

        return G

    def build_poi_flow_graph(self) -> nx.DiGraph:
        """
        Build directed POI-to-POI flow graph.

        Aggregates all POI transitions across trajectories.
        Edge weight = number of times this transition was observed.

        POI sequence is determined by ordering positions by timestamp
        within each trajectory.

        Returns:
            Directed NetworkX Graph with POI nodes and transition edges
        """

        # PREFIX
        # poi: < http: // w3id.org / trajecto / poi  # >
        # PREFIX
        # trajecto: < http: // w3id.org / trajecto  # >
        # PREFIX
        # rdf: < http: // www.w3.org / 1999 / 02 / 22 - rdf - syntax - ns  # >
        # PREFIX
        # xsd: < http: // www.w3.org / 2001 / XMLSchema  # >
        # PREFIX
        # rdfs: < http: // www.w3.org / 2000 / 01 / rdf - schema  # >
        # PREFIX
        # dul: < http: // www.ontologydesignpatterns.org / ont / dul / DUL.owl  # >
        #
        # SELECT ?poi1 ?poi2
        # WHERE
        # {
        # ?pos1
        # dul: isPartOf ?trajectory.
        # ?pos2
        # dul: isPartOf ?trajectory.
        #
        # ?pos1
        # trajecto: hasInteraction ?int1.
        # ?int1
        # a
        # poi: MeaningfulInteraction.
        # ?int1
        # poi: interactsWith ?poi1.
        #
        # ?pos2
        # trajecto: hasInteraction ?int2.
        # ?int2
        # poi: interactsWith ?poi2.
        # ?int2
        # a
        # poi: MeaningfulInteraction.
        #
        # BIND(STRAFTER(STR(?pos1), "rawPosition_") AS ?suffix1)
        # BIND(STRAFTER(?suffix1, "_") AS ?idx1Str)
        # BIND(xsd: integer(?idx1Str) AS ?idx1)
        #
        # BIND(STRAFTER(STR(?pos2), "rawPosition_") AS ?suffix2)
        # BIND(STRAFTER(?suffix2, "_") AS ?idx2Str)
        # BIND(xsd: integer(?idx2Str) AS ?idx2)
        #
        # FILTER(?idx2 > ?idx1)
        # FILTER(?poi1 != ?poi2)  # If you want different POIs
        #
        # FILTER
        # NOT
        # EXISTS
        # {
        # ?pos3
        # dul: isPartOf ?trajectory.  # Same trajectory!
        # ?pos3
        # trajecto: hasInteraction ?int3.
        # ?int3
        # poi: interactsWith ?poi3.
        # ?int3
        # a
        # poi: MeaningfulInteraction.
        #
        # # Compare using string operations directly in FILTER
        # FILTER(
        #     xsd:integer(STRAFTER(STRAFTER(STR(?pos3), "rawPosition_"), "_")) > ?idx1 & &
        #                                                                         xsd: integer(
        #     STRAFTER(STRAFTER(STR(?pos3), "rawPosition_"), "_")) < ?idx2
        # )
        # }
        # } limit
        # 100

        # Get all POIs with meaningful interactions per trajectory, ordered by timestamp
        query = """
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

        df = self._execute_query(query)

        G = nx.DiGraph()

        if df.empty:
            return G

        # Count transitions by iterating through each trajectory's POI sequence
        transition_counts = {}

        for traj in df['traj'].unique():
            traj_df = df[df['traj'] == traj].sort_values('timestamp')
            pois = traj_df['poi'].tolist()

            # Find consecutive POI pairs (transitions)
            for i in range(len(pois) - 1):
                poi1, poi2 = pois[i], pois[i + 1]
                if poi1 != poi2:  # Skip self-loops
                    edge = (poi1, poi2)
                    transition_counts[edge] = transition_counts.get(edge, 0) + 1

        # Build the graph from transition counts
        for (poi1, poi2), count in transition_counts.items():
            G.add_edge(poi1, poi2, weight=count, transitions=count)

        # Add node type
        for node in G.nodes():
            G.nodes[node]['type'] = 'poi'

        return G

    def build_trajectory_sequence_graph(self) -> nx.DiGraph:
        """
        Build directed graph of POI visit sequences within trajectories.

        Each edge represents a direct transition from one POI to another
        within a trajectory, preserving order determined by timestamp.

        Returns:
            Directed Graph with trajectory-specific POI transitions
        """
        # Get all POIs with meaningful interactions per trajectory, ordered by timestamp
        query = """
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

        df = self._execute_query(query)

        G = nx.DiGraph()

        if df.empty:
            return G

        # Build edges by iterating through each trajectory's POI sequence
        for traj in df['traj'].unique():
            traj_df = df[df['traj'] == traj].sort_values('timestamp')
            pois = traj_df['poi'].tolist()

            # Find consecutive POI pairs (transitions)
            for i in range(len(pois) - 1):
                poi1, poi2 = pois[i], pois[i + 1]
                if poi1 != poi2:  # Skip self-loops
                    if G.has_edge(poi1, poi2):
                        G[poi1][poi2]['weight'] += 1
                        G[poi1][poi2]['trajectories'].append(traj)
                    else:
                        G.add_edge(poi1, poi2, weight=1, trajectories=[traj])

        # Add node type
        for node in G.nodes():
            G.nodes[node]['type'] = 'poi'

        return G

    def build_user_trajectory_graph(self) -> nx.Graph:
        """
        Build bipartite User-Trajectory graph.

        Returns:
            NetworkX Graph with user and trajectory nodes
        """
        query = """
        SELECT ?user ?traj
        WHERE {
            ?user a trajecto:Person .
            ?user trajecto:formsTrajectory ?traj .
        }
        """

        df = self._execute_query(query)

        G = nx.Graph()

        if df.empty:
            return G

        for _, row in df.iterrows():
            user = row['user']
            traj = row['traj']

            if user not in G:
                G.add_node(user, type='user', bipartite=0)
            if traj not in G:
                G.add_node(traj, type='trajectory', bipartite=1)

            G.add_edge(user, traj)

        return G

    # ========== Metadata Extraction ==========

    def get_poi_metadata(self) -> pd.DataFrame:
        """
        Get POI metadata (labels, types, categories).

        Returns:
            DataFrame with POI metadata
        """
        query = """
        SELECT ?poi ?label ?type
        WHERE {
            ?poi a trajecto:PointOfInterest .
            OPTIONAL { ?poi rdfs:label ?label }
            OPTIONAL { ?poi a ?type . FILTER(?type != trajecto:PointOfInterest) }
        }
        """
        return self._execute_query(query)

    def get_trajectory_metadata(self) -> pd.DataFrame:
        """
        Get trajectory metadata (user, purpose, navigation style).

        Returns:
            DataFrame with trajectory metadata
        """
        query = """
        SELECT ?traj ?user ?purpose ?navStyle
        WHERE {
            ?user trajecto:formsTrajectory ?traj .
            OPTIONAL { ?traj trajecto:hasPurpose ?purpose }
            OPTIONAL { ?traj patterns:exhibitsPattern ?navStyle }
        }
        """
        return self._execute_query(query)

    def get_user_metadata(self) -> pd.DataFrame:
        """
        Get user metadata.

        Returns:
            DataFrame with user metadata
        """
        query = """
        SELECT ?user ?label (COUNT(DISTINCT ?traj) AS ?numTrajectories)
        WHERE {
            ?user a trajecto:Person .
            OPTIONAL { ?user rdfs:label ?label }
            OPTIONAL { ?user trajecto:formsTrajectory ?traj }
        }
        GROUP BY ?user ?label
        """
        return self._execute_query(query)

    def get_trajectory_poi_sequences(self) -> Dict[str, List[str]]:
        """
        Get POI visit sequences for each trajectory.

        Returns unique POI sequences, removing consecutive duplicates
        (multiple positions interacting with the same POI).

        Returns:
            Dictionary mapping trajectory IRI to list of POI IRIs in visit order
        """
        query = """
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

        df = self._execute_query(query)

        sequences = {}
        if df.empty:
            return sequences

        for traj in df['traj'].unique():
            traj_df = df[df['traj'] == traj].sort_values('timestamp')
            pois = traj_df['poi'].tolist()

            # Remove consecutive duplicates (keep only transitions to different POIs)
            unique_pois = [pois[0]] if pois else []
            for poi in pois[1:]:
                if poi != unique_pois[-1]:
                    unique_pois.append(poi)

            sequences[traj] = unique_pois

        return sequences

    # ========== Graph Statistics ==========

    def get_graph_statistics(self, graph: nx.Graph) -> Dict[str, Any]:
        """
        Compute basic statistics for a graph.

        Args:
            graph: NetworkX graph

        Returns:
            Dictionary with graph statistics
        """
        stats = {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'density': nx.density(graph),
            'is_connected': nx.is_connected(graph) if not graph.is_directed() else nx.is_weakly_connected(graph),
        }

        if stats['is_connected'] and graph.number_of_nodes() > 1:
            if not graph.is_directed():
                stats['diameter'] = nx.diameter(graph)
                stats['avg_shortest_path'] = nx.average_shortest_path_length(graph)
            stats['avg_clustering'] = nx.average_clustering(graph)

        if graph.number_of_nodes() > 0:
            degrees = [d for _, d in graph.degree()]
            stats['avg_degree'] = sum(degrees) / len(degrees)
            stats['max_degree'] = max(degrees)
            stats['min_degree'] = min(degrees)

        return stats


# Convenience functions for quick graph building
def build_all_graphs(endpoint: str = None) -> Dict[str, nx.Graph]:
    """
    Build all available graph types.

    Args:
        endpoint: Optional SPARQL endpoint URL

    Returns:
        Dictionary mapping graph name to NetworkX graph
    """
    builder = TrajectoNXGraphBuilder(endpoint)

    return {
        'user_poi': builder.build_user_poi_graph(),
        'poi_covisit': builder.build_poi_covisit_graph(),
        'user_similarity': builder.build_user_similarity_graph(),
        'trajectory_poi': builder.build_trajectory_poi_graph(),
        'trajectory_similarity': builder.build_trajectory_similarity_graph(),
        'poi_flow': builder.build_poi_flow_graph(),
        'user_trajectory': builder.build_user_trajectory_graph(),
    }


def print_graph_summary(graphs: Dict[str, nx.Graph]):
    """Print summary statistics for all graphs."""
    builder = TrajectoNXGraphBuilder()

    print("=" * 60)
    print("GRAPH SUMMARY")
    print("=" * 60)

    for name, graph in graphs.items():
        print(f"\n{name}:")
        stats = builder.get_graph_statistics(graph)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
