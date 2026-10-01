"""
KG-native analytics (notebook 21).

Four analytics families that exist because the data is a knowledge graph:

  1. POI centrality & structural roles over the co-visitation and flow
     projections, with materialization back to the KG
     (trajecto:pageRank, trajecto:betweennessCentrality,
      trajecto:communityId, trajecto:hasCentralityRole) so SPARQL and the
     recommender's explanation layer can reference structural roles.
  2. Multi-hop relatedness: path-length distribution over the merged
     covisit+flow graph and explicit triple chains explaining how two
     POIs relate.
  3. Class-level aggregation: subsumption-aware contingency tables
     (purpose × interaction type, navigation × temporal context,
      category × engagement) with chi-square association tests.
  4. Temporal-sliced centrality: the same projections filtered per
     temporal context; centrality shift across slices and community
     stability (NMI).

All graphs are built from ONE interaction extraction (meaningful
interactions with timestamps) plus small metadata queries, so a notebook
can run every family without an engine load. Community detection uses
greedy modularity (networkx 2.6; swap to Louvain on nx >= 3).
"""

import itertools
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd
import requests
from scipy.stats import chi2_contingency

from .fuseki_config import query_endpoint, update_endpoint

PREFIXES = """
PREFIX trajecto: <http://w3id.org/trajecto#>
PREFIX poi: <http://w3id.org/trajecto/poi#>
PREFIX patterns: <http://w3id.org/trajecto/patterns#>
PREFIX context: <http://w3id.org/trajecto/context#>
PREFIX purpose: <http://w3id.org/trajecto/purpose#>
PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""

GENERIC_TYPES = {'PointOfInterest', 'SignificantPOI', 'NamedIndividual'}

MATERIALIZED_ANALYTICS_PREDICATES = [
    'http://w3id.org/trajecto#pageRank',
    'http://w3id.org/trajecto#betweennessCentrality',
    'http://w3id.org/trajecto#communityId',
    'http://w3id.org/trajecto#hasCentralityRole',
]


def _query(sparql: str) -> List[Dict]:
    r = requests.post(query_endpoint, data={'query': PREFIXES + sparql},
                      headers={'Accept': 'application/json'}, timeout=300)
    r.raise_for_status()
    return [{k: v['value'] for k, v in b.items()}
            for b in r.json()['results']['bindings']]


def _update(sparql: str) -> bool:
    r = requests.post(update_endpoint, data={'update': PREFIXES + sparql},
                      timeout=300)
    return r.status_code in (200, 204)


# ── Shared extraction ─────────────────────────────────────────────────

def load_interaction_records() -> pd.DataFrame:
    """One row per meaningful interaction: traj, poi, ts."""
    rows = _query("""
    SELECT ?traj ?poi ?ts WHERE {
        ?pos dul:isPartOf ?traj .
        ?pos trajecto:hasInteraction ?i .
        ?i a poi:MeaningfulInteraction ;
           poi:interactsWith ?poi .
        OPTIONAL { ?pos trajecto:timestamp ?ts }
    }""")
    return pd.DataFrame(rows)


def load_poi_metadata() -> Dict[str, Dict]:
    rows = _query("""
    SELECT DISTINCT ?poi ?cat ?label WHERE {
        ?i poi:interactsWith ?poi .
        OPTIONAL { ?poi a ?cat . FILTER(!isBlank(?cat)) }
        OPTIONAL { ?poi <http://www.w3.org/2000/01/rdf-schema#label> ?label }
    }""")
    meta = {}
    for r in rows:
        m = meta.setdefault(r['poi'], {'categories': set(), 'label': None})
        cat = r.get('cat', '').split('#')[-1]
        if cat and cat not in GENERIC_TYPES:
            m['categories'].add(cat)
        if r.get('label'):
            m['label'] = r['label']
    for poi, m in meta.items():
        m['category'] = sorted(m['categories'])[0] if m['categories'] else ''
        m['label'] = m['label'] or poi.split('_')[-1]
    return meta


def load_temporal_contexts() -> Dict[str, set]:
    rows = _query("""
    SELECT ?traj ?ctx WHERE { ?traj context:hasTemporalContext ?ctx }""")
    out = defaultdict(set)
    for r in rows:
        out[r['traj']].add(r['ctx'].split('#')[-1])
    return dict(out)


# ── Graph construction (from the shared extraction) ───────────────────

def build_graphs(records: pd.DataFrame,
                 trajs: Optional[set] = None
                 ) -> Tuple[nx.Graph, nx.DiGraph]:
    """Co-visitation (undirected, weight = #trajectories visiting both)
    and flow (directed, weight = #observed consecutive transitions)."""
    covisit = nx.Graph()
    flow = nx.DiGraph()
    for traj, grp in records.groupby('traj'):
        if trajs is not None and traj not in trajs:
            continue
        pois = list(dict.fromkeys(
            grp.sort_values('ts', na_position='last')['poi']))
        for a, b in itertools.combinations(sorted(set(pois)), 2):
            w = covisit.get_edge_data(a, b, {}).get('weight', 0)
            covisit.add_edge(a, b, weight=w + 1)
        for a, b in zip(pois, pois[1:]):
            if a == b:
                continue
            w = flow.get_edge_data(a, b, {}).get('weight', 0)
            flow.add_edge(a, b, weight=w + 1)
    return covisit, flow


# ── 1. Centrality & structural roles ──────────────────────────────────

def centrality_analysis(covisit: nx.Graph, flow: nx.DiGraph,
                        meta: Dict[str, Dict]) -> pd.DataFrame:
    deg = dict(covisit.degree(weight='weight'))
    btw = nx.betweenness_centrality(covisit, weight='weight')
    try:
        eig = nx.eigenvector_centrality(covisit, weight='weight',
                                        max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eig = {n: 0.0 for n in covisit}
    pr = nx.pagerank(flow, weight='weight') if flow.number_of_nodes() \
        else {}
    win = dict(flow.in_degree(weight='weight'))
    wout = dict(flow.out_degree(weight='weight'))

    communities = list(nx.algorithms.community.greedy_modularity_communities(
        covisit, weight='weight')) if covisit.number_of_nodes() else []
    comm_of = {}
    for cid, nodes in enumerate(communities):
        for n in nodes:
            comm_of[n] = cid

    pois = sorted(set(covisit) | set(flow))
    rows = []
    for p in pois:
        rows.append({
            'poi': p,
            'label': meta.get(p, {}).get('label', p),
            'category': meta.get(p, {}).get('category', ''),
            'covisit_degree': deg.get(p, 0),
            'betweenness': btw.get(p, 0.0),
            'eigenvector': eig.get(p, 0.0),
            'pagerank': pr.get(p, 0.0),
            'flow_in': win.get(p, 0),
            'flow_out': wout.get(p, 0),
            'community': comm_of.get(p, -1),
        })
    df = pd.DataFrame(rows)

    # Structural roles via corpus quantiles (kg-analytics-design table)
    q = {c: df[c].quantile([0.5, 0.8]).tolist()
         for c in ('covisit_degree', 'betweenness', 'pagerank')}

    def role(r):
        hi_deg = r['covisit_degree'] >= q['covisit_degree'][1]
        hi_btw = r['betweenness'] >= q['betweenness'][1]
        hi_pr = r['pagerank'] >= q['pagerank'][1]
        io = (r['flow_in'] + 1) / (r['flow_out'] + 1)
        if hi_deg and hi_pr:
            return 'Hub'
        if hi_btw:
            return 'Gateway'
        if hi_pr and io > 1.5:
            return 'Destination'
        if r['flow_out'] > 0 and io < 0.67:
            return 'Origin'
        return 'Peripheral'

    df['role'] = df.apply(role, axis=1)
    return df.sort_values('pagerank', ascending=False)


def materialize_centrality(df: pd.DataFrame, batch: int = 200,
                           top_n: Optional[int] = None) -> int:
    """INSERT the centrality annotations; returns triples written.
    Predicates are covered by MATERIALIZED_ANALYTICS_PREDICATES (also
    added to semantic_reasoning.MATERIALIZED_PREDICATES for cleanup)."""
    # replace previous run
    for pred in MATERIALIZED_ANALYTICS_PREDICATES:
        _update(f'DELETE {{ ?s <{pred}> ?o }} WHERE {{ ?s <{pred}> ?o }}')
    rows = df if top_n is None else df.head(top_n)
    triples, written = [], 0
    for _, r in rows.iterrows():
        s = f'<{r.poi}>'
        triples += [
            f'{s} trajecto:pageRank "{r.pagerank:.6f}"^^xsd:float .',
            f'{s} trajecto:betweennessCentrality '
            f'"{r.betweenness:.6f}"^^xsd:float .',
            f'{s} trajecto:communityId "{int(r.community)}"^^xsd:integer .',
            f'{s} trajecto:hasCentralityRole trajecto:{r.role}POI .',
        ]
        if len(triples) >= batch * 4:
            if _update('INSERT DATA { ' + '\n'.join(triples) + ' }'):
                written += len(triples)
            triples = []
    if triples and _update('INSERT DATA { ' + '\n'.join(triples) + ' }'):
        written += len(triples)
    return written


# ── 2. Multi-hop relatedness ──────────────────────────────────────────

def path_length_distribution(covisit: nx.Graph, flow: nx.DiGraph
                             ) -> pd.Series:
    merged = nx.Graph()
    merged.add_edges_from(covisit.edges())
    merged.add_edges_from((u, v) for u, v in flow.edges())
    if not merged.number_of_nodes():
        return pd.Series(dtype=int)
    giant = merged.subgraph(max(nx.connected_components(merged), key=len))
    lengths = Counter()
    for _, dists in nx.all_pairs_shortest_path_length(giant, cutoff=6):
        lengths.update(dists.values())
    lengths.pop(0, None)
    return pd.Series(lengths).sort_index()


def explain_relation(poi_a: str, poi_b: str, covisit: nx.Graph,
                     flow: nx.DiGraph, meta: Dict[str, Dict]
                     ) -> Optional[List[str]]:
    """Shortest evidence chain between two POIs as readable triples."""
    merged = nx.Graph()
    for u, v, d in covisit.edges(data=True):
        merged.add_edge(u, v, kind='poi:covisitedWith', w=d['weight'])
    for u, v, d in flow.edges(data=True):
        if not merged.has_edge(u, v):
            merged.add_edge(u, v, kind='poi:followedBy', w=d['weight'])
    if poi_a not in merged or poi_b not in merged:
        return None
    try:
        path = nx.shortest_path(merged, poi_a, poi_b)
    except nx.NetworkXNoPath:
        return None
    chain = []
    for u, v in zip(path, path[1:]):
        e = merged[u][v]
        lu = meta.get(u, {}).get('label', u)
        lv = meta.get(v, {}).get('label', v)
        chain.append(f'{lu} —{e["kind"]} (w={e["w"]})→ {lv}')
    return chain


# ── 3. Class-level aggregation ────────────────────────────────────────

CLASS_QUERIES = {
    'purpose_x_interaction': """
        SELECT ?a ?b (COUNT(?i) AS ?n) WHERE {
            ?traj trajecto:hasPurpose ?ap .
            ?pos dul:isPartOf ?traj .
            ?pos trajecto:hasInteraction ?i .
            ?i a ?bt .
            FILTER(?bt IN (poi:PassingInteraction,
                           poi:MeaningfulInteraction,
                           poi:ImmersiveInteraction))
            BIND(STRAFTER(STR(?ap), "#") AS ?a)
            BIND(STRAFTER(STR(?bt), "#") AS ?b)
        } GROUP BY ?a ?b""",
    'navstyle_x_temporal': """
        SELECT ?a ?b (COUNT(DISTINCT ?traj) AS ?n) WHERE {
            ?traj patterns:exhibitsPattern ?ap .
            ?traj context:hasTemporalContext ?bt .
            BIND(STRAFTER(STR(?ap), "#") AS ?a)
            BIND(STRAFTER(STR(?bt), "#") AS ?b)
        } GROUP BY ?a ?b""",
    'engagement_x_purpose': """
        SELECT ?a ?b (COUNT(DISTINCT ?traj) AS ?n) WHERE {
            ?traj trajecto:hasEngagementLevel ?ap .
            ?traj trajecto:hasPurpose ?bt .
            BIND(STRAFTER(STR(?ap), "#") AS ?a)
            BIND(STRAFTER(STR(?bt), "#") AS ?b)
        } GROUP BY ?a ?b""",
}


def class_level_tables() -> Dict[str, Dict]:
    """Contingency tables + chi-square for the three class pairs."""
    out = {}
    for name, q in CLASS_QUERIES.items():
        rows = _query(q)
        if not rows:
            out[name] = {'table': pd.DataFrame(), 'chi2': None, 'p': None}
            continue
        df = pd.DataFrame(rows)
        df['n'] = df['n'].astype(int)
        table = df.pivot_table(index='a', columns='b', values='n',
                               fill_value=0, aggfunc='sum')
        chi2, p = None, None
        if table.shape[0] > 1 and table.shape[1] > 1:
            chi2, p, _, _ = chi2_contingency(table.values)
        out[name] = {'table': table, 'chi2': chi2, 'p': p}
    return out


# ── 4. Temporal-sliced centrality ─────────────────────────────────────

TEMPORAL_SLICES = ['Morning', 'Afternoon', 'Evening', 'Night',
                   'Weekday', 'Weekend']


def temporal_centrality(records: pd.DataFrame,
                        traj_contexts: Dict[str, set],
                        meta: Dict[str, Dict],
                        slices: List[str] = None,
                        min_pois: int = 5) -> Tuple[pd.DataFrame, Dict]:
    """PageRank per temporal slice + cross-slice shift; community NMI
    between slice partitions (shared POIs only)."""
    from sklearn.metrics import normalized_mutual_info_score

    slices = slices or TEMPORAL_SLICES
    pr_by_slice, comm_by_slice = {}, {}
    for sl in slices:
        trajs = {t for t, ctx in traj_contexts.items() if sl in ctx}
        cov, flo = build_graphs(records, trajs=trajs)
        if flo.number_of_nodes() < min_pois:
            continue
        pr_by_slice[sl] = nx.pagerank(flo, weight='weight')
        if cov.number_of_nodes() >= min_pois:
            comms = nx.algorithms.community.greedy_modularity_communities(
                cov, weight='weight')
            comm_by_slice[sl] = {n: cid for cid, nodes in enumerate(comms)
                                 for n in nodes}

    pois = sorted({p for pr in pr_by_slice.values() for p in pr})
    rows = []
    for p in pois:
        vals = {sl: pr_by_slice[sl].get(p, 0.0) for sl in pr_by_slice}
        series = pd.Series(vals)
        mean = series.mean()
        rows.append({'poi': p,
                     'label': meta.get(p, {}).get('label', p),
                     'category': meta.get(p, {}).get('category', ''),
                     **vals,
                     'mean_pagerank': mean,
                     'cv': series.std() / mean if mean > 0 else 0.0})
    # A corpus whose timestamps are relative offsets rather than wall-clock
    # times has no temporal slices, so there is nothing to compare across them.
    # Return an empty frame with the expected columns instead of raising -- the
    # capability legitimately does not transfer to that data, and the rest of
    # the analytics is unaffected.
    if not rows:
        return {'shift': pd.DataFrame(columns=['poi', 'label', 'category',
                                               'mean_pagerank', 'cv']),
                'nmi': {}, 'slices': list(pr_by_slice)}
    shift = pd.DataFrame(rows).sort_values('mean_pagerank', ascending=False)

    nmi = {}
    names = list(comm_by_slice)
    for a, b in itertools.combinations(names, 2):
        shared = sorted(set(comm_by_slice[a]) & set(comm_by_slice[b]))
        if len(shared) >= 10:
            nmi[f'{a}~{b}'] = float(normalized_mutual_info_score(
                [comm_by_slice[a][p] for p in shared],
                [comm_by_slice[b][p] for p in shared]))
    return shift, nmi
