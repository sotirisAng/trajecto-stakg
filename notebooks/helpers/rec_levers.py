"""
Additional rankers and fusions for the recommender evaluation.

Rankers follow the evaluation contract:
    ranker(rec_engine, traj_uri, input_seq, top_k, [exclude_traj=...])
so they plug into baseline_evaluation.evaluate_method and the
eval_protocols runners unchanged.

Levers:
  markov2_rec         — recency-weighted second-order flow ranker
  two_stage           — retrieve-then-rank candidate generation
                        (graph-connected ∪ popularity head ∪ radius)
  rrf                 — reciprocal-rank fusion of component rankers
  repeat_rec          — rank the user's own previously visited POIs
                        (for the repeats-allowed session-transfer protocol)
  LRFusion            — logistic regression over per-candidate
                        signal vectors, trained on a tune split only


Ranking note: every sort below breaks ties on the POI URI as well as the
score. Candidates are drawn from a set, so a score-only sort resolves ties by
set iteration order, which varies between processes, so two identical runs
could otherwise differ.
"""

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .baselines import (
    _candidate_filter, _haversine, _pack, _popularity, popularity_rec,
)


# ── 1.4 Second-order / recency-weighted flow ──────────────────────────

def markov2_scores(rec, input_seq: List[str], exclude: str, candidates: set,
                   w_last: float = 0.7, w_prev: float = 0.3,
                   w_chain: float = 0.3) -> Dict[str, float]:
    """Markov-2 scores over the candidates (unnormalized):

    score(dst) = w_last·W(last→dst) + w_prev·W(prev→dst)
               + w_chain·Σ_mid Ŵ(last→mid)·Ŵ(mid→dst)
    with all edge weights leak-free via _flow_weight(exclude_traj=...).
    """
    scores = defaultdict(float)
    if not input_seq or rec.flow_graph is None:
        return scores

    def add_from(src, weight):
        if not rec.flow_graph.has_node(src):
            return
        out = []
        for _, dst in rec.flow_graph.out_edges(src):
            w = rec._flow_weight(src, dst, exclude_traj=exclude)
            if w > 0:
                out.append((dst, float(w)))
        total = sum(w for _, w in out)
        if total <= 0:
            return
        for dst, w in out:
            if dst in candidates:
                scores[dst] += weight * w / total

    last = input_seq[-1]
    add_from(last, w_last)
    if len(input_seq) >= 2:
        add_from(input_seq[-2], w_prev)

    # 2-hop chains from the last POI
    if w_chain > 0 and rec.flow_graph.has_node(last):
        first = []
        for _, mid in rec.flow_graph.out_edges(last):
            w = rec._flow_weight(last, mid, exclude_traj=exclude)
            if w > 0:
                first.append((mid, float(w)))
        t1 = sum(w for _, w in first)
        for mid, w1 in first:
            if t1 <= 0 or not rec.flow_graph.has_node(mid):
                continue
            second = []
            for _, dst in rec.flow_graph.out_edges(mid):
                w2 = rec._flow_weight(mid, dst, exclude_traj=exclude)
                if w2 > 0:
                    second.append((dst, float(w2)))
            t2 = sum(w for _, w in second)
            for dst, w2 in second:
                if t2 > 0 and dst in candidates:
                    scores[dst] += w_chain * (w1 / t1) * (w2 / t2)
    return scores


def markov2_rec(rec, traj_uri: str, input_seq: List[str], top_k: int,
                exclude_traj: Optional[str] = None,
                w_last: float = 0.7, w_prev: float = 0.3,
                w_chain: float = 0.3, restrict_to: set = None) -> List[Dict]:
    """Flow ranker over the last TWO observed POIs plus 2-hop chains
    (markov2_scores). Backs off to leak-free popularity when no flow
    evidence remains."""
    exclude = exclude_traj or traj_uri
    visited = set(input_seq)
    candidates = _candidate_filter(rec, visited, restrict_to)
    scores = markov2_scores(rec, input_seq, exclude, candidates,
                            w_last, w_prev, w_chain)
    if not scores:
        return popularity_rec(rec, traj_uri, input_seq, top_k,
                              exclude_traj=exclude, restrict_to=restrict_to)
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    total = sum(scores.values())
    return [_pack(rec, p, s / total, 'markov2')
            for p, s in ranked[:top_k]]


# ── 1.3 Two-stage retrieve-then-rank ──────────────────────────────────

def two_stage_candidates(rec, input_seq: List[str],
                         exclude_traj: Optional[str] = None,
                         pop_head: int = 100, radius_m: float = 1000.0,
                         hops: int = 2) -> set:
    """Union candidate generator: graph-connected ∪ popularity head ∪
    radius around the prefix centroid. Visited and excluded-category POIs
    are already dropped by _candidate_filter."""
    visited = set(input_seq)
    base = _candidate_filter(rec, visited)
    cand = set()

    # graph-connected (flow successors + covisit neighbors, up to `hops`)
    frontier = set(input_seq)
    for _ in range(hops):
        nxt = set()
        for src in frontier:
            if rec.flow_graph is not None and rec.flow_graph.has_node(src):
                for _, dst in rec.flow_graph.out_edges(src):
                    if rec._flow_weight(src, dst,
                                        exclude_traj=exclude_traj) > 0:
                        nxt.add(dst)
            if rec.covisit_graph is not None \
                    and rec.covisit_graph.has_node(src):
                for nb in rec.covisit_graph.neighbors(src):
                    if rec._covisit_weight(src, nb,
                                           exclude_traj=exclude_traj) > 0:
                        nxt.add(nb)
        cand |= nxt & base
        frontier = nxt

    # popularity head (leak-free)
    ranked = sorted(((p, _popularity(rec, p, exclude_traj)) for p in base),
                    key=lambda x: (-x[1], x[0]))
    cand |= {p for p, _ in ranked[:pop_head]}

    # radius around prefix centroid
    coords = [(rec.poi_metadata[p]['lat'], rec.poi_metadata[p]['lon'])
              for p in input_seq
              if rec.poi_metadata.get(p, {}).get('lat') is not None]
    if coords:
        clat = sum(c[0] for c in coords) / len(coords)
        clon = sum(c[1] for c in coords) / len(coords)
        for p in base:
            m = rec.poi_metadata.get(p, {})
            if m.get('lat') is None:
                continue
            if _haversine(clat, clon, m['lat'], m['lon']) <= radius_m:
                cand.add(p)
    return cand


def two_stage(base_ranker: Callable, pop_head: int = 100,
              radius_m: float = 1000.0, hops: int = 2) -> Callable:
    """Wrap a ranker: full ranking filtered (order-preserving) to the
    two-stage candidate set."""
    def ranker(rec, traj_uri, input_seq, top_k, exclude_traj=None):
        exclude = exclude_traj or traj_uri
        cand = two_stage_candidates(rec, input_seq, exclude_traj=exclude,
                                    pop_head=pop_head, radius_m=radius_m,
                                    hops=hops)
        pool = len(rec.all_poi_uris)
        try:
            recs = base_ranker(rec, traj_uri, input_seq, pool,
                               exclude_traj=exclude)
        except TypeError:
            recs = base_ranker(rec, traj_uri, input_seq, pool)
        return [r for r in recs if r['poi'] in cand][:top_k]
    return ranker


# ── Reciprocal-rank fusion ────────────────────────────────────────────

def rrf(components: Sequence[Callable], k_rrf: int = 60,
        weights: Optional[Sequence[float]] = None) -> Callable:
    def ranker(rec, traj_uri, input_seq, top_k, exclude_traj=None,
               restrict_to=None):
        pool = len(rec.all_poi_uris)
        scores, meta = {}, {}
        for i, comp in enumerate(components):
            w = 1.0 if weights is None else weights[i]
            kw = {}
            if exclude_traj is not None:
                kw['exclude_traj'] = exclude_traj
            if restrict_to is not None:
                kw['restrict_to'] = restrict_to
            try:
                recs = comp(rec, traj_uri, input_seq, pool, **kw)
            except TypeError:
                kw.pop('restrict_to', None)
                try:
                    recs = comp(rec, traj_uri, input_seq, pool, **kw)
                except TypeError:
                    recs = comp(rec, traj_uri, input_seq, pool)
            for rank, r in enumerate(recs, 1):
                scores[r['poi']] = scores.get(r['poi'], 0.0) \
                    + w / (k_rrf + rank)
                meta.setdefault(r['poi'], r)
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return [dict(meta[p], score=s) for p, s in ranked[:top_k]]
    return ranker


# ── 1.5 Repeat ranker (session transfer with repeats) ─────────────────

def repeat_rec(rec, traj_uri: str, input_seq: List[str],
               top_k: int, exclude_traj: Optional[str] = None,
               recency_halflife: int = 10) -> List[Dict]:
    """Rank the POIs the user has already visited, by frequency with a
    recency boost (later positions in the observed sequence weigh more).
    Only meaningful under protocols whose holdout allows repeats."""
    if not input_seq:
        return []
    n = len(input_seq)
    scores = defaultdict(float)
    for idx, p in enumerate(input_seq):
        scores[p] += 2.0 ** (-(n - 1 - idx) / recency_halflife)
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    total = sum(scores.values())
    return [_pack(rec, p, s / total, 'repeat')
            for p, s in ranked[:top_k]]


# ── 1.6 Learned fusion (logistic regression) ──────────────────────────

class LRFusion:
    """Logistic regression over per-candidate signal vectors.

    Features per candidate: the unified signal dict (raw values), plus
    leak-free popularity, markov2 probability, and inverse distance to
    the prefix centroid. Trained on tune-half trajectories only; the
    trained model is a plug-in ranker for the test half.
    """

    def __init__(self, rec, unified_ranker: Callable,
                 signal_names: Sequence[str]):
        self.rec = rec
        self.unified_ranker = unified_ranker
        self.signal_names = list(signal_names)
        self.feature_names = self.signal_names + ['pop', 'markov2', 'prox']
        self.coef = None

    def _features(self, traj_uri, input_seq, exclude_traj):
        rec = self.rec
        pool = len(rec.all_poi_uris)
        recs = self.unified_ranker(rec, traj_uri, input_seq, pool,
                                   exclude_traj=exclude_traj)
        m2 = {r['poi']: r['score'] for r in
              markov2_rec(rec, traj_uri, input_seq, pool,
                          exclude_traj=exclude_traj)}
        max_pop = max(rec._poi_popularity.values()) or 1
        coords = [(rec.poi_metadata[p]['lat'], rec.poi_metadata[p]['lon'])
                  for p in input_seq
                  if rec.poi_metadata.get(p, {}).get('lat') is not None]
        clat = sum(c[0] for c in coords) / len(coords) if coords else None
        clon = sum(c[1] for c in coords) / len(coords) if coords else None

        pois, X = [], []
        for r in recs:
            sig = r.get('signals') or {}
            row = [float(sig.get(s, 0.0) or 0.0) for s in self.signal_names]
            row.append(_popularity(rec, r['poi'], exclude_traj) / max_pop)
            row.append(m2.get(r['poi'], 0.0))
            meta = rec.poi_metadata.get(r['poi'], {})
            if clat is not None and meta.get('lat') is not None:
                d = _haversine(clat, clon, meta['lat'], meta['lon'])
                row.append(1.0 / (1.0 + d / 500.0))
            else:
                row.append(0.0)
            pois.append(r['poi'])
            X.append(row)
        return pois, np.array(X, dtype=np.float64)

    def fit(self, trajectories, split_fn, neg_per_traj: int = 40,
            l2: float = 1.0, epochs: int = 300, lr: float = 0.5):
        Xs, ys = [], []
        for t in trajectories:
            input_seq, hold = split_fn(t)
            if not hold:
                continue
            hold_set = set(hold)
            pois, X = self._features(t, input_seq, exclude_traj=t)
            pos = [i for i, p in enumerate(pois) if p in hold_set]
            if not pos:
                continue
            neg = [i for i, p in enumerate(pois) if p not in hold_set]
            neg = neg[:neg_per_traj]           # hard negatives: top-ranked
            for i in pos:
                Xs.append(X[i]); ys.append(1.0)
            for i in neg:
                Xs.append(X[i]); ys.append(0.0)
        X = np.array(Xs); y = np.array(ys)
        self.mu = X.mean(axis=0); self.sd = X.std(axis=0) + 1e-9
        Xn = (X - self.mu) / self.sd
        n, d = Xn.shape
        w = np.zeros(d); b = 0.0
        pos_w = (y == 0).sum() / max((y == 1).sum(), 1)
        sw = np.where(y == 1, pos_w, 1.0)
        for _ in range(epochs):
            z = Xn @ w + b
            p = 1.0 / (1.0 + np.exp(-z))
            g = (p - y) * sw
            w -= lr * (Xn.T @ g / n + l2 * w / n)
            b -= lr * g.mean()
        self.coef, self.intercept = w, b
        return self

    def ranker(self):
        def _rank(rec, traj_uri, input_seq, top_k, exclude_traj=None):
            pois, X = self._features(traj_uri, input_seq,
                                     exclude_traj=exclude_traj or traj_uri)
            if len(pois) == 0:
                return []
            z = ((X - self.mu) / self.sd) @ self.coef + self.intercept
            order = np.argsort(-z)
            out = []
            for i in order[:top_k]:
                out.append(_pack(rec, pois[i], float(z[i]), 'lr_fusion'))
            return out
        return _rank


# ── Cold-start rankers ─────────────────────────────────────

def tagpop_rec(rec, traj_uri: str, input_seq: List[str], top_k: int,
               exclude_traj: Optional[str] = None) -> List[Dict]:
    """Tag-conditioned popularity: rank candidates by visit count among
    trajectories sharing a purpose with the target, backing off to
    leak-free global popularity. Needs no user history — the cold-start
    content prior."""
    exclude = exclude_traj or traj_uri
    visited = set(input_seq)
    candidates = _candidate_filter(rec, visited)
    purposes = rec.traj_metadata.get(traj_uri, {}).get('purposes', set())

    counts = defaultdict(float)
    if purposes:
        for ot, meta in rec.traj_metadata.items():
            if ot == exclude:
                continue
            if meta.get('purposes', set()) & purposes:
                for p in rec.traj_poi_sets.get(ot, set()):
                    if p in candidates:
                        counts[p] += 1.0
    max_g = max(rec._poi_popularity.values()) or 1
    for p in candidates:
        counts[p] += 0.3 * _popularity(rec, p, exclude) / max_g
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    mx = ranked[0][1] if ranked else 1.0
    return [_pack(rec, p, s / max(mx, 1e-9), 'tagpop')
            for p, s in ranked[:top_k]]


def _load_pagerank(rec) -> Dict[str, float]:
    if getattr(rec, '_kg_pagerank', None) is None:
        import requests as _rq
        from .fuseki_config import query_endpoint as _qe
        q = ('PREFIX trajecto: <http://w3id.org/trajecto#> '
             'SELECT ?p ?pr WHERE { ?p trajecto:pageRank ?pr }')
        try:
            r = _rq.post(_qe, data={'query': q},
                         headers={'Accept': 'application/json'}, timeout=60)
            rec._kg_pagerank = {
                b['p']['value']: float(b['pr']['value'])
                for b in r.json()['results']['bindings']}
        except Exception:
            rec._kg_pagerank = {}
    return rec._kg_pagerank


def centrality_rec(rec, traj_uri: str, input_seq: List[str], top_k: int,
                   exclude_traj: Optional[str] = None) -> List[Dict]:
    """Rank by materialized flow PageRank (kg_analytics, NB21) — a
    user-independent structural prior. Backs off to leak-free
    popularity when no centrality is materialized."""
    pr = _load_pagerank(rec)
    visited = set(input_seq)
    candidates = _candidate_filter(rec, visited)
    if not pr:
        return popularity_rec(rec, traj_uri, input_seq, top_k,
                              exclude_traj=exclude_traj or traj_uri)
    ranked = sorted(((p, pr.get(p, 0.0)) for p in candidates),
                    key=lambda x: (-x[1], x[0]))
    mx = ranked[0][1] if ranked and ranked[0][1] > 0 else 1.0
    return [_pack(rec, p, s / mx, 'centrality')
            for p, s in ranked[:top_k]]
