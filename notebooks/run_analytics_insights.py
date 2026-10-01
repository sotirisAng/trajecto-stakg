"""Turn the analytics layer from described capability into reported findings.

Section 7.8 of the chapter claimed three analyses it never reported (behavioural
analogues, archetype clustering, cross-dataset comparison) and a preference-drift
study it never ran. This produces all of them, plus the tag cross-tabulations
that were computed by the analytics notebooks and left in CSVs.

    python run_analytics_insights.py            # all parts
    python run_analytics_insights.py --part tags|space|drift

Outputs land in output/analytics-insights/.
"""
import argparse
import pathlib
import re
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / 'output' / 'analytics-insights'
OUT.mkdir(parents=True, exist_ok=True)

CORPORA = [('analytics', 'semantic_features', 'osm-rome', 'OSM Rome'),
           ('analytics-geolife', 'semantic_features-geolife',
            'geolife', 'GeoLife'),
           ('analytics-bird', None, 'bird', 'BIRD')]


# ── Part 1: what the tags say when crossed with each other ─────────────

def tags():
    """Cross-tabulations of independently derived tags.

    Purpose comes from Stage 7, interaction class from Stage 3. Neither was
    fitted to the other, so their agreement is evidence about the tag
    semantics rather than a property of one classifier.
    """
    rows = []
    for adir, _, _, name in CORPORA:
        p = HERE / 'output' / adir / 'class_purpose_x_interaction.csv'
        if not p.exists():
            continue
        df = pd.read_csv(p).set_index('a')
        for purpose, r in df.iterrows():
            total = r.sum()
            if total < 50:          # too few to read a ratio from
                continue
            rows.append({
                'corpus': name,
                'purpose': purpose.replace('Purpose', ''),
                'interactions': int(total),
                'immersive_pct': 100 * r['ImmersiveInteraction'] / total,
                'meaningful_pct': 100 * r['MeaningfulInteraction'] / total,
                'passing_pct': 100 * r['PassingInteraction'] / total,
            })
    eng = pd.DataFrame(rows).sort_values(
        ['corpus', 'immersive_pct'], ascending=[True, False])
    eng.to_csv(OUT / 'engagement_by_purpose.csv', index=False)
    print('\n=== engagement signature of each purpose ===')
    print(eng.to_string(index=False, float_format=lambda x: f'{x:5.1f}'))

    # Rank correlation between the immersive share and the passing share:
    # if the tags are meaningful these must move in opposite directions.
    from scipy.stats import spearmanr
    print('\nimmersive vs passing share, per corpus:')
    for name, g in eng.groupby('corpus'):
        if len(g) >= 3:
            rho, p = spearmanr(g['immersive_pct'], g['passing_pct'])
            print(f'  {name:<10} rho={rho:+.2f}  p={p:.3f}  (n={len(g)} purposes)')

    nav = []
    for adir, _, _, name in CORPORA:
        p = HERE / 'output' / adir / 'class_navstyle_x_temporal.csv'
        if not p.exists():
            continue
        df = pd.read_csv(p).set_index('a')
        for style, r in df.iterrows():
            night, morning = r.get('Night', 0), r.get('Morning', 0)
            wd, we = r.get('Weekday', 0), r.get('Weekend', 0)
            nav.append({'corpus': name, 'style': style,
                        'night': int(night), 'morning': int(morning),
                        'weekday': int(wd), 'weekend': int(we),
                        'night_share_pct': 100 * night / max(night + morning, 1),
                        'weekend_share_pct': 100 * we / max(wd + we, 1)})
    navdf = pd.DataFrame(nav)
    navdf.to_csv(OUT / 'navigation_by_temporal.csv', index=False)
    print('\n=== when each navigation style happens ===')
    print(navdf.to_string(index=False, float_format=lambda x: f'{x:5.1f}'))
    return eng, navdf


# ── Part 2: the three analyses section 7.8.3 promised, in the class space ──
#
# These were first attempted over the exported 67-dimensional feature matrix,
# with k-means for the archetypes and cosine for the analogues. That is
# generic vector analytics that happens to be fed by a KG, and it answered
# nothing the ontology could not answer directly. Redone here over the OWL
# classification layer, so the archetypes are the profile classes the reasoner
# derives, similarity is information-content-weighted overlap of asserted and
# inferred classes, and the cross-dataset comparison is a statement about
# which parts of the ontology each domain populates.


def _ontology(kg):
    import sys as _sys
    import atexit
    CFG = pathlib.Path('helpers/fuseki_config.py')
    orig = CFG.read_text()
    atexit.register(lambda: CFG.write_text(orig))
    CFG.write_text(re.sub(r'^graph_name = "[^"]*"', f'graph_name = "{kg}"',
                          orig, count=1, flags=re.M))
    for m in [m for m in list(_sys.modules) if 'helpers' in m]:
        del _sys.modules[m]
    from helpers.ontological_recommender import OntologicalRecommender
    rec = OntologicalRecommender()
    if kg not in rec.endpoint:
        raise RuntimeError(f'endpoint {rec.endpoint} != {kg}')
    rec.load_classifications()
    CFG.write_text(orig)
    return rec


def classes():
    per_corpus = {}
    for _, _, kg, name in CORPORA:
        try:
            rec = _ontology(kg)
        except Exception as exc:
            print(f'{name}: classification unavailable ({type(exc).__name__})')
            continue
        if not rec.class_counts:
            print(f'{name}: no inferred profile classes')
            continue
        per_corpus[name] = rec

    # (b) archetypes, as the ontology assigns them
    print('\n=== behavioural archetypes, as OWL profile classes ===')
    rows = []
    for name, rec in per_corpus.items():
        total = len(rec.classifications)
        print(f'\n  {name}: {total} classified trajectories, '
              f'{len(rec.class_counts)} profile classes')
        for cls, n in sorted(rec.class_counts.items(), key=lambda x: -x[1]):
            rows.append({'corpus': name, 'class': cls, 'n': n,
                         'share_pct': round(100 * n / total, 1),
                         'ic_weight': round(rec.ic_weights.get(cls, 0.0), 3)})
        for r in [r for r in rows if r['corpus'] == name][:8]:
            print(f"    {r['class']:<32}{r['n']:>6}{r['share_pct']:>7.1f}%"
                  f"   IC {r['ic_weight']:.2f}")
    cdf = pd.DataFrame(rows)
    cdf.to_csv(OUT / 'profile_classes.csv', index=False)

    # (c) cross-dataset comparison: which parts of the ontology each domain
    #     populates. This is the portability claim stated at class level.
    if len(per_corpus) >= 2:
        piv = cdf.pivot(index='class', columns='corpus',
                        values='share_pct').fillna(0.0)
        piv.to_csv(OUT / 'profile_classes_cross_corpus.csv')
        print('\n=== which profile classes each domain populates (% of corpus) ===')
        print(piv.to_string(float_format=lambda x: f'{x:6.1f}'))
        shared = (piv > 0).all(axis=1).sum()
        print(f'\n  classes populated in every corpus: {shared} of {len(piv)}')

    # (a) analogues, by information-content-weighted class overlap. A rare
    #     shared class says more than a common one, which is what IC encodes
    #     and what plain vector cosine cannot express.
    print('\n=== behavioural analogues by shared ontology classes ===')
    for name, rec in per_corpus.items():
        items = [(t, c) for t, c in rec.classifications.items() if c]
        if len(items) < 2:
            continue
        seed_uri, seed_cls = max(items, key=lambda tc: sum(
            rec.ic_weights.get(c, 0) for c in tc[1]))
        scored = []
        for uri, cls in items:
            if uri == seed_uri:
                continue
            inter = seed_cls & cls
            union = seed_cls | cls
            if not union:
                continue
            num = sum(rec.ic_weights.get(c, 0) for c in inter)
            den = sum(rec.ic_weights.get(c, 0) for c in union)
            if den:
                scored.append((num / den, uri, inter))
        scored.sort(key=lambda x: (-x[0], x[1]))
        print(f"\n  {name} seed {seed_uri.split('#')[-1]}"
              f"  [{', '.join(sorted(seed_cls))}]")
        for sim, uri, inter in scored[:3]:
            print(f"    {sim:.3f}  {uri.split('#')[-1]:<28}"
                  f"shared: {', '.join(sorted(inter))}")
        pd.DataFrame([{'corpus': name, 'seed': seed_uri, 'match': u,
                       'ic_jaccard': round(s, 4),
                       'shared_classes': ';'.join(sorted(i))}
                      for s, u, i in scored[:10]]).to_csv(
            OUT / f'analogues_{name.replace(" ", "_").lower()}.csv', index=False)


# ── Part 3: the preference drift study section 7.8.5 promised ──────────

def drift(kg='osm-rome'):
    import requests
    # The category of a preference record is written under TWO predicates:
    # user:categoryType on 29,065 records and user:prefersPoiCategory on 135.
    # Querying the ontology's declared property alone returns 0.5% of the
    # layer, which is what made this study look impossible at first. COALESCE
    # over both is the honest read until the pipeline is made to agree with
    # the ontology; the split itself is reported as a finding.
    Q = """
    PREFIX user: <http://w3id.org/trajecto/user#>
    SELECT ?profile ?snap ?ts ?cat ?score WHERE {
      ?profile user:hasSnapshot ?snap .
      ?snap user:updateTimestamp ?ts ; user:hasPreferences ?pref .
      ?pref user:preferenceScore ?score .
      OPTIONAL { ?pref user:categoryType ?ct }
      OPTIONAL { ?pref user:prefersPoiCategory ?pc }
      BIND(COALESCE(?ct, ?pc) AS ?cat)
      FILTER(BOUND(?cat))
    }
    """
    r = requests.post(f'http://localhost:3030/{kg}/sparql', data={'query': Q},
                      headers={'Accept': 'application/sparql-results+json'},
                      timeout=600).json()['results']['bindings']
    if not r:
        print('no snapshot data; is the KG populated?')
        return
    df = pd.DataFrame([{'profile': b['profile']['value'],
                        'snap': b['snap']['value'],
                        'ts': b['ts']['value'],
                        'cat': b['cat']['value'].split('#')[-1],
                        'score': float(b['score']['value'])} for b in r])
    print(f'\n=== preference drift: {df.snap.nunique()} snapshots, '
          f'{df.profile.nunique()} profiles, {len(df)} preference records ===')

    steps, ent = [], []
    for prof, g in df.groupby('profile'):
        snaps = (g.groupby(['snap', 'ts']).size().reset_index()
                 .sort_values('ts')['snap'].tolist())
        if len(snaps) < 2:
            continue
        vecs = []
        for s in snaps:
            v = g[g.snap == s].set_index('cat')['score']
            vecs.append(v)
        cats = sorted(set().union(*[set(v.index) for v in vecs]))
        M = np.array([[v.get(c, 0.0) for c in cats] for v in vecs])
        for i in range(1, len(M)):
            a, b = M[i - 1], M[i]
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            steps.append({'profile': prof, 'step': i,
                          'l1_change': float(np.abs(b - a).sum()),
                          'cosine_change': float(1 - (a @ b) / denom) if denom else 0.0})
        for i, row in enumerate(M):
            p = row / row.sum() if row.sum() > 0 else row
            p = p[p > 0]
            ent.append({'profile': prof, 'step': i,
                        'entropy': float(-(p * np.log(p)).sum()),
                        'n_categories': int((row > 0).sum())})
    sdf, edf = pd.DataFrame(steps), pd.DataFrame(ent)
    if sdf.empty:
        print('every profile has a single snapshot; no drift to measure')
        return
    sdf.to_csv(OUT / 'preference_drift_steps.csv', index=False)
    edf.to_csv(OUT / 'preference_entropy.csv', index=False)

    print(f'profiles with >1 snapshot: {sdf.profile.nunique()}, '
          f'update steps: {len(sdf)}')
    band = sdf.groupby(sdf.step.clip(upper=10))['l1_change'].agg(['mean', 'count'])
    print('\nmean L1 change in the preference vector, by update index:')
    print(band.to_string(float_format=lambda x: f'{x:7.3f}'))
    from scipy.stats import spearmanr
    rho, p = spearmanr(sdf.step, sdf.l1_change)
    print(f'\nstep vs magnitude of change: rho={rho:+.3f}, p={p:.2g}  '
          f'({"stabilises" if rho < 0 and p < 0.05 else "no stabilisation trend"})')
    eb = edf.groupby(edf.step.clip(upper=10))[['entropy', 'n_categories']].mean()
    print('\nmean preference entropy and breadth, by snapshot index:')
    print(eb.to_string(float_format=lambda x: f'{x:7.3f}'))
    rho2, p2 = spearmanr(edf.step, edf.entropy)
    print(f'step vs entropy: rho={rho2:+.3f}, p={p2:.2g}  '
          f'({"diversifies" if rho2 > 0 and p2 < 0.05 else "no diversification trend"})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=('tags', 'classes', 'drift', 'all'),
                    default='all')
    a = ap.parse_args()
    if a.part in ('tags', 'all'):
        tags()
    if a.part in ('classes', 'all'):
        classes()
    if a.part in ('drift', 'all'):
        drift()
