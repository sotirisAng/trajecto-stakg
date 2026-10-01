"""
Tag-stability harness.

Measures the reliability of the tag layer — the pipeline's primary
deliverable — via perturbation, without mutating the KG:

  stability      drop one POI interaction at a time from a trajectory and
                 re-infer its tags; stability(tag) = fraction of drops that
                 preserve the full-trajectory tag.
  predictivity   infer tags from the first n−1 interactions (temporal
                 order) and compare to the full-trajectory tags — can the
                 tag layer be trusted online, before the visit ends?

Tags re-inferred in memory, mirroring the pipeline rules exactly:
  dominant POI       stage 5: argmax(0.8·lognorm dwell + 0.2·inv-norm dist)
  dominant category  category of the dominant POI
  engagement         SR-10: High ≥3 immersive; Medium ≥5 meaningful & <3
                     immersive; Low otherwise
  purpose            stage 7 spirit: purpose implied by the dominant
                     category (category→purpose map)
  nav (interaction)  the interaction-count components of stage 6:
                     Targeted (=2 interactions), Exploratory (>10);
                     speed-based Casual is invariant under POI dropout and
                     excluded from the harness by construction

A sanity column reports agreement between the in-memory re-inference on the
FULL trajectory and the tags stored in the KG, which validates the mirror
itself before any perturbation is trusted.
"""

import math
import re
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd
import requests

from .fuseki_config import query_endpoint

PREFIXES = """
PREFIX trajecto: <http://w3id.org/trajecto#>
PREFIX poi: <http://w3id.org/trajecto/poi#>
PREFIX patterns: <http://w3id.org/trajecto/patterns#>
PREFIX spatial: <http://w3id.org/trajecto/spatial#>
PREFIX dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#>
"""

# Category -> purpose, inverted from the pipeline's purpose->category map
CATEGORY_PURPOSE = {}
_PURPOSE_CATEGORY = {
    'CulturalPurpose': {'Museum', 'Gallery', 'Artwork', 'ArtworkPOI',
                        'Monument', 'Cultural', 'CulturalPOI',
                        'HistoricalLandmark', 'HistoricalSite', 'Historical',
                        'ReligiousSite', 'Religious', 'HeritageBuilding',
                        'Church', 'Exhibit', 'IconicBuilding', 'Landmark',
                        'ViewingPoint'},
    'CommercialPurpose': {'RetailStore', 'ShoppingMall', 'MarketArea',
                          'CommercialPOI', 'Commercial', 'CommercialStreet',
                          'Bookstore'},
    'RecreationalPurpose': {'Park', 'Beach', 'LeisureFacility',
                            'SportsFacility', 'NaturalLandmark',
                            'RecreationalPOI', 'Recreational', 'Leisure'},
    'EducationalPurpose': {'Library', 'University', 'Educational',
                           'EducationalPOI', 'School'},
    'AccommodationPurpose': {'Hotel', 'Accommodation', 'AccommodationPOI',
                             'Hostel'},
    'DiningPurpose': {'Restaurant', 'Cafe', 'FastFood', 'FoodCourt', 'Bar',
                      'DiningFacility', 'Dining'},
    'TransportationPurpose': {'BusStation', 'TrainStation', 'Airport',
                              'Transportation', 'TransportationPOI'},
}
for purpose, cats in _PURPOSE_CATEGORY.items():
    for c in cats:
        CATEGORY_PURPOSE.setdefault(c, purpose)

# BIRD derives purpose from an artwork's creation era, not its category --
# bird/notebooks/helpers/artwork_categories.ERA_TO_PURPOSE. Mirroring Rome's
# category->purpose rule there produced 0.000 agreement with the stored tags:
# the rule is domain-specific by design, and the ontology supports specialising
# it. A corpus supplying creation periods uses this map instead.
ERA_TO_PURPOSE = {
    'Pre-1600': 'ClassicalArtsPurpose',
    '17th-century': 'ClassicalArtsPurpose',
    '18th-century': 'ClassicalArtsPurpose',
    '19th-century': 'ModernArtsPurpose',
    'Early-20th-century': 'ContemporaryArtsPurpose',
    'Late-20th-century+': 'ContemporaryArtsPurpose',
}


def era_bucket(period):
    """Bucket a free-text creation period into an ERA_TO_PURPOSE key."""
    if not period:
        return None
    years = [int(y) for y in re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', str(period))]
    if not years:
        return None
    y = max(years)
    if y < 1600:
        return 'Pre-1600'
    if y < 1700:
        return '17th-century'
    if y < 1800:
        return '18th-century'
    if y < 1900:
        return '19th-century'
    if y < 1950:
        return 'Early-20th-century'
    return 'Late-20th-century+'


GENERIC_TYPES = {'PointOfInterest', 'SignificantPOI', 'NamedIndividual',
                 'POIInteraction', 'MeaningfulInteraction',
                 'PassingInteraction', 'ImmersiveInteraction',
                 'DominantInteraction', 'StartInteraction',
                 'DestinationInteraction'}


def _query(sparql: str) -> List[Dict]:
    r = requests.post(query_endpoint, data={'query': PREFIXES + sparql},
                      headers={'Accept': 'application/json'}, timeout=300)
    r.raise_for_status()
    return [{k: v['value'] for k, v in b.items()}
            for b in r.json()['results']['bindings']]


def load_interactions(min_pois: int = 3) -> Dict[str, List[Dict]]:
    """traj -> ordered interaction records (poi, category, dwell, dist,
    immersive, meaningful, timestamp)."""
    rows = _query("""
    SELECT ?traj ?int ?poi ?cat ?dwell ?dist ?ts
           ?isImm ?isMean
    WHERE {
        ?pos dul:isPartOf ?traj .
        ?pos trajecto:hasInteraction ?int .
        ?int poi:interactsWith ?poi .
        OPTIONAL { ?pos trajecto:timestamp ?ts }
        OPTIONAL { ?int poi:hasDwellTimeInSeconds ?dwell }
        OPTIONAL { ?int spatial:hasDistance ?dist }
        OPTIONAL { ?poi a ?cat .
                   FILTER(!isBlank(?cat)) }
        BIND(EXISTS { ?int a poi:ImmersiveInteraction } AS ?isImm)
        BIND(EXISTS { ?int a poi:MeaningfulInteraction } AS ?isMean)
    }""")
    per_int = {}
    for r in rows:
        key = (r['traj'], r['int'])
        rec = per_int.setdefault(key, {
            'traj': r['traj'], 'int': r['int'], 'poi': r['poi'],
            'cats': set(), 'dwell': None, 'dist': None, 'ts': None,
            'imm': r.get('isImm') == 'true',
            'meaningful': r.get('isMean') == 'true',
        })
        cat = r.get('cat', '')
        local = cat.split('#')[-1] if cat else ''
        if local and local not in GENERIC_TYPES:
            rec['cats'].add(local)
        if r.get('dwell') is not None:
            try:
                rec['dwell'] = float(r['dwell'])
            except ValueError:
                pass
        if r.get('dist') is not None:
            try:
                rec['dist'] = float(r['dist'])
            except ValueError:
                pass
        if r.get('ts'):
            rec['ts'] = r['ts']

    trajs = defaultdict(list)
    for rec in per_int.values():
        rec['category'] = sorted(rec['cats'])[0] if rec['cats'] else ''
        trajs[rec['traj']].append(rec)
    out = {}
    for t, recs in trajs.items():
        if len({r['poi'] for r in recs}) >= min_pois:
            recs.sort(key=lambda r: (r['ts'] or '', r['int']))
            out[t] = recs
    return out


def load_stored_tags() -> Dict[str, Dict]:
    rows = _query("""
    SELECT ?traj ?eng ?dompoi ?purpose WHERE {
        { ?traj trajecto:hasEngagementLevel ?eng }
        UNION { ?pos dul:isPartOf ?traj . ?pos trajecto:hasInteraction ?d .
                ?d a poi:DominantInteraction ; poi:interactsWith ?dompoi }
        UNION { ?traj trajecto:hasPurpose ?purpose }
    }""")
    tags = defaultdict(dict)
    for r in rows:
        t = r['traj']
        if 'eng' in r:
            tags[t]['engagement'] = r['eng'].split('#')[-1]
        if 'dompoi' in r:
            tags[t]['dominant_poi'] = r['dompoi']
        if 'purpose' in r:
            tags[t].setdefault('purposes', set()).add(
                r['purpose'].split('#')[-1])
    return tags


# ── In-memory tag inference (mirrors pipeline rules) ──────────────────

def infer_tags(recs: List[Dict]) -> Dict[str, Optional[str]]:
    if not recs:
        return {'dominant_poi': None, 'dominant_category': None,
                'engagement': None, 'purpose': None, 'nav_interaction': None}

    # Stage 5 scoring: 0.8 * log-norm dwell + 0.2 * inv-norm distance
    dwells = [math.log1p(r['dwell']) if r['dwell'] else 0.0 for r in recs]
    dists = [r['dist'] if r['dist'] is not None else None for r in recs]
    max_dw = max(dwells) or 1.0
    known_d = [d for d in dists if d is not None]
    max_di = max(known_d) if known_d else 1.0
    best, best_score = None, -1.0
    for r, dw, di in zip(recs, dwells, dists):
        inv = 1.0 - (di / max_di) if (di is not None and max_di > 0) else 0.0
        score = 0.8 * (dw / max_dw) + 0.2 * inv
        if score > best_score:
            best, best_score = r, score
    dominant_poi = best['poi']
    dominant_category = best['category'] or None

    # SR-10 engagement
    n_imm = sum(1 for r in recs if r['imm'])
    n_mean = sum(1 for r in recs if r['meaningful'])
    if n_imm >= 3:
        engagement = 'HighEngagement'
    elif n_mean >= 5:
        engagement = 'MediumEngagement'
    else:
        engagement = 'LowEngagement'

    purpose = CATEGORY_PURPOSE.get(dominant_category or '', None)

    n_int = len(recs)
    if n_int == 2:
        nav = 'TargetedSegment'
    elif n_int > 10:
        nav = 'ExploratoryMovement'
    else:
        nav = None

    return {'dominant_poi': dominant_poi,
            'dominant_category': dominant_category,
            'engagement': engagement, 'purpose': purpose,
            'nav_interaction': nav}


TAGS = ['dominant_poi', 'dominant_category', 'engagement', 'purpose',
        'nav_interaction']


def run_harness(min_pois: int = 3, verbose: bool = True) -> pd.DataFrame:
    trajs = load_interactions(min_pois=min_pois)
    stored = load_stored_tags()
    if verbose:
        print(f'{len(trajs)} trajectories with >= {min_pois} POIs')

    stability = {t: [] for t in TAGS}
    predictivity = {t: [] for t in TAGS}
    sanity = {'engagement': [], 'dominant_poi': [], 'purpose': []}

    for traj, recs in trajs.items():
        full = infer_tags(recs)

        st = stored.get(traj, {})
        if 'engagement' in st and full['engagement']:
            sanity['engagement'].append(
                float(st['engagement'] == full['engagement']))
        if 'dominant_poi' in st and full['dominant_poi']:
            sanity['dominant_poi'].append(
                float(st['dominant_poi'] == full['dominant_poi']))
        if st.get('purposes') and full['purpose']:
            sanity['purpose'].append(
                float(full['purpose'] in st['purposes']))

        # drop-one stability (drop all interactions of one POI at a time)
        pois = sorted({r['poi'] for r in recs})
        for drop in pois:
            kept = [r for r in recs if r['poi'] != drop]
            if not kept:
                continue
            drops = infer_tags(kept)
            for t in TAGS:
                if full[t] is None and drops[t] is None:
                    continue
                stability[t].append(float(drops[t] == full[t]))

        # prefix predictivity (first n-1 interactions in temporal order)
        prefix = infer_tags(recs[:-1])
        for t in TAGS:
            if full[t] is None and prefix[t] is None:
                continue
            predictivity[t].append(float(prefix[t] == full[t]))

    rows = []
    for t in TAGS:
        rows.append({
            'tag': t,
            'stability': (sum(stability[t]) / len(stability[t])
                          if stability[t] else None),
            'n_stability': len(stability[t]),
            'predictivity': (sum(predictivity[t]) / len(predictivity[t])
                             if predictivity[t] else None),
            'n_predictivity': len(predictivity[t]),
            'kg_agreement': (sum(sanity[t]) / len(sanity[t])
                             if t in sanity and sanity[t] else None),
            'n_kg': len(sanity.get(t, [])),
        })
    return pd.DataFrame(rows)
