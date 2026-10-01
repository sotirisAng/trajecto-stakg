"""
Semantic Reasoning Engine for BIRD Museum Dataset

SHACL validation + SPARQL materialization for museum trajectory reasoning.
Adapted from Trajecto's semantic_reasoning.py:
- No temporal context (BIRD has elapsed timestamps, not wall-clock)
- Uses poi:hasType for artwork categories instead of rdf:type class hierarchy
- Similarity: 2 scored facets (category_jaccard, navigation); no temporal
"""

from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

try:
    from .fuseki_config import fuseki_server, graph_name, query_endpoint, update_endpoint
    from .fuseki_queries import execute_sparql_query, execute_sparql_update, format_results
except ImportError:
    from fuseki_config import fuseki_server, graph_name, query_endpoint, update_endpoint
    from fuseki_queries import execute_sparql_query, execute_sparql_update, format_results

RULES_DIR = Path(__file__).resolve().parent.parent.parent / 'sparql-rules'
SHACL_DIR = Path(__file__).resolve().parent.parent.parent / 'shacl'

PREFIXES = """
PREFIX trajecto: <http://w3id.org/trajecto#>
PREFIX poi:      <http://w3id.org/trajecto/poi#>
PREFIX patterns: <http://w3id.org/trajecto/patterns#>
PREFIX user:     <http://w3id.org/trajecto/user#>
PREFIX purpose:  <http://w3id.org/trajecto/purpose#>
PREFIX dul:      <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#>
PREFIX datacron: <http://www.datacron-project.eu/datAcron#>
PREFIX rdf:      <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs:     <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:      <http://www.w3.org/2001/XMLSchema#>
"""

# Similarity facet weights (no temporal for BIRD)
SIMILARITY_WEIGHTS = {
    'category_jaccard': 0.60,
    'navigation': 0.40,
}

# Predicates materialized by the reasoning rules
MATERIALIZED_PREDICATES = [
    'http://w3id.org/trajecto/poi#covisitedWith',
    'http://w3id.org/trajecto/poi#followedBy',
    'http://w3id.org/trajecto#hasEngagementLevel',
    'http://w3id.org/trajecto#hasJaccardCategorySimilarity',
    'http://w3id.org/trajecto#hasNavigationMatch',
    'http://w3id.org/trajecto#hasPurposeMatch',
    'http://w3id.org/trajecto#withTrajectory',
    'http://w3id.org/trajecto#similarityScore',
    'http://w3id.org/trajecto/user#hasInferredPreference',
    'http://w3id.org/trajecto/user#hasAffinityFor',
    'http://w3id.org/trajecto/user#forCategory',
    'http://w3id.org/trajecto/user#immersiveFraction',
    'http://w3id.org/trajecto/user#meaningfulFraction',
    'http://w3id.org/trajecto/user#totalVisits',
]


class SemanticReasoningEngine:
    """SHACL validation + SPARQL materialization for trajectory reasoning."""

    def __init__(self, endpoint=None, upd_endpoint=None):
        self.query_endpoint = endpoint or query_endpoint
        self.update_endpoint = upd_endpoint or update_endpoint
        self.shacl_endpoint = f"{fuseki_server}/{graph_name}/shacl?graph=default"

    # ------------------------------------------------------------------ #
    # SHACL Validation
    # ------------------------------------------------------------------ #

    def run_validation(self, shapes_file: str = 'trajectory-validation.ttl') -> dict:
        """
        Validate data quality. Tries Fuseki /shacl endpoint first;
        falls back to SPARQL-based checks if SHACL service is unavailable.
        """
        shapes_path = SHACL_DIR / shapes_file
        if shapes_path.exists():
            try:
                with open(shapes_path, encoding='utf-8') as f:
                    shapes_ttl = f.read()
                response = requests.post(
                    self.shacl_endpoint,
                    data=shapes_ttl.encode('utf-8'),
                    headers={'Content-Type': 'text/turtle', 'Accept': 'application/ld+json'},
                    timeout=120
                )
                response.raise_for_status()
                report = response.json()
                return self._parse_shacl_report(report)
            except requests.exceptions.RequestException:
                pass

        print("SHACL endpoint not available, using SPARQL-based validation.")
        return self._sparql_validation()

    def _sparql_validation(self) -> dict:
        """Run data quality checks via SPARQL as fallback for SHACL."""
        results = []

        checks = [
            (
                "Trajectory without navigation pattern",
                "sh:Warning",
                PREFIXES + """
                SELECT (COUNT(?traj) AS ?cnt) WHERE {
                    ?traj a datacron:Trajectory .
                    FILTER NOT EXISTS { ?traj patterns:exhibitsPattern ?p . }
                }"""
            ),
            (
                "Trajectory without purpose",
                "sh:Warning",
                PREFIXES + """
                SELECT (COUNT(?traj) AS ?cnt) WHERE {
                    ?traj a datacron:Trajectory .
                    FILTER NOT EXISTS { ?traj trajecto:hasPurpose ?p . }
                }"""
            ),
            (
                "Interaction without POI link",
                "sh:Violation",
                PREFIXES + """
                SELECT (COUNT(?int) AS ?cnt) WHERE {
                    ?int a poi:POIInteraction .
                    FILTER NOT EXISTS { ?int poi:interactsWith ?poi . }
                }"""
            ),
            (
                "Interaction without dwell time",
                "sh:Warning",
                PREFIXES + """
                SELECT (COUNT(?int) AS ?cnt) WHERE {
                    ?int a poi:POIInteraction .
                    FILTER NOT EXISTS { ?int poi:hasDwellTimeInSeconds ?d . }
                }"""
            ),
        ]

        violations = 0
        warnings = 0

        for message, severity, query in checks:
            result = execute_sparql_query(self.query_endpoint, query)
            df = format_results(result)
            count = int(df.iloc[0]['cnt']) if not df.empty else 0
            if count > 0:
                results.append({
                    'severity': severity,
                    'message': f"{message}: {count} instances",
                    'focusNode': '',
                    'path': '',
                })
                if severity == 'sh:Violation':
                    violations += count
                else:
                    warnings += count

        conforms = violations == 0
        return {
            'conforms': conforms,
            'violations': violations,
            'warnings': warnings,
            'total': len(results),
            'results': results,
        }

    def _parse_shacl_report(self, report: any) -> dict:
        """Parse JSON-LD SHACL validation report into structured dict."""
        if isinstance(report, list):
            report = report[0] if report else {}

        conforms = report.get('http://www.w3.org/ns/shacl#conforms', [{}])
        if isinstance(conforms, list):
            conforms = conforms[0].get('@value', True) if conforms else True
        elif isinstance(conforms, dict):
            conforms = conforms.get('@value', True)

        results = []
        raw_results = report.get('http://www.w3.org/ns/shacl#result', [])
        if not isinstance(raw_results, list):
            raw_results = [raw_results]

        for r in raw_results:
            if not isinstance(r, dict):
                continue
            severity = self._extract_jsonld_value(r, 'http://www.w3.org/ns/shacl#resultSeverity')
            message = self._extract_jsonld_value(r, 'http://www.w3.org/ns/shacl#resultMessage')
            focus = self._extract_jsonld_value(r, 'http://www.w3.org/ns/shacl#focusNode')
            path = self._extract_jsonld_value(r, 'http://www.w3.org/ns/shacl#resultPath')
            results.append({
                'severity': self._shorten_uri(severity),
                'message': message,
                'focusNode': self._shorten_uri(focus),
                'path': self._shorten_uri(path),
            })

        violations = sum(1 for r in results if r['severity'] == 'sh:Violation')
        warnings = sum(1 for r in results if r['severity'] == 'sh:Warning')

        return {
            'conforms': conforms,
            'violations': violations,
            'warnings': warnings,
            'total': len(results),
            'results': results,
        }

    @staticmethod
    def _extract_jsonld_value(obj, key):
        val = obj.get(key, [{}])
        if isinstance(val, list):
            val = val[0] if val else {}
        if isinstance(val, dict):
            return val.get('@value', val.get('@id', ''))
        return str(val)

    @staticmethod
    def _shorten_uri(uri):
        if not uri:
            return ''
        prefixes = {
            'http://www.w3.org/ns/shacl#': 'sh:',
            'http://w3id.org/trajecto#': 'trajecto:',
            'http://w3id.org/trajecto/poi#': 'poi:',
            'http://w3id.org/trajecto/patterns#': 'patterns:',
            'http://w3id.org/trajecto/user#': 'user:',
            'http://w3id.org/trajecto/purpose#': 'purpose:',
        }
        for full, short in prefixes.items():
            if uri.startswith(full):
                return short + uri[len(full):]
        return uri

    # ------------------------------------------------------------------ #
    # SPARQL Rule Execution
    # ------------------------------------------------------------------ #

    def _load_rule(self, rule_file: str) -> str:
        """Load SPARQL rule from file."""
        path = RULES_DIR / rule_file
        if not path.exists():
            raise FileNotFoundError(f"Rule file not found: {path}")
        with open(path, encoding='utf-8') as f:
            return f.read()

    def execute_rule(self, rule_file: str, params: Optional[dict[str, str]] = None) -> bool:
        """Execute a SPARQL UPDATE rule from file. Returns success."""
        query = self._load_rule(rule_file)
        if params:
            for key, value in params.items():
                query = query.replace(f'${key}', value)
        return execute_sparql_update(self.update_endpoint, query)

    def execute_relationship_rules(self) -> dict[str, bool]:
        """Execute SR-05, SR-06: POI co-visitation and sequence."""
        results = {}
        for rule in ['SR-05-covisitation.sparql',
                      'SR-06-sequence-succession.sparql']:
            name = rule.split('.')[0]
            results[name] = self.execute_rule(rule)
        return results

    def execute_engagement_rules(self) -> dict[str, bool]:
        """Execute SR-10: engagement levels."""
        return {'SR-10': self.execute_rule('SR-10-engagement-levels.sparql')}

    def execute_similarity_batch(self) -> dict[str, bool]:
        """Compute SR-11 (category Jaccard) in Python, SR-12/SR-14 via SPARQL.

        SR-11 is too expensive as SPARQL (double self-join on 42K interaction triples).
        Python computes Jaccard from category sets and materializes via INSERT DATA.
        SR-12 (nav match) and SR-14 (purpose match) are lightweight SPARQL joins.
        """
        results = {}

        # SR-11: category Jaccard in Python
        results['SR-11-category-jaccard'] = self._compute_category_jaccard()

        # SR-12, SR-14: lightweight, keep as SPARQL
        for rule in ['SR-12-navigation-match.sparql',
                      'SR-14-purpose-match.sparql']:
            name = rule.split('.')[0]
            results[name] = self.execute_rule(rule)
        return results

    def _compute_category_jaccard(self) -> bool:
        """Compute SR-11 category Jaccard similarity in Python and materialize."""
        # 1. Fetch category sets per trajectory
        query = PREFIXES + """
        SELECT ?traj ?cat WHERE {
            ?pos dul:isPartOf ?traj .
            ?pos trajecto:hasInteraction ?int .
            ?int a poi:MeaningfulInteraction .
            ?int poi:interactsWith ?poi .
            ?poi poi:hasType ?cat .
        }
        """
        result = execute_sparql_query(self.query_endpoint, query)
        if not result or not result.get('results', {}).get('bindings'):
            print("  SR-11: no category data found")
            return False

        traj_cats = defaultdict(set)
        for b in result['results']['bindings']:
            traj_cats[b['traj']['value']].add(b['cat']['value'])

        # 2. Fetch purpose blocking keys
        pq = PREFIXES + """
        SELECT ?traj ?purpose WHERE { ?traj trajecto:hasPurpose ?purpose . }
        """
        pr = execute_sparql_query(self.query_endpoint, pq)
        traj_purposes = defaultdict(set)
        if pr and pr.get('results', {}).get('bindings'):
            for b in pr['results']['bindings']:
                traj_purposes[b['traj']['value']].add(b['purpose']['value'])

        # 3. Compute Jaccard for pairs sharing a purpose, threshold > 0.3
        trajs = sorted(traj_cats.keys())
        triples = []
        for i in range(len(trajs)):
            for j in range(i + 1, len(trajs)):
                a, b_uri = trajs[i], trajs[j]
                # Blocking: must share a purpose
                if not (traj_purposes[a] & traj_purposes[b_uri]):
                    continue
                intersection = len(traj_cats[a] & traj_cats[b_uri])
                union = len(traj_cats[a] | traj_cats[b_uri])
                if union == 0:
                    continue
                jaccard = intersection / union
                if jaccard > 0.3:
                    triples.append(
                        f'<{a}> trajecto:hasJaccardCategorySimilarity '
                        f'[ trajecto:withTrajectory <{b_uri}> ; '
                        f'trajecto:similarityScore {jaccard:.4f} ] .')

        if not triples:
            print("  SR-11: no pairs above threshold")
            return True

        # 4. Materialize in batches
        BATCH = 50
        success = True
        for i in range(0, len(triples), BATCH):
            chunk = triples[i:i + BATCH]
            insert_q = PREFIXES + "INSERT DATA {\n" + "\n".join(chunk) + "\n}"
            if not execute_sparql_update(self.update_endpoint, insert_q):
                success = False

        print(f"  SR-11: {len(triples)} pairs materialized (Python)")
        return success

    def execute_preference_rules(self) -> dict[str, bool]:
        """Execute SR-20, SR-21: user preference inference."""
        results = {}
        for rule in ['SR-20-preference-inference.sparql',
                      'SR-21-preference-affinity.sparql']:
            name = rule.split('.')[0]
            results[name] = self.execute_rule(rule)
        return results

    # ------------------------------------------------------------------ #
    # Pipeline
    # ------------------------------------------------------------------ #

    def run_full_pipeline(self, include_similarity_batch=False) -> dict:
        """
        Execute full reasoning pipeline:
        1. SHACL validation
        2. Relationships (SR-05, SR-06)
        3. Engagement (SR-10)
        4. Preferences (SR-20, SR-21)
        5. Optionally: batch similarity (SR-11, SR-12, SR-14)
        """
        results = {}

        print("Step 1: SHACL validation...")
        results['validation'] = self.run_validation()
        print(f"  Conforms: {results['validation']['conforms']}, "
              f"Violations: {results['validation']['violations']}, "
              f"Warnings: {results['validation']['warnings']}")

        print("\nStep 2: Relationship rules (SR-05, SR-06)...")
        results['relationships'] = self.execute_relationship_rules()

        print("\nStep 3: Engagement rules (SR-10)...")
        results['engagement'] = self.execute_engagement_rules()

        print("\nStep 4: Preference rules (SR-20, SR-21)...")
        results['preferences'] = self.execute_preference_rules()

        if include_similarity_batch:
            print("\nStep 5: Batch similarity (SR-11, SR-12, SR-14)...")
            results['similarity'] = self.execute_similarity_batch()

        return results

    def clear_reasoning_triples(self) -> bool:
        """Remove all triples materialized by reasoning rules.

        Uses SELECT + batch DELETE DATA instead of DELETE WHERE to avoid
        Fuseki's expensive full-graph scan on wildcard patterns.
        """
        success = True
        for pred in MATERIALIZED_PREDICATES:
            # 1. Fetch explicit s/o pairs
            select_q = f"SELECT ?s ?o WHERE {{ ?s <{pred}> ?o . }}"
            result = execute_sparql_query(self.query_endpoint, select_q)
            if not result or not result.get('results', {}).get('bindings'):
                continue

            bindings = result['results']['bindings']

            # 2. Batch DELETE DATA with explicit triples (100 per batch)
            BATCH = 100
            for i in range(0, len(bindings), BATCH):
                chunk = bindings[i:i + BATCH]
                triples = []
                for b in chunk:
                    s = b['s']['value']
                    o = b['o']['value']
                    if b['o']['type'] == 'literal':
                        # Escape backslashes and quotes; use triple-quotes for multiline
                        o_escaped = o.replace('\\', '\\\\').replace('"', '\\"')
                        needs_long = '\n' in o or '\r' in o
                        q = '"""' if needs_long else '"'
                        dt = b['o'].get('datatype', '')
                        lang = b['o'].get('xml:lang', '')
                        if lang:
                            triples.append(f'<{s}> <{pred}> {q}{o_escaped}{q}@{lang} .')
                        elif dt:
                            triples.append(
                                f'<{s}> <{pred}> {q}{o_escaped}{q}^^<{dt}> .')
                        else:
                            triples.append(f'<{s}> <{pred}> {q}{o_escaped}{q} .')
                    else:
                        triples.append(f'<{s}> <{pred}> <{o}> .')

                delete_q = "DELETE DATA {\n" + "\n".join(triples) + "\n}"
                if not execute_sparql_update(self.update_endpoint, delete_q):
                    success = False

        return success

    # ------------------------------------------------------------------ #
    # Trajectory Profile Queries
    # ------------------------------------------------------------------ #

    def get_trajectory_profile(self, uri: str) -> dict:
        """Get the semantic profile of a trajectory using existing KG properties."""
        query = PREFIXES + f"""
        SELECT ?prop ?value WHERE {{
            BIND(<{uri}> AS ?traj)
            {{
                ?traj trajecto:hasEngagementLevel ?val .
                BIND("engagementLevel" AS ?prop)
                BIND(STR(?val) AS ?value)
            }}
            UNION
            {{
                ?traj patterns:exhibitsPattern ?val .
                BIND("navigationStyle" AS ?prop)
                BIND(STR(?val) AS ?value)
            }}
            UNION
            {{
                ?traj trajecto:hasPurpose ?val .
                BIND("purpose" AS ?prop)
                BIND(STR(?val) AS ?value)
            }}
        }}
        """
        result = execute_sparql_query(self.query_endpoint, query)
        df = format_results(result)
        if df.empty:
            return {}

        profile = defaultdict(list)
        for _, row in df.iterrows():
            val = self._shorten_uri(row['value'])
            profile[row['prop']].append(val)

        out = {}
        for k, v in profile.items():
            out[k] = v[0] if len(v) == 1 else v
        return out

    def get_engagement_distribution(self) -> dict[str, int]:
        """Count trajectories per engagement level."""
        query = PREFIXES + """
        SELECT ?level (COUNT(?traj) AS ?count)
        WHERE {
            ?traj trajecto:hasEngagementLevel ?level .
        }
        GROUP BY ?level
        ORDER BY DESC(?count)
        """
        result = execute_sparql_query(self.query_endpoint, query)
        df = format_results(result)
        if df.empty:
            return {}
        return {self._shorten_uri(row['level']): int(row['count'])
                for _, row in df.iterrows()}

    def get_engagement_level(self, uri: str) -> str:
        """Get engagement level for a specific trajectory."""
        query = PREFIXES + f"""
        SELECT ?level WHERE {{
            <{uri}> trajecto:hasEngagementLevel ?level .
        }}
        """
        result = execute_sparql_query(self.query_endpoint, query)
        df = format_results(result)
        if df.empty:
            return 'Unknown'
        return self._shorten_uri(df.iloc[0]['level'])

    def get_purpose_distribution(self) -> dict[str, int]:
        """Count trajectories per purpose."""
        query = PREFIXES + """
        SELECT ?purpose (COUNT(?traj) AS ?count)
        WHERE {
            ?traj trajecto:hasPurpose ?purpose .
        }
        GROUP BY ?purpose
        ORDER BY DESC(?count)
        """
        result = execute_sparql_query(self.query_endpoint, query)
        df = format_results(result)
        if df.empty:
            return {}
        return {self._shorten_uri(row['purpose']): int(row['count'])
                for _, row in df.iterrows()}

    def get_profile_groups(self) -> list[dict]:
        """Group trajectories by purpose + navigation style."""
        query = PREFIXES + """
        SELECT ?purpose ?navStyle (COUNT(?traj) AS ?count)
               (GROUP_CONCAT(DISTINCT STR(?traj); separator=",") AS ?trajectories)
        WHERE {
            ?traj trajecto:hasPurpose ?purpose .
            ?traj patterns:exhibitsPattern ?navStyle .
        }
        GROUP BY ?purpose ?navStyle
        ORDER BY DESC(?count)
        """
        result = execute_sparql_query(self.query_endpoint, query)
        df = format_results(result)
        if df.empty:
            return []
        groups = []
        for _, row in df.iterrows():
            groups.append({
                'purpose': self._shorten_uri(row['purpose']),
                'navigationStyle': self._shorten_uri(row['navStyle']),
                'count': int(row['count']),
                'trajectories': row['trajectories'].split(',') if row['trajectories'] else [],
            })
        return groups

    def get_cross_tabulation(self) -> pd.DataFrame:
        """Cross-tabulate purpose vs navigation style."""
        query = PREFIXES + """
        SELECT ?purpose ?navStyle (COUNT(?traj) AS ?count)
        WHERE {
            ?traj trajecto:hasPurpose ?purpose .
            ?traj patterns:exhibitsPattern ?navStyle .
        }
        GROUP BY ?purpose ?navStyle
        ORDER BY ?purpose ?navStyle
        """
        result = execute_sparql_query(self.query_endpoint, query)
        df = format_results(result)
        if df.empty:
            return pd.DataFrame()
        df['purpose'] = df['purpose'].apply(self._shorten_uri)
        df['navStyle'] = df['navStyle'].apply(self._shorten_uri)
        df['count'] = df['count'].astype(int)
        return df.pivot_table(index='purpose', columns='navStyle',
                              values='count', fill_value=0)

    def get_engagement_by_purpose(self) -> pd.DataFrame:
        """Cross-tabulate engagement level vs purpose."""
        query = PREFIXES + """
        SELECT ?purpose ?level (COUNT(?traj) AS ?count)
        WHERE {
            ?traj trajecto:hasPurpose ?purpose .
            ?traj trajecto:hasEngagementLevel ?level .
        }
        GROUP BY ?purpose ?level
        ORDER BY ?purpose ?level
        """
        result = execute_sparql_query(self.query_endpoint, query)
        df = format_results(result)
        if df.empty:
            return pd.DataFrame()
        df['purpose'] = df['purpose'].apply(self._shorten_uri)
        df['level'] = df['level'].apply(self._shorten_uri)
        df['count'] = df['count'].astype(int)
        return df.pivot_table(index='purpose', columns='level',
                              values='count', fill_value=0)

    def spot_check_profile(self, uri: str) -> dict:
        """Verify a trajectory's profile against its actual interactions."""
        profile = self.get_trajectory_profile(uri)

        query = PREFIXES + f"""
        SELECT ?poi ?artworkType ?intType ?dwell WHERE {{
            ?pos dul:isPartOf <{uri}> .
            ?pos trajecto:hasInteraction ?int .
            ?int a ?intType .
            ?int poi:interactsWith ?poi .
            ?poi poi:hasType ?artworkType .
            FILTER(?intType IN (poi:PassingInteraction, poi:MeaningfulInteraction,
                                poi:ImmersiveInteraction, poi:DominantInteraction))
            OPTIONAL {{ ?int poi:hasDwellTimeInSeconds ?dwell . }}
        }}
        ORDER BY ?artworkType
        """
        result = execute_sparql_query(self.query_endpoint, query)
        df = format_results(result)

        return {
            'profile': profile,
            'interactions': df,
            'trajectory': uri,
        }

    # ------------------------------------------------------------------ #
    # Similarity (On-Demand)
    # ------------------------------------------------------------------ #

    def get_similar_trajectories(self, uri: str, min_score: float = 0.3,
                                  top_k: int = 10) -> list[dict]:
        """
        On-demand faceted similarity: 2 scored facets (category_jaccard, navigation).
        Purpose is the blocking key.
        """
        query = PREFIXES + f"""
        SELECT ?other ?catJaccard ?navMatch ?aggregate
        WHERE {{
            BIND(<{uri}> AS ?input)

            # Blocking: shared purpose
            ?input trajecto:hasPurpose ?sharedPurpose .
            ?other trajecto:hasPurpose ?sharedPurpose .
            FILTER(?other != ?input)

            # Navigation match
            OPTIONAL {{
                SELECT ?input ?other (1.0 AS ?navScore) WHERE {{
                    ?input patterns:exhibitsPattern ?p .
                    ?other patterns:exhibitsPattern ?p .
                }}
                LIMIT 1
            }}
            BIND(COALESCE(?navScore, 0.0) AS ?navMatch)

            # Category Jaccard (shared / union of artwork types)
            OPTIONAL {{
                SELECT ?input ?other
      (xsd:float(COUNT(DISTINCT ?sharedCat)) / xsd:float(IF(COUNT(DISTINCT ?anyCat) < 1, 1, COUNT(DISTINCT ?anyCat))) AS ?catScore)
                WHERE {{
                    {{
                        ?posI dul:isPartOf ?input .
                        ?posI trajecto:hasInteraction ?intI .
                        ?intI a poi:MeaningfulInteraction .
                        ?intI poi:interactsWith ?poiI .
                        ?poiI poi:hasType ?catI .

                        ?posO dul:isPartOf ?other .
                        ?posO trajecto:hasInteraction ?intO .
                        ?intO a poi:MeaningfulInteraction .
                        ?intO poi:interactsWith ?poiO .
                        ?poiO poi:hasType ?catI .

                        BIND(?catI AS ?sharedCat)
                        BIND(?catI AS ?anyCat)
                    }}
                    UNION
                    {{
                        ?posIA dul:isPartOf ?input .
                        ?posIA trajecto:hasInteraction ?intIA .
                        ?intIA a poi:MeaningfulInteraction .
                        ?intIA poi:interactsWith ?poiIA .
                        ?poiIA poi:hasType ?catIA .
                        BIND(?catIA AS ?anyCat)
                    }}
                    UNION
                    {{
                        ?posOA dul:isPartOf ?other .
                        ?posOA trajecto:hasInteraction ?intOA .
                        ?intOA a poi:MeaningfulInteraction .
                        ?intOA poi:interactsWith ?poiOA .
                        ?poiOA poi:hasType ?catOA .
                        BIND(?catOA AS ?anyCat)
                    }}
                }}
                GROUP BY ?input ?other
            }}
            BIND(COALESCE(?catScore, 0.0) AS ?catJaccard)

            # Weighted aggregate (no temporal)
            BIND(
                {SIMILARITY_WEIGHTS['category_jaccard']} * ?catJaccard +
                {SIMILARITY_WEIGHTS['navigation']} * ?navMatch
                AS ?aggregate
            )
            FILTER(?aggregate >= {min_score})
        }}
        ORDER BY DESC(?aggregate)
        LIMIT {top_k}
        """
        result = execute_sparql_query(self.query_endpoint, query)
        df = format_results(result)
        if df.empty:
            return []

        similar = []
        for _, row in df.iterrows():
            similar.append({
                'trajectory': row['other'],
                'aggregate': float(row['aggregate']),
                'facets': {
                    'category_jaccard': float(row['catJaccard']),
                    'navigation': float(row['navMatch']),
                },
            })
        return similar

    def get_similarity_distribution(self) -> pd.DataFrame:
        """Get distribution of materialized similarity scores (batch mode)."""
        query = PREFIXES + """
        SELECT ?trajA ?trajB ?catJ ?nav WHERE {
            ?trajA trajecto:hasJaccardCategorySimilarity ?cj .
            ?cj trajecto:withTrajectory ?trajB .
            ?cj trajecto:similarityScore ?catJ .

            OPTIONAL {
                ?trajA trajecto:hasNavigationMatch ?nm .
                ?nm trajecto:withTrajectory ?trajB .
                ?nm trajecto:similarityScore ?nav .
            }
        }
        """
        result = execute_sparql_query(self.query_endpoint, query)
        df = format_results(result)
        if df.empty:
            return pd.DataFrame()

        for col in ['catJ', 'nav']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        w = SIMILARITY_WEIGHTS
        df['aggregate'] = (
            w['category_jaccard'] * df['catJ'] +
            w['navigation'] * df['nav']
        )
        return df

    # ------------------------------------------------------------------ #
    # User Preferences
    # ------------------------------------------------------------------ #

    def get_inferred_preferences(self, user_uri: str) -> list[dict]:
        """Get inferred preferences for a user from SR-20."""
        query = PREFIXES + f"""
        SELECT ?cat ?score ?immFrac ?meanFrac ?visits WHERE {{
            <{user_uri}> user:hasInferredPreference ?pref .
            ?pref user:forCategory ?cat .
            ?pref user:preferenceScore ?score .
            OPTIONAL {{ ?pref user:immersiveFraction ?immFrac . }}
            OPTIONAL {{ ?pref user:meaningfulFraction ?meanFrac . }}
            OPTIONAL {{ ?pref user:totalVisits ?visits . }}
        }}
        ORDER BY DESC(?score)
        """
        result = execute_sparql_query(self.query_endpoint, query)
        df = format_results(result)
        if df.empty:
            return []
        prefs = []
        for _, row in df.iterrows():
            prefs.append({
                'category': self._shorten_uri(row['cat']),
                'score': float(row['score']),
                'immersiveFraction': float(row.get('immFrac', 0)),
                'meaningfulFraction': float(row.get('meanFrac', 0)),
                'totalVisits': int(float(row.get('visits', 0))),
            })
        return prefs

    def get_user_affinities(self, user_uri: str) -> list[str]:
        """Get categories the user has affinity for (from SR-21)."""
        query = PREFIXES + f"""
        SELECT ?cat WHERE {{
            <{user_uri}> user:hasAffinityFor ?cat .
        }}
        """
        result = execute_sparql_query(self.query_endpoint, query)
        df = format_results(result)
        if df.empty:
            return []
        return [self._shorten_uri(row['cat']) for _, row in df.iterrows()]

    # ------------------------------------------------------------------ #
    # Recommendations
    # ------------------------------------------------------------------ #

    def recommend(self, trajectory_uri: str, top_k: int = 10) -> list[dict]:
        """
        Run all recommendation rules (SR-30..SR-34) and aggregate.
        POIs covered by more rules score higher.
        """
        rules = {
            'SR-30': 'SR-30-profile-collaborative.sparql',
            'SR-31': 'SR-31-preference-gap.sparql',
            'SR-32': 'SR-32-sequence-continuation.sparql',
            'SR-33': 'SR-33-covisitation-recs.sparql',
            'SR-34': 'SR-34-purpose-alignment.sparql',
        }

        poi_data = defaultdict(lambda: {'rules': [], 'categories': set(), 'details': {}})

        for rule_name, rule_file in rules.items():
            query = self._load_rule(rule_file)
            query = query.replace('$TRAJECTORY_URI', trajectory_uri)
            result = execute_sparql_query(self.query_endpoint, query)
            df = format_results(result)
            if df.empty:
                continue

            for _, row in df.iterrows():
                poi = row['poi']
                poi_data[poi]['rules'].append(row.get('rule', rule_name))
                poi_data[poi]['categories'].add(self._shorten_uri(row['poiCat']))
                if 'prefScore' in row:
                    poi_data[poi]['details']['prefScore'] = float(row['prefScore'])
                if 'sourcePoi' in row:
                    poi_data[poi]['details']['sourcePoi'] = row['sourcePoi']
                if 'dominantPoi' in row:
                    poi_data[poi]['details']['dominantPoi'] = row['dominantPoi']
                if 'similarTraj' in row:
                    poi_data[poi]['details'].setdefault('similarTrajs', set())
                    poi_data[poi]['details']['similarTrajs'].add(row['similarTraj'])

        ranked = []
        for poi, data in poi_data.items():
            score = len(set(data['rules'])) / len(rules)
            ranked.append({
                'poi': poi,
                'poiLabel': self._shorten_uri(poi),
                'categories': list(data['categories']),
                'score': score,
                'ruleCoverage': len(set(data['rules'])),
                'rules': list(set(data['rules'])),
                'details': {k: (list(v) if isinstance(v, set) else v)
                            for k, v in data['details'].items()},
            })

        ranked.sort(key=lambda x: (-x['score'], x['poi']))
        return ranked[:top_k]

    def recommend_with_explanations(self, trajectory_uri: str,
                                     top_k: int = 10) -> list[dict]:
        """Recommend with full rule provenance and explanations."""
        recs = self.recommend(trajectory_uri, top_k)

        rule_explanations = {
            'SR-30:profile-collaborative': 'From trajectories with similar semantic profile',
            'SR-31:preference-gap': 'Matches inferred user preferences (unvisited category)',
            'SR-32:sequence-continuation': 'Next in common visit sequence',
            'SR-33:covisitation': 'Co-visited with dominant artwork in other trajectories',
            'SR-34:purpose-alignment': 'Aligns with trajectory purpose',
        }

        for rec in recs:
            rec['explanations'] = [
                rule_explanations.get(r, r) for r in rec['rules']
            ]

        return recs

    def get_rule_yield(self, trajectory_uri: str) -> dict[str, int]:
        """Count how many POIs each recommendation rule produces."""
        rules = {
            'SR-30': 'SR-30-profile-collaborative.sparql',
            'SR-31': 'SR-31-preference-gap.sparql',
            'SR-32': 'SR-32-sequence-continuation.sparql',
            'SR-33': 'SR-33-covisitation-recs.sparql',
            'SR-34': 'SR-34-purpose-alignment.sparql',
        }
        yields = {}
        for rule_name, rule_file in rules.items():
            query = self._load_rule(rule_file)
            query = query.replace('$TRAJECTORY_URI', trajectory_uri)
            result = execute_sparql_query(self.query_endpoint, query)
            df = format_results(result)
            yields[rule_name] = len(df) if not df.empty else 0
        return yields

    # ------------------------------------------------------------------ #
    # POI Relationship Queries
    # ------------------------------------------------------------------ #

    def get_covisitation_links(self, poi_uri: Optional[str] = None, limit: int = 50) -> pd.DataFrame:
        """Get materialized co-visitation relationships."""
        filter_clause = f'FILTER(?poi = <{poi_uri}>)' if poi_uri else ''
        query = PREFIXES + f"""
        SELECT ?poi ?covisited WHERE {{
            ?poi poi:covisitedWith ?covisited .
            {filter_clause}
        }}
        LIMIT {limit}
        """
        result = execute_sparql_query(self.query_endpoint, query)
        return format_results(result)

    def get_sequence_links(self, poi_uri: Optional[str] = None, limit: int = 50) -> pd.DataFrame:
        """Get materialized sequence succession relationships."""
        filter_clause = f'FILTER(?poi = <{poi_uri}>)' if poi_uri else ''
        query = PREFIXES + f"""
        SELECT ?poi ?next WHERE {{
            ?poi poi:followedBy ?next .
            {filter_clause}
        }}
        LIMIT {limit}
        """
        result = execute_sparql_query(self.query_endpoint, query)
        return format_results(result)

    def get_relationship_stats(self) -> dict:
        """Count materialized co-visitation and sequence triples."""
        query = PREFIXES + """
        SELECT
            (COUNT(DISTINCT ?cv) AS ?covisitCount)
            (COUNT(DISTINCT ?sq) AS ?sequenceCount)
        WHERE {
            { ?a poi:covisitedWith ?cv . }
            UNION
            { ?b poi:followedBy ?sq . }
        }
        """
        result = execute_sparql_query(self.query_endpoint, query)
        df = format_results(result)
        if df.empty:
            return {'covisitation': 0, 'sequence': 0}
        return {
            'covisitation': int(df.iloc[0].get('covisitCount', 0)),
            'sequence': int(df.iloc[0].get('sequenceCount', 0)),
        }

    # ------------------------------------------------------------------ #
    # Export for Comparison
    # ------------------------------------------------------------------ #

    def export_for_comparison(self, trajectory_uri: str) -> dict:
        """Export semantic profile and recommendations for comparison."""
        profile = self.get_trajectory_profile(trajectory_uri)
        engagement = self.get_engagement_level(trajectory_uri)
        similar = self.get_similar_trajectories(trajectory_uri, min_score=0.0, top_k=10)
        recs = self.recommend_with_explanations(trajectory_uri, top_k=20)

        return {
            'trajectory': trajectory_uri,
            'profile': profile,
            'engagement': engagement,
            'similar_trajectories': similar,
            'recommendations': recs,
        }

    # ------------------------------------------------------------------ #
    # Counting / Stats
    # ------------------------------------------------------------------ #

    def count_materialized_triples(self) -> dict[str, int]:
        """Count materialized reasoning triples by predicate."""
        pred_values = ' '.join(f'<{p}>' for p in MATERIALIZED_PREDICATES)
        query = f"""
        SELECT ?pred (COUNT(*) AS ?count)
        WHERE {{
            ?s ?pred ?o .
            VALUES ?pred {{ {pred_values} }}
        }}
        GROUP BY ?pred
        ORDER BY DESC(?count)
        """
        result = execute_sparql_query(self.query_endpoint, query)
        df = format_results(result)
        if df.empty:
            return {}
        return {self._shorten_uri(row['pred']): int(row['count'])
                for _, row in df.iterrows()}

    def get_all_trajectories(self) -> list[str]:
        """Get all trajectory URIs in the KG."""
        query = PREFIXES + """
        SELECT DISTINCT ?traj WHERE {
            ?traj a datacron:Trajectory .
        }
        ORDER BY ?traj
        """
        result = execute_sparql_query(self.query_endpoint, query)
        df = format_results(result)
        if df.empty:
            return []
        return df['traj'].tolist()

    def get_all_users(self) -> list[str]:
        """Get all user URIs."""
        query = PREFIXES + """
        SELECT DISTINCT ?user WHERE {
            ?traj user:belongsToUser ?user .
        }
        ORDER BY ?user
        """
        result = execute_sparql_query(self.query_endpoint, query)
        df = format_results(result)
        if df.empty:
            return []
        return df['user'].tolist()
