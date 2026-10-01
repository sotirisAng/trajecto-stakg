"""
Semantic Reasoning Engine for Trajecto

SHACL validation + SPARQL materialization for trajectory reasoning.
Replaces ML-based feature vectors with SPARQL-based inference over
the existing knowledge graph.

Layers:
1. SHACL validation via Fuseki /shacl endpoint
2. SPARQL INSERT WHERE rules for relationships, engagement, similarity, preferences, recommendations
"""

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List
from collections import defaultdict

from .fuseki_config import query_endpoint, update_endpoint

if TYPE_CHECKING:
    import pandas as pd
from .fuseki_queries import execute_sparql_query, execute_sparql_update, format_results
from .shacl_validator import validate_before_import, construct_validation_graph

RULES_DIR = Path(__file__).resolve().parent.parent.parent / 'sparql' / 'semantic-rules'

PREFIXES = """
PREFIX trajecto: <http://w3id.org/trajecto#>
PREFIX poi:      <http://w3id.org/trajecto/poi#>
PREFIX patterns: <http://w3id.org/trajecto/patterns#>
PREFIX context:  <http://w3id.org/trajecto/context#>
PREFIX user:     <http://w3id.org/trajecto/user#>
PREFIX purpose:  <http://w3id.org/trajecto/purpose#>
PREFIX events:   <http://w3id.org/trajecto/events#>
PREFIX profiles: <http://w3id.org/trajecto/profiles#>
PREFIX dul:      <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#>
PREFIX datacron: <http://www.datacron-project.eu/datAcron#>
PREFIX rdf:      <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs:     <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:      <http://www.w3.org/2001/XMLSchema#>
"""

# Categories excluded from recommendations (transportation, utility, navigation)
# Mirrors EXCLUDED_CATEGORIES in context_recommender.py
EXCLUDED_CATEGORIES = {
    # Street furniture / utility
    'SeatingArea', 'PublicService', 'VendingMachine', 'WaterFountain',
    'Embassy',
    'ATM', 'Bank', 'PostOffice',
    'BusStation', 'TrainStation', 'Airport', 'Transportation', 'TransportationPOI',
    'TransitPOI', 'TransitStop',
    'ParkingArea',
    'Elevator', 'Escalator', 'Staircase',
    'Entrance', 'Exit', 'Intersection', 'Junction',
    'Transition', 'TransitionPOI', 'Waypoint', 'NavigationLandmark',
    'Restroom', 'RestRoom', 'ServicePOI',
    # Healthcare (utility, not a leisure/tourist rec)
    'Pharmacy', 'Hospital', 'Clinic', 'HealthcareFacility', 'MedicalFacility',
    'HealthcarePOI', 'DoctorsOffice', 'DentalOffice',
    # Business (not a consumer-leisure rec). ConferenceCenter intentionally
    # NOT listed — it is recommendable only when a scheduled event overlaps
    # the trajectory's time window (gated via event-match logic).
    'OfficeBuilding', 'GovernmentBuilding',
    'nan', 'None', 'Unknown', 'Uncategorized',
}

# Minimum rule coverage to include a recommendation (fraction of 5 rules)
MIN_RULE_COVERAGE = 2

# Similarity facet weights (3 scored facets; purpose is the blocking key, always 1.0 for batch pairs)
SIMILARITY_WEIGHTS = {
    'category_jaccard': 0.65,
    'navigation': 0.25,
    'temporal': 0.10,
}

# Predicates materialized by the reasoning rules — used for surgical cleanup
MATERIALIZED_PREDICATES = [
    'http://w3id.org/trajecto/poi#covisitedWith',
    'http://w3id.org/trajecto/poi#followedBy',
    'http://w3id.org/trajecto#hasEngagementLevel',
    'http://w3id.org/trajecto#hasJaccardCategorySimilarity',
    'http://w3id.org/trajecto#hasNavigationMatch',
    'http://w3id.org/trajecto#hasTemporalOverlap',
    'http://w3id.org/trajecto#withTrajectory',
    'http://w3id.org/trajecto#similarityScore',
    'http://w3id.org/trajecto/user#hasAffinityFor',
    # SR-20's preference nodes are removed through user:hasInferredPreference
    # in clear_reasoning_triples: user:preferenceScore is shared with the
    # Stage 8/9 profile snapshots, so deleting it by predicate erased them.
    # SR-40..44: aggregate properties
    'http://w3id.org/trajecto#hasMeaningfulInteractionCount',
    'http://w3id.org/trajecto#hasImmersiveInteractionCount',
    'http://w3id.org/trajecto#hasTotalDwellTime',
    'http://w3id.org/trajecto#hasUniqueCategoryCount',
    'http://w3id.org/trajecto#hasPOICount',
    # SR-41: trajectory-level interaction links for OWL reasoning
    'http://w3id.org/trajecto#includesInteraction',
    # SR-50..54: event instances (cleaned via rdf:type + trajecto:hasEvent)
    'http://w3id.org/trajecto#hasEvent',
    # KG-analytics materialization (helpers/kg_analytics.py)
    'http://w3id.org/trajecto#pageRank',
    'http://w3id.org/trajecto#betweennessCentrality',
    'http://w3id.org/trajecto#communityId',
    'http://w3id.org/trajecto#hasCentralityRole',
]

# Event-related predicates — cleaned by deleting event individuals and their properties
EVENT_CLASSES = [
    'http://w3id.org/trajecto/events#EngagementEscalation',
    'http://w3id.org/trajecto/events#EngagementDrop',
    'http://w3id.org/trajecto/events#CategoryDiscovery',
    'http://w3id.org/trajecto/events#CategoryTransition',
    'http://w3id.org/trajecto/events#DominantEncounter',
]

EVENT_PREDICATES = [
    'http://w3id.org/trajecto/events#atPosition',
    'http://w3id.org/trajecto/events#eventSequenceIndex',
    'http://w3id.org/trajecto/events#fromInteractionType',
    'http://w3id.org/trajecto/events#toInteractionType',
    'http://w3id.org/trajecto/events#discoveredCategory',
    'http://w3id.org/trajecto/events#fromCategory',
    'http://w3id.org/trajecto/events#toCategory',
]


_URI_PREFIXES = {
    'http://w3id.org/trajecto#': 'trajecto:',
    'http://w3id.org/trajecto/poi#': 'poi:',
    'http://w3id.org/trajecto/patterns#': 'patterns:',
    'http://w3id.org/trajecto/context#': 'context:',
    'http://w3id.org/trajecto/user#': 'user:',
    'http://w3id.org/trajecto/purpose#': 'purpose:',
    'http://w3id.org/trajecto/events#': 'events:',
    'http://w3id.org/trajecto/profiles#': 'profiles:',
    'http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#': 'dul:',
    'http://www.datacron-project.eu/datAcron#': 'datacron:',
    'http://www.w3.org/1999/02/22-rdf-syntax-ns#': 'rdf:',
    'http://www.w3.org/2000/01/rdf-schema#': 'rdfs:',
    'http://www.w3.org/2001/XMLSchema#': 'xsd:',
}


class SemanticReasoningEngine:
    """SHACL validation + SPARQL materialization for trajectory reasoning."""

    def __init__(self, endpoint=None, upd_endpoint=None):
        self.query_endpoint = endpoint or query_endpoint
        self.update_endpoint = upd_endpoint or update_endpoint

    @staticmethod
    def _shorten_uri(uri) -> str:
        """Shorten a full URI to prefixed form; return input unchanged if not a URI."""
        if uri is None:
            return ''
        s = str(uri)
        for full, short in _URI_PREFIXES.items():
            if s.startswith(full):
                return short + s[len(full):]
        return s

    # ------------------------------------------------------------------ #
    # SHACL Validation (delegates to shacl_validator module)
    # ------------------------------------------------------------------ #

    def run_validation(self, stage: str = 'stage-S1-reasoning') -> Dict:
        """Validate current KG state against stage-specific SHACL shapes.

        Fetches relevant triples via batched CONSTRUCTs (one per graph
        pattern), validates locally with pyshacl.
        """
        proposed = construct_validation_graph(stage)
        return validate_before_import(
            proposed, stage, halt_on_violations=False
        )

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

    def execute_rule(self, rule_file: str, params: Dict[str, str] = None) -> bool:
        """Execute a SPARQL UPDATE rule from file. Returns success."""
        query = self._load_rule(rule_file)
        if params:
            for key, value in params.items():
                query = query.replace(f'${key}', value)
        return execute_sparql_update(self.update_endpoint, query)

    def execute_relationship_rules(self) -> Dict[str, bool]:
        """Execute SR-05, SR-06: POI co-visitation and sequence."""
        results = {}
        for rule in ['SR-05-covisitation.sparql',
                      'SR-06-sequence-succession.sparql']:
            name = rule.split('.')[0]
            results[name] = self.execute_rule(rule)
        return results

    def execute_engagement_rules(self) -> Dict[str, bool]:
        """Execute SR-10: engagement levels."""
        return {'SR-10': self.execute_rule('SR-10-engagement-levels.sparql')}

    def execute_similarity_batch(self) -> Dict[str, bool]:
        """Batch similarity: fetch raw mappings, compute in Python, INSERT results.

        Replaces the old approach of running SR-11..SR-13 as monolithic SPARQL
        UPDATEs, which caused O(n^2) deep-join explosions in Fuseki.
        """
        results = {}

        # --- Fetch raw mappings ---
        traj_cats = self._fetch_trajectory_categories()
        traj_nav = self._fetch_trajectory_navigation()
        traj_temp = self._fetch_trajectory_temporal()
        traj_purpose = self._fetch_trajectory_purposes()

        # Build purpose blocks: only compare trajectories sharing a purpose
        purpose_blocks = defaultdict(set)
        for traj, purposes in traj_purpose.items():
            for p in purposes:
                purpose_blocks[p].add(traj)

        # Collect unique pairs via blocking key
        pairs = set()
        for block in purpose_blocks.values():
            trajs = sorted(block)
            for i in range(len(trajs)):
                for j in range(i + 1, len(trajs)):
                    pairs.add((trajs[i], trajs[j]))

        # --- Compute scores ---
        cat_scores = {}   # (a, b) -> float
        nav_scores = {}
        temp_scores = {}

        for a, b in pairs:
            # Category Jaccard
            cats_a = traj_cats.get(a, set())
            cats_b = traj_cats.get(b, set())
            union = cats_a | cats_b
            if union:
                cat_scores[(a, b)] = len(cats_a & cats_b) / len(union)

            # Navigation match (1.0 if any shared pattern)
            nav_a = traj_nav.get(a, set())
            nav_b = traj_nav.get(b, set())
            if nav_a & nav_b:
                nav_scores[(a, b)] = 1.0

            # Temporal Jaccard
            temp_a = traj_temp.get(a, set())
            temp_b = traj_temp.get(b, set())
            t_union = temp_a | temp_b
            if t_union:
                score = len(temp_a & temp_b) / len(t_union)
                if score > 0.0:
                    temp_scores[(a, b)] = score

        # --- INSERT results in batches ---
        results['SR-11-category-jaccard'] = self._insert_similarity_triples(
            cat_scores, 'trajecto:hasJaccardCategorySimilarity', threshold=0.3)
        results['SR-12-navigation-match'] = self._insert_similarity_triples(
            nav_scores, 'trajecto:hasNavigationMatch', threshold=0.0)
        results['SR-13-temporal-overlap'] = self._insert_similarity_triples(
            temp_scores, 'trajecto:hasTemporalOverlap', threshold=0.0)

        return results

    # ---- Fetch helpers for batch similarity ----

    def _fetch_trajectory_categories(self) -> Dict[str, set]:
        """Trajectory -> set of POI category URIs (MeaningfulInteraction only)."""
        query = PREFIXES + """
        SELECT ?traj ?cat WHERE {
            ?pos dul:isPartOf ?traj .
            ?pos trajecto:hasInteraction ?int .
            ?int a poi:MeaningfulInteraction .
            ?int poi:interactsWith ?poi .
            ?poi a ?cat .
            FILTER(?cat != trajecto:PointOfInterest)
            FILTER(?cat != poi:SignificantPOI)
        }
        """
        result = execute_sparql_query(self.query_endpoint, query)
        df = format_results(result)
        mapping = defaultdict(set)
        for _, row in df.iterrows():
            mapping[row['traj']].add(row['cat'])
        return dict(mapping)

    def _fetch_trajectory_navigation(self) -> Dict[str, set]:
        """Trajectory -> set of navigation pattern URIs."""
        query = PREFIXES + """
        SELECT ?traj ?pattern WHERE {
            ?traj patterns:exhibitsPattern ?pattern .
        }
        """
        result = execute_sparql_query(self.query_endpoint, query)
        df = format_results(result)
        mapping = defaultdict(set)
        for _, row in df.iterrows():
            mapping[row['traj']].add(row['pattern'])
        return dict(mapping)

    def _fetch_trajectory_temporal(self) -> Dict[str, set]:
        """Trajectory -> set of temporal context URIs."""
        query = PREFIXES + """
        SELECT ?traj ?ctx WHERE {
            ?traj context:hasTemporalContext ?ctx .
        }
        """
        result = execute_sparql_query(self.query_endpoint, query)
        df = format_results(result)
        mapping = defaultdict(set)
        for _, row in df.iterrows():
            mapping[row['traj']].add(row['ctx'])
        return dict(mapping)

    def _fetch_trajectory_purposes(self) -> Dict[str, set]:
        """Trajectory -> set of purpose URIs (blocking key)."""
        query = PREFIXES + """
        SELECT ?traj ?purpose WHERE {
            ?traj trajecto:hasPurpose ?purpose .
        }
        """
        result = execute_sparql_query(self.query_endpoint, query)
        df = format_results(result)
        mapping = defaultdict(set)
        for _, row in df.iterrows():
            mapping[row['traj']].add(row['purpose'])
        return dict(mapping)

    def _insert_similarity_triples(self, scores: Dict[tuple, float],
                                   predicate: str, threshold: float = 0.0,
                                   batch_size: int = 500) -> bool:
        """Insert similarity score triples in batches."""
        filtered = [(pair, s) for pair, s in scores.items() if s > threshold]
        if not filtered:
            return True

        success = True
        for i in range(0, len(filtered), batch_size):
            batch = filtered[i:i + batch_size]
            triples = []
            for (a, b), score in batch:
                triples.append(
                    f'<{a}> {predicate} [ '
                    f'trajecto:withTrajectory <{b}> ; '
                    f'trajecto:similarityScore {score:.6f} ] .')
            insert = PREFIXES + "INSERT DATA {\n" + "\n".join(triples) + "\n}"
            if not execute_sparql_update(self.update_endpoint, insert):
                success = False
        return success

    def execute_aggregate_rules(self) -> Dict[str, bool]:
        """Execute SR-40..44: aggregate properties for OWL reasoning."""
        results = {}
        for rule in ['SR-40-aggregate-interaction-counts.sparql',
                      'SR-41-trajectory-interaction-links.sparql',
                      'SR-42-aggregate-dwell-time.sparql',
                      'SR-43-aggregate-category-count.sparql',
                      'SR-44-aggregate-poi-count.sparql']:
            name = rule.split('.')[0]
            results[name] = self.execute_rule(rule)
        return results

    def execute_event_rules(self) -> Dict[str, bool]:
        """Execute SR-50..54: trajectory event materialization."""
        results = {}
        for rule in ['SR-50-engagement-escalation.sparql',
                      'SR-51-engagement-drop.sparql',
                      'SR-52-category-discovery.sparql',
                      'SR-53-category-transition.sparql',
                      'SR-54-dominant-encounter.sparql']:
            name = rule.split('.')[0]
            results[name] = self.execute_rule(rule)
        return results

    def execute_preference_rules(self) -> Dict[str, bool]:
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

    def run_full_pipeline(self, include_similarity_batch=False,
                          include_aggregates=False, include_events=False) -> Dict:
        """
        Execute full reasoning pipeline:
        1. SHACL validation
        2. Relationships (SR-05, SR-06)
        3. Engagement (SR-10)
        4. Preferences (SR-20, SR-21)
        5. Optionally: batch similarity (SR-11..SR-13)
        6. Optionally: aggregate properties (SR-40..44)
        7. Optionally: event materialization (SR-50..54)
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
            print("\nStep 5: Batch similarity (SR-11..SR-13)...")
            results['similarity'] = self.execute_similarity_batch()

        if include_aggregates:
            print("\nStep 6: Aggregate properties (SR-40..44)...")
            results['aggregates'] = self.execute_aggregate_rules()

        if include_events:
            print("\nStep 7: Event materialization (SR-50..54)...")
            results['events'] = self.execute_event_rules()

        return results

    def clear_reasoning_triples(self) -> bool:
        """Remove all triples materialized by reasoning rules.

        Surgical deletion: only removes triples whose predicate is in
        MATERIALIZED_PREDICATES, avoiding damage to pre-existing triples
        in the poi:, trajecto:, or user: namespaces.

        Also removes event individuals and their properties.
        """
        success = True

        # 1. Delete event individual properties (events: namespace predicates)
        if EVENT_PREDICATES:
            evt_pred_values = ' '.join(f'<{p}>' for p in EVENT_PREDICATES)
            evt_query = f"""
            DELETE {{ ?s ?p ?o }}
            WHERE {{
                VALUES ?p {{ {evt_pred_values} }}
                ?s ?p ?o .
            }}
            """
            if not execute_sparql_update(self.update_endpoint, evt_query):
                success = False

        # 2. Delete event rdf:type triples
        if EVENT_CLASSES:
            for cls in EVENT_CLASSES:
                type_query = f"""
                DELETE WHERE {{
                    ?s a <{cls}> .
                }}
                """
                if not execute_sparql_update(self.update_endpoint, type_query):
                    success = False

        # 3. Delete SR-20 inferred-preference nodes (only the nodes, so the
        #    profile snapshots that share their property names survive)
        inferred_query = """
        PREFIX user: <http://w3id.org/trajecto/user#>
        DELETE { ?u user:hasInferredPreference ?n . ?n ?p ?o }
        WHERE  { ?u user:hasInferredPreference ?n . OPTIONAL { ?n ?p ?o } }
        """
        if not execute_sparql_update(self.update_endpoint, inferred_query):
            success = False

        # 4. Delete standard materialized predicates
        pred_values = ' '.join(f'<{p}>' for p in MATERIALIZED_PREDICATES)
        query = f"""
        DELETE {{ ?s ?p ?o }}
        WHERE {{
            VALUES ?p {{ {pred_values} }}
            ?s ?p ?o .
        }}
        """
        if not execute_sparql_update(self.update_endpoint, query):
            success = False

        return success

    # ------------------------------------------------------------------ #
    # Trajectory Profile Queries
    # ------------------------------------------------------------------ #

    def get_trajectory_profile(self, uri: str) -> Dict:
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
            UNION
            {{
                ?traj context:hasTemporalContext ?val .
                BIND("temporalContext" AS ?prop)
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

        # Flatten singletons
        out = {}
        for k, v in profile.items():
            out[k] = v[0] if len(v) == 1 else v
        return out

    def get_engagement_distribution(self) -> Dict[str, int]:
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

    def get_purpose_distribution(self) -> Dict[str, int]:
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

    def get_profile_groups(self) -> List[Dict]:
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

    def get_cross_tabulation(self) -> 'pd.DataFrame':
        """Cross-tabulate purpose vs navigation style."""
        import pandas as pd
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

    def get_engagement_by_purpose(self) -> 'pd.DataFrame':
        """Cross-tabulate engagement level vs purpose."""
        import pandas as pd
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

    def spot_check_profile(self, uri: str) -> Dict:
        """Verify a trajectory's profile against its actual interactions."""
        profile = self.get_trajectory_profile(uri)

        # Get actual interactions
        query = PREFIXES + f"""
        SELECT ?poi ?cat ?intType ?dwell WHERE {{
            ?pos dul:isPartOf <{uri}> .
            ?pos trajecto:hasInteraction ?int .
            ?int a ?intType .
            ?int poi:interactsWith ?poi .
            ?poi a ?cat .
            FILTER(?cat != trajecto:PointOfInterest)
            FILTER(?cat != poi:SignificantPOI)
            FILTER(?intType IN (poi:PassingInteraction, poi:MeaningfulInteraction,
                                poi:ImmersiveInteraction, poi:DominantInteraction))
            OPTIONAL {{ ?int poi:hasDwellTimeInSeconds ?dwell . }}
        }}
        ORDER BY ?cat
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
                                  top_k: int = 10) -> List[Dict]:
        """
        On-demand faceted similarity: compute similarity between the input
        trajectory and all others sharing a purpose, returning top_k above min_score.

        3 scored facets (category_jaccard, navigation, temporal).
        Purpose is the blocking key, so only trajectories sharing a purpose are compared.
        """
        query = PREFIXES + f"""
        SELECT ?other ?catJaccard ?navMatch ?temporalOverlap ?aggregate
        WHERE {{
            BIND(<{uri}> AS ?input)

            # Blocking: shared purpose
            ?input trajecto:hasPurpose ?sharedPurpose .
            ?other trajecto:hasPurpose ?sharedPurpose .
            FILTER(?other != ?input)

            # Navigation match — EXISTS is correlated per (input, other)
            # pair; the previous LIMIT-1 subquery evaluated bottom-up and
            # kept one arbitrary pair, zeroing the facet for all others.
            BIND(IF(EXISTS {{
                ?input patterns:exhibitsPattern ?navP .
                ?other patterns:exhibitsPattern ?navP .
            }}, 1.0, 0.0) AS ?navMatch)

            # Temporal overlap (Jaccard)
            OPTIONAL {{
                SELECT ?input ?other
                (xsd:float(COUNT(DISTINCT ?sharedCtx)) / xsd:float(IF(COUNT(DISTINCT ?anyCtx) < 1, 1, COUNT(DISTINCT ?anyCtx))) AS ?tempScore)
                WHERE {{
                    {{
                        ?input context:hasTemporalContext ?ctxI .
                        ?other context:hasTemporalContext ?ctxI .
                        BIND(?ctxI AS ?sharedCtx)
                        BIND(?ctxI AS ?anyCtx)
                    }}
                    UNION
                    {{
                        ?input context:hasTemporalContext ?ctxA .
                        BIND(?ctxA AS ?anyCtx)
                    }}
                    UNION
                    {{
                        ?other context:hasTemporalContext ?ctxB .
                        BIND(?ctxB AS ?anyCtx)
                    }}
                }}
                GROUP BY ?input ?other
            }}
            BIND(COALESCE(?tempScore, 0.0) AS ?temporalOverlap)

            # Category Jaccard (shared / union of category types)
            OPTIONAL {{
                SELECT ?input ?other
      (xsd:float(COUNT(DISTINCT ?sharedCat)) / xsd:float(IF(COUNT(DISTINCT ?anyCat) < 1, 1, COUNT(DISTINCT ?anyCat))) AS ?catScore)
                WHERE {{
                    {{
                        ?posI dul:isPartOf ?input .
                        ?posI trajecto:hasInteraction ?intI .
                        ?intI a poi:MeaningfulInteraction .
                        ?intI poi:interactsWith ?poiI .
                        ?poiI a ?catI .
                        FILTER(?catI != trajecto:PointOfInterest)
                        FILTER(?catI != poi:SignificantPOI)

                        ?posO dul:isPartOf ?other .
                        ?posO trajecto:hasInteraction ?intO .
                        ?intO a poi:MeaningfulInteraction .
                        ?intO poi:interactsWith ?poiO .
                        ?poiO a ?catI .

                        BIND(?catI AS ?sharedCat)
                        BIND(?catI AS ?anyCat)
                    }}
                    UNION
                    {{
                        ?posIA dul:isPartOf ?input .
                        ?posIA trajecto:hasInteraction ?intIA .
                        ?intIA a poi:MeaningfulInteraction .
                        ?intIA poi:interactsWith ?poiIA .
                        ?poiIA a ?catIA .
                        FILTER(?catIA != trajecto:PointOfInterest)
                        FILTER(?catIA != poi:SignificantPOI)
                        BIND(?catIA AS ?anyCat)
                    }}
                    UNION
                    {{
                        ?posOA dul:isPartOf ?other .
                        ?posOA trajecto:hasInteraction ?intOA .
                        ?intOA a poi:MeaningfulInteraction .
                        ?intOA poi:interactsWith ?poiOA .
                        ?poiOA a ?catOA .
                        FILTER(?catOA != trajecto:PointOfInterest)
                        FILTER(?catOA != poi:SignificantPOI)
                        BIND(?catOA AS ?anyCat)
                    }}
                }}
                GROUP BY ?input ?other
            }}
            BIND(COALESCE(?catScore, 0.0) AS ?catJaccard)

            # Weighted aggregate
            BIND(
                {SIMILARITY_WEIGHTS['category_jaccard']} * ?catJaccard +
                {SIMILARITY_WEIGHTS['navigation']} * ?navMatch +
                {SIMILARITY_WEIGHTS['temporal']} * ?temporalOverlap
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
                    'temporal': float(row['temporalOverlap']),
                },
            })
        return similar

    def get_similarity_distribution(self) -> 'pd.DataFrame':
        """Get distribution of materialized similarity scores (batch mode)."""
        import pandas as pd
        query = PREFIXES + """
        SELECT ?trajA ?trajB ?catJ ?nav ?temp WHERE {
            ?trajA trajecto:hasJaccardCategorySimilarity ?cj .
            ?cj trajecto:withTrajectory ?trajB .
            ?cj trajecto:similarityScore ?catJ .

            OPTIONAL {
                ?trajA trajecto:hasNavigationMatch ?nm .
                ?nm trajecto:withTrajectory ?trajB .
                ?nm trajecto:similarityScore ?nav .
            }
            OPTIONAL {
                ?trajA trajecto:hasTemporalOverlap ?to .
                ?to trajecto:withTrajectory ?trajB .
                ?to trajecto:similarityScore ?temp .
            }
        }
        """
        result = execute_sparql_query(self.query_endpoint, query)
        df = format_results(result)
        if df.empty:
            return pd.DataFrame()

        for col in ['catJ', 'nav', 'temp']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        w = SIMILARITY_WEIGHTS
        df['aggregate'] = (
            w['category_jaccard'] * df['catJ'] +
            w['navigation'] * df['nav'] +
            w['temporal'] * df['temp']
        )
        return df

    # ------------------------------------------------------------------ #
    # User Preferences
    # ------------------------------------------------------------------ #

    def get_inferred_preferences(self, user_uri: str) -> List[Dict]:
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

    def get_user_affinities(self, user_uri: str) -> List[str]:
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

    def recommend(self, trajectory_uri: str, top_k: int = 10) -> List[Dict]:
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
                # Capture rule-specific details
                if 'prefScore' in row:
                    poi_data[poi]['details']['prefScore'] = float(row['prefScore'])
                if 'sourcePoi' in row:
                    poi_data[poi]['details']['sourcePoi'] = row['sourcePoi']
                if 'dominantPoi' in row:
                    poi_data[poi]['details']['dominantPoi'] = row['dominantPoi']
                if 'similarTraj' in row:
                    poi_data[poi]['details'].setdefault('similarTrajs', set())
                    poi_data[poi]['details']['similarTrajs'].add(row['similarTraj'])

        # Rank by rule coverage (more rules = higher score)
        ranked = []
        for poi, data in poi_data.items():
            categories = list(data['categories'])

            # Filter: excluded categories (transportation, utility, navigation)
            cat_names = {c.split(':')[-1] if ':' in c else c for c in categories}
            if cat_names & EXCLUDED_CATEGORIES:
                continue
            if not cat_names - {''}:
                continue

            rule_count = len(set(data['rules']))

            # Filter: minimum rule coverage
            if rule_count < MIN_RULE_COVERAGE:
                continue

            score = rule_count / len(rules)  # Normalized 0-1
            ranked.append({
                'poi': poi,
                'poiLabel': self._shorten_uri(poi),
                'categories': categories,
                'score': score,
                'ruleCoverage': rule_count,
                'rules': list(set(data['rules'])),
                'details': {k: (list(v) if isinstance(v, set) else v)
                            for k, v in data['details'].items()},
            })

        ranked.sort(key=lambda x: (-x['score'], x['poi']))
        return ranked[:top_k]

    def recommend_with_explanations(self, trajectory_uri: str,
                                     top_k: int = 10) -> List[Dict]:
        """Recommend with full rule provenance and explanations."""
        recs = self.recommend(trajectory_uri, top_k)

        rule_explanations = {
            'SR-30:profile-collaborative': 'From trajectories with similar semantic profile',
            'SR-31:preference-gap': 'Matches inferred user preferences (unvisited category)',
            'SR-32:sequence-continuation': 'Next in common visit sequence',
            'SR-33:covisitation': 'Co-visited with dominant POI in other trajectories',
            'SR-34:purpose-alignment': 'Aligns with trajectory purpose',
        }

        for rec in recs:
            rec['explanations'] = [
                rule_explanations.get(r, r) for r in rec['rules']
            ]

        return recs

    def get_rule_recommendations(self, trajectory_uri: str) -> Dict[str, set]:
        """Execute SR-30..34 and return per-rule POI URI sets (no aggregation).

        Returns:
            {'SR-30': {poi_uri, ...}, 'SR-31': {poi_uri, ...}, ...}
        """
        rules = {
            'SR-30': 'SR-30-profile-collaborative.sparql',
            'SR-31': 'SR-31-preference-gap.sparql',
            'SR-32': 'SR-32-sequence-continuation.sparql',
            'SR-33': 'SR-33-covisitation-recs.sparql',
            'SR-34': 'SR-34-purpose-alignment.sparql',
        }
        per_rule = {}
        for rule_name, rule_file in rules.items():
            query = self._load_rule(rule_file)
            query = query.replace('$TRAJECTORY_URI', trajectory_uri)
            result = execute_sparql_query(self.query_endpoint, query)
            df = format_results(result)
            if df.empty:
                per_rule[rule_name] = set()
            else:
                per_rule[rule_name] = set(df['poi'].tolist())
        return per_rule

    def get_rule_yield(self, trajectory_uri: str) -> Dict[str, int]:
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

    def get_covisitation_links(self, poi_uri: str = None, limit: int = 50) -> 'pd.DataFrame':
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

    def get_sequence_links(self, poi_uri: str = None, limit: int = 50) -> 'pd.DataFrame':
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

    def get_relationship_stats(self) -> Dict:
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

    def export_for_comparison(self, trajectory_uri: str) -> Dict:
        """Export semantic profile and recommendations for comparison with ML."""
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

    def count_materialized_triples(self) -> Dict[str, int]:
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

    def get_all_trajectories(self) -> List[str]:
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

    def get_all_users(self) -> List[str]:
        """Get all user URIs."""
        query = PREFIXES + """
        SELECT DISTINCT ?user WHERE {
            ?user trajecto:formsTrajectory ?traj  .
        }
        ORDER BY ?user
        """
        result = execute_sparql_query(self.query_endpoint, query)
        df = format_results(result)
        if df.empty:
            return []
        return df['user'].tolist()
