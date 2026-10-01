"""
SHACL Validation for Trajecto Pipeline

Validates proposed triples against stage-specific SHACL shapes before
they enter the knowledge graph. Uses pyshacl (local validation) —
no dependency on Fuseki's /shacl endpoint.

Usage:
    # Before bulk .ttl upload (notebook 2)
    report = validate_ttl_file('../files/rdf/trajectory_data.ttl')

    # Before SPARQL INSERT DATA (notebooks 3-9)
    report = validate_before_import(proposed_graph, 'stage-06-navigation',
                                    context_construct='CONSTRUCT ...')

    # Before SR-* INSERT WHERE (S1)
    proposed = preview_insert_where(insert_sparql)
    report = validate_before_import(proposed, 'stage-S1-reasoning')
"""

import re
from pathlib import Path
from typing import Dict, Optional

from pyshacl import validate as shacl_validate
from rdflib import Graph
from rdflib.namespace import RDF, SH

from .fuseki_config import query_endpoint

SHACL_DIR = Path(__file__).resolve().parent.parent.parent / 'shacl'
SHACL_STAGES_DIR = SHACL_DIR / 'stages'

PREFIXES = """
PREFIX trajecto: <http://w3id.org/trajecto#>
PREFIX poi:      <http://w3id.org/trajecto/poi#>
PREFIX patterns: <http://w3id.org/trajecto/patterns#>
PREFIX context:  <http://w3id.org/trajecto/context#>
PREFIX user:     <http://w3id.org/trajecto/user#>
PREFIX purpose:  <http://w3id.org/trajecto/purpose#>
PREFIX events:   <http://w3id.org/trajecto/events#>
PREFIX dul:      <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#>
PREFIX datacron: <http://www.datacron-project.eu/datAcron#>
PREFIX rdf:      <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs:     <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:      <http://www.w3.org/2001/XMLSchema#>
"""

_URI_PREFIXES = {
    'http://www.w3.org/ns/shacl#': 'sh:',
    'http://w3id.org/trajecto#': 'trajecto:',
    'http://w3id.org/trajecto/poi#': 'poi:',
    'http://w3id.org/trajecto/patterns#': 'patterns:',
    'http://w3id.org/trajecto/context#': 'context:',
    'http://w3id.org/trajecto/user#': 'user:',
    'http://w3id.org/trajecto/purpose#': 'purpose:',
    'http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#': 'dul:',
    'http://www.datacron-project.eu/datAcron#': 'datacron:',
}


class ValidationError(Exception):
    """Raised when SHACL validation finds critical violations."""

    def __init__(self, report: Dict):
        self.report = report
        violations = report.get('violations', 0)
        super().__init__(f"SHACL validation failed: {violations} violation(s)")


def validate_before_import(
    proposed: Graph,
    stage: str,
    context_construct: Optional[str] = None,
    halt_on_violations: bool = True,
) -> Dict:
    """Validate proposed triples against stage shapes before writing to KG.

    Args:
        proposed: rdflib Graph of triples to be inserted.
        stage: Stage name, e.g. 'stage-02-base-rdf'. Maps to
               shacl/stages/{stage}.ttl.
        context_construct: Optional SPARQL CONSTRUCT query to fetch existing
                          KG triples needed for validation context.
                          Merged with proposed before validation.
        halt_on_violations: If True, raise ValidationError on sh:Violation.

    Returns:
        Dict with keys: conforms, violations, warnings, total, results.
    """
    shapes_path = SHACL_STAGES_DIR / f'{stage}.ttl'
    if not shapes_path.exists():
        raise FileNotFoundError(f"SHACL shapes not found: {shapes_path}")

    shapes = Graph().parse(shapes_path)
    data = Graph()
    data += proposed
    if context_construct:
        data += construct_from_fuseki(context_construct)

    conforms, report_graph, report_text = shacl_validate(
        data, shacl_graph=shapes, abort_on_first=False
    )
    result = _parse_report(conforms, report_graph)

    _print_summary(stage, result)

    if halt_on_violations and result['violations'] > 0:
        raise ValidationError(result)
    return result


def validate_ttl_file(
    file_path: str,
    stage: str = 'stage-02-base-rdf',
    halt_on_violations: bool = True,
) -> Dict:
    """Validate a Turtle file before uploading to Fuseki.

    No context_construct needed — the file contains the complete proposed data.
    """
    proposed = Graph().parse(file_path, format='turtle')
    print(f"Parsed {len(proposed)} triples from {Path(file_path).name}")
    return validate_before_import(
        proposed, stage, halt_on_violations=halt_on_violations
    )


def preview_insert_where(insert_sparql: str) -> Graph:
    """Convert an INSERT WHERE query to CONSTRUCT WHERE and execute it.

    Extracts the INSERT { ... } block and the WHERE { ... } block,
    rewrites as CONSTRUCT { ... } WHERE { ... }, executes against Fuseki,
    and returns the result as an rdflib Graph.

    Used for SR-* rules in S1 to preview what would be inserted.
    """
    # Extract INSERT { ... } and WHERE { ... }
    # Handle INSERT { ... } WHERE { ... } pattern
    match = re.search(
        r'INSERT\s*\{(.*?)\}\s*WHERE\s*\{(.*)\}',
        insert_sparql,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        raise ValueError("Could not parse INSERT WHERE query")

    insert_block = match.group(1).strip()
    where_block = match.group(2).strip()

    # Extract prefixes from original query
    prefix_lines = re.findall(r'PREFIX\s+\S+\s+<[^>]+>', insert_sparql, re.IGNORECASE)
    prefix_block = '\n'.join(prefix_lines)

    construct_query = f"{prefix_block}\nCONSTRUCT {{\n{insert_block}\n}} WHERE {{\n{where_block}\n}}"

    return construct_from_fuseki(construct_query)


def construct_from_fuseki(construct_sparql: str) -> Graph:
    """Execute a SPARQL CONSTRUCT query against Fuseki and return an rdflib Graph."""
    import requests
    from urllib.parse import urlencode

    # Add prefixes if not present
    if 'PREFIX' not in construct_sparql.upper():
        construct_sparql = PREFIXES + construct_sparql

    response = requests.post(
        query_endpoint,
        data=urlencode({'query': construct_sparql}),
        headers={
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'text/turtle',
        },
        timeout=600,
    )
    response.raise_for_status()

    g = Graph()
    if response.text.strip():
        g.parse(data=response.text, format='turtle')
    return g


# ---------------------------------------------------------------------------
# Atomic CONSTRUCT queries — one per independent graph pattern.
#
# Each query touches a single subject type with at most one OPTIONAL,
# so Fuseki never builds a cross-pattern Cartesian product.
#
# RawPositions are deliberately excluded: potentially millions of triples
# that dominate runtime. Position validation is covered at stage-02 via
# validate_ttl_file(). Side effect: the inverse-path minCount check on
# Trajectory (sh:minCount 1 on [^dul:isPartOf]) will raise Warnings for
# every trajectory. Acceptable — severity is Warning, not Violation.
# ---------------------------------------------------------------------------

_CONSTRUCTS = {
    'trajectories': """CONSTRUCT { ?t a datacron:Trajectory }
        WHERE { ?t a datacron:Trajectory }""",

    'temporal_context': """CONSTRUCT { ?t context:hasTemporalContext ?v }
        WHERE { ?t a datacron:Trajectory ; context:hasTemporalContext ?v }""",

    'navigation_patterns': """CONSTRUCT { ?t patterns:exhibitsPattern ?v }
        WHERE { ?t a datacron:Trajectory ; patterns:exhibitsPattern ?v }""",

    'purpose': """CONSTRUCT { ?t trajecto:hasPurpose ?v }
        WHERE { ?t a datacron:Trajectory ; trajecto:hasPurpose ?v }""",

    'engagement': """CONSTRUCT { ?t trajecto:hasEngagementLevel ?v }
        WHERE { ?t a datacron:Trajectory ; trajecto:hasEngagementLevel ?v }""",

    'interactions': """CONSTRUCT {
        ?i a poi:POIInteraction ;
           poi:interactsWith ?p ;
           poi:hasDwellTimeInSeconds ?d .
    } WHERE {
        ?i a poi:POIInteraction ; poi:interactsWith ?p .
        OPTIONAL { ?i poi:hasDwellTimeInSeconds ?d }
    }""",

    'pois': """CONSTRUCT { ?p a trajecto:PointOfInterest }
        WHERE { ?p a trajecto:PointOfInterest }""",

    # Materializes a direct trajectory -> interaction edge ONLY for dominant interactions,
    # with the DominantInteraction type asserted. Lets ex:DominantSingletonShape walk a
    # simple trajecto:hasInteraction path and count how many carry rdf:type DominantInteraction
    # without requiring RawPosition triples in the validation graph.
    'dominant_interactions': """CONSTRUCT {
        ?t trajecto:hasInteraction ?i .
        ?i a poi:DominantInteraction .
    } WHERE {
        ?t a datacron:Trajectory .
        ?pos dul:isPartOf ?t ;
             trajecto:hasInteraction ?i .
        ?i a poi:DominantInteraction .
    }""",

    'user_profiles': """CONSTRUCT {
        ?p a user:UserProfile ; user:profileCreated ?c .
    } WHERE {
        ?p a user:UserProfile .
        OPTIONAL { ?p user:profileCreated ?c }
    }""",
}

# Which atomic queries each stage needs. Cumulative by design.
_STAGE_QUERIES = {
    'stage-02-base-rdf': [
        'trajectories', 'interactions', 'pois',
    ],
    'stage-03-enrichment': [
        'trajectories', 'temporal_context', 'interactions', 'pois',
    ],
    'stage-05-poi-scoring': [
        'trajectories', 'temporal_context', 'interactions', 'pois',
        'dominant_interactions',
    ],
    'stage-06-navigation': [
        'trajectories', 'temporal_context', 'navigation_patterns',
        'interactions', 'pois',
    ],
    'stage-07-classification': [
        'trajectories', 'temporal_context', 'navigation_patterns', 'purpose',
        'interactions', 'pois',
    ],
    'stage-08-user-profiles': [
        'trajectories', 'temporal_context', 'navigation_patterns', 'purpose',
        'interactions', 'pois', 'user_profiles',
    ],
    'stage-S1-reasoning': [
        'trajectories', 'temporal_context', 'navigation_patterns', 'purpose',
        'engagement', 'interactions', 'pois', 'user_profiles',
    ],
}


def construct_validation_graph(stage: str, verbose: bool = True) -> Graph:
    """Fetch validation context for a stage as a merged rdflib Graph.

    Each independent graph pattern is fetched as a separate CONSTRUCT
    query to avoid Cartesian-product blowup in Fuseki, then unioned
    in Python.
    """
    query_keys = _STAGE_QUERIES.get(stage)
    if query_keys is None:
        raise ValueError(
            f"Unknown stage '{stage}'. Known: {', '.join(_STAGE_QUERIES)}"
        )

    merged = Graph()
    for i, key in enumerate(query_keys, 1):
        g = construct_from_fuseki(_CONSTRUCTS[key])
        if verbose:
            print(f"  CONSTRUCT {i}/{len(query_keys)} ({key}): {len(g)} triples")
        merged += g
    return merged


# Backwards compatibility alias
construct_stage03_graph = lambda verbose=True: construct_validation_graph(
    'stage-03-enrichment', verbose=verbose
)


def _parse_report(conforms: bool, report_graph: Graph) -> Dict:
    """Parse a pyshacl validation report graph into a structured dict."""
    results = []

    for report_node in report_graph.subjects(RDF.type, SH.ValidationResult):
        severity_uri = report_graph.value(report_node, SH.resultSeverity)
        message = report_graph.value(report_node, SH.resultMessage)
        focus = report_graph.value(report_node, SH.focusNode)
        path = report_graph.value(report_node, SH.resultPath)

        results.append({
            'severity': _shorten_uri(str(severity_uri)) if severity_uri else '',
            'message': str(message) if message else '',
            'focusNode': _shorten_uri(str(focus)) if focus else '',
            'path': _shorten_uri(str(path)) if path else '',
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


def _shorten_uri(uri: str) -> str:
    """Shorten a full URI to prefixed form."""
    if not uri:
        return ''
    for full, short in _URI_PREFIXES.items():
        if uri.startswith(full):
            return short + uri[len(full):]
    return uri


def _print_summary(stage: str, result: Dict) -> None:
    """Print a concise validation summary."""
    status = 'PASS' if result['conforms'] else 'FAIL'
    print(f"SHACL [{stage}]: {status} — "
          f"{result['violations']} violation(s), {result['warnings']} warning(s)")
    if not result['conforms']:
        # Show violations first, then warnings
        violations = [r for r in result['results'] if r['severity'] == 'sh:Violation']
        warnings = [r for r in result['results'] if r['severity'] == 'sh:Warning']
        for r in violations[:10]:
            print(f"  [{r['severity']}] {r['message']}"
                  f"{' @ ' + r['focusNode'] if r['focusNode'] else ''}")
        if len(violations) > 10:
            print(f"  ... and {len(violations) - 10} more violation(s)")
        shown_warnings = min(10 - min(len(violations), 10), len(warnings))
        for r in warnings[:shown_warnings]:
            print(f"  [{r['severity']}] {r['message']}"
                  f"{' @ ' + r['focusNode'] if r['focusNode'] else ''}")
        remaining = len(warnings) - shown_warnings
        if remaining > 0:
            print(f"  ... and {remaining} more warning(s)")
