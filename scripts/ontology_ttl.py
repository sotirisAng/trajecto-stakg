"""Write an ontology module as sectioned, human-readable Turtle.

rdflib's Turtle writer orders subjects by type and IRI, so the ontology
header lands mid-file and a module's own terms mix with stub declarations of
terms it borrows. This writer puts the header first, then one section per
kind of term, then the borrowed terms, then anything left (SWRL rules).
"""

from rdflib import BNode, Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS

PREFIXES = {
    "trajecto": "http://w3id.org/trajecto#",
    "poi": "http://w3id.org/trajecto/poi#",
    "spaces": "http://w3id.org/trajecto/spaces#",
    "purpose": "http://w3id.org/trajecto/purpose#",
    "purposeskos": "http://w3id.org/trajecto/purpose/skos#",
    "context": "http://w3id.org/trajecto/context#",
    "patterns": "http://w3id.org/trajecto/patterns#",
    "spatial": "http://w3id.org/trajecto/spatial#",
    "tdwell": "http://w3id.org/trajecto/temporal-dwell#",
    "events": "http://w3id.org/trajecto/events#",
    "scheduled": "http://w3id.org/trajecto/scheduled#",
    "profiles": "http://w3id.org/trajecto/profiles#",
    "user": "http://w3id.org/trajecto/user#",
    "datacron": "http://www.datacron-project.eu/datAcron#",
    "dul": "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#",
    "sf": "http://www.opengis.net/ont/sf#",
    "geo": "http://www.w3.org/2003/01/geo/wgs84_pos#",
    "time": "http://www.w3.org/2006/time#",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "swrl": "http://www.w3.org/2003/11/swrl#",
    "swrlb": "http://www.w3.org/2003/11/swrlb#",
    "swrla": "http://swrl.stanford.edu/ontologies/3.3/swrla.owl#",
    "dcterms": "http://purl.org/dc/terms/",
    "vann": "http://purl.org/vocab/vann/",
    "bibo": "http://purl.org/ontology/bibo/",
    "skos": str(SKOS),
    "owl": str(OWL),
    "rdf": str(RDF),
    "rdfs": str(RDFS),
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}

SECTIONS = [
    ("Classes", {OWL.Class, RDFS.Class}),
    ("Object properties", {OWL.ObjectProperty}),
    ("Data properties", {OWL.DatatypeProperty}),
    ("Annotation properties", {OWL.AnnotationProperty}),
    ("Concept schemes", {SKOS.ConceptScheme}),
    ("Concepts", {SKOS.Concept}),
    ("Individuals", {OWL.NamedIndividual}),
]


def _closure(g, node, out):
    """Copy node's triples and those of every blank node it reaches."""
    for p, o in g.predicate_objects(node):
        out.add((node, p, o))
        if isinstance(o, BNode) and (o, None, None) not in out:
            _closure(g, o, out)


def _turtle(graph, prefix_lines):
    """Serialize graph without its @prefix lines, collecting those into prefix_lines."""
    for k, v in PREFIXES.items():
        graph.bind(k, v, override=True, replace=True)
    body = []
    for line in graph.serialize(format="turtle").splitlines():
        if line.startswith("@prefix"):
            prefix_lines.add(line)
        else:
            body.append(line)
    return "\n".join(body).strip()


def _block(g, subject, prefix_lines):
    sub = Graph()
    _closure(g, subject, sub)
    return _turtle(sub, prefix_lines)


def _banner(title):
    return f"\n#################################################################\n#    {title}\n#################################################################\n"


def write_module(g, path, namespace):
    """Serialize g to path; terms under `namespace` are the module's own."""
    prefix_lines = set()
    lines = []
    done = set()
    for onto in sorted(g.subjects(RDF.type, OWL.Ontology)):
        lines.append(_block(g, onto, prefix_lines))
        done.add(onto)

    def key(s):
        return str(s).rsplit("#", 1)[-1].rsplit("/", 1)[-1].lower()

    own = [s for s in set(g.subjects()) if isinstance(s, URIRef) and str(s).startswith(namespace)]
    for title, types in SECTIONS:
        members = sorted((s for s in own if s not in done and types & set(g.objects(s, RDF.type))), key=key)
        if members:
            lines.append(_banner(title))
            for s in members:
                lines.append(_block(g, s, prefix_lines) + "\n")
                done.add(s)
    rest_own = sorted((s for s in own if s not in done), key=key)
    if rest_own:
        lines.append(_banner("Other terms"))
        for s in rest_own:
            lines.append(_block(g, s, prefix_lines) + "\n")
            done.add(s)
    borrowed = sorted((s for s in set(g.subjects()) if isinstance(s, URIRef) and s not in done), key=str)
    if borrowed:
        lines.append(_banner("Terms reused from other modules and vocabularies"))
        for s in borrowed:
            lines.append(_block(g, s, prefix_lines) + "\n")
            done.add(s)
    # Top-level blank nodes (SWRL rules) that no named term reaches.
    reached = Graph()
    for s in done:
        _closure(g, s, reached)
    rest = Graph()
    for t in g:
        if t not in reached:
            rest.add(t)
    if len(rest):
        lines.append(_banner("Rules"))
        lines.append(_turtle(rest, prefix_lines))
    order = {f"@prefix {k}: <{v}> .": i for i, (k, v) in enumerate(PREFIXES.items())}
    head = sorted(prefix_lines, key=lambda line: (order.get(line, len(order)), line))
    path.write_text("\n".join(head + [""] + lines).rstrip() + "\n", encoding="utf-8")

    check = Graph().parse(path, format="turtle")
    if len(check) != len(g):  # blank nodes make isomorphism costly; size is the cheap guard
        raise RuntimeError(f"{path}: wrote {len(check)} triples, expected {len(g)}")
