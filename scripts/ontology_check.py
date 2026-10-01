#!/usr/bin/env python3
"""Ontology consistency and representation check (evaluation axis A).

Loads every Trajecto module, runs HermiT, and reports the findings an
ontology-engineering reviewer looks for first:

  consistency      does the reasoner classify without contradiction
  undeclared       properties/classes used in axioms but never declared
  punning          an IRI used as both class and individual
  phantom types    configuration individuals asserted as instances of the
                   class they configure -- these become Points of Interest
                   under the subclass closure
  versionIRI       modules sharing a versionIRI
  orphans          declared classes with no superclass and no instances

    python scripts/ontology_check.py            # report
    python scripts/ontology_check.py --self-check
"""

import argparse
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ONTOLOGY = ROOT / "ontology"
TRAJECTO = "http://w3id.org/trajecto"


def load_graph():
    from rdflib import Graph
    g = Graph()
    files = ([ONTOLOGY / "trajecto.ttl"] + sorted((ONTOLOGY / "modules").glob("*.ttl"))
             + sorted((ONTOLOGY / "imports").glob("*.owl")))
    loaded = []
    for f in files:
        fmt = ("turtle" if f.suffix == ".ttl" else "xml")
        try:
            g.parse(str(f), format=fmt)
            loaded.append(f.name)
        except Exception as exc:
            print(f"  ! could not parse {f.name}: {type(exc).__name__}")
    return g, loaded


def q(g, sparql):
    return list(g.query(sparql))


# The OWL 2 datatype map (owl2-syntax #Datatype_Maps). Anything outside it puts
# the ontology in OWL 2 Full and no DL reasoner will load it.
OWL2_DATATYPES = {
    "PlainLiteral", "XMLLiteral", "Literal", "real", "rational",
    "decimal", "integer", "nonNegativeInteger", "nonPositiveInteger",
    "positiveInteger", "negativeInteger", "long", "int", "short", "byte",
    "unsignedLong", "unsignedInt", "unsignedShort", "unsignedByte",
    "double", "float", "string", "normalizedString", "token", "language",
    "Name", "NCName", "NMTOKEN", "boolean", "hexBinary", "base64Binary",
    "anyURI", "dateTime", "dateTimeStamp",
}

PRE = """
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX poi:  <http://w3id.org/trajecto/poi#>
PREFIX trajecto: <http://w3id.org/trajecto#>
"""


def check(g):
    findings = defaultdict(list)

    # 1. punning: an IRI declared both as a class and as a named individual
    for r in q(g, PRE + """SELECT DISTINCT ?x WHERE {
        ?x a owl:Class . ?x a owl:NamedIndividual .
        FILTER(STRSTARTS(STR(?x), '%s')) }""" % TRAJECTO):
        findings["punning"].append(str(r[0]))

    # 2. phantom types: a configuration individual asserted as an instance of
    #    a class that is a POI subclass -> inferred to be a PointOfInterest
    for r in q(g, PRE + """SELECT DISTINCT ?cfg ?cls WHERE {
        ?cfg poi:hasTypicalDuration ?d ; a ?cls .
        ?cls rdfs:subClassOf* trajecto:PointOfInterest .
        FILTER(?cls != owl:NamedIndividual) }"""):
        findings["phantom_poi"].append(f"{str(r[0]).split('#')[-1]} a {str(r[1]).split('#')[-1]}")

    # 3. properties used in restrictions but never declared
    for r in q(g, PRE + """SELECT DISTINCT ?p WHERE {
        ?r a owl:Restriction ; owl:onProperty ?p .
        FILTER(STRSTARTS(STR(?p), '%s'))
        FILTER NOT EXISTS { ?p a owl:ObjectProperty }
        FILTER NOT EXISTS { ?p a owl:DatatypeProperty }
        FILTER NOT EXISTS { ?p a owl:AnnotationProperty } }""" % TRAJECTO):
        findings["undeclared_property"].append(str(r[0]))

    # 4. classes referenced as a superclass but never declared
    for r in q(g, PRE + """SELECT DISTINCT ?c WHERE {
        ?s rdfs:subClassOf ?c .
        FILTER(STRSTARTS(STR(?c), '%s')) FILTER(!isBlank(?c))
        FILTER NOT EXISTS { ?c a owl:Class } }""" % TRAJECTO):
        findings["undeclared_class"].append(str(r[0]))

    # 4b. an IRI declared under more than one property type -- illegal in OWL 2 DL
    for r in q(g, PRE + """SELECT ?p (COUNT(DISTINCT ?t) AS ?k) WHERE {
        ?p a ?t . FILTER(?t IN (owl:ObjectProperty, owl:DatatypeProperty, owl:AnnotationProperty))
        FILTER(STRSTARTS(STR(?p), '%s')) } GROUP BY ?p HAVING(COUNT(DISTINCT ?t) > 1)""" % TRAJECTO):
        findings["property_punning"].append(f"{str(r[0]).replace(TRAJECTO,'')} declared {r[1]} ways")

    # 4c. datatypes outside the OWL 2 datatype map
    # attribute each to the vocabulary that declares it: a datatype inherited
    # from a reused ontology is not a defect in this contribution
    for r in q(g, PRE + """SELECT DISTINCT ?p ?dt WHERE {
        ?p rdfs:range ?dt .
        FILTER(STRSTARTS(STR(?dt), 'http://www.w3.org/2001/XMLSchema#')) }"""):
        name = str(r[1]).split("#")[-1]
        if name in OWL2_DATATYPES:
            continue
        own = str(r[0]).startswith(TRAJECTO)
        findings["non_owl2_datatype" if own else "inherited_datatype"].append(
            f"xsd:{name} as range of {str(r[0]).split('#')[-1]}"
            + ("" if own else f"  [{str(r[0]).split('/')[2]}]"))
    seen_lit = set()
    for _s, _p, _o in g:
        dt = getattr(_o, "datatype", None)
        if not dt or not str(dt).startswith("http://www.w3.org/2001/XMLSchema#"):
            continue
        name = str(dt).split("#")[-1]
        if name in OWL2_DATATYPES:
            continue
        own = str(_p).startswith(TRAJECTO)
        key = (name, own)
        if key in seen_lit:
            continue
        seen_lit.add(key)
        findings["non_owl2_datatype" if own else "inherited_datatype"].append(
            f"xsd:{name} as value of {str(_p).split('#')[-1]}")

    # 5. shared versionIRI across ontologies
    seen = defaultdict(list)
    for r in q(g, PRE + """SELECT ?o ?v WHERE { ?o a owl:Ontology ; owl:versionIRI ?v }"""):
        seen[str(r[1])].append(str(r[0]))
    for v, os_ in seen.items():
        if len(os_) > 1:
            findings["shared_versionIRI"].append(f"{v} <- {', '.join(os_)}")

    # 6. object properties with neither domain nor range.
    #    A property that declares owl:inverseOf or rdfs:subPropertyOf inherits
    #    both by swapping/specialisation, so it is complete without them; so
    #    is a deprecated alias declared equivalent to its replacement.
    for r in q(g, PRE + """SELECT DISTINCT ?p WHERE {
        ?p a owl:ObjectProperty .
        FILTER(STRSTARTS(STR(?p), '%s'))
        FILTER NOT EXISTS { ?p rdfs:domain ?d }
        FILTER NOT EXISTS { ?p rdfs:range ?r }
        FILTER NOT EXISTS { ?p owl:inverseOf ?inv }
        FILTER NOT EXISTS { ?p rdfs:subPropertyOf ?sup }
        FILTER NOT EXISTS { ?p owl:equivalentProperty ?eq } }""" % TRAJECTO):
        findings["no_domain_or_range"].append(str(r[0]))

    return findings


def reason(g):
    """HermiT consistency over the merged modules.

    The modules owl:import IRIs under http://w3id.org/trajecto/... which do not
    resolve over the network, so owlready2 cannot follow them. Merging every
    module into one import-free file first sidesteps that and reasons over
    exactly the axioms the project ships.
    """
    try:
        import owlready2
        from rdflib import RDF, OWL, URIRef
    except ImportError:
        return "owlready2/rdflib not installed"

    SWRL = "http://www.w3.org/2003/11/swrl#"
    merged = g
    for s_, p_, o_ in list(merged.triples((None, OWL.imports, None))):
        merged.remove((s_, p_, o_))          # everything is already in here

    # HermiT refuses SWRL rules that use built-in atoms, so the rule layer is
    # stripped and reported separately. This is an expressivity boundary worth
    # stating, not a defect: DL consistency and rule-layer behaviour are
    # different claims and should not be conflated.
    # Annotations are not part of the OWL 2 DL fragment, so they are excluded
    # from the reasoner input by definition. This also removes the xsd:duration
    # literals on the dwell parameters -- that datatype is outside the OWL 2
    # datatype map, which is precisely why those parameters are annotations.
    ann_props = {pr for pr in merged.subjects(RDF.type, OWL.AnnotationProperty)}
    n_ann = 0
    for pr in ann_props:
        for t in list(merged.triples((None, pr, None))):
            merged.remove(t); n_ann += 1

    # An annotation property carries no DL range, and xsd:duration is outside
    # the OWL 2 datatype map, so the leftover range declarations go too.
    XS = "http://www.w3.org/2001/XMLSchema#"
    n_dt = 0
    for t in list(merged):
        _s, _p, _o = t
        bad = (str(_o).startswith(XS) and str(_o).split("#")[-1] not in OWL2_DATATYPES)
        dt = getattr(_o, "datatype", None)
        bad = bad or (dt and str(dt).startswith(XS)
                      and str(dt).split("#")[-1] not in OWL2_DATATYPES)
        if bad:
            merged.remove(t); n_dt += 1

    rules = {s2 for s2, p2, o2 in merged if str(o2).startswith(SWRL + "Imp")}
    n_rules = len(rules)
    for s2, p2, o2 in list(merged):
        if str(s2).startswith(SWRL) or str(p2).startswith(SWRL) or str(o2).startswith(SWRL):
            merged.remove((s2, p2, o2))
    tmp = Path("/tmp/trajecto_merged.owl")
    merged.serialize(destination=str(tmp), format="xml")

    world = owlready2.World()
    # deliberately NOT adding MODULES to onto_path: the merged file is
    # import-free, and onto_path would make owlready2 re-load the originals
    # (re-introducing the annotations and datatypes just stripped)
    onto = world.get_ontology(f"file://{tmp}").load()
    try:
        with onto:
            owlready2.sync_reasoner_hermit(world, infer_property_values=False, debug=0)
        return (f"DL fragment CONSISTENT — {len(list(onto.classes())):,} classes "
                f"classified; excluded {n_rules} SWRL rule(s) with built-in atoms "
                f", {n_ann} annotation assertion(s) and {n_dt} non-OWL2-datatype "
                f"triple(s)")
    except owlready2.OwlReadyInconsistentOntologyError:
        return f"DL fragment INCONSISTENT — HermiT found a contradiction "\
               f"({n_rules} SWRL rule(s) excluded)"
    except Exception as exc:
        return f"reasoner error: {type(exc).__name__}: {str(exc)[:300]}"


def self_check():
    from rdflib import Graph
    g = Graph()
    g.parse(data="""
    @prefix owl:<http://www.w3.org/2002/07/owl#> .
    @prefix rdfs:<http://www.w3.org/2000/01/rdf-schema#> .
    @prefix poi:<http://w3id.org/trajecto/poi#> .
    @prefix trajecto:<http://w3id.org/trajecto#> .
    poi:Church a owl:Class ; rdfs:subClassOf trajecto:PointOfInterest .
    trajecto:PointOfInterest a owl:Class .
    poi:ChurchCfg a owl:NamedIndividual, poi:Church ; poi:hasTypicalDuration "PT60M" .
    poi:Dual a owl:Class, owl:NamedIndividual .
    """, format="turtle")
    f = check(g)
    assert any("ChurchCfg a Church" in x for x in f["phantom_poi"]), f["phantom_poi"]
    assert any(x.endswith("#Dual") for x in f["punning"]), f["punning"]
    print("ontology_check self-check passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--reason", action="store_true", help="also run HermiT (slow)")
    a = ap.parse_args()
    if a.self_check:
        self_check(); return

    print("loading modules...")
    g, loaded = load_graph()
    print(f"  {len(loaded)} files, {len(g):,} triples\n")

    f = check(g)
    labels = {
        "punning": "IRIs used as both class and individual",
        "phantom_poi": "config individuals inferred to be POIs",
        "undeclared_property": "properties in restrictions but never declared",
        "undeclared_class": "classes used as superclass but never declared",
        "property_punning": "properties declared under more than one property type",
        "non_owl2_datatype": "non-OWL2 datatypes in Trajecto-authored terms",
        "inherited_datatype": "non-OWL2 datatypes inherited from reused vocabularies",
        "shared_versionIRI": "modules sharing a versionIRI",
        "no_domain_or_range": "object properties with neither domain nor range",
    }
    total = 0
    for k, lbl in labels.items():
        items = sorted(set(f[k]))
        total += len(items)
        print(f"{'OK  ' if not items else 'FAIL'}  {len(items):4}  {lbl}")
        for x in items[:6]:
            print(f"          {x.replace(TRAJECTO, '')}")
        if len(items) > 6:
            print(f"          ... and {len(items)-6} more")
    print(f"\n{total} findings")

    if a.reason:
        print("\nrunning HermiT...")
        print(" ", reason(g))


if __name__ == "__main__":
    main()
