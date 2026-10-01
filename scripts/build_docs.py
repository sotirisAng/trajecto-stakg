#!/usr/bin/env python3
"""Build the WIDOCO documentation of every ontology module into docs/.

The folder layout mirrors the ontology IRIs, so that w3id.org/trajecto/<module>
can redirect to <pages>/<module>/:

    docs/index.html            http://w3id.org/trajecto
    docs/poi/index.html        http://w3id.org/trajecto/poi
    docs/purpose/skos/...      http://w3id.org/trajecto/purpose/skos

Metadata comes from each module's header; WIDOCO gets it through a generated
configuration file so that author order, license name and citation are right.

    java -version                                    # 17 or later
    curl -LO https://github.com/dgarijo/Widoco/releases/download/v1.4.25/widoco-1.4.25-jar-with-dependencies_JDK-17.jar
    python scripts/build_docs.py widoco-1.4.25-jar-with-dependencies_JDK-17.jar
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from rdflib import Graph, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF

ROOT = Path(__file__).resolve().parent.parent
ONTOLOGY = ROOT / "ontology"
DOCS = ROOT / "docs"
REPO = "https://github.com/sotirisAng/trajecto-stakg"
VANN = "http://purl.org/vocab/vann/"
BASE = "http://w3id.org/trajecto"
AUTHORS = ["Sotirios Angelis", "Konstantinos Kotis"]
INSTITUTION = "University of the Aegean"
CITATION = ("Angelis, S. (2026). Semantic Trajectory Analytics and Recommender Systems: Engineering Semantic "
            "Trajectories as Knowledge Graphs. Doctoral dissertation, University of the Aegean.")

REFERENCES = """<ul>
<li>S. Angelis. Semantic Trajectory Analytics and Recommender Systems: Engineering Semantic Trajectories as
Knowledge Graphs. Doctoral dissertation, University of the Aegean, 2026.</li>
<li>G. M. Santipantakis, G. A. Vouros, A. Glenis, C. Doulkeridis, A. Vlachou. The datAcron Ontology for Semantic
Trajectories. ESWC 2017 Satellite Events, LNCS 10577, pp. 26-30, Springer, 2017.
<a href="https://doi.org/10.1007/978-3-319-70407-4_6">doi:10.1007/978-3-319-70407-4_6</a></li>
<li>A. Gangemi, N. Guarino, C. Masolo, A. Oltramari, L. Schneider. Sweetening Ontologies with DOLCE. EKAW 2002,
LNCS 2473, pp. 166-181, Springer, 2002. <a href="https://doi.org/10.1007/3-540-45810-7_18">doi:10.1007/3-540-45810-7_18</a></li>
<li>D. Doumanas, G. Bouchouras, A. Soularidis, K. Kotis, G. Vouros. From Human- to LLM-Centered Collaborative
Ontology Engineering. Applied Ontology, 2024. <a href="https://doi.org/10.1177/15705838241305067">doi:10.1177/15705838241305067</a></li>
</ul>"""


def tidy(html):
    """Fill WIDOCO's reference placeholder and drop the empty changelog."""
    html = html.replace("Add your references here. It is recommended to have them as a list.", REFERENCES)
    return html.replace('<div id="changelog">null</div>', "")


# Parents before children: docs/purpose/ must exist before docs/purpose/skos/.
FILES = ["trajecto.ttl"] + [f"modules/{m}.ttl" for m in (
    "poi", "spaces", "purpose", "purpose-skos", "context", "patterns", "spatial",
    "temporal-dwell", "events", "scheduled-events", "profiles", "user")]


def esc(text):
    """Escape a value for a Java properties file."""
    return str(text).replace("\\", "\\\\").replace("\n", " ").replace(":", "\\:").replace("=", "\\=")


def config(g, onto, rel_path):
    def one(p):
        v = g.value(onto, p)
        return "" if v is None else str(v)

    imports = sorted(str(o) for o in g.objects(onto, OWL.imports))
    description = one(DCTERMS.description)
    abstract = description.split(". ")[0].rstrip(".") + "."
    intro = (f"Source: {REPO}/blob/main/ontology/{rel_path}. "
             f"The whole ontology is documented from {BASE} (the core module), which imports every module.")
    fields = {
        "ontologyTitle": one(DCTERMS.title),
        "ontologyName": one(URIRef("http://www.w3.org/2000/01/rdf-schema#label")),
        "ontologyPrefix": one(URIRef(VANN + "preferredNamespacePrefix")),
        "ontologyNamespaceURI": one(URIRef(VANN + "preferredNamespaceUri")),
        "thisVersionURI": one(OWL.versionIRI),
        "latestVersionURI": str(onto),
        "previousVersionURI": one(OWL.priorVersion),
        "ontologyRevisionNumber": one(OWL.versionInfo),
        "dateCreated": one(DCTERMS.created),
        "dateModified": one(DCTERMS.modified),
        "dateIssued": one(DCTERMS.issued),
        "licenseURI": "https://creativecommons.org/publicdomain/zero/1.0/",
        "licenseName": "CC0 1.0 Universal",
        "licenseIconURL": "https://licensebuttons.net/p/zero/1.0/88x31.png",
        "status": "Ontology Specification",
        "citeAs": CITATION,
        "abstract": abstract,
        "description": description,
        "introduction": intro,
        "publisher": one(DCTERMS.publisher),
        "authors": ";".join(AUTHORS),
        "authorsInstitution": ";".join([INSTITUTION] * len(AUTHORS)),
        "importedOntologyNames": ";".join(i[len(BASE) + 1:] if i.startswith(BASE + "/") else
                                          "datAcron" if "datacron" in i.lower() else i for i in imports),
        "importedOntologyURIs": ";".join(imports),
        "source": f"{REPO}/blob/main/ontology/{rel_path}",
        "RDFXMLSerialization": "ontology.owl",
        "TurtleSerialization": "ontology.ttl",
        "NTSerialization": "ontology.nt",
        "JSONLDSerialization": "ontology.jsonld",
    }
    return "\n".join(f"{k}={esc(v)}" for k, v in fields.items()) + "\n"


def main(jar):
    DOCS.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for rel in FILES:
            g = Graph().parse(ONTOLOGY / rel, format="turtle")
            onto = next(g.subjects(RDF.type, OWL.Ontology))
            out = DOCS / str(onto)[len(BASE):].lstrip("/")
            conf = Path(tmp) / (Path(rel).stem + ".properties")
            conf.write_text(config(g, onto, rel), encoding="utf-8")
            cmd = ["java", "-jar", jar, "-ontFile", str(ONTOLOGY / rel), "-outFolder", str(out),
                   "-confFile", str(conf), "-rewriteAll", "-uniteSections", "-webVowl"]
            if rel.endswith("poi.ttl"):
                cmd.append("-includeAnnotationProperties")  # the dwell thresholds are annotation properties
            print(f"{rel} -> {out.relative_to(ROOT)}")
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            html = tidy((out / "index-en.html").read_text(encoding="utf-8"))
            (out / "index-en.html").write_text(html, encoding="utf-8")
            (out / "index.html").write_text(html, encoding="utf-8")
    (DOCS / ".nojekyll").touch()  # GitHub Pages: serve the folders as they are


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
