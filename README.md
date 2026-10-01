# Trajecto STaKG

Trajecto is a semantic trajectory knowledge graph (STaKG). It turns raw GPS traces into an RDF knowledge graph of
trajectories, the points of interest (POIs) they interact with, their movement patterns, purposes and temporal context,
and the user profiles learned from them. The graph serves three uses: a queryable repository of semantic trajectories,
KG-applied analytics, and a POI recommender.

This repository holds the Trajecto ontology, the nine-stage pipeline that builds the graph, the analytics and
recommender code, the evaluation scripts, and the results reported in chapter 7 of:

> Angelis, S. (2026). *Semantic Trajectory Analytics and Recommender Systems: Engineering Semantic Trajectories as
> Knowledge Graphs*. Doctoral dissertation, University of the Aegean.

The pipeline was run on three datasets: OSM Rome (public OpenStreetMap GPS traces), GeoLife (Microsoft Research) and
BIRD (museum visitors).

## Contents

| Path | What it holds |
|---|---|
| `ontology/` | The Trajecto ontology: core module `trajecto.ttl`, twelve modules in `modules/`, imported ontologies, Protégé catalog |
| `docs/` | WIDOCO documentation of every module, published at the w3id IRIs |
| `notebooks/0-9` | Pipeline stages 1 to 9 (notebook 0 loads the raw data) |
| `notebooks/10-18, 21` | Semantic features, KG explorer, semantic rules, OWL classification, graph analytics, profile cards, recommender, baselines |
| `notebooks/helpers/` | The pipeline, analytics and recommender code |
| `notebooks/run_*.py`, `make_*.py` | Evaluation runs and figures of chapter 7 |
| `sparql/` | Enrichment queries (stages 3 to 9) and semantic rules SR-05 to SR-54 |
| `shacl/` | SHACL shapes, cumulative per pipeline stage |
| `mappings/` | YARRRML mapping templates (CSV to RDF) |
| `bird/` | The BIRD instantiation: its loader, stages and analytics notebooks |
| `scripts/` | OWL classification, ontology and SHACL checks, profiles module and documentation generators |
| `data/` | The OSM Rome traces; download instructions for GeoLife and BIRD |
| `files/` | Caches that make the pipeline reproducible offline: OSM POIs, LLM POI classifications, scheduled events |
| `results/` | The evaluation outputs behind the thesis tables |
| `w3id/` | The `.htaccess` for the w3id.org/trajecto redirect |

## Ontology

The ontology lives at `http://w3id.org/trajecto` (CC0). The core module imports the other eleven modules and the purpose
concept scheme, so loading `ontology/trajecto.ttl` in Protégé (with `ontology/catalog-v001.xml` next to it) loads the
whole ontology offline.

| Module | IRI | Scope |
|---|---|---|
| core | `http://w3id.org/trajecto` | Person, entities of interest, interactions; reuses datAcron and DUL |
| poi | `http://w3id.org/trajecto/poi` | POI taxonomy, interaction hierarchy, dwell thresholds per POI type |
| spaces | `http://w3id.org/trajecto/spaces` | Cultural and commercial spaces, artifacts, visit contexts |
| purpose | `http://w3id.org/trajecto/purpose` | Movement purposes, characteristics and transitions |
| purpose concept scheme | `http://w3id.org/trajecto/purpose/skos` | SKOS purposes that trajectories are assigned |
| context | `http://w3id.org/trajecto/context` | Temporal, social, routine and environmental context |
| patterns | `http://w3id.org/trajecto/patterns` | Movement and behavioral patterns, navigation styles |
| spatial | `http://w3id.org/trajecto/spatial` | Proximity analysis, zones, distance metrics, clusters |
| temporal-dwell | `http://w3id.org/trajecto/temporal-dwell` | Dwell events, activity periods, dwell analytics |
| events | `http://w3id.org/trajecto/events` | Events detected in a trajectory |
| scheduled-events | `http://w3id.org/trajecto/scheduled` | Externally scheduled events hosted at POIs |
| profiles | `http://w3id.org/trajecto/profiles` | Trajectory profile classes assigned by an OWL reasoner |
| user | `http://w3id.org/trajecto/user` | User profiles, EWMA preference snapshots |

`ontology/README.md` lists what changed in version 1.1.0. To check the ontology and rebuild its documentation:

```bash
python scripts/ontology_check.py --reason      # structure checks + HermiT consistency
python scripts/generate_profiles_module.py     # regenerate modules/profiles.ttl
python scripts/build_docs.py widoco-1.4.25-jar-with-dependencies_JDK-17.jar
```

## Requirements

- Python 3.9 and `pip install -r requirements.txt`
- Apache Jena Fuseki on `http://localhost:3030` (the thesis used 5.4.0)
- Java 17 or later, for RMLMapper and WIDOCO
- [RMLMapper 7.3.3](https://github.com/RMLio/rmlmapper-java/releases): save `rmlmapper-7.3.3-r374-all.jar` in `mappings/`
- [yarrrml-parser](https://github.com/RMLio/yarrrml-parser): `npm install -g @rmlio/yarrrml-parser`
- An OpenAI API key in `.env` (see `.env.example`) for POIs missing from `files/poi_classification_cache.json`

The Fuseki dataset is chosen with the `TRAJECTO_KG` environment variable (default `osm-rome`). Stage 2 creates the
dataset and its `<name>_inference` twin and loads the ontology into both.

## Building a knowledge graph

OSM Rome, from the traces in this repository:

```bash
tar -xzf data/osm-rome/rome_center.tar.gz -C data/osm-rome
gunzip -k files/osm_cache.json.gz
cd notebooks
TRAJECTO_KG=osm-rome python run_pipeline.py --only 0 1 2 3 4 5 6 7 8 9 10 12 13
TRAJECTO_KG=osm-rome python -m papermill 21-kg-analytics.ipynb output/21-out.ipynb
```

Scheduled events (used by the recommender to gate conference, convention and exhibition venues) are loaded from
`files/events/` after stage 2:

```bash
cd notebooks
TRAJECTO_KG=osm-rome python -c "from helpers.event_ingester import ingest_fixture; \
from helpers.fuseki_config import query_endpoint, update_endpoint; \
print(ingest_fixture('../files/events/rome-events.json', query_endpoint, update_endpoint))"
```

GeoLife and BIRD: see `data/geolife/README.md` and `data/bird/README.md`.

## Reproducing the evaluation

Run from `notebooks/` against a built KG. Each script writes to `notebooks/output/evaluation/`.

| Thesis result | Command |
|---|---|
| Baseline comparison, radius and transfer protocols (OSM Rome) | `TRAJECTO_KG=osm-rome python run_osm_rome.py` |
| Same, GeoLife | `TRAJECTO_KG=geolife TRAJECTO_FEATURES=output/semantic_features-geolife/trajectory_features.json python run_geolife.py` |
| Radius protocol on the shared candidate set | `python rerun_radius_protocol.py --kg <kg>` |
| User-clustered significance | `python run_clustered_significance.py --kg <kg>` |
| Cold start | `TRAJECTO_KG=<kg> python run_coldstart.py` |
| Preference signal ablation | `python run_preference_ablation.py --kg <kg>` |
| Profile decay ablation | `TRAJECTO_KG=<kg> python run_decay_ablation.py` |
| Competency questions and portability | `python cq_coverage.py --kg <kg>` |
| Tag stability and predictivity | `python run_tag_reliability.py --kg <kg>` |
| SHACL conformance per stage | `python ../scripts/shacl_conformance.py --kg <kg>` |
| Cross-corpus analytics | `python run_analytics_insights.py` |
| Figures (written to `notebooks/output/figures/`) | `make_cross_dataset_figure.py`, `make_map_figure.py` |

`results/` holds the outputs these produced for the thesis; `results/README.md` maps each file to its table.

## Known issues

Left unchanged, so that the thesis results reproduce:

- Stage 13 (OWL classification) sees only purpose, navigation pattern and engagement. Notebook 12 does not run rules
  SR-41 (`trajecto:includesInteraction`) and SR-50 to SR-54 (`trajecto:hasEvent`), and
  `scripts/run_owl_classification.py` reads `trajecto:hasTemporalContext` while the KG stores
  `context:hasTemporalContext`. 89 of the 117 profile classes (visitor, focused, interaction, event, temporal and
  most archetype classes) are therefore never assigned; the KGs carry 13 distinct classes.
- `poi:Bench`, `poi:WaterFountain` and `poi:VendingMachine` are not linked to their own dwell configurations, so their
  interactions use the default thresholds.
- Notebook 14 queries `patterns:ImmersiveExploration`, which no stage writes (the class is `patterns:ImmersiveMovement`).

Fixed in this release:

- `passing-interaction.sparql`, `immersive-interaction.sparql` and notebook 7 found a POI class's dwell configuration
  through an `rdf:type` link that the ontology dropped in September 2026 (it made the configurations POIs). With the
  current ontology they found none: zero passing and zero immersive interactions, and no thresholds for the
  trajectory endpoints in stage 7. They now match the configuration by name, which gives the same pairs.
- The default dwell thresholds are restored (`ontology/README.md`).

## License

Code: MIT (`LICENSE`). Ontology: CC0 1.0. OSM Rome traces: © OpenStreetMap contributors, public GPS traces under the
[OpenStreetMap terms](https://www.openstreetmap.org/copyright); the KG keeps the OSM user names of identifiable traces.
POI data in `files/osm_cache.json.gz`: © OpenStreetMap contributors, ODbL.
