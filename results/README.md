# Results

The evaluation outputs reported in chapter 7 and appendix D of the thesis, copied from `notebooks/output/` after the
final runs. Scripts write fresh copies to `notebooks/output/evaluation/`. File names drop the dataset suffix; the
folder gives the corpus.

| File (per corpus folder) | Produced by | Thesis |
|---|---|---|
| `protocol_fullpool.csv` | `run_osm_rome.py` (OSM Rome), `run_geolife.py` (GeoLife) | Baseline comparison table |
| `protocol_radius500.csv`, `protocol_radius1000.csv` | same | Within-radius protocol |
| `protocol_radius500_fixed.csv`, `protocol_radius1000_fixed.csv` | `rerun_radius_protocol.py` | Radius protocol on a shared candidate set |
| `protocol_transfer.csv`, `protocol_transfer_repeats.csv` | `run_osm_rome.py`, `run_geolife.py` | Session transfer, novel POIs and repeats |
| `significance.csv`, `clustered_significance.csv` | same, `run_clustered_significance.py` | Paired tests, clustered by user |
| `per_trajectory_results.json` | same | Per-trajectory ranks behind every table |
| `coldstart_raw.csv`, `coldstart_slices.csv` | `run_coldstart.py` | Cold-start slices |
| `preference_ablation.csv` | `run_preference_ablation.py` | Effect of the user-preference signal |
| `decay_ablation.csv` | `run_decay_ablation.py` | Snapshot vs. all-trajectory preferences |
| `cq_coverage.csv` | `cq_coverage.py` | Competency-question answer counts |
| `tag_reliability.csv` | `run_tag_reliability.py` | Tag stability and predictivity |
| `poi_centrality.csv`, `temporal_centrality.csv`, `class_*.csv` | notebook 21 | KG-applied analytics |
| `semantic_features.json` | notebook 10 | 67-dimensional trajectory features used by the recommender |
| `logs/` | the runs above, `scripts/shacl_conformance.py`, stage 13 | Console output, including SHACL conformance |

`cross-corpus/` holds the outputs of `run_analytics_insights.py`: profile classes per corpus, cross-corpus analogues,
engagement by purpose and navigation style by temporal context.

The tag reliability, SHACL, centrality and feature files come from the KG before the profile layer (stages 8 and 9)
was rebuilt; that rebuild touched only user profiles, so they are unchanged in the final KG.
