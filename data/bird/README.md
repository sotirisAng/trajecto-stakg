# BIRD

BIRD museum visitor dataset (51 visitors of the Nancy Museum of Fine Arts). It is not redistributed here.

> A. Worm, F. Marchal, S. Castagnos. BIRD: A Museum Open Dataset Combining Behavior Patterns and Identity Types to
> Better Model Visitors' Experience. Adjunct Proceedings of the 33rd ACM Conference on User Modeling, Adaptation and
> Personalization (UMAP 2025), pp. 18-22. <https://doi.org/10.1145/3708319.3733686>

The dataset is released under CC BY-NC-SA 4.0 by its authors. Place its repository here as
`data/bird/bird-dataset-main/` (so that `data/bird/bird-dataset-main/data/` exists), then run the BIRD pipeline:

```bash
cd bird/notebooks
python run_pipeline.py --all  # stages and analytics; Fuseki dataset "bird" (bird/notebooks/helpers/fuseki_config.py)
```

BIRD replaces stages 1 and 2 (GPX parsing, Overpass lookup, LLM classification) with a loader for the artwork
observations (B0, B1) and runs the other stages through its own notebooks (B2 to B6, BF1 to BF6).
