# GeoLife

GeoLife GPS Trajectories 1.3, Microsoft Research Asia:
<https://www.microsoft.com/en-us/download/details.aspx?id=52367>. It is not redistributed here.

> Y. Zheng, X. Xie, W.-Y. Ma. GeoLife: A Collaborative Social Networking Service among User, Location and Trajectory.
> IEEE Data Engineering Bulletin 33(2), 2010, pp. 32-39.

## Building the GeoLife KG of the thesis (1,160 trajectories, 96 users)

```bash
# 1. unzip the download here, so that data/geolife/Geolife Trajectories 1.3/Data/ exists
python data/geolife/extract_walking_trajectories.py        # -> walking_trajectories.csv

# 2. keep the trajectories that can yield >= 3 POI interactions, from the raw traces of all users
cd notebooks
python screen_geolife.py --all-users --emit-csv            # -> all_users_candidates.csv

# 3. run the pipeline on them
export TRAJECTO_KG=geolife
export TRAJECTO_SOURCES=geolife TRAJECTO_SOURCE=geolife
export TRAJECTO_GEOLIFE_CSV="../data/geolife/Geolife Trajectories 1.3/all_users_candidates.csv"
export TRAJECTO_CSV_DIR=../files/csv-geolife
export TRAJECTO_FEATURES_DIR=output/semantic_features-geolife
python run_pipeline.py --only 0 1 2 3 4 5 6 7 8 9 10 12 13
python -m papermill 21-kg-analytics.ipynb output/21-geolife.ipynb
```

Notebook 3's Wikidata linking takes over an hour on GeoLife. `files/osm_cache.json.gz` (gunzip it first) holds the
Overpass answers for the GeoLife stops, so stage 1 does not query Overpass again.
