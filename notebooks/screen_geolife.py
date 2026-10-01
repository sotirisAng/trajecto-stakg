"""Screen GeoLife walking trajectories before running the ingestion pipeline.

The GeoLife instantiation was built from `walking_trajectories_1-20.csv` (the
first 20 user ids), which yielded 47 evaluable trajectories from 9 users. User
id correlates with nothing: screening the full walking extraction finds 205
candidates from 24 users, 75% of them in users never ingested. Resolving their
stops against the OSM layer (72% of stops have a named node within 100 m)
predicts 127 evaluable trajectories from 19 users over 458 distinct POIs --
a conservative floor, since it omits the origin/destination pseudo-stops that
find_pois_near_stops injects.

This decides which trajectories are worth the expensive path (Overpass at 2 s
per stop, RDF generation, enrichment) using only local arithmetic. It mirrors
the pipeline's stop detector -- CONFIG['stop_detection'], 20 m diameter and
60 s duration -- so its count is directly comparable to the eligibility
threshold (>= 3 POI interactions).

Validation: restricted to the 18 already-ingested users it predicts 52
evaluable trajectories where the pipeline produced 47 (+10%). It over-predicts
because a stop only becomes a POI interaction if OSM has a named node within
CONFIG['osm_query']['radius_meters'].

    python screen_geolife.py                      # screen, write both CSVs
    python screen_geolife.py --prewarm            # + bulk-fetch OSM and fill
                                                  #   files/osm_cache.json
    python screen_geolife.py --self-check         # no data needed

Outputs (output/evaluation/):
    geolife_screen.csv        one row per trajectory, all metrics
    geolife_candidates.csv    the evaluable subset, for the pipeline
    geolife_stops.csv         their stop centroids, for cache prewarming
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "geolife" / "Geolife Trajectories 1.3" / "walking_trajectories.csv"
RAW = ROOT / "data" / "geolife" / "Geolife Trajectories 1.3" / "Data"
OUT = Path(__file__).resolve().parent / "output" / "evaluation"

MAX_DIAM_M, MIN_DUR_S = 20.0, 60.0   # == CONFIG['stop_detection'] in notebook 1
SAME_PLACE_M = 50.0                  # two stops this close are one POI
MIN_POINTS, MIN_MINUTES = 20, 5.0
MIN_STOPS = 3                        # == evaluation's min_seq_length
POI_RADIUS = 100.0                   # == CONFIG['osm_query']['radius_meters']
BEIJING_BBOX = (39.75, 116.15, 40.15, 116.65)   # south, west, north, east


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def find_stops(lats, lons, secs):
    """Stop centroids: maximal runs staying within MAX_DIAM_M of their anchor
    for at least MIN_DUR_S. Same rule as MovingPandas' TrajectoryStopDetector,
    without the dependency."""
    out, i, n = [], 0, len(lats)
    while i < n:
        j = i + 1
        while j < n and haversine(lats[i], lons[i], lats[j], lons[j]) <= MAX_DIAM_M / 2:
            j += 1
        if j - i >= 2 and (secs[j - 1] - secs[i]) >= MIN_DUR_S:
            out.append((float(np.mean(lats[i:j])), float(np.mean(lons[i:j]))))
            i = j          # a closed stop is consumed
        else:
            i += 1         # not a stop, slide on
    return out


def collapse(points, sep_m=SAME_PLACE_M):
    """Drop stops within sep_m of one already kept -- revisiting the same
    shopfront is one POI, not several."""
    keep = []
    for p in points:
        if not any(haversine(p[0], p[1], q[0], q[1]) <= sep_m for q in keep):
            keep.append(p)
    return keep


def screen(src=SRC):
    df = pd.read_csv(src, parse_dates=["datetime"])
    df["secs"] = (df["datetime"] - pd.Timestamp("1970-01-01")).dt.total_seconds()
    print(f"{len(df):,} points / "
          f"{df.groupby(['user_id', 'trajectory_file']).ngroups:,} trajectories / "
          f"{df.user_id.nunique()} users")

    rows, stop_rows = [], []
    for (user, traj), g in df.groupby(["user_id", "trajectory_file"], sort=False):
        g = g.sort_values("secs")
        lats, lons = g.latitude.to_numpy(), g.longitude.to_numpy()
        stops = collapse(find_stops(lats, lons, g.secs.to_numpy()))
        rows.append({
            "user": user, "traj": traj, "points": len(g),
            "minutes": (g.secs.iloc[-1] - g.secs.iloc[0]) / 60.0,
            "extent_m": haversine(lats.min(), lons.min(), lats.max(), lons.max()),
            "distinct_stops": len(stops),
            "lat": float(lats.mean()), "lon": float(lons.mean()),
        })
        for lat, lon in stops:
            stop_rows.append({"user": user, "traj": traj, "lat": lat, "lon": lon})
    return pd.DataFrame(rows), pd.DataFrame(stop_rows)


def read_plt(path):
    """GeoLife .plt: six header lines, then lat,lon,0,alt,days,date,time."""
    out = []
    for line in path.read_text(errors="ignore").splitlines()[6:]:
        f = line.split(",")
        if len(f) < 7:
            continue
        try:
            out.append((float(f[0]), float(f[1]), float(f[3]),
                        float(f[4]) * 86400.0, f"{f[5]}T{f[6]}"))
        except ValueError:
            pass
    return out


_INDEX = None


def poi_index(bbox=BEIJING_BBOX):
    """Local named-node index; loaded once, no network after the first fetch."""
    global _INDEX
    if _INDEX is None:
        from helpers.osm_bulk_cache import BulkPOIs
        _INDEX = BulkPOIs(bbox)
    return _INDEX


def recommendable_pois(stop_points):
    """Distinct POIs at these stops that survive the recommender's exclusion
    list -- the quantity the evaluation's min_seq_length gate actually counts.

    Screening on raw stop counts overpredicts badly: on the walking corpus 185
    trajectories had >=3 distinct POIs but only 53 survived, because 36% of
    Beijing's named nodes are bus and train stations.
    """
    from helpers.poi_classifier import classify_by_tags
    from helpers.context_recommender import EXCLUDED_CATEGORIES
    idx = poi_index()
    keep = set()
    for lat, lon in stop_points:
        hit = idx.answer(lat, lon, POI_RADIUS)
        if not hit or not hit["elements"]:
            continue
        # notebook 1 keeps only the closest named node (max_closest_pois=1)
        near = min(hit["elements"],
                   key=lambda e: (e["lat"] - lat) ** 2 + (e["lon"] - lon) ** 2)
        cls = classify_by_tags(str(near.get("tags", {})))
        cat = cls[0].split(":")[-1] if cls else ""
        if cat and cat not in EXCLUDED_CATEGORIES:
            keep.add(near["id"])
    return keep


def screen_raw(bbox=BEIJING_BBOX, min_points=MIN_POINTS):
    """Screen every .plt of all 182 users, ignoring the transport-mode labels.

    The walking extraction needs labels.txt, which only 69 users have -- that
    structurally excluded 113 users and 7,764 files. Mode turned out not to
    predict yield (mixed-mode data screens at the same rate as walking-only),
    so this filters on the outcome instead: does the trajectory produce enough
    recommendable POIs to be evaluable.
    """
    users = sorted(d for d in RAW.iterdir() if d.is_dir())
    n_files = sum(len(list((u / "Trajectory").glob("*.plt"))) for u in users)
    print(f"{len(users)} users, {n_files:,} .plt files")
    rows, stop_rows = [], []
    for k, u in enumerate(users, 1):
        tdir = u / "Trajectory"
        if not tdir.exists():
            continue
        for f in sorted(tdir.glob("*.plt")):
            pts = read_plt(f)
            if len(pts) < min_points:
                continue
            lats = np.array([p[0] for p in pts])
            lons = np.array([p[1] for p in pts])
            secs = np.array([p[3] for p in pts])
            if not (bbox[0] <= lats.mean() <= bbox[2]
                    and bbox[1] <= lons.mean() <= bbox[3]):
                continue
            st = collapse(find_stops(lats, lons, secs))
            rec = recommendable_pois(st) if len(st) >= MIN_STOPS else set()
            rows.append({
                "user": u.name, "traj": f.name, "points": len(pts),
                "minutes": (secs[-1] - secs[0]) / 60.0,
                "extent_m": haversine(lats.min(), lons.min(),
                                      lats.max(), lons.max()),
                "distinct_stops": len(st), "rec_pois": len(rec),
                "lat": float(lats.mean()), "lon": float(lons.mean()),
            })
            for lat, lon in st:
                stop_rows.append({"user": u.name, "traj": f.name,
                                  "lat": lat, "lon": lon})
        if k % 20 == 0:
            print(f"  {k}/{len(users)} users, {len(rows):,} trajectories kept")
    return pd.DataFrame(rows), pd.DataFrame(stop_rows)


def in_bbox(df, bbox):
    s, w, n, e = bbox
    return df[df.lat.between(s, n) & df.lon.between(w, e)]


def report(scr, bbox=BEIJING_BBOX):
    base = (scr.points >= MIN_POINTS) & (scr.minutes >= MIN_MINUTES)
    print("\n=== screening cascade ===")
    for label, mask in [
        ("trajectories scanned", scr.points > 0),
        (f"  >= {MIN_POINTS} points, >= {MIN_MINUTES:.0f} min", base),
        ("  >= 1 distinct stop", base & (scr.distinct_stops >= 1)),
        (f"  >= {MIN_STOPS} distinct stops", base & (scr.distinct_stops >= MIN_STOPS)),
    ]:
        sub = scr[mask]
        print(f"{label:36} {len(sub):6,} ({len(sub)/len(scr):5.1%})  users {sub.user.nunique():3}")

    cand = in_bbox(scr[base & (scr.distinct_stops >= MIN_STOPS)], bbox)
    print(f"{'  ... within bbox':36} {len(cand):6,} "
          f"({len(cand)/len(scr):5.1%})  users {cand.user.nunique():3}")

    if "rec_pois" in scr.columns:
        cand = cand[cand.rec_pois >= MIN_STOPS]
        print(f"{'  ... >= 3 RECOMMENDABLE POIs  <- EVALUABLE':36} {len(cand):6,} "
              f"({len(cand)/len(scr):5.1%})  users {cand.user.nunique():3}")
        print(f"\ndistinct recommendable POIs in the evaluable set: "
              f"{int(cand.rec_pois.sum()):,} (with repeats across trajectories)")
    print(f"\nstops to resolve against OSM: {int(cand.distinct_stops.sum()):,}")
    print("Trajectories outside the bbox share no POIs with the corpus: they add "
          "to N but contribute\nceiling 0 and no co-visitation edges, so they are "
          "dropped rather than ingested.")
    return cand


def self_check():
    """Stop detection and collapsing, on hand-built input."""
    # 5 min parked, 5 min walking away, 5 min parked again -> 2 stops
    lats = [39.9500] * 6 + [39.9500 + 0.0009 * k for k in range(1, 6)] + [39.9550] * 6
    lons = [116.3500] * 17
    secs = list(range(0, 17 * 60, 60))
    stops = find_stops(np.array(lats), np.array(lons), np.array(secs))
    assert len(stops) == 2, stops
    # both stops are ~550 m apart, so collapsing keeps both
    assert len(collapse(stops)) == 2
    # the same two stops 10 m apart collapse to one
    assert len(collapse([(39.95, 116.35), (39.95001, 116.35)])) == 1
    # a fast pass-through is not a stop: 20 points, each 30 m on, 10 s apart
    moving = np.array([39.95 + 0.00027 * k for k in range(20)])
    assert find_stops(moving, np.full(20, 116.35), np.arange(20) * 10) == []
    print("screen_geolife self-check passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prewarm", action="store_true",
                    help="bulk-fetch the bbox from Overpass and fill files/osm_cache.json")
    ap.add_argument("--emit-csv", action="store_true",
                    help="write walking_trajectories_candidates.csv (same schema as the "
                         "source) next to it, for notebook 0's geolife loader")
    ap.add_argument("--all-users", action="store_true",
                    help="screen raw .plt files for all 182 users instead of the "
                         "walking CSV, counting only recommendable POIs")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    if args.self_check:
        self_check()
        return

    if not SRC.exists():
        sys.exit(f"missing {SRC}\nRun extract_walking_trajectories.py first.")

    OUT.mkdir(parents=True, exist_ok=True)
    if args.all_users:
        scr, stops = screen_raw()
        prefix = "geolife_all"
    else:
        scr, stops = screen()
        prefix = "geolife"
    cand = report(scr)

    scr.to_csv(OUT / f"{prefix}_screen.csv", index=False)
    cand.to_csv(OUT / f"{prefix}_candidates.csv", index=False)
    keep = set(zip(cand.user, cand.traj))
    cand_stops = stops[[(u, t) in keep for u, t in zip(stops.user, stops.traj)]]
    cand_stops.to_csv(OUT / f"{prefix}_stops.csv", index=False)
    print(f"\nwrote {prefix}_screen.csv / _candidates.csv / _stops.csv -> {OUT}")

    if args.emit_csv and args.all_users:
        rows = []
        for u, t in sorted(keep):
            for lat, lon, alt, _secs, ts in read_plt(RAW / u / "Trajectory" / t):
                rows.append({"user_id": u, "trajectory_file": t,
                             "latitude": lat, "longitude": lon,
                             "altitude": alt, "datetime": ts, "mode": "unknown"})
        sub = pd.DataFrame(rows)
        dest = SRC.with_name("all_users_candidates.csv")
        sub.to_csv(dest, index=False)
        print(f"\nwrote {dest.name}: {len(sub):,} points / "
              f"{len(keep):,} trajectories / {cand.user.nunique()} users")
    elif args.emit_csv:
        src = pd.read_csv(SRC)
        keep = set(zip(cand.user, cand.traj))
        sub = src[[(u, t) in keep for u, t in zip(src.user_id, src.trajectory_file)]]
        dest = SRC.with_name("walking_trajectories_candidates.csv")
        sub.to_csv(dest, index=False)
        print(f"\nwrote {dest.name}: {len(sub):,} points / "
              f"{sub.groupby(['user_id','trajectory_file']).ngroups:,} trajectories / "
              f"{sub.user_id.nunique()} users")

    if args.prewarm:
        from helpers.osm_bulk_cache import BEIJING as BBOX, BulkPOIs
        print(f"\nfetching named nodes for bbox {BBOX} (tiled, cached on disk)...")
        idx = BulkPOIs(BBOX)
        print(f"indexed {len(idx.elements):,} named nodes")
        stats = idx.prewarm(zip(cand_stops.lat, cand_stops.lon),
                            ROOT / "files" / "osm_cache.json", radius=100.0)
        print(f"osm_cache.json: {stats['before']:,} -> {stats['after']:,} entries "
              f"(+{stats['added']:,}, {stats['outside_bbox']:,} outside bbox)")
        print("Pipeline stop centroids differ slightly from the screen's, so "
              "coverage is partial;\nsee osm_bulk_cache.__doc__ option (b) for "
              "the one-line notebook change that makes it total.")


if __name__ == "__main__":
    main()
