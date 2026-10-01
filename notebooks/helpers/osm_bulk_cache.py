"""Bulk OSM POI fetch + local resolver, as a drop-in for per-stop Overpass calls.

Notebook 1 queries Overpass once per stop group at
`node(around:100,lat,lon); node.sn[name]; out geom;`, serialized with a 2 s
delay (CONFIG['osm_query']), so a corpus with ~2,000 stops costs well over an
hour of wall clock and depends on a rate-limited public endpoint staying up.

Every one of those queries is a bounded lookup inside one metropolitan area, so
one bbox fetch answers all of them:

    idx = BulkPOIs(BEIJING)      # tiled bbox fetch, cached on disk
    idx.answer(lat, lon, 100)    # same JSON shape notebook 1 expects

Two ways to use it against the existing pipeline:

  (a) prewarm() writes exact-key entries into files/osm_cache.json. Notebook 1
      picks them up unchanged; anything it asks for that was not prewarmed
      still falls through to the network. Nothing to modify, partial coverage
      (the pipeline's MovingPandas stop centroids differ slightly from the
      screen's).

  (b) for full coverage, one line in notebook 1's query_osm_single, right after
      the cache-miss check:

          hit = BULK.answer(lat, lon, radius) if BULK else None
          if hit is not None:
              return (cache_key, hit, True)

      Not applied here.

Overpass usage: one GET per bbox, results cached under
notebooks/output/osm_bulk/. Re-running is free.
"""

import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests

OVERPASS_URL = "http://overpass-api.de/api/interpreter"
# Overpass rejects the default python-requests User-Agent with 406 Not
# Acceptable. Notebook 1 sends no UA and is subject to the same block.
HEADERS = {"User-Agent": "Trajecto-thesis/1.0 (academic research; semantic trajectory KG)"}
CACHE_DIR = Path(__file__).resolve().parent.parent / "output" / "osm_bulk"
CELL_DEG = 0.005                      # ~450 m; a 100 m query needs the 3x3 ring
TILE_DEG = 0.1                        # ~11 km; one `node[name]` query's practical limit
MAX_SPLIT_DEPTH = 3                   # quarter a tile that still times out

# (south, west, north, east)
BEIJING = (39.75, 116.15, 40.15, 116.65)
ROME    = (41.85, 12.42, 41.95, 12.55)


def _bbox_slug(bbox: Tuple[float, float, float, float]) -> str:
    return "_".join(f"{v:.4f}" for v in bbox)


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def fetch_tile(bbox: Tuple[float, float, float, float],
               refresh: bool = False, timeout: int = 180) -> Path:
    """One Overpass call for one bbox. Cached on disk; returns the path.

    Mirrors notebook 1's filter (`node[name]`) so the resolver returns the same
    candidate set the per-stop query would have.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"nodes_{_bbox_slug(bbox)}.json"
    if path.exists() and not refresh:
        return path

    s, w, n, e = bbox
    query = f"[out:json][timeout:{timeout}];node[name]({s},{w},{n},{e});out geom;"
    for attempt, wait in enumerate([0, 10, 30, 60]):
        if wait:
            time.sleep(wait)
        resp = requests.get(OVERPASS_URL, params={"data": query},
                            timeout=timeout, headers=HEADERS)
        if resp.status_code == 200:
            path.write_text(json.dumps(resp.json()))
            return path
        if resp.status_code not in (429, 504) or attempt == 3:
            resp.raise_for_status()
    raise TimeoutError(f"Overpass timed out on tile {bbox}")


def fetch_area(bbox: Tuple[float, float, float, float],
               tile_deg: float = TILE_DEG, refresh: bool = False,
               depth: int = 0) -> List[Path]:
    """Fetch `bbox` as a grid of tiles, splitting any tile Overpass times out on.

    A whole metropolitan bbox exceeds what one `node[name]` query can return --
    Beijing at 0.4 x 0.5 degrees answers 504 -- so the area is always tiled.
    A tile that still times out is quartered, up to `MAX_SPLIT_DEPTH`.
    """
    s, w, n, e = bbox
    paths: List[Path] = []
    lat = s
    while lat < n - 1e-9:
        lon = w
        lat_hi = min(lat + tile_deg, n)
        while lon < e - 1e-9:
            lon_hi = min(lon + tile_deg, e)
            tile = (round(lat, 4), round(lon, 4), round(lat_hi, 4), round(lon_hi, 4))
            try:
                paths.append(fetch_tile(tile, refresh=refresh))
            except (TimeoutError, requests.HTTPError):
                if depth >= MAX_SPLIT_DEPTH:
                    raise
                paths.extend(fetch_area(tile, tile_deg / 2.0, refresh, depth + 1))
            lon = lon_hi
        lat = lat_hi
    return paths


class BulkPOIs:
    """Grid-indexed named nodes for one bbox, answering around-queries locally."""

    def __init__(self, bbox: Tuple[float, float, float, float], refresh: bool = False):
        self.bbox = bbox
        by_id: Dict[int, Dict] = {}
        for path in fetch_area(bbox, refresh=refresh):
            for el in json.loads(path.read_text()).get("elements", []):
                if el.get("lat") is not None and el.get("lon") is not None:
                    by_id[el["id"]] = el          # tile edges overlap; dedupe
        self.elements: List[Dict] = list(by_id.values())
        self.grid: Dict[Tuple[int, int], List[Dict]] = {}
        for el in self.elements:
            self.grid.setdefault(self._cell(el["lat"], el["lon"]), []).append(el)

    @staticmethod
    def _cell(lat: float, lon: float) -> Tuple[int, int]:
        return (int(math.floor(lat / CELL_DEG)), int(math.floor(lon / CELL_DEG)))

    def contains(self, lat: float, lon: float) -> bool:
        s, w, n, e = self.bbox
        return s <= lat <= n and w <= lon <= e

    def answer(self, lat: float, lon: float, radius: float = 100.0) -> Optional[Dict]:
        """Overpass-shaped response for `node(around:radius,lat,lon)[name]`.

        None when the point is outside the indexed bbox — the caller must fall
        back to the network rather than treat it as "no POIs here".
        """
        if not self.contains(lat, lon):
            return None
        # radius must stay inside the 3x3 ring the scan covers
        assert radius <= CELL_DEG * 111_000, "radius exceeds one grid cell"
        ci, cj = self._cell(lat, lon)
        out = []
        for i in range(ci - 1, ci + 2):
            for j in range(cj - 1, cj + 2):
                for el in self.grid.get((i, j), ()):
                    if _haversine(lat, lon, el["lat"], el["lon"]) <= radius:
                        out.append(el)
        return {"elements": out}

    def prewarm(self, points: Iterable[Tuple[float, float]],
                cache_file: Path, radius: float = 100.0) -> Dict[str, int]:
        """Write exact-key entries into notebook 1's osm_cache.json.

        Key format is notebook 1's: f"{lat:.6f},{lon:.6f},{radius}". Existing
        entries are never overwritten, so a real Overpass answer always wins
        over a synthesised one.
        """
        cache_file = Path(cache_file)
        cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
        before, added, outside = len(cache), 0, 0
        for lat, lon in points:
            hit = self.answer(lat, lon, radius)
            if hit is None:
                outside += 1
                continue
            key = f"{lat:.6f},{lon:.6f},{radius}"
            if key not in cache:
                cache[key] = hit
                added += 1
        cache_file.write_text(json.dumps(cache, indent=2))
        return {"before": before, "added": added,
                "outside_bbox": outside, "after": len(cache)}


def demo() -> None:
    """Self-check on a synthetic index — no network."""
    idx = BulkPOIs.__new__(BulkPOIs)
    idx.bbox = (39.9, 116.3, 40.0, 116.4)
    idx.elements = [
        {"id": 1, "lat": 39.9500, "lon": 116.3500, "tags": {"name": "near"}},
        {"id": 2, "lat": 39.9505, "lon": 116.3500, "tags": {"name": "56m away"}},
        {"id": 3, "lat": 39.9600, "lon": 116.3500, "tags": {"name": "1.1km away"}},
    ]
    idx.grid = {}
    for el in idx.elements:
        idx.grid.setdefault(BulkPOIs._cell(el["lat"], el["lon"]), []).append(el)

    got = {e["id"] for e in idx.answer(39.9500, 116.3500, 100.0)["elements"]}
    assert got == {1, 2}, got                       # 3 is out of range
    assert idx.answer(39.9500, 116.3500, 10.0)["elements"][0]["id"] == 1
    assert idx.answer(0.0, 0.0, 100.0) is None      # outside bbox -> None, not empty
    # a node in the neighbouring cell is still found (3x3 ring, not just own cell).
    # 39.9498 and 39.9500 straddle a cell boundary (multiples of CELL_DEG) but
    # are only ~22 m apart, so a cell-local scan would wrongly miss both nodes.
    assert BulkPOIs._cell(39.9498, 116.35) != BulkPOIs._cell(39.9500, 116.35)
    edge = idx.answer(39.9498, 116.3500, 100.0)["elements"]
    assert {e["id"] for e in edge} == {1, 2}, edge
    print("osm_bulk_cache self-check passed")


if __name__ == "__main__":
    demo()
