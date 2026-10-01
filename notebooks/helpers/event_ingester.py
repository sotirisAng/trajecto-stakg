"""
Scheduled-events ingester for Trajecto.

Writes scheduled:ScheduledEvent triples into Fuseki. Two sources:

  - FixtureSource: reads a JSON list (the fixtures in files/events/).
  - TicketmasterSource: queries the Ticketmaster Discovery API for a bbox
    + time range (Rome, Athens, any city with real events).

POI resolution: snap venue (lat, lon) to the nearest POI in Fuseki within
DEFAULT_POI_SNAP_METERS; events whose venue cannot be matched are skipped.

Ingestion is idempotent by (externalSource, externalId): re-running the
same source clears prior events from that source before inserting.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import requests

from .fuseki_queries import execute_sparql_query, execute_sparql_update

# Best-effort .env load so TICKETMASTER_CONSUMER_KEY is picked up without
# requiring callers to source the file manually. Silent if python-dotenv
# is absent — env vars exported in the shell still work.
try:
    from dotenv import load_dotenv  # type: ignore

    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    load_dotenv(_PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

SCHEDULED_NS = "http://w3id.org/trajecto/scheduled#"
POI_NS = "http://w3id.org/trajecto/poi#"
TRAJECTO_NS = "http://w3id.org/trajecto#"
DATACRON_NS = "http://www.datacron-project.eu/datAcron#"

DEFAULT_POI_SNAP_METERS = 150.0

# Normalized event-category vocabulary. Keep in sync with
# EVENT_ELIGIBLE_CATEGORIES in unified_recommender.py.
EVENT_CATEGORIES = {
    "Concert", "Exhibition", "Conference", "Festival",
    "Sports", "Theater", "ReligiousService", "Other",
}

_PREFIX_BLOCK = f"""
PREFIX scheduled: <{SCHEDULED_NS}>
PREFIX poi:       <{POI_NS}>
PREFIX trajecto:  <{TRAJECTO_NS}>
PREFIX datacron:  <{DATACRON_NS}>
PREFIX rdf:       <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs:      <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:       <http://www.w3.org/2001/XMLSchema#>
"""


# ── Unified internal record ─────────────────────────────────────────────

@dataclass
class EventRecord:
    external_id: str
    external_source: str       # "fixture", "ticketmaster", ...
    label: str
    category: str              # must be in EVENT_CATEGORIES
    start: datetime            # tz-aware preferred
    end: datetime
    lat: float
    lon: float

    def normalized_category(self) -> str:
        return self.category if self.category in EVENT_CATEGORIES else "Other"

    def event_uri(self) -> str:
        safe_id = self.external_id.replace("/", "_").replace(" ", "_")
        return f"{SCHEDULED_NS}{self.external_source}_{safe_id}"


# ── Sources ─────────────────────────────────────────────────────────────

class EventSource:
    """Abstract event source. Subclasses implement fetch()."""

    source_tag: str = "unknown"

    def fetch(self) -> List[EventRecord]:
        raise NotImplementedError


class FixtureSource(EventSource):
    """Reads a JSON fixture file with a list of events.

    Expected shape (list of objects):
        {
          "id": "beijing-2008-summer-palace-concert",
          "label": "Summer Palace Lantern Festival",
          "category": "Festival",
          "start": "2008-09-14T19:00:00+08:00",
          "end":   "2008-09-14T22:00:00+08:00",
          "lat":   39.999,
          "lon":   116.275
        }
    """

    source_tag = "fixture"

    def __init__(self, path: str | Path, source_tag: str = "fixture"):
        self.path = Path(path)
        self.source_tag = source_tag

    def fetch(self) -> List[EventRecord]:
        with open(self.path, encoding="utf-8") as f:
            raw = json.load(f)
        records = []
        for item in raw:
            records.append(EventRecord(
                external_id=item["id"],
                external_source=self.source_tag,
                label=item["label"],
                category=item.get("category", "Other"),
                start=datetime.fromisoformat(item["start"]),
                end=datetime.fromisoformat(item["end"]),
                lat=float(item["lat"]),
                lon=float(item["lon"]),
            ))
        return records


class TicketmasterSource(EventSource):
    """Ticketmaster Discovery API adapter.

    Requires env var TICKETMASTER_CONSUMER_KEY (Ticketmaster's term for the
    API key — the Consumer Key of the default application created at signup).
    TICKETMASTER_API_KEY is accepted as a fallback. Free tier caps at 5000
    req/day, 5 req/s.

    Params:
      bbox: (min_lat, min_lon, max_lat, max_lon)
      start / end: tz-aware datetimes bounding the query window
      max_pages: page cap (each page returns up to size=200)
    """

    source_tag = "ticketmaster"
    BASE = "https://app.ticketmaster.com/discovery/v2/events.json"

    # Ticketmaster classification segment → our vocab
    _SEGMENT_MAP = {
        "Music": "Concert",
        "Sports": "Sports",
        "Arts & Theatre": "Theater",
        "Film": "Other",
        "Miscellaneous": "Other",
    }

    def __init__(self, bbox: Tuple[float, float, float, float],
                 start: datetime, end: datetime,
                 api_key: Optional[str] = None,
                 max_pages: int = 10,
                 source_tag: Optional[str] = None):
        self.bbox = bbox
        self.start = start
        self.end = end
        self.api_key = (
            api_key
            or os.environ.get("TICKETMASTER_CONSUMER_KEY")
            or os.environ.get("TICKETMASTER_API_KEY")
        )
        if not self.api_key:
            raise RuntimeError(
                "Ticketmaster API key not set. Provide via the api_key arg or "
                "set TICKETMASTER_CONSUMER_KEY (preferred) or "
                "TICKETMASTER_API_KEY in the environment.")
        self.max_pages = max_pages
        if source_tag:
            self.source_tag = source_tag

    def _classify(self, tm_event: dict) -> str:
        classifications = tm_event.get("classifications") or []
        if not classifications:
            return "Other"
        segment = (classifications[0].get("segment") or {}).get("name", "")
        genre = (classifications[0].get("genre") or {}).get("name", "")
        # Refine a couple of common genres
        if genre in ("Festival", "Festivals"):
            return "Festival"
        if genre in ("Classical", "Opera"):
            return "Concert"
        if segment == "Arts & Theatre" and genre in ("Fine Art", "Exhibition"):
            return "Exhibition"
        return self._SEGMENT_MAP.get(segment, "Other")

    @staticmethod
    def _venue_coords(tm_event: dict) -> Optional[Tuple[float, float]]:
        venues = ((tm_event.get("_embedded") or {}).get("venues")) or []
        if not venues:
            return None
        loc = venues[0].get("location") or {}
        try:
            return float(loc["latitude"]), float(loc["longitude"])
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _parse_start(tm_event: dict) -> Optional[datetime]:
        dates = tm_event.get("dates", {}).get("start", {})
        iso = dates.get("dateTime")
        if iso:
            try:
                return datetime.fromisoformat(iso.replace("Z", "+00:00"))
            except ValueError:
                pass
        date_only = dates.get("localDate")
        time_only = dates.get("localTime") or "00:00:00"
        if date_only:
            try:
                return datetime.fromisoformat(f"{date_only}T{time_only}")
            except ValueError:
                return None
        return None

    def fetch(self) -> List[EventRecord]:
        min_lat, min_lon, max_lat, max_lon = self.bbox
        records: List[EventRecord] = []
        page = 0
        while page < self.max_pages:
            params = {
                "apikey": self.api_key,
                "size": 200,
                "page": page,
                "latlong": f"{(min_lat + max_lat)/2},{(min_lon + max_lon)/2}",
                "radius": int(_approx_radius_km(min_lat, min_lon,
                                                max_lat, max_lon)),
                "unit": "km",
                "startDateTime": self.start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "endDateTime": self.end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            resp = requests.get(self.BASE, params=params, timeout=30)
            if resp.status_code == 429:
                time.sleep(2.0)
                continue
            resp.raise_for_status()
            payload = resp.json()
            events = (payload.get("_embedded") or {}).get("events") or []
            if not events:
                break
            for e in events:
                coords = self._venue_coords(e)
                start = self._parse_start(e)
                if not coords or not start:
                    continue
                end = start
                end_iso = (e.get("dates", {}).get("end") or {}).get("dateTime")
                if end_iso:
                    try:
                        end = datetime.fromisoformat(
                            end_iso.replace("Z", "+00:00"))
                    except ValueError:
                        pass
                records.append(EventRecord(
                    external_id=e["id"],
                    external_source=self.source_tag,
                    label=e.get("name", "(unnamed event)"),
                    category=self._classify(e),
                    start=start,
                    end=end,
                    lat=coords[0],
                    lon=coords[1],
                ))
            page_info = payload.get("page", {})
            if page + 1 >= page_info.get("totalPages", 0):
                break
            page += 1
        return records


def _approx_radius_km(min_lat, min_lon, max_lat, max_lon) -> float:
    """Half-diagonal of bbox in km, used for Ticketmaster's radius param."""
    lat_km = (max_lat - min_lat) * 111.0
    lon_km = (max_lon - min_lon) * 111.0 * math.cos(
        math.radians((max_lat + min_lat) / 2))
    return math.hypot(lat_km, lon_km) / 2.0


# ── POI resolver ────────────────────────────────────────────────────────

class POIResolver:
    """Spatial nearest-POI snap within a tolerance radius (meters).

    Loads POI coordinates from Fuseki once, then does in-memory nearest-
    neighbor lookups. Haversine distance, linear scan (POI counts are in
    the low thousands — KD-tree not worth the dependency).
    """

    def __init__(self, endpoint: str, tolerance_m: float = DEFAULT_POI_SNAP_METERS):
        self.endpoint = endpoint
        self.tolerance_m = tolerance_m
        self._pois: List[Tuple[str, float, float]] = []

    def load(self) -> None:
        query = _PREFIX_BLOCK + """
        SELECT ?poi ?lat ?lon WHERE {
          ?poi a trajecto:PointOfInterest ;
               datacron:hasLatitude ?lat ;
               datacron:hasLongitude ?lon .
        }
        """
        res = execute_sparql_query(self.endpoint, query)
        if not res:
            return
        for b in res.get("results", {}).get("bindings", []):
            try:
                self._pois.append((
                    b["poi"]["value"],
                    float(b["lat"]["value"]),
                    float(b["lon"]["value"]),
                ))
            except (KeyError, ValueError):
                continue

    def resolve(self, lat: float, lon: float) -> Optional[str]:
        if not self._pois:
            return None
        best_uri, best_d = None, float("inf")
        for uri, plat, plon in self._pois:
            d = _haversine_m(lat, lon, plat, plon)
            if d < best_d:
                best_d = d
                best_uri = uri
        if best_d <= self.tolerance_m:
            return best_uri
        return None


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# ── Ingester ────────────────────────────────────────────────────────────

class EventIngester:
    """Write EventRecord batches into Fuseki as trajecto:ScheduledEvent triples."""

    def __init__(self, query_endpoint: str, update_endpoint: str,
                 tolerance_m: float = DEFAULT_POI_SNAP_METERS):
        self.query_endpoint = query_endpoint
        self.update_endpoint = update_endpoint
        self.resolver = POIResolver(query_endpoint, tolerance_m)

    def ingest(self, source: EventSource, clear_source: bool = True) -> dict:
        """Fetch events from source, snap to POIs, insert into Fuseki.

        Returns a summary dict: {source, fetched, resolved, inserted, skipped}.
        """
        self.resolver.load()
        records = source.fetch()

        if clear_source:
            self._clear_source(source.source_tag)

        inserted, skipped = 0, 0
        batch: List[str] = []
        for rec in records:
            poi_uri = self.resolver.resolve(rec.lat, rec.lon)
            if poi_uri is None:
                skipped += 1
                continue
            batch.append(self._triples_for(rec, poi_uri))
            inserted += 1

        if batch:
            insert = _PREFIX_BLOCK + "INSERT DATA {\n" + "\n".join(batch) + "\n}"
            ok = execute_sparql_update(self.update_endpoint, insert)
            if not ok:
                return {
                    "source": source.source_tag,
                    "fetched": len(records),
                    "resolved": inserted,
                    "inserted": 0,
                    "skipped": skipped,
                    "error": "insert failed",
                }

        return {
            "source": source.source_tag,
            "fetched": len(records),
            "resolved": inserted,
            "inserted": inserted,
            "skipped": skipped,
        }

    def _clear_source(self, source_tag: str) -> None:
        """Delete all events previously ingested under source_tag."""
        q = _PREFIX_BLOCK + f"""
        DELETE {{
          ?e ?p ?o .
        }}
        WHERE {{
          ?e a scheduled:ScheduledEvent ;
             scheduled:externalSource "{source_tag}" ;
             ?p ?o .
        }}
        """
        execute_sparql_update(self.update_endpoint, q)

    @staticmethod
    def _triples_for(rec: EventRecord, poi_uri: str) -> str:
        uri = rec.event_uri()
        label_literal = _escape_literal(rec.label)
        cat = rec.normalized_category()
        return (
            f"  <{uri}> a scheduled:ScheduledEvent ;\n"
            f"    rdfs:label \"{label_literal}\" ;\n"
            f"    scheduled:hostedAt <{poi_uri}> ;\n"
            f"    scheduled:startTime \"{rec.start.isoformat()}\"^^xsd:dateTime ;\n"
            f"    scheduled:endTime \"{rec.end.isoformat()}\"^^xsd:dateTime ;\n"
            f"    scheduled:eventCategory \"{cat}\" ;\n"
            f"    scheduled:externalSource \"{rec.external_source}\" ;\n"
            f"    scheduled:externalId \"{_escape_literal(rec.external_id)}\" ."
        )


def _escape_literal(s: str) -> str:
    return s.replace("\\", "\\\\").replace("\"", "\\\"")


# ── Convenience entry points ────────────────────────────────────────────

def ingest_fixture(path: str | Path, query_endpoint: str,
                   update_endpoint: str, source_tag: str = "fixture",
                   clear_source: bool = True) -> dict:
    src = FixtureSource(path, source_tag=source_tag)
    ing = EventIngester(query_endpoint, update_endpoint)
    return ing.ingest(src, clear_source=clear_source)


def ingest_ticketmaster(bbox: Tuple[float, float, float, float],
                        start: datetime, end: datetime,
                        query_endpoint: str, update_endpoint: str,
                        source_tag: str = "ticketmaster",
                        api_key: Optional[str] = None,
                        max_pages: int = 10,
                        clear_source: bool = True) -> dict:
    src = TicketmasterSource(bbox, start, end, api_key=api_key,
                             max_pages=max_pages, source_tag=source_tag)
    ing = EventIngester(query_endpoint, update_endpoint)
    return ing.ingest(src, clear_source=clear_source)
