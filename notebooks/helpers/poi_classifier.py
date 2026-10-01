"""
Rule-based POI classifier using OSM tags.

Maps OSM `all_tags` (dict or dict-repr string) to Trajecto ontology POI classes.
Returns a list of `poi:<ClassName>` strings (matching the LLM classifier's output
format) or None when no rule matches. Callers should fall back to the LLM on None.
"""

from __future__ import annotations

import ast
from typing import Optional

# --- Direct value → class mappings -----------------------------------------
# Each mapping is keyed by OSM tag key; value maps OSM tag value → ontology class.

AMENITY = {
    # food & drink
    "restaurant": "Restaurant",
    "fast_food": "FastFood",
    "food_court": "FoodCourt",
    "cafe": "Cafe",
    "ice_cream": "Cafe",
    "bar": "Bar",
    "pub": "Bar",
    "biergarten": "Bar",
    "nightclub": "Nightclub",
    # culture / entertainment
    "theatre": "Theater",
    "cinema": "Cinema",
    "arts_centre": "Gallery",
    "library": "Library",
    "community_centre": "PublicService",
    # education
    "school": "School",
    "kindergarten": "School",
    "college": "University",
    "university": "University",
    # religion
    "place_of_worship": "Religious",
    "monastery": "Religious",
    # transit
    "bus_station": "BusStation",
    "ferry_terminal": "TransitStop",
    "taxi": "TransitStop",
    # services / amenities
    "bank": "Bank",
    "atm": "ATM",
    "post_office": "PostOffice",
    "pharmacy": "Pharmacy",
    "hospital": "Hospital",
    "clinic": "Clinic",
    "doctors": "Clinic",
    "dentist": "Clinic",
    "police": "PublicService",
    "fire_station": "PublicService",
    "townhall": "PublicService",
    "courthouse": "PublicService",
    "embassy": "Embassy",
    "marketplace": "MarketArea",
    "fountain": "Fountain",
    "drinking_water": "WaterFountain",
    "toilets": "Restroom",
    "bench": "SeatingArea",
    "shelter": "SeatingArea",
    "parking": "ParkingArea",
    "parking_entrance": "ParkingArea",
    "vending_machine": "VendingMachine",
    "fuel": "ServicePOI",
}

SHOP_MALL = {"mall", "department_store"}
BOOK_SHOPS = {"books"}

TOURISM = {
    "museum": "Museum",
    "gallery": "Gallery",
    "artwork": "Artwork",
    "attraction": "Landmark",
    "viewpoint": "ViewingPoint",
    "information": "Information",
    "hotel": "Hotel",
    "hostel": "Hostel",
    "guest_house": "Hotel",
    "motel": "Hotel",
    "apartment": "Accommodation",
    "chalet": "Accommodation",
    "camp_site": "Accommodation",
    "theme_park": "Recreational",
    "zoo": "Recreational",
    "aquarium": "Recreational",
    "picnic_site": "Recreational",
}

HISTORIC = {
    "monument": "Monument",
    "memorial": "Monument",
    "castle": "HistoricalLandmark",
    "fort": "HistoricalLandmark",
    "ruins": "HistoricalSite",
    "archaeological_site": "HistoricalSite",
    "tomb": "HistoricalSite",
    "city_gate": "HistoricalLandmark",
    "building": "HistoricalLandmark",
    "manor": "HistoricalLandmark",
    "church": "Religious",
    "wayside_shrine": "Religious",
    "wayside_cross": "Religious",
}

LEISURE = {
    "park": "Park",
    "garden": "Park",
    "nature_reserve": "Park",
    "playground": "Playground",
    "pitch": "SportsFacility",
    "sports_centre": "SportsFacility",
    "stadium": "Stadium",
    "fitness_centre": "Gymnasium",
    "swimming_pool": "SportsFacility",
    "marina": "LeisureFacility",
    "beach_resort": "Beach",
    "dog_park": "Park",
    "water_park": "Recreational",
}

NATURAL = {
    "beach": "Beach",
    "peak": "NaturalLandmark",
    "cave_entrance": "NaturalLandmark",
    "spring": "NaturalLandmark",
    "waterfall": "NaturalLandmark",
}

PUBLIC_TRANSPORT = {
    "station": "TransitStop",
    "stop_position": "TransitStop",
    "platform": "TransitStop",
    "stop_area": "TransitStop",
}

RAILWAY = {
    "station": "TrainStation",
    "halt": "TrainStation",
    "subway_entrance": "TrainStation",
    "tram_stop": "TransitStop",
}

AEROWAY = {
    "aerodrome": "Airport",
    "terminal": "Airport",
}

OFFICE_DIPLOMATIC = {"diplomatic"}

PLACE_LANDMARK = {"square": "Landmark", "neighbourhood": "Landmark"}

INFORMATION_VALUE = {
    "map": "Map",
    "board": "InfoDisplay",
    "terminal": "InformationKiosk",
    "office": "InformationDesk",
    "guidepost": "InformationSign",
    "tactile_map": "Map",
    "audioguide": "InfoDisplay",
}


def _coerce_tags(tags) -> dict:
    """Accept dict or Python-literal string; return a dict (empty on failure)."""
    if isinstance(tags, dict):
        return tags
    if not tags or not isinstance(tags, str):
        return {}
    try:
        parsed = ast.literal_eval(tags)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def classify_by_tags(tags) -> Optional[list[str]]:
    """
    Classify a POI from its OSM tags using deterministic rules.

    Returns
        list of `poi:<Class>` strings (1+ entries), or None if no rule matched.
    """
    t = _coerce_tags(tags)
    if not t:
        return None

    classes: list[str] = []

    def add(cls: str) -> None:
        label = f"poi:{cls}"
        if label not in classes:
            classes.append(label)

    tourism = t.get("tourism")
    if tourism in TOURISM:
        add(TOURISM[tourism])

    historic = t.get("historic")
    if historic in HISTORIC:
        add(HISTORIC[historic])

    amenity = t.get("amenity")
    if amenity in AMENITY:
        add(AMENITY[amenity])

    shop = t.get("shop")
    if shop:
        if shop in SHOP_MALL:
            add("ShoppingMall")
        elif shop in BOOK_SHOPS:
            add("RetailStore")
        else:
            add("RetailStore")

    leisure = t.get("leisure")
    if leisure in LEISURE:
        add(LEISURE[leisure])

    natural = t.get("natural")
    if natural in NATURAL:
        add(NATURAL[natural])

    railway = t.get("railway")
    if railway in RAILWAY:
        add(RAILWAY[railway])

    aeroway = t.get("aeroway")
    if aeroway in AEROWAY:
        add(AEROWAY[aeroway])

    # Bus stops: `highway=bus_stop` or `bus=yes` (often combined with public_transport)
    if t.get("highway") == "bus_stop" or t.get("bus") == "yes":
        add("BusStation")

    # Subway/metro stops expressed as `railway=stop` + `subway=yes`
    if t.get("subway") == "yes" and "poi:TrainStation" not in classes:
        add("TrainStation")

    public_transport = t.get("public_transport")
    if public_transport in PUBLIC_TRANSPORT and not any(
        c in classes for c in ("poi:TrainStation", "poi:BusStation")
    ):
        add(PUBLIC_TRANSPORT[public_transport])

    # Junctions and entrances
    if t.get("highway") == "motorway_junction":
        add("Junction")
    if t.get("entrance") and not classes:
        add("Entrance")

    if t.get("office") in OFFICE_DIPLOMATIC or t.get("diplomatic"):
        add("Embassy")

    if tourism == "information":
        info_val = t.get("information")
        if info_val in INFORMATION_VALUE:
            add(INFORMATION_VALUE[info_val])

    place = t.get("place")
    if place in PLACE_LANDMARK and not classes:
        add(PLACE_LANDMARK[place])

    if t.get("man_made") == "tower" and not classes:
        add("Tower")

    if t.get("building") == "church" and "poi:Religious" not in classes:
        add("Religious")

    if not classes:
        return None

    return classes


if __name__ == "__main__":
    # Sanity checks
    samples = [
        {"amenity": "restaurant", "name": "X"},
        {"tourism": "museum"},
        {"historic": "monument"},
        {"shop": "clothes"},
        {"leisure": "park"},
        {"public_transport": "station", "railway": "station"},
        {"tourism": "information", "information": "map"},
        {"office": "diplomatic", "diplomatic": "embassy"},
        {"name": "nothing useful"},
        "{'amenity': 'cafe', 'name': 'A'}",
    ]
    for s in samples:
        print(s, "->", classify_by_tags(s))
