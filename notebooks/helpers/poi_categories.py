"""
POI Categories for the Trajecto system.

This module contains the official list of POI categories available in the system,
plus alias mappings for GPT-classifier names that don't match the ontology directly.
"""

# Official POI categories in the Trajecto system
POI_CATEGORIES = [
    'Accommodation',
    'Commercial',
    'Cultural',
    'Educational',
    'Entertainment',
    'Artwork',
    'Exhibit',
    'Gallery',
    'Historical',
    'HistoricalLandmark',
    'HistoricalSite',
    'Monument',
    'Museum',
    'Information',
    'Landmark',
    'Recreational',
    'Religious',
    'LeisureFacility',
    'SportsFacility',
    'Cafe',
    'FastFood',
    'FoodCourt',
    'Restaurant',
    'CommercialStreet',
    'MarketArea',
    'RetailStore',
    'ShoppingMall',
    'Library',
    'School',
    'University',
    'NightlifeFacility',
    'Bar',
    'Nightclub',
    'PerformanceVenue',
    'Cinema',
    'Theater',
    'Beach',
    'Park'
]

# Hierarchical grouping (for reference)
# Group members use canonical POI_CATEGORIES names (ontology local names) —
# display names with spaces never match observed categories.
CATEGORY_GROUPS = {
    'Cultural': ['Artwork', 'Exhibit', 'Gallery', 'Museum'],
    'Historical': ['HistoricalLandmark', 'HistoricalSite', 'Monument', 'Historical'],
    'Food & Drink': ['Cafe', 'FastFood', 'FoodCourt', 'Restaurant', 'Bar'],
    'Shopping': ['CommercialStreet', 'MarketArea', 'RetailStore', 'ShoppingMall'],
    'Education': ['Library', 'School', 'University'],
    'Entertainment': ['NightlifeFacility', 'Bar', 'Nightclub', 'PerformanceVenue', 'Cinema', 'Theater'],
    'Recreation': ['LeisureFacility', 'SportsFacility', 'Beach', 'Park']
}

# Popular categories for initial preferences (suggestions)
POPULAR_CATEGORIES = [
    'Museum',
    'Cafe',
    'Restaurant',
    'Park',
    'Gallery',
    'HistoricalSite',
    'Beach',
    'ShoppingMall',
    'Cinema',
    'Bar'
]


# Alias mapping: GPT classifier names → canonical POI_CATEGORIES names
CATEGORY_ALIASES = {
    # Suffix mismatches (GPT adds "POI" suffix)
    'ArtworkPOI': 'Artwork',
    'CulturalPOI': 'Cultural',
    'ReligiousPOI': 'Religious',
    'RecreationalPOI': 'Recreational',
    'EducationalPOI': 'Educational',
    'LandmarkPOI': 'Landmark',
    'InformationPOI': 'Information',
    'AccommodationPOI': 'Accommodation',
    'EntertainmentPOI': 'Entertainment',
    'CommercialPOI': 'Commercial',
    # Semantic mappings
    'DiningFacility': 'Restaurant',
    'Hotel': 'Accommodation',
    'Hostel': 'Accommodation',
    'ViewingPoint': 'Landmark',
    'NavigationLandmark': 'Landmark',
    'NaturalLandmark': 'Landmark',
    'VisualLandmark': 'Landmark',
    'Temple': 'Religious',
    'PublicSquare': 'Landmark',
    'Zoo': 'Recreational',
    'Bookstore': 'RetailStore',
}


# Class names the POI classifier (OSM rules + LLM) has produced that the POI
# module does not declare -> the declared POI class to use instead. None drops
# the type; the POI keeps trajecto:PointOfInterest. Applied where notebook 2
# adds classifications to the KG. Targets carry a dwell
# configuration; Information, GatheringArea and EventSpace are avoided
# because the module places them outside PointOfInterest.
UNDECLARED_CLASS_MAP = {
    'ArtworkPOI': 'Artwork', 'ServicePOI': 'Service',
    'TransportationPOI': 'Transportation', 'TransitPOI': 'Transportation',
    'AmenityPOI': 'Amenity', 'EducationalPOI': 'Educational',
    'ReligiousPOI': 'Religious', 'InformationPOI': 'Service',
    'CulturalPOI': 'Cultural', 'AccommodationPOI': 'Accommodation',
    'RecreationalPOI': 'Recreational', 'EntertainmentPOI': 'Entertainment',
    'CommercialPOI': 'Commercial', 'LandmarkPOI': 'Landmark',
    'PromotionalPOI': 'Promotional', 'TransitionPOI': 'Transition',
    'DiningFacility': 'Dining', 'TransitStop': 'BusStation',
    'TransitStation': 'TrainStation', 'PublicTransportStation': 'TrainStation',
    'NaturalLandmark': 'Landmark', 'PublicSquare': 'Landmark',
    'Bookstore': 'RetailStore', 'BikeRentalStation': 'Transportation',
    'GovernmentOffice': 'PublicService', 'GovernmentPOI': 'PublicService',
    'AdministrativeBoundary': 'PublicService', 'PoliticalParty': 'Service',
    'Association': 'Service', 'Community': 'Service', 'Aquifer': 'Amenity',
    'None': None,
}

def resolve_category(name):
    """
    Resolve a category name to its canonical POI_CATEGORIES name.

    Checks aliases first, then direct match. Returns None if unresolvable.
    """
    if name in POI_CATEGORIES:
        return name
    if name in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[name]
    return None


def get_all_categories():
    """Get all available POI categories."""
    return sorted(POI_CATEGORIES)


def get_popular_categories():
    """Get suggested popular categories for initial preferences."""
    return sorted(POPULAR_CATEGORIES)


def get_category_group(category):
    """Get the group that a category belongs to."""
    for group, categories in CATEGORY_GROUPS.items():
        if category in categories:
            return group
    return 'Other'


def validate_category(category):
    """Check if a category is valid."""
    return category in POI_CATEGORIES


if __name__ == "__main__":
    print("POI Categories Module")
    print("=" * 60)
    print(f"\nTotal categories: {len(POI_CATEGORIES)}")
    print("\nAll categories:")
    for i, cat in enumerate(get_all_categories(), 1):
        print(f"  {i:2d}. {cat}")

    print("\n\nPopular categories (for initial preferences):")
    for i, cat in enumerate(get_popular_categories(), 1):
        print(f"  {i:2d}. {cat}")
