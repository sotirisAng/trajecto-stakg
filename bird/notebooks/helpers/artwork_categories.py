# Artwork category constants and groupings for BIRD museum dataset
# Replaces poi_categories.py from the main Trajecto pipeline

ARTWORK_CATEGORIES = [
    'Painting', 'Sculpture', 'Photography', 'Print',
    'MixedMedia', 'Mirror', 'Installation', 'Unknown'
]

# Map raw artwork medium/type strings from artworks_dataset.csv to categories
ARTWORK_TYPE_MAPPING = {
    # Painting
    'Oil on canvas': 'Painting',
    'Oil on wood': 'Painting',
    'Oil on cardboard': 'Painting',
    'Oil on paper pasted on wood': 'Painting',
    'Oil on plywood': 'Painting',
    'Oil on parquet': 'Painting',
    'Oil on copper': 'Painting',
    'Oil on canvas (isorel)': 'Painting',
    'Oil and sand on canvas': 'Painting',
    'Oil and sand on plywood': 'Painting',
    'Oil and varnish on canvas': 'Painting',
    'Oil on wood transposed onto canvas': 'Painting',
    'Oil/Tempera on wood': 'Painting',
    'Oil, ink and gouache on paper mounted on canvas': 'Painting',
    'acrylic on canvas': 'Painting',
    'Acrylic wall paint': 'Painting',
    'Tempera on wood': 'Painting',
    'Fixed under glass and oil': 'Painting',
    'Mounted under painted and eglomized glass': 'Painting',
    # Sculpture
    'Bronze': 'Sculpture',
    'Marble': 'Sculpture',
    'Terracotta': 'Sculpture',
    'Terracotta bas-relief': 'Sculpture',
    'Carved wood, painted and lacquered in gold; glass ball': 'Sculpture',
    # Photography
    'Modern black and white photographic prints': 'Photography',
    # Print
    'Color xylographic print': 'Print',
    'Color Xylography': 'Print',
    'Burin engraving': 'Print',
    # Mixed media
    'Mixed media and sand glued on wood': 'MixedMedia',
    'Felt and canvas glued on cardboard laminated on wood': 'MixedMedia',
    'Lacquered and formica mouldings on isorel': 'MixedMedia',
    # Mirror
    'Painted mirror and photographic collage': 'Mirror',
    'Painted and engraved mirror': 'Mirror',
    'Bag mirror / silk-screen printing': 'Mirror',
    # Installation
    'Video installation': 'Installation',
    # Unknown
    '-': 'Unknown',
}

# No categories excluded from recommendations in museum context
EXCLUDED_CATEGORIES = set()

# Museum rooms (37 rooms across 3 floors)
ROOM_LIST = [
    'Room_0', 'Room_1', 'Room_2', 'Room_3', 'Room_5', 'Room_7', 'Room_8',
    'Room_9', 'Room_10', 'Room_11', 'Room_12', 'Room_13', 'Room_14',
    'Room_16', 'Room_18', 'Room_19', 'Room_20', 'Room_21', 'Room_22',
    'Room_23', 'Room_24', 'Room_25', 'Room_26', 'Room_27', 'Room_28',
    'Room_29', 'Room_30', 'Room_31', 'Room_32', 'Room_33', 'Room_34',
    'Room_35', 'Room_36', 'Room_37', 'Room_38', 'Room_39', 'Room_40',
]

# Era buckets parsed from poi:creationPeriod date strings
ERA_BUCKETS = [
    'Pre-1600', '17th-century', '18th-century',
    '19th-century', 'Early-20th-century', 'Late-20th-century+',
]

# Era-based purpose mapping
ERA_TO_PURPOSE = {
    'Pre-1600': 'ClassicalArtsPurpose',
    '17th-century': 'ClassicalArtsPurpose',
    '18th-century': 'ClassicalArtsPurpose',
    '19th-century': 'ModernArtsPurpose',
    'Early-20th-century': 'ContemporaryArtsPurpose',
    'Late-20th-century+': 'ContemporaryArtsPurpose',
}


def resolve_category(artwork_type):
    """Map a raw artwork type string to a canonical category."""
    return ARTWORK_TYPE_MAPPING.get(artwork_type, 'Unknown')


def resolve_era(date_string):
    """Parse a date/period string into an era bucket."""
    import re
    if not date_string or date_string.strip() in ('-', '', 'Undated'):
        return 'Unknown'
    date_string = date_string.strip()
    m = re.search(r'(\d{4})', date_string)
    if m:
        year = int(m.group(1))
        if year < 1600:
            return 'Pre-1600'
        elif year < 1700:
            return '17th-century'
        elif year < 1800:
            return '18th-century'
        elif year < 1900:
            return '19th-century'
        elif year < 1950:
            return 'Early-20th-century'
        else:
            return 'Late-20th-century+'
    low = date_string.lower()
    if '15th' in low or '16th' in low:
        return 'Pre-1600'
    if '17th' in low:
        return '17th-century'
    if '18th' in low:
        return '18th-century'
    if '19th' in low:
        return '19th-century'
    if '20th' in low:
        return 'Early-20th-century'
    return 'Unknown'


def resolve_room(room_id):
    """Validate a room ID."""
    if room_id in ROOM_LIST:
        return room_id
    return None


def get_all_categories():
    """Return all artwork categories."""
    return list(ARTWORK_CATEGORIES)


def get_category_group(category):
    """Return the group a category belongs to (identity for artwork types)."""
    if category in ARTWORK_CATEGORIES:
        return category
    return 'Unknown'


def validate_category(category):
    """Check if a category is valid."""
    return category in ARTWORK_CATEGORIES
