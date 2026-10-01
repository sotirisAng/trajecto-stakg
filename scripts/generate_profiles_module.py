"""
Generate the profiles module (ontology/modules/profiles.ttl)

Produces 117 OWL defined classes (owl:equivalentClass) for trajectory
archetypes and user profiles. No primitive classes or properties, only
compositional definitions over vocabulary from other modules. Each class gets
an rdfs:comment that states its definition in words.

    python scripts/generate_profiles_module.py
"""

import re
import sys
from pathlib import Path

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ontology_ttl import write_module  # noqa: E402

OUTPUT = Path(__file__).resolve().parent.parent / 'ontology' / 'modules' / 'profiles.ttl'
ONTOLOGY = 'http://w3id.org/trajecto/profiles'
VERSION = '1.1.0'
COMMENTS = {}  # class local name -> definition in words

# Namespaces
NS = {
    'profiles': 'http://w3id.org/trajecto/profiles#',
    'trajecto': 'http://w3id.org/trajecto#',
    'poi': 'http://w3id.org/trajecto/poi#',
    'patterns': 'http://w3id.org/trajecto/patterns#',
    'context': 'http://w3id.org/trajecto/context#',
    'purposeskos': 'http://w3id.org/trajecto/purpose/skos#',
    'events': 'http://w3id.org/trajecto/events#',
    'user': 'http://w3id.org/trajecto/user#',
    'datacron': 'http://www.datacron-project.eu/datAcron#',
    'dul': 'http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#',
    'owl': 'http://www.w3.org/2002/07/owl#',
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
    'xsd': 'http://www.w3.org/2001/XMLSchema#',
}

# 38 POI types to generate visitor classes for (excluding transport/utility/generic)
POI_TYPES = [
    'Accommodation', 'Commercial', 'Cultural', 'Educational', 'Entertainment',
    'Artwork', 'Exhibit', 'Gallery', 'Historical', 'HistoricalLandmark',
    'HistoricalSite', 'Monument', 'Museum', 'Information', 'Landmark',
    'Recreational', 'Religious', 'LeisureFacility', 'SportsFacility',
    'Cafe', 'FastFood', 'FoodCourt', 'Restaurant',
    'CommercialStreet', 'MarketArea', 'RetailStore', 'ShoppingMall',
    'Library', 'School', 'University',
    'NightlifeFacility', 'Bar', 'Nightclub', 'PerformanceVenue',
    'Cinema', 'Theater', 'Beach', 'Park',
]

# Category groups for focused/immersive classes
# Keys = group name, values = list of poi: class URIs (union members)
CATEGORY_GROUPS = {
    'Cultural': ['Artwork', 'Exhibit', 'Gallery', 'Museum', 'Cultural'],
    'Historical': ['Historical', 'HistoricalLandmark', 'HistoricalSite', 'Monument'],
    'Dining': ['Cafe', 'FastFood', 'FoodCourt', 'Restaurant'],
    'Shopping': ['CommercialStreet', 'MarketArea', 'RetailStore', 'ShoppingMall', 'Commercial'],
    'Entertainment': ['NightlifeFacility', 'Bar', 'Nightclub', 'PerformanceVenue', 'Cinema', 'Theater', 'Entertainment'],
    'Recreation': ['LeisureFacility', 'SportsFacility', 'Beach', 'Park', 'Recreational'],
}

# Purpose SKOS concepts → (SKOS local name, class label prefix)
# hasPurpose stores purposeskos: URIs derived from POI class hierarchy via skos:related
PURPOSES = [
    ('CulturalPurpose', 'Cultural'),
    ('DiningExperience', 'Dining'),
    ('Shopping', 'Shopping'),
    ('RecreationalPurpose', 'Recreation'),
    ('CommercialPurpose', 'Commercial'),
    ('LeisureActivity', 'Leisure'),
]

# Navigation styles
NAV_STYLES = [
    ('CasualMovement', 'Casual'),
    ('ExploratoryMovement', 'Explorer'),
    ('ImmersiveMovement', 'Immersive'),
    ('TargetedSegment', 'Targeted'),
]

# Temporal contexts
TEMPORAL_CONTEXTS = [
    ('Morning', 'context'),
    ('Afternoon', 'context'),
    ('Evening', 'context'),
    ('Night', 'context'),
    ('Weekday', 'context'),
    ('Weekend', 'context'),
    ('Spring', 'context'),
    ('Summer', 'context'),
    ('Fall', 'context'),
    ('Winter', 'context'),
    ('HolidayPeriod', 'context'),  # Stage 4 marks holidays with context:SpecialPeriod
]

# Engagement levels
ENGAGEMENT_LEVELS = ['HighEngagement', 'MediumEngagement', 'LowEngagement']


def xml_header():
    ns_attrs = '\n     '.join(f'xmlns:{k}="{v}"' for k, v in NS.items())
    return f'''<?xml version="1.0"?>
<rdf:RDF xmlns="{NS['profiles']}"
     xml:base="http://w3id.org/trajecto/profiles"
     {ns_attrs}>
    <owl:Ontology rdf:about="http://w3id.org/trajecto/profiles"/>

'''


def xml_footer():
    return '</rdf:RDF>\n'


def section_comment(title):
    return f'''
    <!--
    ///////////////////////////////////////////////////////////////////////////////////////
    //
    // {title}
    //
    ///////////////////////////////////////////////////////////////////////////////////////
     -->

'''


def visitor_class(poi_type):
    """Generate existential visitor class: trajectory has >=1 meaningful interaction with this POI type."""
    class_name = f'{poi_type}Visitor'
    COMMENTS[class_name] = f'A trajectory with at least one meaningful interaction at a poi:{poi_type}.'
    return f'''    <!-- {NS['profiles']}{class_name} -->

    <owl:Class rdf:about="{NS['profiles']}{class_name}">
        <owl:equivalentClass>
            <owl:Class>
                <owl:intersectionOf rdf:parseType="Collection">
                    <rdf:Description rdf:about="{NS['datacron']}Trajectory"/>
                    <owl:Restriction>
                        <owl:onProperty rdf:resource="{NS['trajecto']}includesInteraction"/>
                        <owl:someValuesFrom>
                            <owl:Class>
                                <owl:intersectionOf rdf:parseType="Collection">
                                    <rdf:Description rdf:about="{NS['poi']}MeaningfulInteraction"/>
                                    <owl:Restriction>
                                        <owl:onProperty rdf:resource="{NS['poi']}interactsWith"/>
                                        <owl:someValuesFrom rdf:resource="{NS['poi']}{poi_type}"/>
                                    </owl:Restriction>
                                </owl:intersectionOf>
                            </owl:Class>
                        </owl:someValuesFrom>
                    </owl:Restriction>
                </owl:intersectionOf>
            </owl:Class>
        </owl:equivalentClass>
        <rdfs:label xml:lang="en">{poi_type} Visitor</rdfs:label>
    </owl:Class>

'''


def focused_trajectory_class(group_name, poi_types):
    """Generate focused trajectory class: >=3 meaningful interactions within a category group."""
    class_name = f'{group_name}FocusedTrajectory'
    COMMENTS[class_name] = (f'A trajectory with at least three meaningful interactions at {group_name.lower()} POIs '
                            f'({", ".join(poi_types)}).')
    union_members = '\n                                    '.join(
        f'<rdf:Description rdf:about="{NS["poi"]}{t}"/>' for t in poi_types
    )
    return f'''    <!-- {NS['profiles']}{class_name} -->

    <owl:Class rdf:about="{NS['profiles']}{class_name}">
        <owl:equivalentClass>
            <owl:Class>
                <owl:intersectionOf rdf:parseType="Collection">
                    <rdf:Description rdf:about="{NS['datacron']}Trajectory"/>
                    <owl:Restriction>
                        <owl:onProperty rdf:resource="{NS['trajecto']}includesInteraction"/>
                        <owl:minQualifiedCardinality rdf:datatype="{NS['xsd']}nonNegativeInteger">3</owl:minQualifiedCardinality>
                        <owl:onClass>
                            <owl:Class>
                                <owl:intersectionOf rdf:parseType="Collection">
                                    <rdf:Description rdf:about="{NS['poi']}MeaningfulInteraction"/>
                                    <owl:Restriction>
                                        <owl:onProperty rdf:resource="{NS['poi']}interactsWith"/>
                                        <owl:someValuesFrom>
                                            <owl:Class>
                                                <owl:unionOf rdf:parseType="Collection">
                                                    {union_members}
                                                </owl:unionOf>
                                            </owl:Class>
                                        </owl:someValuesFrom>
                                    </owl:Restriction>
                                </owl:intersectionOf>
                            </owl:Class>
                        </owl:onClass>
                    </owl:Restriction>
                </owl:intersectionOf>
            </owl:Class>
        </owl:equivalentClass>
        <rdfs:label xml:lang="en">{group_name} Focused Trajectory</rdfs:label>
    </owl:Class>

'''


def immersive_visitor_class(group_name, poi_types):
    """Generate immersive visitor class: has immersive interaction at POI in this group."""
    class_name = f'Immersive{group_name}Visitor'
    COMMENTS[class_name] = (f'A trajectory with at least one immersive interaction at a {group_name.lower()} POI '
                            f'({", ".join(poi_types)}).')
    union_members = '\n                                    '.join(
        f'<rdf:Description rdf:about="{NS["poi"]}{t}"/>' for t in poi_types
    )
    return f'''    <!-- {NS['profiles']}{class_name} -->

    <owl:Class rdf:about="{NS['profiles']}{class_name}">
        <owl:equivalentClass>
            <owl:Class>
                <owl:intersectionOf rdf:parseType="Collection">
                    <rdf:Description rdf:about="{NS['datacron']}Trajectory"/>
                    <owl:Restriction>
                        <owl:onProperty rdf:resource="{NS['trajecto']}includesInteraction"/>
                        <owl:someValuesFrom>
                            <owl:Class>
                                <owl:intersectionOf rdf:parseType="Collection">
                                    <rdf:Description rdf:about="{NS['poi']}ImmersiveInteraction"/>
                                    <owl:Restriction>
                                        <owl:onProperty rdf:resource="{NS['poi']}interactsWith"/>
                                        <owl:someValuesFrom>
                                            <owl:Class>
                                                <owl:unionOf rdf:parseType="Collection">
                                                    {union_members}
                                                </owl:unionOf>
                                            </owl:Class>
                                        </owl:someValuesFrom>
                                    </owl:Restriction>
                                </owl:intersectionOf>
                            </owl:Class>
                        </owl:someValuesFrom>
                    </owl:Restriction>
                </owl:intersectionOf>
            </owl:Class>
        </owl:equivalentClass>
        <rdfs:label xml:lang="en">Immersive {group_name} Visitor</rdfs:label>
    </owl:Class>

'''


def purpose_nav_class(purpose_skos, purpose_label, nav_style, nav_label):
    """Generate purpose-navigation composite: trajectory with this purpose and nav style."""
    class_name = f'{purpose_label}{nav_label}'
    COMMENTS[class_name] = (f'A trajectory with purpose purposeskos:{purpose_skos} that exhibits the '
                            f'patterns:{nav_style} navigation pattern.')
    return f'''    <!-- {NS['profiles']}{class_name} -->

    <owl:Class rdf:about="{NS['profiles']}{class_name}">
        <owl:equivalentClass>
            <owl:Class>
                <owl:intersectionOf rdf:parseType="Collection">
                    <rdf:Description rdf:about="{NS['datacron']}Trajectory"/>
                    <owl:Restriction>
                        <owl:onProperty rdf:resource="{NS['trajecto']}hasPurpose"/>
                        <owl:hasValue rdf:resource="{NS['purposeskos']}{purpose_skos}"/>
                    </owl:Restriction>
                    <owl:Restriction>
                        <owl:onProperty rdf:resource="{NS['patterns']}exhibitsPattern"/>
                        <owl:hasValue rdf:resource="{NS['patterns']}{nav_style}"/>
                    </owl:Restriction>
                </owl:intersectionOf>
            </owl:Class>
        </owl:equivalentClass>
        <rdfs:label xml:lang="en">{purpose_label} {nav_label}</rdfs:label>
    </owl:Class>

'''


def temporal_class(context_name, ns_prefix):
    """Generate temporal profile class."""
    class_name = f'{context_name}Trajectory'
    value = 'SpecialPeriod' if context_name == 'HolidayPeriod' else context_name
    COMMENTS[class_name] = f'A trajectory with temporal context context:{value}.'
    return f'''    <!-- {NS['profiles']}{class_name} -->

    <owl:Class rdf:about="{NS['profiles']}{class_name}">
        <owl:equivalentClass>
            <owl:Class>
                <owl:intersectionOf rdf:parseType="Collection">
                    <rdf:Description rdf:about="{NS['datacron']}Trajectory"/>
                    <owl:Restriction>
                        <owl:onProperty rdf:resource="{NS['context']}hasTemporalContext"/>
                        <owl:hasValue rdf:resource="{NS[ns_prefix]}{value}"/>
                    </owl:Restriction>
                </owl:intersectionOf>
            </owl:Class>
        </owl:equivalentClass>
        <rdfs:label xml:lang="en">{context_name} Trajectory</rdfs:label>
    </owl:Class>

'''


def engagement_class(level):
    """Generate engagement profile class."""
    label = level.replace('Engagement', ' Engagement')
    class_name = f'{level}Trajectory'
    COMMENTS[class_name] = f'A trajectory whose engagement level is trajecto:{level}.'
    return f'''    <!-- {NS['profiles']}{class_name} -->

    <owl:Class rdf:about="{NS['profiles']}{class_name}">
        <owl:equivalentClass>
            <owl:Class>
                <owl:intersectionOf rdf:parseType="Collection">
                    <rdf:Description rdf:about="{NS['datacron']}Trajectory"/>
                    <owl:Restriction>
                        <owl:onProperty rdf:resource="{NS['trajecto']}hasEngagementLevel"/>
                        <owl:hasValue rdf:resource="{NS['trajecto']}{level}"/>
                    </owl:Restriction>
                </owl:intersectionOf>
            </owl:Class>
        </owl:equivalentClass>
        <rdfs:label xml:lang="en">{label} Trajectory</rdfs:label>
    </owl:Class>

'''


def interaction_profile_class(class_name, label, interaction_types):
    """Generate interaction profile class using someValuesFrom for each type."""
    COMMENTS[class_name] = ('A trajectory that includes ' +
                            ' and '.join(f'at least one poi:{t}' for t in interaction_types) + '.')
    restrictions = ''
    for itype in interaction_types:
        restrictions += f'''                    <owl:Restriction>
                        <owl:onProperty rdf:resource="{NS['trajecto']}includesInteraction"/>
                        <owl:someValuesFrom rdf:resource="{NS['poi']}{itype}"/>
                    </owl:Restriction>
'''
    return f'''    <!-- {NS['profiles']}{class_name} -->

    <owl:Class rdf:about="{NS['profiles']}{class_name}">
        <owl:equivalentClass>
            <owl:Class>
                <owl:intersectionOf rdf:parseType="Collection">
                    <rdf:Description rdf:about="{NS['datacron']}Trajectory"/>
{restrictions}                </owl:intersectionOf>
            </owl:Class>
        </owl:equivalentClass>
        <rdfs:label xml:lang="en">{label}</rdfs:label>
    </owl:Class>

'''


def event_arc_class(class_name, label, components):
    """Generate event arc class as intersection of named profile/event classes."""
    COMMENTS.setdefault(class_name, 'A trajectory that is ' + ' and '.join(
        'a profiles:' + c.rsplit('#', 1)[-1] for c in components) + '.')
    members = '\n                    '.join(
        f'<rdf:Description rdf:about="{c}"/>' for c in components
    )
    return f'''    <!-- {NS['profiles']}{class_name} -->

    <owl:Class rdf:about="{NS['profiles']}{class_name}">
        <owl:equivalentClass>
            <owl:Class>
                <owl:intersectionOf rdf:parseType="Collection">
                    {members}
                </owl:intersectionOf>
            </owl:Class>
        </owl:equivalentClass>
        <rdfs:label xml:lang="en">{label}</rdfs:label>
    </owl:Class>

'''


def event_existence_class(class_name, label, event_class):
    """Generate class for trajectory that has at least one event of a given type."""
    COMMENTS[class_name] = f'A trajectory with at least one events:{event_class} event.'
    return f'''    <!-- {NS['profiles']}{class_name} -->

    <owl:Class rdf:about="{NS['profiles']}{class_name}">
        <owl:equivalentClass>
            <owl:Class>
                <owl:intersectionOf rdf:parseType="Collection">
                    <rdf:Description rdf:about="{NS['datacron']}Trajectory"/>
                    <owl:Restriction>
                        <owl:onProperty rdf:resource="{NS['trajecto']}hasEvent"/>
                        <owl:someValuesFrom rdf:resource="{NS['events']}{event_class}"/>
                    </owl:Restriction>
                </owl:intersectionOf>
            </owl:Class>
        </owl:equivalentClass>
        <rdfs:label xml:lang="en">{label}</rdfs:label>
    </owl:Class>

'''


def user_frequency_class(group_name, visitor_class_name):
    """Generate user-level frequency class: user with >=3 trajectories of a visitor type."""
    class_name = f'Frequent{group_name}Visitor'
    COMMENTS[class_name] = (f'A user profile with at least three trajectories classified as '
                            f'profiles:{visitor_class_name}.')
    return f'''    <!-- {NS['profiles']}{class_name} -->

    <owl:Class rdf:about="{NS['profiles']}{class_name}">
        <owl:equivalentClass>
            <owl:Class>
                <owl:intersectionOf rdf:parseType="Collection">
                    <rdf:Description rdf:about="{NS['user']}UserProfile"/>
                    <owl:Restriction>
                        <owl:onProperty rdf:resource="{NS['user']}hasTrajectory"/>
                        <owl:minQualifiedCardinality rdf:datatype="{NS['xsd']}nonNegativeInteger">3</owl:minQualifiedCardinality>
                        <owl:onClass rdf:resource="{NS['profiles']}{visitor_class_name}"/>
                    </owl:Restriction>
                </owl:intersectionOf>
            </owl:Class>
        </owl:equivalentClass>
        <rdfs:label xml:lang="en">Frequent {group_name} Visitor</rdfs:label>
    </owl:Class>

'''


def generate():
    parts = [xml_header()]
    count = 0

    # --- Per-type existential visitor classes (38 types) ---
    parts.append(section_comment('Per-Type Visitor Classes (Trajectory Level)'))
    for poi_type in POI_TYPES:
        parts.append(visitor_class(poi_type))
        count += 1

    # --- Per-group focused trajectory classes (6 groups) ---
    parts.append(section_comment('Category Group Focused Trajectory Classes'))
    for group_name, poi_types in CATEGORY_GROUPS.items():
        parts.append(focused_trajectory_class(group_name, poi_types))
        count += 1

    # --- Per-group immersive visitor classes (6 groups) ---
    parts.append(section_comment('Category Group Immersive Visitor Classes'))
    for group_name, poi_types in CATEGORY_GROUPS.items():
        parts.append(immersive_visitor_class(group_name, poi_types))
        count += 1

    # --- Purpose-navigation composites (24 classes) ---
    parts.append(section_comment('Purpose-Navigation Composite Classes'))
    for purpose_skos, purpose_label in PURPOSES:
        for nav_style, nav_label in NAV_STYLES:
            parts.append(purpose_nav_class(purpose_skos, purpose_label, nav_style, nav_label))
            count += 1

    # --- Temporal profile classes (12) ---
    parts.append(section_comment('Temporal Profile Classes'))
    for ctx_name, ns_prefix in TEMPORAL_CONTEXTS:
        parts.append(temporal_class(ctx_name, ns_prefix))
        count += 1

    # --- Engagement profile classes (3) ---
    parts.append(section_comment('Engagement Profile Classes'))
    for level in ENGAGEMENT_LEVELS:
        parts.append(engagement_class(level))
        count += 1

    # --- Interaction profile classes (5) ---
    parts.append(section_comment('Interaction Profile Classes'))
    interaction_classes = [
        ('ImmersiveDominantTrajectory', 'Immersive Dominant Trajectory',
         ['ImmersiveInteraction', 'DominantInteraction']),
        ('HasDominantInteraction', 'Has Dominant Interaction',
         ['DominantInteraction']),
        ('HasStartInteraction', 'Has Start Interaction',
         ['StartInteraction']),
        ('HasDestinationInteraction', 'Has Destination Interaction',
         ['DestinationInteraction']),
        ('HasImmersiveInteraction', 'Has Immersive Interaction',
         ['ImmersiveInteraction']),
    ]
    for cname, label, itypes in interaction_classes:
        parts.append(interaction_profile_class(cname, label, itypes))
        count += 1

    # --- Event existence classes (5) ---
    parts.append(section_comment('Event Existence Classes'))
    event_classes = [
        ('HasEngagementEscalation', 'Has Engagement Escalation', 'EngagementEscalation'),
        ('HasEngagementDrop', 'Has Engagement Drop', 'EngagementDrop'),
        ('HasCategoryDiscovery', 'Has Category Discovery', 'CategoryDiscovery'),
        ('HasCategoryTransition', 'Has Category Transition', 'CategoryTransition'),
        ('HasDominantEncounter', 'Has Dominant Encounter', 'DominantEncounter'),
    ]
    for cname, label, evt_class in event_classes:
        parts.append(event_existence_class(cname, label, evt_class))
        count += 1

    # --- Event arc classes (~8) ---
    parts.append(section_comment('Event Arc Classes'))
    P = NS['profiles']
    # EngagementFocusedArc: high engagement + dominant encounter
    parts.append(event_arc_class(
        'EngagementFocusedArc', 'Engagement Focused Arc',
        [f'{P}HighEngagementTrajectory', f'{P}HasDominantEncounter']))
    count += 1

    # Manually add CasualToImmersiveArc (needs hasValue restriction for nav style)
    COMMENTS['CasualToImmersiveArc'] = ('A casual trajectory (patterns:CasualMovement) with an engagement escalation '
                                        'event and at least one immersive interaction.')
    parts.append(f'''    <!-- {NS['profiles']}CasualToImmersiveArc -->

    <owl:Class rdf:about="{NS['profiles']}CasualToImmersiveArc">
        <owl:equivalentClass>
            <owl:Class>
                <owl:intersectionOf rdf:parseType="Collection">
                    <rdf:Description rdf:about="{NS['profiles']}HasEngagementEscalation"/>
                    <owl:Restriction>
                        <owl:onProperty rdf:resource="{NS['patterns']}exhibitsPattern"/>
                        <owl:hasValue rdf:resource="{NS['patterns']}CasualMovement"/>
                    </owl:Restriction>
                    <rdf:Description rdf:about="{NS['profiles']}HasImmersiveInteraction"/>
                </owl:intersectionOf>
            </owl:Class>
        </owl:equivalentClass>
        <rdfs:label xml:lang="en">Casual to Immersive Arc</rdfs:label>
    </owl:Class>

''')
    count += 1

    # DiscoveryDrivenArc: >=2 category discoveries + exploratory
    COMMENTS['DiscoveryDrivenArc'] = ('An exploratory trajectory (patterns:ExploratoryMovement) with at least two '
                                      'category discovery events.')
    parts.append(f'''    <!-- {NS['profiles']}DiscoveryDrivenArc -->

    <owl:Class rdf:about="{NS['profiles']}DiscoveryDrivenArc">
        <owl:equivalentClass>
            <owl:Class>
                <owl:intersectionOf rdf:parseType="Collection">
                    <owl:Restriction>
                        <owl:onProperty rdf:resource="{NS['trajecto']}hasEvent"/>
                        <owl:minQualifiedCardinality rdf:datatype="{NS['xsd']}nonNegativeInteger">2</owl:minQualifiedCardinality>
                        <owl:onClass rdf:resource="{NS['events']}CategoryDiscovery"/>
                    </owl:Restriction>
                    <owl:Restriction>
                        <owl:onProperty rdf:resource="{NS['patterns']}exhibitsPattern"/>
                        <owl:hasValue rdf:resource="{NS['patterns']}ExploratoryMovement"/>
                    </owl:Restriction>
                </owl:intersectionOf>
            </owl:Class>
        </owl:equivalentClass>
        <rdfs:label xml:lang="en">Discovery Driven Arc</rdfs:label>
    </owl:Class>

''')
    count += 1

    # --- Multi-dimensional archetype classes (~10) ---
    parts.append(section_comment('Multi-Dimensional Archetype Classes'))
    archetypes = [
        ('MorningCulturalExplorer', 'Morning Cultural Explorer', [
            f'{P}CulturalExplorer', f'{P}MorningTrajectory']),
        ('WeekendImmersiveDiner', 'Weekend Immersive Diner', [
            f'{P}WeekendTrajectory', f'{P}ImmersiveDiningVisitor', f'{P}HighEngagementTrajectory']),
        ('MorningCulturalCasual', 'Morning Cultural Casual', [
            f'{P}CulturalCasual', f'{P}MorningTrajectory']),
        ('WeekendRecreationExplorer', 'Weekend Recreation Explorer', [
            f'{P}WeekendTrajectory', f'{P}RecreationExplorer']),
        ('EveningRecreationTargeted', 'Evening Recreation Targeted', [
            f'{P}EveningTrajectory', f'{P}RecreationTargeted']),
        ('HighEngagementCulturalExplorer', 'High Engagement Cultural Explorer', [
            f'{P}HighEngagementTrajectory', f'{P}CulturalExplorer']),
        ('SummerRecreationCasual', 'Summer Recreation Casual', [
            f'{P}SummerTrajectory', f'{P}RecreationCasual']),
        ('WeekdayDiningCasual', 'Weekday Dining Casual', [
            f'{P}WeekdayTrajectory', f'{P}DiningCasual']),
        ('HistoricalDiscoveryExplorer', 'Historical Discovery Explorer', [
            f'{P}HistoricalFocusedTrajectory', f'{P}HasCategoryDiscovery']),
        ('ImmersiveCulturalMorning', 'Immersive Cultural Morning', [
            f'{P}ImmersiveCulturalVisitor', f'{P}MorningTrajectory']),
    ]
    for cname, label, components in archetypes:
        parts.append(event_arc_class(cname, label, components))
        count += 1

    # --- User-level frequency classes (6 groups) ---
    parts.append(section_comment('User-Level Frequency Classes'))
    # Map group names to the first visitor class that covers the group
    group_visitor_map = {
        'Cultural': 'CulturalVisitor',
        'Historical': 'HistoricalVisitor',
        'Dining': 'RestaurantVisitor',
        'Shopping': 'RetailStoreVisitor',
        'Entertainment': 'EntertainmentVisitor',
        'Recreation': 'ParkVisitor',
    }
    for group_name, visitor_cls in group_visitor_map.items():
        parts.append(user_frequency_class(group_name, visitor_cls))
        count += 1

    parts.append(xml_footer())

    g = Graph().parse(data=''.join(parts), format='xml')
    onto = URIRef(ONTOLOGY)
    for p, o in [
        (URIRef('http://purl.org/dc/terms/title'), Literal('Trajecto Trajectory Profiles Module', lang='en')),
        (RDFS.label, Literal('trajecto-profiles', lang='en')),
        (URIRef('http://purl.org/dc/terms/description'), Literal(
            'Trajectory profile classes defined by necessary and sufficient conditions over purpose, navigation '
            'pattern, engagement level, interaction types, temporal context and trajectory events. An OWL reasoner '
            'assigns them to trajectories.',
            lang='en')),
        (URIRef('http://purl.org/dc/terms/creator'), Literal('Sotirios Angelis')),
        (URIRef('http://purl.org/dc/terms/creator'), Literal('Konstantinos Kotis')),
        (URIRef('http://purl.org/dc/terms/publisher'), Literal(
            'i-Lab, Department of Cultural Technology and Communication, University of the Aegean')),
        (URIRef('http://purl.org/dc/terms/license'), URIRef('https://creativecommons.org/publicdomain/zero/1.0/')),
        (URIRef('http://purl.org/dc/terms/created'), Literal('2026-03-29', datatype=XSD.date)),
        (URIRef('http://purl.org/dc/terms/modified'), Literal('2026-10-01', datatype=XSD.date)),
        (URIRef('http://purl.org/dc/terms/issued'), Literal('2026-10-01', datatype=XSD.date)),
        (OWL.versionInfo, Literal(VERSION)),
        (OWL.versionIRI, URIRef(f'{ONTOLOGY}/{VERSION}')),
        (OWL.priorVersion, URIRef(f'{ONTOLOGY}/1.0')),
        (URIRef('http://purl.org/vocab/vann/preferredNamespacePrefix'), Literal('profiles')),
        (URIRef('http://purl.org/vocab/vann/preferredNamespaceUri'), Literal(NS['profiles'])),
    ] + [(OWL.imports, URIRef(f'http://w3id.org/trajecto/{m}'))
         for m in ('events', 'poi', 'patterns', 'context', 'purpose/skos', 'user')]:
        g.add((onto, p, o))
    for prop in ('title', 'description', 'creator', 'publisher', 'license', 'created', 'modified', 'issued'):
        g.add((URIRef(f'http://purl.org/dc/terms/{prop}'), RDF.type, OWL.AnnotationProperty))
    for prop in ('preferredNamespacePrefix', 'preferredNamespaceUri'):
        g.add((URIRef(f'http://purl.org/vocab/vann/{prop}'), RDF.type, OWL.AnnotationProperty))
    for cls, label in list(g.subject_objects(RDFS.label)):  # "CommercialStreet Visitor" -> "Commercial Street Visitor"
        g.set((cls, RDFS.label, Literal(re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', str(label)), lang='en')))
    for name, text in COMMENTS.items():
        cls = URIRef(NS['profiles'] + name)
        g.add((cls, RDFS.comment, Literal(text, lang='en')))
        g.add((cls, RDFS.isDefinedBy, onto))
    write_module(g, OUTPUT, NS['profiles'])
    print(f"Generated {count} defined classes -> {OUTPUT}")


if __name__ == '__main__':
    generate()
