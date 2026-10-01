"""
Dynamic User Profile Learning Module
======================================

This module implements snapshot-based user profile learning from trajectory behavior.

Core Concepts:
- Preference updates using EWMA: P_new = 0.85 × P_old + 0.15 × O_trajectory
- Evidence-based confidence growth with diminishing returns
- Bootstrap discovered preferences from first observation

Author: Trajecto Project
Version: 1.0
"""

import math
from typing import Dict, Tuple, Any


# Fixed Parameters (POC)
LEARNING_RATE = 0.15  # Fixed learning rate (alpha)
CONFIDENCE_BASE_RATE = 0.15  # Base growth rate for confidence

# Interaction type scores
INTERACTION_SCORES = {
    'Dominant': 1.0,
    'DominantInteraction': 1.0,
    'Immersive': 0.75,
    'ImmersiveInteraction': 0.75,
    'Meaningful': 0.5,
    'MeaningfulInteraction': 0.5
}


def update_preference(p_old: float, o_observed: float, alpha: float = LEARNING_RATE) -> float:
    """
    Update preference score using Exponentially Weighted Moving Average (EWMA).

    Formula: P_new = (1 - α) × P_old + α × O_trajectory

    Args:
        p_old: Previous preference score (0.0 to 1.0)
        o_observed: Observed interaction score (0.0 to 1.0)
        alpha: Learning rate (default: 0.15)

    Returns:
        Updated preference score (0.0 to 1.0)

    Example:
        >>> round(update_preference(0.80, 1.0), 2)  # Museum preference after dominant interaction
        0.83
    """
    return (1 - alpha) * p_old + alpha * o_observed


def update_confidence(c_old: float, p_old: float, o_observed: float, n_observations: int) -> float:
    """
    Update confidence based on evidence strength and diminishing returns.

    Confidence grows when:
    - Observations match preferences (high agreement)
    - Strong interactions occur (dominant > immersive > meaningful)
    - Growth slows with more observations (diminishing returns)

    Args:
        c_old: Previous confidence level (0.0 to 1.0)
        p_old: Previous preference score (0.0 to 1.0)
        o_observed: Observed interaction score (0.0 to 1.0)
        n_observations: Number of observations for this category

    Returns:
        Updated confidence level (0.0 to 1.0)

    Example:
        >>> update_confidence(0.30, 0.80, 1.0, 1)  # First museum visit confirms preference
        0.42
    """
    # How well does observation match preference?
    agreement = 1.0 - abs(p_old - o_observed)

    # Weight by interaction strength
    evidence_strength = agreement * o_observed

    # Diminishing returns with more observations
    growth_rate = CONFIDENCE_BASE_RATE / math.sqrt(max(n_observations, 1))

    # Update confidence (capped at 1.0)
    c_new = c_old + (evidence_strength * growth_rate)
    return min(c_new, 1.0)


def get_interaction_score(interaction_type: str) -> float:
    """
    Map interaction type to observation score.

    Args:
        interaction_type: Type of POI interaction (e.g., 'Dominant', 'Immersive', 'Meaningful')

    Returns:
        Observation score (1.0 for Dominant, 0.75 for Immersive, 0.5 for Meaningful)

    Example:
        >>> get_interaction_score('DominantInteraction')
        1.0
        >>> get_interaction_score('Meaningful')
        0.5
    """
    # Extract base type if it includes "Interaction" suffix
    base_type = interaction_type.replace('Interaction', '')

    # Try both forms
    score = INTERACTION_SCORES.get(interaction_type)
    if score is None:
        score = INTERACTION_SCORES.get(base_type, 0.5)  # Default to meaningful

    return score


def initialize_discovered_preference(o_observed: float) -> Tuple[float, float]:
    """
    Bootstrap a new preference from first observation (discovered category).

    When a user interacts with a POI category not in their initial preferences,
    we bootstrap from the actual behavior rather than assuming neutral (0.5).

    Args:
        o_observed: Observed interaction score (first encounter)

    Returns:
        tuple: (preference_score, confidence_level)

    Example:
        >>> initialize_discovered_preference(1.0)  # First art gallery visit (dominant)
        (1.0, 0.15)
    """
    p_new = o_observed  # Bootstrap directly from observation
    c_new = o_observed * CONFIDENCE_BASE_RATE  # Low initial confidence
    return p_new, c_new


def create_snapshot(user_id: str,
                   trajectory_id: str,
                   trajectory_end_time: str,
                   interactions: Dict[str, str],
                   previous_preferences: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create new profile snapshot after trajectory completion.

    Args:
        user_id: User identifier
        trajectory_id: Trajectory identifier
        trajectory_end_time: ISO timestamp when trajectory ended
        interactions: Dict mapping POI category to interaction type
        previous_preferences: Previous snapshot preferences

    Returns:
        Dictionary containing snapshot data with updated preferences

    Example:
        >>> interactions = {'Museum': 'Dominant', 'Cafe': 'Meaningful'}
        >>> prev_prefs = {
        ...     'Museum': {'score': 0.80, 'confidence': 0.30, 'observation_count': 0}
        ... }
        >>> snapshot = create_snapshot('User_123', 'Traj_001', '2025-10-11T17:30:00',
        ...                           interactions, prev_prefs)
        >>> round(snapshot['preferences']['Museum']['score'], 2)
        0.83
    """
    # Initialize new preferences from previous snapshot
    new_prefs = {}
    for category, pref_data in previous_preferences.items():
        new_prefs[category] = pref_data.copy()

    # Update each visited category
    for category, interaction_type in interactions.items():
        o_observed = get_interaction_score(interaction_type)

        # Check if this is a newly discovered category
        if category not in new_prefs:
            # Bootstrap from first observation
            p_new, c_new = initialize_discovered_preference(o_observed)
            new_prefs[category] = {
                'score': p_new,
                'confidence': c_new,
                'observation_count': 1
            }
        else:
            # Update existing preference
            p_old = new_prefs[category]['score']
            c_old = new_prefs[category]['confidence']
            n_obs = new_prefs[category]['observation_count']

            # Increment observation count
            n_obs_new = n_obs + 1

            # Update preference and confidence
            new_prefs[category]['score'] = update_preference(p_old, o_observed)
            new_prefs[category]['confidence'] = update_confidence(c_old, p_old, o_observed, n_obs_new)
            new_prefs[category]['observation_count'] = n_obs_new

    # Create snapshot
    snapshot = {
        'user_id': user_id,
        'trajectory_id': trajectory_id,
        'timestamp': trajectory_end_time,
        'preferences': new_prefs
    }

    return snapshot


def format_preference_as_turtle(snapshot_id: str,
                                user_id: str,
                                snapshot: Dict[str, Any],
                                base_uri: str = "http://w3id.org/trajecto/") -> str:
    """
    Format snapshot as Turtle RDF for insertion into triplestore.

    Args:
        snapshot_id: Unique identifier for this snapshot
        user_id: User identifier
        snapshot: Snapshot dictionary from create_snapshot()
        base_uri: Base URI for RDF resources

    Returns:
        String containing Turtle RDF representation
    """
    # The profile IRI comes from the snapshot IRI, not from user_id: user_id is
    # the rdfs:label, and a label with escaped characters ("Bianca%20Millan")
    # prefixes to a different IRI than the profile ("Bianca%2520Millan_profile").
    profile_uri = snapshot_id.rsplit('_snapshot_', 1)[0] + '_profile'
    turtle = f"""
<{profile_uri}> user:hasSnapshot <{snapshot_id}>.
<{snapshot_id}> a user:ProfileSnapshot ;
    user:updateTimestamp "{snapshot['timestamp']}"^^xsd:dateTime ;
    user:learningRate {LEARNING_RATE} ;\n"""

    # Add preferences
    for category, pref_data in snapshot['preferences'].items():
        turtle += f"""    user:hasPreferences [
        a user:PoiCategoryPreference ;
        user:prefersPoiCategory <{category}> ;
        user:preferenceScore {pref_data['score']:.6f} ;
        user:confidenceLevel {pref_data['confidence']:.6f} ;
        user:observationCount {pref_data['observation_count']}
    ] ;\n"""

    # Remove trailing semicolon and newline, add period
    turtle = turtle.rstrip(" ;\n") + " .\n"

    # Link snapshot to trajectory if available
    if 'trajectory_id' in snapshot:
        turtle += f"\n<{snapshot['trajectory_id']}> user:generatedSnapshot <{snapshot_id}> .\n"

    return turtle


def get_confidence_interpretation(confidence: float) -> str:
    """
    Get human-readable interpretation of confidence level.

    Args:
        confidence: Confidence level (0.0 to 1.0)

    Returns:
        Interpretation string
    """
    if confidence < 0.3:
        return "Low (initial)"
    elif confidence < 0.6:
        return "Medium"
    elif confidence < 0.8:
        return "High"
    else:
        return "Very High"


if __name__ == "__main__":
    # Run doctests
    import doctest
    doctest.testmod()

    print("Profile Learning Module loaded successfully!")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Confidence Base Rate: {CONFIDENCE_BASE_RATE}")
    print(f"Interaction Scores: {INTERACTION_SCORES}")
