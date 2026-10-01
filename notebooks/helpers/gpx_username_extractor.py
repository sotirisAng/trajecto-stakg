"""
GPX Username Extractor

This module provides functions to extract user information from GPX files
that contain OSM trace data with <url> tags in the format /user/{username}/traces/{id}.

For tracks without explicit usernames, pseudonymous identifiers are assigned
from a predefined list.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET


class GPXUsernameExtractor:
    """Extract and manage usernames from GPX track files."""

    def __init__(self, pseudonymous_users_file: str = None):
        """
        Initialize the username extractor.

        Args:
            pseudonymous_users_file: Path to JSON file containing pseudonymous user list
        """
        if pseudonymous_users_file is None:
            # Default path relative to notebooks directory
            pseudonymous_users_file = Path(__file__).parent.parent.parent / 'files' / 'pseudonymous_users.json'

        self.pseudonymous_users_file = Path(pseudonymous_users_file)
        self.pseudonymous_users = self._load_pseudonymous_users()
        self.pseudonymous_user_index = 0
        self.track_user_mapping = {}  # Maps track index to username

    def _load_pseudonymous_users(self) -> List[str]:
        """Load the list of pseudonymous users from JSON file."""
        try:
            with open(self.pseudonymous_users_file) as f:
                data = json.load(f)
                return data.get('users', [])
        except FileNotFoundError:
            print(f"Warning: Pseudonymous users file not found at {self.pseudonymous_users_file}")
            print("Using default pseudonymous user pattern")
            return [f"anonymous_user_{i:03d}" for i in range(1, 101)]

    def extract_username_from_url(self, url_text: str) -> Optional[str]:
        """
        Extract username from OSM trace URL.

        Args:
            url_text: URL string in format /user/{username}/traces/{trace_id}

        Returns:
            Username string if found, None otherwise

        Example:
            >>> extractor = GPXUsernameExtractor()
            >>> extractor.extract_username_from_url("/user/PeachyOne/traces/11353345")
            'PeachyOne'
        """
        if not url_text:
            return None

        # Pattern: /user/{username}/traces/{id}
        pattern = r'/user/([^/]+)/traces/\d+'
        match = re.search(pattern, url_text)

        if match:
            return match.group(1)
        return None

    def get_next_pseudonymous_user(self) -> str:
        """
        Get the next pseudonymous user from the list.

        Returns:
            Pseudonymous username string
        """
        if self.pseudonymous_user_index >= len(self.pseudonymous_users):
            # Cycle back to start if we run out
            self.pseudonymous_user_index = 0

        username = self.pseudonymous_users[self.pseudonymous_user_index]
        self.pseudonymous_user_index += 1
        return username

    def parse_gpx_tracks_with_users(self, gpx_file_path: str) -> List[Dict]:
        """
        Parse GPX file and extract track metadata including usernames.

        Args:
            gpx_file_path: Path to GPX file

        Returns:
            List of dictionaries containing track metadata:
            {
                'track_index': int,
                'username': str,
                'username_source': 'explicit' or 'pseudonymous',
                'track_name': str (optional),
                'track_desc': str (optional),
                'url': str (optional),
                'segment_count': int
            }
        """
        tracks_metadata = []

        try:
            tree = ET.parse(gpx_file_path)
            root = tree.getroot()

            # Handle namespace in GPX 1.0
            ns = {'gpx': 'http://www.topografix.com/GPX/1/0'}
            tracks = root.findall('gpx:trk', ns) if ns else root.findall('trk')

            # Fallback if namespace doesn't work
            if not tracks:
                tracks = root.findall('trk')

            for track_idx, track in enumerate(tracks):
                # Extract track metadata
                name_elem = track.find('gpx:name', ns) if ns else track.find('name')
                desc_elem = track.find('gpx:desc', ns) if ns else track.find('desc')
                url_elem = track.find('gpx:url', ns) if ns else track.find('url')

                track_name = name_elem.text if name_elem is not None else None
                track_desc = desc_elem.text if desc_elem is not None else None
                url_text = url_elem.text if url_elem is not None else None

                # Extract username from URL
                username = self.extract_username_from_url(url_text)
                username_source = 'explicit'

                # If no username found, assign pseudonymous user
                if username is None:
                    username = self.get_next_pseudonymous_user()
                    username_source = 'pseudonymous'

                # Count segments in this track
                segments = track.findall('gpx:trkseg', ns) if ns else track.findall('trkseg')
                if not segments:
                    segments = track.findall('trkseg')

                track_metadata = {
                    'track_index': track_idx,
                    'username': username,
                    'username_source': username_source,
                    'track_name': track_name,
                    'track_desc': track_desc,
                    'url': url_text,
                    'segment_count': len(segments)
                }

                tracks_metadata.append(track_metadata)
                self.track_user_mapping[track_idx] = username

        except Exception as e:
            print(f"Error parsing GPX file {gpx_file_path}: {e}")
            return []

        return tracks_metadata

    def get_user_for_track(self, track_index: int) -> Optional[str]:
        """
        Get the username associated with a specific track index.

        Args:
            track_index: Index of the track (0-based)

        Returns:
            Username string if track exists, None otherwise
        """
        return self.track_user_mapping.get(track_index)

    def generate_summary_report(self, tracks_metadata: List[Dict]) -> str:
        """
        Generate a human-readable summary of track usernames.

        Args:
            tracks_metadata: List of track metadata dictionaries

        Returns:
            Formatted summary string
        """
        explicit_count = sum(1 for t in tracks_metadata if t['username_source'] == 'explicit')
        pseudonymous_count = sum(1 for t in tracks_metadata if t['username_source'] == 'pseudonymous')

        report = [
            "=" * 70,
            "GPX Track Username Extraction Summary",
            "=" * 70,
            f"Total tracks processed: {len(tracks_metadata)}",
            f"Tracks with explicit usernames: {explicit_count}",
            f"Tracks with pseudonymous usernames: {pseudonymous_count}",
            "",
            "Track Details:",
            "-" * 70
        ]

        for track in tracks_metadata:
            source_indicator = "✓" if track['username_source'] == 'explicit' else "⚠"
            report.append(
                f"{source_indicator} Track {track['track_index']}: "
                f"{track['username']:20s} ({track['username_source']:12s}) "
                f"- {track['segment_count']} segment(s)"
            )

        report.append("=" * 70)
        return "\n".join(report)


# Convenience functions for quick use

def extract_usernames_from_gpx(gpx_file_path: str,
                               pseudonymous_users_file: str = None) -> List[Dict]:
    """
    Quick function to extract usernames from a GPX file.

    Args:
        gpx_file_path: Path to GPX file
        pseudonymous_users_file: Optional path to pseudonymous users JSON file

    Returns:
        List of track metadata dictionaries

    Example:
        >>> tracks = extract_usernames_from_gpx('path/to/file.gpx')
        >>> for track in tracks:
        ...     print(f"Track {track['track_index']}: {track['username']}")
    """
    extractor = GPXUsernameExtractor(pseudonymous_users_file)
    tracks_metadata = extractor.parse_gpx_tracks_with_users(gpx_file_path)
    return tracks_metadata


def print_gpx_username_summary(gpx_file_path: str,
                               pseudonymous_users_file: str = None):
    """
    Print a formatted summary of usernames in a GPX file.

    Args:
        gpx_file_path: Path to GPX file
        pseudonymous_users_file: Optional path to pseudonymous users JSON file
    """
    extractor = GPXUsernameExtractor(pseudonymous_users_file)
    tracks_metadata = extractor.parse_gpx_tracks_with_users(gpx_file_path)
    print(extractor.generate_summary_report(tracks_metadata))
