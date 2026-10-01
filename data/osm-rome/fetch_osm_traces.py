#!/usr/bin/env python3
"""
Fetch public GPS traces from OpenStreetMap for the Rome city center (the OSM Rome corpus).

This script fetches the first 50 pages of trackpoints from the OSM API
for the bounding box of the Rome city center.
"""

import time
import requests
from pathlib import Path


# Configuration
# BBOX = "23.71564,37.96660,23.73952,37.98049" #Athens
# BBOX = "23.71073,37.96140,23.76025,37.98642" #Athens center
# BBOX = "26.0912,38.9637,26.6240,39.2014" #Lesvos
# BBOX = "22.93551,40.62679,22.96280,40.63994" #Thessaloniki center
BBOX = "12.44734,41.88217,12.49686,41.90720" #Rome center
BASE_URL = "https://api.openstreetmap.org/api/0.6/trackpoints"
NUM_PAGES = 50
OFFSET = 0
OUTPUT_DIR = "rome_center"
DELAY_BETWEEN_REQUESTS = 1.0  # seconds, to be respectful to OSM servers


def fetch_trackpoints(bbox, page):
    """
    Fetch a single page of trackpoints from OSM API.

    Args:
        bbox: Bounding box string (minlon,minlat,maxlon,maxlat)
        page: Page number to fetch

    Returns:
        tuple: (success: bool, data: str or None, error: str or None)
    """
    url = f"{BASE_URL}?bbox={bbox}&page={page}"

    headers = {
        'User-Agent': 'Trajecto Research Project (educational use)'
    }

    try:
        print(f"Fetching page {page}... ", end='', flush=True)
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code == 200:
            print(f"✓ ({len(response.content)} bytes)")
            return True, response.text, None
        else:
            error_msg = f"HTTP {response.status_code}"
            print(f"✗ {error_msg}")
            return False, None, error_msg

    except requests.exceptions.RequestException as e:
        error_msg = f"Request failed: {str(e)}"
        print(f"✗ {error_msg}")
        return False, None, error_msg


def main():
    """Main execution function."""

    # Create output directory
    script_dir = Path(__file__).parent
    output_path = script_dir / OUTPUT_DIR
    output_path.mkdir(exist_ok=True)

    print(f"Fetching {NUM_PAGES} pages of GPS traces from OpenStreetMap")
    print(f"Bounding box: {BBOX}")
    print(f"Output directory: {output_path}")
    print("-" * 60)

    # Statistics
    successful = 0
    failed = 0
    total_bytes = 0

    # Fetch each page
    for page in range(NUM_PAGES):
        success, data, error = fetch_trackpoints(BBOX, page+OFFSET)

        if success and data:
            # Save to file
            output_file = output_path / f"page_{page+OFFSET:02d}.gpx"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(data)

            successful += 1
            total_bytes += len(data.encode('utf-8'))
        else:
            failed += 1
            # Log error
            error_file = output_path / f"page_{page:02d}_error.txt"
            with open(error_file, 'w') as f:
                f.write(f"Failed to fetch page {page}\n")
                f.write(f"Error: {error}\n")

        # Delay between requests (except after the last one)
        if page < NUM_PAGES - 1:
            time.sleep(DELAY_BETWEEN_REQUESTS)

    # Summary
    print("-" * 60)
    print("Completed:")
    print(f"  Successful: {successful}/{NUM_PAGES} pages")
    print(f"  Failed: {failed}/{NUM_PAGES} pages")
    print(f"  Total data: {total_bytes:,} bytes ({total_bytes/1024:.1f} KB)")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
