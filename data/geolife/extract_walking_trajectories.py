#!/usr/bin/env python3
"""
Extract walking and running trajectories from Geolife dataset.

This script processes user directories that contain a labels.txt file,
identifies time periods where the transportation mode is 'walk' or 'run',
and extracts the corresponding GPS points from trajectory files.
"""

from datetime import datetime
from pathlib import Path
import csv


def parse_labels_file(labels_path):
    """
    Parse labels.txt file and extract walk/run time periods.

    Returns a list of tuples: (start_time, end_time, mode)
    """
    walking_periods = []

    with open(labels_path) as f:
        lines = f.readlines()

    # Skip header line
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        parts = line.split('\t')
        if len(parts) < 3:
            continue

        start_str, end_str, mode = parts[0], parts[1], parts[2].lower()

        if mode in ('walk', 'run'):
            try:
                start_time = datetime.strptime(start_str, '%Y/%m/%d %H:%M:%S')
                end_time = datetime.strptime(end_str, '%Y/%m/%d %H:%M:%S')
                walking_periods.append((start_time, end_time, mode))
            except ValueError as e:
                print(f"Warning: Could not parse time in {labels_path}: {e}")

    return walking_periods


def parse_plt_file(plt_path):
    """
    Parse a PLT trajectory file.

    Returns a list of tuples: (lat, lon, altitude, datetime, original_line)
    """
    points = []

    with open(plt_path) as f:
        lines = f.readlines()

    # Skip first 6 lines (header)
    for line in lines[6:]:
        line = line.strip()
        if not line:
            continue

        parts = line.split(',')
        if len(parts) < 7:
            continue

        try:
            lat = float(parts[0])
            lon = float(parts[1])
            altitude = float(parts[3])
            date_str = parts[5]
            time_str = parts[6]

            point_time = datetime.strptime(f"{date_str} {time_str}", '%Y-%m-%d %H:%M:%S')
            points.append((lat, lon, altitude, point_time, line))
        except (ValueError, IndexError) as e:
            continue

    return points


def get_trajectory_time_range(plt_path):
    """
    Get the start and end time of a trajectory file without loading all points.
    """
    start_time = None
    end_time = None

    with open(plt_path) as f:
        lines = f.readlines()

    # Get first point (after header)
    for line in lines[6:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) >= 7:
            try:
                start_time = datetime.strptime(f"{parts[5]} {parts[6]}", '%Y-%m-%d %H:%M:%S')
                break
            except ValueError:
                continue

    # Get last point
    for line in reversed(lines[6:]):
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) >= 7:
            try:
                end_time = datetime.strptime(f"{parts[5]} {parts[6]}", '%Y-%m-%d %H:%M:%S')
                break
            except ValueError:
                continue

    return start_time, end_time


def time_ranges_overlap(traj_start, traj_end, walk_start, walk_end):
    """Check if two time ranges overlap."""
    return traj_start <= walk_end and traj_end >= walk_start


def extract_walking_points(points, walking_periods):
    """
    Extract points that fall within walking/running periods.

    Returns a list of tuples: (lat, lon, altitude, datetime, mode, original_line)
    """
    walking_points = []

    for lat, lon, altitude, point_time, original_line in points:
        for walk_start, walk_end, mode in walking_periods:
            if walk_start <= point_time <= walk_end:
                walking_points.append((lat, lon, altitude, point_time, mode, original_line))
                break  # Point matched, no need to check other periods

    return walking_points


def process_user_directory(user_path, walking_periods):
    """
    Process all trajectory files for a user and extract walking points.

    Returns a list of dictionaries with trajectory info and walking points.
    """
    results = []
    trajectory_dir = user_path / 'Trajectory'

    if not trajectory_dir.exists():
        print(f"Warning: No Trajectory directory found in {user_path}")
        return results

    plt_files = list(trajectory_dir.glob('*.plt'))

    for plt_file in plt_files:
        # First, check if trajectory overlaps with any walking period
        traj_start, traj_end = get_trajectory_time_range(plt_file)

        if traj_start is None or traj_end is None:
            continue

        # Check for overlap with any walking period
        has_overlap = False
        for walk_start, walk_end, _ in walking_periods:
            if time_ranges_overlap(traj_start, traj_end, walk_start, walk_end):
                has_overlap = True
                break

        if not has_overlap:
            continue

        # Parse the full file and extract walking points
        points = parse_plt_file(plt_file)
        walking_points = extract_walking_points(points, walking_periods)

        if walking_points:
            results.append({
                'file': plt_file.name,
                'trajectory_start': traj_start,
                'trajectory_end': traj_end,
                'total_points': len(points),
                'walking_points': walking_points
            })

    return results


def main():
    # Base directory
    base_dir = Path(__file__).parent / 'Geolife Trajectories 1.3'  # the extracted download
    data_dir = base_dir / 'Data'

    # Output file
    output_file = base_dir / 'walking_trajectories.csv'
    summary_file = base_dir / 'walking_summary.txt'

    # Statistics
    total_users_processed = 0
    total_trajectories_with_walking = 0
    total_walking_points = 0
    users_with_walking = []

    # Collect all walking data
    all_walking_data = []

    # Process each user directory that has labels.txt
    for user_dir in sorted(data_dir.iterdir()):
        if not user_dir.is_dir():
            continue

        labels_file = user_dir / 'labels.txt'
        if not labels_file.exists():
            continue

        user_id = user_dir.name
        print(f"Processing user {user_id}...")

        # Parse walking periods from labels
        walking_periods = parse_labels_file(labels_file)

        if not walking_periods:
            print(f"  No walk/run periods found for user {user_id}")
            continue

        print(f"  Found {len(walking_periods)} walk/run periods")

        # Process trajectories
        results = process_user_directory(user_dir, walking_periods)

        if results:
            total_users_processed += 1
            users_with_walking.append(user_id)

            for result in results:
                total_trajectories_with_walking += 1
                total_walking_points += len(result['walking_points'])

                # Add to all_walking_data
                for lat, lon, altitude, point_time, mode, original_line in result['walking_points']:
                    all_walking_data.append({
                        'user_id': user_id,
                        'trajectory_file': result['file'],
                        'latitude': lat,
                        'longitude': lon,
                        'altitude': altitude,
                        'datetime': point_time.isoformat(),
                        'mode': mode
                    })

            print(f"  Extracted {sum(len(r['walking_points']) for r in results)} walking points from {len(results)} trajectories")

    # Write CSV output
    print(f"\nWriting results to {output_file}...")

    with open(output_file, 'w', newline='') as f:
        fieldnames = ['user_id', 'trajectory_file', 'latitude', 'longitude', 'altitude', 'datetime', 'mode']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_walking_data)

    # Write summary
    print(f"Writing summary to {summary_file}...")

    with open(summary_file, 'w') as f:
        f.write("Geolife Walking Trajectories Extraction Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Users with labels.txt processed: {total_users_processed}\n")
        f.write(f"Trajectories containing walking/running: {total_trajectories_with_walking}\n")
        f.write(f"Total walking/running points extracted: {total_walking_points}\n\n")
        f.write("Users with walking data:\n")
        for user_id in users_with_walking:
            f.write(f"  - {user_id}\n")

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Users with labels.txt processed: {total_users_processed}")
    print(f"Trajectories containing walking/running: {total_trajectories_with_walking}")
    print(f"Total walking/running points extracted: {total_walking_points}")
    print("\nOutput files:")
    print(f"  - {output_file}")
    print(f"  - {summary_file}")


if __name__ == '__main__':
    main()
