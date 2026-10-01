#!/usr/bin/env python3
"""
BIRD Museum Pipeline Runner

Runs all BIRD notebooks in sequence using papermill.
Outputs are saved to the 'output/pipeline/' directory with execution logs.

Usage:
    python run_pipeline.py              # Run full pipeline (B0-B6)
    python run_pipeline.py --from 2     # Start from notebook B2
    python run_pipeline.py --only 0 1   # Run only B0 and B1
    python run_pipeline.py --all        # Run pipeline + analytics (BF*)
    python run_pipeline.py --dry-run    # Show what would run without executing
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    import papermill as pm
except ImportError:
    print("papermill not installed. Installing...")
    os.system(f"{sys.executable} -m pip install papermill")
    import papermill as pm


# Core pipeline — must run in order
PIPELINE = [
    ('B0-data-loading.ipynb', 'Load raw BIRD dataset, produce CSVs'),
    ('B1-rdf-generation.ipynb', 'Generate RDF, upload to Fuseki, insert artwork types + room + era'),
    ('B2-enrichment.ipynb', 'SPARQL enrichment: sequences, interaction classification'),
    ('B3-dominant-artworks.ipynb', 'Identify dominant and significant artworks'),
    ('B4-navigation-styles.ipynb', 'Classify navigation patterns'),
    ('B5-purpose-classification.ipynb', 'Assign era-based purpose to trajectories'),
    ('B6-user-profiles.ipynb', 'Create user profiles with category preferences'),
]

# Analytics — optional, run after pipeline
ANALYTICS = [
    ('BF1-kg-explorer.ipynb', 'Explore KG structure and entity counts'),
    ('BF2-graph-analytics.ipynb', 'Graph analytics: centrality, communities'),
    ('BF3-trajectory-profiles.ipynb', 'Extract 72-dim semantic features, compute similarity'),
    ('BF4-clustering-collab.ipynb', 'Clustering + collaborative filtering'),
    ('BF5-context-recommendations.ipynb', 'Context-aware multi-signal recommendations + evaluation'),
    ('BF6-prediction.ipynb', 'Dwell time + next artwork prediction'),
]


def run_notebook(notebook_path: Path, output_dir: Path, parameters: dict = None):
    """Run a single notebook and save output."""
    output_path = output_dir / f"{notebook_path.stem}_executed.ipynb"

    print(f"\n{'='*60}")
    print(f"Running: {notebook_path.name}")
    print(f"Output:  {output_path.name}")
    print(f"{'='*60}")

    start_time = datetime.now()

    try:
        pm.execute_notebook(
            str(notebook_path),
            str(output_path),
            parameters=parameters or {},
            kernel_name='python3',
            progress_bar=True
        )

        elapsed = datetime.now() - start_time
        print(f"Completed in {elapsed.total_seconds():.1f}s")
        return True, elapsed.total_seconds()

    except pm.PapermillExecutionError as e:
        elapsed = datetime.now() - start_time
        print(f"\nFAILED after {elapsed.total_seconds():.1f}s")
        print(f"Error in cell {e.exec_count}:")
        print(f"  {e.ename}: {e.evalue}")
        return False, elapsed.total_seconds()


def main():
    parser = argparse.ArgumentParser(description='Run BIRD museum notebook pipeline')
    parser.add_argument('--from', dest='start_from', type=int, default=0,
                        help='Start from notebook number (0=B0, 1=B1, ...)')
    parser.add_argument('--only', type=int, nargs='+',
                        help='Run only specific pipeline notebook numbers')
    parser.add_argument('--all', action='store_true',
                        help='Run pipeline + analytics notebooks')
    parser.add_argument('--analytics-only', nargs='*', type=int, default=None,
                        help='Run analytics notebooks. No args = all, or specify numbers (e.g. --analytics-only 4 5)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would run without executing')
    parser.add_argument('--no-stop-on-error', action='store_true',
                        help='Continue running after a notebook fails')
    parser.add_argument('--output-dir', type=str, default='output/pipeline',
                        help='Directory for executed notebooks')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    output_dir = script_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which notebooks to run
    notebooks_to_run = []

    if args.analytics_only is not None:
        indices = args.analytics_only if args.analytics_only else range(len(ANALYTICS))
        for i in indices:
            if i < len(ANALYTICS):
                notebooks_to_run.append((f'A{i}', ANALYTICS[i]))
    elif args.only:
        for i in args.only:
            if i < len(PIPELINE):
                notebooks_to_run.append((f'B{i}', PIPELINE[i]))
    else:
        for i in range(args.start_from, len(PIPELINE)):
            notebooks_to_run.append((f'B{i}', PIPELINE[i]))
        if args.all:
            for i, entry in enumerate(ANALYTICS):
                notebooks_to_run.append((f'A{i}', entry))

    if not notebooks_to_run:
        print("No notebooks selected.")
        return 1

    # Show plan
    print("\nBIRD Museum Pipeline Runner")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print("\nNotebooks to run:")
    for tag, (name, desc) in notebooks_to_run:
        print(f"  [{tag}] {name}")
        print(f"       {desc}")

    if args.dry_run:
        print("\n[DRY RUN] No notebooks executed.")
        return 0

    # Run
    print(f"\nStarting pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    succeeded = []
    failed = []
    timings = {}

    for tag, (name, desc) in notebooks_to_run:
        notebook_path = script_dir / name

        if not notebook_path.exists():
            print(f"\nWARNING: Notebook not found: {name}")
            failed.append((tag, name, "File not found"))
            if not args.no_stop_on_error:
                print("Pipeline stopped. Use --no-stop-on-error to continue.")
                break
            continue

        success, elapsed = run_notebook(notebook_path, output_dir)
        timings[name] = elapsed

        if success:
            succeeded.append((tag, name))
        else:
            failed.append((tag, name, "Execution error"))
            if not args.no_stop_on_error:
                print(f"\nPipeline stopped due to error in {name}")
                print(f"Use --from {tag[1:]} to resume after fixing.")
                break

    # Summary
    total_time = sum(timings.values())
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s")
    print(f"\nSucceeded: {len(succeeded)}")
    for tag, name in succeeded:
        t = timings.get(name, 0)
        print(f"  [{tag}] {name} ({t:.1f}s)")

    if failed:
        print(f"\nFailed: {len(failed)}")
        for tag, name, reason in failed:
            t = timings.get(name, 0)
            print(f"  [{tag}] {name} - {reason} ({t:.1f}s)")
        return 1

    print("\nAll notebooks completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
