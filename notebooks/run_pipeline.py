#!/usr/bin/env python3
"""
Trajecto Pipeline Runner

Runs all notebooks in sequence using papermill.
Outputs are saved to the 'output/' directory with execution logs.

Usage:
    python run_pipeline.py                          # Run all notebooks
    python run_pipeline.py --from 3                 # Start from notebook 3
    python run_pipeline.py --only 0 1 2             # Run only specific notebooks
    python run_pipeline.py --dry-run                # Show what would run
    python run_pipeline.py --kg my-dataset          # Use a custom Fuseki KG name
    python run_pipeline.py --kg my-dataset --from 2 # Custom KG + start from 2
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    import papermill as pm
except ImportError:
    print("papermill not installed. Installing...")
    os.system(f"{sys.executable} -m pip install papermill")
    import papermill as pm


# Notebook pipeline in execution order
PIPELINE = [
    # --- Core pipeline (0-9): data loading → KG construction ---
    ('0-data-loaders.ipynb', 'Load and standardize trajectory data'),
    ('1-trajectory-preprocessing.ipynb', 'Process trajectories, detect stops, find POIs'),
    ('2-update_rules_and_generate_rdf.ipynb', 'Generate RDF, classify POIs, upload to Fuseki'),
    ('3-enrichment_sparql_queries.ipynb', 'Enrich with sequences, temporal context, interactions, Wikidata'),
    ('4-trajectory_holiday_annotation.ipynb', 'Add holiday context annotations'),
    ('5-dominant_and_significant_pois.ipynb', 'Score and classify dominant/significant POIs'),
    ('6-navigation_styles.ipynb', 'Classify navigation patterns (casual, exploratory, targeted, immersive)'),
    ('7-trajectory_classification.ipynb', 'Assign trajectory purposes'),
    ('8-setup_existing_user.ipynb', 'Create user profiles with initial preferences'),
    ('9-snapshot_updates.ipynb', 'EWMA preference updates per trajectory'),

    # --- Feature extraction (10) ---
    ('10-semantic-features-extraction.ipynb', 'Extract 67-dim semantic feature vectors'),

    # --- KG exploration (11) ---
    ('11-knowledge-graph-explorer.ipynb', 'Explore and verify KG structure'),

    # --- Semantic reasoning (12-13): materialization + OWL classification ---
    ('12-trajectory-profiling.ipynb', 'SPARQL materialization: co-visitation, flow, engagement, similarity'),
    ('13-owl-classification.ipynb', 'OWL 2 RL classification and ontological similarity',
     '../scripts/run_owl_classification.py'),

    # --- Analytics & recommendations (14-18) ---
    ('14-graph-analytics.ipynb', 'Graph-theoretic structure + KG-native multi-hop analytics'),
    ('15-trajectory-profile-card.ipynb', 'Per-trajectory tag-stack product surface + visited-POI map'),
    ('16-unified-recommendations.ipynb', 'Nine-signal POI recommendation engine (consumer demo)'),
    ('17-baseline-comparison.ipynb', 'Headline evaluation: baselines, proxy metrics, ceiling-normalized recall'),
    ('18-unified-signal-analysis.ipynb', 'Diagnostics: weight sensitivity, signal contribution, ablation'),
]

FUSEKI_CONFIG_PATH = Path(__file__).parent / 'helpers' / 'fuseki_config.py'


def set_kg_name(new_name: str) -> None:
    """Update graph_name in fuseki_config.py to point to a different KG."""
    config_path = FUSEKI_CONFIG_PATH
    content = config_path.read_text()

    # Replace the graph_name assignment
    updated = re.sub(
        r'^graph_name\s*=\s*"[^"]*"',
        f'graph_name = "{new_name}"',
        content,
        count=1,
        flags=re.MULTILINE,
    )

    if updated == content:
        print(f"WARNING: Could not find graph_name assignment in {config_path}")
        return

    config_path.write_text(updated)
    print(f"Updated fuseki_config.py: graph_name = \"{new_name}\"")


def run_notebook(notebook_path: Path, output_dir: Path, parameters: dict = None):
    """Run a single notebook and save output."""
    notebook_name = notebook_path.stem
    output_path = output_dir / f"{notebook_name}_executed.ipynb"

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
        return True

    except pm.PapermillExecutionError as e:
        elapsed = datetime.now() - start_time
        print(f"\nFAILED after {elapsed.total_seconds():.1f}s")
        print(f"Error in cell {e.exec_count}:")
        print(f"  {e.ename}: {e.evalue}")
        return False


def run_script(script_path: Path):
    """Run a Python script as a subprocess; inherit stdio."""
    print(f"\n{'='*60}")
    print(f"Running script: {script_path.name}")
    print(f"{'='*60}")
    start_time = datetime.now()
    result = subprocess.run([sys.executable, str(script_path)])
    elapsed = datetime.now() - start_time
    if result.returncode == 0:
        print(f"Completed in {elapsed.total_seconds():.1f}s")
        return True
    print(f"\nFAILED after {elapsed.total_seconds():.1f}s (exit {result.returncode})")
    return False


def main():
    parser = argparse.ArgumentParser(description='Run Trajecto notebook pipeline')
    parser.add_argument('--from', dest='start_from', type=int, default=0,
                        help='Start from notebook index (0-indexed)')
    parser.add_argument('--only', type=int, nargs='+',
                        help='Run only specific notebook indices')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would run without executing')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory for executed notebooks (default: output)')
    parser.add_argument('--kg', type=str, default=None,
                        help='Fuseki KG dataset name (updates fuseki_config.py)')
    parser.add_argument('--no-stop-on-error', action='store_true',
                        help='Continue running after a notebook fails')
    args = parser.parse_args()

    # Update KG name if specified
    if args.kg:
        set_kg_name(args.kg)

    # Setup paths
    script_dir = Path(__file__).parent
    output_dir = script_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    # Determine which notebooks to run
    if args.only:
        indices = args.only
    else:
        indices = range(args.start_from, len(PIPELINE))

    notebooks_to_run = [(i, PIPELINE[i]) for i in indices if i < len(PIPELINE)]

    # Show plan
    print("\nTrajecto Pipeline Runner")
    print("=" * 60)
    if args.kg:
        print(f"KG dataset: {args.kg}")
    print(f"Output directory: {output_dir}")
    print("\nNotebooks to run:")
    for i, entry in notebooks_to_run:
        name, desc = entry[0], entry[1]
        prereq = entry[2] if len(entry) > 2 else None
        print(f"  [{i:2d}] {name}")
        print(f"       {desc}")
        if prereq:
            print(f"       prereq: {prereq}")

    if args.dry_run:
        print("\n[DRY RUN] No notebooks executed.")
        return 0

    # Run notebooks
    print(f"\nStarting pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    failed = []
    succeeded = []

    for i, entry in notebooks_to_run:
        name, desc = entry[0], entry[1]
        prereq = entry[2] if len(entry) > 2 else None
        step_path = (script_dir / name).resolve()

        if not step_path.exists():
            print(f"\nWARNING: Step not found: {name}")
            failed.append((i, name, "File not found"))
            if not args.no_stop_on_error:
                print("Pipeline stopped. Use --no-stop-on-error to continue.")
                break
            continue

        if prereq:
            prereq_path = (script_dir / prereq).resolve()
            if not prereq_path.exists():
                print(f"\nWARNING: Prereq not found: {prereq}")
                failed.append((i, name, f"Prereq missing: {prereq}"))
                if not args.no_stop_on_error:
                    break
                continue
            if not run_script(prereq_path):
                failed.append((i, name, f"Prereq failed: {prereq}"))
                if not args.no_stop_on_error:
                    print(f"\nPipeline stopped due to prereq failure for {name}")
                    break
                continue

        if step_path.suffix == '.py':
            success = run_script(step_path)
        else:
            success = run_notebook(step_path, output_dir)

        if success:
            succeeded.append((i, name))
        else:
            failed.append((i, name, "Execution error"))
            if not args.no_stop_on_error:
                print(f"\nPipeline stopped due to error in {name}")
                print("Use --no-stop-on-error to continue past failures.")
                break

    # Summary
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Succeeded: {len(succeeded)}")
    for i, name in succeeded:
        print(f"  [{i:2d}] {name}")

    if failed:
        print(f"\nFailed: {len(failed)}")
        for i, name, reason in failed:
            print(f"  [{i:2d}] {name} - {reason}")
        return 1

    print(f"\nAll {len(succeeded)} notebooks completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
