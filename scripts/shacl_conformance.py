#!/usr/bin/env python3
"""Per-stage SHACL conformance over a live KG (evaluation axis B).

The pipeline already carries 37 shapes across 7 transformation stages and
validates against them during ingestion, but nothing reports the result as a
whole. This runs every stage against a dataset and prints the table.

Conformance is the KG-engineering counterpart to a test suite: it states that
each transformation produced a graph satisfying its own contract. Running it
across instantiations also shows which stages a corpus reaches at all -- a
stage with no data is a different fact from a stage that fails.

    python scripts/shacl_conformance.py                       # default KG
    python scripts/shacl_conformance.py --kg osm-rome --kg bird
    python scripts/shacl_conformance.py --self-check
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))

STAGES = [
    ("stage-02-base-rdf",       "base RDF from the RML mapping"),
    ("stage-03-enrichment",     "sequences, temporal context, interactions"),
    ("stage-05-poi-scoring",    "dominant and significant POIs"),
    ("stage-06-navigation",     "navigation patterns"),
    ("stage-07-classification", "trajectory purposes"),
    ("stage-08-user-profiles",  "user profiles and preferences"),
    ("stage-S1-reasoning",      "materialised reasoning layer"),
]


def shape_count(stage):
    p = ROOT / "shacl" / "stages" / f"{stage}.ttl"
    if not p.exists():
        return 0
    t = p.read_text(encoding="utf-8")
    return t.count("sh:NodeShape") + t.count("sh:PropertyShape")


def run_stage(stage):
    """Returns (n_triples, conforms, violations, warnings) or an error string."""
    from helpers import shacl_validator as sv
    try:
        g = sv.construct_validation_graph(stage, verbose=False)
    except Exception as exc:
        return f"context error: {type(exc).__name__}"
    if len(g) == 0:
        return (0, None, 0, 0)          # stage not reached by this corpus
    try:
        # halt_on_violations=False: we are reporting, not gating an import
        r = sv.validate_before_import(g, stage, halt_on_violations=False)
    except Exception as exc:
        return f"validation error: {type(exc).__name__}"
    return (len(g), r["conforms"], r["violations"], r["warnings"])


def self_check():
    assert len(STAGES) == 7, STAGES
    assert all(shape_count(s) > 0 for s, _ in STAGES), \
        [(s, shape_count(s)) for s, _ in STAGES]
    total = sum(shape_count(s) for s, _ in STAGES)
    assert total == 37, f"expected 37 shapes, found {total}"
    print(f"shacl_conformance self-check passed ({total} shapes, 7 stages)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", action="append", default=None)
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        self_check(); return

    import re
    cfg = ROOT / "notebooks" / "helpers" / "fuseki_config.py"
    original = cfg.read_text()
    kgs = a.kg or [re.search(r'^graph_name = "([^"]*)"', original, re.M).group(1)]

    try:
        for kg in kgs:
            cfg.write_text(re.sub(r'^graph_name = "[^"]*"',
                                  f'graph_name = "{kg}"', original, count=1, flags=re.M))
            for mod in [m for m in list(sys.modules) if "helpers" in m]:
                del sys.modules[mod]
            print(f"\n=== {kg} ===")
            print(f"{'stage':26} {'shapes':>6} {'triples':>9} {'result':>12} {'viol':>5} {'warn':>5}")
            for stage, _desc in STAGES:
                out = run_stage(stage)
                n_sh = shape_count(stage)
                if isinstance(out, str):
                    print(f"{stage:26} {n_sh:>6} {'':>9} {out:>12}")
                    continue
                n, conforms, v, w = out
                if conforms is None:
                    verdict = "not reached"
                elif v > 0:
                    verdict = "VIOLATIONS"
                elif w > 0:
                    verdict = "warn only"      # conforms=False but nothing gating
                else:
                    verdict = "CONFORMS"
                print(f"{stage:26} {n_sh:>6} {n:>9,} {verdict:>12} {v:>5} {w:>5}")
        print("\nviolations are the gating signal; sh:Warning results are advisory.")
        print("Note: the validation graph is a PROJECTION built from per-stage")
        print("CONSTRUCTs, so a shape whose context is not projected reports a")
        print("warning for every focus node. Verified example: 'Trajectory should")
        print("have at least one position' warns 1,258 times on Rome, while the")
        print("full KG has 0 trajectories without a position. Such warnings")
        print("characterise the projection, not the data.")
    finally:
        cfg.write_text(original)
        restored = re.search(r'^graph_name = "([^"]*)"', original, re.M).group(1)
        print(f"\nfuseki_config restored to {restored}")


if __name__ == "__main__":
    main()
