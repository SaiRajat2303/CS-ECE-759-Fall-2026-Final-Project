#!/usr/bin/env python3
"""Group timing rows and report medians plus spread. Stdlib only."""
import csv
import statistics
import sys
from collections import defaultdict

SCHEMA = "3"
KEY = ("gpu_name", "scene", "n", "mem_mode", "layout", "block", "timing_mode")
STAGES = ("ms_h2d", "ms_bounds", "ms_morton", "ms_sort", "ms_hierarchy", "ms_refit")


def complete(r):
    try:
        float(r.get("ms_total") or "")
        int(r.get("warmup_iters") or "")
    except ValueError:
        return False
    return r.get("schema") == SCHEMA


def load(paths):
    rows, skipped = [], 0
    for p in paths:
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                if complete(r):
                    rows.append(r)
                else:
                    skipped += 1
    return rows, skipped


def stat(vals):
    med = statistics.median(vals)
    spread = 100.0 * (max(vals) - min(vals)) / med if med else 0.0
    return med, spread


def main(paths):
    rows, skipped = load(paths)
    if skipped:
        print(f"skipped {skipped} incomplete or old-schema lines")
    if not rows:
        print("no rows")
        return 1

    gpus = sorted({r["gpu_name"] for r in rows})
    if len(gpus) > 1:
        print(f"WARNING: rows from {len(gpus)} GPUs: {gpus}")
        print("Timings from different GPUs are not comparable.\n")

    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in KEY)].append(r)

    hdr = f"{'scene':>14} {'n':>9} {'lay':>4} {'mode':>5} {'cnt':>4} {'total':>9} {'sprd%':>6}"
    for s in STAGES:
        hdr += f" {s[3:]:>9}"
    hdr += f" {'h2d%':>6} {'build%':>6}"
    print(hdr)

    order = lambda k: (k[0], k[1], int(k[2]), k[4], k[6])
    for key in sorted(groups, key=order):
        g = groups[key]
        med, spread = stat([float(r["ms_total"]) for r in g])
        _, scene, n, _, layout, _, mode = key
        line = f"{scene:>14} {int(n):>9} {layout:>4} {mode:>5} {len(g):>4} {med:>9.3f} {spread:>6.1f}"
        if mode == "chain":
            meds = {s: stat([float(r[s]) for r in g])[0] for s in STAGES}
            for s in STAGES:
                line += f" {meds[s]:>9.3f}"
            build = meds["ms_hierarchy"] + meds["ms_refit"]
            line += f" {100.0 * meds['ms_h2d'] / med:>5.1f}% {100.0 * build / med:>5.1f}%"
        print(line)
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(f"usage: {sys.argv[0]} <csv> [csv ...]")
        sys.exit(2)
    sys.exit(main(args))
