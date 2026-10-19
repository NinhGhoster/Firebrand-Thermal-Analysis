# -*- coding: utf-8 -*-
"""Firebrand Thermal Analysis — headless command line, for HPC (Spartan).

Detects and tracks firebrands in a Level 1.1 NetCDF thermal recording using
temperature, and writes counts, trajectories, velocities and per-firebrand
temperature histories. The desktop app is FirebrandThermalAnalysis.py.

    python FTA_CLI.py run S9.nc --out results/S9 --label fibrous_bark/E_radiata/High/50kW/S9
    python FTA_CLI.py run S9.nc --out test --stop 3000 --qa --set persist_frac=0.2
    python FTA_CLI.py aggregate results/ --out firebrand_counts.csv
    python FTA_CLI.py review-make results/S9 S9.nc --batch review/b1 --label fibrous_bark/E_radiata/High/50kW/S9
    python FTA_CLI.py review-score review/b1 verdicts.json

No GUI, no FLIR SDK: only numpy, scipy, opencv-python-headless and netCDF4.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import sys

from fta_track import Params, run_recording


def _parse_sets(pairs):
    p = Params()
    fields = {f.name: f for f in dataclasses.fields(Params)}
    for item in pairs or []:
        key, _, val = item.partition("=")
        key = key.strip()
        if key not in fields:
            sys.exit(f"unknown parameter {key!r}; known: {', '.join(sorted(fields))}")
        cur = getattr(p, key)
        if val.lower() in ("none", ""):
            setattr(p, key, None)
        elif isinstance(cur, bool):
            setattr(p, key, val.lower() in ("1", "true", "yes"))
        elif isinstance(cur, int) and not isinstance(cur, bool):
            setattr(p, key, int(val))
        else:
            setattr(p, key, float(val))
    return p


def cmd_run(a) -> int:
    p = _parse_sets(a.set)
    res = run_recording(a.nc, a.out, p, label=a.label or "", qa=a.qa, start=a.start, stop=a.stop)
    fb = sum(1 for t in res["tracks"] if t["is_firebrand"])
    print(f"done: {fb} firebrands, {len(res['tracks'])} confirmed tracks, "
          f"{len(res['detections'])} detections, {res['seconds']:.0f} s -> {a.out}")
    return 0


# Factor levels by position in the Level 2 path (the run label).
_FACTORS = {
    "branchlet": ["fuel_type", "species", "hrr", "structure"],
    "fibrous_bark": ["fuel_type", "species", "hazard_rating", "hrr", "section"],
    "candlebark": ["fuel_type", "species", "hrr", "shape", "length"],
    "preliminary_branchlet_test": ["fuel_type"],
}


def cmd_aggregate(a) -> int:
    rows = []
    for root, _, files in os.walk(a.results):
        if "summary.json" not in files:
            continue
        with open(os.path.join(root, "summary.json")) as fh:
            s = json.load(fh)
        label = s.get("label") or os.path.relpath(root, a.results)
        parts = label.split("/")
        row = {"recording": label}
        for name, val in zip(_FACTORS.get(parts[0], ["fuel_type"]), parts):
            row[name] = val
        row.update({
            "firebrand_count": s["firebrand_count"],
            "confirmed_tracks": s["confirmed_tracks"],
            "ignition_time_s": s["ignition_time_s"],
            "duration_s": s["n_frames"] / s["fps"] if s.get("fps") else None,
            "fps": s["fps"],
            "m_per_px": s["m_per_px"],
            "code_version": s["code_version"],
        })
        rows.append(row)
    if not rows:
        sys.exit(f"no summary.json found under {a.results}")
    cols = ["recording", "fuel_type", "species", "hazard_rating", "hrr", "structure", "section",
            "shape", "length", "firebrand_count", "confirmed_tracks", "ignition_time_s",
            "duration_s", "fps", "m_per_px", "code_version"]
    rows.sort(key=lambda r: r["recording"])
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} recordings -> {a.out}")
    return 0


def cmd_review_make(a) -> int:
    from fta_track.review import make_batch
    n = make_batch(a.results, a.nc, a.batch, a.label, n_firebrand=a.n_firebrand,
                   n_rejected=a.n_rejected, seed=a.seed)
    print(f"{n} clips from {a.label} -> {a.batch}/index.html")
    return 0


def cmd_review_score(a) -> int:
    from fta_track.review import score
    out = score(a.batch, a.verdicts)
    p, r = out["precision_of_counted_firebrands"], out["real_firebrands_among_rejected"]
    print(f"counted firebrands judged real: {p['yes']}/{p['yes'] + p['no']} (unsure {p['unsure']})")
    print(f"rejected tracks judged real:    {r['yes']}/{r['yes'] + r['no']} (unsure {r['unsure']})")
    print(f"-> {a.batch}/scores.json")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="FTA_CLI.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="detect and track firebrands in one NetCDF recording")
    r.add_argument("nc", help="Level 1.1 NetCDF file (temperature in degC)")
    r.add_argument("--out", required=True, help="output directory for this recording")
    r.add_argument("--label", help="Level 2 path of the run, e.g. branchlet/Acacia/150kW/Twigs/51")
    r.add_argument("--qa", action="store_true", help="also write qa.mp4, a half-resolution overlay video")
    r.add_argument("--start", type=int, default=0, help="first frame (default 0)")
    r.add_argument("--stop", type=int, default=None, help="stop before this frame (default: end)")
    r.add_argument("--set", action="append", metavar="NAME=VALUE",
                   help="override a parameter (repeatable); see fta_track/params.py")
    r.set_defaults(func=cmd_run)

    g = sub.add_parser("aggregate", help="collect every summary.json into one counts table")
    g.add_argument("results", help="results root (searched recursively)")
    g.add_argument("--out", required=True, help="CSV to write")
    g.set_defaults(func=cmd_aggregate)

    m = sub.add_parser("review-make", help="render a recording's tracks into a blind yes/no review batch")
    m.add_argument("results", help="results directory of one recording (from `run`)")
    m.add_argument("nc", help="the same recording's NetCDF, to render the clips")
    m.add_argument("--batch", required=True, help="batch directory; clips are appended")
    m.add_argument("--label", required=True, help="Level 2 path of the run")
    m.add_argument("--n-firebrand", type=int, default=20, help="tracks counted as firebrands to sample")
    m.add_argument("--n-rejected", type=int, default=15, help="rejected tracks to sample")
    m.add_argument("--seed", type=int, default=0)
    m.set_defaults(func=cmd_review_make)

    sc = sub.add_parser("review-score", help="score a review batch against the pipeline's decisions")
    sc.add_argument("batch", help="batch directory")
    sc.add_argument("verdicts", help="verdicts.json downloaded from the review page")
    sc.set_defaults(func=cmd_review_score)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
