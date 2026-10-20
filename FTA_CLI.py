# -*- coding: utf-8 -*-
"""Firebrand Thermal Analysis — headless command line, for HPC (Spartan).

Detects and tracks firebrands in a Level 1.1 NetCDF thermal recording using
temperature, and writes counts, trajectories, velocities and per-firebrand
temperature histories. The desktop app is FirebrandThermalAnalysis.py.

    python FTA_CLI.py run S9.nc --out results/S9 --label fibrous_bark/E_radiata/High/50kW/S9
    python FTA_CLI.py run S9.nc --out test --stop 3000 --qa --set persist_frac=0.2
    python FTA_CLI.py retrack results/ --set min_contrast_c=100   # re-track every recording, no NetCDF needed
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


def _parse_overrides(pairs) -> dict:
    """--set NAME=VALUE pairs, typed by each field's declared type, checked before any data is read.

    None (or an empty value) is accepted only for fields declared Optional; a typo or an
    unset shell variable would otherwise crash the run after the hours-long detection pass.
    """
    fields = {f.name: f for f in dataclasses.fields(Params)}
    out = {}
    for item in pairs or []:
        key, eq, val = item.partition("=")
        key, val = key.strip(), val.strip()
        if not eq or key not in fields:
            sys.exit(f"bad --set {item!r}: expected NAME=VALUE with NAME one of {', '.join(sorted(fields))}")
        ftype = str(fields[key].type)
        optional = "Optional" in ftype or "None" in ftype
        if val.lower() in ("none", ""):
            if not optional:
                sys.exit(f"bad --set {item!r}: {key} needs a value ({ftype})")
            out[key] = None
            continue
        try:
            if "bool" in ftype:
                if val.lower() not in ("1", "0", "true", "false", "yes", "no"):
                    raise ValueError(val)
                out[key] = val.lower() in ("1", "true", "yes")
            elif "int" in ftype:
                f = float(val)
                if not f.is_integer():
                    raise ValueError(val)
                out[key] = int(f)
            else:
                out[key] = float(val)
        except ValueError:
            sys.exit(f"bad --set {item!r}: {key} must be {ftype}")
    return out


def cmd_run(a) -> int:
    p = Params(**_parse_overrides(a.set))
    res = run_recording(a.nc, a.out, p, label=a.label or "", qa=a.qa, start=a.start, stop=a.stop)
    fb = sum(1 for t in res["tracks"] if t["is_firebrand"])
    print(f"done: {fb} firebrands, {len(res['tracks'])} confirmed tracks, "
          f"{len(res['detections'])} detections, {res['seconds']:.0f} s -> {a.out}")
    return 0


def cmd_params(a) -> int:
    """Print the resolved parameters and their fingerprints (used by the Slurm array's skip check)."""
    from fta_track.export import DETECTION_PARAMS, params_hash
    d = Params(**_parse_overrides(a.set)).to_dict()
    if a.hash:
        print(params_hash(d))
    elif a.detection_hash:
        print(params_hash(d, DETECTION_PARAMS))
    else:
        print(json.dumps({"params": d, "params_hash": params_hash(d),
                          "detection_params_hash": params_hash(d, DETECTION_PARAMS)}, indent=2))
    return 0


def cmd_retrack(a) -> int:
    from fta_track.export import resolve_retrack_params, retrack
    overrides = _parse_overrides(a.set)
    roots = [root for root, _, files in sorted(os.walk(a.results))
             if "summary.json" in files and "detections.csv.gz" in files]
    if not roots:
        sys.exit(f"no results with detections.csv.gz under {a.results}")
    # check every recording before rewriting any, so a refusal cannot leave the tree half re-tracked
    for root in roots:
        try:
            resolve_retrack_params(root, overrides, a.from_defaults)
        except ValueError as e:
            sys.exit(str(e))
    for root in roots:
        res = retrack(root, overrides, a.from_defaults)
        fb = sum(1 for t in res["tracks"] if t["is_firebrand"])
        print(f"{os.path.relpath(root, a.results)}: {fb} firebrands, {len(res['tracks'])} confirmed tracks")
    print(f"re-tracked {len(roots)} recording(s)")
    return 0


def _factors(label: str) -> dict:
    """Factor levels from the run's Level 2 path."""
    parts = label.split("/")
    tree = parts[0]
    row = {"fuel_type": tree}
    if tree == "branchlet":
        # branchlet/<species>/<HRR>/<structure>/[<class>]/<run>
        row.update(zip(["species", "hrr", "structure"], parts[1:4]))
        if len(parts) == 6:
            row["class"], row["run"] = parts[4], parts[5]
        elif len(parts) == 5:
            row["run"] = parts[4]
    elif tree == "fibrous_bark":
        # fibrous_bark/<species>/<rating>/<HRR>/<section>
        row.update(zip(["species", "hazard_rating", "hrr", "section"], parts[1:5]))
    elif tree == "candlebark":
        # candlebark/<species>/<HRR>/<shape>/<length>/<run>
        row.update(zip(["species", "hrr", "shape", "length", "run"], parts[1:6]))
    elif len(parts) > 1:
        row["run"] = parts[-1]
    return row


def cmd_aggregate(a) -> int:
    rows = []
    for root, _, files in os.walk(a.results):
        if "summary.json" not in files:
            continue
        with open(os.path.join(root, "summary.json")) as fh:
            s = json.load(fh)
        label = s.get("label") or os.path.relpath(root, a.results)
        row = {"recording": label, **_factors(label)}
        f0, f1 = (s.get("frame_range") or [0, s["n_frames"]])[:2]
        row.update({
            "firebrand_count": s["firebrand_count"],
            "confirmed_tracks": s["confirmed_tracks"],
            "ignition_time_s": s["ignition_time_s"],
            "duration_s": (f1 - f0) / s["fps"] if s.get("fps") else None,
            "fps": s["fps"],
            "m_per_px": s["m_per_px"],
            "params_hash": s.get("params_hash"),
            "detection_params_hash": s.get("detection_params_hash"),
            "detection_code_version": s.get("detection_code_version"),
            "tracking_code_version": s.get("tracking_code_version"),
            "retracked": s.get("retracked", False),
        })
        rows.append(row)
    if not rows:
        sys.exit(f"no summary.json found under {a.results}")
    cols = ["recording", "fuel_type", "species", "hazard_rating", "hrr", "structure", "class", "section",
            "shape", "length", "run", "firebrand_count", "confirmed_tracks", "ignition_time_s",
            "duration_s", "fps", "m_per_px", "params_hash", "detection_params_hash",
            "detection_code_version", "tracking_code_version", "retracked"]
    rows.sort(key=lambda r: r["recording"])
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} recordings -> {a.out}")
    hashes = sorted({str(r["params_hash"]) for r in rows})
    if len(hashes) > 1:
        print(f"WARNING: the table mixes {len(hashes)} parameter sets ({', '.join(hashes)}); "
              f"counts from different sets are not comparable", file=sys.stderr)
    return 0


def cmd_review_make(a) -> int:
    from fta_track.review import make_batch
    n = make_batch(a.results, a.nc, a.batch, a.label, n_firebrand=a.n_firebrand,
                   n_rejected=a.n_rejected, seed=a.seed)
    print(f"{n} clips from {a.label} -> {a.batch}/index.html")
    return 0


def cmd_review_merge(a) -> int:
    from fta_track.review import merge_batches
    n = merge_batches(a.out, a.parts)
    print(f"{n} clips from {len(a.parts)} batch(es) -> {a.out}/index.html")
    return 0


def cmd_review_score(a) -> int:
    from fta_track.review import score
    try:
        out = score(a.batch, a.verdicts)
    except ValueError as e:
        sys.exit(str(e))
    p = out["raw_sample"]["precision_of_counted_firebrands"]
    r = out["raw_sample"]["real_firebrands_among_rejected"]
    w = out["weighted"]
    print(f"answered {out['answered']}, unanswered {out['unanswered']}, "
          f"unknown ids {len(out['unknown_ids'])}, invalid values {len(out['invalid_values'])}")
    print(f"sample: counted firebrands judged real {p['yes']}/{p['yes'] + p['no']} (unsure {p['unsure']}); "
          f"rejected tracks judged real {r['yes']}/{r['yes'] + r['no']} (unsure {r['unsure']})")
    print(f"weighted: precision {w['estimated_precision']}, recall among confirmed tracks "
          f"{w['estimated_recall_among_confirmed_tracks']}")
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

    pp = sub.add_parser("params", help="print the resolved parameters and their fingerprints")
    pp.add_argument("--set", action="append", metavar="NAME=VALUE")
    pp.add_argument("--hash", action="store_true", help="print only the fingerprint of all parameters")
    pp.add_argument("--detection-hash", action="store_true", help="print only the detection-stage fingerprint")
    pp.set_defaults(func=cmd_params)

    t = sub.add_parser("retrack", help="re-run tracking and classification from saved detections (no NetCDF)")
    t.add_argument("results", help="one recording's results directory, or a root searched recursively")
    t.add_argument("--set", action="append", metavar="NAME=VALUE",
                   help="override a tracking/classification parameter (repeatable); "
                        "everything else is kept as the run recorded it")
    t.add_argument("--from-defaults", action="store_true",
                   help="start tracking/classification from the current code's defaults instead of the run's "
                        "values (detection parameters always come from the run)")
    t.set_defaults(func=cmd_retrack)

    g = sub.add_parser("aggregate", help="collect every summary.json into one counts table")
    g.add_argument("results", help="results root (searched recursively)")
    g.add_argument("--out", required=True, help="CSV to write")
    g.set_defaults(func=cmd_aggregate)

    m = sub.add_parser("review-make", help="render a recording's tracks into a blind yes/no review batch")
    m.add_argument("results", help="results directory of one recording (from `run`)")
    m.add_argument("nc", help="the same recording's NetCDF, to render the clips")
    m.add_argument("--batch", required=True, help="batch directory; clips are appended")
    m.add_argument("--label", required=True, help="Level 2 path of the run")
    m.add_argument("--n-firebrand", type=int, default=15, help="tracks counted as firebrands to sample")
    m.add_argument("--n-rejected", type=int, default=12,
                   help="rejected tracks to sample, spread over rejection reasons (3+ detections, so too_short is included)")
    m.add_argument("--seed", type=int, default=0)
    m.set_defaults(func=cmd_review_make)

    mg = sub.add_parser("review-merge", help="combine per-recording review batches into one page")
    mg.add_argument("out", help="merged batch directory")
    mg.add_argument("parts", nargs="+", help="batch directories to merge")
    mg.set_defaults(func=cmd_review_merge)

    sc = sub.add_parser("review-score", help="score a review batch against the pipeline's decisions")
    sc.add_argument("batch", help="batch directory")
    sc.add_argument("verdicts", help="verdicts.json downloaded from the review page")
    sc.set_defaults(func=cmd_review_score)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
