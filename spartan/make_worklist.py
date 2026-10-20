"""Build worklist.tsv for fta_track_array.sbatch from the re-encode list.

    python make_worklist.py ~/restructure/reencode_all.tsv \
        [--listing mediaflux_l11_thermal_listing.txt] > worklist.tsv

Input columns: NetCDF path under level1.1/thermal_compressed_threshold299, the
run's Level 2 path, floor, ceiling, source. Output: the first two, both in their
*current* names. The re-encode list predates two candlebark renames:

- run labels there read `curve/60cm/1`, released as
  `cylindrical/60cm/cylindrical1_<HRR>_60cm` (and `flat/20cm/1` as
  `flat/20cm/flat1_<HRR>_20cm`);
- NetCDF files there read `Curve_1_100_kW_60_cm.nc`, renamed on MediaFlux to
  `Curve1_100kW_60cm.nc`.

Both mappings leave names that are already current unchanged, so the script can
be run on its own output. With --listing (one `rclone lsf -R --files-only` of the
Level 1.1 thermal tree, prefixed `level1/thermal/`), every NetCDF path must be in
it and every listed file must be used, or the script fails.
"""

import argparse
import re
import sys

FAMILIES = ("branchlet", "fibrous_bark", "candlebark", "preliminary_branchlet_test")
_OLD_CANDLE_NC = re.compile(r"^(level1/thermal/candlebark/)(Curve|Flat)_(\d+)_(\d+)_kW_(\d+)_cm\.nc$")


def released_label(label: str) -> str:
    parts = label.split("/")
    if (parts[0] == "candlebark" and len(parts) == 6 and parts[5].isdigit()
            and parts[3] in ("curve", "flat")):
        _, species, hrr, shape, length, rep = parts
        shape = "cylindrical" if shape == "curve" else shape
        return "/".join(["candlebark", species, hrr, shape, length, f"{shape}{rep}_{hrr}_{length}"])
    return label


def current_nc(path: str) -> str:
    m = _OLD_CANDLE_NC.match(path)
    if not m:
        return path
    prefix, shape, rep, hrr, length = m.groups()
    return f"{prefix}{shape}{rep}_{hrr}kW_{length}cm.nc"


def build(lines, listing=None):
    out, seen_label, seen_nc = [], set(), set()
    for n, raw in enumerate(lines, 1):
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        cols = [c.strip() for c in line.split("\t")]
        if len(cols) < 2 or not cols[0] or not cols[1]:
            raise ValueError(f"line {n}: expected <NetCDF path> TAB <run path>: {line!r}")
        nc, label = current_nc(cols[0]), released_label(cols[1])
        if not (nc.startswith("level1/thermal/") and nc.endswith(".nc")) or ".." in nc:
            raise ValueError(f"line {n}: not a Level 1.1 NetCDF path (a header?): {cols[0]!r}")
        if label.split("/")[0] not in FAMILIES or ".." in label or label.endswith("/") or "//" in label:
            raise ValueError(f"line {n}: not a run path: {cols[1]!r}")
        if label in seen_label or nc in seen_nc:
            raise ValueError(f"line {n}: duplicate run or NetCDF: {label} / {nc}")
        seen_label.add(label)
        seen_nc.add(nc)
        out.append((nc, label))
    if listing is not None:
        missing = [nc for nc, _ in out if nc not in listing]
        unused = sorted(listing - seen_nc)
        if missing or unused:
            raise ValueError(f"{len(missing)} NetCDF path(s) not on MediaFlux {missing[:5]}; "
                             f"{len(unused)} listed file(s) not in the work list {unused[:5]}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("reencode_list")
    ap.add_argument("--listing", help="file of NetCDF paths on MediaFlux, one per line, prefixed level1/thermal/")
    a = ap.parse_args(argv)
    listing = None
    if a.listing:
        with open(a.listing) as fh:
            listing = {l.strip() for l in fh if l.strip()}
    try:
        with open(a.reencode_list) as fh:
            rows = build(fh, listing)
    except ValueError as e:
        sys.exit(f"make_worklist: {e}")
    for nc, label in rows:
        print(f"{nc}\t{label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
