"""Build worklist.tsv for fta_track_array.sbatch from the re-encode list.

    python make_worklist.py ~/restructure/reencode_all.tsv > worklist.tsv

Input columns: NetCDF path under level1.1/thermal_compressed_threshold299, the
run's Level 2 path, floor, ceiling, source. Output: the first two, with the
Level 2 path in the *released* names. The re-encode list predates one rename:
candlebark runs there read `curve/60cm/1`, released as
`cylindrical/60cm/cylindrical1_<HRR>_60cm` (and `flat/20cm/1` as
`flat/20cm/flat1_<HRR>_20cm`). Every other path is already as released.
"""

import sys


def released_label(label: str) -> str:
    parts = label.split("/")
    if parts[0] != "candlebark" or len(parts) != 6:
        return label
    _, species, hrr, shape, length, rep = parts
    shape = "cylindrical" if shape == "curve" else shape
    return "/".join(["candlebark", species, hrr, shape, length, f"{shape}{rep}_{hrr}_{length}"])


def main(path: str) -> None:
    seen = set()
    with open(path) as fh:
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2:
                continue
            label = released_label(cols[1])
            if label in seen:
                sys.exit(f"duplicate run label: {label}")
            seen.add(label)
            print(f"{cols[0]}\t{label}")


if __name__ == "__main__":
    main(sys.argv[1])
