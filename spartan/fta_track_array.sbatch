#!/bin/bash
#SBATCH --job-name=fta-track
#SBATCH --account=punim1990
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --array=1-261%25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/data/gpfs/projects/punim1990/haninhn/firebrand_tracking/logs/track_%A_%a.out

# Firebrand detection and tracking over all 261 thermal recordings.
#
# One Level 1.1 NetCDF per array task: fetch it from MediaFlux into node-local
# scratch, run FTA_CLI.py, delete it. The set is 1.64 TiB and cannot be staged
# on /data/gpfs, so nothing is held for longer than one task. Concurrency is
# capped at 25 because the MediaFlux SFTP endpoint, not the CPU, is the limit.
# MediaFlux is only ever read: rclone copies from it, never to it.
#
# Submit through submit_array.sh, which creates logs/ (Slurm cannot write a log
# into a missing directory, and every task would fail without a trace) and
# checks the --array range against the work list:
#   spartan/submit_array.sh                              # all 261
#   spartan/submit_array.sh --array=1-3                  # pilot
#   REVIEW=$W/review/parts spartan/submit_array.sh --array=5,90,200
#       also renders review clips while each NetCDF is on disk; combine them with
#       FTA_CLI.py review-merge $W/review/batch1 $W/review/parts/*
#   EXTRA="--set persist_frac=0.2" spartan/submit_array.sh ...
#
# A task is done only when its summary.json carries the same parameter
# fingerprint as this submission. Results made with other parameters stop the
# task loudly instead of being silently mixed into the counts: change tracking
# or classification with `FTA_CLI.py retrack` (no NetCDF needed), or set FORCE=1
# to re-run detection.

set -euo pipefail

W=/data/gpfs/projects/punim1990/haninhn/firebrand_tracking
# <NetCDF path under SRC> TAB <released Level 2 run path>; build it with
#   python $W/FTA/spartan/make_worklist.py ~/restructure/reencode_all.tsv \
#       --listing $W/mediaflux_l11_thermal_listing.txt > $W/worklist.tsv
LIST=${LIST:-$W/worklist.tsv}
SRC=Mediaflux:/Volumes/proj-6600_3d_firebrand-1128.4.1067/level1.1/thermal_compressed_threshold299
PY=$W/venv/bin/python
CLI=$W/FTA/FTA_CLI.py
EXTRA=${EXTRA:-}                          # e.g. EXTRA="--set persist_frac=0.2"
REVIEW=${REVIEW:-}                        # directory for per-recording review batches (optional)
FORCE=${FORCE:-0}

# Self-contained venv under Python 3.12. Do not also load SciPy-bundle: it
# shadows the venv's numpy and breaks the netCDF4 extension. (rclone is /usr/bin.)
module purge
module load foss/2024a Python/3.12.3

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
if [ -z "$LINE" ]; then echo "no entry for task ${SLURM_ARRAY_TASK_ID}"; exit 0; fi
NC_REL=$(cut -s -f1 <<< "$LINE")
LABEL=$(cut -s -f2 <<< "$LINE")

# Validate before anything is fetched or deleted: an empty or malformed label
# would otherwise point OUT at the results root itself.
if [[ ! "$NC_REL" =~ ^level1/thermal/[^[:space:]]+\.nc$ || "$NC_REL" == *..* ]]; then
    echo "bad work-list line ${SLURM_ARRAY_TASK_ID} (NetCDF path): $LINE" >&2; exit 2
fi
if [[ ! "$LABEL" =~ ^(branchlet|fibrous_bark|candlebark|preliminary_branchlet_test)/[^[:space:]]*[^/[:space:]]$ \
      || "$LABEL" == *..* || "$LABEL" == *//* ]]; then
    echo "bad work-list line ${SLURM_ARRAY_TASK_ID} (label): $LINE" >&2; exit 2
fi
OUT=$W/results/$LABEL
case "$OUT" in "$W/results/"?*) ;; *) echo "refusing output path $OUT" >&2; exit 2 ;; esac
PART=$REVIEW/$(tr / _ <<< "$LABEL")

# One task per label at a time, even across resubmissions.
mkdir -p "$(dirname "$OUT")"
exec 9>"$OUT.lock"
if ! flock -n 9; then echo "another task is working on $LABEL; leaving it"; exit 0; fi

WANT=$("$PY" "$CLI" params $EXTRA --hash)
HAVE=""
if [ -s "$OUT/summary.json" ]; then
    HAVE=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("params_hash",""))' "$OUT/summary.json")
fi
RUN=1
if [ -n "$HAVE" ]; then
    if [ "$HAVE" = "$WANT" ]; then
        RUN=0
    elif [ "$FORCE" != "1" ]; then
        echo "$LABEL already has results with parameters $HAVE, not the requested $WANT." >&2
        echo "Use FTA_CLI.py retrack for tracking/classification changes, or FORCE=1 to re-run detection." >&2
        exit 3
    fi
fi
if [ "$RUN" = "0" ] && { [ -z "$REVIEW" ] || [ -s "$PART/manifest.json" ]; }; then
    echo "already done: $LABEL ($HAVE)"; exit 0
fi

WORK=${TMPDIR:-/tmp}/fta_track_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$WORK"
cleanup() { rm -rf -- "$WORK"; }
trap cleanup EXIT

echo "[$(date)] task ${SLURM_ARRAY_TASK_ID}: $NC_REL -> $LABEL (parameters $WANT)"
rclone copyto "$SRC/$NC_REL" "$WORK/in.nc" \
    --retries 5 --low-level-retries 20 --stats 2m --stats-one-line
echo "[$(date)] fetched $(du -h "$WORK/in.nc" | cut -f1)"

if [ "$RUN" = "1" ]; then
    TMPOUT=$OUT.partial.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
    rm -rf -- "${TMPOUT:?}"
    "$PY" "$CLI" run "$WORK/in.nc" --out "$TMPOUT" --label "$LABEL" $EXTRA
    rm -rf -- "${OUT:?}"
    mv -- "$TMPOUT" "$OUT"
fi
if [ -n "$REVIEW" ]; then
    echo "[$(date)] rendering review clips"
    "$PY" "$CLI" review-make "$OUT" "$WORK/in.nc" --batch "$PART" --label "$LABEL"
fi

echo "[$(date)] done"
