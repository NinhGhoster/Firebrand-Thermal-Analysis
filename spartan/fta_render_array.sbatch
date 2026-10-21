#!/bin/bash
#SBATCH --job-name=fta-render
#SBATCH --account=punim1990
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/data/gpfs/projects/punim1990/haninhn/firebrand_tracking/logs/render_%A_%a.out

# Render review clips (and contact sheets) for chosen recordings from results made
# elsewhere, e.g. a retrack with other parameters into its own results tree. One
# NetCDF per array task: fetched from MediaFlux into node-local scratch, read, deleted.
# MediaFlux is only ever read.
#
#   LIST=$W/review/pilot3/worklist.tsv RESULTS=$W/results_pilot3 BATCH=$W/review/pilot3/parts \
#   MAKE_OPTS="--all-counted --near-miss --n-rejected 15" INFALL=$W/results_pilot3_infall \
#   sbatch --array=1-10%10 $W/FTA/spartan/fta_render_array.sbatch
#
# LIST lines are those of worklist.tsv (<NetCDF path> TAB <Level 2 run path>). With
# INFALL, the recording's falling-in-the-flame candidates (FTA_CLI.py infall) are
# rendered too, as a separate part labelled <run path>/infall. Combine the parts with
#   FTA_CLI.py review-merge $W/review/pilot3 $W/review/pilot3/parts/*

set -euo pipefail

W=/data/gpfs/projects/punim1990/haninhn/firebrand_tracking
SRC=Mediaflux:/Volumes/proj-6600_3d_firebrand-1128.4.1067/level1.1/thermal_compressed_threshold299
PY=$W/venv/bin/python
CLI=$W/FTA/FTA_CLI.py
LIST=${LIST:?set LIST to a work list}
RESULTS=${RESULTS:?set RESULTS to the results root to render from}
BATCH=${BATCH:?set BATCH to the directory for the per-recording parts}
MAKE_OPTS=${MAKE_OPTS:-}
INFALL=${INFALL:-}

module purge
module load foss/2024a Python/3.12.3

for d in "$RESULTS" "$BATCH" ${INFALL:+"$INFALL"}; do
    case "$d" in "$W/"?*) ;; *) echo "refusing $d: not under $W" >&2; exit 2 ;; esac
    [[ "$d" != *..* ]] || { echo "refusing $d" >&2; exit 2; }
done
read -ra OPTS <<< "$MAKE_OPTS"
for ((i = 0; i < ${#OPTS[@]}; i++)); do
    case "${OPTS[i]}" in
        --all-counted|--near-miss|--no-sheets) ;;
        --n-rejected|--n-firebrand|--seed|--min-det)
            [[ "${OPTS[i+1]:-}" =~ ^[0-9]+$ ]] || { echo "bad MAKE_OPTS: $MAKE_OPTS" >&2; exit 2; }
            i=$((i + 1)) ;;
        *) echo "MAKE_OPTS may not hold ${OPTS[i]}" >&2; exit 2 ;;
    esac
done

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
if [ -z "$LINE" ]; then echo "no entry for task ${SLURM_ARRAY_TASK_ID}"; exit 0; fi
NC_REL=$(cut -s -f1 <<< "$LINE")
LABEL=$(cut -s -f2 <<< "$LINE")
if [[ ! "$NC_REL" =~ ^level1/thermal/[^[:space:]]+\.nc$ || "$NC_REL" == *..* ]]; then
    echo "bad work-list line (NetCDF path): $LINE" >&2; exit 2
fi
if [[ ! "$LABEL" =~ ^(branchlet|fibrous_bark|candlebark|preliminary_branchlet_test)/[^[:space:]]*[^/[:space:]]$ \
      || "$LABEL" == *..* || "$LABEL" == *//* ]]; then
    echo "bad work-list line (label): $LINE" >&2; exit 2
fi
IN=$RESULTS/$LABEL
[ -s "$IN/summary.json" ] || { echo "no results at $IN" >&2; exit 2; }
PART=$BATCH/$(tr / _ <<< "$LABEL")

WORK=${TMPDIR:-/tmp}/fta_render_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$WORK" "$BATCH"
cleanup() { rm -rf -- "$WORK"; }
trap cleanup EXIT

echo "[$(date)] task ${SLURM_ARRAY_TASK_ID}: $NC_REL -> $LABEL"
rclone copyto "$SRC/$NC_REL" "$WORK/in.nc" --retries 5 --low-level-retries 20 --stats 2m --stats-one-line
echo "[$(date)] fetched $(du -h "$WORK/in.nc" | cut -f1)"
"$PY" "$CLI" review-make "$IN" "$WORK/in.nc" --batch "$PART" --label "$LABEL" "${OPTS[@]+"${OPTS[@]}"}"
if [ -n "$INFALL" ]; then
    if [ -s "$INFALL/$LABEL/summary.json" ]; then
        "$PY" "$CLI" review-make "$INFALL/$LABEL" "$WORK/in.nc" --batch "${PART}__infall" \
            --label "$LABEL/infall" --all-counted --n-rejected 0
    else
        echo "no infall candidates at $INFALL/$LABEL"
    fi
fi
echo "[$(date)] done"
