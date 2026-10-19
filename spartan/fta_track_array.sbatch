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
# Read-only on MediaFlux: outputs go to $W/results only.
#
#   sbatch fta_track_array.sbatch                 # all 261
#   sbatch --array=1-3 fta_track_array.sbatch     # pilot
#
# Re-runnable: a task whose summary.json exists is skipped.

set -euo pipefail

W=/data/gpfs/projects/punim1990/haninhn/firebrand_tracking
LIST=${LIST:-$W/worklist.tsv}            # <NetCDF path under SRC> TAB <Level 2 run path>
SRC=Mediaflux:/Volumes/proj-6600_3d_firebrand-1128.4.1067/level1.1/thermal_compressed_threshold299
PY=$W/venv/bin/python
CLI=$W/FTA/FTA_CLI.py
EXTRA=${EXTRA:-}                          # e.g. EXTRA="--set persist_frac=0.2"

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
if [ -z "$LINE" ]; then echo "no entry for task ${SLURM_ARRAY_TASK_ID}"; exit 0; fi
NC_REL=$(cut -f1 <<< "$LINE")
LABEL=$(cut -f2 <<< "$LINE")
OUT=$W/results/$LABEL

if [ -s "$OUT/summary.json" ]; then echo "already done: $LABEL"; exit 0; fi

WORK=${TMPDIR:-/tmp}/fta_track_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$WORK"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

echo "[$(date)] task ${SLURM_ARRAY_TASK_ID}: $NC_REL -> $LABEL"
rclone copyto "$SRC/$NC_REL" "$WORK/in.nc" \
    --retries 5 --low-level-retries 20 --stats 2m --stats-one-line

echo "[$(date)] tracking $(du -h "$WORK/in.nc" | cut -f1)"
# Self-contained venv under Python 3.12. Do not also load SciPy-bundle: it
# shadows the venv's numpy and breaks the netCDF4 extension.
module purge
module load foss/2024a Python/3.12.3
"$PY" "$CLI" run "$WORK/in.nc" --out "$OUT.partial" --label "$LABEL" $EXTRA
rm -rf "$OUT" && mv "$OUT.partial" "$OUT"

echo "[$(date)] done"
