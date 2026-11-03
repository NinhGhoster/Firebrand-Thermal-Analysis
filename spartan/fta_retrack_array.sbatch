#!/bin/bash
#SBATCH --job-name=fta-retrack
#SBATCH --account=punim1990
#SBATCH --partition=sapphire
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/data/gpfs/projects/punim1990/haninhn/firebrand_tracking/logs/retrack_%A_%a.out

# Re-track and re-classify every recording in a results tree from its saved
# detections (no NetCDF, nothing fetched), one recording per array task.
#
#   RESULTS=$W/results_v2 LIST=$W/worklist.tsv RETRACK_OPTS="--from-defaults" \
#   sbatch --array=1-261%50 $W/FTA/spartan/fta_retrack_array.sbatch
#
# Rewrites the outputs inside RESULTS: run it on a copy of a results tree, never
# on the only one.

set -euo pipefail

W=/data/gpfs/projects/punim1990/haninhn/firebrand_tracking
PY=$W/venv/bin/python
CLI=$W/FTA/FTA_CLI.py
RESULTS=${RESULTS:?set RESULTS to the results tree to re-track}
LIST=${LIST:-$W/worklist.tsv}
RETRACK_OPTS=${RETRACK_OPTS:-}

case "$RESULTS" in "$W/results" | "$W/results/") echo "refusing the main results tree; copy it first" >&2; exit 2 ;; esac
case "$RESULTS" in "$W/"?*) ;; *) echo "refusing $RESULTS: not under $W" >&2; exit 2 ;; esac
[[ "$RESULTS" != *..* ]] || { echo "refusing $RESULTS" >&2; exit 2; }
read -ra OPTS <<< "$RETRACK_OPTS"
for ((i = 0; i < ${#OPTS[@]}; i++)); do
    case "${OPTS[i]}" in
        --from-defaults) ;;
        --set) [[ "${OPTS[i+1]:-}" =~ ^[a-z_0-9]+=[^[:space:]]*$ ]] || { echo "bad RETRACK_OPTS" >&2; exit 2; }; i=$((i + 1)) ;;
        *) echo "RETRACK_OPTS may not hold ${OPTS[i]}" >&2; exit 2 ;;
    esac
done

module purge
module load foss/2024a Python/3.12.3

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
if [ -z "$LINE" ]; then echo "no entry for task ${SLURM_ARRAY_TASK_ID}"; exit 0; fi
LABEL=$(cut -s -f2 <<< "$LINE")
if [[ ! "$LABEL" =~ ^(branchlet|fibrous_bark|candlebark|preliminary_branchlet_test)/[^[:space:]]*[^/[:space:]]$ \
      || "$LABEL" == *..* || "$LABEL" == *//* ]]; then
    echo "bad work-list line (label): $LINE" >&2; exit 2
fi
DIR=$RESULTS/$LABEL
[ -s "$DIR/summary.json" ] && [ -s "$DIR/detections.csv.gz" ] || { echo "no results at $DIR" >&2; exit 2; }
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID}: $LABEL"
"$PY" "$CLI" retrack "$DIR" "${OPTS[@]+"${OPTS[@]}"}"
echo "[$(date)] done"
