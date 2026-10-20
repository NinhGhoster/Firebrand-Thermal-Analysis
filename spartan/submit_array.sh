#!/bin/bash
# Submit fta_track_array.sbatch safely: create the log and results directories
# (Slurm cannot write a log into a missing directory, so every task would fail
# without a trace) and check that every requested index has a work-list line.
#   spartan/submit_array.sh [--array=SPEC] [other sbatch options]
set -euo pipefail
W=/data/gpfs/projects/punim1990/haninhn/firebrand_tracking
LIST=${LIST:-$W/worklist.tsv}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$W/logs" "$W/results"
[ -s "$LIST" ] || { echo "work list $LIST is missing or empty" >&2; exit 1; }
N=$(grep -c . "$LIST")
SPEC="1-261"
for a in "$@"; do case "$a" in --array=*) SPEC=${a#--array=} ;; esac; done
MAXIDX=$(tr ',' '\n' <<< "${SPEC%%\%*}" | sed 's/-/\n/' | grep -o '^[0-9]*' | sort -n | tail -1)
if [ "$MAXIDX" -gt "$N" ]; then
    echo "array index $MAXIDX exceeds the $N lines of $LIST" >&2; exit 1
fi
cd "$W"
exec sbatch --parsable --export=ALL "$@" "$HERE/fta_track_array.sbatch"
