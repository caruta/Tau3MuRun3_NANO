#!/bin/bash
# Usage: source resubmitAll.sh [Year] [channel: signal|control] [tag, default Transformer_v1]
# Runs resubmit.sh in every data directory of that production (YEAR_era*_CHANNEL_TAG).

if [ -z "$2" ]; then
    echo "Usage: source resubmitAll.sh [Year] [channel: signal|control] [tag, default Transformer_v1]"
    return 1 2>/dev/null || exit 1
fi

YEAR=$1
CHANNEL=$2
TAG=${3:-Transformer_v1}

for d in ${YEAR}_era*_${CHANNEL}_${TAG}; do
    [ -f "$d/resubmit.sh" ] || continue    # MC directories have a single task: use crab resubmit directly
    echo -e "\n$d"
    (cd "$d" && source resubmit.sh)
    sleep 1
done
