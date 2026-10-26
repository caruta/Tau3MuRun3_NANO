#!/bin/bash
# HTCondor executable for submit_flatten.sub:
#   run_flatten.sh <CMSSW_BASE> <channel> <isData 0|1> <output_prefix> <filelist>
# Sets up the CMSSW environment (which provides uproot and awkward) and flattens the files
# listed in <filelist> to <output_prefix>_fold<k>.root.
set -e

CMSSW_DIR=$1
CHANNEL=$2
ISDATA=$3
PREFIX=$4
FILELIST=$5

source /cvmfs/cms.cern.ch/cmsset_default.sh
pushd "$CMSSW_DIR/src" > /dev/null
eval "$(scram runtime -sh)"
popd > /dev/null

DATAFLAG=""
if [ "$ISDATA" = "1" ]; then DATAFLAG="--data"; fi

mkdir -p "$(dirname "$PREFIX")"
echo "channel=$CHANNEL data=$ISDATA output=$PREFIX files=$(wc -l < "$FILELIST")"
python3 "$CMSSW_DIR/src/PhysicsTools/Tau3muNANO/scripts/flatten_for_transformer.py" \
    --channel "$CHANNEL" $DATAFLAG -o "$PREFIX" $(cat "$FILELIST")
