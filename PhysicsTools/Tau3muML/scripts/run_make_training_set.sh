#!/bin/bash
# HTCondor wrapper for make_training_set.py: run_make_training_set.sh <CMSSW_BASE> <args...>
CMSSW=$1; shift
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd "$CMSSW/src" && eval "$(scramv1 runtime -sh)" && cd - > /dev/null
echo "host $(hostname), start $(date)"
python3 -u "$CMSSW/src/PhysicsTools/Tau3muML/scripts/make_training_set.py" "$@"
rc=$?
echo "end $(date), exit code $rc"
exit $rc
