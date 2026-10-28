#!/bin/bash
# HTCondor wrapper: run_script.sh <CMSSW_BASE> <script under Tau3muML/scripts> <args...>
CMSSW=$1; SCRIPT=$2; shift 2
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd "$CMSSW/src" && eval "$(scramv1 runtime -sh)" && cd - > /dev/null
echo "host $(hostname), start $(date)"
python3 -u "$CMSSW/src/PhysicsTools/Tau3muML/scripts/$SCRIPT" "$@"
rc=$?
echo "end $(date), exit code $rc"
exit $rc
