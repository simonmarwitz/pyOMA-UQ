#!/bin/bash
# Retry-submit the experimental-study jobs through mbatchd's intermittent
# "Cannot create job info file. Job not submitted." windows (they clear in
# under a couple of minutes). Success is detected by the exact phrase
# "is submitted to queue" -- NOT the substring "submitted", which also occurs
# in the failure line "Job not submitted."
#
# Usage: submit_exp.sh smoke|full [build posthoc]
# NOTE: no `set -u` here. /etc/profile references unset variables, so with
# nounset the source below aborts the whole script before it prints anything
# (exit 1, no output). submit_cdfd.sh omits it for the same reason.
source /etc/profile >/dev/null 2>&1     # sets LSF_ENVDIR and puts bsub on PATH
                                        # ('module load lsf' is not reliable)
command -v bsub >/dev/null || { echo "bsub not on PATH after sourcing /etc/profile"; exit 1; }
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE" || exit 1

TAG=${1:-smoke}
shift || true
COMBOS=${*:-"build posthoc"}

declare -A JID
for c in $COMBOS; do JID[$c]=''; done
MAXTRY=120

for attempt in $(seq 1 $MAXTRY); do
  alldone=1
  for c in $COMBOS; do
    [ -n "${JID[$c]}" ] && continue
    f="exp_${TAG}_${c}.bsub"
    if [ ! -f "$f" ]; then echo "missing $f"; continue; fi
    out=$(bsub < "$f" 2>&1 | grep -E 'is submitted to queue|not submitted|Cannot' | head -1)
    if echo "$out" | grep -q 'is submitted to queue'; then
      JID[$c]=$(echo "$out" | sed -n 's/.*Job <\([0-9]*\)>.*/\1/p')
      echo "[try $attempt] OK   exp_${TAG}_${c} -> job ${JID[$c]}"
    else
      alldone=0
      [ "$attempt" -eq 1 ] && echo "[try $attempt] wait exp_${TAG}_${c}: $out"
    fi
  done
  [ $alldone -eq 1 ] && break
  sleep 8
done

echo '--- final ---'
n_ok=0; n_tot=0
for c in $COMBOS; do
  n_tot=$((n_tot+1))
  if [ -n "${JID[$c]}" ]; then echo "exp_${TAG}_${c}: ${JID[$c]}"; n_ok=$((n_ok+1))
  else echo "exp_${TAG}_${c}: NOT_SUBMITTED"; fi
done
echo "submitted $n_ok/$n_tot"
