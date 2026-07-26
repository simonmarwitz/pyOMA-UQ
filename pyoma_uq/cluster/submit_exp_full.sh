#!/bin/bash
# Shard the full experimental run across LSF jobs and submit them.
#
# WHY SHARD: LSF's queue limit is on CPU time, not wall time
# (Batch72: CPULIMIT 4320 min = 72 CPU-h; TERM_CPULIMIT kills you). The run
# needs ~648 CPU-h per weighting, so no single job can hold it and adding
# workers only spends the budget faster -- job 2029290 died after 2.3 h of wall
# on 32 slots having completed 175 of 1500 samples. Splitting across jobs is
# the only thing that raises the ceiling.
#
# Sizing: ~1556 core-s per epistemic sample (measured, job 2029251), so
# SHARD samples cost SHARD*1556/3600 CPU-h; keep that under ~55 for margin.
#
# Usage: submit_exp_full.sh [SHARD] [WORKERS] [weighting ...]
# no arguments = 125-sample shards, 16 workers, both weightings
# NOTE: no `set -u` -- /etc/profile references unset vars (see submit_exp.sh).
source /etc/profile >/dev/null 2>&1
command -v bsub >/dev/null || { echo "bsub not on PATH"; exit 1; }
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE" || exit 1

SHARD=${1:-125}
WORKERS=${2:-16}
shift 2 2>/dev/null
WEIGHTINGS=${*:-"build posthoc"}

N_EPI=1500
BASE=/scratch/sima9999/modal_uq/schwabach
RD=$BASE/exp_run
GEN=$HERE/generated
mkdir -p "$GEN"

cpuh=$(awk -v s=$SHARD 'BEGIN{printf "%.1f", s*1556/3600}')
echo "shard=$SHARD samples (~$cpuh CPU-h each, limit 72), workers=$WORKERS"

submit_one() {   # $1 = bsub file, $2 = label
  for attempt in $(seq 1 60); do
    out=$(bsub < "$1" 2>&1 | grep -E 'is submitted to queue|not submitted|Cannot' | head -1)
    if echo "$out" | grep -q 'is submitted to queue'; then
      echo "  OK  $2 -> $(echo "$out" | sed -n 's/.*Job <\([0-9]*\)>.*/\1/p')"
      return 0
    fi
    sleep 8
  done
  echo "  FAIL $2: $out"; return 1
}

n_ok=0; n_tot=0
for w in $WEIGHTINGS; do
  for start in $(seq 0 $SHARD $((N_EPI-1))); do
    stop=$((start+SHARD)); [ $stop -gt $N_EPI ] && stop=$N_EPI
    tag=$(printf "%05d_%05d" $start $stop)
    f="$GEN/exp_${w}_${tag}.bsub"
    cat > "$f" <<BSUB
#!/bin/bash
#BSUB -q Batch72
#BSUB -n $WORKERS
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=24000]"
#BSUB -J exp_${w}_${tag}
#BSUB -oo $BASE/logs/exp_${w}_${tag}_%J.log
#BSUB -eo $BASE/logs/exp_${w}_${tag}_%J.err
#BSUB -L /usr/bin/bash
export PYTHONPATH=\$HOME/git_cs2redo/uq_oma_b:\$HOME/git_cs2redo/PolyUQ:\$HOME/git_cs2redo/pyOMA
export SCHWABACH_DIR=\$HOME/scratch/modal_uq/2019_Schwabach
export POLYUQ_JOB_ID=\$LSB_JOBID
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
\$HOME/conda/envs/py311/bin/python \$HOME/git_cs2redo/uq_oma_b/pyoma_uq/cluster/run_exp_cluster.py \\
  --weighting $w --result-dir $RD --state $BASE/samp_state_exp_full.npz \\
  --schwabach \$HOME/scratch/modal_uq/2019_Schwabach \\
  --workers $WORKERS --epi-start $start --epi-stop $stop
BSUB
    # resumable: a completed shard leaves its pickle behind, skip it
    if [ -f "$RD/${w}_shards/stat_db_${tag}.pkl" ]; then
      echo "  skip $w $tag (already done)"; continue
    fi
    n_tot=$((n_tot+1)); submit_one "$f" "$w $tag" && n_ok=$((n_ok+1))
  done
done
echo "submitted $n_ok/$n_tot shards"
echo "when all shards are done, merge with:  submit_exp_merge.sh"
