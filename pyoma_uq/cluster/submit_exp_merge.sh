#!/bin/bash
# Merge the shard pickles and run the statistic level (one job per weighting).
source /etc/profile >/dev/null 2>&1
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE" || exit 1
BASE=/scratch/sima9999/modal_uq/schwabach
RD=$BASE/exp_run
GEN=$HERE/generated; mkdir -p "$GEN"
for w in ${*:-build posthoc}; do
  f="$GEN/exp_merge_${w}.bsub"
  cat > "$f" <<BSUB
#!/bin/bash
#BSUB -q Batch72
#BSUB -n 1
#BSUB -R "rusage[mem=16000]"
#BSUB -J exp_merge_${w}
#BSUB -oo $BASE/logs/exp_merge_${w}_%J.log
#BSUB -eo $BASE/logs/exp_merge_${w}_%J.err
#BSUB -L /usr/bin/bash
export PYTHONPATH=\$HOME/git_cs2redo/uq_oma_b:\$HOME/git_cs2redo/PolyUQ:\$HOME/git_cs2redo/pyOMA
export SCHWABACH_DIR=\$HOME/scratch/modal_uq/2019_Schwabach
export OMP_NUM_THREADS=1
\$HOME/conda/envs/py311/bin/python \$HOME/git_cs2redo/uq_oma_b/pyoma_uq/cluster/run_exp_cluster.py \\
  --weighting $w --result-dir $RD --state $BASE/samp_state_exp_full.npz \\
  --schwabach \$HOME/scratch/modal_uq/2019_Schwabach --merge
BSUB
  for attempt in $(seq 1 60); do
    out=$(bsub < "$f" 2>&1 | grep -E 'is submitted to queue|not submitted|Cannot' | head -1)
    echo "$out" | grep -q 'is submitted to queue' && { echo "OK merge $w -> $(echo "$out" | sed -n 's/.*Job <\([0-9]*\)>.*/\1/p')"; break; }
    sleep 8
  done
done
