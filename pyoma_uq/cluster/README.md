# Cluster job scripts (makalu, LSF)

## The rule that matters

**Never `sample_qmc` on the cluster.** `sample_qmc` draws its Halton sequence
through `scipy.stats.qmc`, which is not reproducible across scipy versions
(cluster 1.11 vs local 1.17). Resampling there moves the epistemic samples out
from under the surrogate the results were fitted on — silently, with no error.

Sample locally, ship the state, restore it:

```python
poly_uq.save_state('samp_state_exp.npz', differential='samp')   # local
poly_uq.load_state('samp_state_exp.npz', differential='samp')   # cluster
```

`run_exp_cluster.py` additionally asserts that the restored `a_ref` samples
equal the block levels computed from the measurement, so a state belonging to a
different data set fails loudly instead of producing plausible nonsense.

## Layout

| file | role |
|---|---|
| `run_exp_cluster.py` | driver: restore state → `run_experimental_pipeline` for one weighting |
| `exp_{smoke,full}_{build,posthoc}.bsub` | LSF job files |
| `submit_exp.sh` | retry-submit loop |

## The other rule that matters: LSF limits CPU time, not wall time

`Batch72` means **72 CPU-hours per job**, not 72 hours of wall
(`CPULIMIT 4320 min`; exceeding it gives `TERM_CPULIMIT`). Adding workers
therefore cannot buy a longer run -- it spends the same budget faster. Job
2029290 asked for 32 slots and died after 2.3 h of wall having completed 175 of
1500 samples, at exactly 72.2 CPU-h.

The full run needs ~648 CPU-h per weighting (1500 samples x ~1556 core-s), so
it *must* be split across jobs:

```bash
./submit_exp_full.sh                 # 125-sample shards, 16 workers, both
./submit_exp_full.sh 100 16 build    # smaller shards, one weighting
./submit_exp_merge.sh                # once every shard pickle exists
```

Each shard writes `<result>/<weighting>_shards/stat_db_<start>_<stop>.pkl`
atomically, so a killed job never leaves a half-written shard, and re-running
`submit_exp_full.sh` skips the shards already present. The merge step refuses
to run if any epistemic sample is missing from the shards rather than quietly
producing a statistic level fitted on a subset.

Queue CPU limits, for sizing: `Batch24` 24 CPU-h, `Batch72` 72 CPU-h,
`InterXL` 666 CPU-h; `BatchXL` and `highmem` publish none but sit on different
host sets and are heavily contended.

## Submitting the smoke

```bash
./submit_exp.sh smoke              # both weightings
./submit_exp.sh smoke build        # one of them
```

`bsub` comes from `source /etc/profile` — `module load lsf` is not reliable on
makalu47. mbatchd intermittently rejects submissions with "Cannot create job
info file. Job not submitted."; the retry loop rides those windows out, and
detects success by the exact phrase `is submitted to queue` (the failure line
also contains the substring `submitted`).

## Threading — measured, not assumed

`OMP_NUM_THREADS=1` plus a process pool over epistemic samples, as in the CDF
jobs. The first attempt did the opposite (one process, 16 BLAS threads), on the
theory that the per-sample SVD/pseudo-inverse chain would thread well. It does
not:

| | s/sample | cores |
|---|---|---|
| local, 4 cores, threaded | 540 | 4 |
| cluster job 2029214, threaded | 310 | 16 |

1.74x for 4x the cores, which put the full run at **72.3 h** — past Batch72's
72 h wall. Epistemic samples are independent, so distributing whole samples
scales properly instead.

`_estimate_stat_parallel` reproduces `PolyUQ.estimate_stat`'s contract exactly
(per-hypercube weights, dedupe of identical weight vectors, `stat_db` entry
shape); each worker restores the same sampling state and computes its own
weights, so there is still no weights exit point.

Note that `_worker_init` re-establishes `SCHWABACH_DIR` inside the child rather
than relying on the parent's assignment: Python 3.14 defaults to
`forkserver`/`spawn` on Linux, where module-level state set in the parent does
not reach workers.
