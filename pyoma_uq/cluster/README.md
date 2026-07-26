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

## Submitting

```bash
./submit_exp.sh smoke              # both weightings
./submit_exp.sh full build         # one of them
```

`bsub` comes from `source /etc/profile` — `module load lsf` is not reliable on
makalu47. mbatchd intermittently rejects submissions with "Cannot create job
info file. Job not submitted."; the retry loop rides those windows out, and
detects success by the exact phrase `is submitted to queue` (the failure line
also contains the substring `submitted`).

## Threading

These jobs deliberately do **not** pin `OMP_NUM_THREADS=1`, unlike the CDF jobs.
There is no process pool here: the per-sample cost is one SVD/pseudo-inverse
chain that grows roughly cubically with the number of block rows, and threaded
BLAS parallelises it directly.
