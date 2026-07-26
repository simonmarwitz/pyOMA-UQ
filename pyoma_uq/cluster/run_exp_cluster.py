"""Cluster driver for the experimental (Schwabach) study, one weighting per job.

Mirrors run_sd_cluster.py: restore the sampling state shipped from the local
machine and run the pipeline against it.

R1 -- NEVER sample_qmc on the cluster. sample_qmc draws its Halton sequence
through scipy.stats.qmc, whose output is not reproducible across scipy
versions (cluster 1.11 vs local 1.17). Resampling here would move the
epistemic samples out from under the surrogate that the local results were
fitted on, silently and without error. The state is restored with
load_state(differential='samp') instead.

Parallelism: a process pool over epistemic samples, with BLAS pinned to one
thread per worker -- the same shape as run_cdf_cluster.py, and for the same
reason. The first smoke run tried the opposite (one process, 16 BLAS threads)
on the theory that the per-sample SVD/pseudo-inverse chain would thread well.
It does not: LSF reported Max Threads 20 on 16 slots for 310 s/sample, against
540 s/sample on 4 local cores -- 1.74x for 4x the cores. At that rate the full
N_epi run needed 72.3 h and did not fit Batch72's 72 h wall. Epistemic samples
are independent, so distributing whole samples scales far better.

_estimate_stat_parallel below reproduces PolyUQ.estimate_stat's contract
(per-hypercube weights, dedupe of identical weight vectors, stat_db entry
shape) but maps the epistemic loop over the pool. Weights stay PolyUQ's --
each worker restores the same sampling state and computes them itself, so
there is still no weights exit point.

Usage:
  run_exp_cluster.py --weighting build \
      --result-dir /scratch/sima9999/modal_uq/schwabach/exp_run \
      --state      /scratch/sima9999/modal_uq/schwabach/samp_state_exp.npz \
      --schwabach  /home/sima9999/scratch/modal_uq/2019_Schwabach
"""
import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_WORKER = {}


def _worker_init(state_path, schwabach, weighting, n_stat):
    """One restored PolyUQ + estimator per worker process."""
    import logging as _logging
    _logging.disable(_logging.WARNING)
    try:
        from alive_progress import config_handler
        config_handler.set_global(disable=True)
    except Exception:
        pass
    import numpy as np
    from polyuq import PolyUQ
    from pyoma_uq.studies import UQ_OMA_experimental as ex

    ex.SCHWABACH_DIR = Path(schwabach)
    vars_ale, vars_epi, levels, offsets = ex.vars_definition_experimental()
    path = 'fast_build' if weighting == 'build' else 'fast_posthoc'
    pq = PolyUQ(vars_ale, vars_epi, dim_ex='cartesian', path=path)
    pq.load_state(str(state_path), differential='samp')   # R1: no resample
    targets = np.linspace(0, 1, n_stat) if n_stat else None
    _WORKER['pq'] = pq
    _WORKER['ex'] = ex
    _WORKER['estimator'] = ex.make_experimental_estimator(
        pq, ex.load_baseline_modes(), offsets, weighting=weighting,
        target_probabilities=targets)


def _run_one(n_epi):
    """One epistemic sample: PolyUQ's weights per hypercube, deduped, then the
    estimator. Mirrors the body of PolyUQ.estimate_stat's weighted branch."""
    pq, estimator = _WORKER['pq'], _WORKER['estimator']
    cache = {}
    for i_imp in range(len(pq.imp_hyc_foc_inds)):
        weights = pq._weights(i_imp=i_imp, n_epi=n_epi, eliminate=False)
        key = weights.tobytes()
        if key not in cache:
            cache[key] = (estimator(n_epi, i_imp, weights), [])
        cache[key][1].append(i_imp)
    return [dict(result, n_epi=n_epi, i_imp_hycs=tuple(i_imps))
            for result, i_imps in cache.values()]


def _estimate_stat_parallel(poly_uq, state, schwabach, weighting, n_stat,
                            workers):
    """PolyUQ.estimate_stat, with its epistemic loop mapped over a pool."""
    stat_db, done = [], 0
    t0 = time.time()
    with ProcessPoolExecutor(
            max_workers=workers, initializer=_worker_init,
            initargs=(state, schwabach, weighting, n_stat)) as pool:
        futures = {pool.submit(_run_one, n): n
                   for n in range(poly_uq.N_mcs_epi)}
        for future in as_completed(futures):
            stat_db.extend(future.result())
            done += 1
            if done % 25 == 0 or done == poly_uq.N_mcs_epi:
                rate = (time.time() - t0) / done
                print(f'  {done}/{poly_uq.N_mcs_epi} samples '
                      f'({rate:.1f} s/sample wall, '
                      f'eta {rate * (poly_uq.N_mcs_epi - done) / 3600:.1f} h)',
                      flush=True)
    stat_db.sort(key=lambda e: (e['n_epi'], e['i_imp_hycs'][0]))
    poly_uq.stat_db = stat_db
    return stat_db


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--weighting', choices=('build', 'posthoc'),
                        required=True)
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--schwabach', type=Path, required=True)
    parser.add_argument('--n-stat', type=int, default=0)
    parser.add_argument('--min-coverage', type=float, default=0.1)
    parser.add_argument('--opt-meth', default='genetic')
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    logging.getLogger('pyOMA').setLevel(logging.WARNING)
    # alive_progress floods a non-TTY bsub log with 100+ MB otherwise
    try:
        from alive_progress import config_handler
        config_handler.set_global(disable=True)
    except Exception:
        pass

    os.environ['SCHWABACH_DIR'] = str(args.schwabach)
    from polyuq import PolyUQ
    from pyoma_uq.studies import UQ_OMA_experimental as ex

    ex.SCHWABACH_DIR = args.schwabach          # module read it at import time
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir / f'run_exp_{args.weighting}.pid').write_text(str(os.getpid()))

    vars_ale, vars_epi, levels, offsets = ex.vars_definition_experimental()
    path = 'fast_build' if args.weighting == 'build' else 'fast_posthoc'
    poly_uq = PolyUQ(vars_ale, vars_epi, dim_ex='cartesian', path=path)
    poly_uq.load_state(str(args.state), differential='samp')   # R1
    print(f'restored state: N_ale={poly_uq.N_mcs_ale} N_epi={poly_uq.N_mcs_epi} '
          f'imp_hyc={len(poly_uq.imp_hyc_foc_inds)}', flush=True)

    # the blocks are a property of the measurement, so the restored aleatory
    # sample values must equal the levels this machine computes; a mismatch
    # means the state belongs to a different data set
    import numpy as np
    restored = poly_uq.inp_samp_prim['a_ref'].iloc[:len(levels)].values
    if not np.allclose(restored, levels, rtol=1e-9, atol=0):
        raise SystemExit('restored a_ref samples differ from the measured block '
                         'levels -- wrong sampling state for this data set')
    print(f'a_ref matches the {len(levels)} measured block levels', flush=True)

    t0 = time.time()
    print(f'>>> exp_{args.weighting} START ({args.workers} workers)', flush=True)

    # phase 1: identification, distributed over the epistemic samples
    _estimate_stat_parallel(poly_uq, args.state, args.schwabach,
                            args.weighting, args.n_stat, args.workers)
    print(f'    identification done in {time.time() - t0:.0f} s, '
          f'{len(poly_uq.stat_db)} stat_db entries', flush=True)

    # phase 2: statistic level, in the parent against the assembled stat_db
    ex.run_experimental_pipeline(
        args.result_dir, weighting=args.weighting, poly_uq=poly_uq,
        n_stat=args.n_stat, min_coverage=args.min_coverage,
        opt_meth=args.opt_meth, skip_identification=True)
    print(f'>>> exp_{args.weighting} DONE in {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
