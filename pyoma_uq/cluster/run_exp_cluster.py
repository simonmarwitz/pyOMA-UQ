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
                            workers, epi_range=None):
    """PolyUQ.estimate_stat, with its epistemic loop mapped over a pool.

    *epi_range* restricts the loop to one shard of epistemic samples; the
    entries it returns are exactly those estimate_stat would have produced for
    that subset.
    """
    if epi_range is None:
        epi_range = range(poly_uq.N_mcs_epi)
    total = len(epi_range)
    stat_db, done = [], 0
    t0 = time.time()
    with ProcessPoolExecutor(
            max_workers=workers, initializer=_worker_init,
            initargs=(state, schwabach, weighting, n_stat)) as pool:
        futures = {pool.submit(_run_one, n): n for n in epi_range}
        for future in as_completed(futures):
            stat_db.extend(future.result())
            done += 1
            if done % 25 == 0 or done == total:
                wall = time.time() - t0
                rate = wall / done
                print(f'  {done}/{total} samples ({rate:.1f} s/sample wall, '
                      f'{wall * workers / done:.0f} core-s/sample, '
                      f'eta {rate * (total - done) / 3600:.1f} h)', flush=True)
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
    parser.add_argument('--epi-start', type=int, default=0)
    parser.add_argument('--epi-stop', type=int, default=None,
                        help='exclusive; default = N_epi')
    parser.add_argument('--merge', action='store_true',
                        help='skip identification, assemble the shard pickles '
                             'and run only the statistic level')
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

    import pickle
    shard_dir = args.result_dir / f'{args.weighting}_shards'
    shard_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if not args.merge:
        # ── phase 1: identification of one shard of epistemic samples ────────
        # Sharding is not an optimisation, it is what makes the run possible.
        # LSF's queue limit is on *CPU* time (TERM_CPULIMIT), not wall time:
        # Batch72 allows 72 CPU-h per job, and the whole run needs ~648 CPU-h
        # per weighting. Adding workers only spends that budget faster -- job
        # 2029290 died after 2.3 h of wall on 32 slots having done 175 of 1500
        # samples. Only splitting across jobs raises the ceiling.
        stop = args.epi_stop if args.epi_stop is not None else poly_uq.N_mcs_epi
        stop = min(stop, poly_uq.N_mcs_epi)
        out = shard_dir / f'stat_db_{args.epi_start:05d}_{stop:05d}.pkl'
        if out.exists():
            print(f'shard {out.name} already present -- nothing to do', flush=True)
            return
        print(f'>>> exp_{args.weighting} shard [{args.epi_start},{stop}) '
              f'START ({args.workers} workers)', flush=True)
        entries = _estimate_stat_parallel(
            poly_uq, args.state, args.schwabach, args.weighting, args.n_stat,
            args.workers, epi_range=range(args.epi_start, stop))
        tmp = out.with_suffix('.tmp')
        with open(tmp, 'wb') as fh:
            pickle.dump(entries, fh)
        os.replace(tmp, out)          # atomic: a partial file never looks done
        print(f'>>> shard [{args.epi_start},{stop}) DONE in '
              f'{time.time() - t0:.0f} s, {len(entries)} entries -> {out.name}',
              flush=True)
        return

    # ── phase 2: merge the shards, then the statistic level ──────────────────
    shards = sorted(shard_dir.glob('stat_db_*.pkl'))
    if not shards:
        raise SystemExit(f'no shard pickles in {shard_dir}')
    stat_db = []
    for shard in shards:
        with open(shard, 'rb') as fh:
            stat_db.extend(pickle.load(fh))
    covered = {e['n_epi'] for e in stat_db}
    missing = sorted(set(range(poly_uq.N_mcs_epi)) - covered)
    if missing:
        raise SystemExit(f'{len(missing)} epistemic samples missing from the '
                         f'shards (first: {missing[:5]}); rerun those shards '
                         'before merging')
    stat_db.sort(key=lambda e: (e['n_epi'], e['i_imp_hycs'][0]))
    poly_uq.stat_db = stat_db
    print(f'merged {len(shards)} shards -> {len(stat_db)} entries covering all '
          f'{poly_uq.N_mcs_epi} epistemic samples', flush=True)

    ex.run_experimental_pipeline(
        args.result_dir, weighting=args.weighting, poly_uq=poly_uq,
        n_stat=args.n_stat, min_coverage=args.min_coverage,
        opt_meth=args.opt_meth, skip_identification=True)
    print(f'>>> exp_{args.weighting} MERGE DONE in {time.time() - t0:.0f} s',
          flush=True)


if __name__ == '__main__':
    main()
