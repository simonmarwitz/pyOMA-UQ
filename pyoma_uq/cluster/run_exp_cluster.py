"""Cluster driver for the experimental (Schwabach) study, one weighting per job.

Mirrors run_sd_cluster.py: restore the sampling state shipped from the local
machine and run the pipeline against it.

R1 -- NEVER sample_qmc on the cluster. sample_qmc draws its Halton sequence
through scipy.stats.qmc, whose output is not reproducible across scipy
versions (cluster 1.11 vs local 1.17). Resampling here would move the
epistemic samples out from under the surrogate that the local results were
fitted on, silently and without error. The state is restored with
load_state(differential='samp') instead.

Unlike the CDF jobs this driver does NOT use a process pool: the cost sits in
one big SVD/pseudo-inverse chain per epistemic sample (roughly cubic in the
number of block rows), which threaded BLAS parallelises directly. So the .bsub
leaves OMP_NUM_THREADS at the slot count rather than pinning it to 1.

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
from pathlib import Path


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
    print(f'>>> exp_{args.weighting} START', flush=True)
    ex.run_experimental_pipeline(
        args.result_dir, weighting=args.weighting, poly_uq=poly_uq,
        n_stat=args.n_stat, min_coverage=args.min_coverage,
        opt_meth=args.opt_meth)
    print(f'>>> exp_{args.weighting} DONE in {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
