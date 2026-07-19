"""Final validation of the completed weighted-OMA run (plan section 4, items 6-7).
Uses the artifacts persisted by run_weighted_uq_pipeline (pole_db.pkl,
weighted_theta.npz, weighted_focals_*.npz) — no recomputation."""
import sys
import pickle
import logging
from pathlib import Path

sys.path.insert(0, '/home/womo1998/dev/uq_oma_a/oma_uq')
logging.disable(logging.INFO)

import numpy as np

RESULT = Path('/home/womo1998/Projects/uq_oma_a/weighted_dev')
ARCHIVE = Path('/home/womo1998/Projects/uq_oma_a/estimations')
KNOWN_F = np.array([0.157, 0.163, 0.179, 0.180, 0.3154, 0.3352, 0.5801,
                    0.6038, 1.1995, 1.2495, 2.0111, 2.0974, 3.035, 3.167,
                    4.285, 4.472])
KNOWN_D = np.array([1.398, 2.027, 8.897, 8.592, 1.460, 1.952, 0.169, 0.166,
                    0.107, 0.103, 0.077, 0.079, np.nan, np.nan, np.nan, np.nan])

from examples import UQ_OMA_weighted as w
from polyuq import PolyUQ

print('=== 1+2. pole database: mode coverage and std sanity ===')
with open(RESULT / 'pole_db.pkl', 'rb') as fh:
    pole_db = pickle.load(fh)
n_eff = np.array([e['n_eff'] for e in pole_db[::36]])  # 36 imp-hyc / epi sample
print(f'{len(pole_db)} pole-db entries; n_eff over epi samples: '
      f'min {n_eff.min():.1f}, median {np.median(n_eff):.1f}, max {n_eff.max():.1f}')
std_f_all = np.concatenate([e['std_f'] for e in pole_db[::36]])
std_d_all = np.concatenate([e['std_d'] for e in pole_db[::36]])
print(f'std_f: {np.mean(np.isfinite(std_f_all))*100:.1f}% finite, '
      f'{np.mean(std_f_all[np.isfinite(std_f_all)] >= 0)*100:.1f}% >= 0')
print(f'std_d: {np.mean(np.isfinite(std_d_all))*100:.1f}% finite')

print()
print('=== 3+5. focal intervals: ordering, nesting, archive comparison ===')
# archived per-sample-OMA study results (final averaged Incompleteness stats)
archived = {}
for mdir in sorted(ARCHIVE.glob('f_sc-*')):
    j = int(mdir.name.split('-')[1])
    try:
        with np.load(mdir / 'polyuq_avg_inc.npz', allow_pickle=True) as arr:
            stats = arr['self.focals_stats'][0]
            mass = np.ravel(arr['self.focals_mass'])
        ok = np.isfinite(stats).all(axis=1) & np.isfinite(mass) & (mass > 0)
        mid = np.average(np.mean(stats[ok], axis=1), weights=mass[ok])
        archived[j] = (mid, stats[ok], mass[ok])
    except Exception as e:
        print(f'  archive f_sc-{j}: {e!r}')
archived_d = {}
for mdir in sorted(ARCHIVE.glob('d_sc-*')):
    j = int(mdir.name.split('-')[1])
    try:
        with np.load(mdir / 'polyuq_avg_inc.npz', allow_pickle=True) as arr:
            stats = arr['self.focals_stats'][0]
            mass = np.ravel(arr['self.focals_mass'])
        ok = np.isfinite(stats).all(axis=1) & np.isfinite(mass) & (mass > 0)
        archived_d[j] = stats[ok]
    except Exception:
        pass

hdr = (f'{"clu":>3s} {"median f":>9s} {"known":>7s} {"off%":>6s} | '
       f'{"weighted f-envelope":>22s} | {"archived f-envelope":>22s} | '
       f'{"d med":>6s} {"known d":>7s} | ordered nested')
print(hdr)
for f_file in sorted(RESULT.glob('weighted_focals_*_f.npz'),
                     key=lambda p: int(p.stem.split('_')[2])):
    label = int(f_file.stem.split('_')[2])
    with np.load(f_file) as arr:
        foc_f, mass = arr['imp_foc'], arr['imp_hyc_mass']
        med_f = float(arr['median'])
    d_file = RESULT / f'weighted_focals_{label}_d.npz'
    med_d, foc_d = np.nan, None
    if d_file.exists():
        with np.load(d_file) as arr:
            foc_d, med_d = arr['imp_foc'], float(arr['median'])

    i_known = np.argmin(np.abs(KNOWN_F - med_f))
    off = (med_f - KNOWN_F[i_known]) / KNOWN_F[i_known] * 100
    ordered = bool(np.all(foc_f[:, 0] <= foc_f[:, 1] + 1e-12))
    # nesting proxy: envelope of high-mass focals inside overall envelope
    env = (np.min(foc_f[:, 0]), np.max(foc_f[:, 1]))
    hm = mass >= np.median(mass)
    nested = bool(np.min(foc_f[hm, 0]) >= env[0] - 1e-12
                  and np.max(foc_f[hm, 1]) <= env[1] + 1e-12)

    arch_txt = f'{"-":>22s}'
    if archived:
        j_near = min(archived, key=lambda j: abs(archived[j][0] - med_f))
        a_mid, a_stats, _ = archived[j_near]
        if abs(a_mid - med_f) / med_f < 0.05:
            arch_txt = (f'[{np.min(a_stats[:,0]):8.4f},'
                        f'{np.max(a_stats[:,1]):9.4f}] ')
    kd = KNOWN_D[i_known]
    print(f'{label:3d} {med_f:9.4f} {KNOWN_F[i_known]:7.4f} {off:6.2f} | '
          f'[{env[0]:8.4f},{env[1]:9.4f}]  | {arch_txt} | '
          f'{med_d:6.4f} {kd if np.isfinite(kd) else float("nan"):7.3f} | '
          f'{ordered!s:7s} {nested!s}')

print()
print('=== 4. weight sensitivity: monotone shift with lamda_vb ===')
vars_ale, vars_epi, _, _ = w.vars_definition_weighted()
pq = PolyUQ(vars_ale, vars_epi, dim_ex='cartesian')
pq.sample_qmc(N_mcs_ale=100, N_mcs_epi=200, seed=1509)
vb = pq.inp_samp_prim['v_b'].values[:100]
pq.vars_inc[0].freeze(2.28)  # c_vb mid-range
means = []
for val in np.linspace(5.618, 6.0, 5):
    pq.vars_inc[1].freeze(val)
    wgt = pq.probabilities_imp(0)
    wgt /= wgt.sum()
    means.append(float(np.sum(wgt * vb)))
print('lamda_vb 5.618 -> 6.0: weighted-mean v_b =', np.round(means, 4))
print('strictly increasing:', bool(np.all(np.diff(means) > 0)))
print()
print('VALIDATION DONE')
