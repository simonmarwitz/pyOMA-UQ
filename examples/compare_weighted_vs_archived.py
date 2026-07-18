"""Deliverable 1 — comparison figures: weighted-covariance SSI (Phase 1
weighted run) vs. the original per-sample-OMA archived study, matched to the
plotting style of UQ_OMA.ipynb. Reads only existing artifacts persisted by
run_weighted_uq_pipeline / the original study — no recomputation.

Per matched mode (frequency f, damping ratio d):
  1. stacked focal-interval bars (archived vs. weighted, side by side, not
     overlaid — different hypercube counts, 144 vs. 72).
  2. belief/plausibility p-box overlay on a shared bin grid (the comparable
     view): weighted solid, archived dashed.
Plus the unit-disk pole-cluster diagram (UQ_OMA.ipynb cell 91) on the
weighted pole_db, and a combined per-mode p-box grid.
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, '/home/womo1998/dev/uq_oma_a/oma_uq')

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from examples import UQ_OMA_weighted as w
from polyuq import compute_belief, plot_focals
from polyuq.plotting import get_pcd

RESULT = Path('/home/womo1998/Projects/uq_oma_a/weighted_dev')
ARCHIVE = Path('/home/womo1998/Projects/uq_oma_a/estimations')
FIGDIR = RESULT / 'review' / 'plots'
FIGDIR.mkdir(parents=True, exist_ok=True)

# reference-mode arcs, UQ_OMA.ipynb cell 91 (verbatim)
REF_MODES_HZ = [0.15696927, 0.16334915, 0.17907126, 0.1797327, 0.3154164,
                0.33520363, 0.58013491, 0.6038054, 1.19951567, 1.2494742,
                2.01113836, 2.0974408, 3.03492814, 3.16726121, 4.2853773,
                4.47227961, 5.75791544, 6.00769074, 7.44085249, 7.76186644,
                9.32672286, 9.72715841, 11.41472027, 11.90226129,
                13.70563292, 14.28735387, 16.19654751, 16.87903915]

XLABEL = {'f': r'Frequency $f$ [\si{\hertz}]',
          'd': r'Damping ratio $\zeta$ [\si{\percent}]'}


def savefig(fig, name):
    # backend='pgf' (not the default usetex/dvips pipeline) so get_pcd's
    # pgf.preamble (siunitx, unicode-math \mathfrak mapping) actually applies —
    # matches notebook cell 39's convention, not cell 91's (which relied on a
    # stray matplotlib.rc('text.latex', preamble=...) call from an earlier
    # notebook cell that has no equivalent in a standalone script).
    fig.savefig(FIGDIR / f'{name}.pdf', backend='pgf')
    fig.savefig(FIGDIR / f'{name}.png', backend='pgf')


def plot_pole_cluster(pole_db):
    """Unit-disk pole-cluster diagram, UQ_OMA.ipynb cell 91, on the weighted pole_db."""
    f = np.concatenate([e['f'] for e in pole_db])
    d = np.concatenate([e['d'] for e in pole_db])
    # mirrors cell 89's hard filters (drop non-physical / out-of-range poles)
    ok = np.isfinite(f) & np.isfinite(d) & (f > 0.1) & (f < 5) & (d < 20)
    omega = f[ok] * 2 * np.pi
    zeta = d[ok] / 100
    mu = -zeta * omega + 1j * omega * np.sqrt(1 - zeta ** 2)
    lamda = np.exp(mu / w.NYQ_MAX)

    # cell 91's 5000x2500 bins assumed the original study's 21e6-pole ensemble;
    # this weighted run has ~n points, several orders of magnitude fewer, so
    # keep the same per-unit-length bin density ratio (real:imag = 2:1 over
    # the plotted (-1,1)x(0,1) range) but scale the count down to ~sqrt(n)
    # bins/unit so density is actually visible.
    n_bins_imag = max(20, int(np.sqrt(lamda.size)))
    bins = (np.linspace(-1.0, 1.0, 2 * n_bins_imag), np.linspace(0, 1.0, n_bins_imag))

    with matplotlib.rc_context(get_pcd('print')):
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax.spines['left'].set_position(('data', 0))
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.hist2d(lamda.real, lamda.imag, density=True, bins=bins,
                  cmap='Greys', norm='log', rasterized=True)
        _zeta = np.linspace(0, 1)
        for f_ref in REF_MODES_HZ:
            _omega = f_ref * 2 * np.pi
            _mu = -_zeta * _omega + 1j * _omega * np.sqrt(1 - _zeta ** 2)
            _lamda = np.exp(_mu / w.NYQ_MAX)
            ax.plot(_lamda.real, _lamda.imag, c='lightgrey', ls='solid', alpha=0.2)
            ax.annotate(f'${f_ref:1.2f} \\si{{\\hertz}}$',
                        (_lamda[0].real, _lamda[0].imag),
                        (_lamda[0].real * 1.05, _lamda[0].imag * 1.05))
        ax.set_xlabel(r'$\mathfrak{R}(\lambda)$')
        ax.set_ylabel(r'$\mathfrak{I}(\lambda)$')
        ax.set_xlim((-0.1, 1.1))
        ax.set_ylim((-0.1, 1.1))
        savefig(fig, 'compare_cluster_example')
        plt.close(fig)


def load_archived(quantity, k):
    npz = ARCHIVE / f'{quantity}_sc-{k}' / 'polyuq_avg_inc.npz'
    with np.load(npz, allow_pickle=True) as arr:
        stats = arr['self.focals_stats'][0]
        mass = np.ravel(arr['self.focals_mass'])
    ok = np.isfinite(stats).all(axis=1) & np.isfinite(mass) & (mass > 0)
    return stats[ok], mass[ok]


def load_weighted(quantity, label):
    with np.load(RESULT / f'weighted_focals_{label}_{quantity}.npz') as arr:
        return arr['imp_foc'], arr['imp_hyc_mass'], float(arr['median'])


def match_label_to_archive(label, n_archived=14):
    """Nearest archived f_sc-k by weighted midpoint, mirroring
    validate_weighted_run.py:80-95 (same k reused for d_sc-k, same run)."""
    _, _, med_f = load_weighted('f', label)
    best_k, best_diff = None, np.inf
    for k in range(n_archived):
        npz = ARCHIVE / f'f_sc-{k}' / 'polyuq_avg_inc.npz'
        if not npz.exists():
            continue
        stats, mass = load_archived('f', k)
        if len(mass) == 0:
            continue
        mid = np.average(np.mean(stats, axis=1), weights=mass)
        diff = abs(mid - med_f)
        if diff < best_diff:
            best_k, best_diff = k, diff
    return best_k, best_diff, med_f


def compare_mode(quantity, label, k):
    arch_stats, arch_mass = load_archived(quantity, k)
    w_foc, w_mass, _ = load_weighted(quantity, label)
    xlabel = XLABEL[quantity]

    # 1. stacked focal intervals, side by side (not overlaid: 144 vs 72 hyc)
    with matplotlib.rc_context(get_pcd('print')):
        fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True)
        plot_focals(arch_stats, arch_mass, ax0)
        ax0.set_title(f'archived ({quantity}\\_sc-{k})')
        ax0.set_xlabel(xlabel)
        ax0.set_ylabel('Cumulative Mass [-]')
        ax0.set_ylim((0, 1))
        plot_focals(w_foc, w_mass, ax1)
        ax1.set_title(f'weighted (cluster {label})')
        ax1.set_xlabel(xlabel)
        fig.subplots_adjust(top=0.9, bottom=0.16, left=0.1, right=0.97, wspace=0.08)
        savefig(fig, f'compare_focals_{label}_{quantity}')
        plt.close(fig)

    # 2. belief-plausibility p-box overlay, shared bin grid
    lo = min(arch_stats.min(), w_foc.min())
    hi = max(arch_stats.max(), w_foc.max())
    bins = np.linspace(lo, hi, 200)
    _, bel_a, pl_a, _ = compute_belief(arch_stats, arch_mass, cumulative=True, bins=bins)
    _, bel_w, pl_w, _ = compute_belief(w_foc, w_mass, cumulative=True, bins=bins)

    with matplotlib.rc_context(get_pcd('print')):
        fig, ax = plt.subplots()
        _pbox_axes(ax, bins, bel_a, pl_a, bel_w, pl_w, xlabel)
        ax.legend(fontsize=8)
        savefig(fig, f'compare_pbox_{label}_{quantity}')
        plt.close(fig)

    # per-mode overlap metrics
    disagreement = float(np.mean(np.abs(bel_w - bel_a)))
    w_env = (w_foc.min(), w_foc.max())
    a_env = (arch_stats.min(), arch_stats.max())
    inter = max(0.0, min(w_env[1], a_env[1]) - max(w_env[0], a_env[0]))
    union = max(w_env[1], a_env[1]) - min(w_env[0], a_env[0])
    envelope_overlap = inter / union if union > 0 else np.nan

    return dict(bins=bins, bel_a=bel_a, pl_a=pl_a, bel_w=bel_w, pl_w=pl_w,
                disagreement=disagreement, envelope_overlap=envelope_overlap)


def _pbox_axes(ax, bins, bel_a, pl_a, bel_w, pl_w, xlabel):
    ax.step(bins, bel_a, where='post', ls='dashed', color='k', lw=0.8, label='archived (bel)')
    ax.fill_between(bins, bel_a, pl_a, step='post', alpha=0.3, color='grey',
                     label='archived (bel-pl)')
    ax.step(bins, bel_w, where='post', ls='solid', color='k', lw=0.8, label='weighted (bel)')
    ax.fill_between(bins, bel_w, pl_w, step='post', alpha=0.3, color='tab:blue',
                     label='weighted (bel-pl)')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Cumulative Probability [-]')
    ax.set_ylim((0, 1))
    ax.set_xlim((bins.min(), bins.max()))


def compare_cdf(quantity, label, k):
    '''
    Deliverable 2 comparison: overlay the weighted parametric aleatory CDF
    (weighted_cdf_{label}_{quantity}.npz, run_weighted_cdf_reconstruction)
    against the archived study's empirical aleatory CDF
    (polyuq_cdf_inc.npz, stat_fun_cdf) — never against a confidence
    interval (plan D2 rationale). Both share the same target_probabilities
    grid (np.linspace(0, 1, 40)) by construction, so index i in one
    dataset's 40 probability levels is directly comparable to index i in
    the other's.

    Two side-by-side plausibility heatmaps (probability vs. value), each on
    its OWN value-bin range (as polyuq.aggregate_mass computes by default) —
    not a shared axis, mirroring D1's side-by-side (not overlaid) choice
    for the focal-interval panels: the weighted parametric envelope can be
    dramatically wider than the archived empirical one for modes whose
    per-epistemic-sample sigma_pop is large relative to the mean (see
    module-level caveat below), so forcing a shared axis would make the
    archived panel's structure invisible.
    '''
    from polyuq import aggregate_mass

    wfile = RESULT / f'weighted_cdf_{label}_{quantity}.npz'
    afile = ARCHIVE / f'{quantity}_sc-{k}' / 'polyuq_cdf_inc.npz'
    if not (wfile.exists() and afile.exists()):
        return None

    with np.load(wfile) as arr:
        w_foc, w_mass = arr['imp_foc'], arr['imp_hyc_mass']
        target_probabilities = arr['target_probabilities']
    with np.load(afile, allow_pickle=True) as arr:
        a_foc = arr['self.focals_stats']
        a_mass = np.ravel(arr['self.focals_mass'])

    bel_w, pl_w, _, bins_w = aggregate_mass(w_foc, w_mass, 10, False)
    bel_a, pl_a, _, bins_a = aggregate_mass(a_foc, a_mass, 10, False)

    xlabel = XLABEL[quantity]
    with matplotlib.rc_context(get_pcd('print')):
        fig, (ax0, ax1) = plt.subplots(1, 2)
        m0 = ax0.pcolormesh(bins_a, target_probabilities, pl_a, cmap='Greys')
        ax0.set_title(f'archived ({quantity}\\_sc-{k})')
        ax0.set_xlabel(xlabel)
        ax0.set_ylabel('Cumulative Probability [-]')
        m1 = ax1.pcolormesh(bins_w, target_probabilities, pl_w, cmap='Greys')
        ax1.set_title(f'weighted parametric (cluster {label})')
        ax1.set_xlabel(xlabel)
        fig.colorbar(m1, ax=[ax0, ax1], label='Plausibility [-]', shrink=0.8)
        savefig(fig, f'compare_cdf_{label}_{quantity}')
        plt.close(fig)
    return dict(bins_w=bins_w, bins_a=bins_a)


def make_grid(results, quantity, labels):
    n = len(results)
    ncols = min(4, n) or 1
    nrows = int(np.ceil(n / ncols))
    with matplotlib.rc_context(get_pcd('print')):
        fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                                  figsize=(ncols * 2.4, nrows * 2.0))
        for ax in axes.flat:
            ax.set_visible(False)
        for ax, label, res in zip(axes.flat, labels, results):
            ax.set_visible(True)
            _pbox_axes(ax, res['bins'], res['bel_a'], res['pl_a'],
                       res['bel_w'], res['pl_w'], '')
            ax.set_ylabel('')
            ax.set_title(f'cluster {label}', fontsize=8)
        fig.suptitle(XLABEL[quantity])
        fig.subplots_adjust(top=0.9, bottom=0.08, left=0.06, right=0.98,
                             hspace=0.5, wspace=0.3)
        savefig(fig, f'compare_pbox_grid_{quantity}')
        plt.close(fig)


def main():
    with open(RESULT / 'pole_db.pkl', 'rb') as fh:
        pole_db = pickle.load(fh)
    plot_pole_cluster(pole_db)

    labels = sorted(int(p.stem.split('_')[2])
                     for p in RESULT.glob('weighted_focals_*_f.npz'))

    report = []
    grids = {'f': ([], []), 'd': ([], [])}
    for label in labels:
        k, diff, med_f = match_label_to_archive(label)
        rel_off = diff / med_f * 100 if med_f else np.inf
        if k is None or rel_off > 5.0:
            print(f'cluster {label}: no close archive match '
                  f'(nearest f_sc-{k}, off {rel_off:.1f}%), skipping')
            continue
        for quantity in ('f', 'd'):
            wfile = RESULT / f'weighted_focals_{label}_{quantity}.npz'
            afile = ARCHIVE / f'{quantity}_sc-{k}' / 'polyuq_avg_inc.npz'
            if not (wfile.exists() and afile.exists()):
                continue
            res = compare_mode(quantity, label, k)
            report.append((label, quantity, k, res['disagreement'], res['envelope_overlap']))
            grids[quantity][0].append(res)
            grids[quantity][1].append(label)

            # verification (plan §Verification/D1): masses are proper mass
            # functions and matched clusters' envelopes actually overlap
            w_foc, w_mass, _ = load_weighted(quantity, label)
            arch_stats, arch_mass = load_archived(quantity, k)
            assert np.isclose(w_mass.sum(), 1.0, atol=1e-8), \
                f'weighted mass does not sum to 1 for cluster {label}/{quantity}: {w_mass.sum()}'
            assert np.isclose(arch_mass.sum(), 1.0, atol=1e-8), \
                f'archived mass does not sum to 1 for {quantity}_sc-{k}: {arch_mass.sum()}'
            assert res['envelope_overlap'] > 0.0, \
                f'weighted/archived envelopes do not overlap for cluster {label}/{quantity}'

            # Deliverable 2: parametric aleatory CDF vs. archived empirical
            # CDF, wherever run_weighted_cdf_reconstruction has been run
            # for this cluster (weighted_cdf_{label}_{quantity}.npz)
            if compare_cdf(quantity, label, k) is not None:
                print(f'  cluster {label}/{quantity}: CDF comparison '
                      f'-> compare_cdf_{label}_{quantity}.{{pdf,png}}')

    for quantity in ('f', 'd'):
        results, glabels = grids[quantity]
        if results:
            make_grid(results, quantity, glabels)

    print(f'{"cluster":>7s} {"qty":>3s} {"arch k":>6s} {"mean|dBel|":>10s} {"env overlap":>11s}')
    for label, quantity, k, dis, ov in report:
        print(f'{label:7d} {quantity:>3s} {k:6d} {dis:10.4f} {ov:11.3f}')
    print(f'\nfigures written to {FIGDIR}')


if __name__ == '__main__':
    main()
