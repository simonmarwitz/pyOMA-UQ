"""Figures and measurements for the pyOMA documentation page.

The pyOMA docs ship only rendered PNGs (the existing PoGER application page does
the same), because reproducing them needs this package *and* the measurement
tree. This module is the generator that produced them.

Modes::

    python -m pyoma_uq.studies.make_doc_figs figs   <result_dir> --out-dir DOC/_static
    python -m pyoma_uq.studies.make_doc_figs pbox   <result_dir> --out-dir <pbox_dir>
    python -m pyoma_uq.studies.make_doc_figs cost   --out-dir <dir>

``figs`` renders everything that is cheap (seconds to a few minutes). ``pbox``
reconstructs the aleatory p-box and takes hours -- run it first, in the
background; ``figs`` picks up its output if present and skips that one figure
otherwise. The two weightings of ``pbox`` are independent and each is serial,
so on a multi-core box run them as two processes::

    ... pbox <result_dir> --out-dir <d> --weighting build   &
    ... pbox <result_dir> --out-dir <d> --weighting posthoc &

``cost`` measures the per-sample identification cost on the machine it runs on.
Run it on an *idle* machine and with ``OMP_NUM_THREADS=1``: the numbers are
meant to be core-seconds, and contended timings are worse than no timings.

Rendering differs from :mod:`make_experimental_figs`, which targets a 150 mm
journal column: web figures are wider, sans-serif, and rendered at 150 dpi
without LaTeX, so they stay legible inside a documentation page.
"""
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Okabe-Ito blue / vermillion, the pair validated for this study (worst
# adjacent deltaE 21.9 protan, 31.2 normal vision). Identity is additionally
# carried by marker shape and a legend, never by colour alone.
BUILD, POSTHOC = '#0072B2', '#D55E00'
#: Third categorical hue, for the per-setup block levels. The three together
#: pass the validator (worst adjacent deltaE 11.0 deutan, 25.8 normal vision).
SETUP_COLOURS = (BUILD, POSTHOC, '#009E73')
SETUP_MARKERS = ('o', 's', '^')
MARKERS = {'build': 'o', 'posthoc': 's'}

#: The two modes the p-box is shown for: the best-covered low-band and
#: high-band mode (coverage 0.67 and 0.68), so the shape is representative
#: rather than an artefact of a sparsely sampled mode.
PBOX_MODES = (19, 21)
PBOX_LEVELS = 11


def web_rc(width=7.2, height=None):
    """rcParams for a documentation-page figure.

    Deliberately not :func:`polyuq.plotting.get_pcd`, which sizes for a 150 mm
    print column and switches on LaTeX when it is available -- both wrong here:
    a doc page is wider, and the LaTeX serif renders small and thin against the
    theme's sans-serif body text.
    """
    if height is None:
        height = width / 1.618
    return {
        'figure.figsize': (width, height),
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'text.usetex': False,
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.labelsize': 10,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.linewidth': 0.6,
        'legend.frameon': False,
    }


def _despine(ax, grid_axis='y'):
    ax.grid(True, axis=grid_axis, lw=0.3, color='0.85', zorder=0)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)


# ── the aleatory input: observed block levels ────────────────────────────────

def fig_levels(out_path):
    """The 18 observed block levels and the proposal density fitted to them.

    This is what an aleatory variable looks like when nothing is simulated: the
    realisations are measured, so their density is a property of the weather
    during the campaign and has to be *estimated*, not chosen.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pyoma_uq.studies import UQ_OMA_experimental as ex

    ex.install_loader()
    levels, offsets = ex.block_levels(ex.N_SEGMENTS)
    proposal, (s, scale) = ex.fit_level_proposal(levels)

    # offsets are cumulative block boundaries and already start at 0, so the
    # per-setup slices are consecutive pairs of it
    bounds = np.asarray(offsets, dtype=int)
    slices = list(zip(bounds[:-1], bounds[1:]))

    with matplotlib.rc_context(web_rc(height=3.2)):
        fig, ax = plt.subplots()
        grid = np.linspace(0, levels.max() * 1.35, 400)
        density = proposal.pdf(grid) / 1e3
        ax.plot(grid * 1e3, density, color='0.35', lw=1.6, zorder=2,
                label=f'fitted proposal: lognormal($s$ = {s:.2f}, '
                      f'median = {scale * 1e3:.2f})')

        # the observations as three labelled rows below the density, at fixed
        # fractions of the axis so nothing is clipped against the spine
        top = density.max() * 1.45
        for j, (lo, hi) in enumerate(slices):
            ax.plot(levels[lo:hi] * 1e3,
                    np.full(hi - lo, top * (0.10 + 0.07 * j)),
                    SETUP_MARKERS[j % len(SETUP_MARKERS)],
                    color=SETUP_COLOURS[j % len(SETUP_COLOURS)], ms=5, mew=0,
                    ls='none', zorder=3,
                    label=f'setup {j + 1}, {hi - lo} blocks')

        ax.set_xlabel('block response level at the reference channels '
                      '[mm/s$^2$]')
        ax.set_ylabel('density [s$^2$/mm]')
        ax.set_ylim(0, top)
        ax.set_xlim(left=0)
        _despine(ax)
        ax.legend(loc='upper right')
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


# ── pole-to-mode assignment, drawn by pyOMA's own compare_modes ──────────────

def _polish_mac(fig, title='', tick_stride=2):
    """Make ``compare_modes``' MAC matrix legible inside a doc page.

    The function draws the matrix but sets no axis labels and no colour bar,
    and at this campaign's pole count the column labels collide. Only
    presentation is touched here -- the matrix, the transparency mask and the
    red pairing crosses are exactly what ``compare_modes`` produced.
    """
    ax = fig.axes[0]
    ax.set_xlabel('identified poles')
    ax.set_ylabel('reference modes')
    ax.xaxis.set_label_position('top')
    for label in ax.get_xticklabels()[1::tick_stride]:
        label.set_visible(False)
    ax.tick_params(labelsize=6)
    bar = fig.colorbar(ax.images[0], ax=ax, fraction=0.025, pad=0.02)
    bar.set_label('MAC', fontsize=9)
    bar.ax.tick_params(labelsize=8)
    if title:
        ax.set_title(title, fontsize=10, pad=30)


def _polish_fd(fig, title=''):
    """Label the frequency-damping panel and pull its statistics box inside.

    ``compare_modes`` annotates at figure fraction (0.55, 0.7), which overflows
    the right edge at any aspect ratio narrower than its default.
    """
    ax = fig.axes[0]
    ax.set_xlabel('natural frequency [Hz]')
    ax.set_ylabel('damping ratio [%]')
    # the statistics box is an Annotation on the axes, positioned in *figure*
    # fraction coordinates, so it lands off the right edge at any aspect ratio
    # narrower than compare_modes' default
    for text in list(ax.texts) + list(fig.texts):
        text.set_position((0.40, 0.62))
        text.set_fontsize(7.5)
    _despine(ax, grid_axis='both')
    if title:
        ax.set_title(title, fontsize=10)


def _save_compare_modes(call, out_paths, polishers=(), titles=(), width=6.4,
                        height=None):
    """Run something that calls ``compare_modes`` and save the figures it drew.

    ``compare_modes`` draws into pyplot figures it creates itself (``matshow``
    for the MAC matrix, then a plain ``figure`` for the frequency-damping
    plane) and returns only the pairing indices, so the figures are collected
    afterwards by number rather than by return value.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.close('all')
    titles = list(titles) + [''] * len(out_paths)
    polishers = list(polishers) + [None] * len(out_paths)
    with matplotlib.rc_context(web_rc(width=width, height=height)):
        result = call()
        nums = plt.get_fignums()
        if len(nums) < len(out_paths):
            raise RuntimeError(f'compare_modes drew {len(nums)} figures, '
                               f'expected at least {len(out_paths)}')
        for num, out_path, polish, title in zip(nums, out_paths, polishers,
                                                titles):
            fig = plt.figure(num)
            fig.set_size_inches(*matplotlib.rcParams['figure.figsize'])
            if polish is not None:
                polish(fig, title=title)
            fig.savefig(out_path)
        plt.close('all')
    return result


def fig_pairing_baseline(out_mac, out_fd, order=130, band='low'):
    """Harness validation: archived reference modes vs. a fresh identification.

    ``reproduce_baseline(plot=True)`` re-runs the published analysis parameters
    through the current implementation and hands both mode sets to
    ``compare_modes``, which is exactly the diagnostic a practitioner wants
    before trusting any uncertainty propagated through the same code.
    """
    from pyoma_uq.studies import UQ_OMA_experimental as ex

    ex.install_loader()
    table = _save_compare_modes(
        lambda: ex.reproduce_baseline(band=band, orders=[order],
                                      n_segments=1, plot=True),
        (out_mac, out_fd),
        polishers=(_polish_mac, _polish_fd),
        titles=(f'reference modes vs. poles re-identified at order {order}',
                ''),
        width=8.0, height=4.0)
    logger.info('baseline reproduction at order %d:\n%s', order,
                table.to_string(index=False))
    return table


def fig_pairing_sample(out_path, result_dir, n_epi=None, weighting='build'):
    """The same comparison for *one epistemic sample* of the study.

    Where the baseline check pairs a deliberately faithful reproduction, an
    epistemic sample uses a sampled band, lag count and model order, so only a
    subset of the reference modes is resolved at all. That partial pairing is
    what the coverage panel of the results figure aggregates.
    """
    from pyoma_uq.studies import UQ_OMA_experimental as ex

    ex.install_loader()
    baseline = ex.load_baseline_modes()
    poly_uq, offsets = restore_sampling(result_dir)

    if n_epi is None:
        n_epi = _pick_illustrative_sample(result_dir, weighting)
    params = ex.sample_parameters(poly_uq, n_epi)
    ok, reason, resolved = ex.feasible(**params)
    if not ok:
        raise ValueError(f'epistemic sample {n_epi} is infeasible: {reason}')

    K = int(offsets[-1])
    weights = ex.split_weights(np.full(K, 1.0 / K), offsets)
    modal_data, order = ex.identify(params, resolved, weights, weighting,
                                    baseline)

    f_all = modal_data.modal_frequencies[order]
    d_all = modal_data.modal_damping[order]
    phi_all = modal_data.mode_shapes[:, :, order]
    keep = ex.physical_poles(f_all, d_all, params['highpass'],
                             params['lowpass'])

    # damping is already in percent on both sides (a lightly damped mast trips
    # compare_modes' `max(d) <= 1` "not in percent?" heuristic, harmlessly)
    from pyOMA.core.PostProcessingTools import compare_modes
    _save_compare_modes(
        lambda: compare_modes(baseline['f'], baseline['d'], baseline['phi'],
                              f_all[keep], d_all[keep], phi_all[:, keep]),
        (out_path,),
        polishers=(_polish_mac,),
        titles=(f'one epistemic sample: band '
                f'{params["highpass"]:.2f}-{params["lowpass"]:.2f} Hz, '
                f'$p$ = {resolved["num_block_rows"]}, '
                f'$n_\\mathrm{{ord}}$ = {order}',),
        width=8.0, height=4.0)
    return n_epi, params, resolved


def _pick_illustrative_sample(result_dir, weighting):
    """A feasible sample with a mid-range block count that pairs several modes.

    Picked from the stored run rather than at random so the figure is
    reproducible and is not accidentally a degenerate cell.
    """
    import pickle

    with open(Path(result_dir) / weighting / 'stat_db.pkl', 'rb') as fh:
        stat_db = pickle.load(fh)
    best, best_n = None, -1
    for entry in stat_db:
        if 'infeasible' in entry or not entry['keys']:
            continue
        if not 60 <= entry.get('num_block_rows', 0) <= 130:
            continue
        if len(entry['keys']) > best_n:
            best, best_n = entry['n_epi'], len(entry['keys'])
    if best is None:
        raise RuntimeError('no feasible sample in the stored run')
    logger.info('illustrative epistemic sample %d pairs %d reference modes',
                best, best_n)
    return best


# ── results ──────────────────────────────────────────────────────────────────

def fig_envelope_web(summary, out_path):
    """Focal envelope per mode, as a deviation from the reference value.

    A dot-with-range plot rather than bars: the quantity is an interval around
    a reference, not a magnitude from zero, so bars would imply the wrong
    baseline. Zero is the line every interval should straddle.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with matplotlib.rc_context(web_rc(width=7.2, height=5.4)):
        fig, ax = plt.subplots()
        y = np.arange(len(summary))
        for weighting, colour, sign, label in (
                ('build', BUILD, +1, 'build-time weighting'),
                ('posthoc', POSTHOC, -1, 'post-hoc reweighting')):
            lo = (summary[f'{weighting}_f_lo'] / summary['f_2019'] - 1) * 100
            hi = (summary[f'{weighting}_f_hi'] / summary['f_2019'] - 1) * 100
            yy = y + sign * 0.19
            ax.hlines(yy, lo, hi, color=colour, lw=2.2, label=label)
            ax.plot(lo, yy, MARKERS[weighting], color=colour, ms=4, mew=0)
            ax.plot(hi, yy, MARKERS[weighting], color=colour, ms=4, mew=0)

        ax.axvline(0, color='0.25', lw=0.9, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels([f'{f:.3f}' for f in summary['f_2019']])
        ax.set_ylabel('reference natural frequency [Hz]')
        ax.set_xlabel('deviation of the focal envelope from the reference value [%]')
        ax.set_ylim(-0.8, len(summary) - 0.2)
        ax.invert_yaxis()
        _despine(ax, grid_axis='x')
        ax.legend(loc='lower right')
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def fig_width_coverage_web(summary, out_path):
    """Envelope width and per-mode coverage -- two panels, never two y-axes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with matplotlib.rc_context(web_rc(width=7.2, height=4.4)):
        fig, axes = plt.subplots(2, 1, sharex=True,
                                 gridspec_kw={'height_ratios': [2, 1]})
        x = np.arange(len(summary))
        for weighting, colour, label in (('build', BUILD, 'build-time'),
                                         ('posthoc', POSTHOC, 'post-hoc')):
            rel = summary[f'{weighting}_f_w'] / summary['f_2019'] * 100
            axes[0].plot(x, rel, MARKERS[weighting], color=colour, ms=5,
                         mew=0, ls='none', label=label)
        axes[0].set_ylabel('mass-weighted width [% of $f$]')
        # a width is a magnitude, so the axis belongs at zero -- and a non-zero
        # bottom clipped the smallest markers against the spine
        top = np.nanmax(np.concatenate([
            (summary['build_f_w'] / summary['f_2019']).values,
            (summary['posthoc_f_w'] / summary['f_2019']).values])) * 100
        axes[0].set_ylim(0, top * 1.3)
        axes[0].legend(loc='upper right', ncol=2)

        axes[1].plot(x, summary['coverage'] * 100, 'D', color='0.35', ms=4,
                     mew=0, ls='none')
        axes[1].set_ylabel('coverage [%]')
        axes[1].set_ylim(0, 100)

        for ax in axes:
            _despine(ax)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f'{f:.2f}' for f in summary['f_2019']],
                                rotation=90)
        axes[1].set_xlabel('reference natural frequency [Hz]')
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


# ── aleatory p-box ───────────────────────────────────────────────────────────

def restore_sampling(result_dir, n_epi_total=1500, seed=1509):
    """Rebuild the PolyUQ instance the stored run was propagated on.

    ``sample_qmc`` draws a Halton sequence through ``scipy.stats.qmc``, which is
    not reproducible across scipy *versions* -- so this re-sample is only valid
    on a machine whose scipy matches the one that produced the run. The stored
    per-sample parameters are the ground truth for that, and are asserted
    against here: a scipy upgrade then fails loudly instead of silently moving
    the epistemic samples out from under the stored ``stat_db``.
    """
    from polyuq import PolyUQ
    from pyoma_uq.studies import UQ_OMA_experimental as ex

    ex.install_loader()
    vars_ale, vars_epi, levels, offsets = ex.vars_definition_experimental()
    poly_uq = PolyUQ(vars_ale, vars_epi, dim_ex='cartesian', path='fast_build')
    poly_uq.sample_qmc(N_mcs_ale=len(levels), N_mcs_epi=n_epi_total, seed=seed,
                       given_samples={'a_ref': levels})

    reference = pd.read_csv(Path(result_dir) / 'build' /
                            'feasibility_samples.csv')
    for n_epi in (0, 1, n_epi_total // 2, n_epi_total - 1):
        got = ex.sample_parameters(poly_uq, n_epi)
        want = reference.iloc[n_epi]
        for key in ('highpass', 'lowpass', 'nyq_rat', 'tau_max', 'm_lags',
                    'model_order'):
            if not np.isclose(float(got[key]), float(want[key]),
                              rtol=1e-9, atol=1e-9):
                raise RuntimeError(
                    f'the re-drawn epistemic samples do not match the stored '
                    f'run (sample {n_epi}, {key}: {got[key]} != {want[key]}). '
                    'scipy.stats.qmc has changed; the stored stat_db cannot be '
                    'reused with freshly drawn samples.')
    return poly_uq, offsets


def probability_levels(n_levels=PBOX_LEVELS):
    """Probability grid for the p-box.

    Deliberately interior: ``expand_parametric_cdf`` clips to (1e-4, 1-1e-4)
    before taking the quantile, so a grid that includes 0 and 1 evaluates the
    distribution 3.7 standard deviations into its tails. Interval-optimising
    *that* over 128 hypercubes picks the single worst cell and returns bounds
    two orders of magnitude wider than the mode itself -- measured on this
    study: a frequency p-box spanning -141 to +144 Hz. The tails are real but
    they are not what a p-box figure is for.
    """
    return np.linspace(0.02, 0.98, n_levels)


def add_cdf_rows(stat_db, n_levels=PBOX_LEVELS):
    """Expand each entry's per-pole mean and variance to aleatory CDF rows.

    The production run stored only the mean-value statistics, but the CDF rows
    are a *pure function* of what it did store -- per-pole ``f``, ``std_f``,
    ``d``, ``std_d`` and the effective sample size -- so the aleatory p-box can
    be reconstructed without re-identifying anything. This is exactly the
    computation the estimator would have done inline had it been given
    ``target_probabilities``.
    """
    from pyoma_uq.studies.UQ_OMA_weighted import expand_parametric_cdf

    probabilities = probability_levels(n_levels)
    n_expanded = 0
    for entry in stat_db:
        if 'infeasible' in entry or not len(entry['keys']):
            continue
        entry['cdf_f'] = expand_parametric_cdf(
            entry['f'], entry['std_f'], entry['n_eff'], probabilities,
            dist='normal')
        entry['cdf_d'] = expand_parametric_cdf(
            entry['d'], entry['std_d'], entry['n_eff'], probabilities,
            dist='lognormal')   # damping: strictly positive, right-skewed
        n_expanded += 1
    logger.info('expanded %d/%d stat_db entries to %d CDF levels',
                n_expanded, len(stat_db), n_levels)
    return probabilities


def run_pbox(result_dir, out_dir, modes=PBOX_MODES, n_levels=PBOX_LEVELS,
             weightings=('build', 'posthoc'), quantities=('f', 'd'),
             opt_meth='genetic'):
    """Reconstruct the aleatory p-box for a few modes from a finished run.

    Drives ``statistic_level`` + ``estimate_imp`` directly rather than through
    ``run_experimental_pipeline``: the pipeline would additionally recompute the
    mean-value focal sets that the finished run already holds, and it fixes the
    probability grid to ``linspace(0, 1, n_stat)`` -- see
    :func:`probability_levels` for why that grid is unusable here.
    """
    import pickle
    from pyoma_uq.studies import UQ_OMA_experimental as ex

    result_dir, out_dir = Path(result_dir), Path(out_dir)
    probabilities = probability_levels(n_levels)

    for weighting in weightings:
        target_dir = out_dir / weighting
        target_dir.mkdir(parents=True, exist_ok=True)

        poly_uq, _ = restore_sampling(result_dir)
        with open(result_dir / weighting / 'stat_db.pkl', 'rb') as fh:
            poly_uq.stat_db = pickle.load(fh)
        add_cdf_rows(poly_uq.stat_db, n_levels=n_levels)

        for mode in modes:
            for quantity in quantities:
                started = time.time()
                focals = []
                mass = None
                for i_stat in range(n_levels):
                    pq_cdf, hyc_rows = ex.statistic_level(
                        poly_uq, mode, field=f'cdf_{quantity}', i_stat=i_stat)
                    foc, _, _, _, _ = pq_cdf.estimate_imp(
                        interp_fun='rbf', opt_meth=opt_meth,
                        hyc_rows=hyc_rows)
                    focals.append(foc[0])
                    mass = pq_cdf.imp_hyc_mass
                out_path = target_dir / f'cdf_mode{mode}_{quantity}.npz'
                np.savez(out_path, imp_foc=np.stack(focals, axis=0),
                         imp_hyc_mass=mass,
                         target_probabilities=probabilities)
                logger.info('%s mode %d %s: %d levels in %.0f s -> %s',
                            weighting, mode, quantity, n_levels,
                            time.time() - started, out_path.name)
    return out_dir


def fig_pbox(out_path, pbox_dir, result_dir, modes=PBOX_MODES,
             quantities=('f', 'd')):
    """Interval-valued aleatory CDFs -- one panel per mode and quantity.

    A p-box is two bounding CDFs, not one curve: at every probability level the
    epistemic variables are interval-optimised, so the aleatory distribution is
    known only to lie between the bounds. The horizontal gap between them is
    the epistemic contribution and the slope is the aleatory one.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pyoma_uq.studies import UQ_OMA_experimental as ex

    from pyoma_uq.studies.make_experimental_figs import load_summary

    ex.install_loader()
    baseline = ex.load_baseline_modes()
    pbox_dir = Path(pbox_dir)
    summary = load_summary(result_dir)

    series = (('build', BUILD, 'build-time'), ('posthoc', POSTHOC, 'post-hoc'))

    def bounds(mode, quantity, weighting):
        path = pbox_dir / weighting / f'cdf_mode{mode}_{quantity}.npz'
        if not path.exists():
            return None
        with np.load(path) as archive:
            foc = archive['imp_foc']                  # (n_stat, n_hyc, 2)
            probabilities = archive['target_probabilities']
        # no unit conversion: damping is in percent everywhere in this study
        # -- baseline, stat_db and summary alike
        return (np.nanmin(foc[:, :, 0], axis=1),
                np.nanmax(foc[:, :, 1], axis=1), probabilities)

    with matplotlib.rc_context(web_rc(width=7.2, height=5.2)):
        fig, axes = plt.subplots(len(quantities), len(modes), sharey=True)
        axes = np.atleast_2d(axes)
        clipped = False
        for row, quantity in enumerate(quantities):
            for col, mode in enumerate(modes):
                ax = axes[row, col]
                reference = baseline[quantity][mode]
                # Plotted as a deviation from the reference on a symmetric-log
                # axis. A linear axis cannot show this p-box: the bounds are
                # tight at the median (a few 0.01 Hz) and grow by three orders
                # of magnitude toward either tail, because the aleatory sigma
                # itself spans four orders of magnitude across the epistemic
                # cells and the CDF-level surrogate has to interpolate that.
                # Symlog keeps the informative core linear and still shows where
                # the tails go. The same convention as the envelope figure.
                span_lo, span_hi = 0.0, 0.0
                for weighting, colour, label in series:
                    got = bounds(mode, quantity, weighting)
                    if got is None:
                        continue
                    lower, upper, probabilities = got
                    lower, upper = lower - reference, upper - reference
                    span_lo = min(span_lo, lower.min(), upper.min())
                    span_hi = max(span_hi, lower.max(), upper.max())
                    # plain lines, not steps: the bounds sample a *parametric*
                    # CDF on a probability grid, so a staircase would suggest a
                    # discreteness the underlying distribution does not have
                    ax.fill_betweenx(probabilities, lower, upper, color=colour,
                                     alpha=0.16, lw=0)
                    ax.plot(lower, probabilities, color=colour, lw=1.6,
                            label=label)
                    ax.plot(upper, probabilities, color=colour, lw=1.6)

                # the mean-value focal envelope of the same mode, as the scale
                # the p-box has to be judged against
                row_ = summary.loc[summary['mode'] == mode].iloc[0]
                env_lo = float(row_[f'build_{quantity}_lo']) - reference
                env_hi = float(row_[f'build_{quantity}_hi']) - reference
                ax.axvspan(env_lo, env_hi, color='0.55', alpha=0.20, lw=0,
                           zorder=0, label='mean-value focal envelope')

                linthresh = max(abs(env_lo), abs(env_hi), 1e-3)
                ax.set_xscale('symlog', linthresh=linthresh, linscale=0.9)
                # asymmetric limits: the damping bounds are one-sided (the
                # lognormal floors them at zero), and forcing symmetry would
                # spend half of every damping panel on an empty negative side
                left = span_lo * 1.6 if span_lo < -linthresh else -linthresh * 2
                right = span_hi * 1.6 if span_hi > linthresh else linthresh * 2
                ax.set_xlim(left, right)
                # explicit decade ticks: the default symlog locator crowds the
                # linear region and the +-linthresh labels collide with 0
                decades = [10.0 ** e for e in range(-3, 5)
                           if 10.0 ** e > linthresh * 1.5]
                ticks = [-t for t in reversed(decades)] + [0.0] + decades
                ax.set_xticks([t for t in ticks if left <= t <= right])
                ax.axvline(0.0, color='0.25', lw=0.9, ls=(0, (4, 2)), zorder=1,
                           label='reference value')
                unit = 'Hz' if quantity == 'f' else 'pp'
                ax.set_xlabel(f'deviation from reference [{unit}]')
                _despine(ax, grid_axis='both')
                if col == 0:
                    ax.set_ylabel('cumulative probability')
                if row == 0:
                    ax.set_title(f'mode at {baseline["f"][mode]:.3f} Hz',
                                 fontsize=10)
                ax.set_ylim(0, 1)
                ax.tick_params(labelsize=8)
        axes[0, 0].legend(loc='upper left', fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
    if clipped:
        logger.info('p-box axes are windowed on the central levels; the outer '
                    'bounds extend beyond them')
    return out_path


# ── computational cost, measured on the machine this runs on ─────────────────

def measure_cost(result_dir, out_csv, targets=(31, 40, 50, 62, 75),
                 model_order=60, weightings=('build', 'posthoc'),
                 band='low'):
    """Time one identification per target block-row count.

    The block-row count ``p`` is the cost driver: the first-order variance
    propagation solves a system per SVD triplet whose size grows with ``p``, so
    the cost is expected to grow roughly cubically in it and to be independent
    of the number of data blocks. Measuring the exponent is the point.

    Everything but the lag count is held fixed -- one band, one decimation
    factor, one model order -- so that ``p`` is the only thing that varies.
    Sweeping *sampled* parameter sets instead would confound ``p`` with the
    band and the decimated record length.

    Run with ``OMP_NUM_THREADS=1``: the numbers are meant to be core-seconds.
    Epistemic samples are independent, so a workstation runs one per core and
    the wall time per sample is this divided by the core count.

    Each row is appended to *out_csv* as soon as it is measured -- a sweep like
    this runs for hours and must not lose everything if it is interrupted.
    """
    from pyoma_uq.studies import UQ_OMA_experimental as ex

    ex.install_loader()
    baseline = ex.load_baseline_modes()
    _, offsets = restore_sampling(result_dir)
    K = int(offsets[-1])
    weights = ex.split_weights(np.full(K, 1.0 / K), offsets)

    cfg = ex.BANDS_2019[band]
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for target in targets:
        m_lags = 2 * int(target) + 1        # p = (m_lags - 1) // 2 = target
        params = {'highpass': cfg['highpass'], 'lowpass': cfg['lowpass'],
                  'nyq_rat': ex.FS_RAW / cfg['decimation'] / cfg['lowpass'],
                  'tau_max': 175.0, 'm_lags': m_lags,
                  # the order must stay below both m_lags - 2 and n_ref * p
                  'model_order': int(min(model_order, m_lags - 3,
                                         ex.N_REF * target))}
        ok, reason, resolved = ex.feasible(**params)
        if not ok:
            logger.warning('p=%d skipped: %s', target, reason)
            continue
        p = resolved['num_block_rows']

        started = time.time()
        ex.prepare_setups(params['highpass'], params['lowpass'],
                          resolved['decimation_factor'], m_lags,
                          ex.N_SEGMENTS, n_ref=ex.N_REF)
        t_prep = time.time() - started

        for weighting in weightings:
            started = time.time()
            modal_data, order = ex.identify(params, resolved, weights,
                                            weighting, baseline)
            elapsed = time.time() - started
            result = ex.assign_to_baseline(modal_data, order, params, baseline)
            rows.append({'num_block_rows': p, 'm_lags': m_lags,
                         'model_order': order, 'weighting': weighting,
                         'sampling_rate': resolved['sampling_rate'],
                         'prep_s': round(t_prep, 1),
                         'identify_s': round(elapsed, 1),
                         'n_paired': len(result['keys'])})
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f'p={p:4d} {weighting:8s} prep {t_prep:6.1f} s  '
                  f'identify {elapsed:8.1f} s  ({len(result["keys"])} paired)',
                  flush=True)
            del modal_data
    return pd.DataFrame(rows)


def fit_cost_exponent(table):
    """Least-squares exponent of ``identify_s ~ p**k``, per weighting."""
    out = {}
    for weighting, group in table.groupby('weighting'):
        if len(group) < 2:
            continue
        k, log_c = np.polyfit(np.log(group.num_block_rows),
                              np.log(group.identify_s), 1)
        out[weighting] = (float(k), float(np.exp(log_c)))
    return out


# ── driver ───────────────────────────────────────────────────────────────────

def render_all(result_dir, out_dir, pbox_dir=None, prefix='guyed_mast_uq_'):
    """Every figure the documentation page uses."""
    from pyoma_uq.studies.make_experimental_figs import load_summary

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    def target(name):
        return out_dir / f'{prefix}{name}.png'

    written.append(fig_levels(target('levels')))
    fig_pairing_baseline(target('pairing_mac'), target('pairing_fd'))
    written += [target('pairing_mac'), target('pairing_fd')]
    fig_pairing_sample(target('pairing_sample'), result_dir)
    written.append(target('pairing_sample'))

    summary = load_summary(result_dir)
    written.append(fig_envelope_web(summary, target('envelope')))
    written.append(fig_width_coverage_web(summary, target('width_coverage')))

    if pbox_dir is not None and Path(pbox_dir).exists():
        written.append(fig_pbox(target('pbox'), pbox_dir, result_dir))
    else:
        logger.warning('no p-box directory -- run the `pbox` mode first')
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('mode', choices=('figs', 'pbox', 'cost'))
    parser.add_argument('result_dir', type=Path, nargs='?')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--pbox-dir', type=Path)
    parser.add_argument('--n-levels', type=int, default=PBOX_LEVELS)
    parser.add_argument('--weighting', choices=('build', 'posthoc'),
                        action='append',
                        help='restrict the p-box to one weighting, so the two '
                             'can be run as separate processes in parallel '
                             '(they are independent and each is serial)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    if args.mode == 'pbox':
        run_pbox(args.result_dir, args.out_dir, n_levels=args.n_levels,
                 weightings=tuple(args.weighting or ('build', 'posthoc')))
    elif args.mode == 'cost':
        table = measure_cost(args.result_dir,
                             args.out_dir / 'cost_measurements.csv')
        print(table.to_string(index=False))
        for weighting, (k, c) in fit_cost_exponent(table).items():
            print(f'{weighting}: identify_s ~ {c:.3g} * p^{k:.2f}')
    else:
        for path in render_all(args.result_dir, args.out_dir,
                               pbox_dir=args.pbox_dir):
            print('wrote', path)


if __name__ == '__main__':
    main()
