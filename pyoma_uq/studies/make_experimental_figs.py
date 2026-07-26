"""Paper figures for the experimental study.

Two figures, each answering one question:

fig_envelope
    Does the polymorphic focal envelope contain the value the 2019 manual
    analysis reported? Plotted as *relative* deviation from that value, so all
    21 modes share one axis and "contains zero" is the thing the eye checks.
fig_width_coverage
    How wide are the envelopes, and how well is each mode resolved? Two
    stacked panels rather than one dual-axis plot.

Colours are the Okabe-Ito blue/vermillion pair (CVD-validated: worst adjacent
deltaE 21.9 protan, 31.2 normal vision); identity is additionally carried by
marker shape and a legend, never by colour alone.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BUILD, POSTHOC = '#0072B2', '#D55E00'
MARKERS = {'build': 'o', 'posthoc': 's'}


def load_summary(result_dir):
    """Focal envelopes per mode for both weightings, from the run outputs."""
    import re
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from pyoma_uq.studies import UQ_OMA_experimental as ex

    result_dir = Path(result_dir)
    baseline = ex.load_baseline_modes()
    coverage = pd.read_csv(result_dir / 'build' / 'mode_coverage.csv')
    labels = sorted(
        {int(re.search(r'mode(\d+)_', f.name).group(1))
         for f in (result_dir / 'build').glob('focals_*_f.npz')}
        & {int(re.search(r'mode(\d+)_', f.name).group(1))
           for f in (result_dir / 'posthoc').glob('focals_*_f.npz')})

    rows = []
    for label in labels:
        row = {'mode': label, 'f_2019': float(baseline['f'][label]),
               'band': str(baseline['band'][label]),
               'coverage': float(coverage.loc[coverage.label == label,
                                              'coverage'].iloc[0])}
        for weighting in ('build', 'posthoc'):
            for quantity in ('f', 'd'):
                path = result_dir / weighting / f'focals_mode{label}_{quantity}.npz'
                if not path.exists():
                    continue
                with np.load(path) as archive:
                    foc, mass = archive['imp_foc'], archive['imp_hyc_mass']
                ok = np.isfinite(foc).all(axis=1) & np.isfinite(mass)
                row[f'{weighting}_{quantity}_lo'] = float(np.min(foc[ok, 0]))
                row[f'{weighting}_{quantity}_hi'] = float(np.max(foc[ok, 1]))
                row[f'{weighting}_{quantity}_w'] = float(
                    np.average(foc[ok, 1] - foc[ok, 0], weights=mass[ok]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values('f_2019').reset_index(drop=True)


def fig_envelope(summary, out_path):
    """Relative deviation of the focal envelope from the 2019 point estimate.

    A dot-with-range plot rather than a bar chart: the quantity is an interval
    around a reference, not a magnitude from zero, so bars would imply the wrong
    baseline. Zero is the reference line every interval should straddle.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from polyuq.plotting import get_pcd

    with matplotlib.rc_context(get_pcd('print')):
        fig, ax = plt.subplots()
        y = np.arange(len(summary))
        offset = 0.19
        for weighting, colour, sign in (('build', BUILD, +1), ('posthoc', POSTHOC, -1)):
            lo = (summary[f'{weighting}_f_lo'] / summary['f_2019'] - 1) * 100
            hi = (summary[f'{weighting}_f_hi'] / summary['f_2019'] - 1) * 100
            yy = y + sign * offset
            ax.hlines(yy, lo, hi, color=colour, lw=2.0,
                      label=f'{weighting}-time weighting' if weighting == 'build'
                      else 'post-hoc reweighting')
            # 8px-equivalent end caps mark the interval bounds
            ax.plot(lo, yy, MARKERS[weighting], color=colour, ms=3.2, mew=0)
            ax.plot(hi, yy, MARKERS[weighting], color=colour, ms=3.2, mew=0)

        ax.axvline(0, color='0.25', lw=0.8, zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels([f'{f:.3f}' for f in summary['f_2019']])
        ax.set_ylabel('2019 natural frequency [Hz]')
        ax.set_xlabel(r'deviation of the focal envelope from the 2019 value [\%]'
                      if matplotlib.rcParams.get('text.usetex')
                      else 'deviation of the focal envelope from the 2019 value [%]')
        ax.set_ylim(-0.8, len(summary) - 0.2)
        ax.invert_yaxis()
        ax.grid(True, axis='x', lw=0.3, color='0.85', zorder=0)
        ax.set_axisbelow(True)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        ax.legend(loc='lower right', frameon=False)
        fig.tight_layout(pad=0.3)
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def fig_width_coverage(summary, out_path):
    """Envelope width and mode coverage -- two panels, never two y-axes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from polyuq.plotting import get_pcd

    with matplotlib.rc_context(get_pcd('print')):
        fig, axes = plt.subplots(2, 1, sharex=True,
                                 gridspec_kw={'height_ratios': [2, 1]})
        x = np.arange(len(summary))
        ax = axes[0]
        for weighting, colour in (('build', BUILD), ('posthoc', POSTHOC)):
            rel = summary[f'{weighting}_f_w'] / summary['f_2019'] * 100
            ax.plot(x, rel, MARKERS[weighting], color=colour, ms=4, mew=0,
                    ls='none',
                    label='build-time' if weighting == 'build' else 'post-hoc')
        ax.set_ylabel(r'mass-weighted width [\% of $f$]'
                      if matplotlib.rcParams.get('text.usetex')
                      else 'mass-weighted width [% of f]')
        # a width is a magnitude, so the axis belongs at zero -- and a non-zero
        # bottom clipped the smallest markers against the spine
        top = np.nanmax(np.concatenate([
            (summary['build_f_w'] / summary['f_2019']).values,
            (summary['posthoc_f_w'] / summary['f_2019']).values])) * 100
        ax.set_ylim(0, top * 1.28)          # headroom for the legend row
        ax.legend(loc='upper right', frameon=False, ncol=2)

        axes[1].plot(x, summary['coverage'] * 100, 'D', color='0.35', ms=3.2,
                     mew=0, ls='none')
        axes[1].set_ylabel(r'coverage [\%]'
                           if matplotlib.rcParams.get('text.usetex')
                           else 'coverage [%]')
        axes[1].set_ylim(0, 100)

        for ax in axes:
            ax.grid(True, axis='y', lw=0.3, color='0.85', zorder=0)
            ax.set_axisbelow(True)
            for spine in ('top', 'right'):
                ax.spines[spine].set_visible(False)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f'{f:.2f}' for f in summary['f_2019']],
                                rotation=90)
        axes[1].set_xlabel('2019 natural frequency [Hz]')
        fig.tight_layout(pad=0.3)
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('result_dir', type=Path)
    parser.add_argument('--out-dir', type=Path, default=Path('.'))
    parser.add_argument('--ext', default='pdf')
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = load_summary(args.result_dir)
    summary.to_csv(args.out_dir / 'summary_focals.csv', index=False)

    inside = sum(r[f'build_f_lo'] <= r['f_2019'] <= r[f'build_f_hi']
                 for _, r in summary.iterrows())
    print(f'{len(summary)} modes; 2019 value inside the build envelope: '
          f'{inside}/{len(summary)}')
    for fn in (fig_envelope, fig_width_coverage):
        path = args.out_dir / f'{fn.__name__.replace("fig_", "exp_")}.{args.ext}'
        print('wrote', fn(summary, path))


if __name__ == '__main__':
    main()
