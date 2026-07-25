'''
PolyUQ + OMA on *experimental* data — Schwabach guyed-mast multi-setup study.

The numerical companion study (UQ_OMA_weighted) evaluates a structural model
with numerical excitation and a sensing/acquisition model on an
aleatory x epistemic lattice, and hands the resulting correlation estimates to
a weighted OMA estimator. Here the model, the excitation and the acquisition
chain are replaced by *measured* signals, so only two stages remain:

    signal processing  ->  system identification  ->  statistic-level PolyUQ

Everything downstream of the identification (Incompleteness-conditioned
importance weights computed inside PolyUQ, build-time / post-hoc block
weighting, statistic-level interval optimization) is unchanged.

Three structural differences to the numerical study:

* **The aleatory realisations are the measured data blocks.** Each setup's
  record is split into ``n_segments`` non-overlapping blocks; the blocks of
  all setups are pooled into one aleatory ensemble of size
  ``K = sum_j n_b_j``. Their *observed* reference-channel response level plays
  the role of the numerical study's sampled wind speed ``v_b``. Because the
  levels are observed rather than drawn, the importance weights need the
  empirical proposal density (PolyUQ's ``UncertainVariable.proposal``), not
  the uniform proposal implied by ``sample_qmc``.
* **Pole -> mode assignment replaces clustering.** The 2019 study's manually
  selected modes are the baseline; each epistemic sample's poles are paired
  against them with ``pyOMA.core.PostProcessingTools.pair_modes`` directly
  inside the estimator, so the rows leaving the estimator are already keyed by
  a global baseline mode index. No pole database, no OPTICS pass.
* **The analysis band is epistemic.** The 2019 split into a low
  (0.1 .. 1.5 Hz) and a high (1.5 .. 8 Hz) band is an analyst's choice, so
  ``highpass``/``lowpass`` are Imprecision variables of a single unified study
  rather than two separate ones. Together with ``decimation_factor``,
  ``tau_max``, ``m_lags`` and ``model_order`` they are sampled independently
  and screened by an acceptance-rejection predicate, as in the original full
  study (UQ_OMA._stage3mapping's feasibility asserts); rejected samples
  contribute NaN rows, which ``estimate_imp`` tolerates by fitting its
  surrogates on the feasible subset.

Data: /home/womo1998/Projects/2019_Schwabach (override with $SCHWABACH_DIR).
'''
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCHWABACH_DIR = Path(os.environ.get(
    'SCHWABACH_DIR', '/home/womo1998/Projects/2019_Schwabach'))

FS_RAW = 128.0          # sampling rate of the .asc exports
DURATION = 1800.0       # s per ambient record
N_RAW = 230400          # time steps per ambient record

#: The three ambient setups of 14 June 2019 (2 reference + 4 roving sensor
#: pairs each). ``Messung_2`` deletes channels 6/7 (S19 overload) via its
#: ``setup_info``, so the merged DOF set is 26 rather than 28.
SETUPS = ('Messung_1', 'Messung_2', 'Messung_12')

#: Reference channels, in the channel numbering of the .asc columns:
#: 0/1 = R1_y/R1_x (156.78 m), 2/3 = R2_x/R2_y (113.10 m). The 2019 study used
#: all four (see ``N_REF``); restricting to the R2 pair alone is supported by
#: :func:`prepare_setups` but is not sampled, see
#: :func:`vars_definition_experimental`.
REF_CHANNELS_FULL = (0, 1, 2, 3)
REF_CHANNELS_R2 = (2, 3)

#: The 2019 analysis parameters per frequency band, used by
#: :func:`reproduce_baseline` and as the grounding of the Imprecision focal
#: sets. ``num_block_rows`` came from a modal-contribution sweep over
#: ``range(50, 600, 20)``.
BANDS_2019 = {
    'low': dict(highpass=0.1, lowpass=1.5, decimation=25, num_block_rows=210,
                archive='results_ssi_cov/low/no_cables',
                modal='modal_data_210.npz'),
    'high': dict(highpass=1.5, lowpass=8.0, decimation=6, num_block_rows=550,
                 archive='results_ssi_cov/high/no_cables',
                 modal='modal_data_350.npz'),
}


# ── measurement file loading ─────────────────────────────────────────────────

def load_asc(fname, **kwargs):
    '''
    Load one Schwabach ``.asc`` export as a ``(num_timesteps, num_channels)``
    array.

    Every line of these files starts with a tab, so a plain delimited read
    yields a leading all-NaN column. Left in place it survives filtering as
    NaN and only surfaces much later as "array must not contain infs or NaNs"
    inside the subspace SVD, so all-NaN columns are dropped here. (The 2019
    code used ``np.loadtxt``, which skips leading whitespace instead.)
    '''
    signals = pd.read_csv(fname, sep='\t', header=None, **kwargs).values
    keep = ~np.all(np.isnan(signals), axis=0)
    if not keep.all():
        logger.debug('%s: dropping %d all-NaN column(s)',
                     fname, int((~keep).sum()))
    signals = signals[:, keep]
    if np.isnan(signals).any():
        raise ValueError(f'{fname}: signals still contain NaNs after dropping '
                         'all-NaN columns.')
    return signals


def install_loader():
    '''Bind :func:`load_asc` to ``PreProcessSignals.load_measurement_file``.

    pyOMA reads measurement files through a class-level hook; this is the
    documented way to teach it a project-specific format.
    '''
    from pyOMA.core.PreProcessingTools import PreProcessSignals
    PreProcessSignals.load_measurement_file = staticmethod(load_asc)


def _setup_paths(setup):
    '''Config, channel-DOF and measurement paths of one setup.'''
    src = SCHWABACH_DIR / 'modal_source_files' / setup
    return (src / 'setup_info',
            SCHWABACH_DIR / 'Messdaten' / '2019_06_14_asc_Dateien' / f'{setup}.asc',
            src / 'channel_dofs')


# ── signal processing ────────────────────────────────────────────────────────

def decimation_steps(factor, max_step=8):
    '''
    Split an integer decimation *factor* into successive steps of at most
    *max_step*, as pyOMA's ``decimate_signals`` recommends ("to achieve large
    total reduction factors, call this method multiple times with moderate
    per-step factors"). The 2019 low band used 25 = 5 x 5.

    Returns a list of integer factors whose product is *factor*.
    '''
    factor = int(factor)
    if factor < 1:
        raise ValueError(f'decimation factor must be >= 1, got {factor}')
    if factor == 1:
        return []
    steps = []
    remainder = factor
    for divisor in range(max_step, 1, -1):
        while remainder % divisor == 0 and remainder > 1:
            steps.append(divisor)
            remainder //= divisor
    if remainder > 1:
        # prime larger than max_step (e.g. 11, 13): take it in one step
        steps.append(remainder)
    return sorted(steps, reverse=True)


def _prep_key(highpass, lowpass, decimation, m_lags, n_segments, n_ref):
    '''Hashable, rounded identity of one preprocessing configuration.'''
    return (round(float(highpass), 4), round(float(lowpass), 4),
            int(decimation), int(m_lags), int(n_segments), int(n_ref))


_PREP_CACHE = {}


def prepare_setups(highpass, lowpass, decimation, m_lags, n_segments,
                   n_ref=4, setups=SETUPS, cache=True):
    '''
    Signal-processing stage: load, filter, decimate and correlate every setup.

    Reproduces the 2019 chain (``main_Schwabach_2019.poger``): offset
    correction, a 4th-order Butterworth bandpass, decimation, and a
    Blackman-Tukey correlation estimate -- here split into ``n_segments``
    non-overlapping blocks, which are the aleatory realisations of the study.

    Parameters
    ----------
    highpass, lowpass : float
        Bandpass corner frequencies in Hz.
    decimation : int
        Total decimation factor; ``fs = 128 / decimation``. Applied in steps
        of at most 8 (:func:`decimation_steps`).
    m_lags : int
        Number of correlation lags. Must not exceed the block length
        ``1800 * fs / n_segments`` (``corr_blackman_tukey`` raises otherwise);
        :func:`feasible` screens this before we get here.
    n_segments : int
        Number of non-overlapping blocks per setup. Fixed across the study --
        it sets ``N_ale``, which PolyUQ cannot vary per epistemic sample.
    n_ref : int
        2 or 4 reference channels (the R2 pair alone, or both pairs).
    cache : bool
        Memoise on the rounded parameter tuple. Repeated epistemic samples
        with identical preprocessing then cost nothing.

    Returns
    -------
    list of PreProcessSignals
        One per setup, carrying ``corr_matrices`` with ``n_segments`` blocks,
        ready for ``VarPreGERSSI.add_setup``.
    '''
    from pyOMA.core.PreProcessingTools import PreProcessSignals
    install_loader()

    key = _prep_key(highpass, lowpass, decimation, m_lags, n_segments, n_ref)
    if cache and key in _PREP_CACHE:
        return _PREP_CACHE[key]

    if n_ref not in (2, 4):
        raise ValueError(f'n_ref must be 2 or 4, got {n_ref}')
    ref_channels = list(REF_CHANNELS_FULL if n_ref == 4 else REF_CHANNELS_R2)

    steps = decimation_steps(decimation)
    preps = []
    for setup in setups:
        conf_file, meas_file, chan_dofs_file = _setup_paths(setup)
        prep = PreProcessSignals.init_from_config(
            str(conf_file), str(meas_file), str(chan_dofs_file))

        prep.correct_offset()
        prep.filter_signals(highpass=highpass, lowpass=lowpass,
                            order=4, ftype='butter')
        prep.correct_offset()
        for step in steps:
            prep.decimate_signals(step)
            prep.correct_offset()

        prep.ref_channels = ref_channels
        prep.corr_blackman_tukey(int(m_lags), n_segments=int(n_segments))
        preps.append(prep)

    if cache:
        _PREP_CACHE[key] = preps
    return preps


#: Reference band for :func:`block_levels` -- the union of the two 2019
#: analysis bands. Deliberately *not* the sampled band: the excitation level of
#: a block is a property of the measurement, not of the analyst's choices.
LEVEL_BAND = (0.1, 8.0)

_LEVEL_CACHE = {}


def block_levels(n_segments, band=LEVEL_BAND, setups=SETUPS, cache=True):
    '''
    Observed excitation level of every aleatory block, in physical units.

    The broadband response level at the reference channels stands in for the
    (unknown, unmeasured) excitation level: it is the one quantity available
    in every block of every setup, and it is what the modal parameters of this
    geometrically non-linear structure actually depend on.

    Computed from a *canonical* chain -- offset correction and a bandpass over
    ``band``, with **no decimation** -- rather than from the per-epistemic-
    sample preprocessing, for two reasons. Physically, a block's excitation
    level cannot depend on the analyst's filter and decimation choices; those
    are Imprecision variables of this study. Numerically, pyOMA's
    ``decimate_signals`` multiplies the signals by the decimation factor to
    compensate the power loss of downsampling
    (``PreProcessingTools._apply_downsampling``), so levels taken after
    decimation would carry an epistemic variable as a scale factor.

    Block boundaries are ``N // n_segments`` on the raw record, the same
    partition ``corr_blackman_tukey`` uses, so block ``k`` here is block ``k``
    there.

    Returns
    -------
    levels : (K,) ndarray
        Pooled per-block reference-channel RMS in m/s^2, setups concatenated
        in order.
    offsets : (n_setups + 1,) ndarray
        Index boundaries, so ``levels[offsets[j]:offsets[j + 1]]`` are setup
        ``j``'s blocks. Used to split PolyUQ's pooled weight vector back into
        the per-setup list ``VarPreGERSSI`` expects.
    '''
    from pyOMA.core.PreProcessingTools import PreProcessSignals
    install_loader()

    key = (int(n_segments), round(float(band[0]), 4), round(float(band[1]), 4),
           tuple(setups))
    if cache and key in _LEVEL_CACHE:
        return _LEVEL_CACHE[key]

    per_setup = []
    for setup in setups:
        conf_file, meas_file, chan_dofs_file = _setup_paths(setup)
        prep = PreProcessSignals.init_from_config(
            str(conf_file), str(meas_file), str(chan_dofs_file))
        prep.correct_offset()
        prep.filter_signals(highpass=band[0], lowpass=band[1],
                            order=4, ftype='butter')
        prep.correct_offset()

        refs = prep.signals[:, list(REF_CHANNELS_FULL)]
        n_block = refs.shape[0] // int(n_segments)
        per_setup.append(np.array([
            np.sqrt(np.mean(refs[k * n_block:(k + 1) * n_block] ** 2))
            for k in range(int(n_segments))]))

    offsets = np.concatenate(([0], np.cumsum([len(v) for v in per_setup])))
    result = (np.concatenate(per_setup), offsets)
    if cache:
        _LEVEL_CACHE[key] = result
    return result


def split_weights(weights, offsets):
    '''Split a pooled ``(K,)`` weight vector into ``VarPreGERSSI``'s per-setup
    list. Each entry is renormalised inside pyOMA (``_validate_weights``), so
    only the relative weights within a setup matter here.'''
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape[0] != offsets[-1]:
        raise ValueError(f'expected {offsets[-1]} weights, got {weights.shape[0]}')
    return [weights[offsets[j]:offsets[j + 1]] for j in range(len(offsets) - 1)]


# ── baseline modes of the 2019 study ─────────────────────────────────────────

def load_baseline_modes(bands=('low', 'high')):
    '''
    The manually selected modes of the 2019 PoGER analysis, which replace
    clustering as the pole -> mode assignment target.

    ``stabil_data.npz['self.select_modes']`` holds ``(order, mode_index)``
    pairs picked from the stabilization diagram; the corresponding modal
    parameters are read out of ``modal_data_*.npz``. The low- and high-band
    archives share the same 26-channel ``merged_chan_dofs`` ordering, so both
    selections concatenate into one baseline set: 21 modes at
    0.192 .. 1.336 Hz plus 6 at 2.924 .. 5.495 Hz.

    Returns
    -------
    dict with keys ``f`` (n_modes,), ``d`` (n_modes,, in percent),
    ``phi`` (26, n_modes), ``band`` (n_modes,) and ``chan_dofs`` (26, 5),
    sorted by ascending frequency.
    '''
    freqs, damps, shapes, band_of = [], [], [], []
    chan_dofs = None

    for band in bands:
        archive = SCHWABACH_DIR / BANDS_2019[band]['archive']
        with np.load(archive / BANDS_2019[band]['modal'], allow_pickle=True) as md:
            this_chan_dofs = md['self.merged_chan_dofs']
            all_f = md['self.modal_frequencies']
            all_d = md['self.modal_damping']
            all_phi = md['self.mode_shapes']
        with np.load(archive / 'stabil_data.npz', allow_pickle=True) as st:
            select_modes = st['self.select_modes']

        if chan_dofs is None:
            chan_dofs = this_chan_dofs
        elif not np.array_equal(chan_dofs, this_chan_dofs):
            raise ValueError(
                'The archived bands use different merged channel orderings; '
                'they cannot be concatenated into one baseline.')

        for order, i_mode in select_modes:
            freqs.append(all_f[order, i_mode])
            damps.append(all_d[order, i_mode])
            # (channels, modes, orders), as in main_Schwabach_2019.print_mode_info
            shapes.append(all_phi[:, i_mode, order])
            band_of.append(band)

    freqs = np.array(freqs)
    order = np.argsort(freqs)
    return {'f': freqs[order],
            'd': np.array(damps)[order],
            'phi': np.column_stack(shapes)[:, order],
            'band': np.array(band_of)[order],
            'chan_dofs': chan_dofs}


def assert_matching_dofs(baseline, modal_data):
    '''Check that an identification's merged channels match the baseline's.

    ``pair_modes`` compares mode shapes element-wise, so a mismatched channel
    ordering would silently produce meaningless MAC values.
    '''
    merged = np.array(modal_data.merged_chan_dofs)
    expected = np.array(baseline['chan_dofs'])
    if merged.shape[0] != expected.shape[0]:
        raise ValueError(
            f'identification has {merged.shape[0]} merged channels, baseline '
            f'has {expected.shape[0]}')
    got = [(str(row[1]), float(row[2])) for row in merged]
    want = [(str(row[1]), float(row[2])) for row in expected]
    if got != want:
        mismatch = [i for i, (a, b) in enumerate(zip(got, want)) if a != b]
        raise ValueError(
            f'merged channel ordering differs from the baseline at indices '
            f'{mismatch[:5]}: {[got[i] for i in mismatch[:5]]} vs '
            f'{[want[i] for i in mismatch[:5]]}')


# ── baseline reproduction (Stage A acceptance) ───────────────────────────────

def physical_poles(frequencies, damping, f_min, f_max, d_max=5.0):
    '''Indices of poles inside the analysis band with a plausible damping
    ratio -- the automatic stand-in for the 2019 manual pole selection.'''
    return np.where(np.isfinite(frequencies)
                    & (frequencies > f_min) & (frequencies < f_max)
                    & (damping > 0) & (damping < d_max))[0]


def reproduce_baseline(band='low', orders=None, n_segments=1,
                       num_block_rows=None,
                       freq_thresh=0.2, mac_thresh=0.8, plot=False):
    '''
    Stage-A acceptance check: re-run the 2019 identification of one band with
    current pyOMA and pair the result against the archived selected modes.

    Uses the plain point-estimate :class:`PreGERSSI` (no variance) so a full
    multi-order run is cheap, and pairs the baseline against the poles of each
    order in *orders* separately -- the archived selection spans orders 24 to
    170, so no single order can reproduce all of it. The per-order pairing
    count is the diagnostic; the best order is reported in full.

    Parameters
    ----------
    band : {'low', 'high'}
    orders : iterable of int, optional
        Model orders to test. Defaults to ``range(20, 201, 10)``.
    n_segments : int
        Blocks for the correlation estimate. ``1`` reproduces the 2019
        whole-record estimate exactly; higher values show what the block
        splitting costs.
    plot : bool
        Also call ``compare_modes`` at the best order for its MAC matrix.

    Returns
    -------
    pandas.DataFrame
        One row per order: number of baseline modes paired, mean/max |Δf|,
        mean Δd and mean MAC over the paired modes.
    '''
    from pyOMA.core.MultiSetupSSI import PreGERSSI
    from pyOMA.core.PostProcessingTools import pair_modes, compare_modes
    from pyOMA.core.Helpers import calculateMAC

    cfg = BANDS_2019[band]
    if orders is None:
        orders = range(20, 201, 10)
    orders = list(orders)

    install_loader()
    baseline = load_baseline_modes()
    in_band = baseline['band'] == band
    f_base = baseline['f'][in_band]
    d_base = baseline['d'][in_band]
    phi_base = baseline['phi'][:, in_band]

    p = int(cfg['num_block_rows'] if num_block_rows is None else num_block_rows)
    m_lags = 2 * p + 2
    preps = prepare_setups(cfg['highpass'], cfg['lowpass'], cfg['decimation'],
                           m_lags, n_segments, n_ref=4)

    modal_data = PreGERSSI()
    for prep in preps:
        modal_data.add_setup(prep)
    modal_data.pair_channels()
    assert_matching_dofs(baseline, modal_data)
    modal_data.build_subspace_matrices(p, p, 'covariance')
    # the subspace caps the order at min((p+1)*r, q*r); asking for more raises
    order_cap = modal_data.max_model_order
    orders = [order for order in orders if order < order_cap]
    if not orders:
        raise ValueError(f'p={p} with {modal_data.num_ref_channels} references '
                         f'caps the model order at {order_cap}, below every '
                         'requested order')
    modal_data.compute_modal_params(max_model_order=min(max(orders) + 1,
                                                        order_cap))

    rows = []
    best = (None, -1)
    for order in orders:
        f_all = modal_data.modal_frequencies[order]
        d_all = modal_data.modal_damping[order]
        phi_all = modal_data.mode_shapes[:, :, order]
        keep = physical_poles(f_all, d_all, cfg['highpass'], cfg['lowpass'])
        if keep.size == 0:
            rows.append((order, 0, np.nan, np.nan, np.nan, np.nan))
            continue
        inds_b, inds_n, _, _ = pair_modes(
            f_base, f_all[keep], phi_base, phi_all[:, keep],
            freq_thresh=freq_thresh, mac_thresh=mac_thresh)
        if len(inds_b) == 0:
            rows.append((order, 0, np.nan, np.nan, np.nan, np.nan))
            continue
        df = f_all[keep][inds_n] - f_base[inds_b]
        dd = d_all[keep][inds_n] - d_base[inds_b]
        macs = np.diag(calculateMAC(phi_base[:, inds_b],
                                    phi_all[:, keep][:, inds_n]))
        rows.append((order, len(inds_b), np.mean(np.abs(df)),
                     np.max(np.abs(df)), np.mean(dd), np.mean(macs)))
        if len(inds_b) > best[1]:
            best = (order, len(inds_b))

    table = pd.DataFrame(rows, columns=['order', 'n_paired', 'mean_abs_df',
                                        'max_abs_df', 'mean_dd', 'mean_mac'])
    logger.info('Baseline reproduction, band %r (%d archived modes), best '
                'order %s with %d paired', band, len(f_base), *best)

    if plot and best[0] is not None:
        order = best[0]
        f_all = modal_data.modal_frequencies[order]
        d_all = modal_data.modal_damping[order]
        phi_all = modal_data.mode_shapes[:, :, order]
        keep = physical_poles(f_all, d_all, cfg['highpass'], cfg['lowpass'])
        compare_modes(f_base, d_base, phi_base,
                      f_all[keep], d_all[keep], phi_all[:, keep],
                      freq_thresh=freq_thresh, mac_thresh=mac_thresh)

    return table


# ── variable definitions ─────────────────────────────────────────────────────

#: Blocks per setup. Fixed, not sampled: it sets ``N_ale``, which PolyUQ
#: cannot vary per epistemic sample. Six blocks of 300 s leave the 2019 time
#: lags (82 s low band, 52 s high band) comfortably inside a block while
#: keeping the pooled aleatory ensemble at K = 18.
N_SEGMENTS = 6

#: Reference channels used throughout; see vars_definition_experimental.
N_REF = 4


def fit_level_proposal(levels):
    '''
    Proposal density of the observed block levels: a lognormal fitted by
    maximum likelihood with the location fixed at zero.

    This is the density the aleatory samples were *actually* drawn from -- by
    the weather during the campaign, not by us -- and PolyUQ divides the
    assumed level distribution by it to form the importance weights
    (:meth:`UncertainVariable.proposal_dens`). Without it the weighted
    ensemble would represent the product of the assumed and the observed
    density rather than the assumed one.

    Returns the frozen ``scipy.stats.lognorm`` and its ``(s, scale)``.
    '''
    import scipy.stats
    s, _, scale = scipy.stats.lognorm.fit(np.asarray(levels, dtype=float),
                                          floc=0)
    return scipy.stats.lognorm(s, loc=0, scale=scale), (s, scale)


def vars_definition_experimental(levels=None):
    '''
    Variables of the experimental study, mirroring
    ``UQ_OMA_weighted.vars_definition_weighted``.

    Variability (primary aleatory)
        ``a_ref`` -- the block's broadband reference-channel response level,
        lognormal with Incompleteness-imprecise parameters. The analogue of
        the numerical study's wind speed ``v_b``, except that its samples are
        *observed* (:func:`block_levels`), so it carries an explicit
        ``proposal``.

    Incompleteness (secondary epistemic, drives the weights)
        ``s_a``, ``scale_a`` -- shape and median of that lognormal. The narrow
        focal set (mass 0.7) brackets the pooled maximum-likelihood fit of the
        measured levels, the wide one (mass 0.3) spans the between-setup
        spread, i.e. the range of level distributions the campaign could
        plausibly have sampled.

    Imprecision (primary epistemic, interval-optimized)
        The analyst's choices, all grounded in the 2019 study. The 2019 split
        into a low and a high analysis band is itself one of them: the two
        focal sets of ``highpass``/``lowpass`` reproduce 0.1 .. 1.5 Hz and
        1.5 .. 8 Hz, and their cross combinations give a wide band and a
        degenerate one that :func:`feasible` rejects.

    Parameters
    ----------
    levels : (K,) array_like, optional
        Observed block levels, used to fit ``a_ref.proposal``. Defaults to
        ``block_levels(N_SEGMENTS)[0]``.

    Returns
    -------
    (vars_ale, vars_epi, levels, offsets)
    '''
    from polyuq import MassFunction, RandomVariable

    if levels is None:
        levels, offsets = block_levels(N_SEGMENTS)
    else:
        levels = np.asarray(levels, dtype=float)
        offsets = None

    # ── Variability + Incompleteness ────────────────────────────────────────
    # measured: pooled MLE s = 0.450, median = 3.35 mm/s^2; per-setup means
    # 2.32 / 3.09 / 5.72 mm/s^2 with within-setup CVs of 14.5 .. 33.5 %
    s_a = MassFunction('s_a', [(0.38, 0.52), (0.25, 0.75)], [0.7, 0.3],
                       primary=False)
    s_a.pretty_name = r'$s_\mathfrak{a}$'
    scale_a = MassFunction('scale_a', [(3.0e-3, 3.8e-3), (2.2e-3, 5.5e-3)],
                           [0.7, 0.3], primary=False)
    scale_a.pretty_name = r'$\tilde{\mathfrak{a}}$'

    a_ref = RandomVariable('lognorm', 'a_ref', [s_a, 0.0, scale_a],
                           primary=True)
    a_ref.pretty_name = r'$\mathfrak{a}_\mathrm{ref}$'
    a_ref.proposal, _ = fit_level_proposal(levels)

    # ── Imprecision ─────────────────────────────────────────────────────────
    # The focal sets of every Imprecision variable must either tile or nest
    # over the variable's support. sample_qmc draws uniformly over the support
    # (the union hull of the focals), and hypercube membership requires a
    # sample to lie inside the focal of *every* variable, so a gap between two
    # focals is dead sampling volume: with the natural "one focal per 2019
    # band" definition -- highpass (0.08, 0.15) and (1.2, 1.8) -- 61 % of the
    # highpass samples land in the gap and belong to no hypercube at all, and
    # 30 of 64 hypercubes come out empty even at N_epi = 2000 (measured).
    # Tiling focals cost nothing in expressiveness: a Dempster-Shafer focal is
    # an interval of plausible values, not a tight bracket around the choice
    # that was made in 2019, and the masses still carry the belief.
    highpass = MassFunction('highpass', [(0.08, 0.6), (0.6, 1.8)],
                            [0.6, 0.4], primary=True)  # 2019: 0.1 | 1.5
    highpass.pretty_name = r'$f_\mathrm{hp}$'

    lowpass = MassFunction('lowpass', [(1.2, 4.0), (4.0, 9.0)],
                           [0.5, 0.5], primary=True)  # 2019: 1.5 | 8.0
    lowpass.pretty_name = r'$f_\mathrm{lp}$'

    # NOT an independent variable: the decimation factor follows from the
    # band. Sampling it freely makes the anti-alias constraint reject ~90 % of
    # all samples and leaves most Imprecision hypercubes structurally empty
    # (measured: 35 of 64 with zero feasible samples), because a low sampling
    # rate and a high lowpass corner are simply incompatible. What the analyst
    # is actually free to choose is *how much* oversampling to keep above the
    # band -- 2019 kept f_s / f_lp = 3.41 (low band) and 2.67 (high band),
    # both inside the first focal set. See resolve_decimation.
    nyq_rat = MassFunction('nyq_rat', [(2.5, 4.0), (4.0, 10.0)], [0.6, 0.4],
                           primary=True)
    nyq_rat.pretty_name = r'$\sfrac{f_\mathrm{s}}{f_\mathrm{lp}}$'

    tau_max = MassFunction('tau_max', [(20.0, 175.0)], [1.0], primary=True)
    tau_max.pretty_name = r'$\tau_\mathrm{max}$'

    # Narrower than the full study's (50, 1000), on measured grounds. The
    # first-order variance propagation costs roughly O(p^3) with
    # p = (m_lags - 1) // 2, so the upper end of that range is unaffordable at
    # any useful N_epi. It also buys nothing: sweeping the point-estimate
    # identification over p (reproduce_baseline(num_block_rows=...), 21
    # archived low-band modes, n_segments=6) pairs
    #   p =  50 -> 18/21   p = 100 -> 19/21   p = 150 -> 19/21
    #   p =  75 -> 18/21   p = 125 -> 18/21   p = 210 -> 17/21
    # a plateau from p ~ 50 upward that *falls off* at the 2019 choice: 422
    # lags out of a 1536-sample block leave the high-lag correlations too noisy
    # (the 2019 study estimated them from the whole 1800 s record, where p=210
    # still gives 19/21). Splitting into blocks for the variance estimate is
    # therefore what caps the useful lag count, not the identification itself.
    # The focals span the plateau and stop where it starts to fall off. The
    # 2019 lag counts (422 low band, 1102 high band) sit above the upper
    # bound: the cost is cubic, so including them would put most of the UQ
    # budget into a region the sweep has already shown to be no better -- and
    # that penalty is characterised far more cheaply by the point-estimate
    # sweep above than by spending epistemic samples on it.
    m_lags = MassFunction('m_lags', [(100, 300), (150, 250)], [0.4, 0.6],
                          primary=True)
    m_lags.pretty_name = r'$M$'

    model_order = MassFunction('model_order', [(40, 120), (20, 200)],
                               [0.6, 0.4], primary=True)
    model_order.pretty_name = r'$n_\mathrm{ord}$'

    # n_ref is NOT an Imprecision variable, deliberately. Singleton focals
    # (2,) and (4,) span a support of [2, 4] that sample_qmc fills with
    # uniform integers, so a third of the samples come out as n_ref = 3 --
    # meaningless (references come in orthogonal pairs) and, worse, a member
    # of no hypercube. Expressing a genuinely discrete choice would need
    # focal-restricted sampling, which PolyUQ does not have. Fixed at the 2019
    # choice of both reference pairs.
    vars_ale = [a_ref]
    vars_epi = [s_a, scale_a,
                highpass, lowpass, nyq_rat, tau_max, m_lags, model_order]
    return vars_ale, vars_epi, levels, offsets


def resolve_decimation(lowpass, nyq_rat, fs_raw=FS_RAW, max_step=8):
    '''
    The decimation factor implied by a band and an oversampling ratio.

    The largest factor that still leaves ``f_s >= nyq_rat * lowpass``, snapped
    down to the nearest value that factors into steps of at most *max_step* --
    pyOMA decimates in successive moderate steps, and a large prime factor
    would force one coarse step with a correspondingly poor anti-alias filter.
    2019's choices are recovered: ``(1.5, 3.41) -> 25`` and ``(8.0, 2.67) -> 6``.
    '''
    ideal = int(np.floor(fs_raw / (nyq_rat * lowpass)))
    for factor in range(max(ideal, 1), 0, -1):
        if max(decimation_steps(factor), default=1) <= max_step:
            return factor
    return 1


# ── acceptance-rejection: numerical feasibility of a parameter combination ───

def decimated_length(n_raw, decimation):
    '''Number of time steps left after :func:`decimation_steps`, which floors
    at every step (``PreProcessingTools._apply_downsampling``).'''
    n = int(n_raw)
    for step in decimation_steps(decimation):
        n = int(np.floor(n / step))
    return n


def num_block_rows(m_lags):
    '''``p = q`` for a given lag count.

    ``build_subspace_matrices`` needs ``p + q + 1 <= m_lags``; with ``p = q``
    that is ``2p + 1 <= m_lags``, i.e. ``p = (m_lags - 1) // 2`` -- the plain
    ``m_lags // 2`` of the full study overshoots by one for even lag counts.
    '''
    return (int(m_lags) - 1) // 2


def feasible(highpass, lowpass, nyq_rat, tau_max, m_lags, model_order,
             n_ref=N_REF, n_segments=N_SEGMENTS, n_raw=N_RAW,
             fs_raw=FS_RAW):
    '''
    Screen one epistemic sample for numerical feasibility.

    The analysis parameters are sampled independently, so many combinations
    are not realisable -- a band whose highpass exceeds its lowpass, a lag
    count longer than the data block it is estimated from, a model order above
    what the subspace supports. The original full study handles this the same
    way, with asserts inside its stage-3 mapping
    (``UQ_OMA._stage3mapping``: ``model_order + 2 < m_lags``,
    ``m_lags > N // (n_blocks + 1)``).

    Rejected samples contribute NaN statistic rows, which ``estimate_imp``
    tolerates by fitting its surrogates on the feasible subset
    (``PolyUQ.from_propagated_samples``). Screening here rather than letting
    pyOMA raise keeps the reason legible and countable.

    Returns
    -------
    (ok, reason, resolved) : (bool, str or None, dict)
        *resolved* carries the derived quantities (``decimation_factor``,
        ``sampling_rate``, ``num_block_rows``) so callers need not recompute
        them; it is complete only when *ok*.
    '''
    m_lags = int(m_lags)
    model_order = int(model_order)
    n_ref = int(n_ref)
    resolved = {}

    if highpass + 0.05 >= lowpass:
        return False, 'degenerate band', resolved

    decimation_factor = resolve_decimation(lowpass, nyq_rat, fs_raw=fs_raw)
    fs = fs_raw / decimation_factor
    resolved.update(decimation_factor=decimation_factor, sampling_rate=fs)
    # resolve_decimation guarantees fs >= nyq_rat * lowpass >= 2.5 * lowpass,
    # so the anti-alias margin can no longer be violated by construction

    if m_lags > int(np.ceil(tau_max * fs)):
        return False, 'm_lags beyond the estimated correlation length', resolved

    n_block = decimated_length(n_raw, decimation_factor) // int(n_segments)
    resolved['block_length'] = n_block
    # Unreachable for the sampled ranges -- block length and correlation length
    # are both inversely proportional to the decimation factor, so their ratio
    # is tau_max * n_segments / DURATION and the correlation length binds first
    # for any tau_max below DURATION / n_segments = 300 s. Kept because it goes
    # live as soon as n_segments or tau_max's focals change, and because
    # corr_blackman_tukey would otherwise raise from deep inside pyOMA.
    if m_lags > n_block:
        return False, 'm_lags beyond the block length', resolved

    if model_order + 2 >= m_lags:
        return False, 'model_order too high for m_lags', resolved

    p = num_block_rows(m_lags)
    resolved['num_block_rows'] = p
    if p < 1:
        return False, 'no block rows left', resolved
    # max_model_order = min((p + 1) * r, q * r) = r * p for q = p
    if model_order > n_ref * p:
        return False, 'model_order beyond the subspace order cap', resolved

    return True, None, resolved


def sample_parameters(poly_uq, n_epi):
    '''The realized analysis parameters of epistemic sample *n_epi*, typed as
    the pyOMA calls need them.'''
    inp = poly_uq.inp_samp_prim
    return {'highpass': float(inp['highpass'].iloc[n_epi]),
            'lowpass': float(inp['lowpass'].iloc[n_epi]),
            'nyq_rat': float(inp['nyq_rat'].iloc[n_epi]),
            'tau_max': float(inp['tau_max'].iloc[n_epi]),
            'm_lags': int(inp['m_lags'].iloc[n_epi]),
            'model_order': int(inp['model_order'].iloc[n_epi]),
            }


def feasibility_report(poly_uq, n_segments=N_SEGMENTS):
    '''
    Dry run of :func:`feasible` over the whole epistemic sample set -- no
    identification, so it costs milliseconds.

    Run this before committing to a production sweep: it gives the acceptance
    rate overall and per Imprecision hypercube, which is what ``N_epi`` has to
    be sized against. A hypercube with too few feasible samples cannot support
    a surrogate, and ``estimate_imp`` would extrapolate rather than fail.

    Returns
    -------
    (per_sample, per_hypercube) : (pandas.DataFrame, pandas.DataFrame)
    '''
    rows = []
    for n_epi in range(poly_uq.N_mcs_epi):
        params = sample_parameters(poly_uq, n_epi)
        ok, reason, resolved = feasible(n_segments=n_segments, **params)
        rows.append(dict(params, **resolved, n_epi=n_epi, ok=ok,
                         reason='' if ok else reason))
    per_sample = pd.DataFrame(rows)

    # (n_imp_hyc, N_mcs_epi) membership of every epistemic sample
    in_hyc = poly_uq.hypercube_sample_indices(
        poly_uq.inp_samp_prim, list(poly_uq.vars_imp),
        hyc_foc_inds=poly_uq.imp_hyc_foc_inds,
        N_mcs_epi=poly_uq.N_mcs_epi)
    ok = per_sample['ok'].values
    per_hypercube = pd.DataFrame({
        'i_imp': np.arange(in_hyc.shape[0]),
        'n_samples': in_hyc.sum(axis=1),
        'n_feasible': (in_hyc & ok[np.newaxis, :]).sum(axis=1),
        'mass': poly_uq.imp_hyc_mass})
    return per_sample, per_hypercube


# ── system identification + inline pole -> mode assignment ──────────────────

def _empty_result(reason):
    '''Result of an epistemic sample that produced no usable poles.

    ``keys`` is empty, so :meth:`PolyUQ.stat_rows` selects nothing for this
    sample and every mode's statistic row carries NaN there -- which
    ``estimate_imp`` handles by fitting its surrogates on the feasible subset.
    '''
    return {'keys': [], 'point': np.empty((0, 4)), 'rejected': reason}


def identify(params, resolved, weights_per_setup, weighting, baseline,
             n_segments=N_SEGMENTS, n_ref=N_REF, convention='substitution',
             cached=None):
    '''
    One multi-setup identification with first-order uncertainties.

    Parameters
    ----------
    params, resolved : dict
        Sampled and derived analysis parameters (:func:`sample_parameters`,
        :func:`feasible`).
    weights_per_setup : list or None
        Per-setup block weights, as ``VarPreGERSSI`` expects them; ``None``
        for the unweighted estimator.
    weighting : {'build', 'posthoc'}
        ``'build'`` folds the weights into the subspace matrices, so the point
        estimate moves with them. ``'posthoc'`` reweights only the variances of
        an already-built unweighted identification -- the caller passes that
        build back in through *cached* so it is reused across the epistemic
        sample's hypercubes.
    cached : VarPreGERSSI, optional
        A previously built unweighted identification of the *same* epistemic
        sample (``'posthoc'`` only).

    Returns
    -------
    (modal_data, order) : the identification and the evaluated model order.
    '''
    from pyOMA.core.MultiSetupSSI import VarPreGERSSI

    order = int(params['model_order'])
    if weighting == 'posthoc' and cached is not None:
        cached.apply_block_weights(weights=weights_per_setup,
                                   convention=convention)
        return cached, order

    preps = prepare_setups(params['highpass'], params['lowpass'],
                           resolved['decimation_factor'], params['m_lags'],
                           n_segments, n_ref=n_ref)
    # the mode-shape variance factors are never used (only std_f / std_d feed
    # the statistic rows), and they dominate the cache, so 'freqdamp' only
    modal_data = VarPreGERSSI(
        cache_variance_factors=(weighting == 'posthoc'), cache='freqdamp')
    for prep in preps:
        modal_data.add_setup(prep)
    modal_data.pair_channels()
    assert_matching_dofs(baseline, modal_data)

    p = int(resolved['num_block_rows'])
    if weighting == 'build':
        modal_data.build_subspace_matrices(p, p, 'covariance',
                                           weights=weights_per_setup)
        modal_data.compute_modal_params(orders=[order])
    else:
        modal_data.build_subspace_matrices(p, p, 'covariance')
        modal_data.compute_modal_params(orders=[order])
        if weights_per_setup is not None:
            modal_data.apply_block_weights(weights=weights_per_setup,
                                           convention=convention)
    return modal_data, order


def assign_to_baseline(modal_data, order, params, baseline,
                       freq_thresh=0.2, mac_thresh=0.8, d_max=5.0):
    '''
    Pair the identified poles against the 2019 baseline modes -- the step that
    replaces clustering.

    Runs inside the estimator, immediately after the identification, so the
    rows leaving it are already keyed by a *global* baseline mode index. There
    is no pole database to cluster afterwards: PolyUQ's ``stat_db`` entries
    can be selected by key directly (:meth:`PolyUQ.stat_rows`).

    Baseline modes outside the sampled band simply fail to pair, which is the
    intended behaviour -- a narrow band is expected to resolve fewer modes, and
    that shows up as reduced coverage rather than as an error.

    Returns
    -------
    dict with ``keys`` (baseline mode indices), the paired poles' ``f``, ``d``,
    ``std_f``, ``std_d``, ``mac`` and the ``point`` array
    ``(n_paired, 4) = [f, d, std_f, std_d]``.
    '''
    from pyOMA.core.PostProcessingTools import pair_modes
    from pyOMA.core.Helpers import calculateMAC

    f_all = modal_data.modal_frequencies[order]
    d_all = modal_data.modal_damping[order]
    phi_all = modal_data.mode_shapes[:, :, order]
    std_f_all = modal_data.std_frequencies[order]
    std_d_all = modal_data.std_damping[order]

    keep = physical_poles(f_all, d_all, params['highpass'], params['lowpass'],
                          d_max=d_max)
    if keep.size == 0:
        return _empty_result('no physical poles')

    inds_base, inds_cell, _, _ = pair_modes(
        baseline['f'], f_all[keep], baseline['phi'], phi_all[:, keep],
        freq_thresh=freq_thresh, mac_thresh=mac_thresh)
    if len(inds_base) == 0:
        return _empty_result('no baseline mode paired')

    sel = keep[inds_cell]
    f, d = f_all[sel], d_all[sel]
    std_f, std_d = std_f_all[sel], std_d_all[sel]
    mac = np.diag(calculateMAC(baseline['phi'][:, inds_base], phi_all[:, sel]))

    return {'keys': [int(i) for i in inds_base],
            'f': f, 'd': d, 'std_f': std_f, 'std_d': std_d, 'mac': mac,
            'point': np.column_stack([f, d, std_f, std_d])}


def make_experimental_estimator(poly_uq, baseline, offsets,
                                weighting='build', n_segments=N_SEGMENTS,
                                n_ref=N_REF, convention='substitution',
                                target_probabilities=None,
                                freq_thresh=0.2, mac_thresh=0.8, d_max=5.0):
    '''
    Estimator for :meth:`PolyUQ.estimate_stat`, following its injected-callable
    protocol: PolyUQ computes the Incompleteness-conditioned importance weights
    internally and calls ``estimator(n_epi, i_imp, weights)`` with them. There
    is no weights exit point; this wrapper only consumes them.

    Per call it (1) screens the epistemic sample with :func:`feasible`,
    (2) gets the block-wise correlations for the resolved preprocessing,
    (3) splits PolyUQ's pooled ``(K,)`` weight vector into ``VarPreGERSSI``'s
    per-setup list, (4) identifies (:func:`identify`) and (5) assigns the poles
    to baseline modes (:func:`assign_to_baseline`).

    ``weighting='posthoc'`` builds one *unweighted* identification per
    epistemic sample, cached in this closure, and only refreshes the variances
    per hypercube via ``apply_block_weights``.
    '''
    if weighting not in ('build', 'posthoc'):
        raise ValueError(f"weighting must be 'build' or 'posthoc', got {weighting!r}")
    from pyoma_uq.studies.UQ_OMA_weighted import expand_parametric_cdf

    K = int(offsets[-1])
    state = {'n_epi': None, 'obj': None}

    def estimator(n_epi, i_imp, weights):
        params = sample_parameters(poly_uq, n_epi)
        ok, reason, resolved = feasible(n_segments=n_segments, n_ref=n_ref,
                                        **params)
        if not ok:
            logger.debug('epistemic sample %d rejected: %s', n_epi, reason)
            # 'infeasible' marks the acceptance-rejection screen specifically,
            # as opposed to a sample that was identified but in which a given
            # mode simply did not appear -- mode_coverage must not conflate the
            # two, or the coverage denominator silently shrinks
            return dict(_empty_result(reason), infeasible=reason)

        if weights is None:
            weights = np.full(K, 1.0 / K)
        weights_per_setup = split_weights(weights, offsets)
        n_eff = float(poly_uq.kish_n_eff(weights))

        if state['n_epi'] != n_epi:
            state.update(n_epi=n_epi, obj=None)
        try:
            modal_data, order = identify(
                params, resolved, weights_per_setup, weighting, baseline,
                n_segments=n_segments, n_ref=n_ref, convention=convention,
                cached=state['obj'] if weighting == 'posthoc' else None)
        except Exception as exc:      # a numerically feasible but failing cell
            logger.warning('epistemic sample %d failed to identify: %r',
                           n_epi, exc)
            return _empty_result(f'identification failed: {exc!r}')
        if weighting == 'posthoc':
            state['obj'] = modal_data

        result = assign_to_baseline(modal_data, order, params, baseline,
                                    freq_thresh=freq_thresh,
                                    mac_thresh=mac_thresh, d_max=d_max)
        result['n_eff'] = n_eff
        result['order'] = order
        result.update({key: resolved[key] for key in
                       ('decimation_factor', 'sampling_rate', 'num_block_rows')})
        if target_probabilities is not None and len(result['keys']):
            result['cdf_f'] = expand_parametric_cdf(
                result['f'], result['std_f'], n_eff, target_probabilities,
                dist='normal')
            result['cdf_d'] = expand_parametric_cdf(
                result['d'], result['std_d'], n_eff, target_probabilities,
                dist='lognormal')  # damping: strictly positive, right-skewed
        return result

    return estimator


# ── statistic-level processing ───────────────────────────────────────────────

def mode_selection(poly_uq, label):
    '''``(entry, row)`` pairs of one baseline mode, for :meth:`stat_rows`.

    Because :func:`assign_to_baseline` keys the estimator's rows by the global
    baseline mode index, selecting a mode is a lookup rather than a clustering
    result.
    '''
    return [(i, entry['keys'].index(label))
            for i, entry in enumerate(poly_uq.stat_db)
            if label in entry['keys']]


def mode_coverage(poly_uq, baseline):
    '''
    Per baseline mode, in how many epistemic samples it was found.

    Reported against the *feasible* samples, not all of them: a mode cannot be
    identified in a sample that was rejected before any identification ran, so
    counting those would conflate two different failures. A sample that *was*
    identified but in which this mode did not appear stays in the denominator
    -- that is exactly the miss the coverage is meant to report.
    '''
    feasible_epi = {entry['n_epi'] for entry in poly_uq.stat_db
                    if 'infeasible' not in entry}
    rows = []
    for label in range(len(baseline['f'])):
        found = {poly_uq.stat_db[i]['n_epi'] for i, _ in
                 mode_selection(poly_uq, label)}
        rows.append({'label': label, 'f_baseline': baseline['f'][label],
                     'band': baseline['band'][label], 'n_found': len(found),
                     'coverage': len(found) / max(len(feasible_epi), 1)})
    return pd.DataFrame(rows)


def statistic_level(poly_uq, label, field='point', i_stat=0,
                    out_name=None):
    '''
    Statistic-level PolyUQ instance for one baseline mode and one statistic.

    ``field='point'`` with ``i_stat`` 0..3 selects the mean value of ``f``,
    ``d``, ``std_f`` or ``std_d``; ``field='cdf_f'``/``'cdf_d'`` with ``i_stat``
    a probability level gives the aleatory CDF value at that level.

    Returns ``(pq_stat, hyc_rows)``; call
    ``pq_stat.estimate_imp(hyc_rows=hyc_rows)``.
    '''
    selection = mode_selection(poly_uq, label)
    if not selection:
        raise ValueError(f'baseline mode {label} was never identified')
    rows, hyc_rows = poly_uq.stat_rows(selection, field=field, i_stat=i_stat)
    if out_name is None:
        out_name = f'{field}_{i_stat}_mode{label}'
    pq_stat = poly_uq.to_statistic_level(rows, out_name=out_name)
    if hyc_rows is not None:
        hyc_rows = poly_uq.expand_hyc_rows(pq_stat, hyc_rows)
    return pq_stat, hyc_rows


POINT_FIELDS = {'f': 0, 'd': 1, 'std_f': 2, 'std_d': 3}


def run_experimental_pipeline(result_dir, weighting='build', N_epi=1000,
                              seed=1509, n_segments=N_SEGMENTS,
                              min_coverage=0.1, quantities=('f', 'd'),
                              n_stat=0, opt_meth='genetic', labels=None,
                              vars_fun=None):
    '''
    The complete experimental study: sampling, weighted identification with
    inline mode assignment, and statistic-level interval optimization.

    Parameters
    ----------
    weighting : {'build', 'posthoc'}
        Build-time weighted identification ("OMA-enabled") or post-hoc block
        reweighting of one unweighted build per epistemic sample
        ("OMA-optimized").
    n_stat : int
        Number of aleatory CDF levels to expand per pole. ``0`` (default)
        computes only the mean-value focal intervals from ``field='point'``;
        a positive value additionally reconstructs the aleatory p-box.
    min_coverage : float
        Baseline modes found in fewer than this fraction of the feasible
        epistemic samples are reported but not interval-optimized.
    labels : iterable of int, optional
        Restrict to these baseline modes (for quick checks).
    vars_fun : callable, optional
        Replaces :func:`vars_definition_experimental` as the source of
        ``(vars_ale, vars_epi, levels, offsets)``. Lets a variant study -- or a
        deliberately cheap smoke run -- reuse the whole pipeline without
        editing the module.

    Returns ``(poly_uq, baseline, coverage, results)``.
    '''
    import pickle
    from polyuq import PolyUQ

    result_dir = Path(result_dir) / weighting
    result_dir.mkdir(parents=True, exist_ok=True)

    install_loader()
    baseline = load_baseline_modes()
    vars_ale, vars_epi, levels, offsets = (
        vars_definition_experimental if vars_fun is None else vars_fun)()

    path = 'fast_build' if weighting == 'build' else 'fast_posthoc'
    poly_uq = PolyUQ(vars_ale, vars_epi, dim_ex='cartesian', path=path)
    poly_uq.sample_qmc(N_mcs_ale=len(levels), N_mcs_epi=N_epi, seed=seed,
                       given_samples={'a_ref': levels})

    per_sample, per_hyc = feasibility_report(poly_uq, n_segments=n_segments)
    logger.info('Feasible epistemic samples: %d/%d (%.1f %%); hypercubes '
                'without a feasible sample: %d/%d',
                per_sample.ok.sum(), N_epi, per_sample.ok.mean() * 100,
                int((per_hyc.n_feasible == 0).sum()), len(per_hyc))
    per_sample.to_csv(result_dir / 'feasibility_samples.csv', index=False)
    per_hyc.to_csv(result_dir / 'feasibility_hypercubes.csv', index=False)

    target_probabilities = (np.linspace(0, 1, n_stat) if n_stat else None)
    estimator = make_experimental_estimator(
        poly_uq, baseline, offsets, weighting=weighting,
        n_segments=n_segments, target_probabilities=target_probabilities)

    # weighted=True for BOTH weightings, as in UQ_OMA_weighted's
    # run_weighted_identification: 'posthoc' builds once per epistemic sample
    # inside the estimator and only refreshes the variances per weight group,
    # so it still needs PolyUQ to hand it the weights. Driving it with
    # weighted=False would produce plainly unweighted statistics instead.
    # eliminate=False: no Imprecision variable has secondary-Variability focal
    # bounds here, so the elimination is a structural no-op.
    poly_uq.estimate_stat(estimator, weighted=True, eliminate=False)
    with open(result_dir / 'stat_db.pkl', 'wb') as fh:
        pickle.dump(poly_uq.stat_db, fh)

    coverage = mode_coverage(poly_uq, baseline)
    coverage.to_csv(result_dir / 'mode_coverage.csv', index=False)

    results = {}
    for _, row in coverage.iterrows():
        label = int(row['label'])
        if labels is not None and label not in labels:
            continue
        if row['coverage'] < min_coverage:
            logger.info('Baseline mode %d (%.3f Hz): found in %.0f %% of the '
                        'feasible samples - skipping interval optimization.',
                        label, row['f_baseline'], row['coverage'] * 100)
            continue
        results[label] = {}
        for quantity in quantities:
            try:
                pq_stat, hyc_rows = statistic_level(
                    poly_uq, label, field='point',
                    i_stat=POINT_FIELDS[quantity])
                imp_foc, _, intp_errors, _, _ = pq_stat.estimate_imp(
                    interp_fun='rbf', opt_meth=opt_meth, hyc_rows=hyc_rows)
            except Exception as exc:
                logger.error('Baseline mode %d, %s: statistic-level '
                             'processing failed: %r', label, quantity, exc)
                continue
            results[label][quantity] = {'imp_foc': imp_foc[0],
                                        'imp_hyc_mass': pq_stat.imp_hyc_mass,
                                        'intp_errors': intp_errors}
            np.savez(result_dir / f'focals_mode{label}_{quantity}.npz',
                     imp_foc=imp_foc[0], imp_hyc_mass=pq_stat.imp_hyc_mass,
                     intp_errors=intp_errors,
                     f_baseline=row['f_baseline'])

            if not n_stat:
                continue
            # aleatory p-box: one interval optimization per probability level,
            # stacked into the (n_stat, n_hyc, 2) layout of the numerical
            # study's polyuq_cdf_inc.npz
            try:
                foc_rows, mass = [], None
                for i_stat in range(n_stat):
                    pq_cdf, hyc_rows_cdf = statistic_level(
                        poly_uq, label, field=f'cdf_{quantity}', i_stat=i_stat)
                    foc, _, _, _, _ = pq_cdf.estimate_imp(
                        interp_fun='rbf', opt_meth=opt_meth,
                        hyc_rows=hyc_rows_cdf)
                    foc_rows.append(foc[0])
                    mass = pq_cdf.imp_hyc_mass
            except Exception as exc:
                logger.error('Baseline mode %d, %s: CDF reconstruction '
                             'failed: %r', label, quantity, exc)
                continue
            results[label][quantity]['cdf'] = {
                'imp_foc': np.stack(foc_rows, axis=0), 'imp_hyc_mass': mass,
                'target_probabilities': target_probabilities}
            np.savez(result_dir / f'cdf_mode{label}_{quantity}.npz',
                     imp_foc=np.stack(foc_rows, axis=0), imp_hyc_mass=mass,
                     target_probabilities=target_probabilities)
    return poly_uq, baseline, coverage, results


if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('result_dir', type=Path, nargs='?')
    parser.add_argument('--weighting', choices=('build', 'posthoc'),
                        default='build')
    parser.add_argument('--n-epi', type=int, default=1000)
    parser.add_argument('--n-stat', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1509)
    parser.add_argument('--band', choices=('low', 'high'), default='low')
    parser.add_argument('--n-segments', type=int, default=1)
    parser.add_argument('--reproduce-baseline', action='store_true',
                        help='Stage-A check: re-run the 2019 identification '
                             'and pair it against the archived modes.')
    parser.add_argument('--feasibility-only', action='store_true',
                        help='Dry-run the acceptance-rejection screen and '
                             'exit; costs milliseconds and sizes N_epi.')
    args = parser.parse_args()

    if args.reproduce_baseline:
        table = reproduce_baseline(band=args.band, n_segments=args.n_segments)
        print(table.to_string(index=False))
    elif args.feasibility_only:
        from polyuq import PolyUQ
        install_loader()
        vars_ale, vars_epi, levels, _ = vars_definition_experimental()
        pq = PolyUQ(vars_ale, vars_epi, dim_ex='cartesian', path='fast_build')
        pq.sample_qmc(N_mcs_ale=len(levels), N_mcs_epi=args.n_epi,
                      seed=args.seed, given_samples={'a_ref': levels})
        per_sample, per_hyc = feasibility_report(pq)
        print(f'accepted {per_sample.ok.sum()}/{args.n_epi} '
              f'({per_sample.ok.mean() * 100:.1f} %)')
        print(per_sample.loc[~per_sample.ok, 'reason'].value_counts().to_string())
        print(per_hyc.to_string(index=False))
    elif args.result_dir is not None:
        run_experimental_pipeline(args.result_dir, weighting=args.weighting,
                                  N_epi=args.n_epi, seed=args.seed,
                                  n_stat=args.n_stat)
    else:
        parser.error('give a result_dir, --reproduce-baseline or '
                     '--feasibility-only')
