'''
OMA as the weighted statistical estimator — reduced case study.

Implements the dissertation outlook "Toward OMA-Specific PolyUQ Processing
Strategies" (Algorithm alg:proposed): instead of running a full OMA system
identification once per (aleatory, epistemic) sample pair, the deterministic
model is evaluated on the sample lattice only up to the correlation-function
estimate (stage2corr_mapping). A weighted OMA estimator then consumes, per
epistemic sample and Imprecision hypercube, all N_ale correlation estimates
(or decimated signals) at once, with Incompleteness-conditioned importance
weights obtained by freezing the secondary Incompleteness variables (c_vb,
lamda_vb) and evaluating the wind-speed pdf on the aleatory samples
(PolyUQ.probabilities_imp). Identified poles carry first-order variance
estimates, are clustered across all epistemic cells (cluster_modes_weighted),
and the per-mode statistics Theta[q, n_e] are processed on the statistic
level: a surrogate over the epistemic samples, interval-optimized within the
combined Imprecision x Incompleteness hypercubes via
PolyUQ.from_propagated_samples(...).estimate_imp().

Three OMA estimators (``method``) x two weighting mechanisms (``weighting``)
= 6 case-study runs, dispatched by run_weighted_identification /
run_weighted_uq_pipeline / run_weighted_uq_pipeline_all:

    method   estimator                        weighting  mechanism
    'sc'     VarSSIRef, covariance-driven      'build'    build-time weighted
    'sd'     VarSSIRef, projection (data-driven)          identification --
    'cf'     VarPLSCF                                     the point estimate
                                                           moves with the
                                                           weights ("OMA-
                                                           enabled PolyUQ")
                                                'posthoc'  post-hoc block-
                                                           weight reweighting
                                                           (Tier A/B): ONE
                                                           unweighted build
                                                           per epistemic
                                                           sample, then
                                                           apply_block_weights
                                                           per hypercube --
                                                           point estimate
                                                           stays fixed at the
                                                           unweighted mean
                                                           ("OMA-optimized
                                                           PolyUQ")

All three estimators expose both mechanisms with real (non-degenerate)
variance: VarSSIRef-projection's build-time weighting uses
build_subspace_mat's default (non-experimental) weighted-mean reading (every
block's own per-block/joint-LQ projection stays as in the unweighted build;
weights only pick a weighted mean over the independent per-block estimates),
not the pre-LQ-weighted experimental_weighted_projection=True reading, which
has no variance at all and is not used here. See B.3-style caveats in
weighted_data_driven_identification_build's and weighted_plscf_identification's
docstrings, and project_case_study_2_redo.md memory, for what remains
approximate in each combination.

Reduced case-study definition (fixed during planning):
N_ale = 100 aleatory samples, N_epi = 200 epistemic samples. Epistemic
variables: model_order, tau_max, decimation_factor, n_locations, T
(Imprecision); c_vb, lamda_vb (secondary Incompleteness, drive the weights);
DAQ_noise_rms (primary Incompleteness). All other acquisition parameters of
the original study (UQ_OMA.vars_definition) are fixed to representatives of
their highest-mass focal sets. m_lags is derived (ceil(tau_max * fs)), the
estimator is fixed to Blackman-Tukey.

The recording duration T is a genuinely Imprecise variable: a single-focal
MassFunction whose one interval is bounded below by Delta_t1 and above by
Delta_t2, two independent truncated-normal RandomVariables, rather than a
plain numeric interval — PolyUQ's native focal-nesting mechanism
(MassFunction.support()/numeric_focal() recurse into a nested
UncertainVariable's own support/frozen value). Delta_t1 ~ TruncNorm(mean=10
min, std=1 min, truncated +-2 min); Delta_t2 ~ TruncNorm(mean=30 min, std=5
min, truncated +-5 min) — deliberately large stds so the effect of a *true*
Imprecision variable (one that actually moves stage2corr_mapping) on the
propagated modal-parameter uncertainty is clearly measurable, unlike the
secondary Incompleteness variables (c_vb, lamda_vb) which only reweight.
Delta_t1/Delta_t2 are *secondary-Variability* (listed in vars_ale, not
vars_epi, with primary=False) rather than secondary Incompleteness like
c_vb/lamda_vb: T's focal bound is formed by aleatory samples, so their
per-aleatory-sample density is handled by PolyUQ.probabilities_imp's
hyc_hyp_vars/inp_suppl_ale machinery, not by the epistemic-freeze path
c_vb/lamda_vb use.

Case-study degeneracy (unaffected by T): T has a single focal set (mass
1.0), so — like DAQ_noise_rms/decimation_factor — it does not itself branch
the Imprecision hypercube grid; per-hypercube weights stay independent of q
exactly as before T was added. Delta_t1/Delta_t2's densities still enter
compute_weights (via hyc_hyp_vars) and change the weight *values* relative
to not having T at all, but uniformly across every hypercube, so OMA still
effectively runs once per epistemic sample. The generic per-q loop (with a
dedupe fast path) is implemented anyway to match the dissertation
algorithm.

Development data: responses are synthesized locally from the mast's stored
modal solution (model.toy_response); a response_provider callable is the
swap-in point for the original cluster data.

Outlook (not implemented): first-order propagation of weight perturbations,
d(x_bar)/d(w_n) = x_n - x_bar under renormalization, through the fast
algorithm's Jacobian chain would give d(Theta)/d(w_n) from already-computed
quantities, enabling gradient-based Incompleteness optimization directly on
the weights, bypassing the statistic-level surrogate.
'''
import logging
import multiprocessing
import os
from pathlib import Path

import numpy as np
import pandas as pd

from polyuq import MassFunction, RandomVariable, PolyUQ

logger = logging.getLogger(__name__)

FS_M = 70.0          # sampling rate of the synthesized "mechanical" response
N_GEN = 2 ** 18      # generated timesteps per aleatory record (~3746 s;
                     # covers Delta_t2's <=35 min = 2100 s focal support)
NUM_NODES = 203      # mast model nodes, as in stage2mapping
NYQ_MAX = 35.0       # pole-mapping Nyquist for clustering, as in cluster_modes

# Fixed representatives of the highest-mass focal sets of the variables
# dropped from the epistemic set (see UQ_OMA.vars_definition, stage 2/3).
FIXED_PARAMS = {
    'DTC': 15.8,                            # (1.6, 30) mass 0.7
    'sensitivity_nominal': 1.02,            # (1.02,) mass 0.5
    'sensitivity_deviation_percent': 5.0,   # (5.,) mass 0.4
    'spectral_noise_slope': -0.55,          # (-0.8, -0.3) mass 1.0
    'sensor_noise_rms': 5.5e-6,             # (1e-6, 1e-5) mass 0.4
    'range_estimation_duration': 90.0,      # (60, 120) mass 0.5
    'range_estimation_margin': 3.5,         # (2., 5.) mass 0.6
    'anti_aliasing_cutoff_factor': 0.425,   # (0.4, 0.45) mass 0.7
    'quant_bit_factor': 6,                  # (6,) mass 0.6 -> 24 bit
}


def vars_definition_weighted():
    '''
    Variable definitions of the reduced case study; focal sets and masses
    from UQ_OMA.vars_definition except: tau_max upper bounds reduced to stay
    well below T's shortest focal support, decimation_factor reduced to its
    "risk-loving" focal (fs 3.9 ... 10 Hz), m_lags removed (derived as
    ceil(tau_max * fs)).

    T (recording duration) is a genuinely Imprecise MassFunction with a
    single focal set (mass 1.0): a 2-tuple interval [Delta_t1, Delta_t2]
    whose lower and upper bounds are themselves independent truncated-normal
    RandomVariables, rather than plain numeric bounds — PolyUQ's native
    focal-nesting mechanism (MassFunction.support()/numeric_focal() recurse
    into each nested UncertainVariable). Delta_t1/Delta_t2 are primary=False
    and listed in vars_ale (secondary Variability), matching how c_vb/
    lamda_vb are nested under v_b but on the aleatory axis instead of the
    Incompleteness one. Because T has only one focal set, it does not itself
    branch the Imprecision hypercube grid (see module docstring).

    Returns (vars_ale, vars_epi, arg_vars, fixed_params); arg_vars maps
    stage2corr_mapping arguments to variable names (model_order acts on the
    OMA stage only and is not a lattice argument).
    '''
    c = MassFunction('c_vb', [(2.267, 2.3), (1.96, 2.01)], [0.75, 0.25],
                     primary=False)  # incompleteness, secondary
    c.pretty_name = r'$k_{\mathfrak{v}}$'
    lamda = MassFunction('lamda_vb', [(5.618, 5.649), (5.91, 6.0)],
                         [0.75, 0.25], primary=False)  # incompleteness, secondary
    lamda.pretty_name = r'$\lambda_{\mathfrak{v}}$'

    v_b = RandomVariable('weibull_min', 'v_b', [c, lamda], primary=True)
    v_b.pretty_name = r'$\mathfrak{v}_\mathrm{b}$'

    n_locations = MassFunction('n_locations', [(4,), (8,), (12,)],
                               [0.2, 0.5, 0.3], primary=True)  # imprecision
    n_locations.pretty_name = r'$n_\mathrm{loc}$'

    DAQ_noise_rms = MassFunction('DAQ_noise_rms', [(2.5e-6, 3e-3)], [1.0],
                                 primary=True)  # incompleteness, primary (X^c,q)
    DAQ_noise_rms.pretty_name = r'$\sigma_{\eta_\mathrm{D}}$'

    decimation_factor = MassFunction('decimation_factor', [(7, 18)], [1.0],
                                     primary=True)  # imprecision
    decimation_factor.pretty_name = r'$f_\mathrm{s}$'

    tau_max = MassFunction('tau_max', [(20.0, 60.0), (40.0, 90.0)],
                           [0.5, 0.5], primary=True)  # imprecision
    tau_max.pretty_name = r'$\tau_\mathrm{max}$'

    model_order = MassFunction('model_order', [(10, 30), (20, 60), (10, 100)],
                               [0.4, 0.4, 0.2], primary=True)  # imprecision
    model_order.pretty_name = r'$n_\mathrm{ord}$'

    Delta_t1 = RandomVariable('truncnorm', 'Delta_t1', [-2.0, 2.0, 10. * 60, 60.],
                              primary=False)  # variability, secondary: mean 10 min, std 1 min, +-2 min
    Delta_t1.pretty_name = r'$\Delta t_1$'
    Delta_t2 = RandomVariable('truncnorm', 'Delta_t2', [-1.0, 1.0, 30. * 60, 300.],
                              primary=False)  # variability, secondary: mean 30 min, std 5 min, +-5 min
    Delta_t2.pretty_name = r'$\Delta t_2$'
    T = MassFunction('T', [(Delta_t1, Delta_t2)], [1.0],
                     primary=True)  # imprecision, a single interval bounded by secondary-Variability samples
    T.pretty_name = r'$T$'

    vars_ale = [v_b, Delta_t1, Delta_t2]
    vars_epi = [c, lamda,
                n_locations, DAQ_noise_rms, decimation_factor, tau_max,
                model_order, T]

    arg_vars = {'n_locations': n_locations.name,
                'DAQ_noise_rms': DAQ_noise_rms.name,
                'decimation_factor': decimation_factor.name,
                'tau_max': tau_max.name,
                'duration': T.name,
                'v_b': v_b.name}

    return vars_ale, vars_epi, arg_vars, dict(FIXED_PARAMS)


# ── Development-data response provider ───────────────────────────────────────

class ToyResponseProvider:
    '''
    Response provider backed by model.toy_response: synthesizes (and caches
    per id_ale) ambient acceleration responses at the candidate sensor nodes
    from the mast's stored modal solution. The interface — call with
    (id_ale, v_b), get (t_vals, accel(N, n_nodes, 2), nodes) — is the
    swap-in point for the original cluster data (response.npz records).
    '''

    def __init__(self, result_dir, mech_npz):
        self.result_dir = Path(result_dir)
        self.mech_npz = Path(mech_npz)
        self._frf = None

    def _get_frf(self):
        if self._frf is None:
            from model.mechanical import MechanicalDummy
            from model import toy_response
            mech = MechanicalDummy.load(str(self.mech_npz))
            _, frf = toy_response.modal_frf(mech, N_GEN, FS_M)
            self._frf = frf
        return self._frf

    def __call__(self, id_ale, v_b):
        from model import toy_response
        fpath = self.result_dir / id_ale / 'response.npz'
        if fpath.exists():
            with np.load(fpath) as arr:
                return arr['t_vals'], arr['a_freq_time'], arr['meas_nodes']
        fpath.parent.mkdir(parents=True, exist_ok=True)
        seed = int.from_bytes(bytes(id_ale, 'utf-8'), 'big') % (2 ** 32)
        t_vals, accel = toy_response.generate_response(
            fpath, self._get_frf(), FS_M, v_b, seed)
        return t_vals, accel, np.asarray(toy_response.CANDIDATE_NODES)


# ── (b) lattice mapping: response -> acquisition -> correlation estimate ─────

def stage2corr_mapping(n_locations, DAQ_noise_rms, decimation_factor, tau_max,
                       duration, v_b, jid, result_dir, response_provider,
                       fixed_params=None, skip_existing=True,
                       cache_signal=False):
    '''
    Per-(aleatory, epistemic) lattice function: synthesized response ->
    channel selection -> acquisition chain (sensor, range, sampling,
    DAQ noise; reusing UQ_OMA.stage2mapping's steps with fixed sensor
    parameters) -> high-pass + Blackman-Tukey correlation at
    m_lags = ceil(tau_max * fs).

    Deliberate deviation from stage2mapping: the sensor setup is selected
    with a seed derived from the *epistemic* id, not the aleatory id — the
    algorithm requires spectral estimators to be constant along the aleatory
    axis so their estimates can be averaged (channel layout must not depend
    on aleatory parameters). Sequential-variability noise seeds remain
    aleatory-derived, as in the original.

    duration : float
        Recording length in seconds, passed straight through to
        ``Acquire.sample``. The frozen numeric value of the Imprecision
        variable T (vars_definition_weighted), constant along the aleatory
        axis like the other epistemic arguments.

    cache_signal : bool, optional
        Also persist the decimated, filtered signal (the input to the
        correlation step) as <result_dir>/<id_ale>/<id_epi>/signal.npz
        (float32) -- the data-driven / projection SSI analogue of corr.npz,
        consumed by load_decimated_signals / weighted_data_driven_identification
        (plan Deliverable 3, method='sd'). ``False`` (default) reproduces the
        original covariance-only lattice exactly.

    Returns the correlation matrix (n_l, n_r, m_lags) and caches it as
    <result_dir>/<id_ale>/<id_epi>/corr.npz (float32).
    '''
    from model.acquisition import Acquire, sensor_position
    from pyOMA.core.PreProcessingTools import PreProcessSignals

    if fixed_params is None:
        fixed_params = FIXED_PARAMS
    result_dir = Path(result_dir)

    id_ale, id_epi = jid.split('_')
    seed_ale = int.from_bytes(bytes(id_ale, 'utf-8'), 'big')
    seed_epi = int.from_bytes(bytes(id_epi, 'utf-8'), 'big')

    this_result_dir = result_dir / id_ale / id_epi
    corr_file = this_result_dir / 'corr.npz'
    signal_file = this_result_dir / 'signal.npz'
    needed = [corr_file] + ([signal_file] if cache_signal else [])
    if skip_existing and all(f.exists() for f in needed):
        try:
            with np.load(corr_file) as arr:
                return arr['corr_matrix']
        except Exception as e:
            logger.warning(repr(e))
            os.remove(corr_file)

    t_vals, accel, gen_nodes = response_provider(id_ale, v_b)

    # channel selection: constant along the aleatory axis (seed_epi)
    setups = sensor_position(int(n_locations), NUM_NODES, 'distributed')
    i_setup = seed_epi % setups.shape[0]
    sensor_nodes = setups[i_setup, :]

    gen_nodes = list(np.asarray(gen_nodes, dtype=int))
    quant = 2  # acceleration
    channel_defs = []
    signal = np.empty((2 * len(sensor_nodes), accel.shape[0]))
    for i_node, node in enumerate(sensor_nodes):
        node_ind = gen_nodes.index(int(node))
        for dof in (0, 1):  # lateral dofs (uy, uz), labeled x, y as in the
            # original response records where the axial dof is omitted
            channel_defs.append((int(node), dof, quant))
            signal[2 * i_node + dof, :] = accel[:, node_ind, dof]
    channel_defs = np.array(channel_defs, dtype=int)

    acqui = Acquire(t_vals, signal, None, channel_defs, jobname=jid)

    sensitivity_nominal = fixed_params['sensitivity_nominal']
    sensitivity_deviation = (fixed_params['sensitivity_deviation_percent']
                             / 100 * sensitivity_nominal)
    acqui.apply_sensor(DTC=fixed_params['DTC'],
                       sensitivity_nominal=sensitivity_nominal,
                       sensitivity_deviation=sensitivity_deviation,
                       spectral_noise_slope=fixed_params['spectral_noise_slope'],
                       noise_rms=fixed_params['sensor_noise_rms'],
                       seed=seed_ale)
    meas_range = acqui.estimate_meas_range(
        sample_dur=fixed_params['range_estimation_duration'],
        margin=fixed_params['range_estimation_margin'],
        seed=seed_ale)

    decimation_factor = int(decimation_factor)
    quantization_bits = fixed_params['quant_bit_factor'] * 4
    anti_aliasing_cutoff = (fixed_params['anti_aliasing_cutoff_factor']
                            * acqui.sampling_rate / decimation_factor)
    acqui.sample(dec_fact=decimation_factor, aa_cutoff=anti_aliasing_cutoff,
                 bits=quantization_bits, meas_range=meas_range,
                 duration=duration)
    # add noise here, because sampling (decimation) removes all noise again
    acqui.add_noise(noise_power=DAQ_noise_rms ** 2, seed=seed_ale)

    pd_kwargs = acqui.to_prep_data()
    ref_channels = np.where(channel_defs[:, 0] == 201)[0]
    assert ref_channels.shape[0] == 2

    prep_signals = PreProcessSignals(**pd_kwargs, ref_channels=ref_channels)
    # Fix transients from close-to-DC components in spectral (sensor) noise
    prep_signals.filter_signals(highpass=0.1)

    fs = prep_signals.sampling_rate
    m_lags = int(np.ceil(tau_max * fs))
    prep_signals.corr_blackman_tukey(m_lags)
    corr_matrix = prep_signals.corr_matrices_bt[0]

    this_result_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(corr_file,
                        corr_matrix=corr_matrix.astype(np.float32),
                        sampling_rate=fs, m_lags=m_lags,
                        channel_defs=channel_defs)
    if cache_signal:
        np.savez_compressed(signal_file,
                            signal=prep_signals.signals.astype(np.float32),
                            sampling_rate=fs, channel_defs=channel_defs)
    return corr_matrix


# ── local-parallel lattice dispatch ──────────────────────────────────────────

_WORKER_STATE = {}


def _init_worker(result_dir, mech_npz):
    _WORKER_STATE['provider'] = ToyResponseProvider(result_dir, mech_npz)


def _run_lattice_task(task):
    args, jid, result_dir = task
    try:
        stage2corr_mapping(jid=jid, result_dir=result_dir,
                           response_provider=_WORKER_STATE['provider'], **args)
        return jid, None
    except Exception as e:
        return jid, repr(e)


def run_lattice(poly_uq, arg_vars, result_dir, mech_npz, n_procs=None,
                cache_signal=False):
    '''
    Evaluate stage2corr_mapping on the full N_ale x N_epi lattice with local
    multiprocessing. Responses are generated once per aleatory sample
    (cached by the provider), correlation estimates once per lattice cell.

    cache_signal : bool, optional
        Forwarded to stage2corr_mapping: also cache the decimated signal per
        cell for the data-driven / projection SSI pipeline (method='sd').
    '''
    result_dir = Path(result_dir)
    inp = poly_uq.inp_samp_prim
    N_ale, N_epi = poly_uq.N_mcs_ale, poly_uq.N_mcs_epi

    # pre-generate responses sequentially (one shared FRF build)
    provider = ToyResponseProvider(result_dir, mech_npz)
    for n_ale in range(N_ale):
        provider(f'a{n_ale:04d}', inp['v_b'].iloc[n_ale])

    tasks = []
    for n_epi in range(N_epi):
        args_epi = {arg: inp[name].iloc[n_epi]
                    for arg, name in arg_vars.items() if arg != 'v_b'}
        args_epi['cache_signal'] = cache_signal
        for n_ale in range(N_ale):
            args = dict(args_epi, v_b=inp['v_b'].iloc[n_ale])
            jid = f'a{n_ale:04d}_e{n_epi:04d}'
            tasks.append((args, jid, result_dir))

    if n_procs is None:
        n_procs = max(1, (os.cpu_count() or 2) - 2)
    failures = []
    with multiprocessing.Pool(n_procs, initializer=_init_worker,
                              initargs=(result_dir, mech_npz)) as pool:
        for jid, err in pool.imap_unordered(_run_lattice_task, tasks,
                                            chunksize=8):
            if err is not None:
                failures.append((jid, err))
                logger.error(f'{jid}: {err}')
    if failures:
        raise RuntimeError(f'{len(failures)} lattice cells failed, '
                           f'first: {failures[0]}')


def load_corr_matrices(result_dir, n_epi, N_ale):
    '''Stack the N_ale cached correlation estimates of epistemic sample n_epi.'''
    result_dir = Path(result_dir)
    corr, fs = [], None
    for n_ale in range(N_ale):
        fpath = (result_dir / f'a{n_ale:04d}' / f'e{n_epi:04d}' / 'corr.npz')
        with np.load(fpath) as arr:
            corr.append(arr['corr_matrix'].astype(np.float64))
            fs = float(arr['sampling_rate'])
    return np.stack(corr, axis=0), fs


def load_decimated_signals(result_dir, n_epi, N_ale):
    '''List the N_ale cached decimated signals of epistemic sample n_epi
    (plan Deliverable 3; requires the lattice to have been run with
    stage2corr_mapping(..., cache_signal=True) / run_lattice(...,
    cache_signal=True)). A list, not a stacked array: signals need not all
    have the exact same length.'''
    result_dir = Path(result_dir)
    signals, fs = [], None
    for n_ale in range(N_ale):
        fpath = (result_dir / f'a{n_ale:04d}' / f'e{n_epi:04d}' / 'signal.npz')
        with np.load(fpath) as arr:
            signals.append(arr['signal'].astype(np.float64))
            fs = float(arr['sampling_rate'])
    return signals, fs


# ── (c) one weighted OMA run ─────────────────────────────────────────────────

def _dummy_prep_signals(n_l, ref_channels, fs):
    '''Minimal PreProcessSignals carrying the metadata VarSSIRef needs
    (channel counts, reference channels, sampling rate) — the signals
    themselves are never touched on the external-correlation path.'''
    from pyOMA.core.PreProcessingTools import PreProcessSignals
    signals = np.zeros((64, n_l))
    signals[0, :] = 1.0  # avoid degenerate all-zero channels
    return PreProcessSignals(signals, fs, ref_channels=list(ref_channels))


def weighted_modal_identification(corr_matrices, weights, model_order, fs,
                                  ref_channels=(0, 1), max_num_block_columns=80):
    '''
    One VarSSIRef run on externally provided per-aleatory-sample correlation
    estimates with importance weights: weighted subspace matrix, weighted
    Hankel covariance (Kish n_eff), fast variance propagation, modal
    parameters at the single sampled model order.

    Returns dict with arrays over the identified poles of that order:
    frequencies f, damping ratios d (percent), their standard deviations,
    mode shapes phi and the effective sample size n_eff.
    '''
    from pyOMA.core.VarSSIRef import VarSSIRef

    corr_matrices = np.asarray(corr_matrices)
    _, n_l, n_r, m_lags = corr_matrices.shape

    num_block_columns = min((m_lags - 1) // 2, max_num_block_columns)
    model_order = int(model_order)
    # the state-space order cannot exceed the subspace matrix rank
    model_order = min(model_order, n_r * num_block_columns,
                      n_l * num_block_columns)

    prep_dummy = _dummy_prep_signals(n_l, ref_channels, fs)
    varssi = VarSSIRef(prep_dummy)
    varssi.build_subspace_mat(num_block_columns=num_block_columns,
                              subspace_method='covariance',
                              weights=weights, corr_matrices=corr_matrices)
    varssi.compute_state_matrices(max_model_order=model_order + 1,
                                  lsq_method='pinv')
    varssi.prepare_sensitivities(variance_algo='fast')
    varssi.compute_modal_params(orders=[model_order])

    f_row = varssi.modal_frequencies[model_order, :model_order]
    keep = f_row > 0
    return {'f': f_row[keep],
            'd': varssi.modal_damping[model_order, :model_order][keep],
            'std_f': varssi.std_frequencies[model_order, :model_order][keep],
            'std_d': varssi.std_damping[model_order, :model_order][keep],
            'phi': varssi.mode_shapes[:, :model_order, model_order][:, keep],
            'n_eff': varssi.n_eff,
            'order': model_order}


def _signal_to_hankel_block(signal, num_block_rows, ref_channels, n_l):
    '''One external raw block-Hankel matrix (stacked [Y_minus; Y_plus]) from
    a single decimated signal, mirroring VarSSIRef._build_subspace_projection's
    internal per-block construction with num_blocks=1 (the whole signal is
    one "block") -- the projection-method analogue of a single aleatory
    sample's correlation estimate for the covariance method.'''
    n_r = len(ref_channels)
    total_time_steps = signal.shape[0]
    N = total_time_steps - 2 * num_block_rows
    Y_minus = np.zeros((num_block_rows * n_r, N))
    Y_plus = np.zeros(((num_block_rows + 1) * n_l, N))
    for ii in range(num_block_rows):
        Y_minus[(num_block_rows - ii - 1) * n_r:(num_block_rows - ii) * n_r, :] = \
            signal[ii:ii + N, ref_channels].T
    for ii in range(num_block_rows + 1):
        Y_plus[ii * n_l:(ii + 1) * n_l, :] = \
            signal[num_block_rows + ii:num_block_rows + ii + N].T
    return np.vstack((Y_minus, Y_plus))


def weighted_data_driven_identification(signals, weights, model_order,
                                        num_block_rows, fs,
                                        ref_channels=(0, 1)):
    '''
    One VarSSIRef run on externally provided per-aleatory-sample decimated
    signals with POST-HOC importance weights -- the data-driven /
    projection-method analogue of weighted_modal_identification, used for
    method='sd', weighting='posthoc'. ``num_block_rows`` mirrors the
    original study's SSIDataCV convention (m_lags // 2, m_lags = ceil(
    tau_max * fs); UQ_OMA.py stage2n3mapping) so results are comparable to
    the archived {f,d}_sd-* study.

    Identifies the *unweighted* projection subspace once from the external
    Hankel matrices (uniform block mean), computes its fast variance
    normally, then reweights mean and variance together via
    compute_modal_params_weighted. Its point estimate is that of the
    unweighted build (frozen-linearization semantics -- see that method's
    docstring and its projection-specific caveat: block samples are defined
    after the joint LQ normalization, so this is more approximate than the
    covariance method's post-hoc reweighting).

    See weighted_data_driven_identification_build for the build-time
    weighted alternative (weighting='build'), whose point estimate genuinely
    moves with the weights.

    Returns dict with the same keys as weighted_modal_identification: f, d,
    std_f, std_d, phi, n_eff (of the *given* weights, not the unweighted
    build), order.
    '''
    from pyOMA.core.VarSSIRef import VarSSIRef

    n_l = signals[0].shape[1]
    hankel_matrices = [_signal_to_hankel_block(s, num_block_rows, ref_channels, n_l)
                       for s in signals]

    model_order = int(model_order)
    n_r = len(ref_channels)
    # the state-space order cannot exceed the subspace matrix rank
    model_order = min(model_order, n_r * num_block_rows, n_l * num_block_rows)

    prep_dummy = _dummy_prep_signals(n_l, ref_channels, fs)
    varssi = VarSSIRef(prep_dummy)
    varssi.build_subspace_mat(num_block_columns=num_block_rows,
                              num_block_rows=num_block_rows,
                              subspace_method='projection',
                              hankel_matrices=hankel_matrices)
    varssi.compute_state_matrices(max_model_order=model_order + 1,
                                  lsq_method='pinv')
    varssi.prepare_sensitivities(variance_algo='fast')
    varssi.compute_modal_params_weighted(weights, orders=[model_order])
    _, n_eff = VarSSIRef._validate_weights(weights, len(signals))

    f_row = varssi.modal_frequencies[model_order, :model_order]
    keep = f_row > 0
    return {'f': f_row[keep],
            'd': varssi.modal_damping[model_order, :model_order][keep],
            'std_f': varssi.std_frequencies[model_order, :model_order][keep],
            'std_d': varssi.std_damping[model_order, :model_order][keep],
            'phi': varssi.mode_shapes[:, :model_order, model_order][:, keep],
            'n_eff': n_eff,
            'order': model_order}


def weighted_data_driven_identification_build(signals, weights, model_order,
                                              num_block_rows, fs,
                                              ref_channels=(0, 1)):
    '''
    One VarSSIRef run on externally provided per-aleatory-sample decimated
    signals with BUILD-TIME importance weights -- the data-driven /
    projection-method analogue of weighted_modal_identification, used for
    method='sd', weighting='build'.

    Uses VarSSIRef.build_subspace_mat's default (non-experimental) weighted
    projection: every block's own per-block/joint-LQ projection is built
    exactly as in the unweighted case, and the weights only pick a weighted
    mean over those independent per-block estimates -- the same structure
    as the covariance method's weighted mean -- so the point estimate
    genuinely moves with the weights and variance_algo='fast' applies
    normally (unlike the pre-LQ-weighted experimental_weighted_projection
    reading, which has no variance at all).

    See weighted_data_driven_identification for the post-hoc alternative
    (weighting='posthoc'), whose point estimate stays fixed at the
    unweighted mean.

    Returns dict with the same keys as weighted_modal_identification: f, d,
    std_f, std_d, phi, n_eff, order.
    '''
    from pyOMA.core.VarSSIRef import VarSSIRef

    n_l = signals[0].shape[1]
    hankel_matrices = [_signal_to_hankel_block(s, num_block_rows, ref_channels, n_l)
                       for s in signals]

    model_order = int(model_order)
    n_r = len(ref_channels)
    # the state-space order cannot exceed the subspace matrix rank
    model_order = min(model_order, n_r * num_block_rows, n_l * num_block_rows)

    prep_dummy = _dummy_prep_signals(n_l, ref_channels, fs)
    varssi = VarSSIRef(prep_dummy)
    varssi.build_subspace_mat(num_block_columns=num_block_rows,
                              num_block_rows=num_block_rows,
                              subspace_method='projection',
                              hankel_matrices=hankel_matrices,
                              weights=weights)
    varssi.compute_state_matrices(max_model_order=model_order + 1,
                                  lsq_method='pinv')
    varssi.prepare_sensitivities(variance_algo='fast')
    varssi.compute_modal_params(orders=[model_order])

    f_row = varssi.modal_frequencies[model_order, :model_order]
    keep = f_row > 0
    return {'f': f_row[keep],
            'd': varssi.modal_damping[model_order, :model_order][keep],
            'std_f': varssi.std_frequencies[model_order, :model_order][keep],
            'std_d': varssi.std_damping[model_order, :model_order][keep],
            'phi': varssi.mode_shapes[:, :model_order, model_order][:, keep],
            'n_eff': varssi.n_eff,
            'order': model_order}


def weighted_plscf_identification(corr_matrices, weights, model_order, fs,
                                  ref_channels=(0, 1)):
    '''
    One VarPLSCF run on externally provided per-aleatory-sample correlation
    estimates with BUILD-TIME importance weights -- the pLSCF analogue of
    weighted_modal_identification, used for method='cf', weighting='build'.
    Each aleatory correlation estimate is treated as one pLSCF "block"
    (VarPLSCF.build_half_spectra's num_blocks/corr_matrices, its
    block-covariance counterpart to VarSSIRef's weighted subspace matrix);
    weights enter the half-spectrum mean and its block-covariance the same
    way (1/n_eff, generalizing Steffensen et al. 2025 Eq. 26).

    Returns dict with the same keys as weighted_modal_identification: f, d,
    std_f, std_d, phi, n_eff, order. n_eff is Kish's effective sample size
    of ``weights`` (VarPLSCF.n_eff after build_half_spectra).
    '''
    from pyOMA.core.VarPLSCF import VarPLSCF

    corr_matrices = np.asarray(corr_matrices)
    n_b, n_l, n_r, m_lags = corr_matrices.shape

    model_order = int(model_order)
    # compute_modal_params(max_model_order) requires max_model_order <=
    # nperseg - 1 (PLSCF._setup_compute_params), and this class always
    # calls it with model_order + 1 (its loop covers orders 1..model_order
    # inclusive), so model_order itself is capped at nperseg - 2; nperseg
    # is the full lag count here.
    model_order = min(model_order, m_lags - 2)

    prep_dummy = _dummy_prep_signals(n_l, ref_channels, fs)
    varplscf = VarPLSCF(prep_dummy)
    varplscf.build_half_spectra(nperseg=m_lags, num_blocks=n_b,
                                weights=weights, corr_matrices=corr_matrices)
    varplscf.compute_modal_params(model_order + 1, modal_contrib=False)

    f_row = varplscf.modal_frequencies[model_order]
    keep = f_row > 0
    return {'f': f_row[keep],
            'd': varplscf.modal_damping[model_order][keep],
            'std_f': varplscf.std_frequencies[model_order][keep],
            'std_d': varplscf.std_damping[model_order][keep],
            'phi': varplscf.mode_shapes[:, :, model_order][:, keep],
            'n_eff': varplscf.n_eff,
            'order': model_order}


# ── (c2) shared post-hoc block-weight reweighting (all 3 methods) ───────────

def _build_unweighted_sc(corr_matrices, model_order, fs, ref_channels=(0, 1),
                         max_num_block_columns=80):
    '''One unweighted VarSSIRef covariance build with Tier A variance-factor
    caching, for weighting='posthoc': build once per epistemic sample, then
    apply_block_weights per Imprecision hypercube without rebuilding.'''
    from pyOMA.core.VarSSIRef import VarSSIRef

    corr_matrices = np.asarray(corr_matrices)
    _, n_l, n_r, m_lags = corr_matrices.shape
    num_block_columns = min((m_lags - 1) // 2, max_num_block_columns)
    model_order = int(model_order)
    model_order = min(model_order, n_r * num_block_columns, n_l * num_block_columns)

    prep_dummy = _dummy_prep_signals(n_l, ref_channels, fs)
    varssi = VarSSIRef(prep_dummy)
    varssi.build_subspace_mat(num_block_columns=num_block_columns,
                              subspace_method='covariance',
                              weights=None, corr_matrices=corr_matrices)
    varssi.compute_state_matrices(max_model_order=model_order + 1,
                                  lsq_method='pinv')
    varssi.prepare_sensitivities(variance_algo='fast')
    varssi.compute_modal_params(orders=[model_order], cache_variance_factors=True)
    return varssi, model_order


def _build_unweighted_sd(signals, model_order, num_block_rows, fs,
                         ref_channels=(0, 1)):
    '''Projection analogue of _build_unweighted_sc.'''
    from pyOMA.core.VarSSIRef import VarSSIRef

    n_l = signals[0].shape[1]
    hankel_matrices = [_signal_to_hankel_block(s, num_block_rows, ref_channels, n_l)
                       for s in signals]
    model_order = int(model_order)
    n_r = len(ref_channels)
    model_order = min(model_order, n_r * num_block_rows, n_l * num_block_rows)

    prep_dummy = _dummy_prep_signals(n_l, ref_channels, fs)
    varssi = VarSSIRef(prep_dummy)
    varssi.build_subspace_mat(num_block_columns=num_block_rows,
                              num_block_rows=num_block_rows,
                              subspace_method='projection',
                              hankel_matrices=hankel_matrices)
    varssi.compute_state_matrices(max_model_order=model_order + 1,
                                  lsq_method='pinv')
    varssi.prepare_sensitivities(variance_algo='fast')
    varssi.compute_modal_params(orders=[model_order], cache_variance_factors=True)
    return varssi, model_order


def _build_unweighted_cf(corr_matrices, model_order, fs, ref_channels=(0, 1)):
    '''pLSCF analogue of _build_unweighted_sc.'''
    from pyOMA.core.VarPLSCF import VarPLSCF

    corr_matrices = np.asarray(corr_matrices)
    n_b, n_l, n_r, m_lags = corr_matrices.shape
    model_order = int(model_order)
    # see weighted_plscf_identification: max_model_order = model_order + 1
    # must stay <= nperseg - 1, so model_order itself caps at nperseg - 2.
    model_order = min(model_order, m_lags - 2)

    prep_dummy = _dummy_prep_signals(n_l, ref_channels, fs)
    varplscf = VarPLSCF(prep_dummy)
    varplscf.build_half_spectra(nperseg=m_lags, num_blocks=n_b,
                                corr_matrices=corr_matrices)
    varplscf.compute_modal_params(model_order + 1, modal_contrib=False,
                                  cache_variance_factors=True)
    return varplscf, model_order


_POSTHOC_BUILD_FUNS = {'sc': _build_unweighted_sc,
                       'sd': _build_unweighted_sd,
                       'cf': _build_unweighted_cf}


def _extract_pole_dict(obj, model_order, n_eff, plscf=False):
    '''Pole-dict extraction shared by the posthoc path (after
    apply_block_weights) -- mirrors the trailing block of
    weighted_modal_identification / weighted_data_driven_identification(_build)
    / weighted_plscf_identification. VarSSIRef's per-order arrays are
    (max_order+1, max_order) and truncate a row to :model_order; VarPLSCF's
    are (max_model_order, max_modes) and use the whole row (max_modes
    already bounds it).'''
    if plscf:
        f_row = obj.modal_frequencies[model_order]
        d_row = obj.modal_damping[model_order]
        std_f_row = obj.std_frequencies[model_order]
        std_d_row = obj.std_damping[model_order]
        phi = obj.mode_shapes[:, :, model_order]
    else:
        f_row = obj.modal_frequencies[model_order, :model_order]
        d_row = obj.modal_damping[model_order, :model_order]
        std_f_row = obj.std_frequencies[model_order, :model_order]
        std_d_row = obj.std_damping[model_order, :model_order]
        phi = obj.mode_shapes[:, :model_order, model_order]
    keep = f_row > 0
    return {'f': f_row[keep], 'd': d_row[keep],
            'std_f': std_f_row[keep], 'std_d': std_d_row[keep],
            'phi': phi[:, keep], 'n_eff': n_eff, 'order': model_order}


# ── (d) Algorithm alg:proposed main loop ─────────────────────────────────────

def elimination_mask(poly_uq, i_imp, n_epi):
    '''
    Per-hypercube weight elimination (Algorithm alg:proposed): aleatory
    samples whose secondary-Variability values fall outside the Imprecision
    focal bounds of hypercube i_imp receive zero weight. No Imprecision
    variable of this case study has secondary-Variability focal bounds, so
    the mask is all-True (documented degeneracy); the generic case would
    evaluate poly_uq.hypercube_sample_indices on those bound variables.
    '''
    return np.ones(poly_uq.N_mcs_ale, dtype=bool)


def compute_weights(poly_uq, i_imp, n_epi):
    '''
    Incompleteness-conditioned importance weights for epistemic sample
    n_epi and Imprecision hypercube i_imp: freeze the secondary
    Incompleteness variables to their sampled values (as in
    PolyUQ.optimize_inc) and evaluate the renormalized aleatory pdf on the
    v_b samples, then apply the per-hypercube elimination mask.
    '''
    for var in poly_uq.vars_inc:
        var.freeze(poly_uq.inp_suppl_epi[var.name].iloc[n_epi])
    weights = poly_uq.probabilities_imp(i_imp)
    weights = weights * elimination_mask(poly_uq, i_imp, n_epi)
    total = np.sum(weights)
    if total <= 0:
        raise ValueError(f'All weights eliminated in hypercube {i_imp} '
                         f'at epistemic sample {n_epi}.')
    return weights / total


_BUILD_IDENTIFICATION_FUNS = {
    'sc': weighted_modal_identification,
    'sd': weighted_data_driven_identification_build,
    'cf': weighted_plscf_identification,
}
_PLSCF_METHODS = {'cf'}


def _load_cell_data(poly_uq, result_dir, method, n_epi, max_num_block_rows):
    '''Per-epistemic-sample lattice data for ``method``: 'sc'/'cf' load the
    cached correlation stack (both consume corr_matrices), 'sd' loads the
    cached decimated signals and derives num_block_rows from tau_max.'''
    inp = poly_uq.inp_samp_prim
    if method in ('sc', 'cf'):
        cell_data, fs = load_corr_matrices(result_dir, n_epi, poly_uq.N_mcs_ale)
        extra_args = ()
    else:
        cell_data, fs = load_decimated_signals(result_dir, n_epi, poly_uq.N_mcs_ale)
        m_lags = int(np.ceil(inp['tau_max'].iloc[n_epi] * fs))
        extra_args = (min(m_lags // 2, max_num_block_rows),)
    return cell_data, fs, extra_args


def run_weighted_identification(poly_uq, result_dir, method='sc', weighting='build',
                                identification_fun=None,
                                max_num_block_rows=80, convention='substitution'):
    '''
    Main loop of Algorithm alg:proposed: for each epistemic sample and each
    Imprecision hypercube, compute the weights and run the weighted
    identification once — with a dedupe fast path across hypercubes that
    yield identical weights (always the case here, see module docstring).

    ``method`` selects the OMA estimator and its cached lattice data:

    'sc'
        Covariance-driven SSI (weighted_modal_identification /
        weighted_data_driven_identification_build's covariance sibling) on
        cached correlation estimates (load_corr_matrices / corr.npz).
    'sd'
        Data-driven / projection SSI on cached decimated signals
        (load_decimated_signals / signal.npz) -- requires the lattice to
        have been run with run_lattice(..., cache_signal=True) /
        stage2corr_mapping(..., cache_signal=True).
    'cf'
        pLSCF (VarPLSCF) on the same cached correlation estimates as 'sc'.

    ``weighting`` selects how the Incompleteness-conditioned importance
    weights enter each method (the two case-study "methods" 2/3 of the
    dissertation outlook -- see the module docstring):

    'build'
        Build-time weighted identification, one build per distinct weight
        vector within an epistemic sample's hypercubes (dedupe cache as
        before): weighted_modal_identification (sc),
        weighted_data_driven_identification_build (sd),
        weighted_plscf_identification (cf). The point estimate itself moves
        with the weights.
    'posthoc'
        Post-hoc block-weight reweighting (Tier A/B): builds ONE unweighted
        identification per epistemic sample (cached across all its
        hypercubes, regardless of weight-vector dedupe -- this is the
        cheap part of the reweighting), then for each hypercube calls
        apply_block_weights on that single build to refresh only the
        variances. The point estimate stays fixed at the unweighted mean
        for every hypercube of that epistemic sample; ``identification_fun``
        is not used here (the internal _build_unweighted_* / apply_block_weights
        pair replaces it).

    Returns a pole database: list of dicts with keys n_epi, i_imp and the
    identification outputs.
    '''
    if method not in ('sc', 'sd', 'cf'):
        raise ValueError(f"'method' must be 'sc', 'sd' or 'cf', got {method!r}.")
    if weighting not in ('build', 'posthoc'):
        raise ValueError(f"'weighting' must be 'build' or 'posthoc', got {weighting!r}.")
    if identification_fun is None:
        identification_fun = _BUILD_IDENTIFICATION_FUNS[method]

    N_epi = poly_uq.N_mcs_epi
    n_imp_hyc = len(poly_uq.imp_hyc_foc_inds)
    inp = poly_uq.inp_samp_prim
    plscf = method in _PLSCF_METHODS

    pole_db = []
    for n_epi in range(N_epi):
        model_order = int(inp['model_order'].iloc[n_epi])
        cell_data, fs, extra_args = _load_cell_data(
            poly_uq, result_dir, method, n_epi, max_num_block_rows)

        if weighting == 'build':
            cache = {}
            for i_imp in range(n_imp_hyc):
                weights = compute_weights(poly_uq, i_imp, n_epi)
                key = weights.tobytes()
                if key not in cache:
                    cache[key] = identification_fun(
                        cell_data, weights, model_order, *extra_args, fs)
                pole_db.append(dict(cache[key], n_epi=n_epi, i_imp=i_imp,
                                    weights_id=len(cache) - 1))
            logger.info(f'Epistemic sample {n_epi}: {len(cache)} distinct '
                        f'weighted identification(s) for {n_imp_hyc} hypercubes.')
        else:
            build_fun = _POSTHOC_BUILD_FUNS[method]
            obj, resolved_order = build_fun(cell_data, model_order, *extra_args, fs)
            cache = {}
            for i_imp in range(n_imp_hyc):
                weights = compute_weights(poly_uq, i_imp, n_epi)
                key = weights.tobytes()
                if key not in cache:
                    obj.apply_block_weights(weights=weights, convention=convention)
                    _, n_eff = obj._validate_weights(weights, poly_uq.N_mcs_ale)
                    cache[key] = _extract_pole_dict(
                        obj, resolved_order, n_eff, plscf=plscf)
                pole_db.append(dict(cache[key], n_epi=n_epi, i_imp=i_imp,
                                    weights_id=len(cache) - 1))
            logger.info(f'Epistemic sample {n_epi}: {len(cache)} distinct '
                        f'post-hoc reweighting(s) for {n_imp_hyc} hypercubes.')
    return pole_db


# ── (e) global pole clustering ───────────────────────────────────────────────

def cluster_modes_weighted(pole_db, f_min=0.1, f_max=5.0, d_max=20.0,
                           min_samples=0.005, xi=0.05, min_cluster_size=0.02,
                           max_eps=0.0004, eps=0.0003):
    '''
    One global OPTICS pass over all identified poles of all (n_epi, i_imp)
    cells, adapted from UQ_OMA.cluster_modes: poles are mapped onto the unit
    disk via lamda = exp((-zeta*omega + i*omega*sqrt(1-zeta^2)) / nyq_max)
    and clustered on (Re, Im); hard filters 0.1 < f < nyq(f_s) and d < 20 %.
    Standard deviations are carried through.

    Returns (labels, pole_table) where pole_table is a DataFrame with one
    row per retained pole (columns f, d, std_f, std_d, n_eff, n_epi, i_imp)
    and labels assigns each row a cluster (-1 = noise). n_eff (Kish) is a
    property of the identification's weights, constant across all poles of
    one (n_epi, i_imp) cell; carried through for the D2 aleatory-CDF
    reconstruction (sigma_pop = std * sqrt(n_eff), see assemble_theta).
    '''
    from sklearn.cluster import OPTICS, cluster_optics_dbscan

    rows = []
    for entry in pole_db:
        f, d = entry['f'], entry['d']
        std_f, std_d = entry['std_f'], entry['std_d']
        n_eff = entry['n_eff']
        nyq = entry.get('fs', None)
        keep = (f > f_min) & (f < f_max) & (~np.isnan(f)) & (d < d_max) & (d > 0)
        for j in np.where(keep)[0]:
            rows.append((f[j], d[j], std_f[j], std_d[j], n_eff,
                         entry['n_epi'], entry['i_imp']))
    pole_table = pd.DataFrame(rows, columns=['f', 'd', 'std_f', 'std_d',
                                             'n_eff', 'n_epi', 'i_imp'])
    if not len(pole_table):
        return np.empty(0, dtype=int), pole_table

    omega = pole_table['f'].values * 2 * np.pi
    zeta = pole_table['d'].values / 100
    mu = -zeta * omega + 1j * omega * np.sqrt(1 - zeta ** 2)
    lamda = np.exp(mu / NYQ_MAX)
    X = np.hstack((lamda.real[:, np.newaxis], lamda.imag[:, np.newaxis]))

    clust = OPTICS(min_samples=min_samples, xi=xi,
                   min_cluster_size=min_cluster_size, max_eps=max_eps,
                   n_jobs=-1)
    clust.fit(X)
    labels = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=eps)
    return labels, pole_table


def assemble_theta(labels, pole_table, N_epi, n_imp_hyc):
    '''
    Per cluster (mode of interest), assemble the statistic lattice
    Theta[q, n_e] for frequency and damping (NaN where the mode was not
    found in that cell); multiple poles of one cluster in the same cell are
    averaged. Returns dict cluster_label -> dict of (n_imp_hyc, N_epi)
    arrays f, d, std_f, std_d, n_eff, sigma_pop_f, sigma_pop_d.

    sigma_pop_{f,d} (plan Deliverable 2) un-shrink std_{f,d} — the standard
    *error of the (weighted) mean* subspace estimate — back to the
    population aleatory standard deviation the weighting collapsed:
    sigma_pop = std * sqrt(n_eff). Feeds the parametric aleatory-CDF
    reconstruction in parametric_cdf_statistic_level_polyuq.
    '''
    theta = {}
    for label in np.unique(labels):
        if label < 0:
            continue
        sel = pole_table.iloc[labels == label]
        arrs = {key: np.full((n_imp_hyc, N_epi), np.nan)
                for key in ('f', 'd', 'std_f', 'std_d', 'n_eff')}
        counts = np.zeros((n_imp_hyc, N_epi))
        for _, row in sel.iterrows():
            q, n_e = int(row['i_imp']), int(row['n_epi'])
            for key in arrs:
                if counts[q, n_e] == 0:
                    arrs[key][q, n_e] = row[key]
                else:  # running mean over multiple poles in one cell
                    arrs[key][q, n_e] += ((row[key] - arrs[key][q, n_e])
                                          / (counts[q, n_e] + 1))
            counts[q, n_e] += 1
        arrs['count'] = counts
        arrs['sigma_pop_f'] = arrs['std_f'] * np.sqrt(arrs['n_eff'])
        arrs['sigma_pop_d'] = arrs['std_d'] * np.sqrt(arrs['n_eff'])
        theta[int(label)] = arrs
    return theta


# ── (f) statistic-level Imprecision/Incompleteness processing ────────────────

def _primary_copy(var):
    '''Fresh primary=True MassFunction with the focal sets and masses of
    ``var`` (used to lift the secondary Incompleteness variables into the
    statistic-level instance, where their hypercubes become part of H^c_r).'''
    focals = []
    for _, low, high in var._focals:
        if isinstance(high, float) and np.isnan(high):
            focals.append((low,))
        else:
            focals.append((low, high))
    copy = MassFunction(var.name, focals,
                        np.array(var.masses, dtype=float).ravel(),
                        primary=True)
    copy.pretty_name = getattr(var, 'pretty_name', var.name)
    return copy


def _statistic_level_inputs(poly_uq):
    '''
    Shared surrogate inputs for the statistic-level PolyUQ instance of a
    clustered mode: the Imprecision variables, the primary Incompleteness
    variable (DAQ_noise_rms) and — required to make the Incompleteness
    hypercube bounds meaningful — the secondary Incompleteness values
    (c_vb, lamda_vb) that drove the weights, all as primary variables of a
    fresh instance (their cartesian focal products are the combined
    hypercubes H^i_q x H^c_r, 72 in this case study). Used by both
    statistic_level_polyuq (point estimate) and
    parametric_cdf_statistic_level_polyuq (D2 aleatory CDF).
    '''
    vars_stat = [_primary_copy(var) for var in poly_uq.vars_epi]
    inp_names_prim = [var.name for var in poly_uq.vars_imp]
    inp_names_sec = [var.name for var in poly_uq.vars_inc]
    N_epi = poly_uq.N_mcs_epi
    x_t = pd.concat(
        [poly_uq.inp_samp_prim[inp_names_prim].iloc[:N_epi].reset_index(drop=True),
         poly_uq.inp_suppl_epi[inp_names_sec].iloc[:N_epi].reset_index(drop=True)],
        axis=1)
    return vars_stat, x_t


def statistic_level_polyuq(poly_uq, theta_mode, quantity='f'):
    '''
    Build the statistic-level PolyUQ instance for one clustered mode. The
    output samples are Theta[0, n_e]: q-independent due to the documented
    weight-elimination degeneracy, so a single surrogate over the epistemic
    samples covers all Imprecision hypercubes.

    Call .estimate_imp() on the returned instance; its imp_foc[0] are the
    focal intervals F^Theta_{q,r} with masses imp_hyc_mass.
    '''
    vars_stat, x_t = _statistic_level_inputs(poly_uq)

    out = np.asarray(theta_mode[quantity][0, :], dtype=np.float64)
    if np.all(np.isnan(out)):
        raise ValueError('The mode was not found in any epistemic sample.')

    return PolyUQ.from_propagated_samples(
        vars_stat, x_t, out[np.newaxis, :],
        out_name=f'theta_{quantity}')


def parametric_cdf_statistic_level_polyuq(poly_uq, theta_mode, quantity,
                                          target_probabilities):
    '''
    Statistic-level PolyUQ instances for the parametric aleatory CDF of one
    clustered mode (plan Deliverable 2). The weighted estimator never keeps
    individual aleatory correlation samples past identification, so the
    empirical weighted-quantile route (polyuq.stat_fun_cdf, used by the
    original per-sample-OMA study) is not available; instead the aleatory
    distribution of the modal parameter at each epistemic sample n_e is
    taken to be Normal(f[n_e], sigma_pop_f[n_e]) (sigma_pop_f = std_f *
    sqrt(n_eff), the un-shrunk population aleatory std — see
    assemble_theta), and its inverse CDF is evaluated at
    target_probabilities to give one output row per probability level, each
    interval-optimized over the same combined hypercubes as
    statistic_level_polyuq.

    target_probabilities is clipped to (1e-4, 1 - 1e-4) — PolyUQ's own
    default support-truncation percentiles (from_propagated_samples,
    PolyUQ.support) — before evaluating the Normal ppf, which is +-inf at
    the exact 0/1 endpoints; the archived study's empirical quantile
    function has bounded support instead, so its exact endpoints are finite
    (the observed sample min/max).

    Returns a list of PolyUQ instances, one per (clipped) target
    probability, in the given order; imp_hyc_mass is identical across the
    list (same hypercubes and masses every time — only the output values
    differ), so callers typically call .estimate_imp() on each and stack
    the resulting imp_foc[0] into an (n_stat, n_hyc, 2) array (the archived
    study's polyuq_cdf_inc.npz layout).
    '''
    from scipy.stats import norm

    vars_stat, x_t = _statistic_level_inputs(poly_uq)

    f_row = np.asarray(theta_mode[quantity][0, :], dtype=np.float64)
    sigma_row = np.asarray(theta_mode[f'sigma_pop_{quantity}'][0, :],
                           dtype=np.float64)
    if np.all(np.isnan(f_row)):
        raise ValueError('The mode was not found in any epistemic sample.')

    p_eval = np.clip(np.asarray(target_probabilities, dtype=np.float64),
                     1e-4, 1 - 1e-4)
    # scipy's norm.ppf(scale=0) is nan (not the degenerate point mass), so
    # handle sigma_pop -> 0 explicitly: the aleatory CDF collapses to the
    # point estimate at every probability level.
    degenerate = sigma_row <= 0
    safe_sigma = np.where(degenerate, 1.0, sigma_row)
    instances = []
    for p in p_eval:
        out_row = np.where(degenerate, f_row, norm.ppf(p, loc=f_row, scale=safe_sigma))
        instances.append(PolyUQ.from_propagated_samples(
            vars_stat, x_t, out_row[np.newaxis, :],
            out_name=f'theta_{quantity}_cdf'))
    return instances


# ── (g) full pipeline ────────────────────────────────────────────────────────

def run_weighted_uq_pipeline(result_dir, mech_npz,
                             N_ale=100, N_epi=200, seed=1509,
                             n_procs=None, skip_lattice=False,
                             min_found=0.1, method='sc', weighting='build',
                             poly_uq=None):
    '''
    Complete reduced case study: sampling, response generation, local
    lattice, weighted identifications, clustering, statistic-level focal
    intervals. The shared lattice cache (a####/e####/{corr,signal}.npz) is
    written directly under result_dir; reruns skip completed cells. This
    combination's own outputs (pole_db, weighted_theta, weighted_focals_*)
    are written to result_dir/f'{method}_{weighting}'. Clusters found in
    fewer than ``min_found`` of the epistemic samples (spurious poles) are
    reported but not interval-optimized.

    method : {'sc', 'sd', 'cf'}, optional
        OMA estimator, forwarded to run_weighted_identification. 'sd' also
        makes the lattice cache decimated signals (run_lattice(...,
        cache_signal=True)) instead of only correlation estimates.
    weighting : {'build', 'posthoc'}, optional
        Forwarded to run_weighted_identification: build-time weighted
        identification ("OMA-enabled PolyUQ") vs post-hoc block-weight
        reweighting ("OMA-optimized PolyUQ").
    poly_uq : PolyUQ, optional
        Reuse an already-sampled instance instead of resampling -- combined
        with ``skip_lattice=True``, lets run_weighted_uq_pipeline_all share
        one sampling+lattice run across all (method, weighting)
        combinations. ``N_ale``/``N_epi``/``seed`` are ignored when given.

    Returns (poly_uq, pole_db, theta, results) where results maps cluster
    labels to {'f'/'d': {'imp_foc', 'imp_hyc_mass', 'intp_errors'}}.
    '''
    import pickle

    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    combo_dir = result_dir / f'{method}_{weighting}'
    combo_dir.mkdir(parents=True, exist_ok=True)

    vars_ale, vars_epi, arg_vars, _ = vars_definition_weighted()
    if poly_uq is None:
        poly_uq = PolyUQ(vars_ale, vars_epi, dim_ex='cartesian')
        poly_uq.sample_qmc(N_mcs_ale=N_ale, N_mcs_epi=N_epi, seed=seed)
    N_epi = poly_uq.N_mcs_epi

    if not skip_lattice:
        run_lattice(poly_uq, arg_vars, result_dir, mech_npz, n_procs=n_procs,
                    cache_signal=(method == 'sd'))

    pole_db = run_weighted_identification(poly_uq, result_dir, method=method,
                                          weighting=weighting)
    with open(combo_dir / 'pole_db.pkl', 'wb') as f:
        pickle.dump(pole_db, f)

    n_imp_hyc = len(poly_uq.imp_hyc_foc_inds)
    labels, pole_table = cluster_modes_weighted(pole_db)
    theta = assemble_theta(labels, pole_table, N_epi, n_imp_hyc)
    np.savez(combo_dir / 'weighted_theta.npz',
             labels=labels, **{f'theta_{label}_{key}': theta[label][key]
                               for label in theta
                               for key in ('f', 'd', 'std_f', 'std_d', 'n_eff',
                                          'sigma_pop_f', 'sigma_pop_d')})

    results = {}
    for label, theta_mode in sorted(theta.items()):
        found = np.mean(~np.isnan(theta_mode['f'][0]))
        med_f = np.nanmedian(theta_mode['f'])
        if found < min_found:
            logger.info(f'Cluster {label} (median {med_f:.3f} Hz): found in '
                        f'only {found * 100:.0f}% of epistemic samples — '
                        'skipping interval optimization.')
            continue
        logger.info(f'Cluster {label} (median {med_f:.3f} Hz, found in '
                    f'{found * 100:.0f}%): statistic-level processing...')
        results[label] = {}
        for quantity in ('f', 'd'):
            try:
                pq_stat = statistic_level_polyuq(poly_uq, theta_mode, quantity)
                imp_foc, _, intp_errors, _, _ = pq_stat.estimate_imp(
                    interp_fun='rbf', opt_meth='genetic')
            except Exception as e:
                logger.error(f'Cluster {label}, {quantity}: statistic-level '
                             f'processing failed: {e!r}')
                continue
            results[label][quantity] = {'imp_foc': imp_foc[0],
                                        'imp_hyc_mass': pq_stat.imp_hyc_mass,
                                        'intp_errors': intp_errors}
            np.savez(combo_dir / f'weighted_focals_{label}_{quantity}.npz',
                     imp_foc=imp_foc[0], imp_hyc_mass=pq_stat.imp_hyc_mass,
                     intp_errors=intp_errors, median=np.nanmedian(
                         theta_mode[quantity]))
    return poly_uq, pole_db, theta, results


def run_weighted_uq_pipeline_all(result_dir, mech_npz,
                                 N_ale=100, N_epi=200, seed=1509,
                                 n_procs=None, min_found=0.1,
                                 methods=('sc', 'sd', 'cf'),
                                 weightings=('build', 'posthoc')):
    '''
    Runs every (method, weighting) combination (6 by default) of Algorithm
    alg:proposed's two axes: OMA estimator (method: 'sc' VarSSIRef-
    covariance, 'sd' VarSSIRef-projection, 'cf' VarPLSCF) x weighting
    mechanism (weighting: 'build' build-time weighted identification =
    "OMA-enabled PolyUQ", 'posthoc' post-hoc block-weight reweighting =
    "OMA-optimized PolyUQ"). Sampling and the lattice (response ->
    acquisition -> correlation/signal, run_weighted_uq_pipeline's expensive,
    method/weighting-independent part) run ONCE and are shared across every
    combination, with cache_signal=True whenever 'sd' is among ``methods``
    (the only method needing the extra decimated-signal cache). Each
    combination's own identification, clustering and statistic-level
    outputs are written to result_dir/f'{method}_{weighting}' (see
    run_weighted_uq_pipeline).

    Returns dict {(method, weighting): (poly_uq, pole_db, theta, results)},
    one entry per combination, in the order run.
    '''
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    vars_ale, vars_epi, arg_vars, _ = vars_definition_weighted()
    poly_uq = PolyUQ(vars_ale, vars_epi, dim_ex='cartesian')
    poly_uq.sample_qmc(N_mcs_ale=N_ale, N_mcs_epi=N_epi, seed=seed)

    run_lattice(poly_uq, arg_vars, result_dir, mech_npz, n_procs=n_procs,
               cache_signal=('sd' in methods))

    all_results = {}
    for method in methods:
        for weighting in weightings:
            logger.info(f'Running method={method!r}, weighting={weighting!r}...')
            all_results[(method, weighting)] = run_weighted_uq_pipeline(
                result_dir, mech_npz, min_found=min_found, method=method,
                weighting=weighting, poly_uq=poly_uq, skip_lattice=True)
    return all_results


def run_weighted_cdf_reconstruction(poly_uq, theta, result_dir, n_stat=40,
                                    min_found=0.1, labels=None):
    '''
    Deliverable 2: reconstruct each clustered mode's aleatory CDF from
    VarSSIRef's variance and persist weighted_cdf_{label}_{f,d}.npz, in the
    (n_stat, n_hyc, 2) + (n_hyc,) layout of the archived study's
    polyuq_cdf_inc.npz (comparable via D1's plotting, never against a
    confidence interval — see plan D2 rationale). Reuses cached poly_uq/
    theta (e.g. from run_weighted_uq_pipeline, or re-derived cheaply from a
    saved pole_db.pkl via cluster_modes_weighted + assemble_theta) — no
    OMA re-identification.

    n_stat estimate_imp() calls per mode/quantity (genetic optimization
    over the combined hypercubes each time) make this the expensive part of
    D2; restrict ``labels`` to a subset for a quick check before a full
    n_stat=40 x all-clusters run.

    Returns dict cluster_label -> {'f'/'d': {'imp_foc' (n_stat, n_hyc, 2),
    'imp_hyc_mass' (n_hyc,), 'target_probabilities'}}.
    '''
    result_dir = Path(result_dir)
    target_probabilities = np.linspace(0, 1, n_stat)

    results = {}
    for label, theta_mode in sorted(theta.items()):
        if labels is not None and label not in labels:
            continue
        found = np.mean(~np.isnan(theta_mode['f'][0]))
        if found < min_found:
            continue
        results[label] = {}
        for quantity in ('f', 'd'):
            try:
                instances = parametric_cdf_statistic_level_polyuq(
                    poly_uq, theta_mode, quantity, target_probabilities)
                foc_rows = []
                mass = None
                for pq_stat in instances:
                    imp_foc, _, _, _, _ = pq_stat.estimate_imp(
                        interp_fun='rbf', opt_meth='genetic')
                    foc_rows.append(imp_foc[0])
                    mass = pq_stat.imp_hyc_mass
                imp_foc_cdf = np.stack(foc_rows, axis=0)
            except Exception as e:
                logger.error(f'Cluster {label}, {quantity}: CDF '
                            f'reconstruction failed: {e!r}')
                continue
            results[label][quantity] = {
                'imp_foc': imp_foc_cdf, 'imp_hyc_mass': mass,
                'target_probabilities': target_probabilities}
            np.savez(result_dir / f'weighted_cdf_{label}_{quantity}.npz',
                     imp_foc=imp_foc_cdf, imp_hyc_mass=mass,
                     target_probabilities=target_probabilities)
            logger.info(f'Cluster {label}, {quantity}: CDF reconstruction '
                       f'done ({n_stat} probability levels).')
    return results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('result_dir', type=Path)
    parser.add_argument('--mech-npz', type=Path, default=Path(
        os.environ.get('POLYUQ_DATA_DIR', '/home/womo1998/Projects/uq_oma_a')
    ) / 'samples' / 'mechanical.npz')
    parser.add_argument('--n-ale', type=int, default=100)
    parser.add_argument('--n-epi', type=int, default=200)
    parser.add_argument('--n-procs', type=int, default=None)
    parser.add_argument('--seed', type=int, default=1509)
    parser.add_argument('--method', choices=('sc', 'sd', 'cf'), default='sc')
    parser.add_argument('--weighting', choices=('build', 'posthoc'), default='build')
    parser.add_argument('--all-combinations', action='store_true',
                        help='Run all 3 methods x 2 weightings instead of the '
                             'single --method/--weighting combination.')
    args = parser.parse_args()
    if args.all_combinations:
        run_weighted_uq_pipeline_all(args.result_dir, args.mech_npz,
                                     N_ale=args.n_ale, N_epi=args.n_epi,
                                     seed=args.seed, n_procs=args.n_procs)
    else:
        run_weighted_uq_pipeline(args.result_dir, args.mech_npz,
                                 N_ale=args.n_ale, N_epi=args.n_epi,
                                 seed=args.seed, n_procs=args.n_procs,
                                 method=args.method, weighting=args.weighting)
