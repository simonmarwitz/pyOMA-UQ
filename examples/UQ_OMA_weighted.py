'''
OMA as the weighted statistical estimator — reduced case study.

Implements the dissertation outlook "Toward OMA-Specific PolyUQ Processing
Strategies" (Algorithm alg:proposed): instead of running a full OMA system
identification once per (aleatory, epistemic) sample pair, the deterministic
model is evaluated on the sample lattice only up to the correlation-function
estimate (stage2corr_mapping). The weighted-covariance subspace estimator
(pyOMA VarSSIRef with weights and external corr_matrices) then consumes, per
epistemic sample and Imprecision hypercube, all N_ale correlation estimates
at once, with Incompleteness-conditioned importance weights obtained by
freezing the secondary Incompleteness variables (c_vb, lamda_vb) and
evaluating the wind-speed pdf on the aleatory samples
(PolyUQ.probabilities_imp). Identified poles carry first-order variance
estimates (variance_algo='fast'), are clustered across all epistemic cells
(cluster_modes_weighted), and the per-mode statistics Theta[q, n_e] are
processed on the statistic level: a surrogate over the epistemic samples,
interval-optimized within the combined Imprecision x Incompleteness
hypercubes via PolyUQ.from_propagated_samples(...).estimate_imp().

Reduced case-study definition (fixed during planning):
N_ale = 100 aleatory samples of T = 300 s each (~8.3 h of "measurement",
split along the aleatory direction), N_epi = 200 epistemic samples.
Epistemic variables: model_order, tau_max, decimation_factor, n_locations
(Imprecision); c_vb, lamda_vb (secondary Incompleteness, drive the weights);
DAQ_noise_rms (primary Incompleteness). All other acquisition parameters of
the original study (UQ_OMA.vars_definition) are fixed to representatives of
their highest-mass focal sets. m_lags is derived (ceil(tau_max * fs)), the
estimator is fixed to Blackman-Tukey.

Case-study degeneracy (documented, deliberate): the per-hypercube weight-
elimination step of the algorithm only acts on Imprecision variables whose
focal bounds are formed by secondary-Variability samples — this case study
has none, so the weights and hence Theta[q, n_e] are independent of the
Imprecision hypercube q and OMA effectively runs once per epistemic sample.
The generic per-q loop (with a dedupe fast path) is implemented anyway to
match the dissertation algorithm.

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
N_GEN = 2 ** 15      # generated timesteps per aleatory record (468 s > T)
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
    'duration': 300.0,                      # T_seg, fixed by design
}


def vars_definition_weighted():
    '''
    Variable definitions of the reduced case study; focal sets and masses
    from UQ_OMA.vars_definition except: tau_max upper bounds reduced to fit
    T = 300 s segments, decimation_factor reduced to its "risk-loving" focal
    (fs 3.9 ... 10 Hz), m_lags removed (derived as ceil(tau_max * fs)).

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

    vars_ale = [v_b]
    vars_epi = [c, lamda,
                n_locations, DAQ_noise_rms, decimation_factor, tau_max,
                model_order]

    arg_vars = {'n_locations': n_locations.name,
                'DAQ_noise_rms': DAQ_noise_rms.name,
                'decimation_factor': decimation_factor.name,
                'tau_max': tau_max.name,
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
                       v_b, jid, result_dir, response_provider,
                       fixed_params=None, skip_existing=True):
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
    if corr_file.exists() and skip_existing:
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
                 duration=fixed_params['duration'])
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


def run_lattice(poly_uq, arg_vars, result_dir, mech_npz, n_procs=None):
    '''
    Evaluate stage2corr_mapping on the full N_ale x N_epi lattice with local
    multiprocessing. Responses are generated once per aleatory sample
    (cached by the provider), correlation estimates once per lattice cell.
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


def run_weighted_identification(poly_uq, result_dir,
                                identification_fun=weighted_modal_identification):
    '''
    Main loop of Algorithm alg:proposed: for each epistemic sample and each
    Imprecision hypercube, compute the weights and run the weighted
    identification once — with a dedupe fast path across hypercubes that
    yield identical weights (always the case here, see module docstring).

    Returns a pole database: list of dicts with keys n_epi, i_imp and the
    identification outputs.
    '''
    N_epi = poly_uq.N_mcs_epi
    n_imp_hyc = len(poly_uq.imp_hyc_foc_inds)
    inp = poly_uq.inp_samp_prim

    pole_db = []
    for n_epi in range(N_epi):
        model_order = int(inp['model_order'].iloc[n_epi])
        corr_matrices, fs = load_corr_matrices(result_dir, n_epi,
                                               poly_uq.N_mcs_ale)
        cache = {}
        for i_imp in range(n_imp_hyc):
            weights = compute_weights(poly_uq, i_imp, n_epi)
            key = weights.tobytes()
            if key not in cache:
                cache[key] = identification_fun(
                    corr_matrices, weights, model_order, fs)
            pole_db.append(dict(cache[key], n_epi=n_epi, i_imp=i_imp,
                                weights_id=len(cache) - 1))
        logger.info(f'Epistemic sample {n_epi}: {len(cache)} distinct '
                    f'weighted identification(s) for {n_imp_hyc} hypercubes.')
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
    row per retained pole (columns f, d, std_f, std_d, n_epi, i_imp) and
    labels assigns each row a cluster (-1 = noise).
    '''
    from sklearn.cluster import OPTICS, cluster_optics_dbscan

    rows = []
    for entry in pole_db:
        f, d = entry['f'], entry['d']
        std_f, std_d = entry['std_f'], entry['std_d']
        nyq = entry.get('fs', None)
        keep = (f > f_min) & (f < f_max) & (~np.isnan(f)) & (d < d_max) & (d > 0)
        for j in np.where(keep)[0]:
            rows.append((f[j], d[j], std_f[j], std_d[j],
                         entry['n_epi'], entry['i_imp']))
    pole_table = pd.DataFrame(rows, columns=['f', 'd', 'std_f', 'std_d',
                                             'n_epi', 'i_imp'])
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
    arrays f, d, std_f, std_d.
    '''
    theta = {}
    for label in np.unique(labels):
        if label < 0:
            continue
        sel = pole_table.iloc[labels == label]
        arrs = {key: np.full((n_imp_hyc, N_epi), np.nan)
                for key in ('f', 'd', 'std_f', 'std_d')}
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


def statistic_level_polyuq(poly_uq, theta_mode, quantity='f'):
    '''
    Build the statistic-level PolyUQ instance for one clustered mode:
    surrogate inputs x_t are the Imprecision variables, the primary
    Incompleteness variable (DAQ_noise_rms) and — required to make the
    Incompleteness hypercube bounds meaningful — the secondary
    Incompleteness values (c_vb, lamda_vb) that drove the weights, all as
    primary variables of a fresh instance (their cartesian focal products
    are the combined hypercubes H^i_q x H^c_r, 72 in this case study). The
    output samples are Theta[0, n_e]: q-independent due to the documented
    weight-elimination degeneracy, so a single surrogate over the epistemic
    samples covers all Imprecision hypercubes.

    Call .estimate_imp() on the returned instance; its imp_foc[0] are the
    focal intervals F^Theta_{q,r} with masses imp_hyc_mass.
    '''
    vars_stat = [_primary_copy(var) for var in poly_uq.vars_epi]

    inp_names_prim = [var.name for var in poly_uq.vars_imp]
    inp_names_sec = [var.name for var in poly_uq.vars_inc]
    N_epi = poly_uq.N_mcs_epi
    x_t = pd.concat(
        [poly_uq.inp_samp_prim[inp_names_prim].iloc[:N_epi].reset_index(drop=True),
         poly_uq.inp_suppl_epi[inp_names_sec].iloc[:N_epi].reset_index(drop=True)],
        axis=1)

    out = np.asarray(theta_mode[quantity][0, :], dtype=np.float64)
    if np.all(np.isnan(out)):
        raise ValueError('The mode was not found in any epistemic sample.')

    return PolyUQ.from_propagated_samples(
        vars_stat, x_t, out[np.newaxis, :],
        out_name=f'theta_{quantity}')


# ── (g) full pipeline ────────────────────────────────────────────────────────

def run_weighted_uq_pipeline(result_dir, mech_npz,
                             N_ale=100, N_epi=200, seed=1509,
                             n_procs=None, skip_lattice=False,
                             min_found=0.1):
    '''
    Complete reduced case study: sampling, response generation, local
    lattice, weighted identifications, clustering, statistic-level focal
    intervals. Results are cached under result_dir; reruns skip completed
    cells. Clusters found in fewer than ``min_found`` of the epistemic
    samples (spurious poles) are reported but not interval-optimized.
    Returns (poly_uq, pole_db, theta, results) where results maps cluster
    labels to {'f'/'d': {'imp_foc', 'imp_hyc_mass', 'intp_errors'}}.
    '''
    import pickle

    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    vars_ale, vars_epi, arg_vars, _ = vars_definition_weighted()
    poly_uq = PolyUQ(vars_ale, vars_epi, dim_ex='cartesian')
    poly_uq.sample_qmc(N_mcs_ale=N_ale, N_mcs_epi=N_epi, seed=seed)

    if not skip_lattice:
        run_lattice(poly_uq, arg_vars, result_dir, mech_npz, n_procs=n_procs)

    pole_db = run_weighted_identification(poly_uq, result_dir)
    with open(result_dir / 'pole_db.pkl', 'wb') as f:
        pickle.dump(pole_db, f)

    n_imp_hyc = len(poly_uq.imp_hyc_foc_inds)
    labels, pole_table = cluster_modes_weighted(pole_db)
    theta = assemble_theta(labels, pole_table, N_epi, n_imp_hyc)
    np.savez(result_dir / 'weighted_theta.npz',
             labels=labels, **{f'theta_{label}_{key}': theta[label][key]
                               for label in theta
                               for key in ('f', 'd', 'std_f', 'std_d')})

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
            np.savez(result_dir / f'weighted_focals_{label}_{quantity}.npz',
                     imp_foc=imp_foc[0], imp_hyc_mass=pq_stat.imp_hyc_mass,
                     intp_errors=intp_errors, median=np.nanmedian(
                         theta_mode[quantity]))
    return poly_uq, pole_db, theta, results


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
    args = parser.parse_args()
    run_weighted_uq_pipeline(args.result_dir, args.mech_npz,
                             N_ale=args.n_ale, N_epi=args.n_epi,
                             seed=args.seed, n_procs=args.n_procs)
