"""
Orchestration tests for the weighted-OMA pipeline (examples/UQ_OMA_weighted).

The Algorithm alg:proposed logic — variable definitions, Incompleteness-
conditioned weights, per-hypercube loop with dedupe, clustering, statistic-
level assembly — is tested in milliseconds against synthetic pole data and
a counting stand-in for the weighted identification. Only the end-to-end
test touches the real generator chain (markers ``data`` + ``slow``).
"""
import os
from pathlib import Path

import numpy as np
import pytest
from polyuq import PolyUQ

from examples import UQ_OMA_weighted as uw

DATA_DIR = Path(os.environ.get('POLYUQ_DATA_DIR',
                               '/home/womo1998/Projects/uq_oma_a'))
MECH_NPZ = DATA_DIR / 'samples' / 'mechanical.npz'


@pytest.fixture(scope='module')
def poly_uq_small():
    vars_ale, vars_epi, _, _ = uw.vars_definition_weighted()
    pq = PolyUQ(vars_ale, vars_epi, dim_ex='cartesian')
    pq.sample_qmc(N_mcs_ale=64, N_mcs_epi=8, seed=1509)
    return pq


class TestVarsDefinition:

    def test_variable_sets(self):
        vars_ale, vars_epi, arg_vars, fixed = uw.vars_definition_weighted()
        assert [v.name for v in vars_ale] == ['v_b']
        assert len(vars_epi) == 8
        names = {v.name for v in vars_epi}
        assert names == {'c_vb', 'lamda_vb', 'n_locations', 'DAQ_noise_rms',
                         'decimation_factor', 'tau_max', 'model_order', 'T'}
        # secondary Incompleteness variables drive the weights
        secondary = {v.name for v in vars_epi if not v.primary}
        assert secondary == {'c_vb', 'lamda_vb'}
        # T's focal bounds are plain numbers now -- no secondary Variability
        secondary_ale = {v.name for v in vars_ale if not v.primary}
        assert secondary_ale == set()
        # model_order acts on the OMA stage only, not on the lattice
        assert 'model_order' not in arg_vars.values()
        assert arg_vars['duration'] == 'T'
        assert 'duration' not in fixed  # now sampled via T, not fixed
        assert 'm_lags' not in fixed  # derived from tau_max * fs

    def test_hypercube_counts(self, poly_uq_small):
        # Imprecision hypercubes: n_loc(3) x DAQ(1) x dec(1) x tau(2) x ord(3) x T(2)
        assert len(poly_uq_small.imp_hyc_foc_inds) == 36
        assert np.isclose(np.sum(poly_uq_small.imp_hyc_mass), 1.0)


class TestWeights:

    def test_normalized_and_positive(self, poly_uq_small):
        w = uw.compute_weights(poly_uq_small, 0, 0)
        assert w.shape == (64,)
        assert np.isclose(np.sum(w), 1.0)
        assert np.all(w >= 0)
        # Weibull over uniform proposal: informative, but not degenerate
        n_eff = 1.0 / np.sum(w ** 2)
        assert 5 < n_eff < 64

    def test_q_independent_in_this_case_study(self, poly_uq_small):
        # T's two focal sets branch the Imprecision hypercube grid (36 hyc),
        # but T is Imprecision, not Incompleteness: it never enters
        # compute_weights, so the weights stay identical across every
        # hypercube of one epistemic sample (q-independence).
        w_q = [uw.compute_weights(poly_uq_small, i_imp, 0)
               for i_imp in range(36)]
        for w in w_q[1:]:
            assert np.allclose(w, w_q[0])

    def test_depends_on_incompleteness_sample(self, poly_uq_small):
        suppl = poly_uq_small.inp_suppl_epi
        n_other = np.argmax(np.abs(suppl['lamda_vb'].values
                                   - suppl['lamda_vb'].iloc[0]))
        w0 = uw.compute_weights(poly_uq_small, 0, 0)
        w1 = uw.compute_weights(poly_uq_small, 0, int(n_other))
        assert not np.allclose(w0, w1)

    def test_delegates_to_polyuq_weights(self, poly_uq_small):
        # compute_weights is a thin wrapper around PolyUQ._weights(eliminate=
        # True). This plain two-focal T set has no secondary-Variability focal
        # bounds, so the elimination mask is all-True: a structural no-op that
        # only re-normalizes, leaving no weight eliminated.
        w = uw.compute_weights(poly_uq_small, 0, 0)
        assert np.array_equal(
            w, poly_uq_small._weights(i_imp=0, n_epi=0, eliminate=True))
        w_no_elim = poly_uq_small._weights(i_imp=0, n_epi=0, eliminate=False)
        assert np.allclose(w, w_no_elim)   # no-op up to renormalization
        assert np.all(w > 0)               # nothing eliminated


class TestIdentificationLoop:

    N_ALE, N_EPI = 6, 3

    @pytest.fixture
    def fake_lattice(self, tmp_path):
        """Corr caches with recognizable content: value = n_ale + 100*n_epi."""
        for n_epi in range(self.N_EPI):
            for n_ale in range(self.N_ALE):
                d = tmp_path / f'a{n_ale:04d}' / f'e{n_epi:04d}'
                d.mkdir(parents=True)
                np.savez(d / 'corr.npz',
                         corr_matrix=np.full((4, 2, 10),
                                             n_ale + 100 * n_epi,
                                             dtype=np.float32),
                         sampling_rate=7.0, m_lags=10,
                         channel_defs=np.zeros((4, 3), dtype=int))
        return tmp_path

    def test_load_corr_matrices(self, fake_lattice):
        corr, fs = uw.load_corr_matrices(fake_lattice, 2, self.N_ALE)
        assert corr.shape == (self.N_ALE, 4, 2, 10)
        assert corr.dtype == np.float64
        assert fs == 7.0
        assert np.all(corr[3] == 3 + 200)

    def test_dedupe_and_bookkeeping(self, fake_lattice):
        vars_ale, vars_epi, _, _ = uw.vars_definition_weighted()
        pq = PolyUQ(vars_ale, vars_epi, dim_ex='cartesian')
        pq.sample_qmc(N_mcs_ale=self.N_ALE, N_mcs_epi=self.N_EPI, seed=1509)

        calls = []

        def counting_stand_in(corr_matrices, weights, model_order, fs):
            calls.append((corr_matrices.copy(), weights.copy(),
                          model_order, fs))
            return {'f': np.array([1.0, 2.0]), 'd': np.array([1.0, 1.5]),
                    'std_f': np.array([0.01, 0.02]),
                    'std_d': np.array([0.1, 0.2]),
                    'phi': np.zeros((4, 2), dtype=complex),
                    'n_eff': 1.0 / np.sum(weights ** 2),
                    'order': model_order}

        pole_db = uw.run_weighted_identification(
            pq, fake_lattice, identification_fun=counting_stand_in)

        # dedupe: one distinct identification per epistemic sample (T branches
        # the 36-hypercube grid but never enters the weights, so all 36 share
        # one weight vector)
        assert len(calls) == self.N_EPI
        assert len(pole_db) == self.N_EPI * 36
        # each identification received that epistemic sample's corr stack
        for n_epi, (corr, weights, order, fs) in enumerate(calls):
            assert np.all(corr[0] == 100 * n_epi)
            assert np.isclose(np.sum(weights), 1.0)
            assert order == int(pq.inp_samp_prim['model_order'].iloc[n_epi])
        # bookkeeping: every (n_epi, i_imp) cell present exactly once
        cells = {(e['n_epi'], e['i_imp']) for e in pole_db}
        assert len(cells) == self.N_EPI * 36
        assert all(e['weights_id'] == 0 for e in pole_db)


class TestDataDrivenIdentificationLoop:
    """Deliverable 3: method='sd' orchestration (decimated-signal cache,
    num_block_rows derivation, dedupe/bookkeeping) -- mirrors
    TestIdentificationLoop for the covariance method ('sc')."""

    N_ALE, N_EPI = 6, 3

    @pytest.fixture
    def fake_signal_lattice(self, tmp_path):
        """Signal caches with recognizable content: value = n_ale + 100*n_epi."""
        for n_epi in range(self.N_EPI):
            for n_ale in range(self.N_ALE):
                d = tmp_path / f'a{n_ale:04d}' / f'e{n_epi:04d}'
                d.mkdir(parents=True)
                np.savez_compressed(
                    d / 'signal.npz',
                    signal=np.full((500, 4), n_ale + 100 * n_epi, dtype=np.float32),
                    sampling_rate=20.0, channel_defs=np.zeros((4, 3), dtype=int))
        return tmp_path

    def test_load_decimated_signals(self, fake_signal_lattice):
        signals, fs = uw.load_decimated_signals(fake_signal_lattice, 2, self.N_ALE)
        assert len(signals) == self.N_ALE
        assert fs == 20.0
        assert signals[0].dtype == np.float64
        assert all(s.shape == (500, 4) for s in signals)
        assert np.all(signals[3] == 3 + 200)

    def test_dedupe_and_bookkeeping(self, fake_signal_lattice):
        vars_ale, vars_epi, _, _ = uw.vars_definition_weighted()
        pq = PolyUQ(vars_ale, vars_epi, dim_ex='cartesian')
        pq.sample_qmc(N_mcs_ale=self.N_ALE, N_mcs_epi=self.N_EPI, seed=1509)

        calls = []

        def counting_stand_in(signals, weights, model_order, num_block_rows, fs):
            calls.append((len(signals), weights.copy(), model_order,
                          num_block_rows, fs))
            return {'f': np.array([1.0, 2.0]), 'd': np.array([1.0, 1.5]),
                    'std_f': np.array([0.01, 0.02]),
                    'std_d': np.array([0.1, 0.2]),
                    'phi': np.zeros((4, 2), dtype=complex),
                    'n_eff': 1.0 / np.sum(weights ** 2),
                    'order': model_order}

        pole_db = uw.run_weighted_identification(
            pq, fake_signal_lattice, method='sd',
            identification_fun=counting_stand_in)

        # one distinct identification per epistemic sample, see
        # TestIdentificationLoop.test_dedupe_and_bookkeeping
        assert len(calls) == self.N_EPI
        assert len(pole_db) == self.N_EPI * 36
        inp = pq.inp_samp_prim
        for n_epi, (n_signals, weights, order, num_block_rows, fs) in enumerate(calls):
            assert n_signals == self.N_ALE
            assert np.isclose(np.sum(weights), 1.0)
            assert order == int(inp['model_order'].iloc[n_epi])
            assert fs == 20.0
            m_lags = int(np.ceil(inp['tau_max'].iloc[n_epi] * fs))
            assert num_block_rows == min(m_lags // 2, 80)
        cells = {(e['n_epi'], e['i_imp']) for e in pole_db}
        assert len(cells) == self.N_EPI * 36

    def test_invalid_method_raises(self, fake_signal_lattice):
        vars_ale, vars_epi, _, _ = uw.vars_definition_weighted()
        pq = PolyUQ(vars_ale, vars_epi, dim_ex='cartesian')
        pq.sample_qmc(N_mcs_ale=self.N_ALE, N_mcs_epi=self.N_EPI, seed=1509)
        with pytest.raises(ValueError, match="'sc', 'sd' or 'cf'"):
            uw.run_weighted_identification(pq, fake_signal_lattice, method='bogus')

    def test_invalid_weighting_raises(self, fake_signal_lattice):
        vars_ale, vars_epi, _, _ = uw.vars_definition_weighted()
        pq = PolyUQ(vars_ale, vars_epi, dim_ex='cartesian')
        pq.sample_qmc(N_mcs_ale=self.N_ALE, N_mcs_epi=self.N_EPI, seed=1509)
        with pytest.raises(ValueError, match="'build' or 'posthoc'"):
            uw.run_weighted_identification(pq, fake_signal_lattice, weighting='bogus')


class TestPLSCFIdentificationLoop:
    """method='cf' orchestration: same corr.npz cache as 'sc', dispatched to
    weighted_plscf_identification by default -- mirrors
    TestIdentificationLoop."""

    N_ALE, N_EPI = 6, 3

    @pytest.fixture
    def fake_lattice(self, tmp_path):
        """Corr caches with recognizable content: value = n_ale + 100*n_epi."""
        for n_epi in range(self.N_EPI):
            for n_ale in range(self.N_ALE):
                d = tmp_path / f'a{n_ale:04d}' / f'e{n_epi:04d}'
                d.mkdir(parents=True)
                np.savez(d / 'corr.npz',
                         corr_matrix=np.full((4, 2, 10),
                                             n_ale + 100 * n_epi,
                                             dtype=np.float32),
                         sampling_rate=7.0, m_lags=10,
                         channel_defs=np.zeros((4, 3), dtype=int))
        return tmp_path

    def test_dedupe_and_bookkeeping(self, fake_lattice):
        vars_ale, vars_epi, _, _ = uw.vars_definition_weighted()
        pq = PolyUQ(vars_ale, vars_epi, dim_ex='cartesian')
        pq.sample_qmc(N_mcs_ale=self.N_ALE, N_mcs_epi=self.N_EPI, seed=1509)

        calls = []

        def counting_stand_in(corr_matrices, weights, model_order, fs):
            calls.append((corr_matrices.copy(), weights.copy(),
                          model_order, fs))
            return {'f': np.array([1.0, 2.0]), 'd': np.array([1.0, 1.5]),
                    'std_f': np.array([0.01, 0.02]),
                    'std_d': np.array([0.1, 0.2]),
                    'phi': np.zeros((4, 2), dtype=complex),
                    'n_eff': 1.0 / np.sum(weights ** 2),
                    'order': model_order}

        pole_db = uw.run_weighted_identification(
            pq, fake_lattice, method='cf', identification_fun=counting_stand_in)

        assert len(calls) == self.N_EPI
        assert len(pole_db) == self.N_EPI * 36
        for n_epi, (corr, weights, order, fs) in enumerate(calls):
            assert np.all(corr[0] == 100 * n_epi)
            assert np.isclose(np.sum(weights), 1.0)
            assert order == int(pq.inp_samp_prim['model_order'].iloc[n_epi])
        cells = {(e['n_epi'], e['i_imp']) for e in pole_db}
        assert len(cells) == self.N_EPI * 36
        assert all(e['weights_id'] == 0 for e in pole_db)


def _synthetic_two_mode_signals(n_ale=5, n_l=4, fs=50.0, duration=60.0, seed=0):
    """Shared across the identification-function smoke tests: n_ale
    independent realizations of a two-mode (2.0 Hz, 5.0 Hz) system plus
    noise, in the same synthesized-signal style stage2corr_mapping/
    ToyResponseProvider produce for the real lattice."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1.0 / fs)
    signals = []
    for _ in range(n_ale):
        sig = rng.normal(size=(t.size, n_l)) * 0.1
        for f0, zeta, amp in ((2.0, 0.01, 0.5), (5.0, 0.02, 0.5)):
            mode = np.sin(2 * np.pi * f0 * t) * np.exp(-zeta * t)
            sig += mode[:, None] * rng.normal(amp, 0.1 * amp, n_l)
        signals.append(sig)
    return signals


def _synthetic_two_mode_corr_matrices(m_lags, ref_channels=(0, 1), **kwargs):
    """The corr_matrices analogue of _synthetic_two_mode_signals: one
    Blackman-Tukey correlation estimate per synthesized signal, stacked
    (n_ale, n_l, n_r, m_lags) as load_corr_matrices would return."""
    from pyOMA.core.PreProcessingTools import PreProcessSignals

    signals = _synthetic_two_mode_signals(**kwargs)
    fs = kwargs.get('fs', 50.0)
    corr = []
    for sig in signals:
        ps = PreProcessSignals(sig, fs, ref_channels=list(ref_channels))
        ps.corr_blackman_tukey(m_lags)
        corr.append(ps.corr_matrices_bt[0])
    return np.stack(corr, axis=0), fs


class TestDataDrivenIdentification:
    """Smoke test of the full weighted_data_driven_identification wiring
    (external Hankel build -> state matrices -> fast variance -> post-hoc
    reweighting) on a synthetic two-mode signal. Physical-frequency-recovery
    accuracy and the Hankel-construction math are validated at the
    VarSSIRef level in pyOMA's own suite (TestExternalHankelProjection);
    this only checks the pipeline-side plumbing."""

    def test_signal_to_hankel_block_shape(self):
        rng = np.random.default_rng(1)
        n_l, p = 4, 6
        signal = rng.normal(size=(300, n_l))
        block = uw._signal_to_hankel_block(signal, p, (0, 1), n_l)
        n_r = 2
        assert block.shape == (p * n_r + (p + 1) * n_l, 300 - 2 * p)

    def test_returns_expected_shapes_and_finite_variance(self):
        signals = _synthetic_two_mode_signals()
        weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        result = uw.weighted_data_driven_identification(
            signals, weights, model_order=8, num_block_rows=15, fs=50.0)

        assert set(result.keys()) == {'f', 'd', 'std_f', 'std_d', 'phi',
                                      'n_eff', 'order'}
        assert result['order'] == 8
        assert np.isclose(result['n_eff'], 1.0 / np.sum(weights ** 2))
        n_modes = result['f'].shape[0]
        assert result['d'].shape == (n_modes,)
        assert result['std_f'].shape == (n_modes,)
        assert result['phi'].shape == (4, n_modes)
        assert np.all(np.isfinite(result['std_f']) & (result['std_f'] >= 0))
        assert np.all(np.isfinite(result['std_d']) & (result['std_d'] >= 0))
        # the two synthesized modes must be among the identified poles
        for f0 in (2.0, 5.0):
            assert np.min(np.abs(result['f'] - f0)) / f0 < 0.05


class TestDataDrivenIdentificationBuild:
    """weighted_data_driven_identification_build (method='sd',
    weighting='build'): unlike weighted_data_driven_identification, the
    point estimate must genuinely move with the weights -- exercises the
    default (non-experimental) weighted-mean projection build added to
    VarSSIRef alongside this task."""

    def test_returns_expected_shapes_and_finite_variance(self):
        signals = _synthetic_two_mode_signals()
        weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        result = uw.weighted_data_driven_identification_build(
            signals, weights, model_order=8, num_block_rows=15, fs=50.0)

        assert set(result.keys()) == {'f', 'd', 'std_f', 'std_d', 'phi',
                                      'n_eff', 'order'}
        assert result['order'] == 8
        assert np.isclose(result['n_eff'], 1.0 / np.sum(weights ** 2))
        n_modes = result['f'].shape[0]
        assert result['phi'].shape == (4, n_modes)
        assert np.all(np.isfinite(result['std_f']) & (result['std_f'] >= 0))
        assert np.any(result['std_f'] > 0)
        for f0 in (2.0, 5.0):
            assert np.min(np.abs(result['f'] - f0)) / f0 < 0.05

    def test_point_estimate_moves_with_weights(self):
        """The defining difference from weighted_data_driven_identification
        (posthoc): build-time weighting must change f/d themselves, not
        just their standard deviations."""
        signals = _synthetic_two_mode_signals(n_ale=6, seed=3)
        uniform = np.full(6, 1.0 / 6)
        skewed = np.array([0.6, 0.3, 0.05, 0.02, 0.02, 0.01])

        ref = uw.weighted_data_driven_identification_build(
            signals, uniform, model_order=8, num_block_rows=15, fs=50.0)
        wgt = uw.weighted_data_driven_identification_build(
            signals, skewed, model_order=8, num_block_rows=15, fs=50.0)
        assert not np.allclose(np.sort(ref['f']), np.sort(wgt['f']))


class TestPLSCFIdentification:
    """weighted_plscf_identification (method='cf', weighting='build'):
    mirrors TestDataDrivenIdentification for VarPLSCF on synthetic
    correlation estimates built the same way stage2corr_mapping/corr.npz
    does (Blackman-Tukey per aleatory sample)."""

    M_LAGS = 200

    def test_returns_expected_shapes_and_finite_variance(self):
        corr, fs = _synthetic_two_mode_corr_matrices(self.M_LAGS)
        weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        result = uw.weighted_plscf_identification(
            corr, weights, model_order=8, fs=fs)

        assert set(result.keys()) == {'f', 'd', 'std_f', 'std_d', 'phi',
                                      'n_eff', 'order'}
        assert result['order'] == 8
        assert np.isclose(result['n_eff'], 1.0 / np.sum(weights ** 2))
        n_modes = result['f'].shape[0]
        assert result['d'].shape == (n_modes,)
        assert result['std_f'].shape == (n_modes,)
        assert result['phi'].shape == (4, n_modes)
        assert np.all(np.isfinite(result['std_f']) & (result['std_f'] >= 0))
        assert np.all(np.isfinite(result['std_d']) & (result['std_d'] >= 0))
        for f0 in (2.0, 5.0):
            assert np.min(np.abs(result['f'] - f0)) / f0 < 0.05

    def test_order_capped_at_nperseg_minus_two(self):
        """max_model_order = model_order + 1 must stay <= nperseg - 1
        (PLSCF._setup_compute_params), so model_order itself caps at
        nperseg - 2."""
        corr, fs = _synthetic_two_mode_corr_matrices(self.M_LAGS, n_ale=4)
        weights = np.full(4, 0.25)
        result = uw.weighted_plscf_identification(
            corr, weights, model_order=self.M_LAGS + 50, fs=fs)
        assert result['order'] == self.M_LAGS - 2


class TestPosthocBuilders:
    """The shared _build_unweighted_*/_extract_pole_dict/apply_block_weights
    infrastructure behind weighting='posthoc': point estimates must stay
    fixed at the unweighted mean across different weight vectors, while the
    standard deviations respond to them -- the opposite pattern from the
    'build' identification functions."""

    def test_sc_frozen_point_estimate_varying_std(self):
        from pyOMA.core.VarSSIRef import VarSSIRef

        corr, fs = _synthetic_two_mode_corr_matrices(200, n_ale=6, seed=2)
        obj, order = uw._build_unweighted_sc(corr, model_order=8, fs=fs)
        uniform = np.full(6, 1.0 / 6)
        skewed = np.array([0.6, 0.3, 0.05, 0.02, 0.02, 0.01])

        obj.apply_block_weights(weights=uniform)
        _, n_eff_u = VarSSIRef._validate_weights(uniform, 6)
        ref = uw._extract_pole_dict(obj, order, n_eff_u, plscf=False)

        obj.apply_block_weights(weights=skewed)
        _, n_eff_w = VarSSIRef._validate_weights(skewed, 6)
        wgt = uw._extract_pole_dict(obj, order, n_eff_w, plscf=False)

        assert np.allclose(np.sort(ref['f']), np.sort(wgt['f']))
        assert not np.isclose(ref['n_eff'], wgt['n_eff'])
        assert np.any(ref['std_f'] != pytest.approx(wgt['std_f']))

    def test_sd_frozen_point_estimate_varying_std(self):
        from pyOMA.core.VarSSIRef import VarSSIRef

        signals = _synthetic_two_mode_signals(n_ale=6, seed=5)
        obj, order = uw._build_unweighted_sd(
            signals, model_order=8, num_block_rows=15, fs=50.0)
        uniform = np.full(6, 1.0 / 6)
        skewed = np.array([0.6, 0.3, 0.05, 0.02, 0.02, 0.01])

        obj.apply_block_weights(weights=uniform)
        _, n_eff_u = VarSSIRef._validate_weights(uniform, 6)
        ref = uw._extract_pole_dict(obj, order, n_eff_u, plscf=False)
        obj.apply_block_weights(weights=skewed)
        _, n_eff_w = VarSSIRef._validate_weights(skewed, 6)
        wgt = uw._extract_pole_dict(obj, order, n_eff_w, plscf=False)

        assert np.allclose(np.sort(ref['f']), np.sort(wgt['f']))
        assert np.any(ref['std_f'] != pytest.approx(wgt['std_f']))

    def test_cf_frozen_point_estimate_varying_std(self):
        from pyOMA.core.VarPLSCF import VarPLSCF

        corr, fs = _synthetic_two_mode_corr_matrices(200, n_ale=6, seed=7)
        obj, order = uw._build_unweighted_cf(corr, model_order=8, fs=fs)
        uniform = np.full(6, 1.0 / 6)
        skewed = np.array([0.6, 0.3, 0.05, 0.02, 0.02, 0.01])

        obj.apply_block_weights(weights=uniform)
        _, n_eff_u = VarPLSCF._validate_weights(uniform, 6)
        ref = uw._extract_pole_dict(obj, order, n_eff_u, plscf=True)
        obj.apply_block_weights(weights=skewed)
        _, n_eff_w = VarPLSCF._validate_weights(skewed, 6)
        wgt = uw._extract_pole_dict(obj, order, n_eff_w, plscf=True)

        assert np.allclose(np.sort(ref['f']), np.sort(wgt['f']))
        assert np.any(ref['std_f'] != pytest.approx(wgt['std_f']))


class TestPosthocIdentificationLoop:
    """weighting='posthoc' orchestration on the real dispatch path
    (run_weighted_identification): builds ONE unweighted identification per
    epistemic sample -- not per distinct weight vector, since there is
    nothing to dedupe against once the point estimate no longer depends on
    the weights -- and reweights per hypercube via apply_block_weights."""

    N_ALE, N_EPI = 6, 3

    @pytest.fixture
    def fake_lattice(self, tmp_path):
        corr, fs = _synthetic_two_mode_corr_matrices(200, n_ale=self.N_ALE)
        for n_epi in range(self.N_EPI):
            for n_ale in range(self.N_ALE):
                d = tmp_path / f'a{n_ale:04d}' / f'e{n_epi:04d}'
                d.mkdir(parents=True)
                np.savez(d / 'corr.npz',
                         corr_matrix=corr[n_ale].astype(np.float32),
                         sampling_rate=fs, m_lags=corr.shape[-1],
                         channel_defs=np.zeros((4, 3), dtype=int))
        return tmp_path

    def test_builds_once_per_epistemic_sample(self, fake_lattice, monkeypatch):
        vars_ale, vars_epi, _, _ = uw.vars_definition_weighted()
        pq = PolyUQ(vars_ale, vars_epi, dim_ex='cartesian')
        pq.sample_qmc(N_mcs_ale=self.N_ALE, N_mcs_epi=self.N_EPI, seed=1509)

        build_calls = []
        real_build = uw._build_unweighted_sc

        def counting_build(*args, **kwargs):
            build_calls.append(1)
            return real_build(*args, **kwargs)

        monkeypatch.setitem(uw._POSTHOC_BUILD_FUNS, 'sc', counting_build)

        pole_db = uw.run_weighted_identification(
            pq, fake_lattice, method='sc', weighting='posthoc')

        assert len(build_calls) == self.N_EPI
        assert len(pole_db) == self.N_EPI * 36
        cells = {(e['n_epi'], e['i_imp']) for e in pole_db}
        assert len(cells) == self.N_EPI * 36

        # frozen point estimate: f identical across every hypercube of one
        # epistemic sample (weights only ever move the variance here)
        for n_epi in range(self.N_EPI):
            entries = [e for e in pole_db if e['n_epi'] == n_epi]
            f0 = entries[0]['f']
            for e in entries[1:]:
                assert np.allclose(e['f'], f0)

        for e in pole_db:
            assert np.all(np.isfinite(e['std_f']))
            assert np.all(e['std_f'] >= 0)


class TestClusteringAndTheta:

    N_EPI, N_HYC = 30, 3

    @pytest.fixture
    def synthetic_pole_db(self):
        """Two well-separated modes with jitter, one spurious pole per cell."""
        rng = np.random.default_rng(4)
        pole_db = []
        for n_epi in range(self.N_EPI):
            for i_imp in range(self.N_HYC):
                f = np.array([1.0 + rng.normal(0, 0.002),
                              2.0 + rng.normal(0, 0.002),
                              rng.uniform(0.2, 4.5)])
                d = np.array([1.0, 1.5, 15.0]) + rng.normal(0, 0.05, 3)
                pole_db.append({
                    'f': f, 'd': d,
                    'std_f': np.full(3, 0.01), 'std_d': np.full(3, 0.1),
                    'phi': np.zeros((4, 3), dtype=complex),
                    'n_eff': 50.0, 'order': 20,
                    'n_epi': n_epi, 'i_imp': i_imp})
        return pole_db

    def test_two_modes_recovered(self, synthetic_pole_db):
        labels, pole_table = uw.cluster_modes_weighted(
            synthetic_pole_db, min_samples=10, min_cluster_size=20)
        assert len(pole_table) == self.N_EPI * self.N_HYC * 3
        found_medians = []
        for label in set(labels) - {-1}:
            found_medians.append(
                np.median(pole_table['f'].values[labels == label]))
        assert any(abs(m - 1.0) < 0.01 for m in found_medians)
        assert any(abs(m - 2.0) < 0.01 for m in found_medians)

    def test_theta_assembly(self, synthetic_pole_db):
        labels, pole_table = uw.cluster_modes_weighted(
            synthetic_pole_db, min_samples=10, min_cluster_size=20)
        theta = uw.assemble_theta(labels, pole_table, self.N_EPI, self.N_HYC)
        for label, tm in theta.items():
            assert tm['f'].shape == (self.N_HYC, self.N_EPI)
            med = np.nanmedian(tm['f'])
            if abs(med - 1.0) < 0.01 or abs(med - 2.0) < 0.01:
                # the true modes appear in (nearly) every cell; single
                # jittered poles may drop out of the OPTICS cluster
                found = ~np.isnan(tm['f'])
                assert np.mean(found) >= 0.95
                assert np.all(tm['count'][found] >= 1)
                assert np.all(np.isfinite(tm['std_f'][found])
                              & (tm['std_f'][found] >= 0))

    def test_theta_nan_where_mode_missing(self):
        pole_db = [{'f': np.array([1.0]), 'd': np.array([1.0]),
                    'std_f': np.array([0.01]), 'std_d': np.array([0.1]),
                    'phi': np.zeros((2, 1), dtype=complex),
                    'n_eff': 10.0, 'order': 4, 'n_epi': n_epi, 'i_imp': 0}
                   for n_epi in range(20) if n_epi != 7]  # mode missing at 7
        labels, pole_table = uw.cluster_modes_weighted(
            pole_db, min_samples=5, min_cluster_size=5)
        theta = uw.assemble_theta(labels, pole_table, 20, 1)
        assert len(theta) == 1
        tm = next(iter(theta.values()))
        assert np.isnan(tm['f'][0, 7])
        assert np.sum(~np.isnan(tm['f'][0])) == 19

    def test_sigma_pop_undoes_sem_shrink(self, synthetic_pole_db):
        # plan Deliverable 2: sigma_pop = std * sqrt(n_eff)
        labels, pole_table = uw.cluster_modes_weighted(
            synthetic_pole_db, min_samples=10, min_cluster_size=20)
        theta = uw.assemble_theta(labels, pole_table, self.N_EPI, self.N_HYC)
        for tm in theta.values():
            found = ~np.isnan(tm['f'])
            assert np.allclose(tm['sigma_pop_f'][found],
                               tm['std_f'][found] * np.sqrt(tm['n_eff'][found]))
            assert np.allclose(tm['sigma_pop_d'][found],
                               tm['std_d'][found] * np.sqrt(tm['n_eff'][found]))


class TestStatisticLevel:

    def test_instance_construction(self, poly_uq_small):
        rng = np.random.default_rng(0)
        N_epi = poly_uq_small.N_mcs_epi
        theta_mode = {'f': np.tile(
            1.0 + 0.001 * poly_uq_small.inp_samp_prim['model_order']
            .values[:N_epi] + rng.normal(0, 1e-4, N_epi), (36, 1))}
        theta_mode['f'][:, 2] = np.nan  # a cell where the mode was missed
        pq_stat = uw.statistic_level_polyuq(poly_uq_small, theta_mode, 'f')

        # combined Imprecision x Incompleteness hypercubes: 36 x (2*2*1)
        assert len(pq_stat.imp_hyc_foc_inds) == 144
        assert np.isclose(np.sum(pq_stat.imp_hyc_mass), 1.0)
        assert pq_stat.N_mcs_ale == 1
        assert pq_stat.N_mcs_epi == N_epi
        assert set(pq_stat.inp_samp_prim.columns) == {
            'n_locations', 'DAQ_noise_rms', 'decimation_factor', 'tau_max',
            'model_order', 'T', 'c_vb', 'lamda_vb'}
        # all statistic-level variables are primary (their products are the
        # combined hypercubes); NaN cells are carried, not dropped
        assert len(pq_stat.vars_imp) == 8
        assert np.isnan(pq_stat.out_samp[0, 2])

    def test_all_nan_raises(self, poly_uq_small):
        theta_mode = {'f': np.full((36, poly_uq_small.N_mcs_epi), np.nan)}
        with pytest.raises(ValueError, match='not found'):
            uw.statistic_level_polyuq(poly_uq_small, theta_mode, 'f')


class TestParametricCdf:
    """Deliverable 2: reconstruct the aleatory CDF from VarSSIRef's
    variance. Structural properties (collapse at sigma_pop=0, monotone
    quantiles) are checked on the pre-optimization out_samp rows — cheap and
    exact; estimate_imp() itself (genetic optimization over 144 hypercubes)
    is exercised only via a stand-in, matching this file's existing
    philosophy of keeping orchestration tests fast."""

    N_STAT = 5

    def _theta_mode(self, poly_uq_small, sigma):
        rng = np.random.default_rng(1)
        N_epi = poly_uq_small.N_mcs_epi
        f = 1.0 + 0.001 * np.arange(N_epi) + rng.normal(0, 1e-4, N_epi)
        return {'f': np.tile(f, (36, 1)),
                'sigma_pop_f': np.tile(np.full(N_epi, sigma), (36, 1))}

    def test_sigma_pop_zero_collapses_to_point_estimate(self, poly_uq_small):
        theta_mode = self._theta_mode(poly_uq_small, sigma=0.0)
        target_probabilities = np.linspace(0, 1, self.N_STAT)
        instances = uw.parametric_cdf_statistic_level_polyuq(
            poly_uq_small, theta_mode, 'f', target_probabilities)
        assert len(instances) == self.N_STAT
        for inst in instances:
            assert np.allclose(inst.out_samp[0, :], theta_mode['f'][0, :])

    def test_quantiles_monotone_in_target_probabilities(self, poly_uq_small):
        theta_mode = self._theta_mode(poly_uq_small, sigma=0.02)
        target_probabilities = np.linspace(0, 1, self.N_STAT)
        instances = uw.parametric_cdf_statistic_level_polyuq(
            poly_uq_small, theta_mode, 'f', target_probabilities)
        rows = np.stack([inst.out_samp[0, :] for inst in instances], axis=0)
        assert np.all(np.diff(rows, axis=0) > 0)

    def test_all_nan_raises(self, poly_uq_small):
        theta_mode = {'f': np.full((36, poly_uq_small.N_mcs_epi), np.nan),
                      'sigma_pop_f': np.full((36, poly_uq_small.N_mcs_epi), np.nan)}
        with pytest.raises(ValueError, match='not found'):
            uw.parametric_cdf_statistic_level_polyuq(
                poly_uq_small, theta_mode, 'f', np.linspace(0, 1, self.N_STAT))

    def test_cdf_reconstruction_layout(self, poly_uq_small, tmp_path, monkeypatch):
        # stand-in for estimate_imp: the expensive part (genetic optimization)
        # is irrelevant to the (n_stat, n_hyc, 2) layout being verified here
        def fake_estimate_imp(self, **kwargs):
            n_hyc = len(self.imp_hyc_foc_inds)
            imp_foc = np.zeros((1, n_hyc, 2))
            return imp_foc, None, np.zeros(1), None, None
        monkeypatch.setattr(PolyUQ, 'estimate_imp', fake_estimate_imp)

        theta_mode_f = self._theta_mode(poly_uq_small, sigma=0.02)
        theta_mode_d = self._theta_mode(poly_uq_small, sigma=0.2)
        theta = {0: {**theta_mode_f,
                    'd': theta_mode_d['f'], 'sigma_pop_d': theta_mode_d['sigma_pop_f']}}

        results = uw.run_weighted_cdf_reconstruction(
            poly_uq_small, theta, tmp_path, n_stat=self.N_STAT)

        assert set(results.keys()) == {0}
        for quantity in ('f', 'd'):
            r = results[0][quantity]
            n_hyc = 144  # 36 Imprecision x (2*2*1) combined Incompleteness
            assert r['imp_foc'].shape == (self.N_STAT, n_hyc, 2)
            assert r['imp_hyc_mass'].shape == (n_hyc,)

            with np.load(tmp_path / f'weighted_cdf_0_{quantity}.npz') as arr:
                assert arr['imp_foc'].shape == (self.N_STAT, n_hyc, 2)
                assert arr['imp_hyc_mass'].shape == (n_hyc,)
                assert arr['target_probabilities'].shape == (self.N_STAT,)


@pytest.mark.data
@pytest.mark.slow
@pytest.mark.skipif(not MECH_NPZ.exists(),
                    reason=f'mechanical.npz not found at {MECH_NPZ}')
class TestEndToEndSmall:
    """Generator -> acquisition -> correlation -> weighted VarSSIRef ->
    clustering on a tiny lattice; validates against the known mast modes."""

    def test_identification_recovers_modes(self, tmp_path):
        N_ale, N_epi = 5, 2
        vars_ale, vars_epi, arg_vars, _ = uw.vars_definition_weighted()
        pq = PolyUQ(vars_ale, vars_epi, dim_ex='cartesian')
        pq.sample_qmc(N_mcs_ale=N_ale, N_mcs_epi=N_epi, seed=1509)

        provider = uw.ToyResponseProvider(tmp_path, MECH_NPZ)
        inp = pq.inp_samp_prim
        for n_epi in range(N_epi):
            args_epi = {arg: inp[name].iloc[n_epi]
                        for arg, name in arg_vars.items() if arg != 'v_b'}
            for n_ale in range(N_ale):
                uw.stage2corr_mapping(
                    jid=f'a{n_ale:04d}_e{n_epi:04d}', result_dir=tmp_path,
                    response_provider=provider,
                    v_b=inp['v_b'].iloc[n_ale], **args_epi)

        pole_db = uw.run_weighted_identification(pq, tmp_path)
        assert len(pole_db) == N_epi * 36

        for entry in pole_db:
            assert np.all(np.isfinite(entry['std_f']))
            assert np.all(entry['std_f'] >= 0)

        # the lightly damped, well-separated modes must be among the poles
        # of every epistemic sample (>=10 min records now T is in minutes,
        # n_eff ~ 3 of 5 blocks)
        for n_epi in range(N_epi):
            entry = next(e for e in pole_db
                         if e['n_epi'] == n_epi and e['i_imp'] == 0)
            for f_k in (0.5801, 0.6038, 1.1995, 1.2495):
                rel_err = np.min(np.abs(entry['f'] - f_k)) / f_k
                assert rel_err < 0.01, \
                    f'mode {f_k} Hz missing at n_epi={n_epi}'

    def test_identification_recovers_modes_data_driven(self, tmp_path):
        """method='sd' (plan Deliverable 3): data-driven / projection SSI
        via the same lattice, decimated signals cached alongside corr.npz."""
        N_ale, N_epi = 5, 2
        vars_ale, vars_epi, arg_vars, _ = uw.vars_definition_weighted()
        pq = PolyUQ(vars_ale, vars_epi, dim_ex='cartesian')
        pq.sample_qmc(N_mcs_ale=N_ale, N_mcs_epi=N_epi, seed=1509)

        provider = uw.ToyResponseProvider(tmp_path, MECH_NPZ)
        inp = pq.inp_samp_prim
        for n_epi in range(N_epi):
            args_epi = {arg: inp[name].iloc[n_epi]
                        for arg, name in arg_vars.items() if arg != 'v_b'}
            for n_ale in range(N_ale):
                uw.stage2corr_mapping(
                    jid=f'a{n_ale:04d}_e{n_epi:04d}', result_dir=tmp_path,
                    response_provider=provider, cache_signal=True,
                    v_b=inp['v_b'].iloc[n_ale], **args_epi)
            assert (tmp_path / 'a0000' / f'e{n_epi:04d}' / 'signal.npz').exists()

        pole_db = uw.run_weighted_identification(pq, tmp_path, method='sd')
        assert len(pole_db) == N_epi * 36

        for entry in pole_db:
            assert np.all(np.isfinite(entry['std_f']))
            assert np.all(entry['std_f'] >= 0)

        for n_epi in range(N_epi):
            entry = next(e for e in pole_db
                         if e['n_epi'] == n_epi and e['i_imp'] == 0)
            for f_k in (0.5801, 0.6038, 1.1995, 1.2495):
                rel_err = np.min(np.abs(entry['f'] - f_k)) / f_k
                assert rel_err < 0.02, \
                    f'mode {f_k} Hz missing at n_epi={n_epi}'

    def test_identification_recovers_modes_plscf(self, tmp_path):
        """method='cf' (VarPLSCF), weighting='build' (default) -- shares the
        'sc' lattice (corr.npz only, no decimated-signal cache needed)."""
        N_ale, N_epi = 5, 2
        vars_ale, vars_epi, arg_vars, _ = uw.vars_definition_weighted()
        pq = PolyUQ(vars_ale, vars_epi, dim_ex='cartesian')
        pq.sample_qmc(N_mcs_ale=N_ale, N_mcs_epi=N_epi, seed=1509)

        provider = uw.ToyResponseProvider(tmp_path, MECH_NPZ)
        inp = pq.inp_samp_prim
        for n_epi in range(N_epi):
            args_epi = {arg: inp[name].iloc[n_epi]
                        for arg, name in arg_vars.items() if arg != 'v_b'}
            for n_ale in range(N_ale):
                uw.stage2corr_mapping(
                    jid=f'a{n_ale:04d}_e{n_epi:04d}', result_dir=tmp_path,
                    response_provider=provider,
                    v_b=inp['v_b'].iloc[n_ale], **args_epi)

        pole_db = uw.run_weighted_identification(pq, tmp_path, method='cf')
        assert len(pole_db) == N_epi * 36

        for entry in pole_db:
            assert np.all(np.isfinite(entry['std_f']))
            assert np.all(entry['std_f'] >= 0)

        for n_epi in range(N_epi):
            entry = next(e for e in pole_db
                         if e['n_epi'] == n_epi and e['i_imp'] == 0)
            for f_k in (0.5801, 0.6038, 1.1995, 1.2495):
                rel_err = np.min(np.abs(entry['f'] - f_k)) / f_k
                assert rel_err < 0.02, \
                    f'mode {f_k} Hz missing at n_epi={n_epi}'

    def test_identification_recovers_modes_posthoc(self, tmp_path):
        """weighting='posthoc' (method='sc'): point estimate is frozen at
        the unweighted mean, so the same modes must still show up."""
        N_ale, N_epi = 5, 2
        vars_ale, vars_epi, arg_vars, _ = uw.vars_definition_weighted()
        pq = PolyUQ(vars_ale, vars_epi, dim_ex='cartesian')
        pq.sample_qmc(N_mcs_ale=N_ale, N_mcs_epi=N_epi, seed=1509)

        provider = uw.ToyResponseProvider(tmp_path, MECH_NPZ)
        inp = pq.inp_samp_prim
        for n_epi in range(N_epi):
            args_epi = {arg: inp[name].iloc[n_epi]
                        for arg, name in arg_vars.items() if arg != 'v_b'}
            for n_ale in range(N_ale):
                uw.stage2corr_mapping(
                    jid=f'a{n_ale:04d}_e{n_epi:04d}', result_dir=tmp_path,
                    response_provider=provider,
                    v_b=inp['v_b'].iloc[n_ale], **args_epi)

        pole_db = uw.run_weighted_identification(
            pq, tmp_path, method='sc', weighting='posthoc')
        assert len(pole_db) == N_epi * 36

        for entry in pole_db:
            assert np.all(np.isfinite(entry['std_f']))
            assert np.all(entry['std_f'] >= 0)

        for n_epi in range(N_epi):
            entries = [e for e in pole_db if e['n_epi'] == n_epi]
            f0 = entries[0]['f']
            for e in entries[1:]:
                assert np.allclose(e['f'], f0)
            for f_k in (0.5801, 0.6038, 1.1995, 1.2495):
                rel_err = np.min(np.abs(f0 - f_k)) / f_k
                assert rel_err < 0.01, \
                    f'mode {f_k} Hz missing at n_epi={n_epi}'
