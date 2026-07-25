"""
Orchestration tests for the experimental OMA pipeline
(pyoma_uq/studies/UQ_OMA_experimental).

The pipeline logic — variable definitions, the acceptance-rejection screen,
pole -> baseline-mode assignment, and the statistic-level bookkeeping — is
tested in milliseconds against synthetic poles and a counting stand-in for the
identification. Tests that touch the measurement files carry the ``data``
marker; the one end-to-end identification also carries ``slow``.
"""
import types

import numpy as np
import pytest
from polyuq import MassFunction, PolyUQ

from pyoma_uq.studies import UQ_OMA_experimental as ex

pytestmark = pytest.mark.filterwarnings('ignore::RuntimeWarning')

DATA_AVAILABLE = (ex.SCHWABACH_DIR / 'modal_source_files').is_dir()
needs_data = pytest.mark.skipif(not DATA_AVAILABLE,
                                reason='Schwabach measurement data not available')


# ── signal-processing helpers ────────────────────────────────────────────────

class TestDecimation:

    @pytest.mark.parametrize('factor', [1, 2, 4, 5, 6, 8, 12, 16, 20, 25, 32, 40])
    def test_steps_multiply_back(self, factor):
        steps = ex.decimation_steps(factor)
        assert int(np.prod(steps or [1])) == factor
        assert all(s <= 8 for s in steps)

    def test_2019_choices_are_recovered(self):
        # low band: lowpass 1.5 Hz at f_s = 5.12 Hz -> f_s / f_lp = 3.41
        assert ex.resolve_decimation(1.5, 5.12 / 1.5) == 25
        # high band: lowpass 8 Hz at f_s = 21.33 Hz -> 2.67
        assert ex.resolve_decimation(8.0, 21.3333 / 8.0) == 6

    def test_decimation_never_undersamples(self):
        for lowpass in (1.2, 2.0, 4.0, 6.0, 9.0):
            for nyq_rat in (2.5, 4.0, 7.0, 10.0):
                factor = ex.resolve_decimation(lowpass, nyq_rat)
                assert ex.FS_RAW / factor >= nyq_rat * lowpass

    def test_num_block_rows_satisfies_hankel_bound(self):
        for m_lags in range(3, 500):
            p = ex.num_block_rows(m_lags)
            assert p + p + 1 <= m_lags


# ── variable definitions ─────────────────────────────────────────────────────

@needs_data
class TestVarsDefinition:

    def test_variable_sets(self):
        vars_ale, vars_epi, levels, offsets = ex.vars_definition_experimental()
        assert [v.name for v in vars_ale] == ['a_ref']
        assert {v.name for v in vars_epi if not v.primary} == {'s_a', 'scale_a'}
        assert [v.name for v in vars_epi if v.primary] == [
            'highpass', 'lowpass', 'nyq_rat', 'tau_max', 'm_lags', 'model_order']
        # the pooled aleatory ensemble is the blocks of all three setups
        assert len(levels) == 3 * ex.N_SEGMENTS == offsets[-1]

    def test_observed_levels_carry_a_proposal(self):
        vars_ale, _, levels, _ = ex.vars_definition_experimental()
        a_ref = vars_ale[0]
        assert a_ref.proposal is not None
        # the fit must give every observed level a positive density, else the
        # importance weights are undefined
        assert np.all(a_ref.proposal.pdf(levels) > 0)

    def test_imprecision_focals_tile_or_nest(self):
        """Every focal set must reach an endpoint of the variable's support.

        sample_qmc draws uniformly over the support hull, and a sample only
        joins a hypercube if it lies inside that hypercube's focal of *every*
        variable, so a gap between focals is dead sampling volume that leaves
        hypercubes empty.
        """
        _, vars_epi, _, _ = ex.vars_definition_experimental()
        for var in vars_epi:
            if not isinstance(var, MassFunction) or not var.primary:
                continue
            focals = np.asarray(var.numeric_focals, dtype=float)
            # the union of the focals must cover the support hull without a gap
            edges = np.unique(np.concatenate([focals[:, 0], focals[:, 1]]))
            midpoints = 0.5 * (edges[:-1] + edges[1:])
            for midpoint in midpoints:
                assert np.any((focals[:, 0] <= midpoint)
                              & (midpoint <= focals[:, 1])), (
                    f'{var.name} has a gap around {midpoint} between its '
                    'focal sets, which would leave hypercubes unpopulated')

    def test_hypercube_count(self):
        vars_ale, vars_epi, levels, _ = ex.vars_definition_experimental()
        pq = PolyUQ(vars_ale, vars_epi, dim_ex='cartesian', path='fast_build')
        # highpass(2) x lowpass(2) x nyq_rat(2) x tau_max(1) x m_lags(2)
        #   x model_order(2)
        assert len(pq.imp_hyc_foc_inds) == 32
        assert np.isclose(np.sum(pq.imp_hyc_mass), 1.0)


# ── acceptance-rejection ─────────────────────────────────────────────────────

class TestFeasibility:

    BASE = dict(highpass=0.1, lowpass=1.5, nyq_rat=3.41, tau_max=120.0,
                m_lags=422, model_order=100)

    def test_2019_low_band_is_feasible(self):
        ok, reason, resolved = ex.feasible(**self.BASE)
        assert ok, reason
        assert resolved['decimation_factor'] == 25
        assert resolved['num_block_rows'] == 210

    def test_degenerate_band_rejected(self):
        ok, reason, _ = ex.feasible(**dict(self.BASE, highpass=1.5, lowpass=1.5))
        assert not ok and reason == 'degenerate band'

    def test_model_order_above_m_lags_rejected(self):
        ok, reason, _ = ex.feasible(**dict(self.BASE, m_lags=60, model_order=100))
        assert not ok and 'model_order' in reason

    def test_m_lags_beyond_correlation_length_rejected(self):
        ok, reason, _ = ex.feasible(**dict(self.BASE, tau_max=20.0, m_lags=400))
        assert not ok and 'correlation length' in reason

    def test_m_lags_beyond_block_length_rejected(self):
        """The block-length guard, which the sampled ranges never reach.

        Block length and correlation length are both inversely proportional to
        the decimation factor, so the correlation length binds first for any
        ``tau_max`` below ``DURATION / n_segments = 300 s`` -- and ``tau_max``
        is sampled at most to 175 s. Exercised here with a longer ``tau_max``
        so the guard stays covered if those ranges change.
        """
        # 30 min at f_s = 5.12 Hz in 6 blocks is 1536 samples per block, while
        # ceil(400 s * 5.12 Hz) = 2048 lags were estimated
        ok, reason, resolved = ex.feasible(**dict(self.BASE, tau_max=400.0,
                                                  m_lags=1600, model_order=20))
        assert resolved['block_length'] == 1536
        assert not ok and reason == 'm_lags beyond the block length'

    def test_block_length_guard_is_unreachable_when_sampled(self):
        # documents why the guard above needs an out-of-range tau_max
        for tau_max in (20.0, 100.0, 175.0):
            for lowpass, nyq_rat in ((1.3, 2.5), (1.5, 3.41), (8.0, 2.67),
                                     (9.0, 10.0)):
                _, _, resolved = ex.feasible(**dict(
                    self.BASE, lowpass=lowpass, nyq_rat=nyq_rat,
                    tau_max=tau_max, m_lags=60, model_order=20))
                fs = resolved['sampling_rate']
                assert np.ceil(tau_max * fs) <= resolved['block_length']

    def test_anti_alias_cannot_be_violated(self):
        """The derived decimation makes the anti-alias margin unreachable.

        Sampling the decimation factor independently was what made 90 % of the
        epistemic samples infeasible in the first place.
        """
        for lowpass in (1.3, 3.0, 5.5, 8.9):
            for nyq_rat in (2.5, 5.0, 9.9):
                _, _, resolved = ex.feasible(**dict(self.BASE, lowpass=lowpass,
                                                    nyq_rat=nyq_rat,
                                                    m_lags=100, model_order=20))
                assert resolved['sampling_rate'] >= nyq_rat * lowpass


@needs_data
class TestFeasibilityReport:

    @pytest.fixture(scope='class')
    def poly_uq(self):
        vars_ale, vars_epi, levels, _ = ex.vars_definition_experimental()
        pq = PolyUQ(vars_ale, vars_epi, dim_ex='cartesian', path='fast_build')
        pq.sample_qmc(N_mcs_ale=len(levels), N_mcs_epi=500, seed=1509,
                      given_samples={'a_ref': levels})
        return pq

    def test_no_hypercube_is_empty(self, poly_uq):
        """Every hypercube must retain feasible samples, or its belief mass
        cannot be evaluated at all."""
        per_sample, per_hyc = ex.feasibility_report(poly_uq)
        assert (per_hyc.n_feasible > 0).all()
        assert np.isclose(per_hyc.loc[per_hyc.n_feasible > 0, 'mass'].sum(), 1.0)

    def test_acceptance_rate(self, poly_uq):
        per_sample, _ = ex.feasibility_report(poly_uq)
        assert per_sample.ok.mean() > 0.7

    def test_rejection_reasons_are_the_expected_ones(self, poly_uq):
        per_sample, _ = ex.feasibility_report(poly_uq)
        reasons = set(per_sample.loc[~per_sample.ok, 'reason'])
        assert reasons <= {'degenerate band', 'model_order too high for m_lags',
                           'm_lags beyond the estimated correlation length',
                           'm_lags beyond the block length',
                           'model_order beyond the subspace order cap',
                           'no block rows left'}


# ── baseline modes and pole assignment ───────────────────────────────────────

@needs_data
class TestBaseline:

    def test_both_bands_concatenate(self):
        baseline = ex.load_baseline_modes()
        assert baseline['f'].shape == (27,)
        assert baseline['phi'].shape == (26, 27)
        assert np.all(np.diff(baseline['f']) >= 0)
        assert set(baseline['band']) == {'low', 'high'}
        assert np.isfinite(baseline['phi']).all()

    def test_single_band_subset(self):
        low = ex.load_baseline_modes(bands=('low',))
        assert low['f'].shape == (21,)
        assert low['f'].max() < 1.4


def _fake_modal_data(baseline, order=60, noise=1e-4, seed=0):
    """A stand-in identification that reproduces the baseline modes exactly
    (plus a little noise), so the assignment can be checked without pyOMA."""
    rng = np.random.default_rng(seed)
    n_modes = baseline['f'].shape[0]
    n_slots = order + 1
    f = np.zeros((order + 1, n_slots))
    d = np.zeros((order + 1, n_slots))
    phi = np.zeros((26, n_slots, order + 1), dtype=complex)
    std_f = np.zeros((order + 1, n_slots))
    std_d = np.zeros((order + 1, n_slots))
    f[order, :n_modes] = baseline['f'] * (1 + noise * rng.standard_normal(n_modes))
    d[order, :n_modes] = baseline['d']
    phi[:, :n_modes, order] = baseline['phi']
    std_f[order, :n_modes] = 1e-3
    std_d[order, :n_modes] = 1e-2
    return types.SimpleNamespace(
        modal_frequencies=f, modal_damping=d, mode_shapes=phi,
        std_frequencies=std_f, std_damping=std_d)


@needs_data
class TestModeAssignment:

    def test_keys_are_baseline_indices(self):
        baseline = ex.load_baseline_modes()
        modal_data = _fake_modal_data(baseline)
        params = {'highpass': 0.1, 'lowpass': 9.0}
        result = ex.assign_to_baseline(modal_data, 60, params, baseline)
        # a perfect identification recovers every baseline mode, keyed by index
        assert result['keys'] == list(range(27))
        assert result['point'].shape == (27, 4)
        assert np.allclose(result['f'], baseline['f'], rtol=1e-3)
        assert np.all(result['mac'] > 0.99)

    def test_band_limits_which_modes_can_pair(self):
        baseline = ex.load_baseline_modes()
        modal_data = _fake_modal_data(baseline)
        params = {'highpass': 0.1, 'lowpass': 1.5}
        result = ex.assign_to_baseline(modal_data, 60, params, baseline)
        # only the low-band baseline modes lie inside the analysis band
        assert result['keys'] == list(range(21))

    def test_no_poles_in_band_returns_empty(self):
        baseline = ex.load_baseline_modes()
        modal_data = _fake_modal_data(baseline)
        params = {'highpass': 6.0, 'lowpass': 9.0}
        result = ex.assign_to_baseline(modal_data, 60, params, baseline)
        assert result['keys'] == []
        assert 'rejected' in result


# ── statistic-level bookkeeping ──────────────────────────────────────────────

@needs_data
class TestStatisticLevel:

    @pytest.fixture(scope='class')
    def driven(self):
        """Drive estimate_stat with a stand-in estimator: no pyOMA, but the
        real feasibility screen, weights and stat_db bookkeeping."""
        baseline = ex.load_baseline_modes()
        vars_ale, vars_epi, levels, offsets = ex.vars_definition_experimental()
        pq = PolyUQ(vars_ale, vars_epi, dim_ex='cartesian', path='fast_build')
        pq.sample_qmc(N_mcs_ale=len(levels), N_mcs_epi=120, seed=7,
                      given_samples={'a_ref': levels})
        rng = np.random.default_rng(0)

        def estimator(n_epi, i_imp, weights):
            params = ex.sample_parameters(pq, n_epi)
            ok, reason, _ = ex.feasible(**params)
            if not ok:
                return ex._empty_result(reason)
            in_band = np.where((baseline['f'] > params['highpass'])
                               & (baseline['f'] < params['lowpass']))[0]
            if in_band.size == 0:
                return ex._empty_result('no baseline mode in band')
            keys = sorted(int(k) for k in in_band)
            f = baseline['f'][keys] * (1 + 1e-3 * rng.standard_normal(len(keys)))
            d = baseline['d'][keys]
            return {'keys': keys,
                    'point': np.column_stack([f, d, f * 1e-3, d * 1e-2]),
                    'n_eff': float(PolyUQ.kish_n_eff(weights))}

        pq.estimate_stat(estimator, weighted=True, eliminate=False)
        return pq, baseline

    def test_every_epistemic_sample_has_an_entry(self, driven):
        pq, _ = driven
        # the weights depend only on the Incompleteness variables, so all of a
        # sample's Imprecision hypercubes share one weight vector and dedupe
        # to a single entry
        assert len(pq.stat_db) == pq.N_mcs_epi
        assert {e['n_epi'] for e in pq.stat_db} == set(range(pq.N_mcs_epi))

    def test_mode_selection_finds_the_right_rows(self, driven):
        pq, baseline = driven
        for label in (0, 5, 26):
            for i_entry, i_row in ex.mode_selection(pq, label):
                assert pq.stat_db[i_entry]['keys'][i_row] == label

    def test_coverage_is_relative_to_feasible_samples(self, driven):
        pq, baseline = driven
        coverage = ex.mode_coverage(pq, baseline)
        assert len(coverage) == 27
        assert (coverage.coverage <= 1.0).all()
        # low-frequency modes lie in the wider set of admissible bands
        assert coverage.loc[coverage.band == 'low', 'n_found'].sum() > 0

    def test_statistic_level_rows_and_hypercubes(self, driven):
        pq, baseline = driven
        label = int(ex.mode_coverage(pq, baseline)
                    .sort_values('n_found').iloc[-1]['label'])
        pq_stat, hyc_rows = ex.statistic_level(pq, label, field='point',
                                               i_stat=ex.POINT_FIELDS['f'])
        # the Incompleteness variables are lifted to primary, multiplying the
        # 32 Imprecision hypercubes by their 2 x 2 focal products
        assert len(pq_stat.imp_hyc_foc_inds) == 32 * 4
        assert pq_stat.out_samp.shape[1] == pq.N_mcs_epi
        # unfound / rejected samples are NaN, which estimate_imp tolerates
        assert np.isnan(pq_stat.out_samp).any()
        assert np.isfinite(pq_stat.out_samp).any()

    def test_unidentified_mode_raises(self, driven):
        pq, _ = driven
        with pytest.raises(ValueError, match='never identified'):
            ex.statistic_level(pq, 999)


# ── end to end, with the real identification ─────────────────────────────────

@needs_data
@pytest.mark.slow
@pytest.mark.data
class TestEndToEnd:

    def test_one_weighted_identification(self):
        baseline = ex.load_baseline_modes()
        vars_ale, vars_epi, levels, offsets = ex.vars_definition_experimental()
        pq = PolyUQ(vars_ale, vars_epi, dim_ex='cartesian', path='fast_build')
        pq.sample_qmc(N_mcs_ale=len(levels), N_mcs_epi=200, seed=1509,
                      given_samples={'a_ref': levels})
        per_sample, _ = ex.feasibility_report(pq)
        cheapest = per_sample[per_sample.ok].sort_values('num_block_rows')
        n_epi = int(cheapest.iloc[0]['n_epi'])

        estimator = ex.make_experimental_estimator(pq, baseline, offsets,
                                                   weighting='build')
        weights = pq._weights(i_imp=0, n_epi=n_epi, eliminate=False)
        result = estimator(n_epi, 0, weights)

        assert 'rejected' not in result
        assert len(result['keys']) > 0
        assert result['point'].shape == (len(result['keys']), 4)
        assert np.all(np.isfinite(result['f']))
        assert np.all(result['std_f'] >= 0)
        assert np.all(result['mac'] > 0.5)
        assert 1.0 < result['n_eff'] <= len(levels)
