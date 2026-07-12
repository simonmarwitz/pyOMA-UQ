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
        assert len(vars_epi) == 7
        names = {v.name for v in vars_epi}
        assert names == {'c_vb', 'lamda_vb', 'n_locations', 'DAQ_noise_rms',
                         'decimation_factor', 'tau_max', 'model_order'}
        # secondary Incompleteness variables drive the weights
        secondary = {v.name for v in vars_epi if not v.primary}
        assert secondary == {'c_vb', 'lamda_vb'}
        # model_order acts on the OMA stage only, not on the lattice
        assert 'model_order' not in arg_vars.values()
        assert 'duration' in fixed and fixed['duration'] == 300.0
        assert 'm_lags' not in fixed  # derived from tau_max * fs

    def test_hypercube_counts(self, poly_uq_small):
        # Imprecision hypercubes: n_loc(3) x DAQ(1) x dec(1) x tau(2) x ord(3)
        assert len(poly_uq_small.imp_hyc_foc_inds) == 18
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
        w_q = [uw.compute_weights(poly_uq_small, i_imp, 0)
               for i_imp in range(18)]
        for w in w_q[1:]:
            assert np.allclose(w, w_q[0])

    def test_depends_on_incompleteness_sample(self, poly_uq_small):
        suppl = poly_uq_small.inp_suppl_epi
        n_other = np.argmax(np.abs(suppl['lamda_vb'].values
                                   - suppl['lamda_vb'].iloc[0]))
        w0 = uw.compute_weights(poly_uq_small, 0, 0)
        w1 = uw.compute_weights(poly_uq_small, 0, int(n_other))
        assert not np.allclose(w0, w1)

    def test_elimination_mask_degenerate(self, poly_uq_small):
        mask = uw.elimination_mask(poly_uq_small, 0, 0)
        assert mask.dtype == bool
        assert np.all(mask)


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

        # dedupe: one distinct identification per epistemic sample
        assert len(calls) == self.N_EPI
        assert len(pole_db) == self.N_EPI * 18
        # each identification received that epistemic sample's corr stack
        for n_epi, (corr, weights, order, fs) in enumerate(calls):
            assert np.all(corr[0] == 100 * n_epi)
            assert np.isclose(np.sum(weights), 1.0)
            assert order == int(pq.inp_samp_prim['model_order'].iloc[n_epi])
        # bookkeeping: every (n_epi, i_imp) cell present exactly once
        cells = {(e['n_epi'], e['i_imp']) for e in pole_db}
        assert len(cells) == self.N_EPI * 18
        assert all(e['weights_id'] == 0 for e in pole_db)


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


class TestStatisticLevel:

    def test_instance_construction(self, poly_uq_small):
        rng = np.random.default_rng(0)
        N_epi = poly_uq_small.N_mcs_epi
        theta_mode = {'f': np.tile(
            1.0 + 0.001 * poly_uq_small.inp_samp_prim['model_order']
            .values[:N_epi] + rng.normal(0, 1e-4, N_epi), (18, 1))}
        theta_mode['f'][:, 2] = np.nan  # a cell where the mode was missed
        pq_stat = uw.statistic_level_polyuq(poly_uq_small, theta_mode, 'f')

        # combined Imprecision x Incompleteness hypercubes: 18 x (2*2*1)
        assert len(pq_stat.imp_hyc_foc_inds) == 72
        assert np.isclose(np.sum(pq_stat.imp_hyc_mass), 1.0)
        assert pq_stat.N_mcs_ale == 1
        assert pq_stat.N_mcs_epi == N_epi
        assert set(pq_stat.inp_samp_prim.columns) == {
            'n_locations', 'DAQ_noise_rms', 'decimation_factor', 'tau_max',
            'model_order', 'c_vb', 'lamda_vb'}
        # all statistic-level variables are primary (their products are the
        # combined hypercubes); NaN cells are carried, not dropped
        assert len(pq_stat.vars_imp) == 7
        assert np.isnan(pq_stat.out_samp[0, 2])

    def test_all_nan_raises(self, poly_uq_small):
        theta_mode = {'f': np.full((18, poly_uq_small.N_mcs_epi), np.nan)}
        with pytest.raises(ValueError, match='not found'):
            uw.statistic_level_polyuq(poly_uq_small, theta_mode, 'f')


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
        assert len(pole_db) == N_epi * 18

        for entry in pole_db:
            assert np.all(np.isfinite(entry['std_f']))
            assert np.all(entry['std_f'] >= 0)

        # the lightly damped, well-separated modes must be among the poles
        # of every epistemic sample (300 s records, n_eff ~ 3 of 5 blocks)
        for n_epi in range(N_epi):
            entry = next(e for e in pole_db
                         if e['n_epi'] == n_epi and e['i_imp'] == 0)
            for f_k in (0.5801, 0.6038, 1.1995, 1.2495):
                rel_err = np.min(np.abs(entry['f'] - f_k)) / f_k
                assert rel_err < 0.01, \
                    f'mode {f_k} Hz missing at n_epi={n_epi}'
