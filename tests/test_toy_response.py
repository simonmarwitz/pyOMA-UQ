"""
Generator tests — validate the development-data generator
(model/toy_response.py) against the mast's known modal solution.

All tests require the mast's modal archive ``samples/mechanical.npz``
(2 MB) under POLYUQ_DATA_DIR and are marked ``data``. The identification
test additionally carries the ``slow`` marker (~1 min): it synthesizes a
31-minute clean ambient record and verifies that an unweighted
SSI-Cov identification recovers the known modal frequencies — the
validation anchor for everything built on the synthetic data.
"""
import os
from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(os.environ.get('POLYUQ_DATA_DIR',
                               '/home/womo1998/Projects/uq_oma_a'))
MECH_NPZ = DATA_DIR / 'samples' / 'mechanical.npz'

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(not MECH_NPZ.exists(),
                       reason=f'mechanical.npz not found at {MECH_NPZ}'),
]

FS = 70.0
TEST_NODES = np.array([50, 100, 150, 201])


@pytest.fixture(scope='module')
def mech():
    from pyoma_uq.models.mechanical import MechanicalDummy
    return MechanicalDummy.load(str(MECH_NPZ))


@pytest.fixture(scope='module')
def small_frf(mech):
    """Acceleration FRF restricted to four output nodes, N = 2**15."""
    from model import toy_response
    return toy_response.modal_frf(mech, N=2 ** 15, fs=FS,
                                  out_nodes=TEST_NODES)


class TestModalFrf:

    def test_lamda_round_trip(self, mech):
        from pyoma_uq.models.toy_response import reconstruct_lamda
        lamda = reconstruct_lamda(mech.damped_frequencies, mech.modal_damping)
        assert np.allclose(np.imag(lamda) / 2 / np.pi,
                           mech.damped_frequencies)
        assert np.allclose(-np.real(lamda) / np.abs(lamda),
                           mech.modal_damping)
        assert np.all(np.real(lamda) < 0)

    def test_shape_dtype_and_dc(self, small_frf):
        omegas, frf = small_frf
        n_lines = 2 ** 15 // 2 + 1
        assert omegas.shape == (n_lines,)
        assert frf.shape == (n_lines, 40, 8)
        assert frf.dtype == np.complex64
        # acceleration FRF vanishes at DC (H_a = -omega**2 * H_d)
        assert np.all(frf[0] == 0)

    def test_frf_peaks_at_modes(self, small_frf, mech):
        omegas, frf = small_frf
        freqs = omegas / 2 / np.pi
        df = freqs[1] - freqs[0]
        mag = np.sum(np.abs(frf), axis=(1, 2))
        # the lightly damped bending pair around 0.58 / 0.60 Hz must
        # produce local maxima at the correct lines
        for f_k in (0.5801, 0.6038, 1.1995, 1.2495):
            # band narrower than the pair separation (~0.024 / 0.05 Hz)
            band = (freqs > f_k - 0.01) & (freqs < f_k + 0.01)
            f_peak = freqs[band][np.argmax(mag[band])]
            assert abs(f_peak - f_k) <= 2 * df

    def test_nyquist_violation_raises(self, mech):
        from model import toy_response
        with pytest.raises(ValueError):
            # highest mode is 32.7 Hz > 15 Hz Nyquist
            toy_response.modal_frf(mech, N=1024, fs=30.0,
                                   out_nodes=TEST_NODES)

    def test_unknown_node_raises(self, mech):
        from model import toy_response
        with pytest.raises(ValueError):
            toy_response.modal_frf(mech, N=1024, fs=FS,
                                   out_nodes=np.array([50, 999]))


class TestSynthesis:

    def test_seed_reproducibility(self, small_frf):
        from pyoma_uq.models.toy_response import synthesize_response
        _, frf = small_frf
        t1, a1 = synthesize_response(frf, FS, v_b=5.0, seed=3)
        t2, a2 = synthesize_response(frf, FS, v_b=5.0, seed=3)
        _, a3 = synthesize_response(frf, FS, v_b=5.0, seed=4)
        assert np.array_equal(a1, a2)
        assert not np.array_equal(a1, a3)
        assert t1.shape == (2 ** 15,)
        assert a1.shape == (2 ** 15, 4, 2)
        assert a1.dtype == np.float32

    def test_vb_quadratic_scaling(self, small_frf):
        from pyoma_uq.models.toy_response import synthesize_response
        _, frf = small_frf
        _, a1 = synthesize_response(frf, FS, v_b=5.0, seed=3)
        _, a2 = synthesize_response(frf, FS, v_b=10.0, seed=3)
        rms1 = np.sqrt(np.mean(np.square(a1, dtype=float)))
        rms2 = np.sqrt(np.mean(np.square(a2, dtype=float)))
        assert np.isclose(rms2 / rms1, 4.0, rtol=1e-4)

    def test_generate_response_contract(self, small_frf, tmp_path):
        from pyoma_uq.models.toy_response import generate_response
        _, frf = small_frf
        fpath = tmp_path / 'response.npz'
        generate_response(fpath, frf, FS, v_b=5.0, seed=3,
                          out_nodes=TEST_NODES)
        with np.load(fpath) as arr:
            assert arr['a_freq_time'].shape == (2 ** 15, 4, 2)
            assert arr['a_freq_time'].dtype == np.float32
            assert arr['t_vals'].shape == (2 ** 15,)
            assert np.array_equal(arr['meas_nodes'], TEST_NODES)
            assert arr['v_b'] == 5.0
            assert arr['seed'] == 3


@pytest.mark.slow
class TestIdentification:
    """Unweighted SSI-Cov on a long clean record recovers the known modes."""

    # well-separated, lightly damped modes: tight tolerance
    MODES_TIGHT = (0.3154, 0.3352, 0.5801, 0.6038,
                   1.1995, 1.2495, 2.0111, 2.0974)
    # closely spaced first bending pair (zeta 1.4 / 2.0 %): looser
    MODES_LOOSE = (0.1570, 0.1633)
    # the ~9 %-damped TMD-coupled pair at 0.179 / 0.180 Hz is not
    # resolvable by OMA and is deliberately not asserted

    def test_recovers_known_modes(self, mech):
        from model import toy_response
        from pyOMA.core.PreProcessingTools import PreProcessSignals
        from pyOMA.core.SSICovRef import BRSSICovRef

        _, frf = toy_response.modal_frf(mech, N=2 ** 17, fs=FS,
                                        out_nodes=TEST_NODES)
        _, accel = toy_response.synthesize_response(frf, FS, v_b=10.0,
                                                    seed=7)
        sig = accel.reshape((accel.shape[0], -1), order='F').astype(float)

        ps = PreProcessSignals(sig, FS, ref_channels=[3, 7])
        ps.decimate_signals(7)
        ps.corr_welch(m_lags=400)

        obj = BRSSICovRef(ps)
        obj.build_toeplitz_cov(num_block_columns=100)
        obj.compute_modal_params(60)

        f_all, d_all = [], []
        for order in range(20, 60):
            f_row = obj.modal_frequencies[order, :]
            d_row = obj.modal_damping[order, :] / 100.0
            ok = ((f_row > 0.05) & (f_row < 3.0)
                  & (d_row > 0) & (d_row < 0.15))
            f_all.append(f_row[ok])
            d_all.append(d_row[ok])
        f_all = np.concatenate(f_all)
        d_all = np.concatenate(d_all)

        for f_k in self.MODES_TIGHT:
            rel_err = np.abs(f_all - f_k) / f_k
            best = np.argmin(rel_err)
            assert rel_err[best] < 0.005, f'mode {f_k} Hz not identified'
            # stable across model orders
            assert np.sum(rel_err < 0.01) >= 10, f'mode {f_k} Hz unstable'
            # damping of these modes is below 1.5 %; identified damping is
            # biased high on light damping, allow up to 5 %
            assert d_all[best] < 0.05

        for f_k in self.MODES_LOOSE:
            rel_err = np.abs(f_all - f_k) / f_k
            assert np.min(rel_err) < 0.015, f'mode {f_k} Hz not identified'
