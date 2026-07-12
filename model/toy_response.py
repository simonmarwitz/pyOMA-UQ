'''
Development-data generator for the weighted-OMA case study
("Toward OMA-Specific PolyUQ Processing Strategies").

The original case study synthesizes ambient responses of the guyed mast on
an HPC cluster from a 319 GB expanded FRF (``Mechanical.transient_ifrf``).
That FRF is merely the expansion of the mast's non-classical modal solution,
which is stored completely in the 2 MB ``mechanical.npz`` archive
(``MechanicalDummy.load``-able). This module rebuilds a compact
modal-superposition FRF restricted to the candidate sensor locations and
synthesizes acceleration responses by seeded random-phase excitation,
following the pattern of ``pyOMA/tests/system_ambient_ifrf.py``. The
synthetic data therefore share the physical modes of the real case study
(dissertation Table tab:modal_params: 0.157, 0.163, 0.179, ... Hz), so
identified modal parameters can be validated against known values.

The FRF assembly replicates ``Mechanical.frequency_response_non_classical``
(Brincker & Ventura, Introduction to Operational Modal Analysis, p. 99 ff):

.. math::

    H_d(\\omega) = \\sum_n \\frac{\\phi_{inp,n} \\phi_{out,n} / a_n}
                              {i\\omega - \\lambda_n}
                 + \\frac{(\\phi_{inp,n} \\phi_{out,n} / a_n)^*}
                        {i\\omega - \\lambda_n^*}

with :math:`H_a = -\\omega^2 H_d` for accelerations and
:math:`\\lambda_n = 2 \\pi f_{d,n} (-\\zeta_n / \\sqrt{1 - \\zeta_n^2} + i)`
reconstructed from the stored damped frequencies and damping ratios.
'''

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Candidate sensor nodes as enumerated by acquisition.sensor_position for the
# 203-node mast model: every fifth node plus the tip reference node 201.
CANDIDATE_NODES = np.concatenate((np.arange(5, 203, 5), [201]))

# Wind loading acts along the whole mast; a coarse subset of loaded nodes
# keeps the FRF small while exciting all lateral modes.
DEFAULT_INP_NODES = np.arange(10, 201, 10)

# Excitation scale such that the overall acceleration RMS is 5e-3 m/s^2 at
# the median basic wind velocity (v_b = 5 m/s), placing the epistemic DAQ
# noise interval (2.5e-6, 3e-3) m/s^2 RMS between the negligible and the
# dominating regime. Obtained from calibrate_force_scale() with the default
# input/output nodes at N = 2**15, fs = 70 Hz.
DEFAULT_FORCE_SCALE = 0.042


def reconstruct_lamda(damped_frequencies, modal_damping):
    '''
    Reconstruct the complex eigenvalues from damped natural frequencies (Hz)
    and modal damping ratios, inverting the relations used in
    ``Mechanical.numerical_modal_analysis``:
    ``frequencies = imag(lamda) / 2 / pi`` and
    ``damping = -real(lamda) / abs(lamda)``.
    '''
    f_d = np.asarray(damped_frequencies, dtype=float)
    zeta = np.asarray(modal_damping, dtype=float)
    sigma = -zeta * f_d / np.sqrt(1 - zeta ** 2)
    return (sigma + 1j * f_d) * 2 * np.pi


def modal_frf(mech, N, fs, inp_nodes=None, out_nodes=None, out_quant='a'):
    '''
    Assemble the modal-superposition FRF of the mast from its stored
    non-classical modal solution, restricted to the lateral dofs (uy, uz)
    of ``out_nodes``.

    Parameters
    ----------
    mech : MechanicalDummy
        Loaded from ``mechanical.npz``; must provide ``damped_frequencies``,
        ``modal_damping``, ``damped_mode_shapes`` (input expansion),
        ``mode_shapes_comp`` (compensated output shapes, ordered
        [uy nodes 1..203, uz nodes 1..203]) and ``gen_mod_coeff``.
    N : int
        Number of time steps of the excitation; the FRF has N // 2 + 1 lines.
    fs : float
        Sampling rate in Hz; the Nyquist frequency fs / 2 must exceed the
        highest natural frequency (32.7 Hz for the mast model).
    inp_nodes, out_nodes : ndarray of int, optional
        Loaded and responding node numbers; default to DEFAULT_INP_NODES
        and CANDIDATE_NODES.
    out_quant : {'a', 'v', 'd'}
        Response quantity of the returned FRF.

    Returns
    -------
    omegas : ndarray (N // 2 + 1,)
        Angular frequencies of the FRF lines.
    frf : ndarray (N // 2 + 1, 2 * len(inp_nodes), 2 * len(out_nodes)), complex64
        Input and output channels are ordered [uy all nodes, uz all nodes],
        matching the Fortran-order reshape used in
        ``MechanicalDummy.transient_ifrf``.
    '''
    if inp_nodes is None:
        inp_nodes = DEFAULT_INP_NODES
    if out_nodes is None:
        out_nodes = CANDIDATE_NODES
    inp_nodes = np.asarray(inp_nodes, dtype=int)
    out_nodes = np.asarray(out_nodes, dtype=int)

    lamda = reconstruct_lamda(mech.damped_frequencies, mech.modal_damping)
    num_modes = lamda.shape[0]

    fmax = fs / 2
    f_max_mode = np.max(np.imag(lamda)) / 2 / np.pi
    if f_max_mode > fmax:
        raise ValueError(
            f'The Nyquist frequency (={fmax}) is below the highest natural '
            f'frequency (={f_max_mode})')
    omegas = np.fft.rfftfreq(N, 1 / (2 * fmax)) * 2 * np.pi

    # node number -> row index in nodes_coordinates / meas_nodes (nodes 1..203)
    all_nodes = np.asarray(mech.nodes_coordinates, dtype=float)[:, 0].astype(int)
    inp_indices = np.searchsorted(all_nodes, inp_nodes)
    out_indices = np.searchsorted(all_nodes, out_nodes)
    if (np.any(inp_indices >= all_nodes.shape[0])
            or np.any(out_indices >= all_nodes.shape[0])
            or np.any(all_nodes[inp_indices] != inp_nodes)
            or np.any(all_nodes[out_indices] != out_nodes)):
        raise ValueError('Some input or output nodes are not model nodes.')

    # input mode shapes from the (node, dof, mode) expansion, dofs uy=1, uz=2
    mode_shapes_full = mech.damped_mode_shapes
    phi_inp = np.vstack([mode_shapes_full[inp_indices, dof, :num_modes]
                         for dof in (1, 2)])
    # output mode shapes from the compensated (channel, mode) matrix,
    # rows ordered [uy nodes 1..203, uz nodes 1..203]
    num_nodes = all_nodes.shape[0]
    mode_shapes_out = mech.mode_shapes_comp
    phi_out = np.vstack([mode_shapes_out[dof_block * num_nodes + out_indices, :num_modes]
                         for dof_block in (0, 1)])

    logger.info(f'FRF assembly with {N // 2 + 1} frequency lines up to '
                f'{fmax} Hz and {num_modes} modes for {phi_inp.shape[0]} '
                f'input and {phi_out.shape[0]} output channels.')

    # residues (n_inp * n_out, mode) for the modal poles and their conjugates
    a_n = mech.gen_mod_coeff[:num_modes]
    residues = (phi_inp[:, np.newaxis, :] * phi_out[np.newaxis, :, :]
                / a_n[np.newaxis, np.newaxis, :])
    n_inp, n_out = phi_inp.shape[0], phi_out.shape[0]
    residues = residues.reshape((n_inp * n_out, num_modes))

    # frequency kernels for all poles at once: single BLAS call
    poles = np.concatenate((lamda, np.conj(lamda)))
    residues = np.hstack((residues, np.conj(residues)))
    omega_scale = 1 / (1j * omegas[:, np.newaxis] - poles[np.newaxis, :])

    frf = (omega_scale @ residues.T).reshape((N // 2 + 1, n_inp, n_out))

    if out_quant == 'a':
        frf *= -omegas[:, np.newaxis, np.newaxis] ** 2
    elif out_quant == 'v':
        frf *= 1j * omegas[:, np.newaxis, np.newaxis]
    elif out_quant != 'd':
        raise ValueError(f'This output quantity is invalid: {out_quant}')

    return omegas, frf.astype(np.complex64)


def synthesize_response(frf, fs, v_b, seed, force_scale=DEFAULT_FORCE_SCALE):
    '''
    Synthesize one ambient acceleration response by seeded broadband random
    excitation through the modal FRF (random-phase IFFT synthesis as in
    ``system_ambient_ifrf.ambient_ifrf``; wrap-around instead of transients).

    The excitation is white Gaussian force at every input channel with a
    standard deviation of ``force_scale * v_b ** 2``, reflecting the
    quadratic growth of quasi-static wind pressure with the basic wind
    velocity: aleatory samples differ in excitation level and thus in their
    signal-to-noise ratio after acquisition.

    Parameters
    ----------
    frf : ndarray (N // 2 + 1, n_inp, n_out)
        Acceleration FRF from modal_frf.
    fs : float
        Sampling rate in Hz.
    v_b : float
        Basic wind velocity of this aleatory sample.
    seed : int
        Seed for the excitation realization.
    force_scale : float
        Force RMS per unit v_b ** 2.

    Returns
    -------
    t_vals : ndarray (N,)
        Time instants (starting at 1 / fs as in ``transient_ifrf``).
    accel : ndarray (N, n_out // 2, 2), float32
        Acceleration response at the output nodes, dofs (uy, uz).
    '''
    num_lines, n_inp, n_out = frf.shape
    N = 2 * (num_lines - 1)

    rng = np.random.default_rng(seed)
    force = rng.standard_normal((N, n_inp)) * (force_scale * v_b ** 2)
    force_freq = np.fft.rfft(force, axis=0)

    accel_freq = np.einsum('fi,fio->fo', force_freq, frf)
    accel = np.fft.irfft(accel_freq, n=N, axis=0)
    accel = accel.reshape((N, n_out // 2, 2), order='F')

    t_vals = np.arange(1, N + 1) / fs
    return t_vals, accel.astype(np.float32)


def generate_response(out_path, frf, fs, v_b, seed, out_nodes=None,
                      force_scale=DEFAULT_FORCE_SCALE):
    '''
    Synthesize one aleatory response sample and store it in the
    ``response.npz`` layout consumed by the acquisition stage (only the
    acceleration quantity is stored; displacements and velocities are not
    needed by the acceleration channel definitions).
    '''
    if out_nodes is None:
        out_nodes = CANDIDATE_NODES
    t_vals, accel = synthesize_response(frf, fs, v_b, seed, force_scale)
    np.savez_compressed(out_path,
                        t_vals=t_vals,
                        a_freq_time=accel,
                        meas_nodes=np.asarray(out_nodes, dtype=int),
                        v_b=v_b, seed=seed)
    return t_vals, accel


def calibrate_force_scale(frf, fs, v_b_ref=5.0, target_rms=5e-3, seed=0):
    '''
    Return the force_scale for which the RMS over all response channels
    equals ``target_rms`` (m/s^2) at the reference wind velocity — used
    once to fix DEFAULT_FORCE_SCALE; the response is linear in force_scale.
    '''
    _, accel = synthesize_response(frf, fs, v_b_ref, seed, force_scale=1.0)
    rms = np.sqrt(np.mean(np.square(accel, dtype=float)))
    return target_rms / rms
