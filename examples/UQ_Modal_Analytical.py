import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import scipy.stats.qmc
import sys
from polyuq import *
import logging

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)


def mapping_function(E, A, rho, L, omega_u, zeta,
                     add_mass, ice_occ, ice_mass,
                     omega_l=30, fs=200, N=2048, x=np.array([15, 35, 55, 75, 95, 115, 135, 155, 175, 195]),
                     jid=None, result_dir=None, working_dir=None):
    nyq_omega = fs / 2 * 2 * np.pi

    rho += add_mass / A
    rho += ice_occ * ice_mass / A

    num_modes = int((nyq_omega * L / np.pi / np.sqrt(E / rho) * 2 + 1) // 2)

    j = np.arange(1, num_modes + 1, 1)
    omegans = (2 * j - 1) / 2 * np.pi / L * np.sqrt(E / rho)

    alpha, beta = np.linalg.solve(np.array([[1 / omega_l, omega_l], [1 / omega_u, omega_u]]), [2 * zeta, 2 * zeta])

    zetas = 0.5 * (alpha / omegans + beta * omegans)

    omegas = np.fft.fftfreq(N, 1 / fs) * 2 * np.pi
    omegas = omegas[:, np.newaxis]
    frf = np.zeros((N, len(x)), dtype=complex)
    for mode in range(num_modes):
        omegan = omegans[mode]
        kappa = omegan ** 2
        zeta = zetas[mode]

        modal_coordinate = np.abs(np.sin((2 * (mode + 1) - 1) / 2 * np.pi / L * x))
        modal_coordinate = modal_coordinate[np.newaxis,:]
        frf += -omegan ** 2 / (kappa * (1 + 2 * 1j * zeta * omegas / omegan - (omegas / omegan) ** 2)) * modal_coordinate

    fd = omegans * np.sqrt(1 - zetas ** 2) / 2 / np.pi

    return fd, zetas, np.abs(frf)


def mapping_single(E, A, rho, L, omega_u=None, zeta=None,
                  add_mass=None, ice_occ=None, ice_mass=None,):
    if add_mass is not None:
        rho += add_mass / A
    if ice_occ is not None:
        rho += ice_occ * ice_mass / A
    j = 1
    omegan = (2 * j - 1) / 2 * np.pi / L * np.sqrt(E / rho)
    return omegan * np.sqrt(1 - zeta ** 2) / 2 / np.pi


def main():
    Elognorm = RandomVariable('lognorm', 'Elognorm', [0.074, 1.093e11, 9.58e10], primary=False)
    Enorm = RandomVariable('norm', 'Enorm', [2.0576e11, 6.9e9], primary=False)
    E = MassFunction('E', [(2.1e11,), (Elognorm,), (Enorm,)], [0.8, 0.1, 0.1], primary=True)
    logger.debug(Elognorm.support())
    logger.debug(Enorm.support())
    logger.debug(E.support())
    Anorm = RandomVariable('norm', 'Anorm', [0.0343, 1e-4], primary=False)
    A = MassFunction('A', [(0.0343,), (0.0329, 0.0338), (Anorm,)], [0.6, 0.2, 0.2], primary=True)
    logger.debug(Anorm.support())
    logger.debug(A.support())
    rho = MassFunction('rho', [(7850,), (7700, 7850)], [0.7, 0.3], primary=True)
    add_mass = MassFunction('add_mass', [(0.1, 1), ], [1], primary=True)
    ice_days = MassFunction('ice_days', [(28.2 / 365,), (1 / 365, 77 / 365)], [0.3, 0.7], primary=False)
    ice_occ = RandomVariable('bernoulli', 'ice_occ', [ice_days], primary=True)
    ice_mass = MassFunction('ice_mass', [(0.5, 1), ], [1], primary=True)
    logger.debug(rho.support())
    logger.debug(add_mass.support())
    logger.debug(ice_occ.support())
    logger.debug(ice_mass.support())
    L = MassFunction('L', [(200,), (198, 202)], [0.8, 0.2], primary=True)
    logger.debug(L.support())
    # omega_l = MassFunction('omega_l', [(38,), (30,50)], [0.5, 0.5], primary=True)
    omega_u = MassFunction('omega_u', [(440,), (300, 500)], [0.5, 0.5], primary=True)
    zeta = MassFunction('zeta', [(0.047,), (0.047, 0.058), (0.047, 0.084), (0.0016, 0.015)], [0.3, 0.3, 0.2, 0.2], primary=True)
    # logger.debug(omega_l.support())
    logger.debug(omega_u.support())
    logger.debug(zeta.support())

    vars_epi = [E, A, rho, add_mass, ice_mass, L, omega_u, zeta, ice_days]
    # vars_epi = [E, A, rho, L]
    vars_ale = [Elognorm, Enorm, Anorm, ice_occ]
    # vars_ale = [Elognorm, Enorm, Anorm]

    # E, A, rho, L, omega_l, omega_u, zeta, add_mass, ice_occ, ice_mass,
    arg_vars = {'E':E.name, 'A':A.name, 'rho':rho.name, 'L':L.name,
                'omega_u':omega_u.name, 'zeta':zeta.name,
                'add_mass':add_mass.name, 'ice_occ':ice_occ.name, 'ice_mass':ice_mass.name}
    # arg_vars = {'E':E.name, 'A':A.name, 'rho':rho.name, 'L':L.name,
                # }

    dim_ex = 'cartesian'

    # %%snakeviz
    N_mcs_ale = 3
    N_mcs_epi = 10
    use_dm = True
    ret_name = 'frf'
    ret_ind = {'frequencies':105, 'space':8}

    import tempfile
    from polyuq.data_manager import DataManager
    result_dir = os.environ.get('UQ_MODAL_RESULT_DIR', os.path.join(os.getcwd(), 'polyuq_results', 'uq_modal'))

    if False:
        poly_uq = PolyUQ(vars_ale, vars_epi, dim_ex=dim_ex)
        poly_uq.sample_qmc(N_mcs_ale, N_mcs_epi, check_discr=False)
        poly_uq.save_state(os.path.join(result_dir, 'polyuq_samp.npz'))

        if use_dm:
            dm_grid, _, _ = poly_uq.to_data_manager('example',
                                                    working_dir=tempfile.gettempdir(),
                                                    result_dir=result_dir,
                                                    overwrite=True)
            dm_grid.evaluate_samples(mapping_function, arg_vars,
                                     ret_names={'omegans':('modes',), 'zetas':('modes',), 'frf':('frequencies', 'space',)},
                                     default_len={'modes':20, 'frequencies':2048, 'space':10}, dry_run=True)

            poly_uq.from_data_manager(dm_grid, ret_name, ret_ind)
        else:
            poly_uq.propagate(mapping_single, arg_vars)

        poly_uq.save_state(os.path.join(result_dir, 'polyuq_prop.npz'))
    else:
        poly_uq = PolyUQ(vars_ale, vars_epi, dim_ex=dim_ex)
        poly_uq.load_state(os.path.join(result_dir, 'polyuq_prop.npz'))

        if use_dm:
            dm_grid = DataManager.from_existing('example.nc', working_dir=tempfile.gettempdir(),
                                                           result_dir=result_dir)
            poly_uq.from_data_manager(dm_grid, ret_name, ret_ind)

    # print(poly_uq.out_samp)
    poly_uq.estimate_imp(interpolate=True, opt_meth='Powell')

    poly_uq.save_state(os.path.join(result_dir, f'{ret_name}-{ret_ind}', 'polyuq_imp.npz'))

    def stat_fun(a, weight, i_stat):
        return np.average(a, weights=weight)

    n_stat = 1

    # focals_Pf, hyc_mass = poly_uq.estimate_inc(intervals, stat_fun, n_stat)
    focals_Pf, hyc_mass = poly_uq.optimize_inc(stat_fun, n_stat)

    poly_uq.save_state(os.path.join(result_dir, f'{ret_name}-{ret_ind}', 'polyuq_avg_inc.npz'))

    print(focals_Pf, hyc_mass)


if __name__ == '__main__':
    main()
