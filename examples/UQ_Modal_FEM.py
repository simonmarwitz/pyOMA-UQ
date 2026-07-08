import os
import tempfile
import numpy as np
import zlib, zipfile
from model.mechanical import Mechanical, MechanicalDummy
import sys
from polyuq import MassFunction, RandomVariable, PolyUQ, \
stat_fun_avg, stat_fun_cdf, stat_fun_ci, stat_fun_hist, stat_fun_lci, stat_fun_pdf, generate_histogram_bins, aggregate_mass
import logging
import psutil
import time

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARNING)

logger_mech = logging.getLogger('model.mechanical')
logger_mech.setLevel(level=logging.WARNING)

global ansys


def default_mapping(E=2.1e11, a=0.9, b=0.875, t=0.009, rho=7850, N_wire=1.1e+05,
                     A_wire=0.00075, add_mass=55, zeta=0.0428, dD=198, alpha=0,  # structural
                     ice_occ=1, ice_mass=75,  # environmental
                     num_nodes=200, num_modes=14, fs=10, N=2048, meas_locs=np.array([40, 80, 120, 160, 200]),  # algorithmic
                     jid='abcdef123', result_dir=None, working_dir=None, skip_existing=False):
    if working_dir is None:
        import tempfile
        working_dir = tempfile.gettempdir()
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    return mapping_function(E, a, b, t, rho, N_wire, A_wire, add_mass, zeta, dD, alpha, ice_occ, ice_mass, num_nodes, num_modes, fs, N, meas_locs, jid, result_dir, working_dir, skip_existing)
#
# def mapping_validate(E=2.1e11, a=0.9, b=None, t=None, rho=7850, N_wire=None,
#                      A_wire=None, add_mass=None, zeta=None, dD=None,  alpha=0, # structural
#                      ice_occ=None, ice_mass=None, # environmental
#                      num_nodes=200, num_modes=14, fs=10, N=2048, meas_locs=np.array([40,80,120,160,200]), # algorithmic
#                      jid='abcdef123', result_dir=None, working_dir=None, skip_existing=True):
#     A = np.pi*(a*b-(a-t)*(b-t))
#
#     Iy = np.pi/4*(a*b**3 -(a-t)*(b-t)**3)
#     Iz = np.pi/4*(b*a**3 -(b-t)*(a-t)**3)
#
#     Aeq = A_wire/(1 + (A_wire*rho*9.819*70/N_wire)**2*E*A_wire/12/N_wire)
#
#     keq = (E * Aeq * (70**2 / (70**2 + 160**2)) + N_wire)/np.sqrt(70**2 + 160**2)
#
#     rho_pipe = rho + add_mass / A
#      #incremental bernoulli gives [0...0, 0...1] -> [0...0, 50...100
#     if ice_occ:
#         ice_mass *= 50
#         ice_mass += 50
#         rho_pipe += ice_mass / A
#
#     struct_parms = {
#             'L'         : 200,
#
#             'E'         : E,
#             'A'         : A,
#             'rho'       : rho_pipe,
#             'Iy'        : Iy,
#             'Iz'        : Iz,
#
#             'kz_nl'     : 1.7 * keq,
#             'ky_nl'     : 2 * keq,
#             'x_knl'     : 160,
#
#             'm_tmd'     : 800,
#             'ky_tmd'    : 1025.48,
#             'kz_tmd'    : 1025.48,
#             'dy_tmd'    : dD,
#             'dz_tmd'    : dD,
#             'x_tmd'     : 200,
#             }
#     # print(struct_parms)
#     if result_dir is None:
#         result_dir=os.getcwd()
#     savefolder = os.path.join(result_dir, jid)
#
#     # load/generate  mechanical object:
#     # ambient response of a n dof rod,
#     if os.path.exists(os.path.join(savefolder, f'{jid}_mechanical.npz')):
#
#         tim = os.stat(os.path.join(savefolder, f'{jid}_mechanical.npz')).st_ctime
#         try:
#             mech = MechanicalDummy.load(jid, savefolder)
#             # print(mech.struct_parms, type(mech.struct_parms))
#             for parm,val in struct_parms.items():
#                 if not np.isclose(val, mech.struct_parms[parm]):
#                     print(f'{jid} \t Parameter {parm} values differ {val} != {mech.struct_parms[parm]} {time.ctime(tim)}')
#                     # print(os.stat(os.path.join(savefolder, f'{jid}_mechanical.npz')).st_ctime)
#
#                     return (False,tim)
#         except (EOFError, KeyError, zlib.error, zipfile.BadZipFile, OSError):
#             print(f'File {os.path.join(savefolder, f"{jid}_mechanical.npz")} corrupted. Deleting.')
#             #os.remove(os.path.join(savefolder, f'{jid}_mechanical.npz'))
#             return (False,tim)
#         return (True,tim)
#     else:
        # return (False,0)


def mapping_function(E=2.1e11, a=0.9, b=None, t=None, rho=7850, N_wire=None,
                     A_wire=None, add_mass=None, zeta=None, dD=None, alpha=0,  # structural
                     ice_occ=None, ice_mass=None,  # environmental
                     num_nodes=200, num_modes=14, fs=10, N=2048, meas_locs=np.array([40, 80, 120, 160, 200]),  # algorithmic
                     jid='abcdef123', result_dir=None, working_dir=None, skip_existing=True):

    # logger.warning('MEAS_NODES definition has changed to include TMD node (total 6).')
    '''
    The TMD FRF might be more interesting than the 160 m FRF
    160 m FRF was originally chosen to have some effect due to the mode shapes, 
    which should mostly be affected by wire properties,
    otherwise only frequencies and damping really play a role
    
    we would have to blow up the datamanager database and save it again
    from_datamanager should then work with space_ind=2 for the TMD dof
    which means, we could just operate on the smaller set of 8327 samples that
    are yet to be computed, while using all 13717 samples for the other two frf nodes
    without having to recompute everything again
    
    in estimate_imp we have to make sure to start at 5390 and ignore all previous
    samples (which would be nan)
    also in optimize_inc we have to make sure the first 5390 samples are ignored
    a general procedure to ignore all-nan samples would be useful anyway
    
    
    '''
    # print(E,a,b,t,rho,N_wire,A_wire,add_mass,zeta,dD,ice_occ,ice_mass)

    A = np.pi * (a * b - (a - t) * (b - t))

    Iy = np.pi / 4 * (a * b ** 3 - (a - t) * (b - t) ** 3)
    Iz = np.pi / 4 * (b * a ** 3 - (b - t) * (a - t) ** 3)
    Iyz = 0

    # approximately similar second moments of area are achieved with a truss
    # additionally, the diagonal and horizontal bracings have to be considered for the weight
    # they add a weight of 0.016005*rho per meter,
    # so we need a cross-section area of 0.033816 m² to achieve the same weight per meter
    # use a 160x15x15 L Profile at a 1.9 m distance slightly lower area, but similar moments of area
    # (also see photos of Sender Aholming)
    # use only the parts due to Steiner's theorem and reduce to 80 % due to truss flexibility
    # A_L = (2*a*t - t*t)*4 = 0.0183
    # Iy_L = 0.8*A_L*(L/2)**2 = 0.0132
    # Iz_L = 0.8*A_L*(L/2)**2 = 0.0132
    # also see excel workbook 'pipe_to_truss_conversion.xlsx'

    # rotate cross section about angle alpha
    alpha = alpha * 2 * np.pi / 360
    rIy = 0.5 * (Iy + Iz) + 0.5 * (Iy - Iz) * np.cos(2 * alpha) + Iyz * np.sin(2 * alpha)
    rIz = 0.5 * (Iy + Iz) - 0.5 * (Iy - Iz) * np.cos(2 * alpha) - Iyz * np.sin(2 * alpha)
    rIyz = -0.5 * (Iy - Iz) * np.sin(2 * alpha) + Iyz * np.cos(2 * alpha)

    Aeq = A_wire / (1 + (A_wire * rho * 9.819 * 70 / N_wire) ** 2 * E * A_wire / 12 / N_wire)

    keq = (E * Aeq * (70 ** 2 / (70 ** 2 + 160 ** 2)) + N_wire) / np.sqrt(70 ** 2 + 160 ** 2)

    rho_pipe = rho + add_mass / A
     # incremental bernoulli gives [0...0, 0...1] -> [0...0, 50...100
    if ice_occ:
        ice_mass *= 50
        ice_mass += 50
        rho_pipe += ice_mass / A

    struct_parms = {
            'L': 200,

            'E': E,
            'A': A,
            'rho': rho_pipe,
            'Iy': rIy,
            'Iz': rIz,
            'Iyz': rIyz,

            'kz_nl': 1.7 * keq,
            'ky_nl': 2 * keq,
            'x_knl': 160,

            'm_tmd': 800,
            'ky_tmd': 1025.48,
            'kz_tmd': 1025.48,
            'dy_tmd': dD,
            'dz_tmd': dD,
            'x_tmd': 200,
            }
    # print(struct_parms)
    if result_dir is None:
        result_dir = os.getcwd()
    savefolder = os.path.join(result_dir, jid)

    # load/generate  mechanical object:
    # ambient response of a n dof rod,
    mech = None
    if skip_existing:
        if os.path.exists(os.path.join(savefolder, f'{jid}_mechanical.npz')):
            try:
                mech = MechanicalDummy.load(jid, savefolder)
                for parm, val in struct_parms.items():
                    if not np.isclose(val, mech.struct_parms[parm]):
                        # print(f'{jid} \t Parameter {parm} values differ {val} != {mech.struct_parms[parm]}')
                        mech = None
                        break
            except Exception as e:
                # logger.warning(f'File {os.path.join(savefolder, f"{jid}_mechanical.npz")} corrupted. Deleting.')
                print(e)
                os.remove(os.path.join(savefolder, f'{jid}_mechanical.npz'))

    if mech is None or len(mech.meas_nodes) < 6:
        'python docs: The expression x or y first evaluates x; if x is true, its value is returned; otherwise, y is evaluated and the resulting value is returned.'

        global ansys
        if 'ansys' not in globals():
            ansys = Mechanical.start_ansys(working_dir=working_dir, jid=jid)
        # ansys = Mechanical.start_ansys(working_dir, jid)
        mech = Mechanical(ansys=ansys, jobname=jid, wdir=working_dir)

        mech.build_conti(struct_parms, Ldiv=num_nodes, damping=zeta,
                         num_modes=num_modes, meas_locs=meas_locs)

        # ansys.open_gui()
        inp_node = num_nodes  # should be the tip node
        _, frf_z = mech.frequency_response(N, inp_node, 'uz', fmax=fs // 2, out_quant='a')
        _, frf_y = mech.frequency_response(N, inp_node, 'uy', fmax=fs // 2, out_quant='a')

        # simulate unit force at 45° angle
        frf = (frf_z + frf_y) / np.sqrt(2)

        # hack to save the "angled frf" into the mech object
        mech.frf = frf
        mech.save(savefolder)
        mech.ansys.finish()

        for open_file in psutil.Process().open_files():
            if working_dir in open_file.path and 'rst' in open_file.path:
                # print(f'closing {open_file}')
                os.close(open_file.fd)
        # else:
        #     print(f'leaving {open_file}')

    fd = mech.damped_frequencies
    zetas = mech.modal_damping
    frf = mech.frf[:, -3:]
    # print(mech.meas_nodes[-3:])
    # print(E,a,b,t,rho,N_wire,A_wire, add_mass, zeta, dD, ice_occ, ice_mass, fd[0])
    # print(fd.astype('float32'))

    return fd.astype('float32'), zetas.astype('float32'), np.abs(frf).astype('float32')


def test_interpolation(ret_name, ret_ind, N_mcs_ale):

    b = MassFunction('b', [(1.7 / 2, 1.9 / 2), ], [1, ], primary=True)  # meter
    tnorm = RandomVariable('norm', 'tnorm', [6e-3, 1e-4], primary=False)  # meter
    t = MassFunction('t', [(5.9e-3, 6.1e-3), (tnorm,), ], [0.7, 0.3], primary=True)  # meter
    add_mass = MassFunction('add_mass', [(20, 100), (40, 50)], [0.8, 0.2], primary=True)  # kilogram per meter
    muNwire = MassFunction('muNwire', [(60000,), (40000, 180000)], [0.75, 0.25], primary=False)  # Newton
    N_wire = RandomVariable('norm', 'N_wire', [muNwire, 2655], primary=True)  # Newton
    A_wire = MassFunction('A_wire', [(0.0007, 0.0008)], [1, ], primary=True)  # meter^2
    zeta = MassFunction('zeta', [(0.0075, 0.0092), (0.0075, 0.0134), (0.0016, 0.015)], [0.5, 0.3, 0.2], primary=True)  # -
    stddD = MassFunction('vardD', [(0, 25), (10, 15)], [0.2, 0.8], primary=False)  # Newton second per meter
    dD = RandomVariable('norm', 'dD', [197.61, stddD], primary=True)  # Newton second per meter
    ice_days = MassFunction('ice_days', [(28.2 / 365,), (1 / 365, 77 / 365)], [0.3, 0.7], primary=False)  # days
    ice_occ = RandomVariable('bernoulli', 'ice_occ', [ice_days], primary=True)  # boolean
    ice_mass = MassFunction('ice_mass', [(0, ice_occ), ], [1], primary=True, incremental=True)  #

    vars_epi = [b, t, add_mass, muNwire, A_wire, zeta, stddD, ice_mass, ice_days]
    vars_ale = [tnorm, N_wire, dD, ice_occ]

    dim_ex = 'cartesian'

    result_dir = os.environ.get('UQ_MODAL_BEAM_DATA_DIR', os.path.join(os.getcwd(), 'polyuq_results', 'uq_modal_beam')) + '/'

    poly_uq = PolyUQ(vars_ale, vars_epi, dim_ex=dim_ex)

    ret_dir = f'{ret_name}-{".".join(str(e) for e in ret_ind.values())}'

    poly_uq.load_state(os.path.join(result_dir, 'estimations', f'{ret_dir}/polyuq_prop.npz'))

    poly_uq.N_mcs_ale = N_mcs_ale
    poly_uq.estimate_imp(
        interp_fun='rbf',
        opt_meth='genetic',
        plot_res=False,
        plot_intp=False,
        intp_err_warn=20,
        extrp_warn=10,
        start_ale=poly_uq.N_mcs_ale - 10,
        kernel='gaussian',
        epsilon={'frf':4, 'zetas':2, 'damp_freqs':2}[ret_name]
        )
    # poly_uq.save_state(os.path.join(result_dir,'estimations', f'{ret_dir}/polyuq_imp.npz'))
    return poly_uq.intp_errors, poly_uq.intp_exceed, poly_uq.intp_undershot


def test_domain():

    # ansys.exit()
    '''
    E           1.84e+11   2.3e+11    2.07e+11
    a           0.875      0.925      0.9
    b           0.875      0.925      0.9
    t           0.00569    0.00631    0.006
    rho         7.7e+03    7.85e+03   7.78e+03
    N_wire      3.18e+04   1.88e+05   1.1e+05
    A_wire      0.0007     0.0008     0.00075
    add_mass    10         100        55
    zeta        0.0016     0.084      0.0428
    dD          74         321        198
    ice_occ     0          1
    ice_mass    50         100        75
    '''
    titles = ['160 m', '200 m', 'TMD']
    frf_ind = 0
    import matplotlib.pyplot as plt
    import time
    # nominal
    ymin = 0
    ymax = 0.01
    skip_existing = False
    freqs = np.linspace(0, 5, 2048 // 2 + 1, False)
    now = time.time()
    f, zeta, frf = mapping_function(b=0.9, t=0.006, N_wire=110000,
                     A_wire=0.00075, add_mass=60, zeta=0.0083, dD=197.61,  # structural
                    ice_occ=1, ice_mass=0.5,  # environmental
                   jid='nominal12', working_dir=tempfile.gettempdir(), skip_existing=skip_existing)
    print(time.time() - now)

    for frf_ind in range(3):
        frf_ = frf[:, frf_ind]
        plt.figure()
        plt.plot(freqs, frf_, color='dimgrey')
        plt.vlines(x=f, ymin=ymin, ymax=ymax, colors='lightgrey', zorder=-10)
        plt.ylim((ymin, ymax))
        plt.twinx()
        plt.plot(f, zeta, ls='none', marker='x', color='black')
        plt.title(titles[frf_ind])
        plt.ylim((0, 0.14))
        plt.xlim((0, 5))
    # max f,d
    now = time.time()
    f, zeta, frf = mapping_function(b=0.95, t=0.00637190164854557, N_wire=189873.988768885,
                     A_wire=0.0008, add_mass=20, zeta=0.0016, dD=104.634587863608,  # structural
                    ice_occ=0, ice_mass=0,  # environmental
                     jid='max12', working_dir=tempfile.gettempdir(), skip_existing=skip_existing)
    print(time.time() - now)
    for frf_ind in range(3):
        frf_ = frf[:, frf_ind]
        plt.figure()
        plt.plot(freqs, frf_, color='dimgrey')
        plt.vlines(x=f, ymin=ymin, ymax=ymax, colors='lightgrey', zorder=-10)
        plt.ylim((ymin, ymax))
        plt.twinx()
        plt.plot(f, zeta, ls='none', marker='x', color='black')
        plt.title(titles[frf_ind])
        plt.ylim((0, 0.14))
        plt.xlim((0, 5))
    # min f,d
    now = time.time()
    f, zeta, frf = mapping_function(b=0.85, t=0.00562809835145443, N_wire=30126.0112311152,
                     A_wire=0.0007, add_mass=100, zeta=0.015, dD=290.585412136393,  # structural
                    ice_occ=1, ice_mass=1,  # environmental
                     jid='min12', working_dir=tempfile.gettempdir(), skip_existing=skip_existing)
    print(time.time() - now)
    for frf_ind in range(3):
        frf_ = frf[:, frf_ind]
        plt.figure()
        plt.plot(freqs, frf_, color='dimgrey')
        plt.vlines(x=f, ymin=ymin, ymax=ymax, colors='lightgrey', zorder=-10)
        plt.ylim((ymin, ymax))
        plt.twinx()
        plt.plot(f, zeta, ls='none', marker='x', color='black')
        plt.title(titles[frf_ind])
        plt.ylim((0, 0.14))
        plt.xlim((0, 5))

    plt.show()


def vars_definition():
    b = MassFunction('b', [(1.7 / 2, 1.9 / 2), ], [1, ], primary=True)  # meter
    tnorm = RandomVariable('norm', 'tnorm', [6e-3, 1e-4], primary=False)  # meter
    t = MassFunction('t', [(5.9e-3, 6.1e-3), (tnorm, tnorm,), ], [0.7, 0.3], primary=True)  # meter
    # t and tnorm should have had the same name to ensure samples are aligned
    # but that also means, the number of samples must be equal
    add_mass = MassFunction('add_mass', [(20, 100), (40, 50)], [0.8, 0.2], primary=True)  # kilogram per meter
    muNwire = MassFunction('muNwire', [(60000, 60000), (40000, 180000)], [0.75, 0.25], primary=False)  # Newton
    N_wire = RandomVariable('norm', 'N_wire', [muNwire, 2655], primary=True)  # Newton
    A_wire = MassFunction('A_wire', [(0.0007, 0.0008)], [1, ], primary=True)  # meter^2
    zeta = MassFunction('zeta', [(0.0075, 0.0092), (0.0075, 0.0134), (0.0016, 0.015)], [0.5, 0.3, 0.2], primary=True)  # -
    stddD = MassFunction('vardD', [(5, 20), (10, 15)], [0.2, 0.8], primary=False)  # Newton second per meter
    dD = RandomVariable('norm', 'dD', [197.61, stddD], primary=True)  # Newton second per meter
    ice_days = MassFunction('ice_days', [(28.2 / 365, 28.2 / 365), (1 / 365, 77 / 365)], [0.3, 0.7], primary=False)  # days, incompleteness
    ice_occ = RandomVariable('bernoulli', 'ice_occ', [ice_days], primary=True)  # boolean, variability
    # actually ice_occ is not a primary variable, that was just a hack
    # epistemic samples are generated on the support (of ice_mass)
    # for each aleatory sample, the epistemic samples are propagated
    # if ice_occ==0 the model is run with all those ice_masses, even though, they are not present
    # that is why ice_occ had to be passed to the mapping function and thus needed to be primary
    # actually it is secondary
    # in terms of the sensitivities, the model is now only sensitive to ice_occ and not to ice_mass
    # that should not be the case, but ice_mass is sampled, even when not occuring, so that makes it seem non-sensitive
    # we would have to hack around it in sensi, after the input sample lattice is re-combined and flattened
    # and multiply X[ice_mass] with X[ice_occ]
    ice_mass = MassFunction('ice_mass', [(0, ice_occ), ], [1], primary=True, incremental=True)  # , imprecision

    vars_epi = [b, t, add_mass, muNwire, A_wire, zeta, stddD, ice_mass, ice_days]  # 9, inc: 3
    # vars_epi = [E, A, rho, L]
    vars_ale = [tnorm, N_wire, dD, ice_occ]  # 4, sec: 1
    # vars_ale = [Elognorm, Enorm, Anorm]

    # E, a, b, t, rho, add_mass, N_wire, A_wire, zeta, dD, ice_occ, ice_mass,
    arg_vars = {'b':b.name,
                't':t.name,
                'add_mass':add_mass.name,
                'N_wire':N_wire.name,
                'A_wire':A_wire.name,
                'zeta':zeta.name,
                'dD':dD.name,
                'ice_occ':ice_occ.name,  #
                'ice_mass':ice_mass.name, }  # 9
    return vars_ale, vars_epi, arg_vars


def export_datamanager():
    from uncertainty.data_manager import DataManager
    vars_ale, vars_epi, arg_vars = vars_definition()
    dim_ex = 'cartesian'
    result_dir = os.environ.get('UQ_MODAL_BEAM_DATA_DIR', os.path.join(os.getcwd(), 'polyuq_results', 'uq_modal_beam')) + '/'

    dm_grid = DataManager.from_existing('uq_modal_beam.nc',
                                    result_dir=os.path.join(tempfile.gettempdir(), 'testsamples'),
                                    working_dir=tempfile.gettempdir())
    with dm_grid.get_database('out', False) as out_ds:
        out_ds_keep = out_ds.copy()
    out_ds = out_ds_keep

    logger = logging.getLogger('uncertainty.polymorphic_uncertainty')
    logger.setLevel(level=logging.DEBUG)

    for ret_name in ['damp_freqs', 'zetas', 'frf']:
        if ret_name == 'frf':
            n = 1025 * 2
        else:
            n = 14

        for i_n in range(n):
            if ret_name == 'frf':
                ret_ind = {'frequencies':i_n // 2, 'space':i_n % 2}
            else:
                ret_ind = {'modes':i_n}
            ret_dir = f'{ret_name}-{".".join(str(e) for e in ret_ind.values())}'

            print(ret_dir)

            poly_uq = PolyUQ(vars_ale, vars_epi, dim_ex=dim_ex)

            samp_path = os.path.join(result_dir, 'polyuq_samp.npz')
            prop_path = os.path.join(result_dir, 'estimations', f'{ret_dir}/polyuq_prop.npz')
            imp_path = os.path.join(result_dir, 'estimations', f'{ret_dir}/polyuq_imp.npz')

            if os.path.exists(prop_path):
                continue

            for path in [imp_path, prop_path, samp_path]:
                if os.path.exists(path):
                    poly_uq.load_state(path)

                    poly_uq.from_data_manager(None, ret_name, ret_ind, out_ds)
                    if ret_name == 'damp_freqs':
                        poly_uq.out_valid = [np.nanmin(poly_uq.out_samp), np.nanmax(poly_uq.out_samp)]
                    if path == samp_path:
                        path = prop_path
                    poly_uq.save_state(path)
                    break


def est_imp(poly_uq, result_dir, ret_name, ret_ind):
    ret_dir = f'{ret_name}-{".".join(str(e) for e in ret_ind.values())}'

    samp_path = os.path.join(result_dir, 'polyuq_samp.npz')
    prop_path = os.path.join(result_dir, 'estimations', f'{ret_dir}/polyuq_prop.npz')
    imp_path = os.path.join(result_dir, 'estimations', f'{ret_dir}/polyuq_imp.npz')

    poly_uq.load_state(samp_path, differential='samp')
    poly_uq.load_state(prop_path, differential='prop')
    poly_uq.N_mcs_ale = 13717

    if ret_ind.get('space', 0) == 2:
        start_ale = 5391
    else:
        start_ale = 0

    if os.path.exists(imp_path):
        poly_uq.load_state(imp_path, differential='imp')

        samp_fin = np.nonzero(
                np.any(
                    np.isnan(poly_uq.imp_foc[:,:, 0]),
                    axis=1)
            )[0]

        samp_fin = samp_fin[samp_fin > start_ale]

        if len(samp_fin) > 0:
            start_ale = np.min(samp_fin)
        else:
            start_ale = poly_uq.imp_foc.shape[0]

    while start_ale < poly_uq.N_mcs_ale:
        print(f'restarting {ret_dir} at sample {start_ale}')
        end_ale = min(start_ale + 100, poly_uq.N_mcs_ale)
        poly_uq.estimate_imp(
            interp_fun='rbf',
            opt_meth='genetic',
            plot_res=False,
            plot_intp=False,
            intp_err_warn=20,
            extrp_warn=10,
            start_ale=start_ale,
            end_ale=end_ale,
            kernel='gaussian',
            epsilon={'frf':4, 'zetas':2, 'damp_freqs':2}[ret_name]
        )
        poly_uq.save_state(imp_path, differential='imp')
        start_ale = end_ale


def opt_inc(poly_uq, result_dir, ret_name, ret_ind):

    ret_dir = f'{ret_name}-{".".join(str(e) for e in ret_ind.values())}'
    samp_path = os.path.join(result_dir, 'polyuq_samp.npz')
    prop_path = os.path.join(result_dir, 'estimations', f'{ret_dir}/polyuq_prop.npz')
    imp_path = os.path.join(result_dir, 'estimations', f'{ret_dir}/polyuq_imp.npz')

    poly_uq.load_state(samp_path, differential='samp')
    poly_uq.load_state(prop_path, differential='prop')
    poly_uq.load_state(imp_path, differential='imp')

    if False:  # analyze unfinished samples
        samp_fin = np.nonzero(
                np.any(
                    np.isnan(poly_uq.imp_foc[:,:, 0]),
                    axis=1)
            )[0]
        if len(samp_fin) > 0:
            end_ale = np.min(samp_fin)
            poly_uq.N_mcs_ale = end_ale
        else:
            poly_uq.N_mcs_ale = poly_uq.imp_foc.shape[0]

    def run_inc(poly_uq, inc_path, stat_fun, stat_fun_kwargs={}):
        print(stat_fun.__name__)
        if os.path.exists(inc_path) and False:
            print(f'{inc_path} already finished.')
            return
            poly_uq.load_state(inc_path)

        focals_stats, hyc_mass = poly_uq.optimize_inc(stat_fun, stat_fun.n_stat, stat_fun_kwargs)
        poly_uq.save_state(inc_path, differential='inc')

    if True:  # Average
        inc_path_part = 'polyuq_avg_inc.npz'
        inc_path = os.path.join(result_dir, 'estimations', ret_dir, inc_path_part)

        run_inc(poly_uq, inc_path, stat_fun_avg)

    if False:  # Confidence Interval
        inc_path_part = 'polyuq_ci_inc.npz'
        inc_path = os.path.join(result_dir, 'estimations', ret_dir, inc_path_part)

        run_inc(poly_uq, inc_path, stat_fun_ci)

    if False:  # Confidence Interval Length
        inc_path_part = 'polyuq_lci_inc.npz'
        inc_path = os.path.join(result_dir, 'estimations', ret_dir, inc_path_part)

        run_inc(poly_uq, inc_path, stat_fun_lci)

    if ret_name != 'frf':  # Histogram
        nbin_fact = 20
        n_imp_hyc = len(poly_uq.imp_hyc_foc_inds)
        bins_densities = generate_histogram_bins(poly_uq.imp_foc.reshape(poly_uq.N_mcs_ale, n_imp_hyc * 2), 1, nbin_fact / 2)  # divide nbin_fact by 2 to account for reshaping intervals
        stat_fun_hist.n_stat = len(bins_densities) - 1
        cum_dens = False
        stat_fun_kwargs = {'bins_densities':bins_densities, 'cum_dens':cum_dens}  # , 'ax':ax1}

        inc_path_part = 'polyuq_hist_inc.npz'
        inc_path = os.path.join(result_dir, 'estimations', ret_dir, inc_path_part)

        run_inc(poly_uq, inc_path, stat_fun_hist, stat_fun_kwargs)

    if ret_name != 'frf':  # Cumulative Density Function
        n_stat = 40
        target_probabilities = np.linspace(0, 1, n_stat)
        stat_fun_cdf.n_stat = n_stat
        stat_fun_kwargs = {'target_probabilities':target_probabilities}
        inc_path_part = 'polyuq_cdf_inc.npz'
        inc_path = os.path.join(result_dir, 'estimations', ret_dir, inc_path_part)

        run_inc(poly_uq, inc_path, stat_fun_cdf, stat_fun_kwargs)

    if False:  # Probability Density Function
        # too slow, returns nan often and optimizer just fails
        # bin indices should be pre-computed
        # generally -> variability is mixed with imprecision in the output -> misleading interpretations
        if True:
            n_imp_hyc = len(poly_uq.imp_hyc_foc_inds)
            '''
            How to get useful bin width and target_densities?
            depends on the samples and the weights (assumptio: mostly determined by samples)
            samples are present in xx imprecision hypercubes
            for weights take the mean value of distribution parameters
            that should not have too much of an influence?
            use a sufficient margin on target densities, e.g. factor 1.5
            '''
            for var in poly_uq.vars_inc:
                supp = poly_uq.var_supp[var.name].values
                var.freeze(np.mean(supp))

            if 'zeta' in poly_uq.out_name:
                nbin_fact = 5
            else:
                nbin_fact = 10

            n_imp_hyc = len(poly_uq.imp_hyc_foc_inds)
            bins_densities = generate_histogram_bins(poly_uq.imp_foc.reshape(poly_uq.N_mcs_ale, n_imp_hyc * 2), 1, nbin_fact / 2)  # divide nbin_fact by 2 to account for reshaping intervals
            bin_width = bins_densities[1] - bins_densities[0]
            max_dens = 0
            for i_hyc in range(n_imp_hyc):
                weights = poly_uq.probabilities_imp(i_hyc)
                for minmax in range(2):
                    samp = poly_uq.imp_foc[:, i_hyc, minmax]
                    sort_ind = np.argsort(samp)

                    this_max_dens = stat_fun_pdf(samp[sort_ind], weights[sort_ind], None, minmax, None, bin_width, True)
                    max_dens = max(max_dens, this_max_dens)
                # yn = input('next hypercube? (y,n)')
                # if yn=='n':
                #     break
            print(bin_width, max_dens)
            # return
        else:
            bin_width = 123
            max_dens = 1

        n_stat = 20
        target_pdfs = np.linspace(0, 1.2 * max_dens, n_stat)

        stat_fun_kwargs = {'target_densities':target_pdfs, 'bin_width':bin_width}  # , 'ax':ax1}
        stat_fun_pdf.n_stat = n_stat

        inc_path_part = 'polyuq_pdf_inc.npz'
        inc_path = os.path.join(result_dir, 'estimations', ret_dir, inc_path_part)

        run_inc(poly_uq, inc_path, stat_fun_pdf, stat_fun_kwargs)


def est_stoch(poly_uq, result_dir, ret_name, ret_ind):

    def stoch_fun_ecdf(samp_flat, weight_flat, target_probabilities):
        ecdf = np.cumsum(weight_flat)
        ecdf /= ecdf[-1]
        return np.interp(target_probabilities, ecdf, samp_flat)

    def stoch_fun_hist(samp_flat, weight_flat, bins, density):
        hist, _ = np.histogram(samp_flat, bins, weights=weight_flat, density=density)
        return hist

    ret_dir = f'{ret_name}-{".".join(str(e) for e in ret_ind.values())}'
    samp_path = os.path.join(result_dir, 'polyuq_samp_weights.npz')
    prop_path = os.path.join(result_dir, 'estimations', f'{ret_dir}/polyuq_prop.npz')
    imp_path = os.path.join(result_dir, 'estimations', f'{ret_dir}/polyuq_imp.npz')

    poly_uq.load_state(samp_path, differential='samp')
    poly_uq.load_state(prop_path, differential='prop')
    poly_uq.load_state(imp_path, differential='imp')

    if True:
        if ret_name != 'frf':
            # ensure same bins as for inc_avg_pl-ret_name-ret_ind

            inc_path = os.path.join(result_dir, 'estimations', f'{ret_dir}/polyuq_avg_inc.npz')
            poly_uq.load_state(inc_path, differential='inc')
            focals_stats, focals_mass = poly_uq.focals_stats, poly_uq.focals_mass
            _, _, _, bins_bel = aggregate_mass(focals_stats, focals_mass, 10, False)

            # nbin_fact=20
            # n_imp_hyc = len(poly_uq.imp_hyc_foc_inds)
            # bins_densities = generate_histogram_bins(poly_uq.imp_foc.reshape(poly_uq.N_mcs_ale, n_imp_hyc * 2), 1, nbin_fact/2) # divide nbin_fact by 2 to account for reshaping intervals
            stat_fun_hist.n_stat = len(bins_bel) - 1
            stat_fun_kwargs = {'bins':bins_bel, 'density':True}
        else:
            # ensure same bins as for inc_avg_pl-frf-xxx.x
            nbin_fact = 100
            n_hyc = len(poly_uq.hyc_mass(poly_uq.vars_epi))
            n_bins_dens = np.ceil(np.sqrt(n_hyc) * nbin_fact).astype(int)
            if ret_ind['space'] == 1:
                bins_densities = np.linspace(0, 0.012, n_bins_dens)
            elif ret_ind['space'] == 2:
                bins_densities = np.linspace(0, 0.0021, n_bins_dens)
            stat_fun_kwargs = {'bins':bins_densities, 'density':True}

        stoch_path_part = 'polyuq_hist_stoch.npz'
        stoch_path = os.path.join(result_dir, 'estimations', ret_dir, stoch_path_part)
        poly_uq.stat_full_stoch(stoch_fun_hist, stat_fun_kwargs)
        poly_uq.save_state(stoch_path, differential='stoch')

    if True:
        n_stat = 100
        target_probabilities = np.linspace(0, 1, n_stat)
        stat_fun_kwargs = {'target_probabilities':target_probabilities}
        stoch_path_part = 'polyuq_cdf_stoch.npz'
        stoch_path = os.path.join(result_dir, 'estimations', ret_dir, stoch_path_part)
        poly_uq.stat_full_stoch(stoch_fun_ecdf, stat_fun_kwargs)
        poly_uq.save_state(stoch_path, differential='stoch')


def plots():
    import matplotlib
    import matplotlib.pyplot as plt
    np.set_printoptions(precision=6)  # precision=2)

    from helpers import get_pcd
    print_context_dict = get_pcd('print')

    global ansys
    try:
        print(ansys)
    except:
        ansys = None

    mech = Mechanical(ansys, wdir='/run/user/30980/')
    ansys = mech.ansys

    # accelerance plots
    if False:
        fmax = 3
        plt.rc('text.latex', preamble="\\usepackage{siunitx}\\usepackage{xfrac}")
        with matplotlib.rc_context(rc=print_context_dict):
            fig, [ax1, ax2] = plt.subplots(2, 1, gridspec_kw={'height_ratios':[3, 1]}, sharex=True)
            for color, zeta in [('dimgrey', 0.01)
                                , ('grey', 0.05), ('lightgrey', 0.1)
                                ]:
                mech.example_beam(num_nodes=100, num_modes=14, damping=zeta, num_meas_nodes=1, mD=0)

                omegas, frf_y = mech.frequency_response(65536, 100, 'uz', fmax=fmax, out_quant='a')
                frf_y = frf_y[:, 0]
                # np.savez(os.path.join('polyuq_results', 'system_frf', f'UZ_{zeta}.npz'), frf)
                # ax1.plot(omegas/2/np.pi, np.abs(frf), color=color,label=f"$\\zeta={zeta}$")
                #
                # ax2.plot(omegas/2/np.pi,np.angle(frf)/np.pi*180, color=color)
                omegas, frf_z = mech.frequency_response(65536, 100, 'uy', fmax=fmax, out_quant='a')
                frf_z = frf_z[:, 0]
                # np.savez(os.path.join('polyuq_results', 'system_frf', f'UY_{zeta}.npz'),frf)
                # frf = (frf_y + frf_z) / np.sqrt(2)
                # frf_mag = np.sqrt((np.abs(frf_y)*np.cos(np.angle(frf_y)))**2 + (np.abs(frf_z)*np.cos(np.angle(frf_z)))**2)
                # frf_arg = np.arctan2(np.abs(frf_y)*np.cos(np.angle(frf_y)),np.abs(frf_z)*np.cos(np.angle(frf_z)))
                # frf = np.exp(frf_arg*1j)*frf_mag

                frf = 1000 * (frf_y + frf_z) / np.sqrt(2)

                ax1.plot(omegas / 2 / np.pi, np.abs(frf), color=color, label=f"$\\zeta={zeta}$")
                # ax1.plot(omegas/2/np.pi, np.abs(frf_y), color=color, alpha=0.5)
                # ax1.plot(omegas/2/np.pi, np.abs(frf_z), color=color, alpha=0.5)
                ax2.plot(omegas / 2 / np.pi, np.angle(frf) / np.pi * 180, color=color)
                # ax2.plot(omegas/2/np.pi,np.angle(frf_y)/np.pi*180, color=color, alpha=0.5)
                # ax2.plot(omegas/2/np.pi,np.angle(frf_z)/np.pi*180, color=color, alpha=0.5)

            ax1.set_yscale('log')

            # ax1.set_ylim((5e-06, 0.02))
            ax1.set_xlim((0, fmax))
            ax1.legend()
            ax1.set_ylabel('Accelerance $|\mathcal{H}_\mathrm{f-a}| [\si{\milli\meter\per\square\second\per\\newton}]$')
            ax2.set_ylabel('$\\arg\\bigl(H_{f-a}\\bigr)$ [\si{\degree}]')
            ax2.set_xlabel('Frequency [\si{\hertz}]')
            ax2.yaxis.set_major_locator(plt.MultipleLocator(90))
            ax2.yaxis.set_minor_locator(plt.MultipleLocator(45))
            fig.subplots_adjust(top=0.97, bottom=0.125, left=0.105, right=0.97, hspace=0.1)
            # plt.savefig('figures/frf_example_struc_beam.pdf')
            # plt.savefig('figures/frf_example_struc_beam.png')
            plt.show()

    if True:
        fmax = 2
        plt.rc('text.latex', preamble="\\usepackage{siunitx}\\usepackage{xfrac}")
        with matplotlib.rc_context(rc=print_context_dict):
            fig = plt.figure(figsize=(5.54 * 1.3, 1.3 * 1.3), dpi=100)
            ax1 = plt.subplot()
            zeta = 0.05
            color = 'dimgrey'

            mech.example_beam(num_nodes=100, num_modes=8, damping=zeta, num_meas_nodes=1, mD=0)

            omegas, frf_y, _, _ = mech.frequency_response_non_classical(65536, [100], ['uz'], fmax=fmax, out_quant='a')
            frf_y = frf_y[:, 0, 2]

            omegas, frf_z, _, _ = mech.frequency_response_non_classical(65536, [100], ['uy'], fmax=fmax, out_quant='a')
            frf_z = frf_z[:, 0, 1]
            if False:
                labelz = '$x$-$z$-plane'
                labely = '$x$-$y$-plane'
            else:
                labelz = '$x$-$z$-Ebene'
                labely = '$x$-$y$-Ebene'
            ax1.plot(omegas / 2 / np.pi, np.abs(frf_y), color='grey', label=labelz,)

            ax1.plot(omegas / 2 / np.pi, np.abs(frf_z), color='dimgrey', ls='dashed', label=labely)

            ax1.set_yscale('log')
            ax1.set_ylim(ymin=1e-5)
            ax1.set_xlim((0, fmax))
            ax1.xaxis._set_tick_locations([0, 0.2, 0.4, 0.6, 1.4, 1.6, 1.8, 2.0])
            ax1.legend(loc='lower right')
            ax1.set_ylabel('$|\mathcal{H}_\mathrm{f-a}| [\si{\milli\meter\per\square\second\per\\newton}]$', fontdict={'size':9})
            if False:
                ax1.set_xlabel('Frequency [\si{\hertz}]', labelpad=-9.3)

                ax1.annotate('first-order', (0.18, 0.0004) , ha='center')
                ax1.annotate('support translation', (0.34, 0.004), ha='center')
                ax1.annotate('second-order', (0.66, 0.002), ha='center')
                ax1.annotate('third-order', (1.38, 0.003), ha='center')
            else:
                ax1.set_xlabel('Frequenz [\si{\hertz}]', labelpad=-9.3)

                ax1.annotate('erste Ordnung', (0.18, 0.0004) , ha='center')
                ax1.annotate('Starrkörperschw.', (0.34, 0.0035), ha='center')
                ax1.annotate('zweite Ordnung', (0.66, 0.002), ha='center')
                ax1.annotate('dritte Ordnung', (1.38, 0.003), ha='center')

            fig.subplots_adjust(top=0.97, bottom=0.18, left=0.09, right=0.98, hspace=0.1)
            plt.savefig('figures/frf_example_struc_beam_wide.pdf')
            plt.savefig('figures/frf_example_struc_beam_wide.png')
            plt.show()

    # parameter study TMD mass with FRF
    if False:
        fmax = 3
        plt.rc('text.latex', preamble="\\usepackage{siunitx}\\usepackage{xfrac}")
        with matplotlib.rc_context(rc=print_context_dict):
            fig, [ax1, ax2, ax3, ax4] = plt.subplots(4, 1, sharex=True, sharey=True)
            zeta = 0.005
            for damp_mode, ax in enumerate([ax2, ax3, ax4, ax1]):
                damp_mode += 1
                for color, mD in [('dimgrey', 800), ('lightgrey', 1600)]:

                    mech.example_beam(num_nodes=100, num_modes=14, damping=zeta, damp_mode=damp_mode, num_meas_nodes=1, mD=mD)
                    omegas, frf = mech.frequency_response(65536, 100, 'uz', fmax=fmax, out_quant='d')
                    ax.plot(omegas / 2 / np.pi, np.abs(frf), color=color, label=f'$m_D = {mD}$')
                    omegas, frf = mech.frequency_response(65536, 100, 'uy', fmax=fmax, out_quant='d')
                    ax.plot(omegas / 2 / np.pi, np.abs(frf), color=color)

            ax1.set_yscale('log')
            ax1.legend(loc='upper right')
            fig.text(0.02, 0.5, 'Receptance $|\mathcal{H}_\mathrm{f-d}| [\si{\meter\per\\newton}]$', va='center', ha='center', rotation='vertical',)
            # ax1.set_ylabel('Compliance $|\mathcal{H}_\mathrm{f-d}| [\si{\meter\per\\newton}]$')

            fig.subplots_adjust(top=0.97, bottom=0.115, left=0.1, right=0.97, hspace=0.07, wspace=0.035)
            ax4.set_xlabel('Frequency [\si{\hertz}]')
            ax4.set_xlim((0, fmax))
            plt.show()

    # parameter study TMD tuning with FRF
    if False:
        fmax = 3
        plt.rc('text.latex', preamble="\\usepackage{siunitx}\\usepackage{xfrac}")
        with matplotlib.rc_context(rc=print_context_dict):
            fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
            zeta = 0.005
            for linestyle, damp_mode in [('solid', 1), ('dashed', 2), ('dotted', 3), ('solid', 5)]:

                mech.example_beam(num_nodes=100, num_modes=14, damping=zeta, damp_mode=damp_mode, num_meas_nodes=1, mD=800)

                # print(mech.modal(damped=True, use_cache=False))
                omegas, frf = mech.frequency_response(65536, 100, 'uy', fmax=fmax, out_quant='d')
                # print(frf, frf.shape)

                label = f'$j = {damp_mode}$'
                color = '#00000080'
                if damp_mode == 5:
                    label = 'undamped'
                    color = 'lightgrey'

                frf = frf[:, 0]  # top displacement
                # frf = frf[:,1] #tmd displacement
                # frf = frf[:,1] - frf[:,0] # relative displacement

                ax.plot(omegas / 2 / np.pi, np.abs(frf), color=color, label=label, zorder=-1 * damp_mode, ls=linestyle)
                continue
                omegas, frf = mech.frequency_response(65536, 100, 'uy', fmax=fmax, out_quant='d')
                ax.plot(omegas / 2 / np.pi, np.abs(frf), color=color)

            ax.set_yscale('log')
            ax.legend(loc='upper right')
            ax.set_ylabel('Receptance $|\mathcal{H}_\mathrm{f-d}| [\si{\meter\per\\newton}]$')
            # ax1.set_ylabel('Compliance $|\mathcal{H}_\mathrm{f-d}| [\si{\meter\per\\newton}]$')

            fig.subplots_adjust(top=0.97, bottom=0.115, left=0.1, right=0.97, hspace=0.07, wspace=0.035)
            ax.set_xlabel('Frequency [\si{\hertz}]')
            ax.set_xlim((0, 1))
            # plt.savefig('figures/example_tmd_frf.pdf')
            # plt.savefig('figures/example_tmd_frf.png')
            plt.show()

    # Modal parameters
    if False:

        mech.example_beam(num_nodes=100, num_modes=14, damping=0.005, num_meas_nodes=100)
        f, d, phi, kappas, mus, etas = mech.modal(num_modes=14, damped=True, modal_matrices=True, use_meas_nodes=False)
        for i in range(len(f)):
            # print(f'{f[i]:1.4f} & {d[i]*100:1.4f} & {np.sqrt(kappas[i]/mus[i])/2/np.pi*np.sqrt(1-d[i]**2):1.4f}& {etas[i]/2/np.sqrt(mus[i]*kappas[i]):1.4f} & \includegraphics{{figures/introduction/mshs/{i}_2}}  & \includegraphics{{figures/introduction/mshs/{i}_1}}  \\\\')
            # print(f'{f[i]:1.4f} & {d[i]*100:1.4f} & {kappas[i]:1.4f} & {mus[i]:1.1f} & {etas[i]:1.4f} & \includegraphics{{figures/introduction/mshs/{i}_2}}  & \includegraphics{{figures/introduction/mshs/{i}_1}}  \\\\')
            print(f'{f[i]:1.8f} & {d[i]*100:1.3f} & {kappas[i]:1.3f} & {mus[i]:1.1f} & {etas[i]:1.3f} ')

    # Mode shape pictograms
    if False:

        mech.example_beam(num_nodes=100, num_modes=14, damping=0.005, num_meas_nodes=100, damp_mode=2)
        f, d, phi, = mech.modal(num_modes=14, damped=True, modal_matrices=False, use_meas_nodes=False)
        x = np.array(mech.nodes_coordinates)[:, 1]
        ylim = np.abs(phi).max() * 1.1
        for i in range(len(f)):
            for j in [1, 2]:
                plt.figure(figsize=(1.5, 0.57))
                msh = np.abs(phi[:, j, i])
                msh *= np.cos(np.angle(phi[:, j, i]))

                plt.plot([160], [0], color='dimgrey', ls='none', marker='o', markerfacecolor='white', markersize=3)
                plt.plot(x[:-2], msh[:-2], color='black', lw=1)
                plt.plot([0, 200], [0, 0], color='dimgrey', lw=1, ls='dotted')
                plt.plot([-4], [0], color='dimgrey', ls='none', marker='>', markerfacecolor='white', markersize=6)
                plt.plot([0], [0], color='dimgrey', ls='none', marker='o', markerfacecolor='white', markersize=3)

                plt.plot([x[-2], x[-2]], [0, msh[x == x[-2]][0]], color='black', lw=1)
                plt.plot([x[-1], x[-1]], msh[x == x[-1]], color='black', lw=1)
                plt.plot([x[-1]], msh[-1], color='black', ls='none', marker='.')

                plt.ylim((-ylim, ylim))
                plt.xticks([])
                plt.yticks([])
                plt.subplots_adjust(top=1, bottom=0, left=0, right=1)
                plt.gca().set_axis_off()
                # plt.savefig(os.path.join('figures', f'{i}_{j}_wide.pdf'))

    # Mode shape in PlotMSH
    if False:
        from pyOMA.core.PreProcessingTools import GeometryProcessor, PreProcessSignals
        from pyOMA.core.PlotMSH import ModeShapePlot
        from pyOMA.GUI.PlotMSHGUI import start_msh_gui
        from pyOMA.core.PostProcessingTools import MergePoSER
        jid = 'test'

        mech.example_beam(num_nodes=100, num_modes=14, damping=0.005, num_meas_nodes=100, damp_mode=1)
        f, d, phi, = mech.numerical_response_parameters(dofs=[0, 1, 2])

        mech.export_geometry(f'{tempfile.gettempdir()}/{jid}/')

        merged_data = MergePoSER()
        merged_data.mean_damping = d[:, np.newaxis]
        merged_data.mean_frequencies = f[:, np.newaxis]
        merged_data.std_frequencies = np.zeros_like(f[:, np.newaxis])
        merged_data.std_damping = np.zeros_like(d[:, np.newaxis])
        merged_data.merged_mode_shapes = phi[:, np.newaxis,:]  # np.reshape(phi[:,:,:],(np.product(phi.shape[:2]),phi.shape[2]),'C')[:,np.newaxis,:]# (total_dofs, 1, common_modes)
        merged_data.merged_num_channels = merged_data.merged_mode_shapes.shape[0]

        geometry = GeometryProcessor.load_geometry(f'{tempfile.gettempdir()}/{jid}/grid.txt', f'{tempfile.gettempdir()}/{jid}/lines.txt')
        geometry.add_node('A1', (0, -70, 0))
        geometry.add_node('A2', (0, 35, 60.6))
        geometry.add_node('A3', (0, 35, -60.6))
        geometry.add_line(('A1', '80'))
        geometry.add_line(('A2', '80'))
        geometry.add_line(('A3', '80'))
        geometry.add_line(('102', '102'))
        merged_data.merged_chan_dofs = PreProcessSignals.load_chan_dofs(f'{tempfile.gettempdir()}/{jid}/chan_dofs.txt')

        mode_shape_plot = ModeShapePlot(geometry, merged_data=merged_data, beamcolor='dimgrey', scale=0.2, amplitude=50,
                                        save_ani_path='figures/mshs_ani/')

        # Don't forget to re-enable the hack in draw_lines, to draw the TMD mass on the top

        # mode_shape_plot.draw_nodes()
        mode_shape_plot.subplot.view_init(30, 60, -106)

        mode_shape_plot.draw_lines()
        mode_shape_plot.draw_master_slaves()
        mode_shape_plot.draw_chan_dofs()
        start_msh_gui(mode_shape_plot)

    plt.show()


def main():
    # default_mapping(alpha=0)
    # default_mapping(alpha=45)
    # default_mapping(alpha=90)
    # default_mapping(alpha=135)
    # default_mapping(alpha=21)

    # test_domain()
    plots()


if __name__ == '__main__':
    # test_interpolation('frf', {'frequencies':127,'space':1}, 3062)
    # export_datamanager()
    main()
