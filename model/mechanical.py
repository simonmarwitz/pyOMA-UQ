# from importlib import reload; from model import mechanical; mechanical.main()
# reload(mechanical); mechanical.main()

import sys
import time
import numpy as np
import os
import glob
import shutil

import logging
from asyncio.log import logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# global logger
# logger = logger.getLogger('')
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def _get_pyansys():
    try:
        import pyansys
        return pyansys
    except ImportError as exc:
        raise ImportError(
            "pyansys is required for FEM-based mapping. "
            "Install it manually (not on PyPI); see the oma_uq[fem] extra."
        ) from exc


# import scipy.stats
# import scipy.optimize
# import scipy.signal
import scipy.io
import scipy.integrate

import warnings
warnings.filterwarnings("ignore", message="Use of `point_arrays` is deprecated. Use `point_data` instead.")
warnings.filterwarnings("ignore", message="Use of `cell_arrays` is deprecated. Use `cell_data` instead.")

import functools
from contextlib import contextmanager


@contextmanager
def supress_logging(mapdl):
    """Contextmanager to supress logging for a MAPDL instance"""
    prior_log_level = mapdl._log.level
    if prior_log_level != 'CRITICAL':
        mapdl._set_log_level('CRITICAL')
    try:
        yield
    finally:
        if prior_log_level != 'CRITICAL':
            mapdl._set_log_level(prior_log_level)


def session_restore(func):

    @functools.wraps(func)
    def wrapper_session_restore(*args, **kwargs):
        mech = args[0]
        ansys = mech.ansys
        routine_dict = {0.0:ansys.finish,
                        17.0:ansys.prep7,
                        21.0:ansys.slashsolu,
                        31.0:ansys.post1,
                        36.0:ansys.post26,
                        52.0:ansys.aux2 ,
                        53.0:ansys.aux3,
                        62.0:ansys.aux12,
                        65.0:ansys.aux15}
        ansys.get(par='ROUT', entity='ACTIVE', entnum='0', item1='ROUT')
        # ansys.load_parameters()
        current_rout = ansys.parameters['ROUT']
        if current_rout != 0.0:
            ansys.finish()
        value = func(*args, **kwargs)
        if current_rout != 0.0:
            ansys.finish()
            routine_dict[current_rout]()
        return value

    return wrapper_session_restore


class MechanicalDummy(object):

    def __init__(self, jobname):
        self.jobname = jobname

#             build_mdof, free_decay, ambient, impulse_response, modal, modal_comp  , IRF matrix, build_conti, frequency_response
        self.state = [False, False, False, False, False, False, False, False, False]

        self.nonlinear_elements = []
        self.voigt_kelvin_elements = []
        self.coulomb_elements = []
        self.mass_elements = []
        self.beam_elements = []

        # initialize class variables
        # build_mdof
        self.nodes_coordinates = None
        # self.num_nodes = None
        self.num_modes = None
        self.d_vals = None
        self.k_vals = None
        self.masses = None
        self.eps_duff_vals = None
        self.sl_force_vals = None
        self.hyst_vals = None
        self.meas_nodes = None
        self.damping = None

        # free_decay
        self.decay_mode = None
        self.t_vals_decay = None
        self.resp_hist_decay = None

        # ambient
        self.inp_hist_amb = None
        self.t_vals_amb = None
        self.resp_hist_amb = None

        # impulse_response
        self.inp_hist_imp = None
        self.t_vals_imp = None
        self.resp_hist_imp = None
        self.modal_imp_energies = None
        self.modal_imp_amplitudes = None

        # IRF matrix
        self.t_vals_imp = None
        self.IRF_matrix = None
        self.imp_hist_imp_matrix = None
        self.modal_imp_energy_matrix = None
        self.modal_imp_amplitude_matrix = None

        # FRF
        self.omegas = None
        self.frf = None

        # modal
        self.damped_frequencies = None
        self.modal_damping = None
        self.damped_mode_shapes = None
        self.frequencies = None
        self.mode_shapes = None
        self.num_modes = None
        # self.kappas = None
        # self.mus = None
        # self.etas = None
        self.gen_mod_coeff = None

        # signal_parameters
        self.deltat = None
        self.timesteps = None

        # transient_parameters
        self.trans_params = None

    @property
    def num_nodes(self):
        return len(self.nodes_coordinates)

    def build_mdof(self, nodes_coordinates=[(1, 0, 0, 0), ],
                   k_vals=[1], masses=[1], d_vals=None, damping=None,
                   sl_force_vals=None, eps_duff_vals=None, hyst_vals=None,
                   num_modes=None, meas_nodes=None, **kwargs):

        logger.debug('self.build_mdof')
        # if m modes are of interest number of nodes n > 10 m

        num_nodes = len(nodes_coordinates)

        # Nodes
        ntn_conns = [[] for i in range(num_nodes - 1)]
        for i, (node, x, y, z) in enumerate(nodes_coordinates):
            if i < num_nodes - 1:
                ntn_conns[i] = [node]
            if i > 0:
                ntn_conns[i - 1].append(node)

        self.nodes_coordinates = nodes_coordinates
        self.ntn_conns = ntn_conns
        self.num_nodes = num_nodes
        self.num_modes = num_modes
        self.d_vals = d_vals
        self.k_vals = k_vals
        self.masses = masses
        self.eps_duff_vals = eps_duff_vals
        self.sl_force_vals = sl_force_vals
        self.hyst_vals = hyst_vals

        self.damping = damping
        self.meas_nodes = np.array(meas_nodes)

        self.state[0] = True
        for i in range(1, len(self.state)):
            self.state[i] = False

    def build_conti(self, struct_parms, Ldiv, damping=None, num_modes=None, meas_locs=None):

        if num_modes is None:
            num_modes = max(2, int(np.floor(Ldiv / 10) - 1))  # choose at least 1 mode
            logger.info(f'Choosing num_modes as {num_modes} based on the number of nodes {Ldiv}')
        assert num_modes <= Ldiv - 1
        if num_modes > Ldiv / 10:
            logger.warning(f'The number of modes {num_modes} should be less/equal than 0.1 x number of nodes (= {Ldiv}).')

        assert Ldiv >= 3

        L = struct_parms['L']

        x_nodes = np.linspace(0, L, Ldiv)

        x_knl = struct_parms['x_knl']
        x_nodes[np.argmin(np.abs(x_nodes - x_knl))] = x_knl

        x_tmd = struct_parms['x_tmd']
        x_nodes[np.argmin(np.abs(x_nodes - x_tmd))] = x_tmd

        nodes_coordinates = []
        for i, x in enumerate(x_nodes):
            nodes_coordinates.append((i + 1, x, 0, 0))
        nodes_coordinates.append((i + 2, x_knl, 0, 0))
        nodes_coordinates.append((i + 3, x_tmd, 0, 0))
        nodes_coordinates = np.array(nodes_coordinates)

        self.struct_parms = struct_parms
        # self.num_nodes = len(nodes_coordinates)
        self.nodes_coordinates = nodes_coordinates
        self.num_modes = num_modes

        self.damping = damping
        if damping is not None:
            self.damped = True
            self.globdamp = (struct_parms['dy_tmd'] > 0 or struct_parms['dz_tmd'] > 0)

        for i in range(len(self.state)):
            self.state[i] = False
        self.state[7] = True

    def transient_ifrf(self, fy=None, fz=None,
                       inp_nodes=None,
                       inp_dt=None, out_dt=None,
                       out_quant=['d', 'v', 'a'], **kwargs):
        '''
        Compute the vibration response of the system self to arbitrary forcing in 
        y and z direction in the frequency domain. The compliance FRF is computed 
        for as many timesteps / frequency lines as are in the force vectors up to 
        a frequency of half the sampling frequency 1/inp_dt. The sampling resolution
        may be increased by zero-padding the response prior to inverse FFT. The 
        computationally most expensive part is the assembly of the FRF, depending
        on the number of input and output nodes. Once computed, the FRF will be re-
        used if no parameters have changed, so the response to different force
        time histories can be evaluated quickly. Note, that in comparison
        with direct time stepping methods, Fourier transform methods suffer from 
        wrap-around effects but reach steady-state instantly.
        
        Parameters:
        -----------
            fy, fz: np.ndarray (N, n_inp_nodes)
                Lateral force in y / z direction in time domain for all nodes in 
                "inp_nodes". A FFT is performed prior to further computations and 
                the number of output timesteps is N_out = 2 * (N // 2) * inp_dt // out_dt. 
            inp_nodes: np.ndarray((n_imp_nodes,), int), optional
                The numbers of input nodes corresponding to the columns of fy / fz.
                Defaults to all nodes (self.nodes_coordinates[:,0])
            inp_dt: float
                Sample spacing (inverse of the sampling rate) of the input force
                time histories.
            out_dt: float
                Sample spacing (inverse of the sampling rate) of the output 
                response time histories. 
            out_quant: list of ['d','v','a']
                Whether to compute displacements, velocities and/or accelerations.
        
        Other Parameters:
        -----------------
            **kwargs:
                All other keyword arguments are passed on to self.frequency_response_non_classical 
        
        Returns:
        --------
            time_values: ndarray
                Array holding the time instants of the computed responses.
            [d, v, a]: list-of-ndarray(N_out, num_out_nodes, 2
                Response time histories (displacement, velocity, acceleration).
                Note, that the ordering of out the output follows the order in 
                self.dof_ref_out.
        '''

        omegas = self.omegas
        frf = self.frf
        print(f'The pre-computed FRF array is of type {type(frf)}')

        if omegas is None or frf is None:
            raise NotImplementedError("The FRF must be pre-computed in order to call it within MechanicalDummy class.")

        meas_nodes = self.meas_nodes

        assert inp_dt is not None

        if out_dt is None:
            out_dt = inp_dt
        assert out_dt <= inp_dt

        if inp_nodes is None:
            inp_nodes = self.nodes_coordinates[:, 0]
        num_nodes = len(inp_nodes)

        inp_dofs = []
        out_dofs = kwargs.get('out_dofs', inp_dofs)

        if fy is not None:
            fy = np.fft.rfft(fy, axis=0)
            assert num_nodes == fy.shape[1]
            N = 2 * (fy.shape[0] - 1)
            inp_dofs.append('uy')
        if fz is not None:
            fz = np.fft.rfft(fz, axis=0)
            assert num_nodes == fz.shape[1]
            N = 2 * (fz.shape[0] - 1)
            inp_dofs.append('uz')

        load_frf = (omegas is not None and omegas[-1] == 1 / 2 / inp_dt * 2 * np.pi)
        load_frf = (load_frf and frf is not None and np.all(frf.shape == (N // 2 + 1, len(inp_nodes) * len(inp_dofs), len(self.meas_nodes) * len(out_dofs))))

        if not load_frf:
            cause = ''
            if omegas[-1] != 1 / 2 / inp_dt * 2 * np.pi:
                cause += 'frequency resolution, '
            if frf.shape[0] != N // 2 + 1:
                cause += 'number of frequency lines, '
            if frf.shape[1] != len(inp_nodes) * len(inp_dofs):
                cause += 'number of input channels, '
            if frf.shape[2] != len(self.meas_nodes) * len(out_dofs):
                cause += 'number of output channels'
            raise RuntimeError(f"Pre-computed FRF is not compatible with given parameters for: {cause}. Refer to Mechanical.transient_frf()")

        t_end = N * inp_dt
        N_out = int(t_end / out_dt)

        F_freq = np.hstack([fy, fz])  # n_lines, n_inp_nodes*n_inp_dofs

        d_freq = np.empty((N // 2 + 1, 2 * len(meas_nodes)), dtype=complex)
        for i in range(N // 2 + 1):
            np.dot(F_freq[i,:], frf[i,:,:], out=d_freq[i,:])
        d_freq = d_freq.reshape((N // 2 + 1, len(meas_nodes), 2), order='F')

        if 'd' in out_quant:
            d_freq_time = np.fft.irfft(d_freq, n=N_out, axis=0)
        else:
            d_freq_time = None

        if 'v' in out_quant:
            v_freq_time = np.fft.irfft(d_freq * 1j * omegas[:, np.newaxis, np.newaxis], n=N_out, axis=0)
        else:
            v_freq_time = None

        if 'a' in out_quant:
            a_freq_time = np.fft.irfft(d_freq * -1 * omegas[:, np.newaxis, np.newaxis] ** 2, n=N_out, axis=0)
        else:
            a_freq_time = None

        time_values = np.linspace(inp_dt, N_out * out_dt, N_out)  #  ansys also starts at inp_dt

        self.time_values = time_values
        self.d_freq = d_freq
        self.d_freq_time = d_freq_time
        self.v_freq_time = v_freq_time
        self.a_freq_time = a_freq_time

        return time_values, [d_freq_time, v_freq_time, a_freq_time]

    def modal(self, damped=True, num_modes=None, use_cache=True, modal_matrices=False):  # Modal Analysis

        num_nodes = self.num_nodes
        if num_modes is None:
            num_modes = self.num_modes
        assert num_modes <= num_nodes
        if num_modes > 10 * num_nodes:
            logger.warning(f'The number of modes {num_modes} should be greater/equal than 10 number of nodes {num_nodes}.')

        # cached modal analysis results
        # TODO: the logic needs improvement: num_modes may have been different for both types of analyses
        if use_cache:
            if damped and num_modes == self.num_modes:
                frequencies = self.damped_frequencies
                damping = self.modal_damping
                mode_shapes = self.damped_mode_shapes
            elif not damped and num_modes == self.num_modes:
                frequencies = self.frequencies
                mode_shapes = self.mode_shapes
                damping = np.zeros_like(frequencies)
            else:
                frequencies = None
                damping = None
                mode_shapes = None

            gen_mod_coeff = self.gen_mod_coeff
            if modal_matrices and gen_mod_coeff is not None and frequencies is not None:
                return frequencies, damping, mode_shapes, gen_mod_coeff
            elif not modal_matrices and frequencies is not None:
                return frequencies, damping, mode_shapes
        else:
            return

    def get_geometry(self):
        '''
        return (meas)nodes, lines, chan_dofs in a format usable in pyOMA
        '''
        nodes = []
        for meas_node in np.concatenate(([1], self.meas_nodes)):
            for node, x, y, z in self.nodes_coordinates:
                if node == meas_node:
                    nodes.append([meas_node, x, y, z])
                    break
            else:
                logger.warning(f'Meas node {meas_node} was not found in nodes_coordinates')

        lines = []
        meas_node_last = 1  # ansys starts at 1
        for meas_node in self.meas_nodes:
            # how is that node connected to any other node in self.meas_nodes
            # for all_occurences_of_it_in_ntn_conns:
            #     for connect_level in range(num_nodes):
            #         find all nodes connected to it in ntn_conns
            #         check, if any of them are in meas_nodes
            #            store and remove from
            #            pah, that sucks
            lines.append((meas_node_last, meas_node))
            meas_node_last = meas_node

        chan_dofs = []
        channel = 0
        # for channel in range(3):
        #     chan_dofs.append((channel, 1, 0, 0))
        for az, elev in [(0, 0), (270, 0), (0, 90)]:
            for meas_node in self.meas_nodes:
                chan_dofs.append((channel, meas_node, az, elev))
                channel += 1

        return nodes, lines, chan_dofs

    def export_geometry(self, save_dir=None):
        'save under jid_folder, nodes_file, lines_file, chan_dofs_file'
        if save_dir is None:
            import pathlib
            save_dir = str(pathlib.Path.cwd() / 'polyuq_results' / 'datasets')
        os.makedirs(save_dir, exist_ok=True)

        nodes, lines, chan_dofs = self.get_geometry()

        with open(os.path.join(save_dir, 'grid.txt'), 'wt') as f:
            f.write('node_name\tx\ty\tz\n')
            for node, x, y, z in nodes:
                f.write(f'{node}\t{x:e}\t{y:e}\t{z:e}\n')

        with open(os.path.join(save_dir, 'lines.txt'), 'wt') as f:
            f.write('node_name_1\tnode_name_2\n')
            for line_s, line_e in lines:
                f.write(f'{line_s}\t{line_e}\n')

        with open(os.path.join(save_dir, 'chan_dofs.txt'), 'wt') as f:
            f.write('Channel-Nr.\tNode\tAzimuth\tElevation\tChannel Name\n')
            for channel, meas_node, az, elev in chan_dofs:
                f.write(f'{channel}\t{meas_node}\t{az}\t{elev}\t \n')

        return

    def save(self, fpath, emergency_arrays=None):
        '''
        save under save_dir/{jobname}_mechanical.npz
        
        save enables to:
         - restore the ansys model
         - rerun analysis
         - retrieve previously run analyses
        '''
        fdir, file = os.path.split(fpath)
        fname, ext = os.path.splitext(file)

        logger.info(f'Saving Mechanical object to {fpath}')

        out_dict = {}
        if emergency_arrays is not None:
            out_dict.update(emergency_arrays)
        # 0:build_mdof, 1:free_decay, 2:ambient, 3:impulse_response, 4:modal

        out_dict['self.state'] = self.state

        out_dict['self.jobname'] = self.jobname

        if self.state[0]:
            out_dict['self.nodes_coordinates'] = self.nodes_coordinates
            # out_dict['self.num_nodes'] = self.num_nodes
            out_dict['self.num_modes'] = self.num_modes
            out_dict['self.d_vals'] = self.d_vals
            out_dict['self.k_vals'] = self.k_vals
            out_dict['self.masses'] = self.masses
            out_dict['self.eps_duff_vals'] = self.eps_duff_vals
            out_dict['self.sl_force_vals'] = self.sl_force_vals
            out_dict['self.hyst_vals'] = self.hyst_vals
            out_dict['self.meas_nodes'] = self.meas_nodes
            out_dict['self.damping'] = self.damping

        if self.state[1]:
            out_dict['self.decay_mode'] = self.decay_mode
            out_dict['self.t_vals_decay'] = self.t_vals_decay
            out_dict['self.resp_hist_decayd'] = self.resp_hist_decay[0]
            out_dict['self.resp_hist_decayv'] = self.resp_hist_decay[1]
            out_dict['self.resp_hist_decaya'] = self.resp_hist_decay[2]

        if self.state[2]:
            out_dict['self.inp_hist_amb'] = self.inp_hist_amb
            out_dict['self.t_vals_amb'] = self.t_vals_amb
            out_dict['self.resp_hist_ambd'] = self.resp_hist_amb[0]
            out_dict['self.resp_hist_ambv'] = self.resp_hist_amb[1]
            out_dict['self.resp_hist_amba'] = self.resp_hist_amb[2]

        if self.state[3]:
            out_dict['self.inp_hist_imp'] = self.inp_hist_imp
            out_dict['self.t_vals_imp'] = self.t_vals_imp
            out_dict['self.resp_hist_impd'] = self.resp_hist_imp[0]
            out_dict['self.resp_hist_impv'] = self.resp_hist_imp[1]
            out_dict['self.resp_hist_impa'] = self.resp_hist_imp[2]
            out_dict['self.modal_imp_energies'] = self.modal_imp_energies
            out_dict['self.modal_imp_amplitudes'] = self.modal_imp_amplitudes

        if self.state[4]:
            out_dict['self.damped_frequencies'] = self.damped_frequencies
            out_dict['self.modal_damping'] = self.modal_damping
            out_dict['self.damped_mode_shapes'] = self.damped_mode_shapes
            out_dict['self.frequencies'] = self.frequencies
            out_dict['self.mode_shapes'] = self.mode_shapes
            out_dict['self.num_modes'] = self.num_modes
            # out_dict['self.kappas'] = self.kappas
            # out_dict['self.mus'] = self.mus
            # out_dict['self.etas'] = self.etas
            out_dict['self.gen_mod_coeff'] = self.gen_mod_coeff

        if self.state[5]:
            out_dict['self.frequencies_comp'] = self.frequencies_comp
            out_dict['self.modal_damping_comp'] = self.modal_damping_comp
            out_dict['self.mode_shapes_comp'] = self.mode_shapes_comp

        if self.state[2] or self.state[3] or self.state[4] or self.state[6]:
            out_dict['self.trans_params'] = self.trans_params

            out_dict['self.deltat'] = self.deltat
            out_dict['self.timesteps'] = self.timesteps

        if self.state[6]:
            out_dict['self.t_vals_imp'] = self.t_vals_imp
            out_dict['self.IRF_matrix'] = self.IRF_matrix
            out_dict['self.imp_hist_imp_matrix'] = self.imp_hist_imp_matrix
            out_dict['self.modal_imp_energy_matrix'] = self.modal_imp_energy_matrix
            out_dict['self.modal_imp_amplitude_matrix'] = self.modal_imp_amplitude_matrix

        if self.state[7]:
            out_dict['self.struct_parms'] = self.struct_parms
            # out_dict['self.num_nodes'] = self.num_nodes
            out_dict['self.nodes_coordinates'] = self.nodes_coordinates
            out_dict['self.damping'] = self.damping
            out_dict['self.alpha'] = self.alpha
            out_dict['self.beta'] = self.beta
            out_dict['self.damped'] = self.damped
            out_dict['self.globdamp'] = self.globdamp
            out_dict['self.meas_nodes'] = self.meas_nodes

        if self.state[8]:
            out_dict['self.omegas'] = self.omegas
            if not isinstance(self.frf, np.memmap):
                frf = np.memmap(os.path.join(fdir, fname + '_frf.dat'),
                                dtype=np.complex64,
                                mode='w+', shape=self.frf.shape)
                frf[:] = self.frf
                frf.flush()
            else:
                frf = self.frf
            out_dict['self.frf'] = frf.filename
            out_dict['self.dof_ref_out'] = self.dof_ref_out
            out_dict['self.dof_ref_inp'] = self.dof_ref_inp

        np.savez_compressed(fpath, **out_dict)

    @classmethod
    def load(cls, fpath):
        assert os.path.exists(fpath)

        logger.info('Now loading previous results from  {}'.format(fpath))

        in_dict = np.load(fpath, allow_pickle=True)

        jobname = in_dict['self.jobname'].item()

        mech = cls(jobname=jobname)

        return mech._load(fpath, mech)

    def _load(self, fpath, mech):

        def validate_array(arr):
            '''
            Determine whether the argument has a numeric datatype and if
            not convert the argument to a scalar object or a list.
        
            Booleans, unsigned integers, signed integers, floats and complex
            numbers are the kinds of numeric datatype.
        
            Parameters
            ----------
            array : array-like
                The array to check.
            
            '''
            _NUMERIC_KINDS = set('buifc')
            if not arr.shape:
                return arr.item()
            elif arr.dtype.kind in _NUMERIC_KINDS:
                return arr
            else:
                return list(arr)

        fdir, file = os.path.split(fpath)
        fname, ext = os.path.splitext(file)

        in_dict = np.load(fpath, allow_pickle=True)
        state = list(in_dict['self.state'])

        while len(state) < 9:
            state.append(False)

        if state[0]:
            nodes_coordinates = in_dict['self.nodes_coordinates']
            # print(nodes_coordinates, 'should be list of lists')
            # num_nodes = in_dict['self.num_nodes']
            num_modes = in_dict['self.num_modes'].item()
            d_vals = list(in_dict['self.d_vals'])
            k_vals = list(in_dict['self.k_vals'])
            masses = list(in_dict['self.masses'])
            eps_duff_vals = list(in_dict['self.eps_duff_vals'])
            sl_force_vals = list(in_dict['self.sl_force_vals'])
            hyst_vals = list(in_dict['self.hyst_vals'])
            meas_nodes = list(in_dict['self.meas_nodes'])
            # print(in_dict['self.damping'], type(in_dict['self.damping']))
            damping = in_dict['self.damping']
            if damping.size == 1:
                damping = damping.item()
            else:
                damping = list(damping)
                if len(damping) >= 2:
                    if damping[-1] == 1 or damping[-1] == 0:
                        damping[-1] = bool(damping[-1])
            # print(damping)
            # print(meas_nodes)
            mech.build_mdof(nodes_coordinates=nodes_coordinates,
                            k_vals=k_vals, masses=masses, d_vals=d_vals,
                            damping=damping, sl_force_vals=sl_force_vals,
                            eps_duff_vals=eps_duff_vals, hyst_vals=hyst_vals,
                            num_modes=num_modes, meas_nodes=meas_nodes)

        if state[1]:
            mech.decay_mode = in_dict['self.decay_mode'].item()
            mech.t_vals_decay = in_dict['self.t_vals_decay']
            mech.resp_hist_decay = [None, None, None]
            for i, key in enumerate(['self.resp_hist_decayd',
                                     'self.resp_hist_decayv',
                                     'self.resp_hist_decaya']):
                arr = in_dict[key]
                if not arr.shape: arr = arr.item()
                mech.resp_hist_decay[i] = arr

        if state[2]:
            mech.inp_hist_amb = in_dict['self.inp_hist_amb']
            mech.t_vals_amb = in_dict['self.t_vals_amb']
            mech.resp_hist_amb = [None, None, None]
            for i, key in enumerate(['self.resp_hist_ambd',
                                     'self.resp_hist_ambv',
                                     'self.resp_hist_amba']):
                arr = in_dict[key]
                if not arr.shape: arr = arr.item()
                mech.resp_hist_amb[i] = arr

        if state[3]:
            mech.inp_hist_imp = in_dict['self.inp_hist_imp']
            mech.t_vals_imp = in_dict['self.t_vals_imp']
            mech.resp_hist_imp = [None, None, None]
            for i, key in enumerate(['self.resp_hist_impd',
                                     'self.resp_hist_impv',
                                     'self.resp_hist_impa']):
                arr = in_dict[key]
                if not arr.shape: arr = arr.item()
                mech.resp_hist_imp[i] = arr
            mech.modal_imp_energies = in_dict['self.modal_imp_energies']
            mech.modal_imp_amplitudes = in_dict['self.modal_imp_amplitudes']

        if state[4]:
            mech.damped_frequencies = in_dict['self.damped_frequencies']
            mech.modal_damping = in_dict['self.modal_damping']
            mech.damped_mode_shapes = in_dict['self.damped_mode_shapes']
            mech.frequencies = in_dict['self.frequencies']
            mech.mode_shapes = in_dict['self.mode_shapes']
            mech.num_modes = in_dict['self.num_modes']
            # mech.kappas = validate_array(in_dict['self.kappas'])
            # mech.mus = validate_array(in_dict['self.mus'])
            # mech.etas = validate_array(in_dict['self.etas'])
            mech.gen_mod_coeff = validate_array(in_dict.get('self.gen_mod_coeff', mech.gen_mod_coeff))

        if state[5]:
            mech.frequencies_comp = in_dict['self.frequencies_comp']
            mech.modal_damping_comp = in_dict['self.modal_damping_comp']
            mech.mode_shapes_comp = in_dict['self.mode_shapes_comp']

        if state[2] or state[3] or state[4] or state[6]:
            trans_params = in_dict['self.trans_params']
            if trans_params.size > 1:
                mech.trans_params = tuple(trans_params)

            mech.deltat = in_dict['self.deltat'].item()
            mech.timesteps = in_dict['self.timesteps'].item()

        if state[6]:
            mech.t_vals_imp = in_dict['self.t_vals_imp']
            mech.IRF_matrix = in_dict['self.IRF_matrix']
            mech.imp_hist_imp_matrix = in_dict['self.imp_hist_imp_matrix']
            mech.modal_imp_energy_matrix = in_dict['self.modal_imp_energy_matrix']
            mech.modal_imp_amplitude_matrix = in_dict['self.modal_imp_amplitude_matrix']

        if state[7]:
            mech.struct_parms = validate_array(in_dict['self.struct_parms'])
            # mech.num_nodes        = validate_array(in_dict['self.num_nodes'])
            mech.nodes_coordinates = validate_array(in_dict['self.nodes_coordinates'])
            mech.damping = validate_array(in_dict['self.damping'])
            mech.alpha = validate_array(in_dict['self.alpha'])
            mech.beta = validate_array(in_dict['self.beta'])
            mech.damped = validate_array(in_dict['self.damped'])
            mech.globdamp = validate_array(in_dict['self.globdamp'])
            mech.meas_nodes = validate_array(in_dict['self.meas_nodes'])

        if state[8]:
            mech.omegas = in_dict['self.omegas']
            mech.dof_ref_out = in_dict['self.dof_ref_out']
            mech.dof_ref_inp = in_dict['self.dof_ref_inp']
            frf = in_dict['self.frf']
            if not frf.shape:
                # it is a memory map and this parameter is the filename
                size = (mech.omegas.shape[0], mech.dof_ref_inp.shape[0], mech.dof_ref_out.shape[0])
                frfpath = frf.item()
                if not os.path.exists(frfpath):
                    frfpath = os.path.join(fdir, fname + '_frf.dat')
                if not os.path.exists(frfpath):
                    logger.warn(f"FRF memorymap could neither be found in {frf.item()} nor in {frfpath}.")
                    mech.frf = None
                else:
                    if os.access(frfpath, os.W_OK):
                        mode = 'r+'
                    else:
                        mode = 'r'
                    mech.frf = np.memmap(frfpath, dtype=np.complex64, mode=mode, shape=size)
            else:
                mech.frf = frf

        mech.state = state

        return mech


class Mechanical(MechanicalDummy):
    # TODO: Update to newest pyANSYS release

    def __init__(self, ansys=None, jobname=None, wdir=None):
        pyansys = _get_pyansys()

        if wdir is not None:
            if not os.path.isdir(wdir):
                os.makedirs(wdir)

        if ansys is None:
            ansys = self.start_ansys(wdir, jobname)
        assert isinstance(ansys, pyansys.mapdl._MapdlCore)

        try:
            path = os.path.join(ansys.directory, ansys.jobname)
            ansys.finish()
            ansys.clear()
            # for file in glob.glob(f'{path}.*'):
            #     os.remove(file)
            # if os.path.exists(f'{path}/'):
            #     for file in glob.glob(f'{path}/*'):
            #         os.remove(file)
            #     os.rmdir(f'{path}/')
        except (pyansys.errors.MapdlExitedError, NameError) as e:
            logger.exception(e)
            ansys = self.start_ansys(wdir, jobname)

        self.ansys = ansys

        if wdir is not None:
            logger.info(f'Switching working directory:\t {wdir}')
            ansys.cwd(wdir)
        if jobname is not None:
            logger.info(f'Switching job:\t {jobname}')
            ansys.filname(fname=jobname, key=1)
            # self.jobname = jobname
        else:
            logger.info(f'Current job:\t {ansys.jobname}')
            jobname = ansys.jobname

        ansys.config(lab='NOELDB', value=1)
#         ansys.config(lab='NORSTGM',value=1)
        # ansys.output(fname='null',loc='/dev/')
        # ansys.nopr()  # Suppresses the expanded interpreted input data listing. Leads to failures at least in modal
        ansys.nolist()  # Suppresses the data input listing.
        ansys.finish()

        super().__init__(jobname)

    @staticmethod
    def start_ansys(working_dir=None, jid=None,):
        pyansys = _get_pyansys()

        # global ansys
        # try:
            # ansys.finish()
        # except (pyansys.errors.MapdlExitedError, NameError) as e:
            # logger.warning(repr(e))
        if working_dir is None:
            working_dir = os.getcwd()
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)
        os.chdir(working_dir)
        now = time.time()
        if jid is None:
            jid = 'file'
        # avoid too many simultaneous connections to license server
        for i in range(10):
            try:

                ansys = pyansys.launch_mapdl(
                    # exec_file=os.environ.get('ANSYS_EXEC_FILE'),
                    exec_file='/usr/app-soft1/ansys/v201/ansys/bin/ansys201',
                    run_location=working_dir, override=True, loglevel='ERROR',
                    nproc=1, log_apdl=False,
                    log_broadcast=False, jobname=jid,
                    mode='console', additional_switches='-smp -p ansys')
                logger.warning(f'(Re)started ANSYS in {time.time()-now:1.2f} s.')
                break
            except Exception as e:
                logger.warning(f'Starting up ansys failed on {i}. try')
                time.sleep(np.random.random_sample())
        else:
            if "LSB_JOBID" in os.environ:
                lsb_jobid = os.environ["LSB_JOBID"]
                logger.warning(f'Starting ANSYS failed 10 consecutive times; Killing job {lsb_jobid}')
                os.popen(f'bkill {lsb_jobid}')
                raise e

        # ansys.clear()

        return ansys

    @session_restore
    def nonlinear(self, nl_ity=1, d_max=1.5, k_lin=1e5, k_nl=None, linear_part=False):
        '''
        generate a nonlinear spring
        with a linear part and a cubic nonlinear part
        nl_ity: the fraction of linear and nonlinear parts
            for nl_ity = (-0.5,0] it is a nonlinear softening spring,degressive, underlinear
            for nl_ity = 0 it is a linear spring
            for nl_ity = [0,0.5) it is a nonlinear hardening spring,orogressive, overlinear
        d_max: maximal displacement, up to which the force-displacement curve will be defined. ansys will interpolate further points and (probably) issue a warning.
        k_lin: linear stiffness
        k_nl: nonlinear stiffness, if not automatically determined: k_lin and k_nl should match at d_max
        '''
        if k_nl is not None:
            logger.warning('Are you sure, you want to set k_nl manually. Make sure to not overshoot d_max')
        if nl_ity < -0.5 or nl_ity > 0.5:
            raise RuntimeError('The fraction of nonlinearity should not exceed/subceed 0.5/-0.5. It currently is set to {}.'.format(nl_ity))

        ansys = self.ansys
        ansys.prep7()

        ansys.run('nset=rlinqr(0,14)')
        ansys.run('itype=etyiqr(0,14)')
        # ansys.load_parameters()

        if k_nl is None:
            k_at_d_max = k_lin / d_max ** 2
            k_nl = k_at_d_max
        ansys.et(itype=ansys.parameters['ITYPE'] + 1, ename='COMBIN39', kop3='3', inopr=1)
        d = np.linspace(0, d_max, 20, endpoint=True)

        if linear_part:
            F = k_lin * d * (1 - nl_ity) + k_nl * d ** 3 * (nl_ity)
        else:
            F = k_nl * d ** 3 * (nl_ity)
#         plot.plot(d,F)
#         plot.show()
        d = list(d)
        F = list(F)
        command = 'R, {}'.format(ansys.parameters['NSET'] + 1)
        i = 0
        while len(d) >= 1:
            command += ''.join(', {}, {}'.format(d.pop(0), F.pop(0)))
            i += 1
            if i == 3:
                ansys.run(command)
                # print(command)
                command = 'RMORE'
                i = 0
        else:
            ansys.run(command)
            # print(command)

        ansys.run('nset=rlinqr(0,14)')
        ansys.run('itype=etyiqr(0,14)')
        # ansys.load_parameters()

        ansys.finish()
        self.nonlinear_elements.append((ansys.parameters['ITYPE'], ansys.parameters['NSET']))
        return ansys.parameters['ITYPE'], ansys.parameters['NSET']  # itype, nset

        # ansys.r(nset=1, r1=d.pop(0), r2=F.pop(0), r3=d.pop(0), r4=F.pop(0), r5=d.pop(0), r6=F.pop(0))

    @session_restore
    def voigt_kelvin(self, k="", d="", dof='UX', **kwargs):
        # print(k,d)
        ansys = self.ansys

        ansys.prep7()

        ansys.run('nset=rlinqr(0,14)')
        ansys.run('itype=etyiqr(0,14)')
        # ansys.load_parameters()

        ansys.et(itype=ansys.parameters['ITYPE'] + 1, ename='COMBIN14', inopr=1, kop2=['', 'UX', 'UY', 'UZ', 'ROTX', 'ROTY', 'ROTZ'].index(dof))
        #              k,    cv1, cv2
        ansys.r(nset=ansys.parameters['NSET'] + 1, r1=k, r2=d)

#         Omega = 1
#         s=1
#         Wd = d*np.pi*Omega*s**2 # petersen eq. 555
#         print(f"Damping energy at unit displacement and unit circular frequency: {Wd}")

        ansys.run('nset=rlinqr(0,14)')
        ansys.run('itype=etyiqr(0,14)')
        # ansys.load_parameters()

        ansys.finish()
        self.voigt_kelvin_elements.append((ansys.parameters['ITYPE'], ansys.parameters['NSET']))
        return ansys.parameters['ITYPE'], ansys.parameters['NSET']  # itype, nset

    @session_restore
    def coulomb(self, k_1=90000, d=0, f_sl=15000, k_2=10000 , **kwargs):
        '''
        k_1 and slider are in series
        k_1 is the sticking stiffness, and defines the displacement required
        to reach the force, at which the slider breaks loose should be k_2*10


        f_sl determines the constant friction damping force that is applied each half-cycle
            a lower f_sl means less damping, longer decay
            a higher f_sl means higher damping, shorter decay
            should be d_0/k_2

        high f_sl or low k_1 have the same effect

        k_2 is the direct stiffness and in parallel with k_1+slider and
            determines the oscillation frequency if a mass is attached to the element
            apdl modal solver uses k_tot = k_1 + k_2


        '''
        ansys = self.ansys
        ansys.prep7()

        k_tot = kwargs.pop('k_tot', k_2 + k_1)
        k_2 = k_tot - k_1
        if k_tot:
            f_sl_in = f_sl * k_1 / k_tot
        else:
            f_sl_in = f_sl

#         print('Displacement amplitude above which friction force becomes active and below which no more dissipation happens {}'.format(f_sl/k_1))
#         s=1
#         Wd = 4*f_sl_in*(s-f_sl_in/k_1) # Petersen eq 637
#         print(f"Damping energy at {s} displacement: {Wd}")

        ansys.run('nset=rlinqr(0,14)')
        ansys.run('itype=etyiqr(0,14)')
        # ansys.load_parameters()

        ansys.et(itype=ansys.parameters['ITYPE'] + 1, ename='COMBIN40', kop3='3', inopr=1)
        #               K1,     C,     M,    GAP,  FSLIDE, K2
        # print(k_1, d,f_sl_in, k_2)
        ansys.r(nset=ansys.parameters['NSET'] + 1, r1=k_1, r2=d, r3=0.0, r4=0.0, r5=f_sl_in, r6=k_2)
        #               K1,     C,     M,    GAP,  FSLIDE, K2
        # ansys.r(nset=1, r1=0, r2=1000, r3=0, r4=0, r5=0, r6=100)#
#         print(f"nset= {ansys.parameters['NSET']+1}, r1={k_1}, r2={d}, r3={0.0}, r4={0.0}, r5={f_sl_in}, r6={k_2}")
        ansys.run('nset=rlinqr(0,14)')
        ansys.run('itype=etyiqr(0,14)')
        # ansys.load_parameters()

        ansys.finish()
        self.coulomb_elements.append((ansys.parameters['ITYPE'], ansys.parameters['NSET']))
        return ansys.parameters['ITYPE'], ansys.parameters['NSET']  # itype, nset

    @session_restore
    def mass(self, m=""):
        ansys = self.ansys
        ansys.prep7()

        ansys.run('nset=rlinqr(0,14)')
        ansys.run('itype=etyiqr(0,14)')
        # ansys.load_parameters()

        ansys.et(itype=ansys.parameters['ITYPE'] + 1, ename='MASS21', inopr=1, kop3=2)
        ansys.r(nset=ansys.parameters['NSET'] + 1, r1=m)

        ansys.run('nset=rlinqr(0,14)')
        ansys.run('itype=etyiqr(0,14)')
        # ansys.load_parameters()

        ansys.finish()
        self.mass_elements.append((ansys.parameters['ITYPE'], ansys.parameters['NSET']))
        return ansys.parameters['ITYPE'], ansys.parameters['NSET']  # itype, nset

    @session_restore
    def beam(self, E, PRXY, RHO, A, Iyy, Izz, Iyz,):
        ansys = self.ansys
        ansys.prep7()

        ansys.run('itype=etyiqr(0,14)')
        ansys.run('nmat=mpinqr(0,0,16)')
        nmat = ansys.parameters["NMAT"]
        # ansys.load_parameters()

        ansys.et(itype=ansys.parameters['ITYPE'] + 1, ename='BEAM188', kop3=3, inopr=1)
        ansys.sectype(1, "BEAM", "ASEC", "tube", 0)
        '''
        Arbitrary: User-supplied integrated section
        properties instead of basic geometry data.
        Data to provide in the value fields:
        A, Iyy, Iyz, Izz, Iw, J, CGy, CGz, SHy,
        SHz, TKz, TKy
        where
        A = Area of section
        Iyy = Moment of inertia about the y axis
        Iyz = Product of inertia
        Izz = Moment of inertia about the z axis
        Iw = Warping constant
        J = Torsional constant
        CGy = y coordinate of centroid
        CGz = z coordinate of centroid
        SHy = y coordinate of shear center
        SHz = z coordinate of shear center
        TKz = Thickness along Z axis (maximum
        height)
        TKy = Thickness along Y axis (maximum
        width
        '''
        CGy = 0
        CGz = 0
        SHz = 0
        SHy = 0
        Tkz = 0
        Tky = 0

        IW = 0
        J = 1  # assumes torsional motion is restrained
        # no warping, no torsion

        ansys.secdata(A, Iyy, Iyz, Izz, IW, J, CGy, CGz, SHy, SHz, Tkz, Tky)
        # SECOFFSET, CENT
        ansys.run('itype=etyiqr(0,14)')

        ansys.mptemp(nmat, 0)
        ansys.mpdata("EX", nmat, "", E)
        ansys.mpdata("PRXY", nmat, "", PRXY)
        ansys.mpdata("DENS", nmat, "", RHO)

        ansys.run('nmat=mpinqr(0,0,14)')
        # ansys.load_parameters()

        ansys.finish()

        self.beam_elements.append((ansys.parameters['ITYPE'], ansys.parameters['NSET']))
        return ansys.parameters['ITYPE'], ansys.parameters["NMAT"]  # itype, nset, nmat

    def build_conti(self, struct_parms, Ldiv, damping=None, num_modes=None, meas_locs=None):
        '''
        struct_parms = {
                'L'         : 200,

                'E'         : 2.1e11,
                'A'         : 0.0338161033232405,
                'rho'       : 7850,
                'Iy'        : 0.0136045227118697,
                'Iz'        : 0.0136045227118697,
                'Iyz'       : 0,

                'ky_nl'     : 117476.186062221,
                'kz_nl'     : 135649.815292788,
                'x_knl'     : 160,

                'm_tmd'     : 0,
                'ky_tmd'    : 0,
                'kz_tmd'    : 0,
                'dy_tmd'    : 0,
                'dz_tmd'    : 0,
                'x_tmd '    : 200,

            }
        
        argument:
            damping = None -> no damping
                zeta -> global rayleigh, (ALPHAD, BETAD)
                (zeta, zeta) -> global rayleigh (MP,BETD and MP,ALPD)
                (alpha, beta, False) -> global  rayleigh (MP,BETD and MP,ALPD)
        '''
        self.damped_frequencies = None
        self.frequencies = None
        logger.debug('self.build_conti')

        if num_modes is None:
            num_modes = max(2, int(np.floor(Ldiv / 10) - 1))  # choose at least 1 mode
            logger.info(f'Choosing num_modes as {num_modes} based on the number of nodes {Ldiv}')
        assert num_modes <= Ldiv - 1
        if num_modes > Ldiv / 10:
            logger.warning(f'The number of modes {num_modes} should be less/equal than 0.1 x number of nodes (= {Ldiv}).')

        ansys = self.ansys
        ansys.clear()
        ansys.prep7()

        # Nodes
        L = struct_parms['L']

        x_nodes = np.linspace(0, L, Ldiv)

        x_knl = struct_parms['x_knl']
        x_nodes[np.argmin(np.abs(x_nodes - x_knl))] = x_knl

        x_tmd = struct_parms['x_tmd']
        x_nodes[np.argmin(np.abs(x_nodes - x_tmd))] = x_tmd

        nodes_coordinates = []
        for i, x in enumerate(x_nodes):
            nodes_coordinates.append((i + 1, x, 0, 0))
        nodes_coordinates.append((i + 2, x_knl, 0, 0))
        nodes_coordinates.append((i + 3, x_tmd, 0, 0))
        nodes_coordinates = np.array(nodes_coordinates)

        for i, x, y, z in nodes_coordinates:
            ansys.n(node=i, x=x, y=y, z=z)

        ansys.nsel("S", "LOC", comp='x', vmin=x_knl, vmax=x_knl)
        ansys.get("NKNL1", "NODE", "", "NUM", "MIN")
        ansys.get("NKNL2", "NODE", "", "NUM", "MAX")

        ansys.seltol(1e-6)
        ansys.nsel("S", "LOC", comp='x', vmin=x_tmd, vmax=x_tmd)
        ansys.get("NKTMD1", "NODE", "", "NUM", "MIN")
        ansys.get("NKTMD2", "NODE", "", "NUM", "MAX")
        ansys.seltol('')

        if meas_locs is not None:
            idcs = np.argmin(np.sqrt((nodes_coordinates[:, 1, None] - meas_locs) ** 2), axis=0)
            ansys.nsel(type='NONE')
            # for x_loc in meas_locs:
            for i, idx in enumerate(idcs):
                # print(i,meas_locs[i],nodes_coordinates[idx,:])
                n_node = nodes_coordinates[idx, 0]
                ansys.nsel('A', item='NODE', comp='', vmin=n_node, vmax=n_node)

            # ansys.nsel('A', item='NODE',vmin='NKTMD2', vmax='NKTMD2')
            ansys.cm(cname='meas_nodes', entity='NODE')  # and group into component assembly
        else:
            ansys.nsel(type='ALL')
        ansys.cm(cname='meas_nodes', entity='NODE')  # and group into component assembly
        ansys.starvget(parr='meas_nodes', entity="NODE", entnum='', item1="NLIST")
        meas_nodes = np.array(ansys.parameters['meas_nodes'])
        if not meas_nodes.shape:
            meas_nodes = meas_nodes[np.newaxis]
        ansys.nsel(type='ALL')

        tmd_elems = [self.mass(struct_parms['m_tmd']),
                     self.voigt_kelvin(struct_parms['ky_tmd'], struct_parms['dy_tmd'], 'UY'),
                     self.voigt_kelvin(struct_parms['kz_tmd'], struct_parms['dz_tmd'], 'UZ')]
        wire_elems = [self.voigt_kelvin(struct_parms['ky_nl'], dof='UY'), self.voigt_kelvin(struct_parms['kz_nl'], dof='UZ')]
        pipe = self.beam(struct_parms['E'], 0.2, struct_parms['rho'],
                         struct_parms['A'], struct_parms['Iy'], struct_parms['Iz'], struct_parms['Iyz'])

        ansys.type(pipe[0])
        ansys.mat(pipe[1])
        ansys.real("")
        ansys.esys(0)
        ansys.secnum(1)
        for i in range(1, Ldiv):  # actually build ntn_cons beforehand and iterate over them
            ansys.e(i, i + 1)

        ansys.type(tmd_elems[0][0])
        ansys.real(tmd_elems[0][1])
        ansys.e("NKTMD2")

        for et, r in wire_elems:
            ansys.type(et)
            ansys.real(r)
            ansys.e("NKNL1", "NKNL2")

        ansys.type(tmd_elems[1][0])
        ansys.real(tmd_elems[1][1])
        ansys.e("NKTMD1", "NKTMD2")
        ansys.type(tmd_elems[2][0])
        ansys.real(tmd_elems[2][1])
        ansys.e("NKTMD1", "NKTMD2")

        # boundary conditions
        ansys.nsel("S", item='LOC', comp='x', vmin=0, vmax=0)
        ansys.get('NBC', 'node', 0, "num", "min")

        ansys.d(node='ALL', value=0, lab='UY', lab2='UZ', lab3='UX', lab4='ROTX')

        ansys.nsel("ALL")
        ansys.d(node='NKNL2', value=0, lab='UY', lab2='UZ')
        # ansys.d(node='NKNL1', value=0,lab='UY',lab2='UZ')
        ansys.d("ALL", value=0, lab="UX", lab2="ROTX")

        ansys.finish()

        self.struct_parms = struct_parms
        self.nodes_coordinates = nodes_coordinates
        self.num_modes = num_modes

        # ansys.open_gui()

        frequencies, _, _ = self.modal(damped=False, num_modes=num_modes)

        # undamped analysis
        if damping is None or damping == 0:
            alpha = None
            beta = None
            damped = False
        # constant modal damping approximated by global rayleigh proportional damping
        elif isinstance(damping, float):
            omega1 = frequencies[0] * 2 * np.pi
            omega2 = frequencies[-1] * 2 * np.pi
            zeta1 = damping
            zeta2 = damping
            alpha, beta = np.linalg.solve(np.array([[1 / omega1, omega1], [1 / omega2, omega2]]), [2 * zeta1, 2 * zeta2])
            logger.debug(f'Rayleigh parameters: {alpha}, {beta}')
            damped = True
        # variable modal damping  approximated by global rayleigh proportional damping
        elif len(damping) == 2:
            omega1 = frequencies[0] * 2 * np.pi
            omega2 = frequencies[-1] * 2 * np.pi
            zeta1, zeta2 = damping
            alpha, beta = np.linalg.solve(np.array([[1 / omega1, omega1], [1 / omega2, omega2]]), [2 * zeta1, 2 * zeta2])
            logger.debug(f'Rayleigh parameters: {alpha}, {beta}')
            damped = True
        # mass/stiffness/rayleigh proportional damping
        elif len(damping) == 3 and not damping[2]:
            # should already be incorporated into the model
            alpha, beta, _ = damping
            damped = True
        else:
            raise RuntimeError(f'An unsupported damping argument was specified: {damping}.')

        # global proportional damping
        if damped:
            ansys.prep7()

            nmat = pipe[1]
            ansys.mpdata("ALPD", nmat, "", alpha)
            ansys.mpdata("BETD", nmat, "", beta)
            # print(alpha, beta)

            # ansys.alphad(alpha)
            # ansys.betad(beta)
            ansys.finish()

        self.damping = damping
        self.alpha = alpha
        self.beta = beta
        self.damped = damped
        self.globdamp = (struct_parms['dy_tmd'] > 0 or struct_parms['dz_tmd'] > 0)
        self.meas_nodes = np.array(meas_nodes, dtype=int)

        for i in range(len(self.state)):
            self.state[i] = False
        self.state[7] = True

        ansys.finish()

    def build_mdof(self, nodes_coordinates=[(1, 0, 0, 0), ],
                   k_vals=[1], masses=[1], d_vals=None, damping=None,
                   sl_force_vals=None, eps_duff_vals=None, hyst_vals=None,
                   num_modes=None, meas_nodes=None, **kwargs):
        '''
        this function build a MDOF spring mass system

        nodes_coordinates is a list of tuples (node_number, x, y z)
            first node in nodes_coordinates is taken as a fixed support

        masses are applied to every node
        k and d are applied between first and second node, and so on

        sl_forces are the sliding forces for friction damping.
            equivalent friction depends on an assumed omega and displacement,
            therefore must be estimated outside this function

        eps_duff_vals are the factors for the cubic nonlinear springs (connection),

        meas_nodes is a list of nodes of which the transient responses should be saved

        argument:
            damping = None -> no damping
                zeta -> global rayleigh,
                (zeta, zeta) -> global rayleigh
                (alpha,beta,True/False) -> global/elementwise proportional
                (eta, True/False) -> Structural/Hysteretic (global -> only damped modal and harmonic analyses not TRANSIENT! / elementwise -> only transient (IWAN))
                (Friction -> only local, via sl_forces
                IWAN needs an equivalent damping ratio and a stress-strain relationship (G,E?)
                from which a monotonic strain-softening backbone curve \tau vs \gamma (considering shear) according to Masing's rule can be computed (Kausel Eq. 7.307),
                in turn this can be approximated/discretized by N spring-slider systems in parallel (or a nonlinear stiffness
                    TRY IN SDOF first

        TODO:
        (store constrained nodes to exclude them from indexing operations)
        (extend 3D: arguments constraints, ntn_conns)
        (extend nonlinear)
        (extend damping)

        '''
        logger.debug('self.build_mdof')
        # if m modes are of interest number of nodes n > 10 m
        num_nodes = len(nodes_coordinates)

        if num_modes is None:
            num_modes = max(1, int(np.floor(num_nodes / 10) - 1))  # choose at least 1 mode
            logger.info(f'Choosing num_modes as {num_modes} based on the number of nodes {num_nodes}')
        assert num_modes <= num_nodes - 1
        if num_modes > num_nodes / 10:
            logger.warning(f'The number of modes {num_modes} should be less/equal than 0.1 x number of nodes (= {num_nodes}).')

        if d_vals is None:
            if damping is None:
                logger.debug('Performing undamped analysis.')
                d_vals = [0 for _ in range(num_nodes - 1)]
            elif isinstance(damping, float):
                d_vals = [0 for _ in range(num_nodes - 1)]
            else:
                d_vals = [0 for _ in range(num_nodes - 1)]
        elif damping is not None:
            logger.warning('Applying elementwise damping and global proportional at the same time. Not sure how ANSYS handles this.')
        else:
            # pre generated elementwise damping
            pass

        if sl_force_vals is None:
            sl_force_vals = [0 for _ in range(num_nodes - 1)]
        if eps_duff_vals is None:
            eps_duff_vals = [0 for _ in range(num_nodes - 1)]
        if hyst_vals is None:
            hyst_vals = [0 for _ in range(num_nodes - 1)]
#         if isinstance(d_max, (float, int)) or d_max is None:
#             d_max_vec = [None for _ in range(num_nodes)]
#             d_max_vec[-1] = d_max
#             d_max = d_max_vec

        assert len(k_vals) == num_nodes - 1
        assert len(masses) == num_nodes
        assert len(d_vals) == num_nodes - 1
        # assert len(d_max) == num_nodes

        ansys = self.ansys
        ansys.clear()
        ansys.prep7()
        # Nodes
        ntn_conns = [[] for i in range(num_nodes - 1)]
        for i, (node, x, y, z) in enumerate(nodes_coordinates):
            ansys.n(node, x, y, z)
            if i < num_nodes - 1: ntn_conns[i] = [node]
            if i > 0: ntn_conns[i - 1].append(node)

        # Elements
        lastm = None
        for (node, x, y, z), m in zip(nodes_coordinates, masses):
        #     generate mass for every node
            # print(node)
            if m != lastm:
                mass = self.mass(m)

            ansys.type(mass[0])
            ansys.real(mass[1])
            ansys.e(node)
            lastm = m

        # build equivalent linear model first
        last_params = [None for _ in range(5)]

        for params in zip(ntn_conns, k_vals, d_vals, sl_force_vals, eps_duff_vals, hyst_vals):
            #      0, 1, 2,   3,      4,    5
            ntn_conn, k, d, fsl, eps_duff, hyst = params
            # print(ntn_conn)
            if fsl:

                logger.warning("friction untested")
                # fsl is provided, because the equation for its computation contains omega and d_mean, which is not known here
                if last_params[3] != fsl or last_params[1] != k or last_params[2] != d:
                    k_1 = k * 100  # factor could also be provided,
                    coulomb = self.coulomb(k_2=k, k_1=k_1, f_sl=fsl, d=d)
                ansys.type(coulomb[0])
                ansys.real(coulomb[1])
            else:
                if last_params[1] != k or last_params[2] != d:
                    voigt_kelvin = self.voigt_kelvin(k=k, d=d, dof='UZ')
                ansys.type(voigt_kelvin[0])
                ansys.real(voigt_kelvin[1])

            ansys.e(*ntn_conn)
            if eps_duff != 0 or hyst != 0:
                logger.warning("nonlinear untested")
                # assert abs(nl_ity) <= 0.5
                # only the nonlinear part is added here, the linear part is in voigt-kelvin/coulomb

                # deltad = d_max[ntn_conn[0] - 1] + d_max[ntn_conn[1] - 1]  # assuming linear node-to-node connections and node numbering starting with 1

                if last_params[4] != eps_duff:
                    # increase k here by dividing because it was decreased before
                    # TODO: change definition of nonlinear, currently the stiffness is defined for the working point, which is rather impractical, e.g. because of this division
                    nonlinear = None  # self.nonlinear(nl_ity, deltad, k / (1 - nl_ity))
                    raise NotImplementedError('Duffing Oscillator not yet implemented')
                ansys.type(nonlinear[0])
                ansys.real(nonlinear[1])
                ansys.e(*ntn_conn)

            last_params = params

        # boundary conditions
        # Specifies that DOF constraint values are to be accumulated.
        # Needed to restore constraints after forced displacement analyses
        # ansys.dcum(oper='ADD') # does not work

        ansys.d(node=nodes_coordinates[0][0], value=0, lab='UX', lab2='UY', lab3='UZ')  # ,lab4='RX', lab5='RY',lab6='RZ')

        # move constrained node to the end to ensure consistency with ANSYS node indexing -> bull shit
        # nodes_coordinates.append(nodes_coordinates.pop(0))

        for node, x, y, z in nodes_coordinates:
            ansys.d(node=node, value=0, lab='UX', lab2='UY')
        ansys.finish()

        self.nodes_coordinates = nodes_coordinates
        self.ntn_conns = ntn_conns
        self.num_nodes = num_nodes
        self.num_modes = num_modes
        self.d_vals = d_vals
        self.k_vals = k_vals
        self.masses = masses
        self.eps_duff_vals = eps_duff_vals
        self.sl_force_vals = sl_force_vals
        self.hyst_vals = hyst_vals

        # TODO: Account for multiple coloumbs, to get frequency estimates with loose sliders
        # TODO: Account for nonlinear stiffness, i.e. operating point
        frequencies, _, _ = self.modal(damped=False)

        # undamped analysis
        if damping is None or damping == 0:
            alpha = None
            beta = None
            globdamp = False
            damped = False
        # constant modal damping approximated by global rayleigh proportional damping
        elif isinstance(damping, float) and num_modes >= 2:
            omega1 = frequencies[0] * 2 * np.pi
            omega2 = frequencies[-1] * 2 * np.pi
            zeta1 = damping
            zeta2 = damping
            alpha, beta = np.linalg.solve(np.array([[1 / omega1, omega1], [1 / omega2, omega2]]), [2 * zeta1, 2 * zeta2])
            logger.debug(f'Rayleigh parameters: {alpha}, {beta}')
            globdamp = True
            damped = True
        elif isinstance(damping, float) and num_modes == 1:
            omega = frequencies[0] * 2 * np.pi
            alpha = 0
            beta = 2 * damping / omega
            globdamp = False
            damped = True
        # variable modal damping  approximated by global rayleigh proportional damping
        elif len(damping) == 2 and not isinstance(damping[1], bool) and num_modes >= 2:
            omega1 = frequencies[0] * 2 * np.pi
            omega2 = frequencies[-1] * 2 * np.pi
            zeta1, zeta2 = damping
            alpha, beta = np.linalg.solve(np.array([[1 / omega1, omega1], [1 / omega2, omega2]]), [2 * zeta1, 2 * zeta2])
            logger.debug(f'Rayleigh parameters: {alpha}, {beta}')
            globdamp = True
            damped = True
        # mass/stiffness/rayleigh proportional damping either elementwise or global
        elif len(damping) == 3:
            # should already be incorporated into the model
            alpha, beta, globdamp = damping
            damped = True
        # constant structural damping via DMPSTR (modal, harmonic) or IWAN (transient) based on loss factor
        elif len(damping) == 2 and isinstance(damping[1], bool):
            raise RuntimeError('Structural/Hysteretic damping not implemented yet.')
        else:
            raise RuntimeError(f'An unsupported damping argument was specified: {damping}.')

        # global proportional damping
        if damped:
            if globdamp:
                self.ansys.prep7()
                self.ansys.alphad(alpha)
                self.ansys.betad(beta)
                self.ansys.finish()
            else:
                # if only zeta or alpha,beta was provided, but local damping required
                # check/ compute the d_vals
                # d_vals is a list and its length should fit
                # elementwise Rayleigh with (alpha,) beta
                logger.debug('Using elementwise damping.')
                if alpha > 1e-15:
                    logger.warning(f'Elementwise proportional damping takes only a stiffness factor {beta}. Neglecting mass factor {alpha}!')
                for i in range(num_nodes - 1):
                    d_val = k_vals[i] * beta
                    if d_vals[i] and d_vals[i] != d_val:
                        raise RuntimeError("You should not provide global and local damping together")
#                         print(f"Overwriting viscous dashpot value {d_vals[i]} for element {i} with {d_val}")
                    d_vals[i] = d_val

                lastk = None
                lastd = None
                self.ansys.prep7()
                i = 0
                for ntn_conn, k, d in zip(ntn_conns, k_vals, d_vals):
#                     print(ntn_conn)
                    # TODO: This is likely to fail, if non-proportional d_vals are provided
                    if k != lastk and d != lastd:
                        nset = self.voigt_kelvin_elements[i][1]
                        ansys.rmodif(nset, 2, d)
                        lastk = k
                        lastd = d
                        i += 1

                self.ansys.finish()

        if meas_nodes is not None:
            meas_nodes = np.sort(meas_nodes)
            logger.debug(f'Using node components {meas_nodes}')
            ansys.nsel(type='NONE')
#             if 1 not in meas_nodes:
#                 print('Warning: make sure to include the first (constrained) node in meas_nodes for visualization.')
            for meas_node in meas_nodes:
                for node, x, y, z in self.nodes_coordinates:
                    if node == meas_node:
                        break
                else:
                    raise RuntimeError(f'Meas node {meas_node} was not found in nodes_coordinates')
                # print(node)
                ansys.nsel(type='A', item='NODE', vmin=meas_node, vmax=meas_node)  # select only nodes of interest
            # ansys.nsel(type='A', item='NODE', vmin=2,vmax=2) # select only nodes of interest
            # ansys.cm(cname='meas_nodes', entity='NODE') # and group into component assembly
        else:
            ansys.nsel(type='ALL')
            meas_nodes = [node for node, _, _, _ in nodes_coordinates]
            meas_nodes = np.sort(meas_nodes)

        ansys.cm(cname='meas_nodes', entity='NODE')  # and group into component assembly

        ansys.nsel(type='ALL')
        # print(meas_nodes)

        ansys.finish()
        # time.sleep(20)
        self.damping = damping
        self.alpha = alpha
        self.beta = beta
        self.damped = damped
        self.globdamp = globdamp
        self.meas_nodes = np.array(meas_nodes)

        self.state[0] = True
        for i in range(1, len(self.state)):
            self.state[i] = False
        # print(self.state)
#         self.ntn_conns  =ntn_conns
#         self.fric_rats  =fric_rats
#         self.nl_ity_rats=nl_ity_rats
#         self.d_max      =d_max
#         self.f_scale    =f_scale

    def example_rod(self, num_nodes, damping=None, nl_stiff=None, sl_forces=None, freq_scale=1, num_modes=None,  # structural parameters
                    num_meas_nodes=None, meas_nodes=None,  # signal parameters
                    ** kwargs):
        '''
        argument:
            damping = None -> no damping
                      zeta -> global rayleigh,
                      (zeta, zeta) -> global rayleigh
                      (alpha,beta,True/False) -> global/elementwise proportional
                      (eta, True/False) -> Structural/Hysteretic (global -> only damped modal and harmonic analyses not TRANSIENT! / elementwise -> only transient (IWAN))
                      (Friction -> only local via sl_forces
                      IWAN needs an equivalent damping ratio and a stress-strain relationship (G,E?)
                        from which a monotonic strain-softening backbone curve \tau vs \gamma (considering shear) according to Masing's rule can be computed (Kausel Eq. 7.307),
                        in turn this can be approximated/discretized by N spring-slider systems in parallel (or a nonlinear stiffness
                        TRY IN SDOF first
    
        TODO:
        - nonlinear-equivalent
            - stiffness (as in SDOF systems), estimation of d_max
            - friction (equivalent per node and mode, averaging?)
            - structural damping (ANSYS routines, IWAN)
        '''
        pyansys = _get_pyansys()

        ansys = self.ansys

        try:
            ansys.finish()
        except pyansys.errors.MapdlExitedError as e:
            print(e)
            ansys = self.start_ansys()
        ansys.clear()

        # Example structure Burscheid Longitudinal modes
        total_height = 200
        E = 2.1e11 * freq_scale ** 2  # for scaling the frequency band linearly; analytical solution is proportional to sqrt(E); (ratios are fixed for a rod 1:2:3:4:... or similar)
        # I=0.01416
        A = 0.0343
        rho = 7850

        num_nodes = int(num_nodes)

        section_length = total_height / (num_nodes - 1)
        nodes_coordinates = []
        k_vals = [0 for _ in range(num_nodes - 1)]
        masses = [0 for _ in range(num_nodes)]
        d_vals = [0 for i in range(num_nodes - 1)]
        eps_duff_vals = [0 for _ in range(num_nodes - 1)]
        sl_force_vals = [0 for _ in range(num_nodes - 1)]
        hyst_vals = [0 for _ in range(num_nodes - 1)]

        if isinstance(damping, (list, tuple)):
            if len(damping) == 2 and isinstance(damping[1], bool):
                hyst_damp = damping[0]
            else:
                hyst_damp = None
        else:
            hyst_damp = None

        for i in range(num_nodes):
            #  nodes_coordinates.append([i+1,0,0,section_length*i])
            nodes_coordinates.append([i + 1, 0, 0, i])  # to disable Warning "Nodes are not coincident"
            masses[i] += 0.5 * rho * A * section_length
            if i >= 2:
                masses[i - 1] += 0.5 * rho * A * section_length
            if i >= 1:
                k_vals[i - 1] = E * A / section_length

                if nl_stiff is not None:
                    eps_duff_vals[i - 1] = nl_stiff
                if sl_forces is not None:
                    sl_force_vals[i - 1] = sl_forces
                if hyst_damp is not None:
                    hyst_vals[i - 1] = hyst_damp

        masses.append(800)
        d_vals.append(6.651e3)
        k_vals.append(1.1035e6)
        eps_duff_vals.append(0)
        sl_force_vals.append(0)
        hyst_vals.append(0)
        nodes_coordinates.append([i + 2, 0, 0, i + 1])
        if num_modes == num_nodes - 1:
            num_modes += 1

        if num_meas_nodes is None and meas_nodes is None:
            meas_nodes = [node for node, _, _, _ in nodes_coordinates[1:]]  # exclude the constrained node
        elif num_meas_nodes is not None and meas_nodes is None:
            # step = int(np.floor(num_nodes/num_meas_nodes))
            # meas_nodes = np.concatenate(([1],np.arange(step,num_nodes+1,step)))
            meas_nodes = np.rint(np.linspace(2, num_nodes, int(num_meas_nodes))).astype(int)
            if len(meas_nodes) != (num_meas_nodes):
                logger.warning(f'Warning number of meas_nodes generated {len(meas_nodes)} differs from number of meas_nodes specified {num_meas_nodes}')
        elif num_meas_nodes is not None and meas_nodes is not None:
            raise RuntimeError(f'You cannot provide meas_nodes {meas_nodes} and num_meas_nodes {num_meas_nodes} at the same time')

        logger.info(f'Building an example rod structure with {num_nodes} nodes, {damping} % damping, considering the output of {num_modes} modes at {num_meas_nodes} measurement nodes.')

        self.build_mdof(nodes_coordinates=nodes_coordinates,
                        k_vals=k_vals, masses=masses, d_vals=d_vals, damping=damping,
                        sl_force_vals=sl_force_vals, eps_duff_vals=eps_duff_vals, hyst_vals=hyst_vals,
                        meas_nodes=meas_nodes, num_modes=num_modes,)

        return

    def example_beam(self, num_nodes, damping=None, num_modes=None, damp_mode=1,  # structural parameters
                    num_meas_nodes=None,
                    ** kwargs):
        if damp_mode is not None:
            damp_ind = slice((damp_mode - 1) * 2, damp_mode * 2)
            # mu = 0.07 #m/M        #(5)    je groesser umso groesser die bedaempfte frequenzbreite

            # pre-generated "design" solution
            frequencies = [0.18613589, 0.18693765, 0.33319226, 0.35275437,
                           0.66958636, 0.67296631, 1.38965809, 1.39465315, 100, 100]
            modal_masses = [22691.6, 22759.9, 6695.5, 6549.7,
                            20753.4, 22915.7, 14980.4, 15316.2, 53000, 53000]
            M = np.mean(modal_masses[damp_ind])  # modal mass of first mode(s)
            fH = np.mean(frequencies[damp_ind])  # mean of first mode(s)
            # fH=100

            # mD = mu * M
            mD = kwargs.get('mD', 800)
            mu = mD / M
            kappa_opt = 1 / (1 + mu)  # (22)
            zeta_opt = np.sqrt(3 * mu / (8 * (1 + mu) ** 3))  # (23)
            fD = kappa_opt * fH  # (62)
            kD = (2 * np.pi * fD) ** 2 * mD  # (63)
            dD = 2 * mD * (2 * np.pi * fD) * zeta_opt

            print(f"The TMD is designed to damp mode(s) {damp_mode} at {fH:1.3f} Hz with a mass ratio of {mu},"
                  f" a mass {mD:1.3f}, frequency {fD:1.3f} and damping ratio {zeta_opt:1.3f}.")
        else:
            mD = 0
            kD = 1
            dD = 0

        struct_parms = {
                'L': 200,

                'E': 2.1e11,
                'A': 0.0338161033232405,
                'rho': 7850,
                'Iy': 0.0136045227118697,
                'Iz': 0.0136045227118697,
                'Iyz': 0,

                'kz_nl': 117476.186062221,
                'ky_nl': 135649.815292788,
                'x_knl': 160,

                'm_tmd': mD,
                'ky_tmd': kD,
                'kz_tmd': kD,
                'dy_tmd': dD,
                'dz_tmd': dD,
                'x_tmd': 200,
                }
        # print(struct_parms)
        if num_meas_nodes:
            meas_locs = np.linspace(0, struct_parms['L'], num_meas_nodes + 1)[1:]
        else:
            meas_locs = None

        self.build_conti(struct_parms, num_nodes, damping, num_modes, meas_locs=meas_locs)

    def free_decay(self, d0, dscale=1, deltat=None, dt_fact=None, timesteps=None, num_cycles=None, **kwargs):
        '''
        to take d0 parameter, do all the initial displacement things and call transient with passing kwargs
        save last analysis run as a class variable: d0, time histories, signal and analysis parameters


        d0 = initial displacement
            'static' : displacement obtained from static analysis with unit top displacement
            'modes' : arithmetic mean of all mode shapes
            number > 0, mode_index of which to excite
        
        TODO: extend 3D
        '''

        num_nodes = self.num_nodes
        uz = 2

        if d0 == 'static':
            # static analysis for initial conditions (free decay)
    #         disp = mech.static(fz=np.array([[num_nodes//2, -100000000]]))
            disp = self.static(uz=np.array([[num_nodes, 1]]))
            # print(disp[:,:], disp.shape)
            mode_index = None
    #         disp = mech.static(fz=np.array([[num_nodes//2, -100000000]]), uz=np.array([[num_nodes, 1]]))
        elif d0 == 'modes':
            _, _, modeshapes = self.modal(damped=False)
            for mode in range(self.num_modes):
                modeshapes[:,:, mode] /= modeshapes[-1, uz, mode]
            mode_index = None
        elif isinstance(d0, int):
            _, _, modeshapes = self.modal(damped=False)
            for mode in range(self.num_modes):
                modeshapes[:,:, mode] /= modeshapes[-1, uz, mode]
            mode_index = d0
        else:
            raise RuntimeError(f"No initial displacement was specified d0: {d0}.")

        dt_fact, deltat, num_cycles, timesteps, mode_index = self.signal_parameters(dt_fact, deltat, num_cycles, timesteps, mode_index)

        frequencies, damping, _ = self.modal(damped=True)

        logger.info(f"Generating free decay response with f {frequencies[mode_index]:1.3f}, zeta {damping[mode_index]:1.3f}, free_decay mode {mode_index+1}")

        d = np.full((timesteps, num_nodes), np.nan)
        if d0 == 'static':
            d[0,:] = disp[:, uz]
        elif d0 == 'modes':
            d[0,:] = np.mean(modeshapes[:, uz,:], axis=1)
        elif isinstance(d0, int):
            d[0,:] = modeshapes[:, uz, d0]
#             y0=modeshapes[self.meas_nodes-1,uz,d0]
#             omega_d = frequencies[d0]*2*np.pi
#             zeta=damping[d0]
#             print(f'initial acceleration {-omega_d**2*y0}')
#             print(f'initial velocity {omega_d/np.sqrt(1-zeta**2)*zeta*y0}')

        parameter_set = kwargs.pop('parameter_set', None)

        time_values, response_time_history = self.transient(d=d, deltat=deltat, timesteps=timesteps, parameter_set=parameter_set, **kwargs)

        self.decay_mode = d0
        self.t_vals_decay = time_values
        self.resp_hist_decay = response_time_history

        self.state[1] = True

        return time_values, response_time_history

    def ambient(self, f_scale, deltat=None, dt_fact=None, timesteps=None, num_cycles=None, seed=None, **kwargs):
        '''to take f_scale or more advanced parameters and call transient with passing kwargs
        save last analysis run as a class variable: f_scale, time histories, input time histories, signal and analysis parameters
        '''
        num_nodes = self.num_nodes

        dt_fact, deltat, num_cycles, timesteps, _ = self.signal_parameters(dt_fact, deltat, num_cycles, timesteps)

        logger.info(f"Generating ambient response in frequency range [0 ... {self.frequencies[-1]:1.3f}], zeta range [{np.min(self.modal_damping):1.3f}-{np.max(self.modal_damping):1.3f}], number of modes {self.num_modes}")

        logger.warning("WARNING: Gaussian Noise Generator has not been verified!!!!!!")
        rng = np.random.default_rng(seed)

        # f = rng.normal(0, f_scale, (timesteps, num_nodes))

        f = np.empty((timesteps, num_nodes))
        for channel in range(num_nodes):
            phase = rng.uniform(-np.pi, np.pi, (timesteps // 2 + 1,))
            Pomega = f_scale * np.ones_like(phase) * np.exp(1j * phase)
            f[:, channel] = np.fft.irfft(Pomega, timesteps)

        # TODO: implement correlated random processes here
        # TODO: implement random impulses here
        try:
            time_values, response_time_history = self.transient(f=f, deltat=deltat, timesteps=timesteps, **kwargs)
        except Exception as e:
            err_file = os.path.join(self.ansys.directory, self.ansys.jobname + '.err')
            with open(err_file) as f:
                lines_found = []
                block_counter = -1
                while len(lines_found) < 10:
                    try:
                        f.seek(block_counter * 4098, os.SEEK_END)
                    except IOError:
                        f.seek(0)
                        lines_found = f.readlines()
                        break
                    lines_found = f.readlines()
                    block_counter -= 1
                for i in range(1, min(1 + len(lines_found), 11)):  # try to print only the last error message but not more than ten
                    if lines_found[-i].startswith(' *** ERROR ***'):
                        break
                logger.error(str(lines_found[-i:]))
            raise e

        self.inp_hist_amb = f
        self.t_vals_amb = time_values
        self.resp_hist_amb = response_time_history

        self.state[2] = True

        return time_values, response_time_history, f

    def impulse_response(self, impulses=[[], [], [], []], form='sine', mode='combined', deltat=None, dt_fact=None, timesteps=None, num_cycles=None, out_quant=['d'], **kwargs):
        '''
        impulses = list of list containing
            [ref_nodes, imp_forces, imp_times, imp_durs/-num_modes]
        if imp_dur is negative, it is interpreted, as the highest mode, which is excited by half of the wavepower 1/sqrt(2)  (-3dB) (only for half-sine)
        
        form = 'sine', 'step', 'rect'
        
        mode='matrix' return the IRF matrix for each reference node
        mode='combined' apply all impulses at the same time
        
        
        save last analysis run as a class variable: ref_nodes, IRF matrix, signal and analysis parameters
        
        
        Unit impulse -> IRF
        Rectangular impulse -> define impulse length (includes stepped load for infinite length)
        Half-Sine impulse -> define f_max and power at f_max

        '''
        pyansys = _get_pyansys()

        def analytical_constants_half_sine(p0, k, omegan, zeta, tp):
            omegad = omegan * np.sqrt(1 - zeta ** 2)
            omegaf = np.pi / tp

#             print(f'tp {tp},\t Tn {2*np.pi/omegan},\t p0 {p0},\t k {k},\t zeta {zeta},\t omegaf {omegaf},\t omegan {omegan},\t  omegad {omegad}')

            if np.isclose(zeta, 0) and np.isclose(omegaf, omegad):
                raise NotImplementedError('Modal decomposition of impulses at resonance for undamped system is not implemented yet')

            ydyn = np.sqrt(p0 / k / ((1 - (omegaf / omegan) ** 2) ** 2 + (2 * zeta * omegaf / omegan) ** 2))
            A = ydyn ** 2 * 2 * zeta * omegaf / omegan
            B = -ydyn ** 2 * omegaf / omegad * (1 - (omegaf / omegan) ** 2 - 2 * zeta ** 2)
            C = ydyn ** 2 * (1 - (omegaf / omegan) ** 2)
            D = -ydyn ** 2 * 2 * zeta * omegaf / omegan
            E = np.exp(-omegan * zeta * tp) * (A * np.cos(omegad * tp) + B * np.sin(omegad * tp) - np.exp(omegan * zeta * tp) * D)
            F = np.exp(-omegan * zeta * tp) * (-A * np.sin(omegad * tp) + B * np.cos(omegad * tp)) - C * omegaf / omegad - D * zeta * omegan / omegad

            return A, B, C, D, E, F

        def analytical_response_half_sine(t_vals, omegan, zeta, A, B, C, D, E, F, tp, ts=0):

            omegad = omegan * np.sqrt(1 - zeta ** 2)
            omegaf = np.pi / tp

            imp_resp_ex = np.zeros_like(t_vals)

            this_inds = np.logical_and(t_vals > ts, t_vals < ts + tp)
            forced_resp = np.exp(-omegan * zeta * (t_vals[this_inds] - ts))\
                         * (A * np.cos(omegad * (t_vals[this_inds] - ts))
                            +B * np.sin(omegad * (t_vals[this_inds] - ts)))\
                         +C * np.sin(omegaf * (t_vals[this_inds] - ts))\
                         +D * np.cos(omegaf * (t_vals[this_inds] - ts))
            imp_resp_ex[this_inds] = forced_resp

            this_inds = t_vals >= ts + tp
            free_response = np.exp(-omegan * zeta * (t_vals[this_inds] - ts - tp))\
                             * (E * np.cos(omegad * (t_vals[this_inds] - ts - tp))
                                +F * np.sin(omegad * (t_vals[this_inds] - ts - tp)))

            imp_resp_ex[this_inds] = free_response

            return imp_resp_ex

        def work_integrand_half_sine(t, omegan, zeta, A, B, C, D, p_i, tp, ts=0):

            omegad = omegan * np.sqrt(1 - zeta ** 2)
            omegaf = np.pi / tp

            assert t > ts  and t < ts + tp

            forced_resp = np.exp(-omegan * zeta * (t - ts))\
                         * (A * np.cos(omegad * (t - ts))
                            +B * np.sin(omegad * (t - ts)))\
                         +C * np.sin(omegaf * (t - ts))\
                         +D * np.cos(omegaf * (t - ts))
            impulse = -np.cos(omegaf * (t - ts)) * p_i * omegaf

            return forced_resp * impulse

        def generate_forces(impulses, form, frequencies):

            forces = []
            imp_durs = []
            for impulse, (node, imp_force, imp_time, imp_dur) in enumerate(zip(*impulses)):

                f = np.zeros((timesteps, num_nodes))
                if form == 'step':
                    if imp_dur is not None:
                        logger.warning(f'Stepped impulses have infinite length. A finite length tp_i of {imp_dur} was provided. Setting tp_i to {t_total-imp_time}')
                    imp_dur = t_total - imp_time
                elif imp_dur < 0:  # mode number, which should be excited with at least half energy
                    # it must be an integer and less than the total number of modes (it is an index starting at 0)
                    omega = frequencies[-imp_dur] * 2 * np.pi
                    if form == 'sine':
                        imp_dur = 3.72916 / omega
                    elif form == 'rect':
                        imp_dur = 2.78311 / omega
                imp_durs.append(imp_dur)
                assert imp_time + imp_dur <= t_total
                if imp_time < deltat:
                    raise ValueError(f'Starting time {imp_time} at node {node} must be at least deltat {deltat}')
                step_s = int(imp_time / deltat) - 1  # effectively rounds down
                step_e = int((imp_time + imp_dur) / deltat - 1)

                if form == 'step':
                    f[step_s:, node - 1] = imp_force
                elif form == 'rect':  # Stepped or Rectangular
                    f[step_s:step_e, node - 1] = imp_force
                elif form == 'sine':  # Half Sine
                    f[step_s:step_e, node - 1] = np.sin(np.linspace(0, np.pi, step_e - step_s, endpoint=False)) * imp_force
                else:
                    raise ValueError(f'Did not understand value of argument "form": {form}')
                forces.append(f)
            impulses[3] = imp_durs
            return forces

        num_nodes = self.num_nodes

        # get the system matrices for analytical solutions
        _, _, modeshapes = self.modal(use_cache=False)  # use_cache=False has to be done, to assemble the correct system matrices into the .full file
        full_path = os.path.join(self.ansys.directory, self.ansys.jobname + '.full')
        full = pyansys.read_binary(full_path)
        # TODO: Check, that Nodes and DOFS are in the same order as in mode shapes, should be, since both are returned sorted
        _, K, M = full.load_km(as_sparse=False, sort=False)
        K += np.triu(K, 1).T  # convert to full from upper triangular
        M += np.triu(M, 1).T  # convert to full from upper triangular

        dt_fact, deltat, num_cycles, timesteps, _ = self.signal_parameters(dt_fact, deltat, num_cycles, timesteps)

        self.transient_parameters(**kwargs)

        frequencies, damping, _ = self.numerical_response_parameters()  # use numerical response parameters to ensure correct analytical responses

        num_modes = self.num_modes

        if mode == 'matrix':

            if form != 'step':
                logger.warning(f'You have chosen to generate an IRF matrix with other than stepped impulses, but {form}. Know what you are doing.')
            if len(out_quant) > 1:
                raise ValueError(f'For IRF Matrix mode, out_quant can only be a single quantity: {out_quant}')
            quant_ind = ['d', 'v', 'a'].index(out_quant[0])

            ref_nodes, imp_forces, imp_times, imp_durs = impulses
            for ref_node in ref_nodes:
                assert isinstance(ref_node, (int, np.int64))
            num_impulses = len(ref_nodes)
            num_ref_nodes = len(ref_nodes)
            num_meas_nodes = len(self.meas_nodes)

            IRF_matrix = np.zeros((num_ref_nodes, timesteps, num_meas_nodes, 3))
            F_matrix = np.zeros((num_ref_nodes, timesteps))
            energy_matrix = np.zeros((num_ref_nodes, num_modes))
            amplitudes_matrix = np.zeros((num_ref_nodes, num_modes))
            last_ref_node = 0
            for i in range(num_impulses):
                if imp_times[i] != deltat:
                    logger.warning(f'A starting time was specified as {imp_times[i]} which is invalid for IRF matrix mode. Starting time will be set to {deltat}.')
                this_ref_node = ref_nodes[i]
                if this_ref_node <= last_ref_node:
                    logger.warning(f'ref nodes should be in ascending order and not appear twice (last: {last_ref_node}, current: {this_ref_node}')
                last_ref_node = this_ref_node
                this_impulses = [(this_ref_node,), (imp_forces[i],), (deltat,), (imp_durs[1],)]
                time_values, response_time_history, f, energies, amplitudes = self.impulse_response(this_impulses, form, 'combined', deltat, dt_fact, timesteps, num_cycles, out_quant=out_quant, **kwargs)

                if form == 'step' and out_quant[0] == 'd':
                    disp = self.static(((this_ref_node, imp_forces[i]),), use_meas_nodes=True)
                    IRF_matrix[i,:,:,:] = (response_time_history[0] - disp)
                else:
                    IRF_matrix[i,:,:,:] = response_time_history[quant_ind]

                # in normal irf mode, impulses may be at all nodes, at any time
                # in irf matrix mode, impulses are only at ref node and
                # only at the beginning of the time series / signal
                # so we can use only the respective rows from the
                # input, energy and amplitudes matrix

                F_matrix[i,:] = f[:, this_ref_node - 1]
                energy_matrix[i,:] = energies[this_ref_node - 1,:]
                amplitudes_matrix[i,:] = amplitudes[this_ref_node - 1,:]

            # invalidate single signal impulse reponses
            self.inp_hist_imp = None
            self.resp_hist_imp = None
            self.modal_imp_energies = None
            self.modal_imp_amplitudes = None
            self.state[3] = False

            # save IRF matrix results
            self.t_vals_imp = time_values
            self.IRF_matrix = IRF_matrix
            self.imp_hist_imp_matrix = F_matrix
            self.modal_imp_energy_matrix = energy_matrix
            self.modal_imp_amplitude_matrix = amplitudes_matrix
            self.state[6] = True

            return time_values, IRF_matrix, F_matrix, energy_matrix, amplitudes_matrix

        logger.info(f"Generating impulse response in frequency range [0 ... {self.frequencies[-1]:1.3f}], zeta range [{np.min(self.modal_damping):1.3f}-{np.max(self.modal_damping):1.3f}], number of modes {num_modes}")
        t_total = timesteps * deltat

#         nrows = int(np.ceil(np.sqrt(num_meas_nodes)))
#         ncols=int(np.ceil(num_meas_nodes/nrows))
#         fign,axesn = plot.subplots(nrows,ncols,squeeze=False, sharex=True, sharey=True)
#         axesn=axesn.flatten()

        forces = generate_forces(impulses, form, frequencies)
        f = np.sum(forces, axis=0)  # sum up over the list of forces, which are all in the shape (timesteps, num_nodes)
        assert f.shape == (timesteps, num_nodes)

        time_values, response_time_history = self.transient(f=f, deltat=deltat, timesteps=timesteps, out_quant=out_quant, **kwargs)

#         t_vals_fine=np.linspace(0,t_total, timesteps*10)

        # This merges all impulses at the same node into a single line in the array
#         modal_imp_responses = np.zeros((t_vals_fine.size, num_nodes, num_modes))
        modal_imp_energies = np.zeros((num_nodes, num_modes))
        modal_amplitudes = np.zeros((num_nodes, num_modes))

        for i, (node_i, p_i, ts_i, tp_i) in enumerate(zip(*impulses)):

            if node_i in self.meas_nodes:
                nod_ind = np.where(self.meas_nodes == node_i)[0][0]
            else:
                nod_ind = None

#             nrows = int(np.ceil(np.sqrt(num_modes)))
#             ncols=int(np.ceil(num_modes/nrows))
#             figm,axesm = plot.subplots(nrows,ncols,squeeze=False, sharex=True, sharey=True)
#             axesm=axesm.flatten()

            omegaf_i = np.pi / tp_i

            f_i = forces[i][:, node_i - 1]

#             if form=='sine':# Half Sine
#                 t_vals_int = np.linspace(0,tp_i,10*int(tp_i/deltat))
#                 imp_int = -np.cos(omegaf_i*t_vals_int)*p_i*omegaf_i # negative of first derivative, so we don't have to compute the velocities
#             else:
            t_vals_int = np.linspace(0, tp_i, 2)  # we only need start and end values

#             sum_energies=0

            for mode in range(self.num_modes):

                phi_ij = modeshapes[node_i - 1, 2, mode]

                modeshape_j = modeshapes[:,:, mode].flatten()
                kappa_j = np.real(modeshape_j.T.dot(K).dot(modeshape_j))

                zeta_j = damping[mode]
                omegan_j = frequencies[mode] * 2 * np.pi / np.sqrt(1 - zeta_j ** 2)
                if form == 'sine':
                    A, B, C, D, E, F = analytical_constants_half_sine(p_i * np.abs(phi_ij), kappa_j, omegan_j, zeta_j, tp_i)

                    modal_amplitudes[node_i - 1, mode] += np.sqrt(E ** 2 + F ** 2) * np.abs(phi_ij)

#                     modal_imp_responses[:,node_i-1,mode] += analytical_response_half_sine(t_vals_fine, omegan_j, zeta_j, A, B, C, D, E, F, tp_i, ts_i)*np.abs(phi_ij)

                    # compute work with x10 finer resolution than model
                    imp_resp_int = analytical_response_half_sine(t_vals_int, omegan_j, zeta_j, A, B, C, D, E, F, tp_i)
#                     if np.isclose(zeta_j,0) and np.isclose(omegaf_i,frequencies[mode]*2*np.pi):
#                         print('Warning: Untested')
#                         W=(p_i*np.abs(phi_ij))**2/kappa_j*np.pi**2/8
#                     if np.isclose(zeta_j,0):
#                         W=(p_i*np.abs(phi_ij))**2/kappa_j*(omegaf_i*omegan_j/(omegaf_i**2-omegan_j**2))**2*(np.cos(omegan_j*tp_i)+1)

                    # Integration for half-sine impulse by quad integration
                    W = scipy.integrate.quad(work_integrand_half_sine, a=t_vals_int[0], b=t_vals_int[-1], args=(omegan_j, zeta_j, A, B, C, D, p_i * np.abs(phi_ij), tp_i))[0]

                elif form == 'step' or form == 'rect':

#                     this_t_vals = t_vals_fine-ts_i
#                     modal_imp_responses[this_t_vals>=0,node_i -1,mode] += np.abs(phi_ij)*p_i*np.abs(phi_ij)/kappa_j*(1-np.exp(-zeta_j*omegan_j*this_t_vals)*(np.cos(omegan_j*np.sqrt(1-zeta_j**2)*this_t_vals)+zeta_j/np.sqrt(1-zeta_j**2)*np.sin(omegan_j*np.sqrt(1-zeta_j**2)*this_t_vals)))[this_t_vals>=0]
#                     this_t_vals = t_vals_fine-ts_i-tp_i
#                     modal_imp_responses[this_t_vals>=0,node_i -1,mode] += -np.abs(phi_ij)*p_i*np.abs(phi_ij)/kappa_j*(1-np.exp(-zeta_j*omegan_j*this_t_vals)*(np.cos(omegan_j*np.sqrt(1-zeta_j**2)*this_t_vals)+zeta_j/np.sqrt(1-zeta_j**2)*np.sin(omegan_j*np.sqrt(1-zeta_j**2)*this_t_vals)))[this_t_vals>=0]

                    if form == 'rect':
                        modal_amplitudes[node_i - 1, mode] += p_i * np.abs(phi_ij) / kappa_j * np.sqrt((1 + zeta_j ** 2 / (1 - zeta_j ** 2)) * (2 - 2 * np.cos(omegan_j * np.sqrt(1 - zeta_j ** 2) * tp_i))) * np.abs(phi_ij)
                    elif form == 'step':
                        modal_amplitudes[node_i - 1, mode] += p_i * np.abs(phi_ij) / kappa_j * (np.sqrt(1 + zeta_j ** 2 / (1 - zeta_j ** 2))) * np.abs(phi_ij)

                    imp_resp_int = p_i * np.abs(phi_ij) / kappa_j * (1 - np.exp(-zeta_j * omegan_j * t_vals_int) * (np.cos(omegan_j * np.sqrt(1 - zeta_j ** 2) * t_vals_int) + zeta_j / np.sqrt(1 - zeta_j ** 2) * np.sin(omegan_j * np.sqrt(1 - zeta_j ** 2) * t_vals_int)))
                    W = p_i * np.abs(phi_ij) * (imp_resp_int[-1] - imp_resp_int[0])

#                 axesm[mode].plot(t_vals_fine, modal_imp_responses[:,node_i -1,mode]) # in physical coordinates
#                 axesm[mode].plot(time_values, np.abs(phi_ij)*f_i*np.abs(phi_ij)/kappa_j,ls='dashed') # static solution, modal static response to modal impulse transformed to physical static solution to modal impulse and scaled to physical impulse

#                 if form == 'step': shift=p_i*np.abs(phi_ij)**2/kappa_j
#                 else: shift=0
#
#                 axesm[mode].axhline(modal_amplitudes[node_i -1,mode]+shift, color='grey')
#                 axesm[mode].axhline(-modal_amplitudes[node_i -1,mode]+shift, color='grey')

                modal_imp_energies[node_i - 1, mode] += W
#                 sum_energies += W
#             figm.suptitle(f'Modal impulse responses at node: {node_i}')

#             if node_i in self.meas_nodes:
#                 nod_ind = np.where(self.meas_nodes==node_i)[0][0]
#                 W=scipy.integrate.simps(response_time_history[1][:,nod_ind]*f_i,time_values, even='first')
#                 print(f'Total impulse energy of impulse {i}: {W}. Sum of modal impulse energies: {sum_energies} (Should match approximately)')
#
# #                 for l, label in enumerate(['d']):#,'v','a']):
# #                     axesn[nod_ind].plot(time_values, response_time_history[l][:,nod_ind], label=label, ls='none', marker='+')
# #
# #                 for mode in range(self.num_modes):
# #                     axesn[nod_ind].plot(t_vals_fine, modal_imp_responses[:,node_i -1,mode], alpha=.75, label=f'{mode}')
# #                 axesn[nod_ind].plot(t_vals_fine-0.0*deltat, np.sum(modal_imp_responses[:,node_i -1,:],axis=1), alpha=.75, label='modal sum')
# #
# #                 disp = self.static(((node_i,p_i),))[node_i-1,2]
# #                 k_glob = p_i/disp
# #                 axesn[nod_ind].plot(time_values, f[:,node_i-1]/k_glob, ls='dashed')
# #                 axesn[nod_ind].legend()
#
#             else:
#                 nod_ind = None
#                 print(f'Response at node {node_i} not available (not in meas_nodes), only sum of modal energies can be computed: {sum_energies}')
#
#         for nod_ind, node_i in enumerate(self.meas_nodes):
#             W=scipy.integrate.simps(response_time_history[1][:,nod_ind]*f[:,node_i-1],time_values, even='first')
#             print(f'Total impulse energy at node {node_i}: {W}. Sum of energies of all impulses: {np.sum(modal_imp_energies[node_i -1,:])} (Should match approximately)')

        self.inp_hist_imp = f
        self.t_vals_imp = time_values
        self.resp_hist_imp = response_time_history
        self.modal_imp_energies = modal_imp_energies
        self.modal_imp_amplitudes = modal_amplitudes

        # TODO: reduce modal_matrices to size (num_meas_nodes, num_modes)

        self.state[3] = True

        return time_values, response_time_history, f , modal_imp_energies, modal_amplitudes

    # def frequency_response(self, N, inp_node, dof, fmax=None, out_quant='a', modes=None):
    #     '''
    #     Returns the onesided FRF matrix of the linear(ized) system
    #     at N//2 + 1 frequency lines for all nodes in meas_nodes
    #     by default the accelerance with input force at the last node is returned
    #
    #     Uses numerically computed modal parameters and discrete system matrices
    #     The FRF may not be completely equivalent to analytical solutions
    #
    #     inp_node is the ANSYS node number -> index is corresponding to
    #         meas_nodes (if compensated) or
    #         nodes_coordinates if not compensated
    #
    #     dof is currently used for input and output,e.g. if input is in z direction, output cannot be in y direction
    #     '''
    #
    #     if not self.globdamp:
    #         logger.warning('This method assumes proporional damping. \
    #             System is non-proportionally damped. Results might be errorneous. \
    #             Consider using self.frequency_response_non_classical')
    #
    #     nodes_coordinates = self.nodes_coordinates
    #
    #     for i, (node, x, y, z) in enumerate(nodes_coordinates):
    #         if node == inp_node:
    #             inp_node_ind = i
    #             break
    #     else:
    #         raise RuntimeError(f'input node {inp_node} could not be found in nodes_coordinates')
    #
    #     dof_ind = ['ux', 'uy', 'uz'].index(dof)
    #
    #     # too complicated to get compensated (numerical damping, period elongation) modal_matrices, a marginal error will be accepted
    #     # _, _, mode_shapes, kappas, mus, etas = self.modal(modal_matrices=True)
    #     _, _, mode_shapes = self.modal()
    #     frequencies, damping,mode_shapes_n = self.numerical_response_parameters(compensate=True, dofs=[dof_ind])
    #     # we have to distinguish between input and output mode shapes
    #     # output mode shapes are generally only for meas_nodes,
    #     # while input mode shapes must be complete i.e. input node does not have to be in meas_nodes
    #
    #
    #     if fmax is None:
    #         fmax = np.max(frequencies)
    #
    #     df = fmax / (N // 2 + 1)
    #
    #     omegas = np.linspace(0, fmax, N // 2 + 1, False) * 2 * np.pi
    #     assert np.isclose(df * 2 * np.pi, (omegas[-1] - omegas[0]) / (N // 2 + 1 - 1))
    #     omegas = omegas[:, np.newaxis]
    #
    #     num_modes = self.num_modes
    #     num_meas_nodes = len(self.meas_nodes)
    #     omegans = frequencies * 2 * np.pi
    #
    #     frf = np.zeros((N // 2 + 1, num_meas_nodes), dtype=complex)
    #
    #     if modes is None:
    #         modes = range(num_modes)
    #
    #     for mode in modes:
    #
    #         omegan = omegans[mode]
    #         zeta = damping[mode]
    #         # kappa = kappas[mode]
    #         kappa = omegan**2 # modeshapes are mass-normalized
    #         # print(kappas[mode], mus[mode], etas[mode], omegans[mode], damping[mode])
    #         # if mode<2:
    #         #     zeta = zeta / 1
    #         #     kappa = kappa / np.sqrt(2)
    #         mode_shape = mode_shapes_n[:, mode]
    #         modal_coordinate = mode_shapes[inp_node_ind, dof_ind, mode]
    #         # TODO: extend 3D
    #
    #         this_frf = 1 / (kappa * (1 + 2 * 1j * zeta * omegas / omegan - (omegas / omegan)**2)) * modal_coordinate * mode_shape[np.newaxis, :]
    #         this_frf += 1 / (kappa * (1 + 2 * 1j * zeta * omegas / omegan - (omegas / omegan)**2)) * np.conj(modal_coordinate * mode_shape[np.newaxis, :])
    #
    #         frf += this_frf / 2
    #
    #     if out_quant == 'a':
    #         frf *=   -omegas**2
    #     elif out_quant == 'v':
    #         frf *= 1j*omegas
    #     elif out_quant == 'd':
    #         ...
    #     else:
    #         logger.warning(f'This output quantity is invalid: {out_quant}')
    #
    #     self.omegas = omegas
    #     self.frf= frf
    #
    #     self.state[8] = True
    #
    #     return omegas, frf

    def frequency_response_non_classical(self, N, inp_nodes, inp_dofs,
                                         use_meas_nodes=True, out_dofs=['ux', 'uy', 'uz'],
                                         fmax=None, out_quant='a', modes=None, **kwargs):
        '''
        As in Brincker & Ventura: Introduction to Operational Modal Analysis, p. 99 ff
        
        How would we want to return our FRF?
        It should be limited so some nodes of interest to save computational time
        It should be limited to one or more degrees of freedom
        adopt the procedure taken in numerical_response_parameters to get (len(meas_nodes)*len(dof_ind),:num_modes)
        dof_ref should be returned alongside the output
        
        
        '''
        assert isinstance(inp_nodes, (list, tuple))
        assert isinstance(inp_dofs, (list, tuple))

        nodes_coordinates = self.nodes_coordinates

        if not use_meas_nodes:
            logger.warning('Argument use_meas_nodes is not supported (True by default)')
        num_modes = kwargs.pop('num_modes', self.num_modes)
        if isinstance(out_dofs, str):
            out_dofs = [out_dofs]
        n_out_dof = len(out_dofs)
        n_inp_dof = len(inp_dofs)

        _, _, _, gen_mod_coef = self.modal(num_modes=num_modes, modal_matrices=True)
        lamda = self.lamda
        omegans = np.imag(lamda)  # * 2 * np.pi

        if fmax is None:
            fmax = np.max(omegans) / 2 / np.pi
        if not np.max(omegans) / 2 / np.pi <= fmax:
            raise ValueError(f'The provided Nyquist frequency (={fmax}) is below the highest natural frequency to be used (={np.max(omegans) / 2 / np.pi})')
        df = fmax / (N // 2 + 1)
        # omegas = np.linspace(0, fmax, N // 2 + 1, False) * 2 * np.pi
        # assert np.isclose(df * 2 * np.pi, (omegas[-1] - omegas[0]) / (N // 2 + 1 - 1))
        omegas = np.fft.rfftfreq(N, 1 / (2 * fmax)) * 2 * np.pi
        assert np.isclose(df * 2 * np.pi, (omegas[-1] - omegas[0]) / (N // 2 + 1))
        # omegas = omegas[:, np.newaxis, np.newaxis]

        # Assemble output mode shapes
        out_dofs = Mechanical.dofs_str_to_ind(out_dofs)
        meas_nodes = self.meas_nodes
        n_nod = len(meas_nodes)
        _, _, mode_shapes_out = self.numerical_response_parameters(compensate=True, dofs=out_dofs)

        dof_ref_out = np.empty((len(meas_nodes) * n_out_dof, 2))
        for i in range(n_out_dof):
            dof_ref_out[i * n_nod:(i + 1) * n_nod, 0] = meas_nodes
            dof_ref_out[i * n_nod:(i + 1) * n_nod, 1] = out_dofs[i]

        # Assemble input mode shapes
        inp_dofs = Mechanical.dofs_str_to_ind(inp_dofs)
        n_nod = len(inp_nodes)
        _, _, mode_shapes_full = self.modal(damped=True, num_modes=num_modes)

        inp_indices = []
        for inp_node in inp_nodes:
            for i, (node, _, _, _) in enumerate(nodes_coordinates):
                if node == inp_node:
                    inp_indices.append(i)
                    break
            else:
                raise RuntimeError(f'inp_node {inp_node} could not be found in nodes_coordinates')

        mode_shapes_inp = np.full((len(inp_nodes) * n_inp_dof, num_modes), np.nan, dtype=complex)
        for mode in range(num_modes):
            for i, dof in enumerate(inp_dofs):
                mode_shapes_inp[len(inp_nodes) * i:len(inp_nodes) * (i + 1), mode] = mode_shapes_full[inp_indices, dof, mode]

        dof_ref_inp = np.empty((len(inp_nodes) * n_inp_dof, 2))
        for i in range(n_inp_dof):
            dof_ref_inp[i * n_nod:(i + 1) * n_nod, 0] = inp_nodes
            dof_ref_inp[i * n_nod:(i + 1) * n_nod, 1] = inp_dofs[i]

        logger.info(f'FRF computation for non-classical modes with {N // 2 + 1} frequency lines in the frequency range up to {fmax} Hz and {num_modes} modes for {dof_ref_inp.shape[0]} input and {dof_ref_out.shape[0]} output nodes.')

        # inp_node_ind = np.logical_and(dof_ref_out[:,0] == inp_nodes, dof_ref_out[:,1] == dof_ind)
        if kwargs.get('fname_mmap', False):
            fname = kwargs['fname_mmap']
            logger.warning("Using a memory map for the frf array slows down parallel (numba) computation.")
            frf = np.memmap(fname, dtype=np.complex64, mode='w+', shape=(N // 2 + 1, dof_ref_inp.shape[0], dof_ref_out.shape[0]))
            # frf[:,:,:] = 0
        else:
            frf = np.zeros((N // 2 + 1, dof_ref_inp.shape[0], dof_ref_out.shape[0]), dtype=np.complex64)

        if modes is None:
            modes = range(num_modes)

        in_out = np.empty((dof_ref_inp.shape[0], dof_ref_out.shape[0]), dtype=complex)
        in_out_conj = np.empty((dof_ref_inp.shape[0], dof_ref_out.shape[0]), dtype=complex)
        omega_scale = np.empty((N // 2 + 1,), dtype=complex)
        omega_scale_conj = np.empty((N // 2 + 1,), dtype=complex)

        from numba import njit, prange, set_num_threads
        set_num_threads(18)

        @njit(parallel=True)
        def parallel_freq(frf, omega_scale, in_out, omega_scale_conj, in_out_conj):

            for i in prange(N // 2 + 1):
                # most expensive part
                # for i in range((N // 2 + 1) * j, (N // 2 + 1) * (j + 1)):
                frf[i,:,:] += omega_scale[i] * in_out + omega_scale_conj[i] * in_out_conj
            return True

        from alive_progress import alive_bar
        with  alive_bar(len(modes) , force_tty=True) as pbar:
            for mode in modes:
                lambda_n = lamda[mode]

                mode_shape_out = mode_shapes_out[np.newaxis,:, mode]  # complex vector (num_nodes,)
                mode_shape_inp = mode_shapes_inp[:, np.newaxis, mode]  # complex vector (num_nodes,)

                a_n = gen_mod_coef[mode]
                in_out[:,:] = (mode_shape_inp * mode_shape_out) / a_n
                in_out_conj[:,:] = np.conj(in_out)

                omega_scale[:] = (1 / (1j * omegas - lambda_n))
                omega_scale_conj[:] = (1 / (1j * omegas - np.conj(lambda_n)))
                parallel_freq(frf, omega_scale, in_out, omega_scale_conj, in_out_conj)
                # for i in range(N // 2 + 1):
                #     # most expensive part
                #     frf[i,:,:] += omega_scale[i]      *      in_out + omega_scale_conj[i] * in_out_conj

                pbar()

        if out_quant == 'a':
            frf *= -omegas[:, np.newaxis, np.newaxis] ** 2
        elif out_quant == 'v':
            frf *= 1j * omegas[:, np.newaxis, np.newaxis]
        elif out_quant == 'd':
            ...
        else:
            logger.warning(f'This output quantity is invalid: {out_quant}')

        self.omegas = omegas
        self.frf = frf
        if isinstance(frf, np.memmap):
            frf.flush()
        self.dof_ref_out = dof_ref_out
        self.dof_ref_inp = dof_ref_inp

        self.state[8] = True

        return omegas, frf, dof_ref_out, dof_ref_inp

    def modal_ext(self, damped=True, num_modes=None):  # Modal Analysis
        pyansys = _get_pyansys()
        ansys = self.ansys

        print('External (in python) computation of modal parameters using the state-space formulation')
        num_nodes = self.num_nodes
        if num_modes is None:
            num_modes = self.num_modes
        assert num_modes <= num_nodes
        if num_modes > 10 * num_nodes:
            logger.warning(f'The number of modes {num_modes} should be greater/equal than 10 number of nodes {num_nodes}.')

        ansys.run('/SOL')
        ansys.antype('MODAL')

        ansys.outres(item='ERASE')
        ansys.outres(item='ALL', freq='NONE')  # Disable all output
        ansys.nsel(type='ALL')

        ansys.outres(item='NSOL',
                     freq='ALL'
                     )  # Controls the solution data written to the database.
        if damped:
            if not self.globdamp:
                logger.warning("Model is non-proportionally damped. Modal damping values may be errorneous.")
            ansys.modopt(method='QRDAMP', nmode=num_modes, freqb=0,
                         freqe=1e4,
                         cpxmod='cplx',
                         nrmkey='off',
                         )
        else:
            ansys.modopt(method='LANB', nmode=num_modes, freqb=0,
                         freqe=1e4,
                         nrmkey='off',
                         )

        ansys.mxpand(nmode='all', elcalc=1)
        ansys.wrfull()

        full_path = os.path.join(ansys.directory, ansys.jobname + '.full')
        full = pyansys.read_binary(full_path)
        dof_ref, k, m, c = full.load_kmc(as_sparse=False, sort=False)

        k += np.triu(k, 1).T
        m += np.triu(m, 1).T
        c += np.triu(c, 1).T

        # remove all-zero rows and columns
        mask_0 = ~np.all(k == 0, axis=0)
        mask_1 = ~np.all(k == 0, axis=1)

        k_ = k[mask_0,:]
        m_ = m[mask_0,:]
        c_ = c[mask_0,:]
        k_ = k_[:, mask_1]
        m_ = m_[:, mask_1]
        c_ = c_[:, mask_1]

        dof_ref = dof_ref[mask_1,:]

        # assemble state-space matrices as in Brincker and Ventura, Eq.5.103
        o = np.zeros_like(k_)
        A = np.vstack([np.hstack([ o, m_]),
                       np.hstack([ m_, c_])])
        B = np.vstack([np.hstack([-m_, o ]),
                       np.hstack([ o, k_])])

        # solve Eigenvalue problem
        w, v = scipy.linalg.eig(-B, A)

        # compute omega
        omegans = np.imag(w)

        # sort by ascending frequencies
        sort_ind = np.argsort(omegans)
        omegans = omegans[sort_ind]
        # remove complex conjugates and remove all modes higher than num_modes
        conj_ind = omegans > 0
        lamda = w[sort_ind][conj_ind][:num_modes]
        phi = v[v.shape[0] // 2:, sort_ind][:, conj_ind][:,:num_modes]

        for mode in range(num_modes):
            this_phi = phi[:, mode]
            mu = this_phi.T @ m_ @ np.conj(this_phi)
            phi[:, mode] /= np.sqrt(mu)

        return k_, m_, c_, lamda, phi, dof_ref

    @staticmethod
    def dofs_str_to_ind(dofs):
        strdof_list = ['ux', 'uy', 'uz', 'rx', 'ry', 'rz']
        dof_inds = []
        for dof in dofs:
            if dof in strdof_list:
                dof_ind = strdof_list.index(dof)
                dof_inds.append(dof_ind)
            elif dof > 6:
                dof_inds.append(dof)
        return dof_inds

    # def ambient_ifrf(self, f_scale, deltat=None, dt_fact=None, timesteps=None, num_cycles=None, out_quant=['d', 'v', 'a'], dofs=['ux', 'uy', 'uz'], seed=None, **kwargs):
    #     '''
    #     a shortcut function for ambient using the linear frf and the
    #     inverse FFT to generate the signal by orders of magnitude faster
    #     non-linear effects, or non-white noise inputs are not possible
    #     '''
    #     num_nodes = self.num_nodes
    #
    #     dt_fact, deltat, num_cycles, timesteps, _ = self.signal_parameters(dt_fact, deltat, num_cycles, timesteps)
    #
    #     logger.info(f"Generating ambient response (IFFT/FRF-based) in frequency range [0 ... {self.frequencies[-1]:1.3f}], zeta range [{np.min(self.modal_damping):1.3f}-{np.max(self.modal_damping):1.3f}], number of modes {self.num_modes}")
    #
    #     logger.warning("WARNING: Gaussian Noise Generator has not been verified!!!!!!")
    #
    #     meas_nodes = self.meas_nodes
    #     input_nodes = [node for node, x, y, z in self.nodes_coordinates[-2:-1]]
    #
    #     rng = np.random.default_rng(seed)
    #     phase = rng.uniform(-np.pi, np.pi, (timesteps // 2 + 1, num_nodes))
    #     Pomega = f_scale * np.ones_like(phase) * np.exp(1j * phase)
    #
    #     response_time_history = [None, None, None]
    #
    #     for quant_ind, quant in enumerate(['d', 'v', 'a']):
    #         if quant in out_quant:
    #             sig = np.zeros((timesteps, len(meas_nodes), 3))
    #             # compute ifft for each combination of input and output node
    #             # use linear superposition of output signals from each input node
    #
    #             for inp_node_ind, inp_node in enumerate(input_nodes):
    #                 for dof_ind, dof in enumerate(['ux', 'uy', 'uz']):  # use all dofs in order to comply with the output of a transient simulation
    #                     if dof in dofs:
    #                         _, this_frf = self.frequency_response(N=timesteps, inp_node=inp_node, dof=dof,
    #                                                               fmax=1 / deltat / 2, out_quant=quant, )
    #                         for channel in range(this_frf.shape[1]):
    #                             sig[:, channel, dof_ind] += np.fft.irfft(this_frf[:, channel] * Pomega[:, inp_node_ind])
    #             response_time_history[quant_ind] = sig
    #
    #     time_values = np.linspace(deltat, timesteps * deltat, timesteps) #  ansys also starts at deltat
    #
    #     self.inp_hist_amb = None
    #     self.t_vals_amb = time_values
    #     self.resp_hist_amb = response_time_history
    #
    #     self.state[2] = True
    #
    #     return time_values, response_time_history, None

    def static(self, fz=None, uz=None, use_meas_nodes=False):
        '''
        provide fz/uz as a numpy array of [node, displacement] pairs
        assumes UZ DOF, may have to be changed in the future
        constraints are restored after forced-displacement analysis, currently hardcoded
        
        TODO: extend 3D
        '''
        pyansys = _get_pyansys()

        if fz is not None:
            if not isinstance(fz, np.ndarray):
                fz = np.array(fz)
            assert fz.shape[1] == 2

        if uz is not None:
            assert isinstance(uz, np.ndarray)
            assert uz.shape[1] == 2

        ansys = self.ansys

        # Static Response
        ansys.slashsolu()
        ansys.antype('STATIC')

        ansys.nsel(type='ALL')

        if fz is not None:
            for node, fz_ in fz:
                ansys.f(node=node, lab='FZ', value=fz_)

        if uz is not None:
            for node, uz_ in uz:
                ansys.d(node=node, lab='UZ', value=uz_)

        ansys.outres(item='ERASE')

        if use_meas_nodes:
            ansys.outres(item='ALL', freq='NONE')
            ansys.outres(item='NSOL', freq='LAST', cname='MEAS_NODES')

        ansys.solve()

        if fz is not None:
            ansys.fdele(node='ALL', lab='ALL')

        if uz is not None:
            # remove all prescribed displacements including constraints
            ansys.ddele(node='ALL', lab='ALL')
            # restore constraints
            ansys.d(node=self.nodes_coordinates[0][0], value=0, lab='UX', lab2='UY', lab3='UZ')  # ,lab4='RX', lab5='RY',lab6='RZ')
            for node, x, y, z in self.nodes_coordinates:
                ansys.d(node=node, value=0, lab='UX', lab2='UY')

        ansys.finish()

        # self.last_analysis = 'static'
        res = pyansys.read_binary(os.path.join(ansys.directory, ansys.jobname + '.rst'))

        nodes, disp = res.nodal_solution(0)
#         print(nodes,disp)

        return disp

    def modal(self, damped=True, num_modes=None, use_cache=True, reset_sliders=True, modal_matrices=False, use_meas_nodes=False):  # Modal Analysis
        pyansys = _get_pyansys()
        ansys = self.ansys

        ret_vals = super().modal(damped, num_modes, use_cache, modal_matrices)
        if ret_vals is not None:
            return ret_vals

        if num_modes is None:
            num_modes = self.num_modes

        ansys.prep7()
        if self.coulomb_elements and reset_sliders:
            logger.info("Temporarily resetting sliders for modal analysis.")
            real_constants = []
            # reset the sliding capabilities (k1,fslide) for modal analysis,
            # else ktot=k2+k1 would be taken
            for coulomb in self.coulomb_elements:
                if coulomb is None: continue
                nset = coulomb[1]
                # print(ansys.rlist(nset))
                ansys.get('k_coul', entity="rcon", entnum=nset, item1=1,)  # r5=f_sl_in, r6=k_2
                ansys.get('f_coul', entity="rcon", entnum=nset, item1=5,)  # r5=f_sl_in, r6=k_2
                # ansys.load_parameters()
                real_constants.append((ansys.parameters["K_COUL"], ansys.parameters["F_COUL"]))
                # print(real_constants)
                ansys.rmodif(nset, 1, 0)
                ansys.rmodif(nset, 5, 0)
                # print(ansys.rlist(nset))
                # print(real_constants[-1])
        ansys.finish()

        ansys.run('/SOL')
        ansys.antype('MODAL')

        ansys.outres(item='ERASE')
        ansys.outres(item='ALL', freq='NONE')  # Disable all output
        # ansys.nsel(type='S', item='NODE', vmin=20,vmax=20) # select only nodes of interest
        # ansys.nsel(type='A', item='NODE', vmin=2,vmax=2) # select only nodes of interest
        # ansys.nsel(type='ALL')

        ansys.nsel(type='ALL')
        # if modal_matrices and use_meas_nodes:
        #     logger.warning("Ignoring requested subset of measurement nodes: when computing modal matrices, the full set of nodes is required.")
        #     use_meas_nodes = False
        # if use_meas_nodes:
        #     ansys.outres(# item='A',
        #              item='NSOL',
        #              freq='ALL'
        #              ,cname='meas_nodes'# for modal matrices we need the full mode shapes
        #              )  # Controls the solution data written to the database.
        # else:
        ansys.outres(# item='A',
                     item='NSOL',
                     freq='ALL'
                    # ,cname='meas_nodes'# for modal matrices we need the full mode shapes
                     )  # Controls the solution data written to the database.
        if damped:
            if not self.globdamp:
                logger.warning("Model is undamped or non-proportionally damped. Modal damping values may be errorneous.")
            ansys.modopt(method='QRDAMP', nmode=num_modes, freqb=0,
                         freqe=1e4,
                         cpxmod='cplx',
                         nrmkey='off',
                         )
#             ansys.modopt(method='DAMP',nmode=num_modes,freqb=0,
#                          freqe=1e8,
#                          cpxmod='cplx',
#                          nrmkey='on',
#                          )
        else:
            ansys.modopt(method='LANB', nmode=num_modes, freqb=0,
                         freqe=1e4,
                         nrmkey='off',
                         )

        ansys.mxpand(nmode='all', elcalc=1)
        ansys.solve()

        # self.last_analysis = 'modal'

        res = pyansys.read_binary(os.path.join(ansys.directory, ansys.jobname + '.rst'))

        num_modes_ = res.nsets
        if res._resultheader['cpxrst']:  # real and imaginary parts are saved as separate sets
            num_modes_ //= 2
        if num_modes_ != num_modes:
            logger.warning(f'The number of numerical modes {num_modes_} differs from the requested number of modes {num_modes}.')
            num_modes = num_modes_

        nnodes = res._resultheader['nnod']
        # print(nnodes, self.num_nodes)
        # print(res._resultheader['neqv'])
        # assert nnodes == self.num_nodes
        ndof = res._resultheader['numdof']

        mode_shapes = np.full((nnodes, ndof, num_modes), (1 + 1j) * np.nan, dtype=complex)
        frequencies = np.full(num_modes, np.nan)
        damping = np.full(num_modes, np.nan)

        lamda = np.full((num_modes,), np.nan + 1j * np.nan, dtype=complex)
        # print(res.time_values)
        if res._resultheader['cpxrst']:
            # print('damped')
            for mode in range(num_modes):
                # print(res.time_values[2*mode:2*mode+1])
                sigma = res.time_values[2 * mode]  # real part
                omega = res.time_values[2 * mode + 1]  # imaginary part
                if omega < 0: continue  # complex conjugate pair
                # print(omega)
                lamda[mode] = (sigma + 1j * omega) * 2 * np.pi

                frequencies[mode] = omega  # damped frequency
                damping[mode] = -sigma / np.sqrt(sigma ** 2 + omega ** 2)

                mode_shapes[:,:, mode].real = res.nodal_solution(2 * mode)[1]
                mode_shapes[:,:, mode].imag = res.nodal_solution(2 * mode + 1)[1]
            else:
                nnum = res.nodal_solution(0)[0]

        else:
            # print('Undamped')
            frequencies[:] = res.time_values
            for mode in range(num_modes):
                nnum, modal_disp = res.nodal_solution(mode)
                mode_shapes[:,:, mode] = modal_disp
                # print(frequencies[mode])
            mode_shapes = mode_shapes.real
            lamda = None

        # self.msh_ans_rst = np.copy(mode_shapes)
        self.lamda = lamda

        # reduce mode shapes to meas_nodes and translational dof
        if use_meas_nodes:
            _, meas_indices = np.where(self.meas_nodes[:, None] == nnum)
            mode_shapes = mode_shapes[meas_indices,:3,:]
        else:
            # account for internal element nodes by reducing length
            mode_shapes = mode_shapes[nnum <= self.num_nodes,:3,:]

        if modal_matrices:

            ansys.wrfull()
            ansys.finish()

            ansys.aux2()
            '''
            APDL Guide 4.4
            The mode shapes from the .MODE file and the DOF results from 
            the .RST file are in the internal ordering, and they need 
            to be converted before use with any of the matrices from the .FULL file, as shown below:
            '''
            ansys.smat(matrix='Nod2Solv', type='D', method='IMPORT', val1='FULL', val2=f"{self.jobname}.full", val3="NOD2SOLV")
            # Permutation operators can not be exported, so the mode shapes must be redordered to solver ordering in APDL
            if damped:
                ansys.dmat(matrix="PhiI", type="Z", method="IMPORT", val1="MODE", val2=f"{self.jobname}.mode")
            else:
                ansys.dmat(matrix="PhiI", type="D", method="IMPORT", val1="MODE", val2=f"{self.jobname}.mode")
            ansys.mult(m1='Nod2Solv', t1='', m2='PhiI', t2='', m3='PhiB')
            ansys.export(matrix="PhiB", format="MMF", fname="PhiB.bin")
            msh = np.array(scipy.io.mmread('PhiB.bin'))
            os.remove('PhiB.bin')

            ansys.smat(matrix="MatKD", type="D", method="IMPORT", val1="FULL", val2=f"{self.jobname}.full", val3="STIFF")
            ansys.export(matrix="MatKD", format="MMF", fname="MatKD.bin")
            K = scipy.io.mmread('MatKD.bin').toarray()
            os.remove('MatKD.bin')
            ansys.smat(matrix="MatMD", type="D", method="IMPORT", val1="FULL", val2=f"{self.jobname}.full", val3="MASS")
            ansys.export(matrix="MatMD", format="MMF", fname="MatMD.bin")
            M = scipy.io.mmread('MatMD.bin').toarray()
            os.remove('MatMD.bin')
            # try:
            if damped:
                ansys.dmat(matrix="MatCD", type="D", method="IMPORT", val1="FULL", val2=f"{self.jobname}.full", val3="DAMP")
                ansys.export(matrix="MatCD", format="MMF", fname="MatCD.bin")
                C = scipy.io.mmread('MatCD.bin')
                os.remove('MatCD.bin')
            # except Exception as e:
            else:
                # print(e)
                C = np.zeros_like(K)

            # compute modal matrices
            # kappas = np.zeros((num_modes))
            # mus = np.zeros((num_modes))
            # etas = np.zeros((num_modes))
            # generalized complex modal coefficients (Brincker and Ventura, Eq. 5.112)
            gen_mod_coeff = np.zeros((num_modes), dtype=complex)

            for mode in range(num_modes):

                # TODO: should work, since I assume K, M and C are 3D
                # properly remove constraint nodes
                # check complex conjugate?
                msh_f = msh[:, mode]
                lamda_n = lamda[mode]

                # unit normalization would have to be done for returned mode shapes
                # as well, otherwise subsequent calculations may be errorneous (e.g. frfs)
                if False:
                    msh_f /= msh_f[np.argmax(np.abs(msh_f))]

                # kappas[mode]= (msh_f.T @ K @ msh_f.conj()).real
                # mus[mode]   = (msh_f.T @ M @ msh_f.conj()).real
                # etas[mode]  = (msh_f.T @ C @ msh_f.conj()).real
                gen_mod_coeff[mode] = msh_f.T @ M @ msh_f * 2 * lamda_n
                gen_mod_coeff[mode] += msh_f.T @ C @ msh_f

        ansys.finish()

        ansys.prep7()
        if self.coulomb_elements and reset_sliders:
            for coulomb, real_constant in zip(self.coulomb_elements, real_constants):
                nset = coulomb[1]
                ansys.rmodif(nset, 1, real_constant[0])
                ansys.rmodif(nset, 5, real_constant[1])
        ansys.finish()

        self.state[4] = True

        if damped:
            self.damped_frequencies = frequencies
            self.modal_damping = damping
            self.damped_mode_shapes = mode_shapes
        else:
            self.frequencies = frequencies
            self.mode_shapes = mode_shapes
        self.num_modes = num_modes

        if modal_matrices:
            # self.kappas = kappas
            # self.mus = mus
            # self.etas = etas
            self.gen_mod_coeff = gen_mod_coeff
            # print(mode_shapes)
            # print(f'{frequencies} {damping} {np.sqrt(kappas/mus)/2/np.pi*np.sqrt(1-damping**2)} {etas/2/np.sqrt(mus*kappas)}')
            return frequencies, damping, mode_shapes, gen_mod_coeff
        else:
            return frequencies, damping, mode_shapes

    def transient_parameters(self, meth='NMK', parameter_set=None, **kwargs):

        delta = None  # gamma
        alpha = None  # beta
        alphaf = None  # alphaf
        alpham = None  # alpham

        if meth == 'NMK':
            tintopt = 'NMK'
            if isinstance(parameter_set, tuple):
                assert len(parameter_set) == 2
                delta, alpha = parameter_set
            elif isinstance(parameter_set, str):
                if parameter_set == 'AAM':  # Trapezoidal Rule -> Constant Acceleration
                    delta = 1 / 2
                    alpha = 1 / 4
                elif parameter_set == 'LAM':  # Original Newmark 1/6 -> Linear Acceleration
                    delta = 1 / 2
                    alpha = 1 / 6
                else:
                    raise RuntimeError(f"Could not understand parameter set {parameter_set}.")
            elif isinstance(parameter_set, (float, int)):
                rho_inf = parameter_set
                delta = (3 - rho_inf) / (2 * rho_inf + 2)
                alpha = 1 / ((rho_inf + 1) ** 2)
            elif parameter_set is None:
                delta = 1 / 2
                alpha = 1 / 4
            else:
                raise RuntimeError(f"Could not understand parameter set {parameter_set}.")

        elif meth in ['HHT', 'WBZ', 'G-alpha']:
            tintopt = 'HHT'
            if isinstance(parameter_set, (float, int)):
                rho_inf = parameter_set
            elif parameter_set is None:
                logger.info("Spectral radius was not supplied. Using 1")
                rho_inf = 1
            else:
                assert len(parameter_set) == 4
                delta, alpha, alphaf, alpham = parameter_set

            if meth == 'HHT':  # HHT-α Hilber-Hugh Taylor
                alphaf = np.round((1 - rho_inf) / (rho_inf + 1), 8)
                if alphaf > 0.5: raise RuntimeError("ANSYS won't take alphaf>0.5 in the HHT method")
                alpham = 0
                delta = np.round(1 / 2 + alphaf, 8)
                alpha = np.round((1 + alphaf) ** 2 / 4, 9)
            elif meth == 'WBZ':  # WBZ-α Wood-Bosak-Zienkiewicz
                alphaf = 0
                alpham = np.round((rho_inf - 1) / (rho_inf + 1), 8)
                delta = np.round(1 / 2 - alpham, 8)
                alpha = np.round((1 - alpham) ** 2 / 4, 8)
            elif meth == 'G-alpha':
                alpham = np.round((2 * rho_inf - 1) / (rho_inf + 1), 8)
                alphaf = np.round(rho_inf / (rho_inf + 1), 8)
                delta = np.round(1 / 2 - alpham + alphaf, 8)
                alpha = np.round(1 / 4 * (1 - alpham + alphaf) ** 2, 8)
        else:
            raise ValueError(f'Method provided could not be understood: {meth}')

        tintopt = 'HHT'

        logger.debug(f'Transient analysis method {meth} with parameter set {parameter_set} -> gamma {delta}, beta {alpha}, alpha_f {alphaf}, alpha_m {alpham}')

        self.trans_params = (delta, alpha, alphaf, alpham)

        return delta, alpha, alphaf, alpham, tintopt

    def signal_parameters(self, dt_fact=None, deltat=None, num_cycles=None, timesteps=None, mode_index=None):
        # Setup transient simulation parameters

        assert deltat is not None or dt_fact is not None
        assert timesteps is not None or num_cycles is not None

        frequencies, damping, _ = self.modal(damped=True)

        if mode_index is None:
            f_min = frequencies[0]
            f_max = frequencies[-1]
            mode_index = self.num_modes - 1
        else:
            f_min = frequencies[mode_index]
            f_max = frequencies[mode_index]

        if dt_fact is None:
            deltat = np.round(deltat, 8)  # round deltat: in transient stepsize*deltat = loadstep end time.  ansys computes stepsize back from loadsteptime and deltat with finite precision, thus inconsitencies occur
            dt_fact = deltat * f_max
        elif deltat is None:
            deltat = dt_fact / f_max
            deltat = np.round(deltat, 8)

        if timesteps is None:
            # ensure timesteps is a multiple of 2 to avoid problems in any fft based processing
            timesteps = int(np.ceil(num_cycles / f_min / deltat) // 2) * 2
        elif num_cycles is None:
            if timesteps % 2:
                logging.warning(f'Timesteps is {timesteps}, which is not a multiple of 2. In order to avoid problems in FFT-based processing, setting it to {timesteps +1}')
                timesteps += 1
            num_cycles = int(np.floor(timesteps * f_min * deltat))

        Omega = deltat * frequencies * 2 * np.pi
#         print(Omega)
        if (Omega > 0.1).any():
            if self.trans_params is not None:
                #  compute the bifurcation limits for all modes
                gamma, beta, alpha_f, alpha_m = self.trans_params
                if alpha_f is None and alpha_m is  None:
                    Omega_bif = (-0.5 * damping * (-gamma + 0.5) + np.sqrt(damping ** 2 * (beta - 0.5 * gamma) - beta + 0.25 * (gamma + 0.5) ** 2)) / (0.25 * (gamma + 0.5) ** 2 - beta)
                    if (Omega > Omega_bif).any():
                        warnings.warn(f'Omega exceeds bifurcation limit for modes {np.where(Omega>Omega_bif)[0]+1}')
            # timestep, where decay is down to 1e-8
            else:
                t_crit = -1 * np.log(1e-8) / damping[-1] / f_max / 2 / np.pi
                if t_crit <= timesteps * deltat:
                    logger.warning(f'For conditionally stable integration, convergence issues may appear in free-decay mode due to numerical precision leaking into higher modes: {np.where((deltat*frequencies*2*np.pi)>0.1)[0]+1}.')

        self.deltat = deltat
        self.timesteps = timesteps
        logger.info(f"Signal parameters for upcoming transient: deltat {deltat:1.6f}, dt_fact for f_max {dt_fact:1.6f}, timesteps {timesteps}, num_cycles for f_min {num_cycles}")

        return dt_fact, deltat, num_cycles, timesteps, mode_index

    def transient(self, fy=None, fz=None, d=None,
                  timint=1, deltat=None, timesteps=None,
                  out_quant=['d', 'v', 'a'],
                  chunksize=10000, chunk_restart=False, **kwargs):
        pyansys = _get_pyansys()
        ansys = self.ansys

        chunksize = int(chunksize)
        num_chunks = timesteps // chunksize
        if num_chunks > 5 and not chunk_restart:
            logger.info(f"{num_chunks}>5 chunks will be computed. Enabling chunk_restart for efficiency.")
            chunk_restart = True
            ansys.config("NRES", chunksize + 1)
            # sometimes, an additional loadstep is computed, probably due to rounding errors in deltat,
            # setting nres a little higher, avoids failure on  ANTYPE, REST

        # TODO:: Ensure no numerical damping is needed, even if only a subset of modes is to be used later
        delta, alpha, alphaf, alpham, tintopt = self.transient_parameters(**kwargs)

        if deltat is not None and timesteps is not None:
            # check user provided signal parameters
            _, deltat, _, timesteps, _ = self.signal_parameters(deltat=deltat, timesteps=timesteps)
        elif deltat is None and timesteps is not None:
            deltat = self.deltat
            # check user provided signal parameters
            _, deltat, _, timesteps, _ = self.signal_parameters(deltat=deltat, timesteps=timesteps)
        if timesteps is None and deltat is not None:
            timesteps = self.timesteps
            # check user provided signal parameters
            _, deltat, _, timesteps, _ = self.signal_parameters(deltat=deltat, timesteps=timesteps)
        else:
            deltat = self.deltat
            timesteps = self.timesteps

        ansys.slashsolu()
        ansys.nsel(type='ALL')
        ansys.antype('TRANS')
#         ansys.set_log_level("INFO")
        ansys.trnopt(method='FULL', tintopt=tintopt)  # bug: vaout should be tintopt

        if chunk_restart:
            # ansys.run("RESCONTROL, DEFINE, LAST, LAST, -1, , 3") # MAXTotalFiles is not implemented in pyansys rescontrol, let's see, if that stops .ldhi files from growing infinitely
            ansys.rescontrol(action='DEFINE', ldstep='NONE', frequency='LAST')  # Controls file writing for multiframe restarts
        else:
            ansys.rescontrol(action='DEFINE', ldstep='NONE', frequency='NONE', maxfiles=-1)  # Controls file writing for multiframe restarts

        ansys.kbc(1)  # Specifies ramped or stepped loading within a load step.
        ansys.timint(timint)  # Turns on transient effects.

        ansys.tintp(alpha=alpha if alpha is not None else '',
                    delta=delta if delta is not None else '',
                    alphaf=alphaf if alphaf is not None else '',
                    alpham=alpham if alpham is not None else '',
                    oslm="", tol="", avsmooth='')

        ansys.autots('off')
        ansys.nsubst(1)

        ansys.outres(item='ERASE')
        ansys.outres(item='ALL', freq='NONE')  # Disable all output, must be here, else everything, not just meas_nodes will be written to db

        t_start = time.time()

        # according to ansys structural analysis guide to achieve zero initial velocity
        if d is not None:
            ansys.timint('off')
            for i, (node, x, y, z) in enumerate(self.nodes_coordinates):
                if not i: continue  # skip first node, to not change constraints
                ansys.d(node=node, lab='UZ', value=d[0, i])
            ansys.time(deltat / 2)
            ansys.nsubst(2)
            ansys.solve()
            ansys.nsubst(1)
            ansys.timint(timint)
            for i, (node, x, y, z) in enumerate(self.nodes_coordinates):
                if not i: continue  # skip first node
                ansys.ddele(node=node, lab='UZ')

        if 'd' in out_quant:
            ansys.outres(item='NSOL', freq='ALL', cname='MEAS_NODES')

        if 'a' in out_quant:
            ansys.outres(item='A', freq='ALL', cname='MEAS_NODES')

        if 'v' in out_quant:
            ansys.outres(item='V', freq='ALL', cname='MEAS_NODES')

#         ansys.set_log_level("ERROR")
        t_end = t_start
        t_start = time.time()
        logger.debug(f'setup  in {t_start-t_end} s')

        # if chunk_restart:
            # out_a = []
            # out_v = []
            # out_d = []
            # out_t = []

        # pre-allocate arrays to avoid memory errors after all has been computed
        if "a" in out_quant:
            all_disp_a = np.zeros((timesteps, len(self.meas_nodes), 3))
        else:
            all_disp_a = None

        if "v" in out_quant:
            all_disp_v = np.zeros((timesteps, len(self.meas_nodes), 3))
        else:
            all_disp_v = None

        if "d" in out_quant:
            all_disp_d = np.zeros((timesteps, len(self.meas_nodes), 3))
        else:
            all_disp_d = None

        time_values = np.zeros((timesteps,))

        # make sure time series start at t=deltat, a previous solve was done at t=deltat/2, and a constant deltim would shift everything by deltat/2
        # solve for consistent accelerations
        if d is not None:
            ansys.time(deltat)
#             print(deltat)
            ansys.solve()

        pid = os.getpid()
        truncated_fds = []

        for chunknum in range(timesteps // chunksize + 1):

            if (chunknum + 1) * chunksize <= timesteps:
                stepsize = chunksize
            else:
                stepsize = timesteps % chunksize
            if stepsize == 0:
                break
            # print(chunknum, timesteps//chunksize, timesteps%chunksize, stepsize, chunksize, timesteps, (chunknum+1)*chunksize)
            if fy is not None:
                table = np.zeros((stepsize + 1, self.num_nodes + 1))
                table[1:, 0] = np.arange(chunknum * chunksize + 1, chunknum * chunksize + 1 + stepsize) * deltat
                table[0, 1:] = np.arange(1, self.num_nodes + 1)
                table[1:, 1:] = fy[chunknum * chunksize:chunknum * chunksize + stepsize,:]
                fname = os.path.join(ansys.directory, ansys.jobname + '_y.csv')
                np.savetxt(fname, table)

                with supress_logging(ansys):
                    ansys.starset(par='EXCITATIONY')

                ansys.dim(par='EXCITATIONY', type='TABLE', imax=stepsize, jmax=self.num_nodes, kmax="", var1='TIME', var2='NODE')
                ansys.tread(par='EXCITATIONY', fname=f'{ansys.jobname}_y', ext='csv')

                ansys.f(node='ALL', lab='FY', value='%EXCITATIONY%')
            if fz is not None:
                table = np.zeros((stepsize + 1, self.num_nodes + 1))
                table[1:, 0] = np.arange(chunknum * chunksize + 1, chunknum * chunksize + 1 + stepsize) * deltat
                table[0, 1:] = np.arange(1, self.num_nodes + 1)
                table[1:, 1:] = fz[chunknum * chunksize:chunknum * chunksize + stepsize,:]
                fname = os.path.join(ansys.directory, ansys.jobname + '_z.csv')
                np.savetxt(fname, table)

                with supress_logging(ansys):
                    ansys.starset(par='EXCITATIONZ')

                ansys.dim(par='EXCITATIONZ', type='TABLE', imax=stepsize, jmax=self.num_nodes, kmax="", var1='TIME', var2='NODE')
                ansys.tread(par='EXCITATIONZ', fname=f'{ansys.jobname}_z', ext='csv')

                ansys.f(node='ALL', lab='FZ', value='%EXCITATIONZ%')

            # continue
#             ansys.set_log_level('INFO')
            if chunknum == 0:
                ansys.autots('off')
                # might become slightly inaccurate for "nonrational" deltats,
                # but using nsubst gives problems in free decay, because the first timestep was already computed outside the loop
                ansys.deltim(dtime=deltat, dtmin=deltat, dtmax=deltat)

            ansys.time((chunknum * chunksize + stepsize) * deltat)
            ansys.output(fname='null', loc='/dev')
            ansys.solve()
            ansys.output(fname='term')

            t_end = t_start
            t_start = time.time()

            # if chunk_restart:
            # TODO:: immediately process rst and delete afterwards to avoid disk out of space errors
            # shutil.copyfile(os.path.join(ansys.directory, ansys.jobname + '.rst'), os.path.join(ansys.directory, ansys.jobname + f'.rst.{chunknum}'))
            ind_s = chunknum * chunksize
            ind_e = ind_s + stepsize

            res = pyansys.read_binary(os.path.join(ansys.directory, ansys.jobname + f'.rst'))

            # sometimes, an additional sample for the last time step is computed, which we have to discard

            if not len(res.time_values) <= stepsize + 1:
                raise ValueError(f"Size mismatch in ANSYS results {res.time_values[[0,1,-2,-1]]} should be {table[1:, 0][[0,1,-2,-1]]}")
            time_values[ind_s:ind_e] = res.time_values[:stepsize]

            # out_t.append(res.time_values)

            solution_data_info = res._solution_header(0)
            DOFS = solution_data_info['DOFS']
            ux = DOFS.index(1)
            uy = DOFS.index(2)
            uz = DOFS.index(3)

            if 'd' in out_quant:
                assert np.sum(all_disp_d[ind_s:ind_e,:,:]) == 0  # check to make sure, we get indexing right
                all_disp_d[ind_s:ind_e,:,:] = res.nodal_time_history('NSL')[1][:stepsize,:, (ux, uy, uz)]
                # out_d.append(res.nodal_time_history('NSL')[1])
            if 'a' in out_quant:
                assert np.sum(all_disp_a[ind_s:ind_e,:,:]) == 0
                all_disp_a[ind_s:ind_e,:,:] = res.nodal_time_history('ACC')[1][:stepsize,:, (ux, uy, uz)]
                # out_a.append(res.nodal_time_history('ACC')[1])
            if 'v' in out_quant:
                assert np.sum(all_disp_v[ind_s:ind_e,:,:]) == 0
                all_disp_v[ind_s:ind_e,:,:] = res.nodal_time_history('VEL')[1][:stepsize,:, (ux, uy, uz)]
                # out_v.append(res.nodal_time_history('VEL')[1])

            del res

            ansys.get(par='RST_SUBSTEPS', entity='ACTIVE', item1='SOLU', it1num='NCMSS')
            rst_substeps = int(ansys.parameters['RST_SUBSTEPS'])
            ansys.finish()
            # not needed, if we just move the rst file, but ansys throws a warning about a missing rst file
            if False:
                os.remove(os.path.join(ansys.directory, ansys.jobname + '.rst'))
            else:
                ansys.aux3()
                ansys.file(ansys.jobname, 'rst')
                ansys.delete(set='SET', nstart=1, nend=rst_substeps + 1)
                ansys.compress()
                ansys.finish()

            # truncate open file references to free up disk space, probably caused somewhere in pyansys.read_binary
            del_files = os.popen(f'ls -l /proc/{pid}/fd | grep deleted').readlines()
            truncated = 0
            for fd in del_files:
                this_fd = f'/proc/{pid}/fd/{fd.split()[8]}'
                if this_fd in truncated_fds:
                    continue
                with open(this_fd, 'w'):
                    truncated_fds.append(this_fd)
                    truncated += 1
            logger.debug(f'Truncated {truncated} orphaned file references.')

            # ansys.set_log_level('DEBUG')
            # ansys.config(lab='stat')
            ansys.slashsolu()
            ansys.antype(status='rest')  # restart last analysis
            # ansys.set_log_level("INFO")

            # delete .rxxx files
            # for rxxfile in sorted(glob.glob(os.path.join(ansys.directory, ansys.jobname + '.r*[0-9]')))[:-1]:
            #     os.remove(rxxfile)
            # time.sleep(0.05)

            # outres is not restored upon restart
            if True:
                ansys.outres(item='ERASE')
                ansys.outres(item='ALL', freq='NONE')
                if 'd' in out_quant:
                    ansys.outres(item='NSOL', freq='ALL', cname='MEAS_NODES')
                if 'a' in out_quant:
                    ansys.outres(item='A', freq='ALL', cname='MEAS_NODES')
                if 'v' in out_quant:
                    ansys.outres(item='V', freq='ALL', cname='MEAS_NODES')

            # if chunk_restart:
                # try to reset RESCONTROL to delete evergrowing ldhi file
                # that gives wrong res.time_values
            #    ansys.rescontrol(action='NORESTART')
            #    ansys.rescontrol(action='DEFINE', ldstep='LAST', frequency='LAST')  # Controls file writing for multiframe restarts

            freedisk = shutil.disk_usage(ansys.directory).free / (1024 ** 3)
            while freedisk < 1:
                logger.warning(f'Disk is almost full {freedisk} GB. Blocking 30 s.')
                time.sleep(30)
                freedisk = shutil.disk_usage(ansys.directory).free / (1024 ** 3)
            elapsed = t_start - t_end
            remaining = (timesteps - chunknum * chunksize) / chunksize * elapsed
            logger.info(f'{chunknum * chunksize + chunksize} of {timesteps} timesteps in {elapsed//60:1.0f}m{elapsed%60:1.0f}s (Remaining ~{remaining//60:1.0f}m{remaining%60:1.0f}s; Disk free: {freedisk:.2f} GB)')

        ansys.set_log_level("WARNING")
        ansys.finish()

        # self.last_analysis = 'trans'

        # if chunk_restart:
        #     try:
        #         time_values = np.concatenate(out_t)
        #         if 'a' in out_quant:
        #             all_disp_a = np.concatenate(out_a, axis=0)[:, :, (ux, uy, uz)]
        #             del out_a
        #         else:
        #             all_disp_a = None
        #         if 'v' in out_quant:
        #             all_disp_v = np.concatenate(out_v, axis=0)[:, :, (ux, uy, uz)]
        #             del out_v
        #         else:
        #             all_disp_v = None
        #         if 'd' in out_quant:
        #             all_disp_d = np.concatenate(out_d, axis=0)[:, :, (ux, uy, uz)]
        #             del out_d
        #         else:
        #             all_disp_d = None
        #     except np.core._exceptions._ArrayMemoryError:
        #         emergency_dir = kwargs.pop("emergency_dir", None)
        #         if emergency_dir is not None:
        #             logger.error("Could not allocate enough memory for concatenation.")
        #             emergency_dict = {}
        #             if 'a' in out_quant:
        #                 emergency_dict.update({f'a_{i:04d}':arr for i, arr in enumerate(all_disp_a)})
        #             if 'v' in out_quant:
        #                 emergency_dict.update({f'v_{i:04d}':arr for i, arr in enumerate(all_disp_v)})
        #             if 'd' in out_quant:
        #                 emergency_dict.update({f'd_{i:04d}':arr for i, arr in enumerate(all_disp_d)})
        #             self.save(emergency_dir, emergency_arrays=emergency_dict)
        #             raise
        #         else:
        #             logger.error("Could not allocate enough memory for concatenation. Pass 'emergency_dir' kwarg to enable saving of results.")
#
#
#         else:
# #             print("Reading binary")
#             res = pyansys.read_binary(os.path.join(ansys.directory, ansys.jobname + '.rst'))
#
#             time_values = res.time_values
#
#             solution_data_info = res._solution_header(0)
#             DOFS = solution_data_info['DOFS']
#
#             ux = DOFS.index(1)
#             uy = DOFS.index(2)
#             uz = DOFS.index(3)
#
#             if 'd' in out_quant:
#                 all_disp_d = res.nodal_time_history('NSL')[1][:, :, (ux, uy, uz)]
#             else:
#                 all_disp_d = None
#             if 'a' in out_quant:
#                 all_disp_a = res.nodal_time_history('ACC')[1][:, :, (ux, uy, uz)]
#             else:
#                 all_disp_a = None
#             if 'v' in out_quant:
#                 all_disp_v = res.nodal_time_history('VEL')[1][:, :, (ux, uy, uz)]
#             else:
#                 all_disp_v = None
#             t_end = t_start
#             t_start = time.time()
#             logger.info(f'RST parsing in {t_start-t_end} s')
        if len(time_values) != timesteps:
            warnings.warn(f'The number of response values {len(time_values)} differs from the specified number of timesteps {timesteps} -> Convergence or substep errors.')

        return time_values, [all_disp_d, all_disp_v, all_disp_a]

    def transient_ifrf(self, fy=None, fz=None,
                       inp_nodes=None,
                       inp_dt=None, out_dt=None,
                       out_quant=['d', 'v', 'a'], **kwargs):
        '''
        move the method to MechanicalDummy for cases where frf is precomputed to avoid ansys startup
        in case frf is not precomputed we must be in Mechanical
        so we inherit the method in Mechanical and compute frf (if needed) before calling super()
        '''
        omegas = self.omegas
        frf = self.frf

        inp_dofs = []

        if fy is None and fz is None:
            raise ValueError(f"Neither fy nor fz were provided")

        if fy is not None:
            N = fy.shape[0]
            inp_dofs.append('uy')
        if fz is not None:
            N = fz.shape[0]
            inp_dofs.append('uz')

        if inp_nodes is None:
            inp_nodes = self.nodes_coordinates[:, 0]

        out_dofs = kwargs.get('out_dofs', inp_dofs)

        load_frf = (omegas is not None and omegas[-1] == 1 / 2 / inp_dt * 2 * np.pi)
        load_frf = (load_frf and frf is not None and np.all(frf.shape == (N // 2 + 1, len(inp_nodes) * len(inp_dofs), len(self.meas_nodes) * len(out_dofs))))

        if not load_frf:
            omegas, frf, _, _ = self.frequency_response_non_classical(N,
                                                      inp_nodes=inp_nodes,
                                                      inp_dofs=inp_dofs,
                                                      out_dofs=out_dofs,
                                                      fmax=1 / 2 / inp_dt, out_quant='d', **kwargs)

        return super().transient_ifrf(fy, fz, inp_nodes, inp_dt, out_dt, out_quant, out_dofs=out_dofs)

#     def mode_superpos(self, f=None, d=None):
#         ansys = self.ansys
#         # Transient/Harmonic Response
#         deltat=self.deltat
#         timesteps=self.timesteps
#
#         ansys.run('/SOL')
#         ansys.antype('TRANS')
#         ansys.trnopt(method='MSUP')# bug: vaout should be tintopt
#
#
#         ansys.kbc(0) # Specifies ramped or stepped loading within a load step.
#         ansys.timint(1) #Turns on transient effects.
#         ansys.alphad(value=0)
#         ansys.betad(value=0)
#         ansys.dmprat(ratio=0)
#         ansys.mdamp(stloc=1, v1=0, v2=0, v3=0, v4=0, v5=0, v6=0)
# #         ansys.outres(item='ERASE')
#         ansys.outres(item='ALL',freq='ALL')# Enable all output
# #         ansys.nsel(type='S', item='NODE', vmin=30,vmax=40) # select only nodes of interest
# #         #ansys.nsel(type='A', item='NODE', vmin=2,vmax=2) # select only nodes of interest
# #         #ansys.cm(cname='meas_nodes', entity='NODE') # and group into component assembly
# #         ansys.nsel(type='ALL')
# #         ansys.outres(#item='A',
# #                      item='NSOL',
# #                      freq='ALL'
# #                      #,cname='meas_nodes'
# #                      )# Controls the solution data written to the database.
# #         ansys.outres(item='A',
# #                      #item='NSOL',
# #                      freq='ALL'
# #                      #,cname='meas_nodes'
# #                      )# Controls the solution data written to the database.
# #         ansys.outres(item='V',
# #                      #item='NSOL',
# #                      freq='ALL'
# #                      #,cname='meas_nodes'
# #                      )# Controls the solution data written to the database.
# #         ansys.rescontrol(action='DEFINE',ldstep='NONE',frequency='NONE',maxfiles=-1)  # Controls file writing for multiframe restarts
#         ansys.deltim(dtime=deltat, dtmin=deltat, dtmax=deltat, carry='OFF')
#
#         #ansys.solve()
#
#         t_end = deltat*(timesteps-1)
#         t = np.linspace(deltat,stop=t_end,num=timesteps)
#
#         printsteps = list(np.linspace(0,timesteps, 100, dtype=int))
#         dts=[]
#         timesteps=10000
#         t_start = time.time()
#         for lsnum in range(timesteps):
#             if not lsnum %1000:
#                 t_end=t_start
#                 t_start = time.time()
#                 dts.append(t_start-t_end)
#                 print(lsnum, np.mean(dts))
#
#             while lsnum in printsteps:
#                 del printsteps[0]
#                 print('.',end='', flush=True)
#             ansys.time((lsnum+1)*deltat)
#             if f is not None:
#                 ansys.fdele(node='ALL', lab='ALL')
#                 ansys.f(node=20,lab='FZ', value=f[lsnum])
#             if d is not None:
#                 ansys.ddele(node=20, lab='UZ')
#                 if d[lsnum]:
#                     ansys.d(node=20, lab='UZ', value=d[lsnum])
#             #continue
#             ansys.solve()
#         #np.savetxt('dts_amsupsolve',dts)
#         #asd
#         print('.',end='\n', flush=True)
#         ansys.finish()
#         ansys.run('/SOL')
#         ansys.expass(key='ON')
#         ansys.numexp(num=timesteps, begrng=0, endrng=timesteps*deltat)
#         ansys.solve()
#         ansys.finish()
#         self.last_analysis = 'trans'
#
#
#     def sweep(self, deltat = 0.01, timesteps = 1024, f_start=0/2/np.pi, f_end=None, phi=0, ampl= 1000):
#
#         nyq = 1/deltat/2.5
#         if f_end is None:
#             f_end=nyq
#
#         t_end = deltat*timesteps
#         t = np.linspace(deltat,stop=t_end,num=timesteps)
#
#         assert f_end<=nyq
#
#         f = np.sin(2*np.pi*((f_start*t)+(f_end-f_start)/(2*t_end)*t**2+phi))*ampl
#
#         self.deltat = deltat
#         self.timesteps = timesteps
#
#         return f

    def export_ans_mats(self):
        ansys = self.ansys
        jid = self.jobname
        # np.set_printoptions(precision=3, linewidth=200, suppress=True)
        ansys.slashsolu()
        ansys.antype('MODAL')
        ansys.outres(item='ERASE')
        ansys.outres(item='ALL', freq='NONE')

        ansys.nsel(type='ALL')
        ansys.outres(item='NSOL', freq='ALL')
        ansys.modopt(method='QRDAMP', nmode=100, freqb=0,
                     freqe=1e4,
                     cpxmod='cplx',
                     nrmkey='on',
                     )
        ansys.mxpand(nmode='all', elcalc=1)
        ansys.solve()
        ansys.wrfull()
        ansys.finish()
        ansys.aux2()
        # ansys.file(jid,'FULL')
        ansys.dmat(matrix="MatKD", type="D", method="IMPORT", val1="FULL", val2=f"{jid}.full", val3="STIFF")
        ansys.export(matrix="MatKD", format="MMF", fname="MatKD.bin")
        k = np.array(scipy.io.mmread('MatKD.bin'))
        ansys.dmat(matrix="MatMD", type="D", method="IMPORT", val1="FULL", val2=f"{jid}.full", val3="MASS")
        ansys.export(matrix="MatMD", format="MMF", fname="MatMD.bin")
        m = scipy.io.mmread('MatMD.bin')
        try:
            ansys.dmat(matrix="MatCD", type="D", method="IMPORT", val1="FULL", val2=f"{jid}.full", val3="DAMP")
            ansys.export(matrix="MatCD", format="MMF", fname="MatCD.bin")
            c = scipy.io.mmread('MatCD.bin')
        except Exception as e:
            # print(e)
            c = np.zeros_like(k)
        ansys.finish()

        # ansys.smat(matrix='USR2SOLV', type='D', method='IMPORT', val1='FULL', val2=f"{jid}.full", val3='USR2SOLV')
        # ansys.export(matrix="USR2SOLV", format="MMF", fname="USR2SOLV.bin")
        # usr2solv = np.array(scipy.io.mmread('USR2SOLV.bin'))
        #

        # full_path = os.path.join(ansys.directory, ansys.jobname + '.full')
        # full = pyansys.read_binary(full_path)
        # # TODO: Check, that Nodes and DOFS are in the same order in modeshapes and k,m
        # dof_ref, k_, m_ = full.load_km(as_sparse=False, sort=False)
        # k_ += np.triu(k_, 1).T
        # m_ += np.triu(m_, 1).T
    # #     print(dof_ref)
        # for mode in range(num_modes):
            # msh_f = mshs[1:, 2, mode].flatten()
            #
            # kappa = msh_f.T.dot(k).dot(msh_f)
            # mu = msh_f.T.dot(m).dot(msh_f)
            #
            # print(np.sqrt(kappa / mu) / 2 / np.pi)
            #
            # msh_f = mshs[:, :, mode].flatten()
            #
            # kappa = msh_f.T.dot(k_).dot(msh_f)
            # mu = msh_f.T.dot(m_).dot(msh_f)
            # print(np.sqrt(kappa / mu) / 2 / np.pi)

        return k, m, c

    def save(self, fpath, emergency_arrays=None):
        super().save(fpath, emergency_arrays)

    @classmethod
    def load(cls, fpath, ansys=None, wdir=None,):
        assert os.path.exists(fpath)

        logger.info('Now loading previous results from  {}'.format(fpath))

        in_dict = np.load(fpath, allow_pickle=True)

        jobname = in_dict['self.jobname'].item()

        # assert jobname == in_dict['self.jobname'].item()

        mech = cls(ansys, jobname, wdir)

        mech._load(fpath, mech)

        return mech

    def numerical_response_parameters(self, num_modes=None, compensate=True, dofs=[2]):
        '''optionally, compensate with time integration parameters
        compensate: frequencies for spatial and temporal discretization(and mass matrix formulation?)
        compensate: damping ratios for non-constant rayleigh and temporal discretization
        modeshapes are always reduced to the set of meas nodes
        '''

        ndof = len(dofs)
        if num_modes is None:
            num_modes = self.num_modes
#         frequencies, _, _ = self.modal(damped=False, num_modes=num_modes)
        if self.trans_params is None and compensate:
            logger.debug('No transient parameters set. Compensation will be skipped.')
            compensate = False

        frequencies, damping, mode_shapes = self.modal(damped=True, num_modes=num_modes)
        deltat = self.deltat

        meas_nodes = self.meas_nodes
        nodes_coordinates = self.nodes_coordinates
        meas_indices = []
        for meas_node in self.meas_nodes:
            for i, (node, x, y, z) in enumerate(nodes_coordinates):
                if node == meas_node:
                    meas_indices.append(i)
                    break
            else:
                raise RuntimeError(f'meas_node {meas_node} could not be found in nodes_coordinates')

        mode_shapes_n = np.full((len(meas_nodes) * ndof, num_modes), np.nan, dtype=complex)
        for mode in range(num_modes):
            for i, dof in enumerate(dofs):
                mode_shapes_n[len(meas_nodes) * i:len(meas_nodes) * (i + 1), mode] = mode_shapes[meas_indices, dof, mode]

        if not compensate:
            frequencies_n = frequencies
            damping_n = damping
        else:
            # (delta,alpha, alphaf,alpham)
            # gamma,beta, alphaf, alpham
            gamma_, beta_, alpha_f_, alpha_m_ = self.trans_params

            if gamma_ is None: gamma_ = 0
            if beta_ is None: beta_ = 0
            if alpha_f_ is None: alpha_f_ = 0
            if alpha_m_ is None: alpha_m_ = 0

#             h,beta,gamma,zeta,omega,Omega,eta=sympy.symbols('h \\beta \gamma \zeta \omega \Omega \eta', real=True, positive=True)
#             alpha_f, alpha_m, rho_inf=sympy.symbols('$\\alpha_f$ $\\alpha_m$ $\\rho_{\infty}$', real=True)
#
#             A1=sympy.Matrix([[1,0, -beta],[0, 1, -gamma],[Omega**2*(1-alpha_f),2*zeta*Omega*(1-alpha_f), 1-alpha_m]])
#             A2=sympy.Matrix([[1,1,0.5-beta],[0,1,1-gamma],[-Omega**2*(alpha_f),-2*zeta*Omega*(alpha_f), -alpha_m]])
#             A=A1.solve(A2)
            frequencies_n = np.full_like(frequencies, np.nan)
            damping_n = np.full_like(damping, np.nan)
#             for i,(zeta_, freq) in enumerate(zip(damping, frequencies)):
#                 Omega_ = deltat*freq
#                 A_sub = A.subs(beta, beta_).subs(gamma, gamma_).subs(alpha_f, alpha_f_).subs(alpha_m,alpha_m_).subs(zeta, zeta_)
#                 lamda = np.linalg.eigvals(np.array(A_sub.subs(Omega, Omega_)).astype(np.float64))
#                 rho = np.abs(lamda)
#                 phi = np.abs(np.angle(lamda))
#                 j = np.argmax((phi < np.pi)*rho)
#
#                 frequencies_n[i] = np.sqrt(phi[j]**2+np.log(rho[j])**2)
#                 damping_n[i] = -np.log(rho[j])/frequencies_n[i]

            for mode in range(num_modes):

                zeta_ = damping[mode]
                freq_ = frequencies[mode]
                omega_ = freq_ * 2 * np.pi  # damped response frequency, undamped would be a little higher
                # Omega_ = omega_ * deltat

                # If here the damped omega is provided, later Omega_hat_ud is to be used
                # If here the undamped omega is provided, later Omega_hat_dd is to be used
                # Providing damped omega is more logical, as this is the response frequency of the system,
                # also this would be inline with providing the physical parameters for the amplification matrix
                A1 = np.array([[1, 0, -beta_ * deltat ** 2],
                             [0, 1, -gamma_ * deltat  ],
                             [omega_ ** 2 * (1 - alpha_f_), 2 * zeta_ * omega_ * (1 - alpha_f_), 1 - alpha_m_      ]])
                A2 = np.array([[1, 1 * deltat, (0.5 - beta_) * deltat ** 2],
                             [0, 1, (1 - gamma_) * deltat    ],
                             [-omega_ ** 2 * (alpha_f_), -2 * zeta_ * omega_ * (alpha_f_), -alpha_m_            ]])

#                 zeta_ = damping[mode]
#                 freq_ = frequencies[mode]
#                 Omega_ = deltat*freq_*2*np.pi
#                 A1=np.array([[1,0, -beta_],[0, 1, -gamma_],[Omega_**2*(1-alpha_f_),2*zeta_*Omega_*(1-alpha_f_), 1-alpha_m_]])
#                 A2=np.array([[1,1,0.5-beta_],[0,1,1-gamma_],[-Omega_**2*(alpha_f_),-2*zeta_*Omega_*(alpha_f_), -alpha_m_]])
                A = np.linalg.solve(A1, A2)
                lamda = np.linalg.eigvals(A)
                lamda = lamda[np.logical_and(np.real(lamda) != 0, np.imag(lamda) != 0)]
                # find the conjugate eigenvalues
                if np.logical_and(np.real(lamda) < 0, np.imag(lamda) == 0).all():
                    logger.warning("System has only negative, real eigenvalues!")
                    frequencies_n[mode] = np.nan
                    damping_n[mode] = np.nan
                else:
                    loglamda = np.log(lamda)
                    if (np.imag(loglamda) > 0).any():
                        this_loglamda = loglamda[loglamda.imag > 0].max()
                        Omega_hat_dd = np.imag(this_loglamda)  # contains damping twice, as the damped sampling frequency was provided to the amplification matrix
                        Omega_hat_ud = np.abs(this_loglamda)  # undamped sampling frequency for damped response frequency, contains numerical damping

                        zeta_hat = -np.real(this_loglamda) / Omega_hat_ud  # contains both physical and numerical damping

                        frequencies_n[mode] = Omega_hat_ud / 2 / np.pi / deltat
                        damping_n[mode] = zeta_hat

        self.frequencies_comp = frequencies_n
        self.modal_damping_comp = damping_n
        self.mode_shapes_comp = mode_shapes_n

        self.state[5] = True

        return frequencies_n, damping_n, mode_shapes_n


def main():
    pass


if __name__ == '__main__':
    main()
