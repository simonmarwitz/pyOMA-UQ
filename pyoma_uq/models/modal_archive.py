"""Consume a pre-computed modal / FRF archive -- no FEM solver involved.

This is the model-based path's data source: an archive produced elsewhere
(historically by an ANSYS run, but nothing here cares how) holding a structure's
modal solution and, optionally, a memory-mapped compliance FRF. It is what
:class:`~pyoma_uq.studies.UQ_OMA_weighted.FRFResponseProvider` and
:class:`~pyoma_uq.studies.UQ_OMA_weighted.ToyResponseProvider` read.

The ANSYS bridge that *generated* such archives (model building through APDL,
transient solves, element libraries) has been removed: the supported model-based
entry point is a response provider fed by a user-supplied FRF or response, so
the solver is the user's business and not a dependency of this package. Archives
written by the old ``Mechanical``/``MechanicalDummy`` classes load unchanged --
:meth:`ModalArchive.load` reads the same ``.npz`` layout, and ``MechanicalDummy``
remains as an alias.

Key entry points:

* :meth:`ModalArchive.load` -- read a ``mechanical.npz``
* :meth:`ModalArchive.transient_ifrf` -- synthesize a transient response from
  the stored FRF and a given input history
* :meth:`ModalArchive.modal` -- the stored modal solution
"""
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


class ModalArchive(object):

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
            raise RuntimeError(f"Pre-computed FRF is not compatible with given parameters for: {cause}. the archive was written for a different parameter set")

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

        logger.info(f'Saving modal archive to {fpath}')

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




#: The archives predate the rename; keep the old name importable so existing
#: ``.npz`` files and downstream scripts keep working.
MechanicalDummy = ModalArchive
