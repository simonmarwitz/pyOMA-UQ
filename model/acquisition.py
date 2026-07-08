'''
models the signal acquisition:

TODO:
 * sensor FRF simulation
 * predefined noise profiles


'''

import numpy as np
import scipy.signal
import matplotlib
import matplotlib.pyplot as plt
import os
import logging
import signal
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LoggingContext(object):

    def __init__(self, logger, level=None, handler=None, close=True):
        self.logger = logger
        self.level = level
        self.handler = handler
        self.close = close

    def __enter__(self):
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)

    def __exit__(self, et, ev, tb):
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()
        # implicit return of None => don't swallow exceptions


print_context_dict = {'text.usetex':True,
                     'text.latex.preamble':"\\usepackage{siunitx}\n \\usepackage{xfrac}",
                     'font.size':10,
                     'legend.fontsize':10,
                     'xtick.labelsize':10,
                     'ytick.labelsize':10,
                     'axes.labelsize':10,
                     'font.family':'serif',
                     'legend.labelspacing':0.1,
                     'axes.linewidth':0.5,
                     'xtick.major.width':0.2,
                     'ytick.major.width':0.2,
                     'xtick.major.width':0.5,
                     'ytick.major.width':0.5,
                     'figure.figsize':(5.906, 5.906 / 1.618),  # print #150 mm \columnwidth
                     # 'figure.figsize':(5.906/2,5.906/2/1.618),#print #150 mm \columnwidth
                     # 'figure.figsize':(5.53/2,2.96),#beamer
                     # 'figure.figsize':(5.53/2*2,2.96*2),#beamer
                     'figure.dpi':100}
    # figsize=(5.53,2.96)#beamer 16:9
    # figsize=(3.69,2.96)#beamer 16:9
    # plot.rc('axes.formatter',use_locale=True) #german months
# must be manually set due to some matplotlib bugs
# if print_context_dict['text.usetex']:
    # # plt.rc('text.latex',unicode=True)
    # plt.rc('text', usetex=True)
    # plt.rc('text.latex', preamble="\\usepackage{siunitx}\n \\usepackage{xfrac}")


class Acquire(object):
    '''
    A class for processing a signal, simulating ADC signal acquisition:
        apply sensor FRFs
        (add measurement noise)
        sample (apply lowpass AA filter and downsample)
    
    Additionally, characteristics of the generating system and the signal
    itself are modified according to the respective operations:
        - number of modes -> changes due to downsampling
        - modal frequencies -> reduces to new number of modes
        - modal damping -> reduces to new number of modes, might change due to windowing?
        - mode shapes -> reduces to new number of modes, should not be affected, because operators are applied equally to all channels
        - modal amplitudes -> reduces to new number of modes, changes according to sensor frf and SNR
        - signal-to-noise ratio -> changes according to the amount of noise added
        - decimation noise -> results from imperfect filtering before sample reduction
        - signal energies -> might change due to frf, noise and sampling
        - ...
    
    .. TODO::
        * add save / load abilities
        * add functions for verification, plots
    '''

    def __init__(self, t_vals, signal=None, IRF_matrix=None, channel_defs=None,
                 modal_frequencies=None, modal_damping=None, mode_shapes=None,
                 modal_energies=None, modal_amplitudes=None,
                 jobname=None,
                 ):
        '''
        input a synthethised signal (ambient response, impulse response) and corresponding t_vals
        '''
        if IRF_matrix is not None:
            assert signal is None
            # (num_ref_nodes, num_channels, num_timesteps)
            num_channels_ = IRF_matrix.shape[1]
            num_ref_nodes_ = IRF_matrix.shape[0]
            num_timesteps_ = IRF_matrix.shape[2]
            # Flatten the IRF matrix to a signal of shape (num_channels, num_timesteps)
            signal = IRF_matrix.reshape((num_ref_nodes_ * num_channels_, num_timesteps_))
            # Store the shape to restore the IRF matrix in get_signal
            self.re_shape = [num_ref_nodes_, num_channels_, num_timesteps_]
        else:
            self.re_shape = list(signal.shape)

        self.t_vals = t_vals

        self.signal = signal

        self.channel_defs = channel_defs

        if modal_frequencies is not None:
            assert isinstance(modal_frequencies, (list, tuple, np.ndarray))
            assert len(modal_frequencies.shape) == 1
            # self.num_modes = modal_frequencies.size

            assert np.all(np.diff(modal_frequencies) >= 0)  # ensure they are sorted

        self.modal_frequencies = modal_frequencies

        if modal_damping is not None:
            assert isinstance(modal_damping, (list, tuple, np.ndarray))
            assert len(modal_damping.shape) == 1
            # self.num_modes = modal_damping.size

        self.modal_damping = modal_damping

        if mode_shapes is not None:
            assert isinstance(mode_shapes, (list, tuple, np.ndarray))
            assert len(mode_shapes.shape) == 2  # needs reduced mode shapes (channel,mode) type, not (node, dof, mode) 3D type

            # self.num_modes = mode_shapes.shape[1]
            # print(mode_shapes.shape, self.num_channels)
            assert mode_shapes.shape[0] == self.num_channels

        self.mode_shapes = mode_shapes

        if modal_energies is not None:
            assert isinstance(modal_energies, (list, tuple, np.ndarray))
            assert len(modal_energies.shape) == 1

            # self.num_modes = modal_energies.shape[1]
            # IRF matrix modal matrices are shaped (num_ref_nodes, num_modes)
            # impulse response modal matrices are shape (num_nodes, num_modes

        self.modal_energies = modal_energies

        if modal_amplitudes is not None:
            assert isinstance(modal_amplitudes, (list, tuple, np.ndarray))
            assert len(modal_amplitudes.shape) == 1

            # self.num_modes = modal_amplitudes.shape[1]
            # IRF matrix modal matrices are shaped (num_ref_nodes, num_modes)
            # impulse response modal matrices are shape (num_nodes, num_modes

        self.modal_amplitudes = modal_amplitudes
        self.s_vals_psd = None
        self.snr_db_est = None

        self.reset_snr()  # no noise so far

        self.jobname = jobname

        self.is_sensed = False

    @property
    def deltat (self):
        if self.is_sampled:
            t_vals = self.t_vals_samp
        else:
            t_vals = self.t_vals
        return (t_vals[-1] - t_vals[0]) / (t_vals.size - 1)

    @property
    def num_channels(self):
        signal = self.signal
        return signal.shape[0]

    @property
    def num_timesteps(self):
        if self.is_sampled:
            signal = self.signal_samp
        else:
            signal = self.signal
        return signal.shape[1]

    @property
    def num_modes(self):
        if self.modal_frequencies is None:
            return None
        elif self.is_sampled:
            return self.modal_frequencies_samp.shape[0]
        else:
            return self.modal_frequencies.shape[0]

    @property
    def sampling_rate(self):
        return 1 / self.deltat

    @property
    def duration(self):
        return self.num_timesteps * self.deltat

    @property
    def snr_db(self):
        return 10 * np.log10(self.snr[0] / self.snr[1])

    @classmethod
    def init_from_mech(cls, mech, channel_defs=None):

        '''
        Extract the relevant parameters from a mechanical object
        and keeps a reference to that object?
        
        That assumes, only one type of signal (free decay, ambient, impulse)
        is present in the mechanical object (uses the first encountered type)
        
        '''
        logger.info('Initializing Acquire object from a Mechanical object')
        jobname = mech.jobname

        if mech.state[5]:
            modal_frequencies = mech.frequencies_comp
            modal_damping = mech.modal_damping_comp
            # mode_shapes = mech.mode_shapes_comp
            mode_shapes = mech.damped_mode_shapes  # shape (num_meas_nodes, 3, num_modes)
        elif mech.state[4]:
            modal_frequencies = mech.damped_frequencies
            modal_damping = mech.modal_damping
            mode_shapes = mech.damped_mode_shapes  # shape (num_meas_nodes, 3, num_modes)
        else:
            modal_frequencies = None
            modal_damping = None
            mode_shapes = None

        if mech.state[1]:
            t_vals = mech.t_vals_decay
            signals = mech.resp_hist_decay
            kind = 'free decay response'
        elif mech.state[2]:
            t_vals = mech.t_vals_amb
            signals = mech.resp_hist_amb
            kind = 'ambient response'
        elif mech.state[3]:
            t_vals = mech.t_vals_imp
            signals = mech.resp_hist_imp
            kind = 'impulse response'
        elif mech.state[6]:
            # IRF matrix is only a single quantity, usually displacement
            # size is num_ref_nodes, num_timesteps, num_meas_nodes, 3
            t_vals = mech.t_vals_imp
            IRF_matrix = mech.IRF_matrix

        else:
            raise ValueError('No signal has been generated yet.')

        meas_nodes = mech.meas_nodes  # np.ndarray
        nodes_coordinates = mech.nodes_coordinates

        if np.sum(mech.state[1:4]) > 1:
            logger.warning(f'More than one type of signal has been generated. Using the {kind} signal.')

        # signals is size [3][num_timesteps, num_meas_nodes, 3]
        if mech.state[6]:
            quant_avail = [True, False, False]  # assuming displacement
        else:
            quant_avail = [sig is not None for sig in signals]

        if channel_defs in ['ux', 'uy', 'uz']:
            dofs = [['ux', 'uy', 'uz'].index(channel_defs)]
            channel_defs = None
        else:
            dofs = list(range(3))

        if channel_defs in ['d', 'v', 'a']:
            quant = ['d', 'v', 'a'].index(channel_defs)
            assert quant_avail[quant]
            channel_defs = None
        else:
            for quant in [2, 1, 0]:
                if quant_avail[quant]:
                    break

        if isinstance(channel_defs, (list, tuple, np.ndarray)):

            if not isinstance(channel_defs[0], (tuple, list)):  # list of nodes
                nodes = channel_defs
                channel_defs = None
        else:
            nodes = meas_nodes

        if channel_defs is None:
            # may be any of: list of (node, dof, quant)
            #                list of nodes -> all dof and accel
            #                dof: [ux, uy, uz] -> all nodes and accel
            #                quant: [d,v,a]-> all nodes and all dof

            num_nodes = len(nodes)
            num_dof = len(dofs)  # only translational dofs will be used, rotational ignored
            num_quant = 1
            channel_defs = np.empty((num_nodes * num_dof * num_quant, 3), dtype=int)
            ind = 0
            for meas_node in nodes:
                for dof in dofs:
                    channel_defs[ind,:] = [meas_node, dof, quant]
                    ind += 1
            else:
                assert ind == num_nodes * num_dof * num_quant
        else:
            assert isinstance(channel_defs, (tuple, list, np.ndarray))
            for ind in range(len(channel_defs)):  # len refers to the first dimension of an ndarray or to the len of a list
                this_channel_defs = channel_defs[ind]
                if isinstance(this_channel_defs, tuple):
                    this_channel_defs = list(this_channel_defs)
                    channel_defs[ind] = this_channel_defs

                meas_node, dof, quant = this_channel_defs  # for ndarray returns rows, will throw an error anyway if not
                assert meas_node in meas_nodes

                if isinstance(dof, str):
                    dof = ['ux', 'uy', 'uz'].index(dof)
                    channel_defs[ind][1] = dof

                if isinstance(quant, str):
                    quant = ['d', 'v', 'a'].index(quant)
                    channel_defs[ind][2] = quant
                assert quant_avail[quant]
            channel_defs = np.array(channel_defs, dtype=int)

        num_channels = channel_defs.shape[0]
        num_timesteps = len(t_vals)
        if mech.state[6]:
            # IRF matrix is generated only for a single quantity
            # size is num_ref_nodes, num_timesteps, num_meas_nodes, 3
            num_ref_nodes = IRF_matrix.shape[0]
            signal = np.empty((num_ref_nodes, num_channels, num_timesteps))
            for channel, (meas_node, dof, quant) in enumerate(channel_defs):
                node_ind = list(meas_nodes).index(meas_node)
                for ref_node in range(num_ref_nodes):
                    signal[ref_node, channel,:] = IRF_matrix[ref_node,:, node_ind, dof]
        else:
            signal = np.empty((num_channels, num_timesteps))
            for channel, (meas_node, dof, quant) in enumerate(channel_defs):
                node_ind = list(meas_nodes).index(meas_node)
                signal[channel,:] = signals[quant][:, node_ind, dof]

        if mode_shapes is not None:
            num_modes = mode_shapes.shape[2]
            mode_shapes_chan = np.empty((num_channels, num_modes), dtype=complex)
            for channel, (meas_node, dof, _) in enumerate(channel_defs):
                for node_ind, (node, _, _, _) in enumerate(nodes_coordinates):
                    if node == meas_node:
                        break
                else:
                    raise RuntimeError(f'meas_node {meas_node} could not be found in nodes_coordinates')
                    continue
                mode_shapes_chan[channel,:] = mode_shapes[node_ind, dof,:]
            mode_shapes = mode_shapes_chan

        if mech.state[1] or mech.state[2]:
            acqui = cls(t_vals, signal, None, channel_defs,
                        modal_frequencies, modal_damping, mode_shapes,
                        jobname=jobname)

        if mech.state[3]:  # impulse response
            # num_nodes here is all nodes, because system may be excited at every node
            # so we should not reduce these matrices to meas_nodes, channels, whatsoever
            modal_energies = mech.modal_imp_energies  # shape (num_nodes, num_modes)
            modal_amplitudes = mech.modal_imp_amplitudes  # shape (num_nodes, num_modes)

            acqui = cls(t_vals, signal, None, channel_defs,
                        modal_frequencies, modal_damping, mode_shapes,
                        modal_energies, modal_amplitudes,
                        jobname=jobname)

        if mech.state[6]:  # IRF matrix
            # mech.imp_hist_imp_matrix
            modal_energies = mech.modal_imp_energy_matrix  # shape (num_ref_nodes, num_modes)
            modal_amplitudes = mech.modal_imp_amplitude_matrix  # shape (num_ref_nodes, num_modes)
            acqui = cls(t_vals, None, signal, channel_defs,
                        modal_frequencies, modal_damping, mode_shapes,
                        modal_energies, modal_amplitudes,
                        jobname=jobname)

        # mech.trans_params
        if acqui.deltat != mech.deltat and np.isclose(acqui.deltat, mech.deltat):
            acqui.deltat = mech.deltat
        elif not np.isclose(acqui.deltat, mech.deltat):
            raise RuntimeError(f'There is a mismatch between timestep size in data {acqui.deltat}and in the model {mech.deltat}.')

        assert acqui.num_timesteps == mech.timesteps

        if mech.state[4] or mech.state[5]:
            assert acqui.num_modes == mech.num_modes

        # acqui.mech = mech

        return acqui

    def apply_sensor(self, DTC, fn=None,
                     sensitivity_nominal=1, sensitivity_deviation=0,
                     spectral_noise_slope=None, noise_rms=None,
                     spectral_freq=None, noise_profile=None,
                     seed=None):
        '''
        Applies a sensor FRF to the signal and adds spectral noise, scales the mode
        shapes by the sensor FRF.
        
        Currently only the high-pass of the RC sensor amplifier is implemented
        as a first-order butterworth filter.
        
        Spectral noise can be be provided as a noise profile, or in a simplified 
        form, as a spectral slope and noise rms value.
        
        Finally the signal is converted by sensor sensitivity to a voltage, 
        s. t. argument measurement_range in self.sample(...) makes sense
        additionally, slight deviations can be taken into account
        by sampling each channels sensitivity from a Uniform distribution centered
        at the nominal sensitivity and a given interval, e.g. +- 10%
        
        Parameters
        ----------
            DTC: float, required
                The discharge time constant of the sensor RC circuit in seconds
            fn: float, optional
                The (mechanical) resonance frequency of the sensor
            sensitivity_nominal: float, optional
                The nominal sensitivity of the sensor in V m^-1 s^2
            sensitivity_deviation: float, optional
                The deviation of the sensor sensitivity in V m^-1 s^2
                derived e.g. from datasheet specifications such as +-10%
            spectral_noise_slope: float, optional
                The slope of the spectral noise amplitude in log-log-space [m s^-2/sqrt(Hz)]
            noise_rms: float, optional
                The RMS value of the spectral noise in 
            spectral_freq, noise_profile: ndarray, optional
                Spectral noise amplitudes at given frequencies of the noise profile
            seed: int, optional
                Seed for the Random Number Generator for reproducible results
        '''

        if fn is not None:
            raise NotImplementedError("Sensor resonance frequency response is not implemented")

        if self.is_sampled:
            logger.warning(f'Resetting an already sampled signal.')
            self.reset_snr()
            self.is_sampled = False
        if self.is_sensed:
            logger.warning('Resetting an already sensed signal.')
            self.reset_snr()
            self.is_sensed = False

        deltat = self.deltat
        num_timesteps = self.num_timesteps
        num_channels = self.num_channels
        modal_frequencies = self.modal_frequencies

        # Apply highpass RC filter (sensor amplifier discharge time constant)
        f_c = 1 / (2 * np.pi * DTC)
        nyq = self.sampling_rate / 2

        logger.info(f'Applying a {sensitivity_nominal:1.2f} Vm^-1s^2 sensor with a high-pass cutoff at {f_c:1.3f} Hz and spectral noise of {noise_rms} ms^-2')

        b, a = scipy.signal.butter(1, f_c / nyq, 'high')
        meas_sig = scipy.signal.lfilter(b, a, self.signal, axis=1)

        rng = np.random.default_rng(seed)
        sensitivities = rng.uniform(sensitivity_nominal - sensitivity_deviation, sensitivity_nominal + sensitivity_deviation, num_channels)

        # apply FRF to mode shapes
        if modal_frequencies is not None:  # proxy for: "modal characteristics are present"
            # adjust modal characteristics according to sensor frf

            modal_frf = DTC * 2 * np.pi * 1j * modal_frequencies / (1 + DTC * 2 * np.pi * 1j * modal_frequencies)

            if self.mode_shapes is not None:
                self.mode_shapes_sensed = self.mode_shapes * modal_frf * sensitivities[:, np.newaxis]
            if self.modal_amplitudes is not None:
                logger.warning('Amplitude reduction by sensor filters not validated.')
                self.modal_amplitudes_sensed = self.modal_amplitudes * np.abs(modal_frf) * sensitivities[:, np.newaxis]
            if self.modal_energies is not None:
                logger.warning('Energy reduction by sensor filters not validated.')
                self.modal_energies_sensed = self.modal_energies * np.abs(modal_frf) * sensitivities[:, np.newaxis]
            # modal damping and modal frequencies should not be affected

        # Appy spectral noise to the signal
        freq = np.fft.rfftfreq(num_timesteps, deltat)
        # delta_f = freq[1] - freq[0]

        if spectral_noise_slope is not None and noise_rms is not None:
            F0 = noise_rms * np.sqrt(2 * deltat)
            x0 = 10
            with np.errstate(divide='ignore'):
                intercept = F0 / (x0 ** spectral_noise_slope)
                var_asd = freq ** spectral_noise_slope * intercept
            # reset DC component to 0 (assumes np.fft.rfftfreq[0] = 0)
            var_asd[0] = 0
        elif spectral_freq is not None and noise_profile is not None:
            # Interpolate in log-log-space
            with np.errstate(divide='ignore'):
                var_asd = np.power(10.0, np.interp(np.log10(freq), np.log10(spectral_freq), np.log10(noise_profile)))
        else:
            raise RuntimeError(f'Noise characterization arguments were not provided or provided incompletely.')

        rng = np.random.default_rng(seed)

        # power spectral density (PSD) is the time average of the energy normalized to unit frequency
        # amplitude spectral density (ASD) is the square root of the PSD
        # Amplitude/RMS spectrum given, must not be square rooted for IFFT

        # Coloured Noise with randomized magnitude and phase
        # Randomizing the phase is not sufficient, as the PSD segments would all be the equal
        # https://blog.ioces.com/matt/posts/colouring-noise/
        asd = rng.normal(0, 1 / np.sqrt(2 * deltat),
                               (num_channels, num_timesteps // 2 + 1)
                               ) * var_asd * (1 + 1j)
        # reverse unit-frequency normalization of the ASD (PSD: 1 / N, ASD: 1 / sqrt(N) to make it available for IFFT
        # https://physics.stackexchange.com/questions/615349/amplitude-spectral-density-vs-power-spectral-density
        asd *= np.sqrt(num_timesteps)

        # Ensure the 0 Hz and Nyquist components are purely real
        asd[:, 0] = np.abs(asd[:, 0])
        if len(freq) % 2 == 0:
            asd[:, -1] = np.abs(asd[:, -1])

        # Transform to time-domain
        spectral_noise = np.fft.irfft(asd, axis=1)

        channel_powers = np.mean(meas_sig ** 2, axis=1)
        power_noise = np.mean(spectral_noise ** 2, axis=1)
        logger.debug(f'Final average noise power: {np.mean(power_noise):1.3g} (signal: {np.mean(channel_powers):1.3g})')

        # Add to signal
        meas_sig += spectral_noise

        # convert final signal to voltage
        meas_sig *= sensitivities[:, np.newaxis]

        self.sensor_noise = spectral_noise
        # self.sensor_cutoff = f_c
        self.update_snr(power_noise, channel_powers)
        self.sensor_sensitivity = sensitivity_nominal

        self.signal_volt = meas_sig
        self.is_sensed = True
        self.is_sampled = False

        return

    @property
    def is_sampled(self):
        return self._is_sampled

    @is_sampled.setter
    def is_sampled(self, b):
        if not b:
            # invalidate sampled values
            self.signal_samp = None
            # self.num_timesteps_samp = None
            self.t_vals_samp = None
            # self.deltat_samp = None
            # self.num_modes_samp = None
            self.modal_frequencies_samp = None
            self.modal_damping_samp = None
            self.mode_shapes_samp = None
            self.modal_energies_samp = None
            self.modal_amplitudes_samp = None
            self.bits_effective = None
        self._is_sampled = b

    @property
    def is_sensed(self):
        return self._is_sensed

    @is_sensed.setter
    def is_sensed(self, b):
        if not b:
            # invalidate sensed values
            self.mode_shapes_sensed = None
            self.modal_energies_sensed = None
            self.modal_amplitudes_sensed = None
            self.sensor_noise = None
            self.sensor_sensitivity = None
            self.signal_volt = None
            self.is_sampled = False

        self._is_sensed = b

    def sample_helper(self, dec_fact=None, f_max=None, num_modes=None, nyq_rat=2.5, numtaps_fact=21):
        '''
        Helper function to  compute a final sampling frequency from given:
            - decimation factor
            - highest frequency
            - number of modes to remain in the signal
        additionally considering
            - the ratio of the filter cutoff frequency to the final sampling frequency
            - the filter order (= number of FIR filter coefficients or numtaps)
        
        Returns:
            - fs: final sampling frequency
            - numtaps: FIR filter order
            - cutoff: FIR cutoff frequency 
        '''

        dt = self.deltat
        modal_frequencies = self.modal_frequencies

        if num_modes is not None:
            if modal_frequencies is not None:
                f_max = modal_frequencies[num_modes]
            else:
                num_modes = None
                logger.warning(f'num_modes was defined, but modal_frequencies were not provided upon init. Ignoring num_modes!')

        if f_max is not None and dec_fact is None:
            assert nyq_rat > 1
            if nyq_rat < 2:
                logger.warning(f'The sampling rate factor ({nyq_rat}) should be greater than 2.0')
            dec_fact = int(np.ceil(1 / dt / (f_max * nyq_rat)))

        elif f_max is not None and dec_fact is not None:
            f_max = None
            logger.warning('Both f_max and dec_factor were provided. Using dec_factor')

        if not isinstance(dec_fact, int):
            logger.warning('The decimation factor should be an integer')
            dec_fact = int(dec_fact)

        if num_modes is None and modal_frequencies is not None:
            if f_max is None:
                f_max = 1 / dt / dec_fact / nyq_rat
            # compute number of modes in the remaining frequency band
            # assuming the aliasing filter has a perfect roll-off
            num_modes = np.sum(modal_frequencies <= f_max)

        assert isinstance(numtaps_fact, int) and numtaps_fact >= 2

        fs = 1 / dt / dec_fact
        numtaps = numtaps_fact * dec_fact
        cutoff = fs / nyq_rat

        return fs, numtaps, cutoff

    def estimate_meas_range(self, sample_dur=None, margin=3.0, seed=None):
        '''
        imitate the process of finding an appropriate measurement range
        take a measurement for a short duration on site
        choose the maximum amplitude and increase it by a "safety" margin
        
        e.g. a 30 ... 90 s sample_dur is commonly used
        if a percentage of the total duration is to be used compute that beforehand,
        e.g.acqui.estimate_meas_range(sample_dur=acqui.duration/10, margin=1.5)
        
        here, we just choose a random sample of length sample_dur and
        evaluate all channels from this sample (assuming all channels will 
        use the same measurement range and bit resolution
        '''
        if self.is_sampled:
            logger.warning('Signal is sampled already. You should consider to start over.')
        dt = self.deltat
        dur = self.duration
        num_timesteps = self.num_timesteps
        _, signal = self.get_signal()

        if margin is None:
            margin = 3.0

        if sample_dur is None:
            sample_dur = 60  # 60 seconds default

        if sample_dur > dur:
            logger.warning('Sample duration was reduced to total duration.')
            sample_dur = dur

        # sample_dur should not be longer that 1/5th of the total signal length
        if sample_dur > dur / 5:
            logger.warning('Sample duration should be less than a fifth of the total duration.')

        rng = np.random.default_rng(seed)
        start_time = rng.uniform(0, dur - sample_dur)

        start_index = int(np.floor(start_time / dt))
        end_index = start_index + int(np.floor(sample_dur / dt))

        assert end_index <= num_timesteps

        sample = signal[:, start_index:end_index]

        max_amp = np.max(np.abs(sample))

        return max_amp * margin

    def sample(self, dec_fact=None,
               aa_order=4, aa_cutoff=None, aa_ftype='butter',
               bits=16, meas_range=None,
               duration=None):
        '''
        Sampling consists of three steps:
            - anti-aliasing filtering
            - sample-and-hold
            (- in current hardware delta-sigma-digitizers are used that work different, though)
            - quantization
            
        Not considered is:
            - accuracy, e.g. the uncertainty to sample the correct voltage (offset and gain errors)
            - sensitivity, e.g. the smallest absolute amount of change that can be detected by a measurement
                commonly a number of quantization steps is attributed to noise,
                so only a signal change greater than this level can be detected
        
        Simulation of these three steps:
            -apply an (simulated) analog anti-aliasing filter
                use a IIR filter (default butterworth 4th order)
            - then sample-and-hold -> take every nth sample
            - simulate quantization (SCXI 1600 -> 16 bit, NI PXI 6221 -> 16 bit)
         
        Additionally, the noise resulting from these operations is estimated:
            - aliasing noise
            - quantization noise
        '''
        if self.is_sampled:
            raise RuntimeError('Signal is already sampled. Start over to ensure correct SNR calculations.')

        if not self.is_sensed:
            logger.warning('The signal is still in mechanical units, no sensor has been applied.')
            signal = self.signal
        else:
            signal = self.signal_volt

        dt = self.deltat
        t_vals = self.t_vals
        fs_initial = self.sampling_rate
        sensor_sensitivity = self.sensor_sensitivity

        if duration is not None:
            assert duration <= self.duration
            N = int(duration // dt)
            signal = signal[:,:N]
            t_vals = t_vals[:N]
        else:
            N = self.num_timesteps

        num_channels = self.num_channels
        modal_frequencies = self.modal_frequencies

        # dec_fact = int(fs_initial / fs)
        assert isinstance(dec_fact, (int, np.integer))

        dt_dec = dt * dec_fact
        N_dec = int(np.floor(N / dec_fact))
        # ceil would also work, but breaks indexing for aliasing noise estimation
        # with floor though, care must be taken to shorten the time domain signal to N_dec full blocks before slicing

        if aa_cutoff is None:
            aa_cutoff = fs_initial / dec_fact / 2.5
        if aa_cutoff > fs_initial / dec_fact / 2:
            logger.warning(f"The cutoff factor of the anti-aliasing filter is larger than the target Nyquist frequency.")

        if dec_fact > 1:
            logger.info(f'Sampling signal at {1/dt_dec} Hz, using a {aa_order}. order {aa_ftype} anti-aliasing filter with a cutoff frequency of {aa_cutoff} Hz.')
            logger.info(f'Final signal size {N_dec} of {N}.')

            # fir lowpass filter
            # fir_firwin = scipy.signal.firwin(order, cutoff, fs=1 / dt)
            b, a = scipy.signal.iirfilter(
                    aa_order, [aa_cutoff / (fs_initial / 2)],
                    btype='lowpass', ftype=aa_ftype, output='ba')

            # filter signal
            # sig_filt = scipy.signal.lfilter(fir_firwin, [1.0], signal)
            sig_filt = scipy.signal.lfilter(b, a, signal)
        elif dec_fact < 1:
            raise RuntimeError("Sampling frequency must not be greater than the given sampling frequency")
        else:
            sig_filt = np.copy(signal)
        # sig_filt = np.apply_along_axis(lambda y: np.convolve(fir_firwin, y, 'same'), axis, signal)
        # sig_filt =  np.convolve(signal, fir_firwin, 'same')

        fft_filt = np.apply_along_axis(lambda y: np.fft.fft(y), 1, sig_filt)  # /np.sqrt(N)

        # decimate signal
        sig_filt_dec = np.copy(sig_filt[:, 0:N_dec * dec_fact:dec_fact])
        # correct for power loss due to decimation
        # https://en.wikipedia.org/wiki/Downsampling_(signal_processing)#Anti-aliasing_filter
        # sig_filt_dec *= dec_fact

        # fft_filt_dec = np.fft.fft(sig_filt_dec)#/np.sqrt(N_dec)

        t_dec = t_vals[0:N_dec * dec_fact:dec_fact]

        # estimate aliasing noise,
        '''
        this is based on the assumption, that the aliased and folded part
        of the spectrum is fully additive to the remaining part of the
        spectrum, which is not the case as some spectral components may be
        subtractive depending on phase differences
        -> computed noise power is a conservative estimate
        '''
        if dec_fact > 1:
            fft_filt_alias = np.zeros((num_channels, N_dec), dtype=complex)
            fft_filt_non_alias = np.zeros((num_channels, N_dec), dtype=complex)

            fft_filt_non_alias[:,:N_dec // 2] += np.copy(fft_filt[:,:1 * N_dec // 2])
            fft_filt_non_alias[:, N_dec // 2:] += np.copy(fft_filt[:, -1 * N_dec // 2:])

            pos_alias = fft_filt_alias[:,:N_dec // 2]  # is a view, any modifications are reflected to the original array
            neg_alias = fft_filt_alias[:, N_dec // 2:]  # is a view, any modifications are reflected to the original array

            for i in reversed(range(1, dec_fact)):
                # folding and aliasing all parts of the spectrum above the
                # new Nyquist frequency into the remaining spectrum
                slp = slice(i * N_dec // 2, (i + 1) * N_dec // 2)
                sln = slice(-(i + 1) * N_dec // 2, -i * N_dec // 2)

                if i % 2:  # alias and fold
                    neg = fft_filt[:, slp]
                    pos = fft_filt[:, sln]
                else:  # alias
                    pos = fft_filt[:, slp]
                    neg = fft_filt[:, sln]

                pos_alias += pos
                neg_alias += neg

            p_fft_filt_alias = np.mean(np.abs(fft_filt_alias / np.sqrt(N_dec)) ** 2, axis=1)
            p_fft_filt_non_alias = np.mean(np.abs(fft_filt_non_alias / np.sqrt(N_dec)) ** 2, axis=1)
            snr_alias = p_fft_filt_non_alias / p_fft_filt_alias

            # signal powers
            logger.debug(f'Power signal:  {np.mean(np.abs(signal)**2, axis=1)}')
            logger.debug(f'Power filtered signal: {np.mean(np.abs(sig_filt)**2, axis=1)} filtered spectrum: {np.mean(np.abs(fft_filt / np.sqrt(N))**2, axis=1)}')
            logger.debug(f'Power decimated signal: {np.mean(np.abs(sig_filt_dec)**2, axis=1)}')
            logger.debug(f'Power aliased/folded spectrum: {p_fft_filt_alias}')
            logger.debug(f'Power non-alias spectrum: {p_fft_filt_non_alias}')
            logger.debug(f'Power error due to non-additivity:  {np.mean(np.abs(sig_filt_dec)**2, axis=1) -p_fft_filt_alias-p_fft_filt_non_alias}')

            logger.debug(f'SNR alias: {snr_alias} in dB: {np.log10(snr_alias) * 10}')

        if modal_frequencies is not None:
            # proxy for: "modal characteristics are present"
            # adjust modal characteristics according to filter frf and reduced number of modes

            # compute number of modes in the remaining frequency band
            # assuming the aliasing filter has a perfect roll-off
            num_modes = np.sum(modal_frequencies <= aa_cutoff)

            modal_frequencies = modal_frequencies[:num_modes]
            # omegas = self.modal_frequencies[np.newaxis, :] * 2 * np.pi
            if dec_fact == 1:
                modal_frf = np.ones_like(modal_frequencies)
            else:
                # modal_frf = np.sum(fir_firwin[:, np.newaxis] * np.exp(-1j * omegas * np.linspace(0, dt * order, order)[:, np.newaxis]), axis=0)
                _, modal_frf = scipy.signal.freqz(b, a, modal_frequencies / fs_initial * 2 * np.pi)

            if self.modal_damping is not None:
                self.modal_damping_samp = self.modal_damping[:num_modes]
            if self.mode_shapes is not None:
                self.mode_shapes_samp = self.mode_shapes[:,:num_modes] * modal_frf / sensor_sensitivity  # scale modal coordinates by filter frf
            if self.modal_amplitudes is not None:
                self.modal_amplitudes_samp = self.modal_amplitudes[:,:num_modes] * np.abs(modal_frf) / sensor_sensitivity
            if self.modal_energies is not None:
                logger.warning('Energy reduction by FIR filters not validated.')
                self.modal_energies_samp = self.modal_energies[:,:num_modes] * np.abs(modal_frf) / sensor_sensitivity

            self.modal_frequencies_samp = modal_frequencies

        # self.re_shape[-1] = sig_filt_dec.shape[-1]

        # self.num_timesteps_samp = sig_filt_dec.shape[1]
        # assert self.num_timesteps == N_dec
        # self.deltat_samp = dt_dec
        # assert np.isclose(t_dec[1] - t_dec[0], self.deltat)
        if dec_fact > 1:
            # Wikipedia - SNR:  "noise goes down as the square root of the number of averaged samples. "
            self.snr[1] /= np.sqrt(dec_fact)
            self.update_snr(p_fft_filt_alias, p_fft_filt_non_alias)

        # Quantization
        if meas_range is None:
            meas_range = 2 ** bits / (2 ** bits - 2) * np.max(np.abs(sig_filt_dec))
            # factor adds one quantization level to the measurement range
            # to avoid clipping when meas_range was not provided

        bits_effective = np.log2(np.max(np.abs(sig_filt_dec)) / meas_range * 2 ** bits)

        logger.info(f'Quantizing signal in a measurement range of ± {meas_range} V with {bits_effective:1.3f} effective bits.')

        Delta_s = 2 * meas_range / 2 ** bits
        # symmetric quantization does not include 0, so we have to
        # add Delta_s / 2 before quantization, and substract it afterwards
        # division by Delta_s maps to the range [-2**b, 2**b],
        # which is subsequently rounded to integers
        # original signal range is restored by multiplication with Delta_s
        sig_filt_dec_quant = np.rint((sig_filt_dec + Delta_s / 2) / Delta_s) * Delta_s - Delta_s / 2
        # apply clipping of excess values
        ind1 = sig_filt_dec_quant < -meas_range + Delta_s / 2
        sig_filt_dec_quant[ind1] = -meas_range + Delta_s / 2
        ind2 = sig_filt_dec_quant > meas_range - Delta_s / 2
        sig_filt_dec_quant[ind2] = meas_range - Delta_s / 2
        ind = np.logical_or(ind1, ind2)
        if ind.any():
            logger.warning(f'Clipping due to quantization occured for {np.sum(ind)} out of {sig_filt_dec.size} samples')

        # margin_quant_norm = (meas_range - np.max(np.abs(sig_filt_dec), axis=1)) / meas_range

        n_q = sig_filt_dec - sig_filt_dec_quant

        P_S = np.mean(sig_filt_dec ** 2, axis=1)
        P_N = np.mean(n_q ** 2, axis=1)

        logger.debug(f'Power quantized signal: {P_S}')
        logger.debug(f'Power quantization noise: {P_N}')

        snr_quant = 10 * np.log10(P_S / P_N)
        logger.debug(f'SNR quantization: {P_S / P_N} in dB: {snr_quant}')

        self.update_snr(P_N, P_S)

        self.t_vals_samp = t_dec
        self.signal_samp = sig_filt_dec_quant / sensor_sensitivity

        self.bits_effective = bits_effective

        self.is_sampled = True

        return

    def reset_snr(self):
        signal = self.signal
        num_channels = self.num_channels
        self.snr = [np.mean(signal ** 2, axis=1), np.zeros((num_channels,))]

    def estimate_snr(self, n_lines_fac=1):
        _, signal = self.get_signal()
        num_timesteps = self.num_timesteps
        n_lines = num_timesteps // n_lines_fac

        # it increase variance and does not improve the result in any other sense
        # when using less than the maximally possible number of segments
        # n_segments = max(num_timesteps // (n_lines // 2), 1)
        n_segments = max(num_timesteps // (n_lines), 1)
        fs = self.sampling_rate

        num_channels = self.num_channels
        # ref_channels = list(range(num_channels))

        # signals = self.signals

        psd_matrix_shape = (num_channels,
                            num_channels,
                            n_lines // 2 + 1)

        psd_matrix = np.empty(psd_matrix_shape, dtype=complex)

        for channel_1 in range(num_channels):
            for channel_2 in range(num_channels):
                # compute spectrum according to welch, with automatic application of a window and scaling
                # specrum scaling compensates windowing by dividing by window(n_lines).sum()**2
                # density scaling divides by fs * window(n_lines)**2.sum()
                _, Pxy_den = scipy.signal.csd(signal[channel_1,:],
                                              signal[channel_2,: ],
                                              fs,
                                              # nperseg=n_lines // 2,
                                              nperseg=n_lines,
                                              # nfft=n_lines,
                                              noverlap=0,
                                              return_onesided=True,
                                              scaling='density',)

                if channel_1 == channel_2:
                    assert np.isclose(Pxy_den.imag, 0).all()
                    Pxy_den.imag = 0
                # compensate averaging over segments (for power equivalence segments should be summed up)
                Pxy_den *= n_segments
                # reverse 1/Hz of scaling="density"
                Pxy_den *= fs
                # compensate onesided
                Pxy_den /= 2
                # compensate zero-padding
                # Pxy_den /= 2
                # compensate energy loss through short segments
                Pxy_den *= n_lines

                psd_matrix[channel_1, channel_2,:] = Pxy_den

        s_vals_psd = np.empty((num_channels, n_lines // 2 + 1))
        for k in range(n_lines // 2 + 1):
            # might use only real part to account for slightly asynchronous data
            # see [Au (2017): OMA, Chapter 7.5]
            s_vals_psd[:, k] = np.linalg.svd(psd_matrix[:,:, k], True, False)

        self.s_vals_psd = s_vals_psd
        self.snr_db_est = 10 * np.log10(np.mean(s_vals_psd[0,:]) / np.mean(s_vals_psd[1,:]))

        return

    def update_snr(self, power_noise, power_no_noise=None):
        '''
        updates stored signal and noise power values
        
        power_noise should be an array of shape (num_channels,)
        
        power_no_noise should be provided, if the signal power might have been
        modified in another different way prior to adding noise and should
        be an array of shape (num_channels,)
        
        '''

        if power_no_noise is not None:
            # total (unscaled) signal power consists of signal power and all added noise power
            p_total = self.snr[0] + self.snr[1]
            logger.debug(f'Total power theoretic: {p_total}')
            logger.debug(f'Total power actual: {power_no_noise}')
            logger.debug(f'Noise power: {power_noise}')
            # factor, by which signal was modified externally
            sig_factor = power_no_noise / p_total
            # print(sig_factor)
            # scale signal and noise power by factor
            self.snr[1] *= sig_factor  # **0.5
            self.snr[0] *= sig_factor  # **0.5
            logger.debug(f'Total power scaled: {self.snr[0] + self.snr[1]}')

            # print(power_no_noise / (self.snr[0] + self.snr[1]))

        # updated noise power
        # E[(X+Y)^2] = E[X^2] + 2E[X]E[Y] + E[Y^2], where E[X],E[Y] = 0
        self.snr[1] += power_noise
        logger.debug(f'Total noise power: {self.snr[1]}')
        logger.debug(f'Current SNR: {self.snr_db} dB')

    def add_noise(self, snr_db=None, noise_power=None, ref_sig='channel', seed=None):
        '''
        Add noise to a signal preferably before, possibly after sampling
        snr_db or noise power can be global or per channel
        if snr_db is given it overrides noise_power
        
        reference signal is "mean" or "channel"
        
        returns power_signal, power_noise
        
        '''
        _, signal = self.get_signal()
        num_channels = self.num_channels
        num_timesteps = self.num_timesteps

        channel_powers = np.mean(signal ** 2, axis=1)
        if ref_sig == 'channel':
            power_signal = channel_powers
        elif ref_sig == 'mean':
            power_signal = np.full((num_channels,), np.mean(signal ** 2))
        else:
            raise ValueError(f'ref_sig {ref_sig} is neither "mean" nor "channel"')

        logger.debug(f'Signal power: {power_signal}')

        if noise_power is not None:
            if isinstance(noise_power, (float, int)):
                noise_power = np.full((num_channels,), noise_power)
            elif isinstance(noise_power, (list, tuple, np.ndarray)):
                noise_power = np.array(noise_power)
                assert noise_power.ndim == 1
                assert noise_power.size == num_channels
            else:
                raise ValueError(f'Noise power must be either an iterable of length num_channels or a scalar. It is {noise_power}')
            if snr_db is None:
                info_str = f'Applying constant noise.'

        if snr_db is not None:
            if not isinstance(snr_db, (float, int)):
                raise NotImplementedError('Adding noise by per channel definition of SNR is currently not supported.')
            if noise_power is not None:
                info_str = f'Applying constant noise and proportional noise.'
            else:
                noise_power = np.zeros((num_channels,))
                info_str = f'Applying proportional noise of {snr_db} dB.'
            if snr_db > 10:
                logger.warning(f'You are adding only {100/(10**(snr_db/10))} % of signal noise. Try using negative snr_db values.')

            snr = 10 ** (snr_db / 10)
            noise_power += power_signal / snr

        if snr_db is None and noise_power is None:
            raise ValueError('Either snr_db or noise_power (or both) must be provided.')
        # add noise
        # TODO: try spectral- and correlation-based white noise generation
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, np.sqrt(noise_power), (num_timesteps, num_channels)).T
        power_noise = np.mean(noise ** 2, axis=1)

        logger.info(f'{info_str} Final average noise power: {np.mean(power_noise):1.3g} (signal: {np.mean(power_signal):1.3g})')

        logger.debug(f'snr_actual = {power_signal / power_noise} = {10 * np.log10(power_signal / power_noise)} dB')

        signal += noise

        self.update_snr(power_noise, channel_powers)
        if self.is_sampled:
            self.signal_samp = signal
        elif self.is_sensed:
            self.signal_volt = signal
        else:
            self.signal = signal

        return power_signal, power_noise

    def get_signal(self):
        if self.is_sampled:
            t_vals = self.t_vals_samp
            signal = self.signal_samp
            logger.debug('Using a sampled signal')
        elif self.is_sensed:
            t_vals = self.t_vals
            signal = self.signal_volt
            logger.debug('Using a sensed signal')
        else:
            t_vals = self.t_vals
            signal = self.signal
            logger.debug('Using a response signal')

        re_shape = np.copy(self.re_shape)
        re_shape[-1] = self.num_timesteps

        return t_vals, signal.reshape(re_shape)

    def to_prep_data(self):

        if not self.is_sampled:
            logger.warning(f'This signal has not been sampled already. Proceeding anyways.')

        t_vals, signal = self.get_signal()
        assert signal.ndim == 2
        fs = self.sampling_rate
        setup_name = self.jobname

        channel_defs = self.channel_defs  # list of (node, dof, quantity)

        if channel_defs is not None:
            disp_channels = np.where(channel_defs[:, 2] == 0)[0]
            velo_channels = np.where(channel_defs[:, 2] == 1)[0]
            accel_channels = np.where(channel_defs[:, 2] == 2)[0]

            channel_headers = []
            for node, dof, quantity in channel_defs:
                channel_headers.append(f'{node} {["x", "y", "z"][dof]} ')
                # channel_headers.append(f'{["dsp", "vel", "acc", ][quantity]} {node} {["x", "y", "z"][dof]} ')

        else:
            disp_channels = None
            velo_channels = None
            accel_channels = None
            channel_headers = None

        kwargs = {'signals': signal.T,
                  'sampling_rate': fs,
                  'accel_channels': accel_channels,
                  'velo_channels': velo_channels,
                  'disp_channels': disp_channels,
                  'setup_name': setup_name,
                  'channel_headers': channel_headers}

        return kwargs

    def save(self, fpath, differential=None):

        fdir, file = os.path.split(fpath)
        fname, ext = os.path.splitext(file)

        logger.info(f'Saving Acquire object to {fpath}')

        out_dict = {}

        out_dict['self.jobname'] = self.jobname
        out_dict['self.re_shape'] = self.re_shape
        out_dict['self.channel_defs'] = self.channel_defs
        out_dict['self.snr'] = self.snr
        out_dict['self.s_vals_psd'] = self.s_vals_psd
        out_dict['self.snr_db_est'] = self.snr_db_est

        out_dict['self._is_sensed'] = self._is_sensed
        out_dict['self._is_sampled'] = self.is_sampled

        if differential is None:
            out_dict['self.signal'] = self.signal
            out_dict['self.t_vals'] = self.t_vals
            out_dict['self.modal_frequencies'] = self.modal_frequencies
            out_dict['self.modal_damping'] = self.modal_damping
            out_dict['self.mode_shapes'] = self.mode_shapes
            out_dict['self.modal_energies'] = self.modal_energies
            out_dict['self.modal_amplitudes'] = self.modal_amplitudes
        else:
            assert differential in ['sensed', 'sampled', 'nosigs']

        if differential == 'sensed' or (differential is None and self._is_sensed):
            assert self._is_sensed
            out_dict['self.signal_volt'] = self.signal_volt
            out_dict['self.t_vals'] = self.t_vals
            out_dict['self.mode_shapes_sensed'] = self.mode_shapes_sensed
            out_dict['self.modal_energies_sensed'] = self.modal_energies_sensed
            out_dict['self.modal_amplitudes_sensed'] = self.modal_amplitudes_sensed
            out_dict['self.sensor_noise'] = self.sensor_noise
            out_dict['self.sensor_sensitivity'] = self.sensor_sensitivity

        if differential == 'sampled' or (differential is None and self.is_sampled):
            assert self.is_sampled
            out_dict['self.t_vals_samp'] = self.t_vals_samp
            out_dict['self.signal_samp'] = self.signal_samp
            out_dict['self.modal_frequencies_samp'] = self.modal_frequencies_samp
            out_dict['self.modal_damping_samp'] = self.modal_damping_samp
            out_dict['self.mode_shapes_samp'] = self.mode_shapes_samp
            out_dict['self.modal_energies_samp'] = self.modal_energies_samp
            out_dict['self.modal_amplitudes_samp'] = self.modal_amplitudes_samp
            out_dict['self.bits_effective'] = self.bits_effective

        if differential == 'nosigs':
            out_dict['self.modal_frequencies_samp'] = self.modal_frequencies_samp
            out_dict['self.modal_damping_samp'] = self.modal_damping_samp
            out_dict['self.mode_shapes_samp'] = self.mode_shapes_samp
            out_dict['self.modal_energies_samp'] = self.modal_energies_samp
            out_dict['self.modal_amplitudes_samp'] = self.modal_amplitudes_samp
            out_dict['self.bits_effective'] = self.bits_effective

        np.savez_compressed(fpath, **out_dict)

    @classmethod
    def load(cls, fpath, differential=None):
        # ..TODO:: implement differential loading with dummy signal
        assert os.path.exists(fpath)

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

        logger.info('Now loading previous results from  {}'.format(fpath))

        fdir, file = os.path.split(fpath)
        fname, ext = os.path.splitext(file)

        in_dict = np.load(fpath, allow_pickle=True)

        jobname = in_dict['self.jobname'].item()
        channel_defs = validate_array(in_dict['self.channel_defs'])
        re_shape = validate_array(in_dict['self.re_shape'])

        if differential is None:
            signal = validate_array(in_dict['self.signal'])
            t_vals = validate_array(in_dict['self.t_vals'])
            modal_frequencies = validate_array(in_dict['self.modal_frequencies'])
            modal_damping = validate_array(in_dict['self.modal_damping'])
            mode_shapes = validate_array(in_dict['self.mode_shapes'])
            modal_energies = validate_array(in_dict['self.modal_energies'])
            modal_amplitudes = validate_array(in_dict['self.modal_amplitudes'])
        else:
            if not differential in ['sensed', 'sampled', 'nosigs', 'blablabla']:
                logger.warning(f'Unsupported value passed for differential={differential}')
            signal = np.full(re_shape, fill_value=np.nan)
            t_vals = np.full(re_shape[-1], fill_value=np.nan)
            modal_frequencies = None
            modal_damping = None
            mode_shapes = None
            modal_energies = None
            modal_amplitudes = None

        acquire = cls(t_vals, signal, None, channel_defs,
                 modal_frequencies, modal_damping, mode_shapes,
                 modal_energies, modal_amplitudes,
                 jobname,)
        assert np.all(acquire.re_shape == re_shape)

        acquire.s_vals_psd = validate_array(in_dict['self.s_vals_psd'])
        acquire.snr_db_est = validate_array(in_dict['self.snr_db_est'])
        acquire.snr = validate_array(in_dict['self.snr'])

        acquire._is_sensed = validate_array(in_dict['self._is_sensed'])
        acquire._is_sampled = validate_array(in_dict['self._is_sampled'])

        if (acquire._is_sensed and differential is None) or differential == 'sensed':
            assert acquire._is_sensed
            acquire.t_vals = validate_array(in_dict['self.t_vals'])
            acquire.signal_volt = validate_array(in_dict['self.signal_volt'])
            acquire.mode_shapes_sensed = validate_array(in_dict['self.mode_shapes_sensed'])
            acquire.modal_energies_sensed = validate_array(in_dict['self.modal_energies_sensed'])
            acquire.modal_amplitudes_sensed = validate_array(in_dict['self.modal_amplitudes_sensed'])
            acquire.sensor_noise = validate_array(in_dict['self.sensor_noise'])
            acquire.sensor_sensitivity = validate_array(in_dict['self.sensor_sensitivity'])

        if (acquire.is_sampled and differential is None) or differential == 'sampled':
            assert acquire.is_sampled
            acquire.t_vals_samp = validate_array(in_dict['self.t_vals_samp'])
            acquire.signal_samp = validate_array(in_dict['self.signal_samp'])
            acquire.modal_frequencies_samp = validate_array(in_dict['self.modal_frequencies_samp'])
            acquire.modal_damping_samp = validate_array(in_dict['self.modal_damping_samp'])
            acquire.mode_shapes_samp = validate_array(in_dict['self.mode_shapes_samp'])
            acquire.modal_energies_samp = validate_array(in_dict['self.modal_energies_samp'])
            acquire.modal_amplitudes_samp = validate_array(in_dict['self.modal_amplitudes_samp'])
            acquire.bits_effective = validate_array(in_dict['self.bits_effective'])

        if differential == 'nosigs':
            acquire.signal_samp = acquire.signal
            acquire.t_vals_samp = acquire.t_vals
            acquire.modal_frequencies_samp = validate_array(in_dict['self.modal_frequencies_samp'])
            acquire.modal_damping_samp = validate_array(in_dict['self.modal_damping_samp'])
            acquire.mode_shapes_samp = validate_array(in_dict['self.mode_shapes_samp'])
            acquire.modal_energies_samp = validate_array(in_dict['self.modal_energies_samp'])
            acquire.modal_amplitudes_samp = validate_array(in_dict['self.modal_amplitudes_samp'])
            acquire.bits_effective = validate_array(in_dict['self.bits_effective'])

        return acquire


def supylabel2(fig, s, **kwargs):
    defaults = {
        "x": 0.98,
        "y": 0.5,
        "horizontalalignment": "center",
        "verticalalignment": "center",
        "rotation": "vertical",
        "rotation_mode": "anchor",
        "size": plt.rcParams["figure.labelsize"],  # matplotlib >= 3.6
        "weight": plt.rcParams["figure.labelweight"],  # matplotlib >= 3.6
    }
    kwargs["s"] = s
    # kwargs = defaults | kwargs  # python >= 3.9
    kwargs = {**defaults, **kwargs}
    fig.text(**kwargs)


def plot_compare_signals(signal_1, signal_2,
                         fs, N,
                         fs2=None, N2=None):

    num_channels, num_timesteps = signal_1.shape
    _, num_timesteps2 = signal_2.shape

    if fs2 is None:
        fs2 = fs
    if N2 is None:
        N2 = N

    t_vals1 = np.linspace(0, num_timesteps / fs, num_timesteps)
    t_vals2 = np.linspace(0, num_timesteps2 / fs2, num_timesteps2)

    fig, axes = plt.subplots(nrows=num_channels,
                             ncols=2,
                             sharey='col',
                             sharex='col')
    axest = axes[:, 0]
    axesf = axes[:, 1]

    for channel in range(num_channels):
        axt = axest[channel]
        axt.plot(t_vals1, signal_1[channel,:], alpha=0.5)
        axt.plot(t_vals2, signal_2[channel,:], alpha=0.5)

        axf = axesf[channel]
        freq_wl1, spec_wl1 = scipy.signal.csd(signal_1[channel,:], signal_1[channel,:], fs, nperseg=N, scaling='density')
        axf.plot(freq_wl1, 10 * np.log10(np.abs(spec_wl1)), alpha=0.5,)
        freq_wl2, spec_wl2 = scipy.signal.csd(signal_2[channel,:], signal_2[channel,:], fs2, nperseg=N2, scaling='density')
        axf.plot(freq_wl2, 10 * np.log10(np.abs(spec_wl2)), alpha=0.5,)
        # axf.set_yscale('log')
        # axf.axvline(1 / (2 * np.pi * DTC), color='grey')
        axf.yaxis.set_label_position("right")
        axf.yaxis.tick_right()
    axt.set_xlim((0, max(num_timesteps / fs, num_timesteps2 / fs2)))
    axf.set_xlim((min(freq_wl1[1], freq_wl2[1]), max(freq_wl1[-1], freq_wl2[-1])))
    axf.set_xscale('log')
    axf.set_xlabel('Frequency [\si{\hertz}]')
    axt.set_xlabel('Time [\si{\second}]')
    fig.supylabel('Acceleration [\si{\metre\per\second\squared}]')
    supylabel2(fig, 'PSD [\si{\decibel}]')
    return fig


def sensor_position(num_sensors, num_nodes, method):
    assert method in ['lumped', 'distributed']

    # put the tip node always as the last channel
    num_sensors -= 1

    all_positions = np.arange(5, num_nodes, 5)
    num_positions = all_positions.shape[0]
    assert num_positions >= num_sensors
    num_clusters = int(np.floor(num_positions / num_sensors))

    if method == 'distributed':
        clusters = np.reshape(all_positions[:num_sensors * (num_clusters)], (num_sensors, num_clusters)).T
        last_setup = all_positions[-(num_sensors - 1) * (num_clusters) - 1::num_clusters]
    elif method == 'lumped':
        clusters = np.reshape(all_positions[:num_sensors * (num_clusters)], (num_clusters, num_sensors))
        last_setup = np.reshape(all_positions[-num_sensors:], (1, num_sensors))

    clusters = np.vstack((clusters, last_setup))

    # put the tip node always as the last channel (reference channel)
    clusters = np.hstack((clusters, np.ones((num_clusters + 1, 1), dtype=int) * 201))

    return clusters


def ashift(array):
    return np.fft.fftshift(np.abs(array))


def quantify_aliasing_noise(sig, dec_fact, dt=1, numtaps_fact=41, nyq_rat=2.5 , do_plot=False):
    import scipy.signal

    N = sig.size
    fs = 1 / dt

    dt_dec = dt * dec_fact
    N_dec = int(np.floor(N / dec_fact))
    # ceil would also work, but breaks indexing for aliasing noise estimation
    # with floor though, care must be taken to shorten the time domain signal to N_dec full blocks before slicing

    cutoff = fs / nyq_rat / dec_fact
    numtaps = numtaps_fact * dec_fact

    t = np.linspace(0, (N - 1) * dt, N)

    # compute spectrum
    if do_plot:
        fft_sig = np.fft.fft(sig)  # /np.sqrt(N)

    # fir lowpass filter
    # fir_firwin = scipy.signal.firwin(numtaps, cutoff/fs*2)
    fir_firwin = scipy.signal.firwin(numtaps, cutoff, fs=fs)

    # filter signal
    sig_filt = np.convolve(sig, fir_firwin, 'same')

    fft_filt = np.fft.fft(sig_filt)  # /np.sqrt(N)
    fft_filt_alias = np.copy(fft_filt)

    # decimate signal
    sig_filt_dec = np.copy(sig_filt[0:N_dec * dec_fact:dec_fact])
    # correct for power loss due to decimation
    # https://en.wikipedia.org/wiki/Downsampling_(signal_processing)#Anti-aliasing_filter
    sig_filt_dec *= dec_fact

    # fft_dec = np.fft.fft(sig_dec)#/np.sqrt(N_dec)
    fft_filt_dec = np.fft.fft(sig_filt_dec)  # /np.sqrt(N_dec)

    t_dec = t[0::dec_fact]
    if do_plot:
        plt.figure()
        freq = np.fft.fftshift(np.fft.fftfreq(N, d=dt))
        plt.semilogy(freq, ashift(fft_sig), label='original signal', color='whitesmoke')
        plt.semilogy(freq, ashift(fft_filt), label='filtered signal', color='lightgrey')

    fft_filt_alias = np.zeros(N_dec, dtype=complex)
    fft_filt_non_alias = np.zeros(N_dec, dtype=complex)

    fft_filt_non_alias[:N_dec // 2] += np.copy(fft_filt[:1 * N_dec // 2])
    fft_filt_non_alias[N_dec // 2:] += np.copy(fft_filt[-1 * N_dec // 2:])

    pos_alias = fft_filt_alias[:N_dec // 2]
    neg_alias = fft_filt_alias[N_dec // 2:]

    for i in reversed(range(1, dec_fact)):
        slp = slice(i * N_dec // 2, (i + 1) * N_dec // 2)
        sln = slice(-(i + 1) * N_dec // 2, -i * N_dec // 2)

        if i % 2:  # alias and fold
            neg = fft_filt[slp]
            pos = fft_filt[sln]

            if do_plot:
                ls = 'dashed'
                freqp = np.fft.fftfreq(N, d=dt)[sln]
                freqn = np.fft.fftfreq(N, d=dt)[slp]
        else:  # alias
            pos = fft_filt[slp]
            neg = fft_filt[sln]

            if do_plot:
                ls = 'dotted'
                freqp = np.fft.fftfreq(N, d=dt)[slp]
                freqn = np.fft.fftfreq(N, d=dt)[sln]

        pos_alias += pos
        neg_alias += neg
        if do_plot:
            plt.semilogy(freqp, np.abs(pos), color='k', ls=ls)  # ,marker='x')
            plt.semilogy(freqn, np.abs(neg), color='k', ls=ls)  # ,marker='x')
            plt.semilogy(np.fft.fftshift(np.fft.fftfreq(N_dec, dt_dec)), np.copy(ashift(fft_filt_alias)), color='k', ls=ls)
    else:
        if do_plot:
            plt.semilogy(np.fft.fftshift(np.fft.fftfreq(N_dec, dt_dec)), np.copy(ashift(fft_filt_alias + fft_filt_non_alias)), color='k', label='alias')

    p_fft_filt_alias = np.mean(np.abs(fft_filt_alias / np.sqrt(N_dec)) ** 2)
    p_fft_filt_non_alias = np.mean(np.abs(fft_filt_non_alias / np.sqrt(N_dec)) ** 2)
    snr_alias = p_fft_filt_non_alias / p_fft_filt_alias

    if do_plot:
        sig_filt_alias = np.fft.ifft(fft_filt_alias)
        sig_filt_nonalias = np.fft.ifft(fft_filt_non_alias)

        # signal powers
        p_sig = np.mean(np.abs(sig) ** 2)
        p_fft = np.mean(np.abs(fft_sig / np.sqrt(N)) ** 2)
        p_sig_filt = np.mean(np.abs(sig_filt) ** 2)
        p_fft_filt = np.mean(np.abs(fft_filt / np.sqrt(N)) ** 2)
        p_sig_filt_dec = np.mean(np.abs(sig_filt_dec) ** 2)
        p_fft_filt_dec = np.mean(np.abs(fft_filt_dec / np.sqrt(N_dec)) ** 2)
        p_sig_filt_alias = np.mean(np.abs(sig_filt_alias) ** 2)
        p_fft_filt_alias = np.mean(np.abs(fft_filt_alias / np.sqrt(N_dec)) ** 2)
        p_sig_filt_non_alias = np.mean(np.abs(sig_filt_nonalias) ** 2)
        p_fft_filt_non_alias = np.mean(np.abs(fft_filt_non_alias / np.sqrt(N_dec)) ** 2)
        print('Power signal: ', p_sig)
        print('Power filtered signal: ', p_sig_filt, 'filtered spectrum: ', p_fft_filt)
        print('Power decimated signal: ', p_sig_filt_dec)
        print('Power alias spectrum: ', p_fft_filt_alias)
        print('Power non-alias spectrum: ', p_fft_filt_non_alias)
        print('SNR alias: ', snr_alias, ' in dB: ', np.log10(snr_alias) * 10)

        freq_dec = np.fft.fftshift(np.fft.fftfreq(N_dec, d=dt_dec))
        plt.semilogy(freq_dec, ashift(fft_filt_dec), ls='dashed', color='darkgrey', label='filtered decimated')

        plt.xlim(xmin=0, xmax=fs / 2)
        plt.axvline(fs / 2 / dec_fact, color='k')
        plt.axvline(cutoff, ls='dashed', color='k')
        plt.legend()

    return sig_filt_dec, snr_alias


def system_signal_decimate(N, dec_fact, numtap_fact, nyq_rat, do_plot=False, **kwargs):
    fmax = 400
    df = fmax / (N // 2 + 1)
    dt = 1 / df / N
    fs = 1 / dt
    # signal processing parameters
    # dec_fact=4
    # dec_fact=int(inputs[i,1])

    # allocation
    omegas = np.linspace(0, fmax, N // 2 + 1, False) * 2 * np.pi
    t = np.linspace(0, (N - 1) * dt, N)
    frf = np.zeros((N // 2 + 1,), dtype=complex)
    assert df * 2 * np.pi == (omegas[-1] - omegas[0]) / (N // 2 + 1 - 1)

    # system parameters
    L = 200
    E = 2.1e11
    rho = 7850
    num_modes = int(np.floor((fmax * 2 * np.pi * np.sqrt(rho / E) * L / np.pi * 2 + 1) / 2))

    # system generation
    ks = np.random.rand(num_modes) + 0.5
    omegans = (2 * np.arange(1, num_modes + 1) - 1) / 2 * np.pi / L * np.sqrt(E / rho)
    zetas = np.zeros_like(omegans)
    zetas[:] = 0.001
    for k, omegan, zeta in zip(ks, omegans, zetas):
        frf += 1 / (k * (1 + 2 * 1j * zeta * omegas / omegan - (omegas / omegan) ** 2))

    # random input
    phase = (np.random.rand(N // 2 + 1) - 0.5)
    Pomega = np.ones_like(frf) * np.exp(1j * phase * 2 * np.pi)

    # time domain signal
    sig = np.fft.irfft(frf * Pomega)

    return (np.log10(quantify_aliasing_noise(sig, dec_fact, dt, numtap_fact, nyq_rat, do_plot)[1]) * 10,)


def uq_dec_noise():
    import seaborn
    import pathlib
    import tempfile
    from polyuq import data_manager

    num_samples = 10

    title = 'decimation_noise'
    savefolder = str(pathlib.Path.cwd() / 'polyuq_results') + '/'
    result_dir = savefolder

    if not os.path.exists(result_dir + title + '.nc') or False:
        dm = data_manager.DataManager(title=title, working_dir=tempfile.gettempdir(),
                                   result_dir=result_dir,
                                   overwrite=True)
        dm.generate_sample_inputs(names=['N',
                                                   'dec_fact',
                                                   'numtap_fact',
                                                   'nyq_rat'],
                                            distributions=[('choice', [2 ** np.arange(5, 18)]),
                                                           ('integers', (2, 15)),
                                                           ('integers', (5, 61)),
                                                           ('uniform', (2, 4)),
                                                           ],
                                            num_samples=num_samples)

        dm.post_process_samples(db='in', names=['N', 'dec_fact', 'numtap_fact', 'nyq_rat'],)
    elif False:
        # add_samples
        dm = data_manager.DataManager.from_existing(dbfile_in="".join(
            i for i in title if i not in "\\/:*?<>|") + '.nc', result_dir=result_dir)
        dm.enrich_sample_set(total_samples=5000)

    elif False:
        # evaluate input samples
        dm = data_manager.DataManager.from_existing(dbfile_in="".join(
            i for i in title if i not in "\\/:*?<>|") + '.nc', result_dir=result_dir)
        dm.evaluate_samples(func=system_signal_decimate,
                                      arg_vars={'N':'N', 'dec_fact':'dec_fact',
                                                'numtap_fact':'numtap_fact', 'nyq_rat':'nyq_rat'},
                                      ret_names=['snrs'],
                                      chwdir=True, dry_run=False)
    elif True:
        dm = data_manager.DataManager.from_existing(dbfile_in="".join(
            i for i in title if i not in "\\/:*?<>|") + '.nc', result_dir=result_dir)
        with matplotlib.rc_context(rc=print_context_dict):
            dm.post_process_samples(db='merged', scales=['log', 'linear', 'linear', 'linear', 'linear'])
        dm.clear_locks()


def verify_noise_power():

    from model import mechanical
    import pathlib
    import tempfile

    skip_existing = True
    working_dir = tempfile.gettempdir()
    result_dir = str(pathlib.Path.cwd() / 'polyuq_results') + '/'
    jid = 'filter_example'
    # jid='acquire_example'

    num_nodes = 21
    damping = 0.01
    num_modes = 20
    dt_fact = 0.01
    num_cycles = 26
    meas_nodes = list(range(1, 22))

    # load/generate  mechanical object:
    # ambient response of a 20 dof rod,
    f_scale = 1000
    d0 = None
    savefolder = os.path.join(result_dir, jid, 'ambient')

    if not os.path.exists(os.path.join(savefolder, f'{jid}_mechanical.npz')) or not skip_existing:
        ansys = mechanical.Mechanical.start_ansys(working_dir, jid)
        mech = mechanical.generate_mdof_time_hist(ansys=ansys, num_nodes=num_nodes, damping=damping,
                                                  num_modes=num_modes, d0=d0, f_scale=f_scale,
                                                  dt_fact=dt_fact, num_cycles=num_cycles,
                                                  meas_nodes=meas_nodes,)
        mech.save(savefolder)
    else:
        mech = mechanical.MechanicalDummy.load(jid, savefolder)
        # mech = mechanical.Mechanical.load(jid, savefolder)

    # generate analytical FRF
    if isinstance(mech, mechanical.Mechanical):
        N = 2 ** 15
        fs = 256

        _, frf = mech.frequency_response(N, fs // 2)

        phase = (np.random.rand(N // 2 + 1) - 0.5)
        Pomega = 120 * np.ones_like(phase) * np.exp(1j * phase * 2 * np.pi)
        sig = np.empty((N, len(mech.meas_nodes)))
        for channel in range(frf.shape[1]):
            sig[:, channel] = np.fft.irfft(frf[:, channel] * Pomega)  # ambient response
            # sig[:, channel] = np.fft.irfft(frf[:, channel]) # IRF
            # sig[:, channel] = np.fft.irfft(Pomega) # white noise

    # pass mechanical object to acquire
    acqui = Acquire.init_from_mech(mech, channel_defs='uz')

    # use analytical example signals (ambient or impulse response, white noise)
    # acqui = Acquire(np.linspace(0,N/fs,N), sig)

    if False:  # Noise power verification
        print(acqui.snr_db)
        p_sig = 1  # original signal power
        ps = 1  # total signal power
        all_noise = 0  # sum of noise powers = total noise power
        sfac = 1  # signal factor
        for i in range(10):
            sfac = np.random.rand() + 0.5  # random signal power modification
            snr_db = (np.random.rand() - 0.5) * 10  # random snr for adding noise
            snr = 10 ** (snr_db / 10)
            p_sig *= sfac
            ps *= sfac
            all_noise *= sfac
            p_noise = ps / snr
            all_noise += p_noise
            ps += p_noise

            acqui.signal *= sfac
            acqui.add_noise(snr_db)
            print(np.nanmean(acqui.snr_db), 10 * np.log10(p_sig / all_noise))

        return
    elif False:
        # verify noise addition and snr updating
        acqui.add_noise(snr_db=10)
        print(np.nanmean(acqui.snr_db))
        acqui.add_noise(snr_db=10, ref_sig='mean')
        print(np.nanmean(acqui.snr_db))
        acqui.add_noise(noise_power=5e-6)
        print(np.nanmean(acqui.snr_db))
        noise_power = np.random.rand(acqui.num_channels) * 1e-5
        acqui.add_noise(noise_power=noise_power)
        print(np.nanmean(acqui.snr_db))
        acqui.add_noise(snr_db=10, noise_power=noise_power, ref_sig='mean')
        return
    else:
        plt.figure()
        channel = -1
        # verify snr updating with changing signal power
        # verify sampling noise estimation
        acqui.add_noise(snr_db=30)
        print(np.nanmean(acqui.snr_db))  # should be 10*log10(1/(0.001*1))~30
        plt.plot(acqui.t_vals, acqui.signal[channel,:], alpha=.3, label='noise1')
        with LoggingContext(logger, level=logging.INFO):
            acqui.sample(*acqui.sample_helper(8, numtaps_fact=5), 8, acqui.estimate_meas_range(acqui.duration / 10, 20))  # adds 30 db aliasing noise and about 30 db quantization noise
        print(np.nanmean(acqui.snr_db))  # should be 10*log10(1/(0.001*3))~25.23
        plt.plot(acqui.t_vals, acqui.signal[channel,:], alpha=.3, label='sample')
        acqui.add_noise(snr_db=30)
        print(np.nanmean(acqui.snr_db))  # should be 10*log10(1/(0.001*4))~23.98
        plt.plot(acqui.t_vals, acqui.signal[channel,:], alpha=.3, label='noise2')
        plt.legend()
        plt.show()
        return


def filter_example():

    from model import mechanical
    import pathlib
    import tempfile
    skip_existing = True
    working_dir = tempfile.gettempdir()
    result_dir = str(pathlib.Path.cwd() / 'polyuq_results') + '/'
    jid = 'filter_example'
    # jid='acquire_example'

    num_nodes = 21
    damping = 0.01
    num_modes = 20
    dt_fact = 0.01
    num_cycles = 26
    meas_nodes = list(range(1, 22))

    # load/generate  mechanical object:
    if True:
        # ambient response of a 20 dof rod,
        f_scale = 1000
        d0 = None
        savefolder = os.path.join(result_dir, jid, 'ambient')
    else:
        # impulse response of a 20 dof rod
        f_scale = None
        d0 = 1
        savefolder = os.path.join(result_dir, jid, 'impulse')

    if not os.path.exists(os.path.join(savefolder, f'{jid}_mechanical.npz')) or not skip_existing:
        ansys = mechanical.Mechanical.start_ansys(working_dir, jid)
        mech = mechanical.generate_mdof_time_hist(ansys=ansys, num_nodes=num_nodes, damping=damping,
                                                  num_modes=num_modes, d0=d0, f_scale=f_scale,
                                                  dt_fact=dt_fact, num_cycles=num_cycles,
                                                  meas_nodes=meas_nodes,)
        mech.save(savefolder)
    else:
        mech = mechanical.MechanicalDummy.load(jid, savefolder)
        # mech = mechanical.Mechanical.load(jid, savefolder)

    # generate analytical FRF
    if isinstance(mech, mechanical.Mechanical):
        N = 2 ** 15
        fs = 256

        _, frf = mech.frequency_response(N, fs // 2)

        phase = (np.random.rand(N // 2 + 1) - 0.5)
        Pomega = 120 * np.ones_like(phase) * np.exp(1j * phase * 2 * np.pi)
        sig = np.empty((N, len(mech.meas_nodes)))
        for channel in range(frf.shape[1]):
            sig[:, channel] = np.fft.irfft(frf[:, channel] * Pomega)

    # pass mechanical object to acquire
    acqui = Acquire.init_from_mech(mech, channel_defs='uz')

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from mpl_toolkits.axisartist.axislines import AxesZero

    acqui.sample(*acqui.sample_helper(f_max=102.4))

    # pass to PreProcessingTools and create filter example with it
    from pyOMA.core import PreProcessingTools
    figt, axest = plt.subplots(2, 2, sharex=True, sharey=True)
    axest = axest.flat
    figf, axesf = plt.subplots(2, 2, sharex=True, sharey=True)
    axesf = axesf.flat

    label_dict = {'moving_average':'Moving Average', 'brickwall':'Windowed Sinc', 'butter':'Butterworth', 'cheby1':'Chebychev Type I'}

    for filt, order, cutoff, ax1, ax2 in [('moving_average', 12, 58.19, axest[0], axesf[0]),
                                            ('brickwall', 12, 58.19, axest[1], axesf[1]),
                                            ('butter', 6, 58.19, axest[2], axesf[2]),
                                            ('cheby1', 6, 58.19, axest[3], axesf[3])
                                            ]:

        prep_signals = PreProcessingTools.PreProcessSignals(**acqui.to_prep_data())

        # use analytical example data
        # prep_signals = PreProcessingTools.PreProcessSignals(sig, fs)

        channel = 18
        prep_signals.plot_signals(channels=channel, NFFT=prep_signals.total_time_steps // 8, axest=[ax1], axesf=[ax2], window='hamm', color='dimgrey', alpha=1)

        axins = inset_axes(ax1, width='30%', height='18.5%', loc=2, axes_class=AxesZero, borderpad=.75)
        prep_signals.filter_signals(ftype=filt, order=order, lowpass=cutoff, overwrite=True, plot_ax=[axins, ax2])
        prep_signals.plot_signals(channels=channel, NFFT=prep_signals.total_time_steps // 8, axest=[ax1], axesf=[ax2], window='hamm', color='black', alpha=1)

        ax1.annotate(label_dict[filt], (0.05, 0.05), xycoords='axes fraction', backgroundcolor='#ffffff80')
        ax2.annotate(label_dict[filt], (0.05, 0.05), xycoords='axes fraction', backgroundcolor='#ffffff80')

        ax2.axhline(0, color='lightgrey', ls='dotted')
        if filt == 'moving_average':
            axins.set_xlim((-0.05, 0.05))
            axins.set_ylim((-0.01, 0.1))
        elif filt == 'brickwall':
            axins.set_xlim((-0.05, 0.05))
            axins.set_ylim((-0.035, 0.35))
        else:
            axins.set_xlim((-0.01, 0.1))
            axins.set_ylim((-.05, .05))
        # axins.set_title('IRF', {'fontsize': 10})

        axins.set_xticks([])
        axins.set_yticks([])
        for direction in ["xzero", "yzero"]:
            axins.axis[direction].set_axisline_style("-|>")
            axins.axis[direction].set_visible(True)
            lin = axins.axis[direction].line
            lin.set_color('dimgrey')
            lin.set_facecolor('dimgrey')
            lin.set_edgecolor('black')
        for direction in ["left", "right", "bottom", "top"]:
            lin = axins.axis[direction].line
            lin.set_color('dimgrey')
            lin.set_facecolor('dimgrey')
            lin.set_edgecolor('dimgrey')

        ax1.legend().remove()
        ax2.legend().remove()
        if ax1.is_last_col():
            ax1.set_ylabel('')
            ax2.set_ylabel('')
        if not ax1.is_last_row():
            ax1.set_xlabel('')
            ax2.set_xlabel('')
    axesf[-1].set_xlim((0, 2 * 58.19))
    axesf[-1].set_ylim((-62, 3))

    axest[-1].set_xlim((0, prep_signals.duration))

    leg_handlesf = []
    leg_handlest = []
    for label, ls, c in [(None, 'dashed', 'lightgrey'), ('Filtered', 'solid', 'black'), ('Original', 'solid', 'dimgrey')]:
        if label is None:
            line = matplotlib.lines.Line2D([], [], color=c, ls=ls, label='Filter FRF')
            leg_handlesf.append(line)
            line = matplotlib.lines.Line2D([], [], color=c, ls='solid', label='Filter IRF')
            leg_handlest.append(line)
        else:
            line = matplotlib.lines.Line2D([], [], color=c, ls=ls, label=label)
            leg_handlesf.append(line)
            leg_handlest.append(line)
    figf.legend(handles=leg_handlesf, loc=(.77, .13))
    figf.subplots_adjust(.115, .115, .97, .97, .1, .1)
    figt.legend(handles=leg_handlest, loc=(.77, .12))
    figt.subplots_adjust(.115, .115, .97, .97, .1, .1)

    # figf.savefig(Path('figures') / 'filter4_example_freq.pdf')
    # figf.savefig(Path('figures') / 'filter4_example_freq.png')

    # figt.savefig(Path('figures') / 'filter4_example_tim.pdf')
    # figt.savefig(Path('figures') / 'filter4_example_tim.png')

    plt.show()

    return
    # apply sensor frf
    acqui.apply_FRF(type)

    # add sensor / cabling  noise
    acqui.add_noise(snr_db, noise_power)

    # decimate and estimate decimation noise / quantization noise
    acqui.sample(f_max, fs_factor)

    prep_signals = PreProcessingTools.PreProcessSignals(**acqui.to_preprocessdata())

    nodes, lines, chan_dofs = mech.get_geometry()

    geometry = PreProcessingTools.GeometryProcessor(nodes, lines)

    #


if __name__ == '__main__':
    # import sys
    # import os
    # from pathlib import Path
    #
    # import matplotlib
    # import matplotlib.pyplot as plt
    # import numpy as np
    # import pandas as pd
    # import scipy.signal
    #
    # from polyuq import PolyUQ
    # from polyuq.data_manager import DataManager
    #
    # from model.mechanical import Mechanical, MechanicalDummy
    # from examples.UQ_OMA import plot_response_field, vars_definition, stage2mapping
    # from helpers import get_pcd, tex_escape
    #
    # result_dir = Path(os.environ.get('UQ_OMA_RESULT_DIR', '.'))
    # working_dir = Path(tempfile.gettempdir())
    #
    # from model.mechanical import Mechanical, MechanicalDummy
    #
    # if os.path.split(result_dir)[-1] != 'samples':
    #     result_dir = result_dir / 'samples'
    #
    # jid = '8fdfc90a_ee1f66b6'
    # seed = int.from_bytes(bytes(jid, 'utf-8'), 'big')
    #
    # # load structural model
    # mech = MechanicalDummy.load(fpath=result_dir / f'mechanical.npz')
    #
    # if not isinstance(result_dir, Path):
    #     result_dir = Path(result_dir)
    #
    # if not isinstance(working_dir, Path):
    #     working_dir = Path(working_dir)
    #
    # # Set up directories
    # if '_' in jid:
    #     id_ale, id_epi = jid.split('_')
    #     this_result_dir = result_dir / id_ale
    #     this_result_dir = this_result_dir / id_epi
    #
    # # Here's the Hack
    # if os.path.exists(result_dir / id_ale / 'response.npz'):
    #     arr = np.load(result_dir / id_ale / 'response.npz')
    #     t_vals = arr['t_vals']
    #     d_freq_time = arr['d_freq_time']
    #     v_freq_time = arr['v_freq_time']
    #     a_freq_time = arr['a_freq_time']
    # else:
    #     raise RuntimeError(f'{result_dir / id_ale / "response.npz"} does not exist')
    #
    # # Here's the Hack
    # mech.t_vals_amb = t_vals
    # mech.resp_hist_amb = [d_freq_time, v_freq_time, a_freq_time]
    # mech.deltat = t_vals[1] - t_vals[0]
    # mech.timesteps = t_vals.shape[0]
    # mech.state[2] = True
    #
    # n_locations = 9
    #
    # # num_nodes = mech.num_nodes
    # num_nodes = 203
    #
    # setups = sensor_position(n_locations, num_nodes, 'distributed')
    # # select a setup based on a "random" integer modulo the total number of setups
    # i_setup = seed % setups.shape[0]
    # sensor_nodes = setups[i_setup,:]
    # quant = 'a'
    # quant = ['d', 'v', 'a'].index(quant)
    # # list of (node, dof, quant)
    # channel_defs = []
    # for node in sensor_nodes:
    #     # We work around the Hack from transient_ifrf, where we omitted the x-axis in the response,
    #     # by wrongly definining dofs ux and uy instead of uy and uz
    #
    #     for dof in ['ux', 'uy']:
    #         dof = ['ux', 'uy', 'uz'].index(dof)
    #         channel_defs.append((node, dof, quant))
    #
    # acqui = Acquire.init_from_mech(mech, channel_defs)
    #
    # acqui.estimate_snr()

    # N=8192
    # dec_fact=4
    # numtap_fact=41
    # nyq_rat=2.5
    # system_signal_decimate(N,dec_fact, numtap_fact,nyq_rat, True)
    # plt.show()
    uq_dec_noise()
    # verify_noise_power()
    # with matplotlib.rc_context(rc=print_context_dict):
    #    filter_example()()

