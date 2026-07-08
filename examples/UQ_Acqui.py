import sys

from polyuq.data_manager import DataManager, HiddenPrints
from model.mechanical import Mechanical, MechanicalDummy
from model.acquisition import Acquire

from pyOMA.core.PreProcessingTools import PreProcessSignals
from pyOMA.core.SSICovRef import BRSSICovRef

import os
# import uuid
import numpy as np
# from scipy import *
import matplotlib.pyplot as plt
# import pyansys

# import ray


def model_acqui_signal_sysid(jid, result_dir, working_dir, ansys=None,
                             num_nodes=101, num_modes=10, damping=0.01, freq_scale=1.0, num_meas_nodes=20,  # structural properties
                             f_scale=1000, dt_fact=0.01, num_cycles=50,  # time integration properties
                             num_sensors=10, lumped=False, quant='a',  # sensor placement / properties
                             snr_db=-30.0, noise_power=0.0, ref_sig='channel',  # acquisition noise properties
                             nyq_rat=2.5, numtaps_fact=21,  # sampling_properties
                             sample_dur=None, margin=None, bits=16, duration=None,  # quantization properties
                             decimate_factor=None, m_lags=100, window=6, method='welch',  # signal processing parameters
                             order_factor=2,  # system identification parameters
                             ):

    skip_existing = True
    savefolder = os.path.join(result_dir, jid)
    jid_int = int.from_bytes(bytes(jid, 'utf-8'), 'big')

    # load/generate  mechanical object:
    # ambient response of a n dof rod,
    mech = None
    if os.path.exists(os.path.join(savefolder, f'{jid}_mechanical.npz')) and skip_existing:
        import zlib, zipfile
        try:
            mech = Mechanical.load(jid, savefolder)
        except (EOFError, KeyError, zlib.error, zipfile.BadZipFile):
            print(f'File {os.path.join(savefolder, f"{jid}_mechanical.npz")} corrupted. Deleting.')
            raise
            os.remove(os.path.join(savefolder, f'{jid}_mechanical.npz'))

    if mech is None:
        raise RuntimeError()
        if title == 'uq_acqui':
            mech = Mechanical(ansys=ansys, jobname=jid, wdir=working_dir)
            mech.example_rod(num_nodes=num_nodes, damping=damping, num_modes=num_modes, freq_scale=freq_scale, num_meas_nodes=num_meas_nodes)
            mech.ambient_ifrf(f_scale, None, dt_fact, None, num_cycles, [quant], ['uz'], seed=jid_int)
            mech.save(savefolder)
        else:
            # ansys = Mechanical.start_ansys(working_dir, jid)
            mech = Mechanical(ansys=ansys, jobname=jid, wdir=working_dir)
            mech.example_rod(num_nodes=num_nodes, damping=damping, num_modes=num_modes, num_meas_nodes=num_meas_nodes)
            mech.ambient(f_scale, None, dt_fact, None, num_cycles, out_quant=[quant], seed=jid_int)
            mech.numerical_response_parameters()
            # mech = generate_mdof_time_hist(ansys=ansys, num_nodes=num_nodes, damping=damping,
                                           # num_modes=num_modes, f_scale=f_scale,
                                           # dt_fact=dt_fact, num_cycles=num_cycles)
            mech.save(savefolder)

        # mech = mechanical.Mechanical.load(jid, savefolder)
    # sig = mech.resp_hist_amb[2]
    # plt.figure()
    # for channel in range(sig.shape[1]):
        # plt.plot(sig[:,channel,2], label=mech.meas_nodes[channel])
    # plt.legend()
    # plt.show()
    if isinstance(mech, MechanicalDummy):
        frequencies_n, damping_n, mode_shapes_n = mech.frequencies_comp, mech.modal_damping_comp, mech.mode_shapes_comp
    else:
        frequencies_n, damping_n, mode_shapes_n = mech.numerical_response_parameters()

    # pass mechanical object to acquire
    if lumped:
        num_meas_nodes = len(mech.meas_nodes)
        free_nodes = num_meas_nodes - num_sensors
        assert free_nodes > 0
        rng = np.random.default_rng(3 * jid_int)
        start_node = rng.integers(0, free_nodes)
        meas_nodes = mech.meas_nodes[start_node:start_node + num_sensors]
    else:
        # evenly distributed sensors
        num_meas_nodes = len(mech.meas_nodes)
        ind = np.rint(np.linspace(0, num_meas_nodes - 1, num_sensors)).astype(int)
        meas_nodes = mech.meas_nodes[ind]

    assert len(meas_nodes) == num_sensors
    channel_defs = [(meas_node, 'uz', quant) for meas_node in meas_nodes]

    acqui = Acquire.init_from_mech(mech, channel_defs=channel_defs)
    # sig = acqui.signal
    # Fs = 1 / acqui.deltat
    # num_channels = acqui.num_channels
    # fig, axes = plt.subplots(num_channels, 2, sharex='col', sharey='col', tight_layout=True)
    # axes = axes.T
    # for channel in range(num_channels):
    #     axes[0, channel].plot(acqui.t_vals, sig[channel, :], alpha=.5)
    #     axes[1, channel].psd(sig[channel, :], NFFT=8192, Fs=Fs, alpha=.5)

    power_signal, power_noise = acqui.add_noise(snr_db=snr_db, noise_power=noise_power, ref_sig=ref_sig, seed=2 * jid_int)
    '''
    for sampling it is of primary interest, how much noise results from sample-and-hold process
    that means, how much noise results from inproper anti-aliasing -> snr_alias
    in physical sampling this is influenced by the order of the anti-aliasing filter 
    (i.e. the signal power above the cutoff) and the cutoff-to-nyquist-rate
        -> these are our primary derived input quantities (numtaps, fs / cutoff)
    in this simulation low numbers of timesteps and high decimation rates mostly affect this
        -> these are secondary derived input quantities for ensuring a negligible effect (sim_timesteps, dec_rate)
    
    quantization is affected by the number of bits and the chosen measurement range
     in relation to the actual range of signal values -> the (normalized) margin
     no other simulative processes seem to affect it
        -> bits, margin_quant_norm
    
    '''
    fs, numtaps, cutoff = acqui.sample_helper(f_max=frequencies_n[-1], nyq_rat=nyq_rat, numtaps_fact=numtaps_fact)
    dec_rate, sim_timesteps = int(1 / (acqui.deltat * fs)), acqui.num_timesteps
    meas_range = acqui.estimate_meas_range(sample_dur=sample_dur, margin=margin)
    _, _, snr_alias, snr_quant, margin_quant_norm = acqui.sample(fs, numtaps, cutoff, bits, meas_range, duration)
    snr_db_out = acqui.snr_db
    frequencies_a, damping_a, mode_shapes_a = acqui.modal_frequencies, acqui.modal_damping, acqui.mode_shapes

    # sig = acqui.signal
    # for channel in range(num_channels):
    #     axes[0, channel].plot(acqui.t_vals, sig[channel, :], alpha=.5)
    #     node, dof, quantity = acqui.channel_defs[channel]
    #     label = f'{["dsp", "vel", "acc", ][quantity]} {node} {["x", "y", "z"][dof]}'
    #     axes[1, channel].psd(sig[channel, :], NFFT=8192, Fs=Fs, label=label, alpha=.5)
    #     axes[1, channel].legend()
    #     axes[1, channel].set_xlim((0, 150))
    #     for freq in acqui.modal_frequencies:
    #         axes[1, channel].axvline(freq, zorder=-10)
    #
    # plt.show()
    # pass to PreProcessingTools
    prep_signals = PreProcessSignals(**acqui.to_prep_data())
    # prep_signals.plot_data(svd_spect=True)
    if decimate_factor is not None:  # implying signal shall be decimated
        prep_signals.decimate_signals(decimate_factor, highpass=None, order=int(21 * decimate_factor), filter_type='brickwall')
    with HiddenPrints():
        prep_signals.correlation(m_lags, method, window=window)
    # prep_signals.plot_data(svd_spect=True)
    # plt.show()
    # pass to Modal System Identification
    model_order = num_modes * order_factor
    modal_data = BRSSICovRef(prep_signals)
    modal_data.build_toeplitz_cov(num_block_columns=m_lags // 2)
    modal_data.compute_state_matrices(max_model_order=model_order)
    modal_frequencies, modal_damping, mode_shapes, _, modal_contributions = modal_data.single_order_modal(order=model_order)

    sort_ind = np.argsort(modal_frequencies[modal_frequencies != 0])
    modal_damping = modal_damping[sort_ind]
    modal_frequencies = modal_frequencies[sort_ind]
    mode_shapes = mode_shapes[:, sort_ind]
    modal_contributions = modal_contributions[sort_ind]

    if False:
        modal_data.compute_modal_params()

        from pyOMA.core import StabilDiagram
        from pyOMA.GUI import StabilGUI
        stabil_calc = StabilDiagram.StabilCalc(modal_data)
        stabil_plot = StabilDiagram.StabilPlot(stabil_calc)
        StabilGUI.start_stabil_gui(stabil_plot, modal_data, None, prep_signals)

    # all_nod_x = mech.meas_nodes
    # nod_x_a = meas_nodes
    # fig, axes = plt.subplots(3, 3, sharex=True, sharey=True)
    # axes = axes.flat
    # for mode in range(9):
        # this_mode_shape = mode_shapes_n[:, mode] / mode_shapes_n[-1, mode]
        # axes[mode].plot(all_nod_x, np.abs(this_mode_shape),
                        # #np.sign(np.angle(this_mode_shape)) * np.abs(this_mode_shape),
                        # label=f'mech: {frequencies_n[mode]:1.2f}, {damping_n[mode]:1.2f}', ls='none', marker='x')
        # this_mode_shape = mode_shapes_a[:, mode] / mode_shapes_a[-1, mode]
        # axes[mode].plot(nod_x_a, np.abs(this_mode_shape),
                        # #np.sign(np.angle(this_mode_shape)) * np.abs(this_mode_shape),
                        # label=f'acqui: {frequencies_a[mode]:1.2f}, {damping_a[mode]:1.2f}', ls='none', marker='+')
        # id_mode = np.argmin(np.abs(modal_frequencies - frequencies_a[mode]))
        # this_mode_shape = mode_shapes[:, id_mode] / mode_shapes[-1, id_mode]
        # axes[mode].plot(nod_x_a, np.abs(this_mode_shape),
                        # #np.sign(np.angle(this_mode_shape)) * np.abs(this_mode_shape),
                        # label=f'ident: {modal_frequencies[id_mode]:1.2f}, {modal_damping[id_mode]:1.2f}', ls='none', marker='o', fillstyle='none')
        # axes[mode].legend(title=f'mode: {mode}')
    # plt.show()

    print(frequencies_a, modal_frequencies)
    print(damping_a, modal_damping)

    return (frequencies_a, damping_a, mode_shapes_a,  # reference output parameters
            np.array(numtaps), np.array(fs), np.array(cutoff), margin_quant_norm,  # derived input parameters
            np.array(sim_timesteps), np.array(dec_rate),  # control input parameters
            snr_alias, snr_quant, snr_db_out,  # intermediate output parameters
            np.array(model_order),  # derived input parameter
            modal_frequencies, modal_damping, mode_shapes,  # output parameters
            modal_contributions  # intermediate output parameters
            )


def uq_acqui(step, working_dir, result_dir):

    if step == 0 or not os.path.exists(os.path.join(result_dir, f'{title}.nc')):
        data_manager = DataManager(title=title, working_dir=working_dir, result_dir=result_dir)

        # num_nodes=101,
        # num_meas_nodes = 100,
        # f_scale=1000,
        # dt_fact=0.01,
        quant = 'a',
        # noise_power=0.0,
        # ref_sig='channel',
        # sample_dur=None,
        # margin=None,
        # duration=None,
        # decimate_factor=None,
        # nyq_rat
        # data_manager.generate_sample_inputs(names=['num_modes',
        #                                            'damping',
        #                                            'freq_scale',
        #                                            'num_cycles',
        #                                            'num_sensors',
        #                                            'lumped',
        #                                            'snr_db',
        #                                            'bits',
        #                                            'tau_max'],
        #                             distributions=[('integers', (2, 21)),
        #                                            ('uniform', (0.01, 0.1)),
        #                                            ('uniform', (0.5, 2)),
        #                                            ('integers', (500, 1501)),
        #                                            ('integers', (2, 11)),
        #                                            ('integers', (0, 2)),
        #                                            ('uniform', (-30, 10)),
        #                                            ('choice', ([8, 16, 32],)),
        #                                            ('integers', (50, 501)),
        #                                            ],
        #                             num_samples=1000)
        # data_manager.generate_sample_inputs(names=['window',
                                                   # 'order_factor'],
                                    # distributions=[('choice',(['None','hann'],)),
                                                   # ('integers',(2,11)),
                                                    # ],
                                    # num_samples=1000)
        # data_manager.generate_sample_inputs(names=['bits2',
        #                                             'nyq_rat',
        #                                            'numtaps_fact'],
        #                             distributions=[('integers',(8,33)),
        #                                             ('uniform',(2, 4)),
        #                                            ('integers',(11,42)),
        #                                             ],
        #                             num_samples=1000)
        data_manager.generate_sample_inputs(names=['window',
                                                   'method'],
                                    distributions=[('uniform', (0, 16)),
                                                   ('choice', (['welch', 'blackman-tukey'],)),
                                                    ],
                                    num_samples=1000)
        return
    if step == 1:
        data_manager = DataManager.from_existing(dbfile_in=f'{title}.nc', result_dir=result_dir, working_dir=working_dir)
        # data_manager.clear_failed(True)
        data_manager.post_process_samples()
        plt.show()
        return
    if step == 2:
        # ansys = Mechanical.start_ansys()

        data_manager = DataManager.from_existing(dbfile_in=f'{title}.nc', result_dir=result_dir, working_dir=working_dir)
        func_kwargs = {'ansys':None, 'num_nodes':101, 'num_meas_nodes': 100,
                       'f_scale':1000, 'dt_fact':0.01, 'quant':'a', 'noise_power':0.0,
                       'ref_sig':'channel', 'sample_dur':None, 'margin':None, 'duration':None,
                       'decimate_factor':None}
        arg_vars = {'num_modes':'num_modes',
                    'damping':'damping',
                    'freq_scale':'freq_scale',
                    'num_cycles':'num_cycles',
                    'num_sensors':'num_sensors',
                    'lumped':'lumped',
                    'snr_db':'snr_db',
                    'bits':'bits2',
                    'm_lags':'tau_max',
                    'window':'window',
                    'method':'method',
                    'order_factor':'order_factor',
                    'nyq_rat':'nyq_rat',
                    'numtaps_fact':'numtaps_fact'}

        ret_names = {'frequencies_a':('modes',),
                     'damping_a':('modes',),
                     'mode_shapes_a':('channels', 'modes',),  # reference output parameters
                     'numtaps':(),
                     'fs':(),
                     'cutoff':(),
                     'margin':('channels',),  # derived input parameters
                     'sim_steps':(),
                     'dec_rate':(),  # control input parameters
                     'snr_alias':('channels',),
                     'snr_quant':('channels',),
                     'snr_db_out':('channels',),  # intermediate output parameters
                     'model_order':(),  # derived input parameter
                     'modal_frequencies':('modes',),
                     'modal_damping':('modes',),
                     'mode_shapes':('channels', 'modes',),  # output parameters
                     'modal_contributions':('modes',)  # intermediate output parameters
                     }

        # ret_names = ['frequencies_a', 'damping_a', 'mode_shapes_a',  # reference output parameters
        #              'numtaps', 'cutoff', 'meas_range',  # derived input parameters
        #              'snr_alias', 'snr_quant', 'snr_db_out',  # intermediate output parameters
        #              'model_order',  # derived input parameter
        #              'modal_frequencies', 'modal_damping', 'mode_shapes',  # output parameters
        #              'modal_contributions'  # intermediate output parameters
        #              ]
        data_manager.evaluate_samples(func=model_acqui_signal_sysid, arg_vars=arg_vars, ret_names=ret_names, **func_kwargs,
                                      chwdir=True, dry_run=True, default_len={'modes':100, 'channels':10}  # modes = max(num_modes) * max(order_factor) / 2

                                      )
        # ['d6075c96cfb7', 'b5c3a0c90604', 'e2cc12128797', '6251717cb53e','d60c3915887e', '79dc28c2f054', '9e6fd66b51c5']
        os.system(f'bkill 0 -q BatchXL -u {os.environ.get("USER", "")}')
        return
    if step == 3:
        import matplotlib
        from helpers import get_pcd
        pcd = get_pcd()
        data_manager = DataManager.from_existing(dbfile_in=f'{title}.nc', result_dir=result_dir, working_dir=working_dir)
        # data_manager.clear_failed(True)

        with matplotlib.rc_context(rc=pcd):
            # influences of signal acquistion (quantization and sampling)
            if True:
                # with data_manager.get_database('processed', rw=True) as ds:
                    # ds['bits_eff']=(np.log(ds.margin*(2**ds.bits))/np.log(2)).mean('channels')
                names = ['snr_db', 'bits', 'bits_eff', 'this_snr_quant', 'this_snr_db_out']  # effective bits = log(margin / 2**bits)/log(2)
                labels = ['SNR\\textsubscript{dB}', '$b$', '$b_\\text{eff}$', 'SNR\\textsubscript{dB, quant}', 'SNR\\textsubscript{dB, tot}']
                # scales = ['linear','log','log','linear','linear']
                fig = data_manager.post_process_samples(names=names, db='processed', labels=labels, figsize=(5.92, 5.92))  # , scales=scales)
                # fig.subplots_adjust(bottom=0.07, left=0.1)
                fig.savefig(f'figures/{title}_quant.pdf', dpi=300)

            if True:
                names = ['snr_db', 'numtaps_fact', 'nyq_rat', 'sim_steps', 'dec_rate', 'this_snr_alias', 'this_snr_db_out']  # numtaps_fact = numtaps/dec_fact, nyq_rat=fs/cutoff, <- constant
                labels = ['SNR\\textsubscript{dB}', '$\\sfrac{M}{d}$', '$\\sfrac{f_c}{f_s}$', '$N$', '$d$', 'SNR\\textsubscript{dB, alias}', 'SNR\\textsubscript{dB, tot}']
                fig = data_manager.post_process_samples(names=names, db='processed', labels=labels, figsize=(5.92, 5.92))
                fig.subplots_adjust(bottom=0.07, left=0.07)
                fig.savefig(f'figures/{title}_samp.pdf', dpi=300)

            if True:
                names = ['this_snr_db_out', 'freq_diff', 'damp_diff', 'modal_assurance', 'unp_id', 'unp_num']
                labels = ['SNR\\textsubscript{dB, tot}', '$\\Delta_f$', '$\\Delta_{\\zeta}$', 'MAC', '$m_\\text{additional}$', '$m_\\text{missing}$']
                fig = data_manager.post_process_samples(names=names, db='processed', labels=labels, figsize=(5.92, 5.92))
                fig.subplots_adjust(left=0.07)
                fig.savefig(f'figures/{title}_noise.pdf', dpi=300)

            # influences of sensors
            if True:
                names = ['num_sensors', 'lumped', 'freq_diff', 'damp_diff', 'modal_assurance', 'unp_id', 'unp_num']  # quant
                labels = [ '$n_\\text{sensors}$', 'placement', '$\\Delta_f$', '$\\Delta_{\\zeta}$', 'MAC', '$m_\\text{additional}$', '$m_\\text{missing}$']
                fig = data_manager.post_process_samples(names=names, db='processed', labels=labels, figsize=(5.92, 5.92))

                axes = fig.axes
                for ax in axes:
                    sps = ax.get_subplotspec()
                    col, row = sps.colspan.start, sps.rowspan.start
                    if col == 1 and row == 6:
                        axis = ax.xaxis
                    elif row == 1 and col == 0:
                        axis = ax.yaxis
                    else:
                        continue
                    axis.set_ticks([0, 1])
                    axis.set_ticklabels(['lump', 'dist'])
                fig.subplots_adjust(left=0.08)
                fig.savefig(f'figures/{title}_sensors.pdf', dpi=300)

            # influences of signal processing and system identification
            if True:
                names = ['all_n_cycl', 'method', 'tau_max', 'order_factor', 'freq_diff', 'damp_diff', 'modal_assurance', 'unp_id', 'unp_num']
                labels = [ '$n_\\text{cycles}$', 'spectral \\\\ estimator', '$\\tau_\\text{max}$', 'model order \\\\ overrate', '$\\Delta_f$', '$\\Delta_{\\zeta}$', 'MAC', '$m_\\text{additional}$', '$m_\\text{missing}$']
                fig = data_manager.post_process_samples(names=names, db='processed', labels=labels, figsize=(5.92, 5.92))
                axes = fig.axes
                for ax in axes:
                    sps = ax.get_subplotspec()
                    col, row = sps.colspan.start, sps.rowspan.start
                    if col == 1 and row == 8:
                        axis = ax.xaxis
                    elif row == 1 and col == 0:
                        axis = ax.yaxis
                    else:
                        continue
                    axis.set_ticks([0, 1])
                    axis.set_ticklabels(['b.-t.', 'welch'])
                fig.subplots_adjust(left=0.08)
                fig.savefig(f'figures/{title}_sigid.pdf', dpi=300)

            # influences of structural parameters
            if True:
                names = ['damping', 'freq_scale', 'num_modes', 'freq_diff', 'damp_diff', 'modal_assurance', 'unp_id', 'unp_num']
                labels = [ '$\\zeta$', '$\\delta_f$', '$n_\\text{modes}$', '$\\Delta_f$', '$\\Delta_{\\zeta}$', 'MAC', '$m_\\text{additional}$', '$m_\\text{missing}$']
                fig = data_manager.post_process_samples(names=names, db='processed', labels=labels, figsize=(5.92, 5.92))
                fig.subplots_adjust(left=0.07)
                fig.savefig(f'figures/{title}_struct.pdf', dpi=300)

            if False:
                names = None
                data_manager.post_process_samples(names=names, db='processed', draft=False)

            plt.show()
            # data_manager.post_process_samples()

        return
    if step == 4:
        data_manager = DataManager.from_existing(dbfile_in=f'{title}.nc', result_dir=result_dir, working_dir=working_dir)
        data_manager.clear_failed(dryrun=True)


def main():

    global title
    title = 'uq_acqui2'

    import tempfile
    working_dir = os.path.join(tempfile.gettempdir(), str(os.getuid()), 'work')

    base = os.environ.get('UQ_ACQUI_RESULT_DIR', os.path.join(os.getcwd(), 'polyuq_results'))
    result_dir = os.path.join(base, f'{title}/')

    if False:
        jid = 'test123'

        import logging
        # logging.getLogger('uncertainty.data_manager').setLevel(logging.DEBUG)
        logging.getLogger('model.mechanical').setLevel(logging.DEBUG)
        fun_out = model_acqui_signal_sysid(jid, result_dir, working_dir, num_cycles=5, snr_db=30)
        for a in fun_out:
            print(a)
    else:
        # pass
        # import logging
        # logging.getLogger('uncertainty.data_manager').setLevel(logging.DEBUG)
        uq_acqui(3, working_dir, result_dir)


def clear_wdirs():
    wdir = os.path.join('/dev/shm', str(os.getuid()), 'work')

    import ray
    if not ray.is_initialized():
        ray.init(address='auto', _redis_password='5241590000000000')

    @ray.remote
    def clear(wdir):
        import shutil
        import time
        time.sleep(np.random.randint(0, 6))
        if os.path.exists(wdir):
            shutil.rmtree(wdir)
            print(f"Cleared {wdir} on {os.uname()[1]}. still exists? {os.path.exists(wdir)}")
        else:
            print(f"{wdir} on {os.uname()[1]} does not exist")

        time.sleep(30)

    futures = []
    for i in range(60):
        worker_ref = clear.remote(wdir)
        futures.append(worker_ref)
    ray.wait(futures)


if __name__ == '__main__':
    # clear_wdirs()
    # main()
    pass
