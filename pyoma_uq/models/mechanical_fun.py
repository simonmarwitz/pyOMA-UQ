# from importlib import reload; 
from pyoma_uq.models.mechanical import Mechanical

import time
import numpy as np
import matplotlib
#matplotlib.use('qt5Agg')
import matplotlib.pyplot as plot
# plot.ioff()
import os
import sys
import glob
import shutil
import pathlib
import tempfile

import logging
# global logger
# logger = logger.getLogger('')
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

import pyansys

# import scipy.stats
# import scipy.optimize
# import scipy.signal
import scipy.io
# import scipy.integrate
import uuid
import warnings
# import simpleflock

# plot.figure()
# plot.close("all")

# plot.show()
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
    # plot.rc('text', usetex=True)
    # plot.rc('text.latex', preamble="\\usepackage{siunitx}\n \\usepackage{xfrac}")

# import math
# class LogFormatterSciNotation(matplotlib.ticker.LogFormatterMathtext):
#     """
#     Format values following scientific notation in a logarithmic axis.
#     """
#
#     def _non_decade_format(self, sign_string, base, fx, usetex):
#         'Return string for non-decade locations'
#         b = float(base)
#         exponent = math.floor(fx)
#         coeff = b ** fx / b ** exponent
#
#         coeff = round(coeff)
#         return f'$\\num{{{coeff}E{exponent}}}$'
#         return (r'$%s%g E {%d}$') % \
#                                     (sign_string, coeff, exponent)


def response_frequency(time_values, ydata, p0=[1, 0.05, 2 * np.pi, 0]):
#     res=pyansys.read_binary(path)
#     node_numbers=res.geometry['nnum']
#     num_nodes=len(node_numbers)
#     time_values=res.time_values
#     dt = time_values[1]-time_values[0]
#     time_values -= dt
#
#
#     solution_data_info = res.solution_data_info(0)
#     DOFS = solution_data_info['DOFS']
#
#     uz = DOFS.index(3)
#     nnum, all_disp = res.nodal_time_history('NSL')
#     #print(nnum)
#
#     ydata = all_disp[:,np.where(nnum==20)[0][0],uz]
    this_t = time_values

    popt, pcov = scipy.optimize.curve_fit(f=free_decay, xdata=this_t, ydata=ydata, p0=p0)  # ,  bounds=[(-1,0,0,0),(1,1,np.pi/dt,2*np.pi)])
    perr = np.sqrt(np.diag(pcov))
    # print('R: {:1.3f} m, zeta: {:1.3f} \%, f_d: {:1.4f} Hz, phi: {:1.4f}'.format(popt[0], popt[1]*100,  popt[2]/2/np.pi, popt[3]*180/np.pi))
    return popt, perr

# def process(path, last_analysis = 'trans', f = None):
#
#
#     '''
#     nsol,2,2,U,Z,uz
#     store,merge
#     plvar,2
#
#     nsol,3,2,ACC,Z,acz
#     store,merge
#     plvar,3
#     '''
#
#     res=pyansys.read_binary(path)
#
#     if last_analysis=='static': #static
#         nodes, disp = res.nodal_solution(0)
#         uz=disp[nodes==20, 2]#knoten 20, DOF 2 (UZ)
#         print(uz)
#         return uz
#
#     elif last_analysis == 'modal': #modal
#         num_modes = res.nsets#_resultheader['nsets']
#         if res._resultheader['cpxrst']: # real and imaginary parts are saved as separate sets
#             num_modes //= 2
#         nnodes = res._resultheader['nnod']
#         ndof = res._resultheader['numdof']
#
#         mode_shapes = np.full((nnodes,ndof,num_modes), (1+1j)*np.nan, dtype=complex)
#         frequencies = np.full(num_modes, np.nan)
#         damping = np.full(num_modes, np.nan)
#
#         if res._resultheader['cpxrst']:
#             for mode in range(num_modes):
#                 sigma = res.time_values[2*mode]
#                 omega = res.time_values[2*mode+1]
#                 if omega < 0 : continue # complex conjugate pair
#
#                 frequencies[mode] = omega
#                 damping[mode] = -sigma/np.sqrt(sigma**2+omega**2)
#
#                 mode_shapes[:,:,mode].real= res.nodal_solution(2*mode)[1]
#                 mode_shapes[:,:,mode].imag= res.nodal_solution(2*mode+1)[1]
#
#         else:
#             frequencies[:] = res.time_values
#             for mode in range(num_modes):
#                 nnum, modal_disp = res.nodal_solution(mode)
#                 mode_shapes[:,:,mode]= modal_disp
#
#         return frequencies, damping, mode_shapes
#
#     elif last_analysis == 'trans': #transient
#
#         #meas_nodes=res.geometry['components']['MEAS_NODES']
#         node_numbers=res.geometry['nnum']
#         num_nodes=len(node_numbers)
#         time_values=res.time_values
#         #print(time_values)
#         dt = time_values[1]-time_values[0]
#         time_values -= dt
#         #print(dt)
#         #print(res._resultheader['neqv'])
#
#         solution_data_info = res.solution_data_info(0)
#         DOFS = solution_data_info['DOFS']
#
#         uz = DOFS.index(3)
#         nnum, all_disp = res.nodal_time_history('NSL')
#
#         #print(nnum, all_disp.shape)
#         if f is not None:
#             #fix,axes = plot.subplots(nrows=2, ncols=2, sharex='col', sharey='row')
#             fig = plot.figure()
#             ax1 = fig.add_subplot(221)
#             ax2 = fig.add_subplot(222, sharey=ax1)
#             ax3 = fig.add_subplot(223, sharex=ax1)
#             ax4 = fig.add_subplot(224)
#             t = time_values#np.linspace(0,f.shape[0]*dt,f.shape[0])
#             ax2.plot(t,f, marker='+')
#         for node in range(1,num_nodes):
#             if f is not None:
#
#                 ydata = all_disp[:,node,uz]
#                 #ydata = np.concatenate(([1],ydata))
#                 this_t = time_values
#                 #this_t = np.concatenate(([0],time_values))
#
#                 ax1.plot(all_disp[:,node,uz],f, label=str(nnum[node]), marker='+')
#
#                 ax3.plot(ydata,this_t, marker='+')
#                 popt, pcov = scipy.optimize.curve_fit(f=free_decay, xdata = this_t, ydata=ydata, p0=[0.5,0.05,2*np.pi,0])#,  bounds=[(-1,0,0,0),(1,1,np.pi/dt,2*np.pi)])
#                 perr = np.sqrt(np.diag(pcov))
#                 print('R: {:1.3f} m, zeta: {:1.3f} \%, f_d: {:1.4f} Hz, phi: {:1.4f}'.format(popt[0], popt[1]*100,  popt[2]/2/np.pi, popt[3]*180/np.pi))
#                 print(perr)
#                 print(popt)
#                 this_t = np.linspace(0,time_values[-1],len(time_values)*10)
#                 ax3.plot(free_decay(this_t,*popt),this_t)
#
#                 Sxx = np.abs(np.fft.rfft(all_disp[:,node,uz]))
#                 freq = np.fft.rfftfreq(all_disp.shape[0], dt)
#                 #freq, Sxx = scipy.signal.welch(all_disp[:,node,uz], fs=1/dt, nperseg = f.shape[0]//2)
#                 ax4.plot(freq, Sxx, marker='+')
#                 ax4.axvline(20.0)
#                 print(freq[Sxx.argmax()])
#             else:
#                 plot.plot(all_disp[:,node,uz], label=str(nnum[node]))
#
#         ax = ax1
#         #ax.grid(True)
#         ax.legend()
#         ax.spines['left'].set_position('zero')
#         ax.spines['right'].set_color('none')
#         ax.spines['bottom'].set_position('zero')
#         ax.spines['top'].set_color('none')
#         ax.set_xlabel('y [m]')
#         ax.xaxis.set_label_coords(1.05, 0.55)
#         ax.set_ylabel('F [N]', rotation='horizontal')
#         ax.yaxis.set_label_coords(0.45, 1.05)
#
#         xmin,xmax = ax.get_xlim()
#         xlim = max(-1*xmin,xmax)
#         ax.set_xlim((-1*xlim,xlim))
#
#         ymin,ymax = ax.get_ylim()
#         ylim = max(-1*ymin,ymax)
#         ax.set_ylim((-1*ylim,ylim))
#         plot.show()
#     return
#
#     path='/dev/shm/test.csv'
#     t,d=[],[]
#     f=open(path,'rt')
#     f.readline()
#     for line in f:
#         l=line.split()
#         t.append(float(l[0]))
#         d.append(float(l[1]))
#
#     d=np.array(d)
#     t=np.array(t)
#     blocks=1
#     block_length = int(np.floor(len(t)/blocks))
#     H=np.zeros(block_length,dtype=complex)
#     for i in range(blocks):
#         Sx=np.fft.fft(d[i*block_length:(i+1)*block_length])
#         F=np.sin(t[i*block_length:(i+1)*block_length]*2*np.pi*t[i*block_length:(i+1)*block_length]/1000)
#         Sf=np.fft.fft(F)
#         Sff=Sf**2
#         Sfx=Sf*Sx
#         H+=Sfx/Sff
#     H/=blocks
#     fftfreq =np.fft.fftfreq(block_length, t[1]-t[0])
#     fig,axes=plot.subplots(2,1,sharex='col')
#     axes[0].plot(fftfreq, np.abs(H))
#     axes[1].plot(fftfreq, np.angle(H)/np.pi*180)
#     plot.xlim(xmin=0)
#     plot.show()
#
#     plot.plot(t,d)
#     plot.plot(t,np.sin(t*2*np.pi*t/1000))
#
#     plot.show()


def free_decay(t, R, zeta, omega_d, phi=0):
    return R * np.exp(-zeta * omega_d / (np.sqrt(1 - zeta ** 2)) * t) * np.cos(omega_d * t + phi)


def free_decay_acc(t, R, zeta, omega_d, phi=0):

    # return -2*R*omega_d**2*np.exp(-zeta*omega_d/(np.sqrt(1-zeta**2))*t)*np.cos(omega_d*t+phi)
    return -R * omega_d ** 2 * np.exp(-zeta * omega_d / (np.sqrt(1 - zeta ** 2)) * t) * np.cos(omega_d * t + phi)

# def generate_student(ansys, omega, zeta, d0, deltat=None,dt_fact=None,timesteps=None, num_cycles=None, f_scale=None, **kwargs):
#     print(jid)
#     assert deltat is not None or dt_fact is not None
#     assert timesteps is not None or num_cycles is not None
#
#     m    = 1    # kg
#     k_2  = omega**2*m/(1-zeta**2)# N/m
#     d=zeta*(2*np.sqrt(k_2*m))
#
#     f_max = np.sqrt((k_2)/m)/np.pi/2
#
#     if dt_fact is None:
#         dt_fact =deltat*f_max # \varOmega /2/pi
#         print("\\varOmega",dt_fact*2*np.pi)
#     elif deltat is None:
#         deltat = dt_fact/f_max
#
#     if dt_fact*2*np.pi > 0.1: print("Warning \\varOmega > 0.1")
#
#     if timesteps is None:
#         timesteps = int(np.ceil(num_cycles/f_max/deltat))
#     elif num_cycles is None:
#         num_cycles = int(np.floor(timesteps*f_max*deltat))
#     if d0 is not None:
#         print("Simulating system with omega {:1.3f}, zeta {:1.3f}, d0 {:1.3f}. deltat {:1.5f}, dt_fact {:1.2f}, timesteps {}, num_cycles {}".format(omega, zeta, d0, deltat, dt_fact, timesteps, num_cycles))
#     elif f_scale is not None:
#         print("Simulating system with omega {:1.3f}, zeta {:1.3f}, random excitation variance {:1.3f}. deltat {:1.5f}, dt_fact {:1.2f}, timesteps {}, num_cycles {}".format(omega, zeta, f_scale, deltat, dt_fact, timesteps, num_cycles))
#     mech = Mechanical(ansys)
#     voigt_kelvin=mech.voigt_kelvin(k=k_2, d=d)
#     mass = mech.mass(m)
#     mech.build_sdof(
#                mass=mass,
#                voigt_kelvin = voigt_kelvin,
#                d_init=d0
#                )
#
#     mech.modal(True)
#
#     f_max = process(f'{jid}.rst', last_analysis=mech.last_analysis)[0].max()
#     f_analytical = np.sqrt((k_2)/m)/np.pi/2
#     zeta_analytical = d/2/np.sqrt(k_2*m)*100 # in percent of critical damping
#     #print(f_analytical,zeta_analytical)
#
#     mech.timesteps=timesteps
#     mech.deltat = deltat
#
#     #print(timesteps,deltat)
#
#     f=np.zeros(timesteps)
#     if d0 is not None:
#         f[0]=d0
#         mech.transient(d=f,parameter_set= kwargs.pop('parameter_set',None))
#     else:
#         assert f_scale is not None
#         f = np.random.normal(loc=0.0,scale=f_scale, size=(timesteps,))
#         mech.transient(f=f,parameter_set= kwargs.pop('parameter_set',None))
#         #mech.mode_superpos(f)
#
#
#     res=pyansys.read_binary(f"{jid}.rst")
#     node_numbers=res.geometry['nnum']
#     num_nodes=len(node_numbers)
#     time_values=res.time_values
#     dt = time_values[1]-time_values[0]
#     time_values -= dt
#
#
#     solution_data_info = res.solution_data_info(0)
#     DOFS = solution_data_info['DOFS']
#
#     uz = DOFS.index(3)
#     nnum, all_disp = res.nodal_time_history('NSL')
#     #print(all_disp)
#
#     ydata = all_disp[:,np.where(nnum==20)[0][0],uz]
# #     print(ydata)
#     this_t = time_values
#     ty=np.vstack((this_t, ydata))
#     plot.figure()
#
#     plot.gca().plot(this_t, ydata, color='black', **kwargs)
#     plot.plot(this_t, ydata, ls='none', marker='+')
#     plot.show()
#
#     np.savetxt(str(pathlib.Path.cwd() / 'polyuq_results' / '{}.csv'.format(jid)), ty.T)
#     with open(str(pathlib.Path.cwd() / 'polyuq_results' / 'description_new.txt'), "at") as f:
#         f.write("{},\t{:1.3f},\t{:1.4f},\t{:1.5f}\n".format(jid,k_2,d,deltat))
#
# def generate_student_nl(ansys, omega, zeta, d0, deltat=None,dt_fact=None,timesteps=None, num_cycles=None, f_scale=None, save=True, **kwargs):
#
#     global jid
#     oldjid= jid
#     jid=str(uuid.uuid4()).split('-')[-1]
#     ansys.filname(fname=jid, key=1)
#     for file in glob.glob(f'{tempfile.gettempdir()}/{oldjid}.*'):
#         print(f'removing {file}')
#         os.remove(file)
#     print(jid)
#
#     total, used, free = shutil.disk_usage("/dev/shm/")
#     if free/total < 0.1:
#         raise RuntimeError(f'Disk "/dev/shm/ almost full {used} of {total}')
#
#     assert deltat is not None or dt_fact is not None
#     assert timesteps is not None or num_cycles is not None
#
#     m    = 1    # kg
#     k_2  = omega**2*m/(1-zeta**2)# N/m
#     d=zeta*(2*np.sqrt(k_2*m))
#
#     f_max = np.sqrt((k_2)/m)/np.pi/2
#
#     if dt_fact is None:
#         dt_fact =deltat*f_max # \varOmega /2/pi
#         print("\\varOmega",dt_fact*2*np.pi)
#     elif deltat is None:
#         deltat = dt_fact/f_max
#
#     if dt_fact*2*np.pi > 0.1: print("Warning \\varOmega > 0.1")
#
#     nl_ity=kwargs.pop('nl_ity',0)
#
#     if timesteps is None:
#         timesteps = int(np.ceil(num_cycles/f_max/deltat))
#     elif num_cycles is None:
#         num_cycles = int(np.floor(timesteps*f_max*deltat))
#     if d0 is not None:
#         print("Simulating system with omega {:1.3f}, zeta {:1.3f}, d0 {:1.3f}. deltat {:1.5f}, dt_fact {:1.2f}, timesteps {}, num_cycles {}, nonlinearity {}".format(omega, zeta, d0, deltat, dt_fact, timesteps, num_cycles, nl_ity))
#     elif f_scale is not None:
#         print("Simulating system with omega {:1.3f}, zeta {:1.3f}, random excitation variance {:1.3f}, deltat {:1.5f}, dt_fact {:1.2f}, timesteps {}, num_cycles {}, nonlinearity {}".format(omega, zeta, f_scale, deltat, dt_fact, timesteps, num_cycles,nl_ity))
#
#     if d0 is None:
#         sigma_scale = 1 # 68–95–99.7 rule
#         empiric_factor=1/2.5
#         d_max = sigma_scale*f_scale/2/k_2/zeta*empiric_factor
#     else:
#         d_max = d0
#
#     mech = Mechanical(ansys)
#     voigt_kelvin=mech.voigt_kelvin(k=k_2*(1-nl_ity), d=d)
#     nonlinear = mech.nonlinear(nl_ity=nl_ity, d_max=d_max, k_lin=k_2)
#     mass = mech.mass(m)
#     mech.build_sdof(
#                mass=mass,
#                voigt_kelvin = voigt_kelvin,
#                nonlinear = nonlinear,
#                d_init=d0
#                )
#
#     mech.modal(True)
#
#     f_max = process(f'{jid}.rst', last_analysis=mech.last_analysis)[0].max()
#     f_analytical = np.sqrt((k_2)/m)/np.pi/2
#     zeta_analytical = d/2/np.sqrt(k_2*m)*100 # in percent of critical damping
#     #print(f_analytical,zeta_analytical)
#
#     mech.timesteps=timesteps
#     mech.deltat = deltat
#
#     #print(timesteps,deltat)
#
#     f=np.zeros(timesteps)
#     if d0 is not None:
#         f[0]=d0
#         mech.transient(d=f,parameter_set= kwargs.pop('parameter_set',None))
#     else:
#         assert f_scale is not None
#         f = np.random.normal(loc=0.0,scale=f_scale, size=(timesteps,))
#         mech.transient(f=f,parameter_set= kwargs.pop('parameter_set',None))
#         #mech.mode_superpos(f)
#
#
#     res=pyansys.read_binary(f"{jid}.rst")
#     node_numbers=res.geometry['nnum']
#     num_nodes=len(node_numbers)
#     time_values=res.time_values
#     dt = time_values[1]-time_values[0]
#     time_values -= dt
#
#
#     solution_data_info = res.solution_data_info(0)
#     DOFS = solution_data_info['DOFS']
#
#     uz = DOFS.index(3)
#     nnum, all_disp = res.nodal_time_history('NSL')
# #     print(all_disp)
#
#     ydata = all_disp[:,np.where(nnum==20)[0][0],uz]
# #     print(ydata)
#     this_t = time_values
#     ty=np.vstack((this_t, ydata))
#     plot.figure()
#
#     plot.gca().plot(this_t, ydata, color='black', **kwargs)
#     plot.axhline(d_max)
#     plot.axhline(-d_max)
# #     plot.plot(this_t, ydata, ls='none', marker='+')
#     #plot.show()
#     #print("{},\t{:1.3f},\t{:1.4f},\t{:1.5f},\t{:1.5f},\t{:1.3f}\n".format(jid,omega,zeta,d0,deltat,nl_ity))
#     source_folder = str(pathlib.Path.cwd() / 'polyuq_results' / 'datasets_nonlinear_')
#     if d0 is not None:
#         source_folder+='decay/'
#     else:
#         source_folder +='ambient/'
#
#     if save:
#         np.savetxt(f"{source_folder}{jid}.csv", ty.T)
#         np.savetxt(f"{source_folder}inp{jid}.csv",f)
#         with open(f"{source_folder}description.txt", "at") as f:
#             f.write("{},\t{:1.3f},\t{:1.4f},\t{:1.5f},\t{:1.5f},\t{:1.3f}\n".format(jid,k_2,d,d_max,deltat,nl_ity))
#


def generate_sdof_time_hist(ansys,
                            omega, zeta, m=1, fric_visc_rat=0, nl_ity=0,  # structural parameters
                            dscale=None, f_scale=None,  # loading parameters
                            deltat=None, dt_fact=None, timesteps=None, num_cycles=None, num_meas_nodes=None, meas_nodes=None,  # signal parameters
                            savefolder=None, working_dir=tempfile.gettempdir(), jid=None,  # function parameters
                            ** kwargs):
#                      ansys, jid, working_dir=tempfile.gettempdir(), # function parameters
#                             omega, zeta, m=1, fric_visc_rat=0, nl_ity=0, # structural parameters
#                             d0=None, f_scale=None,  # loading parameters
#                             deltat=None, dt_fact=None, timesteps=None, num_cycles=None, # signal parameters
#                             **kwargs):
    assert np.abs(nl_ity) <= 0.5
    assert fric_visc_rat <= 1
    try:
        ansys.finish()
    except pyansys.errors.MapdlExitedError as e:
        logger.exception(e)
        ansys = Mechanical.start_ansys()
    ansys.clear()

    # compute global structural parameters
    k = omega ** 2 * m / (1 - zeta ** 2)  # N/m
    d = 2 * zeta * np.sqrt(k * m)

    # Compute maximum displacement from expected excitation parameters and structural parameters
    # this is used mainly for computation of the nonlinear springs, where dmax should not be exceeded
    if f_scale is not None:  # assuming gaussian white noise excitation
        sigma_scale = 1  # 68–95–99.7 rule
        empiric_factor = 1 / 2.5
        d_max = sigma_scale * f_scale / 2 / k / max(zeta, 1 / 1000) * empiric_factor
    elif dscale is not None:  # assuming free decay
        d_max = dscale
        # TODO: If free-decay whith white noise the maximum of both should be taken
    else:
        d_max = kwargs.pop('d_max', None)
        if d_max is None:
            raise RuntimeError('Neither d0 nor f_scale were defined.')

    # Compute expected absolute displacement from excitation parameters and structural parameters
    # this is used mainly for computation of equivalent friction damping, which should be equivalent on average
    # https://en.wikipedia.org/wiki/Average_absolute_deviation
    # ratio of mean absolute deviation to standard deviation is sqrt(2 / π) = 0.79788456...
    if f_scale is not None:  # assuming gaussian white noise excitation
        # dynamic amplification factor https://en.wikipedia.org/wiki/Structural_dynamics#Damping
        DAF = 1 + np.exp(-zeta * np.pi)
        d_mean = np.sqrt(2 / np.pi) * f_scale / k * DAF
    elif dscale is not None:  # assuming free decay
        d_mean = dscale
        # TODO: If free-decay whith white noise the maximum of both should be taken
    else:
        d_mean = kwargs.pop('d_mean', None)
        if d_mean is None:
            raise RuntimeError('Neither d0 nor f_scale were defined.')

    # compute elemental parameters
    k_lin = k * (1 - nl_ity)
    if nl_ity:
        logger.warning("Should k_lin really be reduced here? check definitions of nonlinear again")
    d_visc = d * (1 - fric_visc_rat)
    d_fric = d * fric_visc_rat

    # generate Elements
    mech = Mechanical(ansys, jobname=jid, wdir=working_dir)

#     if kwargs.pop('remove_mass',False):mass = mech.mass(m*1e-5)
#     else: mass = mech.mass(m)

#     if fric_visc_rat == 0: # only viscous damping
#         voigt_kelvin=mech.voigt_kelvin(k=k_lin, d=d_visc)
#         coulomb = None
#     else:
    if fric_visc_rat > 0 and fric_visc_rat <= 1:

        k_1 = k_lin * 100

        # Computation of equivalent friction damping
        fsl_equiv = d_mean / 2 * (k_1 - np.sqrt(k_1 * (-np.pi * omega * d_fric + k_1)))
        logger.info(f"equivalent slip force {fsl_equiv} for d {d_fric} at omega {omega} and d_mean {d_mean} with k_1 {k_1}")
    else:
        fsl_equiv = None
#         coulomb = mech.coulomb(k_2=k_lin, k_1=k_1, f_sl= f_sl_equiv, d=d_visc)

#     if nl_ity!=0:
#         assert abs(nl_ity)<=0.5
#         nonlinear = mech.nonlinear(nl_ity, d_max, k)
#     else:
#         nonlinear = None

    # build final modal
#     mech.build_sdof(
#        mass=mass,
#        voigt_kelvin = voigt_kelvin,
#        coulomb = coulomb,
#        nonlinear = nonlinear,
#        d_init=d0
#        )
    mech.build_mdof(nodes_coordinates=[(1, 0, 0, 0), (2, 0, 0, 0)],
                   k_vals=[k_lin], masses=[1, 1], d_vals=[d_visc], damping=None,
                   sl_forces=[fsl_equiv], nl_rats=[nl_ity], d_max=[0, d_max],
                   num_modes=1, meas_nodes=[2], **kwargs)

    # check modal results match with expected parameters
    # TODO: account for nonlinear stiffness
    try:
        freqs, damping, _ = mech.modal(True)
        logger.info(f"Numeric solution: f={freqs} zeta={damping*100}")
    except Exception as e:
        err_file = os.path.join(ansys.directory, ansys.jobname + '.err')
        with open(err_file) as f:
            lines_found = []
            block_counter = -1
            while len(lines_found) < 10:
                try:f.seek(block_counter * 4098, os.SEEK_END)
                except IOError:
                    f.seek(0)
                    lines_found = f.readlines()
                    break
                lines_found = f.readlines()
                block_counter -= 1
            logger.error(str(lines_found[-10:]))

            raise e
        logger.warning('Modal analysis failed, probably something is wrong with the model.')

    f_analytical = np.sqrt((k) / m) / np.pi / 2
    zeta_analytical = d / 2 / np.sqrt(k * m) * 100  # in percent of critical damping
    logger.info(f"Analytic solution: f={f_analytical},zeta={zeta_analytical}")

#     # Setup transient simulation parameters
#     assert deltat is not None or dt_fact is not None
#     assert timesteps is not None or num_cycles is not None
#
#     f_max = omega/np.pi/2
#
#     if dt_fact is None:
#         dt_fact =deltat*f_max
#     elif deltat is None:
#         deltat = dt_fact/f_max
#
#     if dt_fact*2*np.pi > 0.1: print("Warning \\varOmega > 0.1")

    if dscale is not None:
        t_vals, resp_hist = mech.free_decay(0, dscale, dt_fact=dt_fact, num_cycles=num_cycles, **kwargs)
    elif f_scale is not None:
        t_vals, resp_hist, inp_hist = mech.ambient(f_scale, dt_fact=dt_fact, num_cycles=num_cycles, **kwargs)
    else:
        logger.info(f"Simulating system with omega {omega:1.3f}, zeta {zeta:1.3f}, fric_visc_rat {fric_visc_rat:1.3f}, nonlinearity {nl_ity}, user_provided excitation, deltat {deltat:1.5f}, dt_fact {dt_fact:1.2f}, timesteps {timesteps}, num_cycles {num_cycles}")
        # provide a custom forcing function for verification purposes
        f = kwargs.pop('f', None)
        t_vals, resp_hist = mech.transient(f=f, **kwargs)

    if kwargs.pop('data_hadidi', None):

       return np.hstack((t_vals[:, np.newaxis], resp_hist[0])), k, d_visc, d_max, fsl_equiv, deltat

    return

def generate_conti(ansys, deltat=None,timesteps=None, num_cycles=None, **kwargs):

    assert deltat is not None
    assert timesteps is not None or num_cycles is not None
    try: os.remove('file.rst')
    except: pass


    parameters = {
                'L'         : 40,

                'E'         : 210e9,
                'A'         : 4.97e-3,
                'rho'       : 7850,
                'Iy'        : 15.64e-6,
                'Iz'        : 15.64e-6,

                'ky_nl'     : 0,
                'kz_nl'     : 0,
                'x_knl'     : 15,

                'm_tmd'     : 0,
                'ky_tmd'    : 0,
                'kz_tmd'    : 0,
                'dy_tmd'    : 0,
                'dz_tmd'    : 0,
                'x_tmd '    : 40,

            }
    initial = {'d0y'       : 0.1,
               'd0z'       : 0.1,
               'x_d0'      : 40}
    mech = Mechanical(ansys)


    mech.build_conti(parameters, Ldiv = 10, initial=initial)


# def accuracy_study(ansys):
#
#     timesteps=1000
#     results = np.zeros((40,8))
#
#     if False:
#         for i in range(40):
#             dt_fact = (i+1)*0.005
#
#             mech = Mechanical(ansys)
#
#             fact=1e5
#             m    = fact/1    # kg
#             k_2  = 1*fact*4*np.pi**2 # 3947841.76 N/m
#             ampl = fact/10   # N
#             d=0.00*2*np.sqrt(k_2*m) # 0
#
#             voigt_kelvin=mech.voigt_kelvin(k=k_2, d=d)
#             mass = mech.mass(m)
#             mech.build_sdof(
#                        mass=mass,
#                        voigt_kelvin = voigt_kelvin,
#                        d_init=1
#                        )
#
#             mech.modal()
#
#             f_max = process('file.rst', last_analysis=mech.last_analysis).max()
#             f_analytical = np.sqrt((k_2)/m)/np.pi/2
#             zeta_analytical = d/2/np.sqrt(k_2*m)*100 # in percent of critical damping
#             print(f_analytical,zeta_analytical)
#
#             deltat = dt_fact/f_max
#             dt_fact =deltat/f_max
#             num_cycles = 16
#             timesteps = int(np.ceil(num_cycles/dt_fact))
#
#             mech.timesteps=timesteps
#             mech.deltat = deltat
#
#             f=np.zeros(timesteps)
#             f[0]=1
#             mech.transient(d=f)
#
#             popt,perr = response_frequency('file.rst')
#             results[i,:4]=popt
#             results[i,4:]=perr
#
#
#         ansys.exit()
#         np.save('newmark_errors.npz', results)
#
#     results = np.load('newmark_errors.npz.npy')
#     dt_fact = np.linspace(0.005,0.2,40)
#     #print('R: {:1.3f} m, zeta: {:1.3f} \%, f_d: {:1.4f} Hz, phi: {:1.4f}'.format(popt[0], popt[1]*100,  popt[2]/2/np.pi, popt[3]*180/np.pi))
#
#     #confidence intervals
#     sqrt_ts = np.sqrt(np.ceil(16/dt_fact))
#     tppf = [scipy.stats.t.ppf(0.95,int(timesteps)) for timsteps in np.ceil(16/dt_fact)]
#     #scaling
#     results[results[:,0]<0,0]*=-1
#     results[:,0]+=-0.5
#     results[:,0]/=0.5
#
#     results[:,4]/=0.5**2
#     results[:,4]/=sqrt_ts
#     results[:,4]*=tppf
#
#
#     results[:,1]*=100
#
#     results[:,5]*=100**2
#     results[:,5]/=sqrt_ts
#     results[:,5]*=tppf
#
#
#     results[:,2]/=2*np.pi
#     results[:,2]-=1# to make it delta_frequency
#
#     results[:,6]/=(2*np.pi)**2
#     results[:,6]/=sqrt_ts
#     results[:,6]*=tppf
#
#
#     results[:,3]=np.arcsin(np.sin(results[:,3]))
#     results[:,3]*=180/np.pi
#     results[results[:,3]<=0,3]*=-1
#
#     results[:,7]*=(180/np.pi)**2
#     results[:,7]/=sqrt_ts
#     results[:,7]*=tppf
#
#     with matplotlib.rc_context(rc=print_context_dict):
#         fig,axes = plot.subplots(2,2, sharex=True)
#         print(fig.get_size_inches())
#         axes = axes.flatten()
#         ylabels = ['$\Delta R$ [\si{\percent}]', '$\Delta \zeta [\si{\percent}]$', '$\Delta f [\si{\percent}]$', '$\Delta \phi [\si{\degree}]$']
#
#         for j in range(4):
#
#             axes[j].errorbar(dt_fact, results[:,j], yerr=results[:,j+4], errorevery=2)
#             axes[j].set_ylabel(ylabels[j])
#             #axes[j].plot(dt_fact[:], results[:,j], marker='+')
#         axes[2].set_xlabel('Frequency ratio $\\nicefrac{f_{\\text{max}}}{f_s} [-]$')
#         axes[3].set_xlabel('Frequency ratio $\\nicefrac{f_{\\text{max}}}{f_s} [-]$')
#         plot.subplots_adjust(left=0.085, right=0.985, top=0.985, bottom=0.085, hspace=0.1, wspace=0.18)
#         plot.show(block=True)


def verify_friction(ansys):
    ########
    # Verification code for friction / generate_damping
    ########
    plot.subplot()
    zeta = 10 / 100
    num_cycles = 8
    d0 = 1
    omega = 2
    generate_sdof_time_hist(ansys, omega=omega, zeta=zeta, fric_visc_rat=0, d0=d0, deltat=0.03, num_cycles=8)
    generate_sdof_time_hist(ansys, omega=omega, zeta=zeta, fric_visc_rat=0.5, d0=d0, deltat=0.03, num_cycles=8)
    generate_sdof_time_hist(ansys, omega=omega, zeta=zeta, fric_visc_rat=1, d0=d0, deltat=0.03, num_cycles=8)
    plot.show()


def hysteresis(ansys):
    '''
    Generate a hysteresis curve for different types of damping and stiffness curves
    For generating a hysteresis, the spring/damper/slider system is displaced by a known force
    (typically a full sine cycle, where a first quarter cycle is needed to reach stead-state and is discarded
    any other forces should not be present, that means, the inertia forces of the mass should be removed
    two options exist: turning off transient effects (stepped static analysis) or removing the mass element
    when transient effects are turned off, velocities are not computed and therefore viscous damping forces are zero
    removing the mass causes the response to grow out of bounds, therefore mass is scaled by a factor
    '''

    def PolyArea(x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    # free-decay time histories for 2 % damping over 10 cycles in one plot
    # linear-viscous, duffing degressive (viscous), duffing progressive (viscous), friction
    if False:
        if False:  # generate data
            omega = 1
            ydata1, ydata2, ydata3, ydata4 = [], [], [], []
            for zeta in [2, 5, 10]:
                ty1, _ = generate_sdof_time_hist(ansys, omega, zeta / 100, fric_visc_rat=0, nl_ity=0, d0=1, dt_fact=1 / 1000, num_cycles=10)
                ty2, _ = generate_sdof_time_hist(ansys, omega, zeta / 100, fric_visc_rat=0, nl_ity=-0.5, d0=1, dt_fact=1 / 1000, num_cycles=10)  # degressive
                ty3, _ = generate_sdof_time_hist(ansys, omega, zeta / 100, fric_visc_rat=0, nl_ity=0.5, d0=1, dt_fact=1 / 1000, num_cycles=10)  # progressive
                ty4, _ = generate_sdof_time_hist(ansys, omega, zeta / 100, fric_visc_rat=1, nl_ity=0, d0=1, dt_fact=1 / 1000, num_cycles=10)
                ydata1.append(ty1)
                ydata2.append(ty2)
                ydata3.append(ty3)
                ydata4.append(ty4)
            np.savez('time_hist_decay_different_systems.npz', ydata1=ydata1, ydata2=ydata2, ydata3=ydata3, ydata4=ydata4)
        else:
            arr = np.load('time_hist_decay_different_systems.npz')  #
            ydata1 = arr['ydata1']
            ydata2 = arr['ydata2']  # degressive
            ydata3 = arr['ydata3']  # progressive
            ydata4 = arr['ydata4']

        with matplotlib.rc_context(rc=print_context_dict):
            fig, axes = plot.subplots(2, 2, True, True)
            for ty1, ty2, ty3, ty4, zeta, color in reversed(list(zip(ydata1, ydata2, ydata3, ydata4, [2, 5, 10], ['black', 'darkgrey', 'lightgray']))):

                axes.flat[0].plot(ty1[0, :], ty1[1, :], color=color, linestyle='solid', label=f'{zeta}', linewidth=1)
                axes.flat[1].plot(ty2[0, :], ty2[1, :], color=color, linestyle='solid', label=f'{zeta}', linewidth=1)
                axes.flat[2].plot(ty3[0, :], ty3[1, :], color=color, linestyle='solid', label=f'{zeta}', linewidth=1)
                axes.flat[3].plot(ty4[0, :], ty4[1, :], color=color, linestyle='solid', label=f'{zeta}', linewidth=1)

            axes.flat[0].set_title('Linear, viscous', fontdict={'fontsize':10})
            axes.flat[1].set_title('Duffing (degressive), viscous', fontdict={'fontsize':10})
            axes.flat[2].set_title('Duffing (progressive), viscous', fontdict={'fontsize':10})
            axes.flat[3].set_title('Linear, friction', fontdict={'fontsize':10})

            axes[0, 0].set_ylabel('y [\si{\metre}]')
            axes[0, 0].yaxis.set_label_coords(-0.1, 0.5)
            axes[1, 0].set_ylabel('y [\si{\metre}]')
            axes[1, 0].yaxis.set_label_coords(-0.1, 0.5)
            axes[1, 0].set_xlabel('t [\si{\second}]')
            # axes[1,0].xaxis.set_label_coords(0.49, -0.07)
            axes[1, 1].set_xlabel('t [\si{\second}]')
            # axes[1,1].xaxis.set_label_coords(0.49, -0.06)

            for ax in axes.flat:
                ax.xaxis.set_major_formatter(plot.FuncFormatter(
                    lambda val, pos: '${:d}\pi$'.format(int(val / np.pi)) if val != 0 else '0'))
                ax.xaxis.set_minor_locator(plot.MultipleLocator(2 * np.pi))
                ax.xaxis.set_major_locator(plot.MultipleLocator(4 * np.pi))
                ax.grid(True)
                ax.set_ylim((-1, 1))
                ax.set_xlim((0, 2 * np.pi * 10))
                ax.set_yticks([-1, -0.5, 0, 0.5, 1],)
                ax.set_yticklabels(['$-1$', '$-\\nicefrac{1}{2}$', '$0$', '$\\nicefrac{1}{2}$', '$1$'])

            fig.subplots_adjust(left=0.08, bottom=0.125, right=0.970, top=0.940, hspace=0.2, wspace=0.1)
            plot.savefig(str(pathlib.Path('figures') / 'time_hist_decay_different_systems.pdf'))
            plot.savefig(str(pathlib.Path('figures') / 'time_hist_decay_different_systems.png'))
            plot.show()

    # massless system loaded with 1.25 sine cycles with 1 %, 5% and 20 % damping (equivalent)
    # 4 plots of hysteresis for linear-viscous, duffing degressive (viscous), duffing progressive (viscous), friction

    if False:

        if False:
            omega = 1
            f = np.sin(np.linspace(0, 2.5 * np.pi, 1250)) * 1
            ydata1, ydata2, ydata3, ydata4 = [], [], [], []
            all_ydata = [ydata1, ydata2, ydata3, ydata4]

            for ydata, nl_ity in zip(all_ydata, [0, 0.5, -0.5]):  # linear,#progressive,#degressive

                for zeta in [2, 5, 10]:
                    # for viscous timint=1 is needed, therefore reduce mass by factor 1e-6 -> remove_mass=True
                    ty, _ = generate_sdof_time_hist(ansys, omega, zeta / 100, fric_visc_rat=0, nl_ity=nl_ity, d_max=1, dt_fact=1 / 1000, num_cycles=1.25, remove_mass=True, timint=1, f=f)
                    ydata.append(ty[1, :])

            ydata = ydata4
            for zeta in [1, 5, 20]:
                # for friction timint=0 and normal mass
                ty, _ = generate_sdof_time_hist(ansys, omega, zeta / 100, fric_visc_rat=1, nl_ity=0, d_max=1, dt_fact=1 / 1000, num_cycles=1.25, remove_mass=False, timint=0, f=f)
                ydata.append(ty[1, :])
            np.savez('hysteresis_different_systems.npz', f=f, ydata1=ydata1, ydata2=ydata2, ydata3=ydata3, ydata4=ydata4)
        else:
            arr = np.load('hysteresis_different_systems.npz')  #
            f = arr['f']
            ydata1 = arr['ydata1']
            ydataprog = arr['ydata2']  # progressive
            ydatadeg = arr['ydata3']  # degressive
            ydata4 = arr['ydata4']
        with matplotlib.rc_context(rc=print_context_dict):
            fig, axes = plot.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
            for ydata, ax in zip([ydata1, ydatadeg, ydataprog, ydata4], axes.flat):
                for y, zeta, color in reversed(list(zip(ydata, [2, 5, 10], ['black', 'darkgrey', 'lightgray']))):
                    x = f[250:]
                    y = y[250:]
                    area = PolyArea(x, y)
                    ax.plot(y, x, color=color, label=f'{zeta}', linewidth=1)
                # ax.legend()
            axes.flat[0].set_title('Linear, viscous', fontdict={'fontsize':10})
            axes.flat[1].set_title('Duffing (degressive), viscous', fontdict={'fontsize':10})
            axes.flat[2].set_title('Duffing (progressive), viscous', fontdict={'fontsize':10})
            axes.flat[3].set_title('Linear, friction', fontdict={'fontsize':10})

            axes[0, 0].set_ylabel('f [\si{\\newton}]')
            axes[0, 0].yaxis.set_label_coords(-0.1, 0.5)
            axes[1, 0].set_ylabel('f [\si{\\newton}]')
            axes[1, 0].yaxis.set_label_coords(-0.1, 0.5)
            axes[1, 0].set_xlabel('y [\si{\metre}]')
            # axes[1,0].xaxis.set_label_coords(0.49, -0.07)
            axes[1, 1].set_xlabel('y [\si{\metre}]')
            # axes[1,1].xaxis.set_label_coords(0.49, -0.06)

            for ax in axes.flat:
                # ax.grid(True)
                ax.set_yticks([-1, -0.5, 0, 0.5, 1],)
                ax.set_yticklabels(['$-1$', '$-\\nicefrac{1}{2}$', '$0$', '$\\nicefrac{1}{2}$', '$1$'])
                ax.set_xticks([-1, -0.5, 0, 0.5, 1],)
                ax.set_xticklabels(['$-1$', '$-\\nicefrac{1}{2}$', '$0$', '$\\nicefrac{1}{2}$', '$1$'])
                ax.set_xlim((-1, 1))
                ax.set_ylim((-1, 1))
            fig.subplots_adjust(left=0.08, bottom=0.125, right=0.970, top=0.940, hspace=0.2, wspace=0.1)
            plot.savefig(str(pathlib.Path('figures') / 'hysteresis_different_systems.pdf'))
            plot.savefig(str(pathlib.Path('figures') / 'hysteresis_different_systems.png'))
            plot.show()

    # effect of forcing amplitude and rate on the hysteresis with 5 % damping (equivalent)
    # 5 plots in a 3x3 subplot, where the middle is the base extending left/right/up/down
    # all four system in each subplot

    if False:

        if False:
            zeta = 20
            ydata1, ydata2, ydata3, ydata4 = [], [], [], []
            fdata = []
            # omega, f, axes
            for omega, f_factor in [(1, 2,),
                                    (0.5, 1),
                                    (1 , 1),
                                    (2, 1),
                                    (1, 0.5)]:

                # TODO: d_max=1 is actually wrong here for increased amplitudes
                f = np.sin(np.linspace(0, 2.5 * np.pi, 1250)) * f_factor
                ty, _ = generate_sdof_time_hist(ansys, 1, zeta / 100, fric_visc_rat=0, nl_ity=0, d_max=1, dt_fact=1 / omega / 1000, timesteps=1250, remove_mass=True, timint=1, f=f)
                ydata1.append(ty)
                ty, _ = generate_sdof_time_hist(ansys, 1, zeta / 100, fric_visc_rat=0, nl_ity=0.5, d_max=1, dt_fact=1 / omega / 1000, timesteps=1250, remove_mass=True, timint=1, f=f)
                ydata2.append(ty)
                ty, _ = generate_sdof_time_hist(ansys, 1, zeta / 100, fric_visc_rat=0, nl_ity=-0.5, d_max=1, dt_fact=1 / omega / 1000, timesteps=1250, remove_mass=True, timint=1, f=f)
                ydata3.append(ty)
                ty, _ = generate_sdof_time_hist(ansys, 1, zeta / 100, fric_visc_rat=1, nl_ity=0, d_max=1, dt_fact=1 / omega / 1000, timesteps=1250, remove_mass=False, timint=0, f=f)
                ydata4.append(ty)

                fdata.append(f)
            np.savez('hysteresis_different_forcing.npz', fdata=fdata, ydata1=ydata1, ydata2=ydata2, ydata3=ydata3, ydata4=ydata4)
        else:
            arr = np.load('hysteresis_different_forcing.npz')  #
            fdata = arr['fdata']
            ydata1 = arr['ydata1']
            ydataprog = arr['ydata2']  # progressive
            ydatadeg = arr['ydata3']  # degressive
            ydata4 = arr['ydata4']

        with matplotlib.rc_context(rc=print_context_dict):
            fig, axes = plot.subplots(nrows=3, ncols=3, sharex=False, sharey=False)
            for ty1, ty2, ty3, ty4, f, ax in zip(ydata1, ydatadeg, ydataprog, ydata4, fdata, (axes[0, 1], axes[1, 0], axes[1, 1], axes[1, 2], axes[2, 1])):
                x = f[250:]

                y = ty1[1, 250:]
                ax.plot(y, x, color='black', label=f'')
                y = ty2[1, 250:]
                # ax.plot(y,x, color='black', label=f'')
                y = ty3[1, 250:]
                # ax.plot(y,x, color='black', label=f'')
                y = ty4[1, 250:]
                # ax.plot(y,x, color='black', label=f'')

            for ax in axes.flat:
                ax.set_xlim((-2, 2))
                ax.set_ylim((-2, 2))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                # ax.spines['right'].set_color('none')
                # ax.spines['top'].set_color('none')

                if ax == axes[1, 1]:
                    ax.spines['left'].set_position('zero')
                    ax.spines['bottom'].set_position('zero')

                    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
                    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)
                    # ax.axis['xzero'].set_axisline_style("-|>")
                    # ax.axis['yzero'].set_axisline_style("-|>")
                    ax.xaxis.set_label_coords(0.9, 0.45)
                    ax.yaxis.set_label_coords(0.45, 0.95)
                    ax.set_ylabel('f [\si{\\newton}]')
                    ax.set_xlabel('y [\si{\metre}]')
                    continue
                #
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                # ax.spines['left'].set_color('none')
                # ax.spines['bottom'].set_color('none')
            plot.annotate('Increasing excitation frequency', xycoords='figure fraction', textcoords='figure fraction', xytext=(0.3, 0.05), xy=(0.9, 0.05), arrowprops=dict(arrowstyle='->', facecolor='black'), verticalalignment='center',)
            plot.annotate('Increasing amplitude', xycoords='figure fraction', textcoords='figure fraction', rotation=90, xytext=(0.05, 0.5), xy=(0.05, 0.9), arrowprops=dict(arrowstyle='->', facecolor='black'), horizontalalignment='center',)
            fig.subplots_adjust(left=0, bottom=0, right=1, top=0.98, hspace=0, wspace=0.03)

            # plot.savefig(str(pathlib.Path('figures') / 'hysteresis_different_forcing.pdf'))
            # plot.savefig(str(pathlib.Path('figures') / 'hysteresis_different_forcing.png'))

            plot.show()


def stepsize_example(ansys):
#########
# Example showing effect of stepsizes
#########
    
    from mpldatacursor import datacursor
    with matplotlib.rc_context(rc=print_context_dict):
        plot.figure(tight_layout=True)
        zeta_ = 0 / 100
        omega_ = 1
        for method, gamma_, beta_, linestyle in [('Average Acceleration', 0.5, 0.25, 'solid'),
                                                     ('Linear Acceleration', 0.5, 1 / 6, 'dashed'),
    #                                                  ('Central Difference',0.5,0,'dotted'),
                                                     ('$\\gamma=0.6, \\beta=0.3025$', 0.6, 0.3025, 'dashdot'),
                                                     ('$\\gamma=0.9142, \\beta=0.5$', np.sqrt(2) - 1 / 2, 1 / 2, (0, (3, 1, 1, 1, 1, 1))), ]:
            generate_sdof_time_hist(ansys, omega=omega_, zeta=zeta_, d0=1, dt_fact=0.1, num_cycles=4, parameter_set=(gamma_, beta_), label=method, linestyle=linestyle)

    #     generate_student(ansys, omega=1, zeta=0, d0=1, dt_fact=0.001, num_cycles=1, parameter_set='LAM', color='grey', label='Analytic')
        t = np.linspace(0, 20, 3000)
        ydata = free_decay(t, R=1 / 2, zeta=zeta_, omega_d=omega_, phi=0)
        plot.plot(t, ydata, color='grey', label='Analytical')
#         plot.plot(t,2*1/2*np.exp(-zeta_*omega_/(np.sqrt(1-zeta_**2))*t), color='grey', label='Envelope', ls='dotted')
#         plot.plot(t,-2*1/2*np.exp(-zeta_*omega_/(np.sqrt(1-zeta_**2))*t), color='grey', label='Envelope', ls='dotted')
        plot.xlim((0, 20))
        plot.ylim((-1.1, 1.1))
        datacursor(formatter='{label}'.format, bbox=None, draggable=True, display='multiple')
    #     plot.legend()
        plot.xlabel('Time [\\si{\\second}]')
        plot.ylabel('Displacement [\\si{\\metre}]')
        plot.show()

#     plot.figure()
#     plot.subplot()
#     import sympy
#     sympy.init_printing()
#     h,beta,gamma,zeta,omega,Omega,eta=sympy.symbols('h \\beta \gamma \zeta \omega \Omega \eta', real=True, positive=True)
#
#     A1=sympy.Matrix([[1,0, -h**2*beta],[0, 1, -h*gamma],[omega**2,2*zeta*omega, 1]])
#     #with this one, Eq. 3.17 gives an oscillating response
#     A2=sympy.Matrix([[1,h,h**2*(0.5-beta)],[0,1,h*(1-gamma)],[0,0,0]])
#     A=A1.solve(A2)
#     dt = 2*np.pi*0.1
#     timesteps =30
#     t = np.linspace(0,stop=dt*timesteps,num=timesteps)
#     for beta_ in [1/6,1/4,0]:
#         Anp = np.array(A.subs(omega,1).subs(zeta,0).subs(h,dt).subs(gamma,0.5).subs(beta,beta_)).astype(np.float64)
#         res = np.zeros((3,timesteps+1))
#         if False:#not beta_:
#             res[0,1]=1 #u0
#             res[0,0]=res[0,1]-dt*res[1,1]+dt**2/2*res[2,1] #u-1
#             res[2,1]=-1 #dot{u}_0
#             k=1
#             m=1
#             khat = m/dt**2
#             a=m/dt**2
#             b=k-2*m/dt**2
#             for i in range(1,timesteps+1):
#                 phat=-a*res[0,i-2]-b*res[0,i-1]
#                 res[0,i] = phat/khat
#             res/=np.pi/2
#         else:
#             res[0,0]=1
#             for i in range(1,timesteps+1):
#
#                 res[:,i]= Anp.dot(res[:,i-1])
#         if False:#not beta_:
#             res = res[:,1:]
#         else:
#             res = res[:,:-1]
#         plot.plot(t,res[0,:], label=f'{beta_}')
#
#     dt = 2*np.pi*0.001
#
#     timesteps =3000
#     t=np.linspace(0,stop=dt*timesteps,num=timesteps)
#     Anp = np.array(A.subs(omega,1).subs(zeta,0).subs(h,dt).subs(gamma,0.25).subs(beta,1/6)).astype(np.float64)
#     res = np.zeros((3,timesteps+1))
#     res[0,0]=1
#     for i in range(1,timesteps+1):
#         res[:,i]= Anp.dot(res[:,i-1])
#     plot.plot(t,res[0,:-1])
#     plot.xlim((0,20))
#     plot.ylim((-1.1,1.1))
#     plot.legend()
#     plot.show()

# def spectr_shift_example(ansys):
#     with matplotlib.rc_context(rc=print_context_dict):
#         fig, axes = plot.subplots(nrows = 2, ncols=1)
#         generate_mdof(ansys, deltat=1/200, num_cycles=8,axes=axes)
#         generate_mdof(ansys, deltat=1/100, num_cycles=8, axes=axes)
#         generate_mdof(ansys, deltat=1/50, num_cycles=8, axes=axes)
#
#
#         axes[1].axvline(-1, color='grey', label="Theoretical")
#
#         axes[0].set_ylabel('Acceleration [\\si{\\metre\\per\\square\\second}]')
#         axes[0].set_xlabel('Time [\\si{\\second}]')
#         axes[1].set_ylabel('PSD [\\si{\\decibel\\per\\hertz}]')
#         axes[1].set_xlabel('Frequency [\\si{\\hertz}]')
#         axes[1].grid(0)
#         axes[1].set_xlim((0,100))
#         axes[1].legend()
#         fig.subplots_adjust(left=0.110, bottom=0.125, right=0.970, top=0.960, hspace=0.340)
#         plot.show()


def  student_data(ansys, ambient=True, nonlinear=True, friction=True):
    # verification study over: zeta 0 %, 1%, 5 %, 20 %, over nonlinearity 0, 0.5, -0.5, over fric_visc_rat 0, 0.5, 1
    # plot.ion()

    omega = np.random.random() * 14 + 1
    zeta = np.random.random() * (10 - 0.1) + 0.1
    dt_fact = np.random.random() * (0.015 - 0.001) + 0.001

    if ambient:
        num_cycles = np.random.randint(300, 2000)  # ambient
        f_scale = np.random.random() * 10
        d0 = None
    else:
        num_cycles = np.random.randint(3, 20)  # free decay
        d0 = np.random.random() * 100
        f_scale = None

    if nonlinear:
        nl_ity = np.random.random() - 0.5
    else:
        nl_ity = 0

    if friction:
        fric_visc_rat = np.random.random()
    else:
        fric_visc_rat = 0

    if ambient and nonlinear and not friction:
        savefolder = str(pathlib.Path.cwd() / 'polyuq_results' / 'datasets_nonlinear_ambient') + '/'
    elif ambient and not nonlinear and not friction:
        savefolder = str(pathlib.Path.cwd() / 'polyuq_results' / 'datasets_linear_ambient') + '/'
    elif not ambient and not nonlinear and not friction:
        savefolder = str(pathlib.Path.cwd() / 'polyuq_results' / 'datasets_linear_decay') + '/'
    elif not ambient and nonlinear and not friction:
        savefolder = str(pathlib.Path.cwd() / 'polyuq_results' / 'datasets_nonlinear_decay') + '/'
    elif not ambient and not nonlinear and friction:
        savefolder = str(pathlib.Path.cwd() / 'polyuq_results' / 'datasets_friction_decay') + '/'
    elif ambient and not nonlinear and friction:
        savefolder = str(pathlib.Path.cwd() / 'polyuq_results' / 'datasets_friction_ambient') + '/'
    elif not ambient and nonlinear and friction:
        savefolder = str(pathlib.Path.cwd() / 'polyuq_results' / 'datasets_general_decay') + '/'
    elif ambient and nonlinear and friction:
        savefolder = str(pathlib.Path.cwd() / 'polyuq_results' / 'datasets_general_ambient') + '/'
    else:
        raise RuntimeError(f'This combination of inputs is not supported: {ambient}, {nonlinear}, {friction}')

#                 generate_sdof_time_hist(ansys, omega=1, zeta=zeta/100, m=1, fric_visc_rat=fric_visc_rat, nl_ity=nl_ity, d0=1, dt_fact=0.001, num_cycles=20, savefolder=None)
    generate_sdof_time_hist(ansys, omega=omega, zeta=zeta / 100, m=1, fric_visc_rat=fric_visc_rat, nl_ity=nl_ity,
                            d0=d0, f_scale=f_scale,
                            dt_fact=dt_fact, num_cycles=num_cycles, savefolder=savefolder)


def student_data_part2(jid, result_dir, omega, zeta, dt_fact, num_cycles, f_scale, d_scale, nl_ity, fric_visc_rat, snr_db, working_dir, **kwargs):
    """

    add noise SNR [-20 dB (S/R = 0.01 Signal zehn mal schwaecher als Rauschen),
                    0 dB (Signalleistung==Rauschleistung),
                   20 dB (S/R=10 Signal zehn mal staerker als Rauschen)]
    reduce sample rate : decimate by factor 6 -> h/T = [0.006 - 0.09] -> samples per cycle = [166 ... 11]  to not mix numerical accuracy effects with each other but rather merge them proportionally

    pre compute covariance up to the total length of time series, even though it is not needed

    Use Data Manager for that for each type feed in the ids and SNR

    define function, that loads, decimates, adds noise and (if ambient: computes correlation function) saves
    the actual definition file can then be updated independently

    """

    global ansys
    try:
        ansys.finish()
    except (pyansys.errors.MapdlExitedError, NameError) as e:
        logger.exception(e)
        ansys = Mechanical.start_ansys(jid=jid, working_dir=working_dir)

    ansys.clear()

    if d_scale is not None:
        if np.isnan(d_scale): d_scale = None

    if f_scale is not None:
        if np.isnan(f_scale): f_scale = None

    array, k, c, d_max, fsl, deltat = generate_sdof_time_hist(ansys,
                                                   omega, zeta / 100, 1, fric_visc_rat, nl_ity ,  # structural parameters
                                                   d_scale, f_scale,  # loading parameters
                                                   dt_fact=dt_fact, num_cycles=num_cycles,  # signal parameters
                                                   savefolder=result_dir, working_dir=working_dir, jid=jid,  # function parameters
                                                   data_hadidi=True, **kwargs)
    plot.plot(array[:,0],array[:,1])
    plot.show()
    snr = 10 ** (snr_db / 10)

    power = np.mean(array[:, 1] ** 2)

    # decimate
    array = array[1::6, :] # why start at 1 here?
    N = array.shape[0]
    # add noise
    noise_power = power / snr
    noise = np.random.normal(0, np.sqrt(noise_power), N)
    power_noise = np.mean(noise ** 2)

    snr_actual = power / power_noise
    snr_actual_db = 10 * np.log10(snr_actual)

    array[:, 1] += noise
    if d_scale is None and f_scale is not None:
        cov = np.zeros((N, 2))
        for i in range(N):
            cov[i, 1] = array[:N - i, 1].dot(array[i:, 1]) / (N - i)
        cov[:, 0] = array[:, 0]

        mdict = {'disp':array, 'corr':cov}
    else:
        mdict = {'disp':array}
    file = os.path.join(result_dir, jid + '.mat')
    scipy.io.savemat(file, mdict, appendmat=True, format='5', long_field_names=False, do_compression=True, oned_as='row')

    dt = (array[-1, 0] - array[0, 0]) / (array.shape[0] - 1)

    return k, c, d_max, fsl, power, power_noise, dt


def identify_student():
    source_num = 0

    source_folder = [str(pathlib.Path.cwd() / 'polyuq_results' / 'datasets') + '/',
                     str(pathlib.Path.cwd() / 'polyuq_results' / 'datasets_ambient') + '/',
                     str(pathlib.Path.cwd() / 'polyuq_results' / 'datasets_nonlinear_decay') + '/',
                     str(pathlib.Path.cwd() / 'polyuq_results' / 'datasets_nonlinear_ambient') + '/'
                     ][source_num]

    snr = 100
    decimate = 0
    single = False

    id_res = []
    id_inp = []
    fig, axes = plot.subplots(nrows=5, ncols=8)
    axes = axes.flatten()
    # with open(f'{source_folder}description_new.txt','tr') as descr:
    with open(f'{source_folder}description.txt', 'tr') as descr:  # , open(f'{source_folder}description_new.txt','tw') as descr_new:
        descr.readline()
        for i, line in enumerate(descr):
            print(line)
            m = 1
            if source_num == 0:
                _id, omega, zeta, R, deltat = [float(s.strip()) if j > 0 else s.strip() for j, s in enumerate(line.split(',')) ]
                k = omega ** 2 * m
                d = zeta * (2 * np.sqrt(k * m))
                # descr_new.write("{},\t{:1.3f},\t{:1.4f},\t{:1.5f}\n".format(_id,k,d,R,deltat))
            elif source_num == 1:
                _id, omega, zeta, deltat = [float(s.strip()) if j > 0 else s.strip() for j, s in enumerate(line.split(',')) ]
                k = omega ** 2 * m
                d = zeta * (2 * np.sqrt(k * m))
                # descr_new.write("{},\t{:1.3f},\t{:1.4f},\t{:1.5f}\n".format(_id,k,d,deltat))
            elif source_num == 2:
                _id, omega, zeta, R, deltat, nl_ity = [float(s.strip()) if j > 0 else s.strip() for j, s in enumerate(line.split(',')) ]
                k = omega ** 2 * m
                d = zeta * (2 * np.sqrt(k * m))
                # descr_new.write("{},\t{:1.3f},\t{:1.4f},\t{:1.5f},\t{:1.5f},\t{:1.3f}\n".format(_id,k,d,R,deltat,nl_ity))
            elif source_num == 3:
                _id, omega, zeta, d_max, deltat, nl_ity = [float(s.strip()) if j > 0 else s.strip() for j, s in enumerate(line.split(',')) ]
                k = omega ** 2 * m
                d = zeta * (2 * np.sqrt(k * m))
                # descr_new.write("{},\t{:1.3f},\t{:1.4f},\t{:1.5f},\t{:1.5f},\t{:1.3f}\n".format(_id,k,d,d_max,deltat,nl_ity))

        # continue

            # print(f'zeta: {zeta*100} \%, omega: {omega} Hz')
            ty = np.loadtxt(f'{source_folder}{_id}.csv')
            # print(deltat, ty[2,0]-ty[1,0])
            if snr:
                # SNR=u_eff,sig^2/u_eff,noise^2 (wikipedia: Signal-Rausch-Verhältnis: Rauschspannungsverhältnis)
                # u_eff,noise^2 = u_eff,sig^2/SNR
                ty[:, 1] += np.random.randn(ty.shape[0]) * np.sqrt(ty[:, 1].var() * snr)  # variance equals rms**2 here because it is a zero-mean process

            if decimate:
                ty = ty[::decimate, :]
            if 'ambient' in source_folder:
                if source_num == 3:
                    d = max(np.abs(ty[:, 1]))
                    k *= (1 + nl_ity * ((d / d_max) ** 2 - 1))
                ydata = scipy.signal.correlate(ty[:, 1], ty[:, 1], mode='full', method='direct')
                ydata = ydata[ydata.shape[0] // 2:, ][:1000]
                xdata = ty[:1000, 0]
            else:
                ydata = ty[:, 1]
                xdata = ty[:, 0]
            # popt, pcov = scipy.optimize.curve_fit(f=free_decay, xdata = ty[:1000,0], ydata=corr[corr.shape[0]//2:,][:1000], p0=[0.5,0.05,2*np.pi,0])#,  bounds=[(-1,0,0,0),(1,1,np.pi/dt,2*np.pi)])
            try:
                popt, pcov = scipy.optimize.curve_fit(f=free_decay, xdata=xdata, ydata=ydata, p0=[0.5, 0.05, 2 * np.pi, 0])  # ,  bounds=[(-1,0,0,0),(1,1,np.pi/deltat,2*np.pi)])
            except Exception as e:
                print('ID failed', e)
                popt = [0, 0, 0, 0]
                pcov = [np.inf, np.inf, np.inf, np.inf]

            perr = np.sqrt(np.diag(pcov))
            # print(' zeta: {:1.3f}+-{:1.3f} \%, omega: {:1.4f}+-{:1.3f} Hz\n\n'.format( popt[1]*100, perr[1]*100, popt[2], perr[2]))
            id_res.append((popt[2], popt[1], xdata[2] - xdata[1]))

            t_synth = np.linspace(0, xdata[-1], 1000)
            # synth=free_decay(ty[:1000,0], *popt)
            synth = free_decay(t_synth, *popt)

            zeta = zeta
            omega = np.sqrt(k / m) * np.sqrt(1 - zeta ** 2)

            deltat = deltat
            id_inp.append((omega, zeta, deltat))
            print(omega, zeta, deltat)

            if single:
                plot.figure()
                plot.plot(xdata, ydata, ls='none', marker=',')
                plot.plot(t_synth, synth)
                plot.show()
                continue
            # axes[i].plot(ty[:1000,0],corr[corr.shape[0]//2:,][:1000], ls='none', marker='+')
            axes[i].plot(xdata, ydata, ls='none', marker=',')
            axes[i].plot(t_synth, synth)
#             if len(id_res)>2:
#                 break
            # if i >4:break
    id_res = np.array(id_res)
    id_inp = np.array(id_inp)
    labels = ['Omega', 'Zeta', 'Delta_t']
    for i in range(2):
        plot.figure()
        plot.plot(id_inp[:, i], id_res[:, i], ls='none', marker='.')
        plot.xlabel(labels[i] + '_simulated')
        plot.ylabel(labels[i] + '_identified')

    plot.show()


def generate_mdof_time_hist(ansys, num_nodes=None, damping=None, nl_stiff=None, sl_forces=None, freq_scale=1, num_modes=None,  # structural parameters
                            d0=None, f_scale=None,  # loading parameters
                            deltat=None, dt_fact=None, timesteps=None, num_cycles=None, num_meas_nodes=None, meas_nodes=None,  # signal parameters
                            savefolder=None, working_dir=tempfile.gettempdir(), jid=None,  # function parameters
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
        d0 = initial displacement
                'static' : displacement obtained from static analysis with unit top displacement
                'modes' : arithmetic mean of all mode shapes
                number > 0, modenumber of which


    TODO:
    - spring/mass elements and assembly
        - validation in static and modal analyses ✓
        - number of nodes needed for consideration of a certain number of modes (convergence study and plot) ✓
    -  viscous damping elements and assembly
        - validation with global (rayleigh damping) in stiffness matrices and modal analysis  ✓
        - automatic rayleigh parameters with analytical solution before ✓
        - comparison of global and element damping in terms of modal damping (numerical) vs number of nodes for x modes and y target damping ratios ✓

    - free decay
        - transient analysis parameters (timestep/dt-omega-ratio, duration/num_cycles, numerical damping if not all modes are required?)
            - set parameters for e.g. ten modes and check the response
        - identification of modal parameters from pure mode free decay and mixed free decay (filtered?) -> verification
    - ambient
        - uncorrelated random processes / correlated random processes
        - identification of modal parameters using OMA -> verification

    MAYBE STOP HERE
    - nonlinear-equivalent
        - stiffness (as in SDOF systems), estimation of d_max
        - friction (equivalent per node and mode, averaging?)
        - structural damping (ANSYS routines, IWAN)
    '''
    try:
        ansys.finish()
    except pyansys.errors.MapdlExitedError as e:
        print(e)
        ansys = Mechanical.start_ansys()
    ansys.clear()
    
    mech = Mechanical(ansys=ansys, jobname=jid, wdir=working_dir)
    mech.example_rod(num_nodes, damping, nl_stiff, sl_forces, freq_scale, num_modes, num_meas_nodes, meas_nodes)

    if kwargs.pop('just_build', False):
        return mech
    
    if kwargs.pop('convergence_study', False):
        disp = mech.static(uz=np.array([[num_nodes, 1]]))
        frequencies, damping, modeshapes = mech.modal(damped=False, num_modes=num_modes)
        return [coords[3] for coords in mech.nodes_coordinates], disp[:, 2], frequencies, modeshapes[:, 2, :].real

    if kwargs.pop('damping_study', False):
        frequencies, damping, modeshapes = mech.modal(damped=True, num_modes=num_modes)
        return frequencies, damping, mech.alpha, mech.beta

    if kwargs.pop('evd_study', False):
        frequencies, damping, modeshapes = mech.modal(damped=True, num_modes=num_modes)
        return mech, frequencies, damping, modeshapes, mech.alpha, mech.beta

    if kwargs.pop('verify', False):

        if d0 is not None:
            t_vals, resp_hist = mech.free_decay(d0, 1, deltat, dt_fact, timesteps, num_cycles, **kwargs)
        elif f_scale is not None:
            t_vals, resp_hist, inp_hist = mech.ambient(f_scale, deltat, dt_fact, timesteps, num_cycles, **kwargs)

#         resp_hist=resp_hist[0] #accelerations

        frequencies, damping, modeshapes = mech.numerical_response_parameters(compensate=False)
        frequencies_n, damping_n, modeshapes_n = mech.numerical_response_parameters()

        timesteps = mech.timesteps
        mode = 0

        mech.ansys.save(f'{mech.jobname}.db')
        mech.ansys.exit()
        meas_nodes = mech.meas_nodes

        if False:
            from pyOMA.core import PreProcessingTools
            from pyOMA.core import PlotMSH

            mech.export_geometry(f'{tempfile.gettempdir()}/{jid}/')
            geometry = PreProcessingTools.GeometryProcessor.load_geometry(f'{tempfile.gettempdir()}/{jid}/grid.txt', f'{tempfile.gettempdir()}/{jid}/lines.txt')
            prep_data = PreProcessingTools.PreProcessSignals(resp_hist, 1 / (t_vals[1] - t_vals[0]))
            chan_dofs = prep_data.load_chan_dofs(f'{tempfile.gettempdir()}/{jid}/chan_dofs.txt')
            prep_data.add_chan_dofs(chan_dofs)

            mode_shape_plot = PlotMSH.ModeShapePlot(geometry, prep_data=prep_data)
            mode_shape_plot.draw_nodes()
            mode_shape_plot.draw_lines()
            mode_shape_plot.draw_master_slaves()
            mode_shape_plot.draw_chan_dofs()
            PlotMSH.start_msh_gui(mode_shape_plot)
        elif True:

            nrows = int(np.ceil(np.sqrt(len(meas_nodes))))
            ncols = int(np.ceil(len(meas_nodes) / nrows))
            fig, axes = plot.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False)
            fig2, axes2 = plot.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False)

            for channel, (ax, ax2, meas_node) in enumerate(zip(axes.flat, axes2.flat, meas_nodes)):
                try:
                    popt, _ = response_frequency(t_vals, resp_hist[0][:, channel], p0=[1, damping[mode], frequencies[mode] * 2 * np.pi, 0])
                    resid_id = np.sqrt(np.mean((resp_hist[0][:, -1] - free_decay(t_vals, *popt)) ** 2))
                    print(resid_id)
                except Exception as e:
                    print(e)
                    popt = [1, 1, 0, 0]
                frequencies_id = popt[2] / 2 / np.pi
                damping_id = popt[1]
                time_fine = np.arange(deltat, deltat * timesteps, 0.0001)

                for i in range(1):
                    ax.plot(t_vals, resp_hist[i][:, channel], marker='x')
                    ax2.psd(resp_hist[i][:, channel], NFFT=25600, Fs=1 / (t_vals[1] - t_vals[0]), label=str(meas_node))

                ax.plot(time_fine, free_decay(time_fine, 1, damping[mode], frequencies[mode] * 2 * np.pi, 0), label='ex')
                ax.plot(time_fine, free_decay(time_fine, 1, damping_n, frequencies_n * 2 * np.pi, 0), label='est')
                ax.plot(time_fine, free_decay(time_fine, *popt), label='_id')

                ax.legend(loc=1)
                ax2.legend(loc=1)
            plot.show(block=True)
            print(frequencies_id, damping_id)
        return

    if kwargs.pop('perf_bench', False):
        if np.isnan(d0): d0 = None
        else: d0 = int(d0)
        if np.isnan(f_scale): f_scale = None

        if d0 is not None:
            t_vals, resp_hist = mech.free_decay(d0, 1, deltat, dt_fact, timesteps, num_cycles, **kwargs)
        elif f_scale is not None:
            t_vals, resp_hist, inp_hist = mech.ambient(f_scale, deltat, dt_fact, timesteps, num_cycles, **kwargs)

        dts = mech.transient_runtimes
        return np.mean(dts), np.min(dts), np.max(dts), np.std(dts)

    if kwargs.pop('verify_compens', False):
        '''
        verification of numerical compensation:
        Arguments:
        nodes_coordinates    -    a linear SDOF system should be generated with
        k_vals, m_vals       -    natural circular frequency should ideally be omega=1 (disp, vel and acc are all unity)
        damping              -    viscous damping can be applied according to the usual procedures
        meth, parameter_set  -    all available time integration algorithms can be used with different parameters

        returns a range of 20 timestepsizes where numerical errors should be present
        for each timestepsize:
            free_decay with n_cycles and curve fit for identification
        return exact, compensated and identified frequencies and damping ratios

        final analyses to be done outside this function

        '''
        freq_ex, damp_ex, _ = mech.modal(damped=True)
        freq_ex = freq_ex[0]
        damp_ex = damp_ex[0]

        mech.free_decay(d0=0, dscale=1, deltat=0.0001, num_cycles=num_cycles, timesteps=2, **kwargs)
        frequencies_n, damping_n, _ = mech.numerical_response_parameters()

        frequencies = np.ones((20,)) * freq_ex
        damping = np.ones((20,)) * damp_ex

        frequencies_id = np.zeros((20,))
        damping_id = np.zeros((20,))
        frequencies_n = np.zeros((20,))
        damping_n = np.zeros((20,))
        resid_id = np.zeros((20,))

        _, axes = plot.subplots(4, 5, sharex=True, sharey=True)

        deltats = np.logspace(-3, -2, 20) * 3 / freq_ex

        for i, (ax, deltat) in enumerate(zip(axes.flat, deltats)):

            t_vals, resp_hist = mech.free_decay(d0=0, dscale=1, deltat=deltat, num_cycles=num_cycles, timesteps=timesteps, **kwargs)
            freq, damp, modeshape = mech.numerical_response_parameters()

            ax.plot(t_vals, resp_hist[0][:, -1], ls='none', marker='x')
            ax.plot(t_vals, resp_hist[1][:, -1], ls='none', marker='x')
            ax.plot(t_vals, resp_hist[2][:, -1], ls='none', marker='x')
            frequencies_n[i] = freq[0]
            damping_n[i] = damp[0]

            y0 = np.real((modeshape[:] / modeshape[-1]))[0]

            t_vals_fine = np.linspace(0, t_vals[-1], 2048)
            ax.plot(t_vals_fine, free_decay(t_vals_fine, y0[0], damp[0], freq[0] * 2 * np.pi, 0), label='comp')
            try:

                popt, _ = response_frequency(t_vals, resp_hist[0][:, -1], p0=[y0[0], damp[0], freq[0] * 2 * np.pi, 0])  # R, zeta, omega_d, phi=0
                resid_id[i] = np.sqrt(np.mean((resp_hist[0][:, -1] - free_decay(t_vals, *popt)) ** 2))
                print(resid_id[i])
                frequencies_id[i] = popt[2] / 2 / np.pi
                damping_id[i] = popt[1]
            except Exception as e:
                popt = [0, 0, 0, 0]
                print(e)
            ax.plot(t_vals_fine, free_decay(t_vals_fine, *popt), label=f'_id f={popt[2]/2/np.pi:.5f} \t d={popt[1]:.5f}')
            ax.legend()

        ax.set_ylim((-4, 4))

        return frequencies, frequencies_n, frequencies_id, damping, damping_n, damping_id, resid_id

    if kwargs.pop('verify_impulse', False):
        dt_fact, deltat, num_cycles, timesteps, _ = mech.signal_parameters(dt_fact, deltat, num_cycles, timesteps)

        ref_nodes = [2, 3]
        imp_durs = [20 * deltat, 40 * deltat]

        imp_times = [1 * deltat, 51 * deltat]
        imp_forces = [10000, 10000]

#         ref_nodes = [2]
#         imp_durs = [ 10*deltat]
#
#         imp_times = [ 20*deltat]
#         imp_forces=[ 10000]

#         for form in ['step','rect','sine']:
#             t_vals, resp_hist, inp_hist= mech.impulse_response([ref_nodes,imp_forces,imp_times,imp_durs], form=form, mode='combined', deltat=deltat, dt_fact=dt_fact, timesteps=timesteps, num_cycles=num_cycles,**kwargs)
        ref_nodes = range(2, num_nodes + 1)
        imp_forces = [196, 56, 29, 19, 14, 11, 9, 8, 8]
        imp_times = [deltat for i in range(num_nodes)]
        imp_durs = [30 * deltat for i in range(num_nodes)]
        out_quant = ['d']
        t_vals, IRF_matrix, F_matrix, ener_mat, amp_mat = mech.impulse_response([ref_nodes, imp_forces, imp_times, imp_durs], form='step', mode='matrix', deltat=deltat, dt_fact=dt_fact, timesteps=timesteps, num_cycles=num_cycles, out_quant=out_quant, **kwargs)

        num_ref_nodes = len(ref_nodes)
        meas_nodes = mech.meas_nodes
        num_meas_nodes = len(meas_nodes)
        jid = ansys.jobname

        mech.export_geometry(f'{tempfile.gettempdir()}/{jid}/')
        from pyOMA.core import PreProcessingTools
        geometry = PreProcessingTools.GeometryProcessor.load_geometry(f'{tempfile.gettempdir()}/{jid}/grid.txt', f'{tempfile.gettempdir()}/{jid}/lines.txt')

        if out_quant[0] == 'd':
            disp_channels = list(range(num_meas_nodes))
            velo_channels = None
            accel_channels = None
        elif out_quant[0] == 'a':
            accel_channels = list(range(num_meas_nodes))
            velo_channels = None
            disp_channels = None
        elif out_quant[0] == 'v':
            velo_channels = list(range(num_meas_nodes))
            disp_channels = None
            accel_channels = None
        channel_headers = meas_nodes
        ref_channels = [i for i, node in enumerate(meas_nodes) if node in ref_nodes]
        dummy_meas = np.zeros((timesteps, num_meas_nodes))

        print(accel_channels, velo_channels, disp_channels, ref_channels,)

        prep_data = PreProcessingTools.PreProcessSignals(dummy_meas, 1 / deltat, ref_channels=ref_channels,
                                                      accel_channels=accel_channels,
                                                      velo_channels=velo_channels,
                                                      disp_channels=disp_channels,
                                                      channel_headers=channel_headers)
        chan_dofs = prep_data.load_chan_dofs(f'{tempfile.gettempdir()}/{jid}/chan_dofs.txt')
        prep_data.add_chan_dofs(chan_dofs)
        prep_data.save_state(f'{tempfile.gettempdir()}/{jid}/prep_data.npz')
        np.savez(f'{tempfile.gettempdir()}/{jid}/IRF_data.npz', t_vals=t_vals, IRF_matrix=IRF_matrix, F_matrix=F_matrix, ener_mat=ener_mat, amp_mat=amp_mat)

        # IRF Matrix
        # geometry
        # ref_channels, accel_channels, velo_channels, disp_channels

        return
    
    

    if d0 is not None:
        t_vals, resp_hist = mech.free_decay(d0, 1, deltat, dt_fact, timesteps, num_cycles, **kwargs)
    elif f_scale is not None:
        t_vals, resp_hist, inp_hist = mech.ambient(f_scale, deltat, dt_fact, timesteps, num_cycles, **kwargs)
    resp_hist = resp_hist[2]  # accelerations

    frequencies, damping, modeshapes = mech.numerical_response_parameters()

    for file in glob.glob(f'{working_dir}{jid}.*'):
        # print(f'removing old files: {file}')
        os.remove(file)

    # for i, node in enumerate(mech.meas_nodes):
        # ydata = resp_hist[:, i]
        # plot.plot(t_vals, ydata, label=f'{node}')
    # plot.legend()
    # plot.show()

    if savefolder is not None:
        mech.save(savefolder)
        mech.export_geometry(savefolder)
    
    return mech

# def mean_confidence_interval(data, confidence=0.95):  # https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    # a = 1.0 * np.array(data)
    # n = len(a)
    # m, se = np.mean(a), scipy.stats.sem(a)
    # h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    # return m, h


def convergence_mdof_undamped(ansys):
    mshs = []
    freqs = []
    coords = []
    omega_true = [(2 * k - 1) / 2 * np.pi / 200 * np.sqrt(2.1e11 / 7850) for k in range(1, 11)]
    num_modes = 10

    x = []
    if False:
        for num_nodes in range(2, 21 ** 2):
            print(num_nodes)
            coord, disp, frq, msh = generate_mdof_time_hist(ansys, num_nodes, num_modes=num_modes, convergence_study=True)
            freqs.append(frq)
            mshs.append(msh)
            coords.append(coord)
            # x.append(num_nodes)

            # freq_errors.append(np.mean(np.abs(frq-frq_true[:len(frq)])/frq_true[:len(frq)]))

            # msh_true = np.array([np.sin(np.array(coords)*(2*i-1)/2*np.pi/-200) for i in range(1,len(frq)+1)]).T
            # msh_errors.append(np.abs(np.abs(msh)-np.abs(msh_true)).flatten())

        np.savez(str(pathlib.Path.cwd() / 'polyuq_results' / 'example_convergence_msh.npz'), msh_errors=mshs, freq_errors=freqs, coords=coords)
        return
    else:
        arr = np.load(str(pathlib.Path.cwd() / 'polyuq_results' / 'example_convergence_msh.npz'), allow_pickle=True)
        mshs = arr['msh_errors']
        freqs = arr['freq_errors']
        coords = arr['coords']
        x = list(range(2, 21 ** 2))
    with matplotlib.rc_context(rc=print_context_dict):
        fig, (ax2, ax1) = plot.subplots(2, 1, sharex=True, squeeze=True)
        for mode, color, marker in zip([8, 4, 1], ['lightgrey', 'darkgrey', 'black'], ['+', 'x', '.']):
            freq_errors = []
            msh_errors = []
            this_x = []
            for j, num_nodes in enumerate(x):
                if num_nodes < 9: continue
                if j % 5: continue  # every fifth point only
                this_x.append(num_nodes)
                # print(num_nodes)
                omega = freqs[j]
                coord = coords[j]
                msh = mshs[j]
                freq_errors.append(np.mean(np.abs(omega[mode - 1] * 2 * np.pi - omega_true[mode - 1]) / omega_true[mode - 1]) * 100)
                # print(omega[mode-1]*2*np.pi, omega_true[mode-1])
                msh_true = np.sin(np.array(coord) * (2 * mode - 1) / 2 * np.pi / -200).T
                # print(msh.shape, msh_true.shape)
                msh_errors.append(np.mean(np.abs(np.abs(msh[:, mode - 1]) - np.abs(msh_true))))

            ax1.plot(this_x, freq_errors, ls='none', marker=marker, color=color, label='$\\epsilon(\\omega_' + str(mode) + ')$')
            ax2.plot(this_x, [np.mean(err) for err in msh_errors], ls='none', marker=marker, color=color, label='$\\epsilon(\\bm{\\phi}_' + str(mode) + ')$')

        # ax1.set_xlabel('\#nodes')
        ax1.legend()
        ax2.legend()
        ax1.set_xlabel('$n$')
        ax1.set_ylabel('$\\nicefrac{\\abs{\\omega_\\text{num}-\\omega_\\text{ex}}}{\\omega_\\text{ex}}$ [\\si{\\percent}]')
        ax2.set_ylabel('$\\overline{\\abs{\\bm{\\phi}_\\text{num}-\\bm{\\phi}_\\text{ex}}}$ [\\si{\\metre}]')
        ax1.set_xlim((0, 400))
        ax2.set_xlim((0, 400))
        ax1.set_ylim((0, 0.05 * 100))
        # ax1.set_yscale('log')
        ax2.set_ylim((0, 1e-13))

        plot.subplots_adjust(left=0.11, right=0.970, top=0.95, bottom=0.115, hspace=0.1)
        plot.savefig(str(pathlib.Path('figures') / 'example_convergence_mdof.pdf'))
        plot.savefig(str(pathlib.Path('figures') / 'example_convergence_mdof.png'))
        plot.show()


def rayleigh_example(ansys):
    '''
    Study constant modal damping over frequency (rayleigh, stiffness-, mass-proportional):
    - Analytical
    - Numerical global
    - Numerical local


    '''
    plot.figure()
    ax1 = plot.subplot()
    plot.figure()
    ax2 = plot.subplot()
    for color, zeta in zip(['lightgrey', 'black'], [0.025, 0.1]):

        num_nodes = 80

        freqs, damps, alpha, beta = generate_mdof_time_hist(ansys, num_nodes, damping=(zeta, zeta), num_modes=8, damping_study=True)
        # print(freqs)
        fs = np.linspace(0, 100, 1000)
        omegas = fs * 2 * np.pi
        zetas_r = 1 / 2 * (alpha / omegas + beta * omegas) * 100
        zetas_m = 1 / 2 * (alpha / omegas + 0 * omegas) * 100
        zetas_k = 1 / 2 * (0 / omegas + beta * omegas) * 100

        if zeta == 0.1:
            ax1.plot(fs, zetas_r, color=color, ls='solid', label=f'Rayleigh')
            ax1.plot(fs, zetas_m, color=color, ls='dashdot', label=f'Stiffness prop.')
            ax1.plot(fs, zetas_k, color=color, ls='dotted', label=f'Mass prop.')
        else:
            ax1.plot(fs, zetas_r, color=color, ls='solid')
            ax1.plot(fs, zetas_m, color=color, ls='dashdot')
            ax1.plot(fs, zetas_k, color=color, ls='dotted')
        # ax1.plot([freqs[0],freqs[-1]],[200*zeta,100*zeta], color=color, ls='dashed',lw=0.75)#target damping

        omegas = freqs * 2 * np.pi
        exact_damps = 1 / 2 * (alpha / omegas + beta * omegas)
        delta_damps = np.abs(damps - exact_damps) / exact_damps
        if zeta == 0.1:
            ax1.plot(freqs, damps * 100, ls='none', marker='x', color=color, label='Rayleigh (numerical)')
            ax2.plot(freqs, delta_damps, ls='none', marker='+', color=color, label='Rayleigh (numerical)')
        else:
            ax1.plot(freqs, damps * 100, ls='none', marker='x', color=color)
            ax2.plot(freqs, delta_damps, ls='none', marker='+', color=color)

        # stiffness proportional
        freqs, damps, _, _ = generate_mdof_time_hist(ansys, num_nodes, damping=(0, beta, True), num_modes=8, damping_study=True)
        omegas = freqs * 2 * np.pi
        exact_damps = 1 / 2 * (0 / omegas + beta * omegas)
        delta_damps = np.abs(damps - exact_damps) / exact_damps
        if zeta == 0.1:
            ax1.plot(freqs, damps * 100, ls='none', marker='+', color=color, label='Stiffness prop. (numerical)')
            ax2.plot(freqs, delta_damps, ls='none', marker='+', color=color, label='Stiffness prop. (numerical)')
        else:
            ax1.plot(freqs, damps * 100, ls='none', marker='+', color=color)
            ax2.plot(freqs, delta_damps, ls='none', marker='+', color=color)

        # mass proportional
        freqs, damps, _, _ = generate_mdof_time_hist(ansys, num_nodes, damping=(alpha, 0, True), num_modes=8, damping_study=True)
        omegas = freqs * 2 * np.pi
        exact_damps = 1 / 2 * (alpha / omegas + 0 * omegas)
        delta_damps = np.abs(damps - exact_damps) / exact_damps
        if zeta == 0.1:
            ax1.plot(freqs, damps * 100, ls='none', marker='1', color=color, label='Mass prop. (numerical)')
            ax2.plot(freqs, delta_damps, ls='none', marker='1', color=color, label='Mass prop. (numerical)')
        else:
            ax1.plot(freqs, damps * 100, ls='none', marker='1', color=color)
            ax2.plot(freqs, delta_damps, ls='none', marker='1', color=color)

        # Elementwise stiffness proportional
        freqs, damps, _, _ = generate_mdof_time_hist(ansys, num_nodes, damping=(0, beta, False), num_modes=8, damping_study=True)
        omegas = freqs * 2 * np.pi
        exact_damps = 1 / 2 * (0 / omegas + beta * omegas)
        delta_damps = np.abs(damps - exact_damps) / exact_damps
        if zeta == 0.1:
            ax1.plot(freqs, damps * 100, ls='none', marker='3', color=color, label='Elementwise (numerical)')
            ax2.plot(freqs, delta_damps, ls='none', marker='3', color=color, label='Elementwise (numerical)')
        else:
            ax1.plot(freqs, damps * 100, ls='none', color=color, marker='3')
            ax2.plot(freqs, delta_damps, ls='none', color=color, marker='3')

    ax1.legend()
    ax1.set_xlim((0, 100))
    ax1.set_ylim((0, 200 * zeta))
    ax1.set_xlabel('Natural frequency $f \\, [\\si{\\hertz}]$')
    ax1.set_ylabel('Damping ratio  $\\zeta \\, [\\si{\\percent}_\\text{crit}]$')

    ax2.legend()
    ax2.set_xlim((0, 100))

    plot.sca(ax1)
    plot.subplots_adjust(top=0.95, bottom=0.125, left=0.11, right=0.960)
    # plot.savefig(str(pathlib.Path('figures') / 'example_rayleigh_damping.pdf'))
    # plot.savefig(str(pathlib.Path('figures') / 'example_rayleigh_damping.png'))

    plot.show()


def eigenvalue_decomposition_example(ansys):

    omega_d = np.pi * 2
    freq_scale = omega_d * np.sqrt(200 ** 2 * 7850 / 2 / 2.1e11)
    freq_scale=1
    np.set_printoptions(precision=2, linewidth=250)
    num_nodes = 11
    num_modes = num_nodes - 1
    import scipy.linalg
    mech, freqs, damps, mshs, alpha, beta = generate_mdof_time_hist(ansys, freq_scale=freq_scale, num_nodes=num_nodes, damping=None, num_modes=num_modes, evd_study=True)
    print('ANSYS undamped modal parameters')
    print(freqs, '\n', damps)

    k, m, c = mech.export_ans_mats()
    print(k,m,c)
    full_path = os.path.join(ansys.directory, ansys.jobname + '.full')
    full = pyansys.read_binary(full_path)
    # TODO: Check, that Nodes and DOFS are in the same order in modeshapes and k,m
    dof_ref, k_, m_ = full.load_km(as_sparse=False, sort=False)
    k_ += np.triu(k_, 1).T
    m_ += np.triu(m_, 1).T
#     print(dof_ref)
    for mode in range(num_modes):
        msh_f = mshs[1:, 2, mode].flatten()

        kappa = msh_f.T.dot(k).dot(msh_f)
        mu = msh_f.T.dot(m).dot(msh_f)
        
        print(np.sqrt(kappa / mu) / 2 / np.pi)

        msh_f = mshs[:, :, mode].flatten()

        kappa = msh_f.T.dot(k_).dot(msh_f)
        mu = msh_f.T.dot(m_).dot(msh_f)
        print(np.sqrt(kappa / mu) / 2 / np.pi)

    # print('ANSYS undamped system matrices')
    # print(k,'\n',m,'\n',c)
    w, vr = scipy.linalg.eig(k, m)
    print('SCIPY modal frequencies')
    print(np.sqrt(w) / 2 / np.pi)

    mech, freqs, damps, mshs, alpha, beta = generate_mdof_time_hist(ansys, freq_scale=freq_scale, num_nodes=num_nodes, damping=0.01, num_modes=num_modes, evd_study=True)
    print(alpha, beta)
    print('ANSYS damped modal parameters')
    print(freqs, '\n', damps)
    a0, a2, a1 = mech.export_ans_mats()
    
    k, m, c = mech.export_ans_mats()
    for mode in range(num_modes):
        msh_f = mshs[1:, 2, mode].flatten()

        kappa = msh_f.T.dot(k).dot(msh_f)
        mu = msh_f.T.dot(m).dot(msh_f)
        eta = msh_f.T.dot(c).dot(msh_f)
        print(mode, kappa.real, mu.real, eta.real)
    # a1*=-1
    print('ANSYS damped system matrices')
    print(a0, '\n', a2, '\n', a1)
    zero = np.zeros_like(a0)
    factor = 1e8
    eye = factor * np.eye(*a0.shape)
    A = np.hstack((np.vstack((zero, -eye)), np.vstack((a0, a1))))  # golub p 415
    B = factor * np.eye(*A.shape)
    B[-(a2.shape[0]):, -(a2.shape[1]):] = a2
    print('Linearized system matrices')
    print(A, '\n', B)
    w, vr = scipy.linalg.eig(A, -B)
    print('SCIPY eigenvalues and eigenvectors')
    print(w, '\n', vr)
    print('SCIPY modal parameters')
    omega0 = np.abs(w)  # brinckers p 102
    zeta = -np.real(w) / omega0
    omegad = np.imag(w)
    phi = vr[num_modes:, ::2]
    modmass = np.diag((phi.T).dot(a2).dot(phi))
    phi /= np.sqrt(modmass)
    print(type(phi))
    print(omega0 / 2 / np.pi, '\n', omegad / 2 / np.pi, '\n', zeta)
    for i in range(num_modes):
        color = plot.plot(np.hstack(([0], phi[:, i].real)), label=omega0[i * 2] / 2 / np.pi)[0].get_color()
        plot.plot(np.hstack((mshs[:, 2, i].real)), color=color, ls='none', marker='x')
        # print([f"{a:.3e}" for a in np.real(phi[:, i])])
        
        kappa = phi[:, i].T.dot(k).dot(phi[:, i])
        mu = phi[:, i].T.dot(m).dot(phi[:, i])
        eta = phi[:, i].T.dot(c).dot(phi[:, i])
        print(mode, omega0[i * 2] / 2 / np.pi, kappa.real, mu.real, eta.real)
    plot.legend()
    plot.show()



# def start_ansys(working_dir=None, jid=None,):
    #
    # # global ansys
    # # try:
        # # ansys.finish()
    # # except (pyansys.errors.MapdlExitedError, NameError) as e:
        # # logger.warning(repr(e))
    # if working_dir is None:
        # working_dir = os.getcwd()
    # os.chdir(working_dir)
    # now = time.time()
    # if jid is None:
        # jid = 'file'
    # ansys = pyansys.launch_mapdl(
        # exec_file=os.environ.get('ANSYS_EXEC_FILE'),
        # run_location=working_dir, override=True, loglevel='WARNING',
        # nproc=1, log_apdl='w',
        # log_broadcast=False, jobname=jid,
        # mode='console', additional_switches='-smp')
        #
    # logger.debug(f'Took {time.time()-now} s to start up ANSYS.')
    #
    # ansys.clear()
    #
    # return ansys


def model_performance():
    from polyuq.data_manager import DataManager

    if False:
        data_manager = DataManager(title='model_perf2', working_dir=tempfile.gettempdir())

        num_nodes = np.array([5, 10, 15, 25, 35, 45, 60, 75, 90, 110, 130, 150, 175, 200, 225, 250])
        rat_meas_nodes = np.random.random(size=num_nodes.size)
        chunksize = np.array([500, 1000, 2000, 4000])
        NUM_NODES, RAT_MEAS_NODES, CHUNKSIZE = [array.flatten() for array in np.meshgrid(num_nodes, rat_meas_nodes, chunksize, indexing='ij')]

        D0 = np.zeros((NUM_NODES.size), dtype=float)
        D0[::3] = 1
        F_SCALE = np.logical_not(D0) * 100000000.0
        F_SCALE[F_SCALE == 0] = np.nan
        D0[D0 == 0] = np.nan

        NUM_MEAS_NODES = np.floor(NUM_NODES * RAT_MEAS_NODES)
        NUM_MEAS_NODES[NUM_MEAS_NODES < 2] = 2

        arrays = [NUM_NODES, D0, F_SCALE, NUM_MEAS_NODES, CHUNKSIZE]
        names = ['num_nodes', 'd0', 'f_scale', 'num_meas_nodes', 'chunksize']

        data_manager.provide_sample_inputs(arrays, names)
        return
    elif False:
        data_manager = DataManager.from_existing(dbfile_in='model_perf2.nc', result_dir=str(pathlib.Path.cwd() / 'polyuq_results'))
        # data_manager.clear_failed(True)
        data_manager.post_process_samples()

        return
    else:
        ansys = Mechanical.start_ansys()

        data_manager = DataManager.from_existing(dbfile_in='model_perf2.nc', result_dir=str(pathlib.Path.cwd() / 'polyuq_results'))
        func_kwargs = {'ansys':ansys, 'damping':0.05, 'dt_fact':0.01, 'timesteps':4001, 'perf_bench':True}
        arg_vars = [('num_nodes', 'num_nodes'), ('d0', 'd0'), ('f_scale', 'f_scale'), ('num_meas_nodes', 'num_meas_nodes'), ('chunksize', 'chunksize')]
#         data_manager.evaluate_samples(func=generate_mdof_time_hist, arg_vars=arg_vars, ret_names=['mean', 'min', 'max', 'std'],**func_kwargs,
#                                       chwdir=False
#                                       )
        data_manager.evaluate_samples(func=generate_mdof_time_hist, arg_vars=arg_vars, ret_names=['num_meas_nodes'], **func_kwargs,
                                      chwdir=False
                                      )

        return


def verify_numerical_accuracy(ansys, m=None, d=1, r=1):
    '''
    check the effect of nsubst, autots, deltim on the accuracy
    read again ansys manual, especially with respect to alpha methods, look for further parameters
    current settings allow full compensation for newmark
    but give equal deviations for all alpha methods
    period elongations are smaller than expected for g-alpha and wbz and lager than expected for hht
    numerical damping is larger than expected for all alpha methods
    '''

    # accuracy_study(ansys) # precursor

    damp = [None, 0.1][d]

    df = 0.05
    # rho = 0.5 euqals df=1/3
    # rho= 1/3 is the last value, that ansys does not complain about
    rho = [1, (-df + 1) / (df + 1), 0.5, 1 / 3 ][r]
    parameter_sets = ['AAM', 'LAM', rho, rho, rho, rho]
    meths = ['NMK', 'NMK', 'NMK', 'HHT', 'WBZ', 'G-alpha']

    if m is not None:  # build data

        meth = meths[m]
        parameter_set = parameter_sets[m]
        omega_d = 2  # *np.pi
        freq_scale = omega_d * np.sqrt(200 ** 2 * 7850 / 2 / 2.1e11)

        if damp is not None:
            freq_scale /= np.sqrt(1 - damp ** 2)  # generate_mdof does not correct for damping, so we have to correct our frequency beforehand

#         freq_scale*=np.pi
#         frequencies, frequencies_n, frequencies_id, damping, damping_n, damping_id = generate_mdof_time_hist(ansys, damping=None, num_nodes=2, freq_scale=freq_scale,
#                                    deltat=0.01, num_cycles=800, meas_nodes=[2],
#                                    verify=True, d0=0,
#                                    meth=meth, parameter_set=parameter_set
#                                    )
#         return
#

        arrs = generate_mdof_time_hist(ansys, damping=damp, num_nodes=2, freq_scale=freq_scale,  # d_0=1, a_0=4, v_pi=2
                                       num_cycles=24, meas_nodes=[2],
                                       verify_compens=True, meth=meth, parameter_set=parameter_set)
        frequencies, frequencies_n, frequencies_id, damping, damping_n, damping_id, resid_id = arrs

        np.savez(file=str(pathlib.Path.cwd() / 'polyuq_results' / f'verify_compens_m{m}_r{r}_d{d}.npz'),
             frequencies=frequencies,
             frequencies_n=frequencies_n,
             frequencies_id=frequencies_id,
             damping=damping,
             damping_n=damping_n,
             damping_id=damping_id,
             resid_id=resid_id)
        plot.suptitle(f'{meth}')
        plot.tight_layout()
        figManager = plot.get_current_fig_manager()
        figManager.window.showMaximized()
        plot.show()
        return
    else:
        linestyles = ["solid", "solid", 'dashed', 'dashdot', 'dotted', (0, (3, 1, 1, 1, 1, 1))]
        markers = ['3', '4', '+', 'x', '1', '2']
        meths = ['AAM', 'LAM', 'NMK', 'HHT-$\\alpha$', 'WBZ-$\\alpha$', 'G-$\\alpha$']
        fig, axes = plot.subplots(2, 4, sharey=True, gridspec_kw={'width_ratios':(3, 1, 3, 1)})
#         fig2,ax2 = plot.subplots(1,1)
        for d in reversed(range(2)):
            color = ['black', 'gray'][d]
            damp = [None, 0.1][d]

            for m in range(2, 6):
                meth = meths[m]
                print(f'verify_compens_m{m}_r{r}_d{d}', os.path.getmtime(str(pathlib.Path.cwd() / 'polyuq_results' / f'verify_compens_m{m}_r{r}_d{d}.npz')))
                arrs = np.load(str(pathlib.Path.cwd() / 'polyuq_results' / f'verify_compens_m{m}_r{r}_d{d}.npz'))
                frequencies, frequencies_n, frequencies_id, damping, damping_n, damping_id, resid_id = [value[:20] for value in arrs.values()]

                periods_id = 1 / frequencies_id
                periods_n = 1 / frequencies_n
                periods = 1 / frequencies

                h = np.logspace(-3, -2, 20) * 3 / frequencies
                h = h[:20]

#                 ax2.plot(resid_id, h/periods, label=f'm{m}-d{d}')

                axes[d, 0].semilogy(periods_n, h / periods, ls=linestyles[m], color=color)
                axes[d, 0].semilogy(periods_id, h / periods, color=color, marker=markers[m], ls='none')
                axes[d, 2].semilogy(damping_n, h / periods, color=color, ls=linestyles[m])
                axes[d, 2].semilogy(damping_id, h / periods, color=color, marker=markers[m], ls='none')
                axes[d, 1].semilogy(np.abs(periods_id - periods_n) / periods * 100, h / periods, color=color, marker=markers[m], ls='none')
                axes[d, 3].semilogy(np.abs(damping_id - damping_n), h / periods, color=color, marker=markers[m], ls='none')

            axes[d, 1].set_xlim((-0.0004, 0.049))
            axes[d, 3].set_xlim((-1e-7, 2.5e-5))
            axes[d, 1].xaxis.set_ticklabels(['-2.5e-2', '0', '2.5e-2', '5e-2'])
            axes[d, 3].xaxis.set_ticklabels(['-2.5e-5', '0', '2.5e-5'])
        axes[1, 0].set_xlabel('$T [\si{\second}]$')
        axes[1, 1].set_xlabel('$|\Delta T|$ [\si{\percent}]')
        axes[1, 2].set_xlabel('$\zeta$ [\si{\percent}]')
        axes[1, 3].set_xlabel('$|\Delta \zeta|$ [\si{\percent}]')

        axes[0, 0].yaxis.set_major_formatter(plot.NullFormatter())
#         axes[0,0].yaxis.set_minor_formatter(LogFormatterSciNotation())
        for d in range(2):
            axes[d, 0].set_ylabel('$\sfrac{h}{T}$')
            axes[d, 0].yaxis.set_label_coords(-0.2, 0.55)
            axes[d, 0].set_ylim((0.003, 0.03))
            axes[d, 0].set_xlim((3.1415, 3.1549))

        leg_handles = []
        for label, ls, mkr in list(zip(meths, linestyles, markers))[2:]:
            line = matplotlib.lines.Line2D([], [], color='black', marker=mkr, ls=ls, label=label)
            leg_handles.append(line)
        axes[0, 0].legend(handles=leg_handles).set_draggable(True)
        fig.subplots_adjust(top=0.970, bottom=0.115, left=0.112, right=0.970, hspace=0.180, wspace=0.1)
        axes[1, 2].set_xlim(xmax=0.1039)
#         plot.suptitle(f'$\\rho = {rho}$')
#         plot.savefig(str(pathlib.Path('figures') / 'example_accuracy_timeint.pdf'))
#         plot.savefig(str(pathlib.Path('figures') / 'example_accuracy_timeint.png'))
#         ax2.legend()
        plot.show()
        return


def IRF_to_ssi(ansys=None, jid=None, **kwargs):

    from pyOMA.core.SSICovRef import BRSSICovRef

    from pyOMA.core.PreProcessingTools import PreProcessSignals, GeometryProcessor

    # Modal Analysis PostProcessing Class e.g. Stabilization Diagram
    from pyOMA.core.StabilDiagram import StabilCalc, StabilPlot
    from pyOMA.GUI.StabilGUI import StabilGUI, start_stabil_gui

    # Modeshape Plot
    from pyOMA.core.PlotMSH import ModeShapePlot
    from pyOMA.GUI.PlotMSHGUI import start_msh_gui

    if ansys is not None:
        omega_d = np.pi / 2
        freq_scale = omega_d * np.sqrt(200 ** 2 * 7850 / 2 / 2.1e11)
        # freq_scale/=np.sqrt(1-0.05**2)
        num_nodes = 10
        num_cycles = 15,
        deltat = 0.005
        mech = generate_mdof_time_hist(ansys, damping=0.01, num_nodes=num_nodes, num_modes=num_nodes - 1,  # freq_scale=freq_scale,
                           just_build=True)

        dt_fact, deltat, num_cycles, timesteps, _ = mech.signal_parameters(deltat=deltat, num_cycles=num_cycles)

        ref_nodes = range(2, num_nodes + 1)
        imp_forces = [196, 56, 29, 19, 14, 11, 9, 8, 8]
        imp_times = [deltat for i in range(num_nodes)]
        imp_durs = [30 * deltat for i in range(num_nodes)]
        out_quant = ['a']
        t_vals, IRF_matrix, F_matrix, ener_mat, amp_mat = mech.impulse_response([ref_nodes, imp_forces, imp_times, imp_durs], form='step', mode='matrix', deltat=deltat, dt_fact=dt_fact, timesteps=timesteps, num_cycles=num_cycles, out_quant=out_quant, **kwargs)

        frequencies, damping, modeshapes = mech.numerical_response_parameters()
        meas_nodes = mech.meas_nodes
        num_meas_nodes = len(meas_nodes)

        jid = ansys.jobname

        mech.export_geometry(f'{tempfile.gettempdir()}/{jid}/')
        import PreProcessingTools

        if out_quant[0] == 'd':
            disp_channels = list(range(num_meas_nodes))
            velo_channels = None
            accel_channels = None
        elif out_quant[0] == 'a':
            accel_channels = list(range(num_meas_nodes))
            velo_channels = None
            disp_channels = None
        elif out_quant[0] == 'v':
            velo_channels = list(range(num_meas_nodes))
            disp_channels = None
            accel_channels = None
        channel_headers = meas_nodes
        ref_channels = [i for i, node in enumerate(meas_nodes) if node in ref_nodes]
        dummy_meas = np.zeros((timesteps, num_meas_nodes))

        prep_data = PreProcessingTools.PreProcessSignals(dummy_meas, 1 / deltat, ref_channels=ref_channels,
                                                      accel_channels=accel_channels,
                                                      velo_channels=velo_channels,
                                                      disp_channels=disp_channels,
                                                      channel_headers=channel_headers)
        chan_dofs = prep_data.load_chan_dofs(f'{tempfile.gettempdir()}/{jid}/chan_dofs.txt')
        prep_data.add_chan_dofs(chan_dofs)
        prep_data.save_state(f'{tempfile.gettempdir()}/{jid}/prep_data.npz')
        np.savez(f'{tempfile.gettempdir()}/{jid}/IRF_data.npz', t_vals=t_vals, IRF_matrix=IRF_matrix, F_matrix=F_matrix, ener_mat=ener_mat, amp_mat=amp_mat, frequencies=frequencies, damping=damping, modeshapes=modeshapes)

    else:
        assert jid is not None

    # creating the geometry for plotting the identified modeshapes
    geometry_data = GeometryProcessor.load_geometry(f'{tempfile.gettempdir()}/{jid}/grid.txt', f'{tempfile.gettempdir()}/{jid}/lines.txt')

    prep_data = PreProcessSignals.load_state(f'{tempfile.gettempdir()}/{jid}/prep_data.npz')

    arrs = np.load(f'{tempfile.gettempdir()}/{jid}/IRF_data.npz')
    t_vals = arrs['t_vals']
    IRF_matrix = arrs['IRF_matrix']
    F_matrix = arrs['F_matrix']
    ener_mat = arrs['ener_mat']
    amp_mat = arrs['amp_mat']
    frequencies = arrs['frequencies']
    damping = arrs['damping']
    modeshapes = arrs['modeshapes']

    print(frequencies, damping, modeshapes)

    n_lags = IRF_matrix.shape[2]
    prep_data.n_lags = n_lags
    prep_data.corr_matrix = IRF_matrix
    prep_data.corr_matrices = [IRF_matrix]

    modal_data = BRSSICovRef(prep_data)
    modal_data.build_toeplitz_cov(num_block_columns=125)
    modal_data.compute_state_matrices(max_model_order=19)

    frequencies_id, damping_id, modeshapes_id, _, modal_contributions = modal_data.single_order_modal(order=18)

    sort_inds = np.argsort(frequencies_id[:9])
    frequencies_id = frequencies_id[sort_inds]
    damping_id = damping_id[sort_inds]
    modal_contributions = modal_contributions[sort_inds]
    modeshapes_id = modeshapes_id[:, sort_inds]

    for mode in range(9):
        modeshapes_id[:, mode] /= modeshapes_id[np.argmax(np.abs(modeshapes_id[:, mode])), mode]
        modeshapes[:, mode] /= modeshapes[np.argmax(np.abs(modeshapes[:, mode])), mode]

        modeshapes_id[:, mode] /= modeshapes_id[-1, mode]
        modeshapes[:, mode] /= modeshapes[-1, mode]

    plot.figure()
    plot.plot(frequencies, ls='none', marker='+')
    plot.plot(frequencies_id, ls='none', marker='x')
    plot.figure()
    plot.plot(damping * 100, ls='none', marker='+')
    plot.plot(damping_id, ls='none', marker='x')
    plot.figure()
    plot.plot(modeshapes)
    plot.plot(modeshapes_id, ls='dotted')
    plot.figure()
    total_energy = np.sum(ener_mat)
    plot.plot(frequencies, np.sum(ener_mat, axis=0) / total_energy, ls='none', marker='+')
    plot.plot(frequencies_id, modal_contributions, ls='none', marker='x')
    plot.show()

    if False:
        modal_data.compute_modal_params()

        stabil_data = StabilCalc(modal_data)
        stabil_plot = StabilPlot(stabil_data)
        start_stabil_gui(stabil_plot, modal_data, geometry_data, prep_data)

        mode_shape_plot = ModeShapePlot(geometry_data, stabil_data, modal_data, prep_data)
        start_msh_gui(mode_shape_plot)


def test():
    '''
    A function to test basic functionality of the Mechanical class
    a lot of things are not yet verified/validated
    but at least the code runs, mistakes will be found later, maybe...
    '''
    
    working_dir = tempfile.gettempdir()
    os.makedirs(working_dir, exist_ok=True)
    os.chdir(working_dir)
    
    save_dir = str(pathlib.Path.cwd() / 'polyuq_results' / 'test_mechanical') + '/'
    
    jid = 'test_mechanical'
    ansys = Mechanical.start_ansys(working_dir, jid)
    
    if True:
        # Example structure Burscheid Longitudinal modes
        total_height = 200
        E = 2.1e11
        # I=0.01416
        A = 0.0343
        rho = 7850
    
        num_nodes = 41
        num_meas_nodes = 10
    
        section_length = total_height / (num_nodes - 1)
        nodes_coordinates = []
        k_vals = [0 for _ in range(num_nodes - 1)]
        masses = [0 for _ in range(num_nodes)]
        d_vals = None  # [0 for i in range(num_nodes-1)]
        eps_duff_vals = [0 for _ in range(num_nodes - 1)]
        sl_force_vals = [0 for _ in range(num_nodes - 1)]
        hyst_vals = [0 for _ in range(num_nodes - 1)]
        
        damping = (0.01, 0.02)
        
        for i in range(num_nodes):
            nodes_coordinates.append([i + 1, 0, 0, 0])  # to disable Warning "Nodes are not coincident"
            masses[i] += 0.5 * rho * A * section_length
            if i >= 2:
                masses[i - 1] += 0.5 * rho * A * section_length
            if i >= 1:
                k_vals[i - 1] = E * A / section_length
                
                # if nl_stiff is not None:
                    # eps_duff_vals[i-1]=nl_stiff
                # if sl_forces is not None:
                    # sl_force_vals[i-1]=sl_forces
                # if hyst_damp is not None:
                    # hyst_vals [i-1] = hyst_damp
    
        meas_nodes = np.rint(np.linspace(1, num_nodes, int(num_meas_nodes + 1))).astype(int)
    
        mech = Mechanical(ansys, jid, wdir=os.getcwd())
        mech.build_mdof(nodes_coordinates=nodes_coordinates,
                        k_vals=k_vals, masses=masses, d_vals=d_vals, damping=damping,
                        sl_force_vals=sl_force_vals, eps_duff_vals=eps_duff_vals,hyst_vals=hyst_vals,
                        meas_nodes=meas_nodes)
        N = 2**16
        fmax = None
        out_quant = 'a'
        inp_node = -1
        mech.frequency_response(N, fmax, out_quant, inp_node)
        
        mech.save(save_dir)
        del mech
        
    mech = Mechanical.load(jid, save_dir, ansys, working_dir)
    
    if not mech.state[1]:
        d0 = 1
        deltat = None
        dt_fact = 0.01
        timesteps = None
        num_cycles = 20
        t_vals_dec, resp_hist_dec = mech.free_decay(d0, 1, deltat, dt_fact, timesteps, num_cycles)
        
        mech.save(save_dir)
        mech.del_test()
        mech = Mechanical.load(jid, save_dir, ansys, working_dir)
    
    if False:#not mech.state[2]:
        f_scale = 1000
        deltat = 1/512
        dt_fact = None
        timesteps = 60*512
        num_cycles = None
        t_vals_amb, resp_hist_amb, inp_hist_amb = mech.ambient(f_scale, deltat, dt_fact, timesteps, num_cycles)
        
        mech.save(save_dir)
        mech = Mechanical.load(jid, save_dir, ansys, working_dir)
    
    if not mech.state[3]:
        deltat = 1/512
        dt_fact = None
        timesteps = 6*512
        num_cycles = None
        ref_nodes = mech.meas_nodes[-5:]
        imp_forces = [100 for _ in ref_nodes]
        imp_times = [deltat for _ in ref_nodes]
        imp_durs = [30 * deltat for _ in ref_nodes]
        out_quant = ['d']
        t_vals_imp, IRF_matrix, F_matrix, ener_mat, amp_mat = \
            mech.impulse_response([ref_nodes, imp_forces, imp_times, imp_durs],
                                  form='step', mode='matrix', deltat=deltat,
                                  dt_fact=dt_fact, timesteps=timesteps,
                                  num_cycles=num_cycles, out_quant=out_quant)
        
        mech.save(save_dir)
        mech = Mechanical.load(jid, save_dir, ansys, working_dir)
        
    if not mech.state[5]:
        frequencies, damping, modeshapes = mech.numerical_response_parameters()
        
        mech.save(save_dir)
        mech = Mechanical.load(jid, save_dir, ansys, working_dir)
    
    


def main():
    
    global ansys
    
    try:
        ansys.finish()
    except (pyansys.errors.MapdlExitedError, NameError) as e:
        logger.exception(e)
        global working_dir
        working_dir = tempfile.gettempdir()
        os.makedirs(working_dir, exist_ok=True)
        os.chdir(working_dir)
        global jid
        jid = str(uuid.uuid4()).split('-')[-1]
        ansys = Mechanical.start_ansys(working_dir, jid)

    ansys.clear()
#         IRF_to_ssi(ansys)
#     else:
#         IRF_to_ssi(jid='5a208083e5a1')
#     return
    # global working_dir
    # working_dir = os.getcwd()
    # working_dir = tempfile.gettempdir()
    # os.makedirs(working_dir, exist_ok=True)
    # os.chdir(working_dir)

#     import glob
#     filelist = glob.glob(f'{tempfile.gettempdir()}/*')
#     for file in filelist:
#         os.remove(file)

    # identify_student()

    # global jid
    # global ansys
    # jid=str(uuid.uuid4()).split('-')[-1]

    '''


    '''


#
    # nodes_coordinates = [(1, 0, 0, 0), (2, 0, 0, 0)]
    # k_vals = [1]
    # d_vals = None
    # damping = 0
    # meas_nodes = [2]
    # num_modes = 1
    #
    # omegas = np.linspace(.1, 10, 25, True)
    # fix, axes = plot.subplots(5, 5, sharex=True, sharey=True)
    # energies = []
    # for omega, ax in zip(omegas, axes.flat):
        # mech = Mechanical(ansys, jid, wdir=os.getcwd())
        # masses = [0, k_vals[0] / omega ** 2]
        #
        # mech.build_mdof(nodes_coordinates=nodes_coordinates,
                    # k_vals=k_vals, masses=masses, d_vals=d_vals, damping=damping,
                    # meas_nodes=meas_nodes, num_modes=num_modes)
                    #
        # deltat = 0.0025
        # timesteps = 801
        #
        # ref_nodes = [2]
        # imp_durs = [1]
        #
        # imp_times = [1 * deltat]
        # imp_forces = [1000]
        #
        # form = "rect"
        #
        # time_values, response_time_history, f , modal_imp_energies, modal_amplitudes = mech.impulse_response([ref_nodes, imp_forces, imp_times, imp_durs], form=form, mode='combined', deltat=deltat, timesteps=timesteps)
        # # print(modal_imp_energies)
        # energies.append(modal_imp_energies[1])
        # ax.plot(time_values, response_time_history[0], label=f'{omega}')
        # ax.axvline(1)
        # ax.set_title(f' $\omega={omega:.2f}$')
    # for ax in axes[-1, :]:
        # ax.set_xlabel("t [s]")
    # for ax in axes[:, 0]:
        # ax.set_ylabel("d [m]")
        #
    # fig, ax = plot.subplots(1, 1)
    # ax.plot(omegas, energies)
    # plot.show()

#     import sys
#     num=int(sys.argv[1])-1
#     m = num%6
#     d=num//6%2
#     r=num//12
# #     ## m 0...5, d = 0..1, r = 0...3
#     verify_numerical_accuracy(ansys, m, d, r)
#     verify_numerical_accuracy(ansys, m=5, d=0, r=0)
#     with matplotlib.rc_context(rc=print_context_dict):
#         verify_numerical_accuracy(ansys, r=1)

#     with matplotlib.rc_context(rc=print_context_dict):
#         rayleigh_example(ansys)

    eigenvalue_decomposition_example(ansys)

#     convergence_mdof_undamped(ansys)

#     import sys
#     # sys.argv = [..., ambient, nonlinear, friction]
#     if len(sys.argv)>3:
#         ambient = int(float(sys.argv[1]))
#         nonlinear = int(float(sys.argv[2]))
#         friction = int(float(sys.argv[3]))
#     else:
#         ambient=False
#         nonlinear=False
#         friction=True
#     student_data(ansys, ambient, nonlinear, friction)

#     hysteresis(ansys)

#     stepsize_example(ansys)


if __name__ == '__main__':

    '''
    Static response
        Nonlinear Stiffness
    Quasi-Static
        Nonlinear Stiffness
        Friction Response
    Modal Analysis
    Dynamic Response (Nyquist Check, Amplitude Check)
       Initial Displacement -> Free Vibration
           Numerical Damping
           Prescribed Damping
       Harmonic Loading
           Nonlinear Stiffness
           Damping Hysteresis
       Impulse
           IRF
       Sweep / Random Loading
           FRF

    MDOF
    Pass all verifications
    Random Field Loading
    Mesh Size
    Meas Nodes Definition
    Full Parametrization
    '''
#     main()

    """
    Test
    decay,ambient
        undamped
        damped
            linear, friction, nonlinear, friction+nonlinear

    """
    test()
    #main()
#     for d_scale, f_scale in [(1, None), (None, 1)]:
#         #if d_scale: continue
#         for zeta in [0, 1.2151028666644974]:
#             #if not zeta: continue
#             for fric_visc_rat, nl_ity in [(0, 0), (1, 0), (0, .5), (1, .5), ]:
# #                 if not fric_visc_rat: continue
# 
#                 if d_scale:
#                     jid = "decay_"
#                 else:
#                     jid = "ambient_"
#                 if zeta:
#                     jid += "damped_"
#                 if nl_ity:
#                     jid += "nonlinear_"
#                 if fric_visc_rat:
#                     jid += "friction_"
# 
#                 student_data_part2(jid=jid, result_dir=str(pathlib.Path.cwd() / 'polyuq_results' / 'test'),
#                            omega=2*np.pi*1.7240461713677018, zeta=zeta,
#                            dt_fact=0.005562, num_cycles=1304,
#                            f_scale=f_scale, d_scale=d_scale,
#                            nl_ity=nl_ity, fric_visc_rat=fric_visc_rat,
#                            snr_db=np.infty, working_dir=tempfile.gettempdir())

