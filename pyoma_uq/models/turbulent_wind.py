import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
import time

import numpy as np
import scipy

from numba import njit, prange, set_num_threads
#set_num_threads(18)

def spectral_wind_field(x_grid, f_w, L, U_bar, sigma, C_uz=10, C_vz=7, seed=None):
    
    n_z = x_grid.shape[0]
    
    n_fw = f_w.shape[0]
    delta_fw = f_w[1, 0] - f_w[0, 0]
    
    logger.info(f'A windfield of {n_z} x {n_fw} samples will be sampled at {delta_fw * (2 * n_fw - 2)} Hz (duration {1 / delta_fw} s).')
    
    # random phases of the windfield
    if seed is None:
        seed = np.random.randint(np.iinfo(np.int32).max)
    logger.info(f'Random seed is: {seed}')
    rng = np.random.default_rng(seed)
    phi_um = rng.random((n_fw, n_z), dtype=np.float32) * 2 * np.pi 
    phi_vm = rng.random((n_fw, n_z), dtype=np.float32) * 2 * np.pi 
    # phi_wm = np.random.random((n_z, n_fw)) * 2 * np.pi 
    
    logger.debug(phi_um)
    logger.debug(phi_vm)
    # Compute basic wind parameters and auto spectral densitities according to [Clobes 2008]

    # transpose column vectors
    U_bar = U_bar[np.newaxis,:]
    sigma = sigma[np.newaxis,:]
    
    L_ux = L[np.newaxis,:]
    L_vx = 0.3 * L_ux
    
    # Normalized frequency
    f_n = f_w * L_ux / U_bar
    S_uu = 4 * f_n / (1 + 70.8 * f_n**2)**(5/6) / f_w * sigma**2
    # print(S_uu.shape)
    f_n = f_w * L_vx / U_bar
    S_vv = 4 * f_n * (1 + 755.2 * f_n**2) / (1 + 283.2 * f_n**2)**(11 / 6) / f_w * sigma**2
    
    # f_n = f_w * z / U_bar
    # S_ww = 2.15 * f_n / (1 + 11.16 * f_n**(5 / 3)) / f_w * sigma**2
    
    # CPSD spectral_assembly
    S_uu_full_b = np.zeros((n_fw, n_z, n_z), dtype=np.float32)
    S_vv_full_b = np.zeros((n_fw, n_z, n_z), dtype=np.float32)
    # S_ww_full_b = np.zeros((u + 1, n_z, n_fw))
    
    # numba just-in-time compilation for parallel computation
    # high memory requirements limit single runs of this function on a given node
    # a single core implementation would take much longer and can not be compensated by parallel sample computation
    @njit(parallel=True)
    def spectral_assembly(i, j, S_uu_full_b, S_vv_full_b, x_grid, f_w, C_uz, C_vz, U_bar):
        if i==j: 
            S_uu_full_b[:, i - j, j] = S_uu[:, i]
            S_vv_full_b[:, i - j, j] = S_vv[:, i]
            # S_ww_full_b[i - j, j,:] = S_ww[i,:]
        else:
            delta_x = np.abs(x_grid[i] - x_grid[j])
            
            common = np.exp(-2 * f_w[:,0] * delta_x / (U_bar[0,i] + U_bar[0,j]))
            
            #coh_u = np.exp(-2 * f_w[:,0] * C_uz * delta_x / (U_bar[0,i] + U_bar[0,j]))
            coh_u = common**(C_uz*2)
            # print(coh_u.shape)
            #r = np.sqrt(coh_u**2 * S_uu[:, i] * S_uu[:, j])
            r = np.sqrt(coh_u * S_uu[:, i] * S_uu[:, j])
            # print(r.shape)
            S_uu_full_b[:, i - j, j] = r #* np.exp(1j*phi)
            
            #coh_v = np.exp(-2 * f_w[:,0] * C_vz * delta_x / (U_bar[0,i] + U_bar[0,j])) 
            coh_v = common**(C_vz*2)
            #r = np.sqrt(coh_v**2 * S_vv[:, i] * S_vv[:, j])
            r = np.sqrt(coh_v * S_vv[:, i] * S_vv[:, j])
            S_vv_full_b[:, i - j, j] = r
            
            # coh_w = np.exp(-2 * f_w * C_wz * delta_x / (U_bar[i] + U_bar[j]))
            # r = np.sqrt(coh_w**2 * S_ww[i,:] * S_ww[j,:])
            # S_ww_full_b[i - j, j, :] = r
    
    for i in range(n_z):
        for j in prange(i + 1):
            spectral_assembly(i, j, S_uu_full_b, S_vv_full_b, x_grid, f_w, C_uz, C_vz, U_bar)
                
    if f_w[0,0] == 0:
        S_uu_full_b[0, :, :] = 0
        S_vv_full_b[0, :, :] = 0
        # S_ww_full_b[:,:,0] = 0
        S_uu_full_b[0, 0, :] = np.ones(n_z)*1e-10
        S_vv_full_b[0, 0, :] = np.ones(n_z)*1e-10
        # S_ww_full_b[0,:,0] = np.ones(n_z)*1e-10
        
    del S_uu
    del S_vv
    
    # Decompose Cross-spectral densities
    S_uu_chol_b = np.empty_like(S_uu_full_b, dtype=np.float32)
    S_vv_chol_b = np.empty_like(S_vv_full_b, dtype=np.float32)
    # S_ww_chol_b = np.empty_like(S_ww_full_b)
    
    u_target = 10
    S_uu_chol_b[0, :u_target, :] = scipy.linalg.cholesky_banded(S_uu_full_b[0, :u_target, :], lower=True)
    S_vv_chol_b[0, :u_target, :] = scipy.linalg.cholesky_banded(S_vv_full_b[0, :u_target, :], lower=True)
    # S_ww_chol_b[:u_target,:,0] = scipy.linalg.cholesky_banded(S_ww_full_b[:u_target,:,0], lower=True)
    
    all_u = np.zeros((n_fw))
    for k in range(n_fw - 1, 0, -1):
        # for lower frequencies, we have stronger coherence and need more off-diagonals
        # we iterate backwards in frequency, increasing u when needed
        # for the first frequency we usually need all bands
        while True:
            try:
                S_uu_chol_b[k, :u_target, :] = scipy.linalg.cholesky_banded(S_uu_full_b[k, :u_target, :], lower=True)
                S_vv_chol_b[k, :u_target, :] = scipy.linalg.cholesky_banded(S_vv_full_b[k, :u_target, :], lower=True)
                # S_ww_chol_b[:u_target,:,k] = scipy.linalg.cholesky_banded(S_ww_full_b[:u_target,:,k], lower=True)
                all_u[k] = u_target
                break
            # catch non-positive-definite correlation matrices
            except Exception as e:
                u_target += 1
                if u_target > n_z:
                    print(k, u_target,  e)
    
    del S_uu_full_b
    del S_vv_full_b
    
    @njit(parallel=True)
    def fourier_assembly(S_chol_b, phase, c_j):
        c_j_b = np.fliplr(S_chol_b * phase)
        for i in prange(n_fw):
            for j in prange(n_z):
                c_j[i, j] = np.trace(c_j_b[:,:,i], offset=n_z - j - 1)
                
    # Fourier coefficient spectral_assembly
    c_uj = np.zeros((n_fw, n_z), dtype=complex)
    c_vj = np.zeros((n_fw, n_z), dtype=complex)
    # c_wj = np.zeros((n_z, n_fww), dtype=complex)
    delta_omega = delta_fw * 2 * np.pi
    # last part is the most expensive as sin(phi_um) has to be repeated n_z times along axis=0)
    # we transpose both arrays because numba implementations of flip and trace do not suport axis arguments
    S_uu_chol_b, phase_u = np.broadcast_arrays(np.transpose(np.abs(S_uu_chol_b, where=S_uu_chol_b!=0) , axes=(1, 2, 0)), 
                                             np.sqrt(2* delta_omega) * np.exp(1j * phi_um.T)[np.newaxis, :, :])
    # ..TODO:: It is actually invalid to just randomize the phase. 
    # Spectrum must be generated as a complex timeseries with N(0, S_uu_chol_b) + j N(0, S_uu_chol_b)
    # https://adsabs.harvard.edu/full/1995A%26A...300..707T
    fourier_assembly(S_uu_chol_b, phase_u, c_uj)
    del S_uu_chol_b
    del phase_u
    
    S_vv_chol_b, phase_v = np.broadcast_arrays(np.transpose(np.abs(S_vv_chol_b, where=S_vv_chol_b!=0), axes=(1, 2, 0)), 
                                             np.sqrt(2* delta_omega) * np.exp(1j * phi_vm.T)[np.newaxis, :, :])
    fourier_assembly(S_vv_chol_b, phase_v, c_vj)
    del S_vv_chol_b
    del phase_v

    return c_uj, c_vj

def temporal_wind_field(c_uj, c_vj, N_m):
    
    n_z = c_uj.shape[1]
    logger.info(f'Transforming windfield to time domain with {N_m} x {n_z} samples.')
    
    u_j = np.zeros((N_m, n_z))
    v_j = np.zeros((N_m, n_z))
    # w_j = np.zeros((n_z, N_m))
    
    for j in range(n_z):
        u_j[:, j] = np.fft.irfft(c_uj[:, j], n=N_m, norm="forward") / 2 / np.pi # d="backward", "ortho", "forward"
        v_j[:, j] = np.fft.irfft(c_vj[:, j], n=N_m, norm="forward") / 2 / np.pi # d="backward", "ortho", "forward"
        # w_j[j,:] = np.fft.irfft(c_wj[j,:], n=N_m, norm="forward") # "backward", "ortho", "forward"
    
    return u_j, v_j

def force_wind_field(u_j, v_j, delta_x=1.0, b=1.9, cscd=1.0, cf=2.86519, rho=1.25):
    logger.info(f'Computing forces from windfield with A_ref={delta_x*b} and c_f = {cf}')
    
    # q_b = 0.5*\rho*v_b^2$
    q_buj = 0.5 * rho * u_j**2 * np.sign(u_j)
    q_bvj = 0.5 * rho * v_j**2 * np.sign(v_j)
    
    # F_w = c_sc_d*c_f*q_p*A_ref
    # A_ref = delta_x * b Lasteinzugsflaeche des Knotens
    F_uj = cscd * cf * q_buj * delta_x * b
    F_vj = cscd * cf * q_bvj * delta_x * b
    
    return F_uj, F_vj

def terrain_parameters(category):
    z_min = [None,2,4,8,16][category]
    # z_0 = [None, 0.01,0.05,0.3,1.05][category] # Rauigkeitslänge
    alpha = [None,0.12,0.16,0.22, 0.3][category]  # Profilexponent
    vm_fact = [None, 1.18, 1.00, 0.77, 0.56][category]
    vm_min = [None, 0.97, 0.86, 0.73, 0.64][category]
    Iv_fact = [None, 0.14,0.19,0.28,0.43][category]
    Iv_min = [None, 0.17, 0.22, 0.29, 0.37][category]
    eps = [None, 0.13, 0.26, 0.37, 0.46][category] # Exponent \epsilon nach Tabelle NA.C.1
    
    return z_min, alpha, vm_fact, vm_min, Iv_fact, Iv_min, eps

def basic_wind_parameters(x_grid, v_b, z_min, alpha, vm_fact, vm_min, Iv_fact, Iv_min, eps):
    logger.info(f'Computing basic wind parameters according to EN 1991-1-4 for v_b {v_b}')
    # Compute basic wind parameters according to DIN EN 1991-1-4 and corresponding NA    
    # Werte nach Tab. NA.B.2
    v_m = vm_fact * v_b * (x_grid/10)**alpha
    v_m[x_grid<=z_min] = vm_min * v_b
    I_v = Iv_fact*(x_grid/10)**(-alpha) # Turbulenzintensität [% der mittleren Windgeschwindigkeit] (? siehe unten)
    I_v[x_grid<=z_min] = Iv_min
    sigma_v = I_v * v_m # Standardabweichung der Turbulenz nach Gl. 4.7
    L = (x_grid/300)**eps * 300 # Integrallängenmaß nach Gl. NA.C.2
    L[x_grid<=z_min] = (z_min/300)**eps * 300
    
    return v_m, sigma_v, L
    

def default_wind_field(category=2, zone=2, duration=36, fs_w=10, fs_m=100, **kwargs):
    
    
    # 200 x 2**17 samples take 1min 42 (CPSD spectral_assembly) + 2min 20s (CPSD decomposition) + 1min 59s(Fourier coefficients spectral_assembly) + 1.4s (IFFT) = 6min 2s; 
    # peak at 270 GB RAM
    
    # Spatial domain grid
    x_grid = np.arange(1,201,1)
    
    # Frequency domain grid
    
    # wind does not have to be generated up to higher frequencies, 
    # as the energy in these bands is negligible
    # fs_w = 10 # Hz, Wind sample rate
    # sample rate is adjusted by zero padding the fft
    # fs_m = 100 # Hz, Model sample rate
    
    # duration = 36 # seconds
    
    if duration is None:
        N = 2**17
        duration = N / fs_w
    else:
        N = int(duration * fs_w)
    
    N_m =  int(duration * fs_m)
    
    f_w = np.fft.rfftfreq(N, 1/fs_w)[np.newaxis,:]
    
    # Geländekategorie I - IV
    z_min, alpha, vm_fact, vm_min, Iv_fact, Iv_min, eps = terrain_parameters(category)
    
    # Windzone 1 - 4
    v_b = [kwargs.get('v_b', None), 22.5, 25.0, 27.5, 30][zone] # m/s Basiswindgeschwindigkeit (v_b = v_b0 in DE vgl. NA S. 5)
    
    v_m, sigma_v, L = basic_wind_parameters(x_grid, v_b, z_min, alpha, vm_fact, vm_min, Iv_fact, Iv_min, eps)
    
    C_uz = 10
    C_vz = 7
    C_wz = None
    logger.info(f'Windfield properties: mean wind speed {v_b}, standard deviation of turbulence {np.mean(sigma_v)}, bandwidth {fs_w/2}, decay factors {[C_uz, C_vz, C_wz]}')
    c_uj, c_vj = spectral_wind_field(x_grid, f_w, 
                                     L, v_m, sigma_v, C_uz, C_vz, C_wz, 
                                     seed=None)
    
    u_j, v_j = temporal_wind_field(c_uj, c_vj, N_m)
    
    
    F_uj, F_vj = force_wind_field(u_j + v_m[:,np.newaxis], v_j, delta_x=x_grid[1]-x_grid[0], 
                                  b=1.9, cscd=1.0, cf=2.86519, rho=1.25)
    
    # plot_windfield(u_j, v_j, F_uj, F_vj)
    
    animate_windfield(x_grid,u_j + v_m[:,np.newaxis], v_j)
    return F_uj, F_vj

def plot_windfield(u_j=None, v_j=None, F_uj=None, F_vj=None, duration=None, height=None):
    import matplotlib.pyplot as plt
    
    for arr in [u_j, v_j, F_uj, F_vj]:
        if arr is not None:
            if duration is None:
                duration = arr.shape[1]
            if height is None:
                height = arr.shape[0]
            break
            
    extent = (0, duration, 0, height)
        
    if u_j is not None and v_j is not None:
        Vmin = min([u_j.min(), v_j.min()])#, w_j.min()])
        Vmax = max([u_j.max(), v_j.max()])#, w_j.max()])
    else:
        Vmin, Vmax = None, None
        
    if F_uj is not None and F_vj is not None:
        Fmin = min([F_uj.min(), F_vj.min()])
        Fmax = max([F_uj.max(), F_vj.max()])
    else:
        Fmin, Fmax = None, None
    
    if u_j is not None:
        fig = plt.figure()
        plt.matshow(u_j, fig.number, origin='lower', aspect='auto', extent=extent, vmin=Vmin, vmax=Vmax)
        plt.xlabel('time [s]')
        plt.ylabel('height [m]')
        cbar = plt.colorbar()
        cbar.set_label('Turbulence $V_u$ [m/s]')
        
    if v_j is not None:
        fig = plt.figure()
        plt.matshow(v_j, fig.number, origin='lower', aspect='auto', extent=extent, vmin=Vmin, vmax=Vmax)
        plt.xlabel('time [s]')
        plt.ylabel('height [m]')
        cbar = plt.colorbar()
        cbar.set_label('Turbulence $V_v$ [m/s]')
        
    if F_uj is not None:
        fig = plt.figure()
        plt.matshow(F_uj, fig.number, origin='lower', aspect='auto', extent=extent, vmin=Fmin, vmax=Fmax)
        plt.xlabel('time [s]')
        plt.ylabel('height [m]')
        cbar = plt.colorbar()
        cbar.set_label('Force $F_{w,u}$ [N]')
    
    if F_vj is not None:
        fig = plt.figure()
        plt.matshow(F_vj, fig.number, origin='lower', aspect='auto', extent=extent, vmin=Fmin, vmax=Fmax)
        plt.xlabel('time [s]')
        plt.ylabel('height [m]')
        cbar = plt.colorbar()
        cbar.set_label('Force $F_{w,v}$ [N]')
    
    plt.show()
    
def animate_windfield(x_grid, u_j, v_j):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    
    n_z = x_grid.shape[0]
    N_m = u_j.shape[1]
    
    lim = np.max([-np.min(u_j), np.max(u_j), -np.min(v_j), np.max(v_j)])
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    z = x_grid
    zero = np.zeros_like(z)
    # to run GUI event loop
    
    u_j_inter = np.empty((3*n_z,), dtype=u_j.dtype)
    u_j_inter[0::3] = zero
    u_j_inter[1::3] = u_j[:,0]
    u_j_inter[2::3] = zero
    
    v_j_inter = np.empty((3*n_z,), dtype=v_j.dtype)
    v_j_inter[0::3] = zero
    v_j_inter[1::3] = v_j[:,0]
    v_j_inter[2::3] = zero
    
    x_grid_inter = np.empty((3*n_z,), dtype=x_grid.dtype)
    x_grid_inter[0::3] = x_grid
    x_grid_inter[1::3] = x_grid
    x_grid_inter[2::3] = x_grid
    
    zero_inter = np.empty((3*n_z,), dtype=zero.dtype)
    zero_inter[0::3] = zero
    zero_inter[1::3] = zero
    zero_inter[2::3] = zero
    
    lines = ax.plot(u_j_inter, v_j_inter, x_grid_inter, alpha=0.5)[0]
    
    
    ax.set_xlim((-lim, lim))
    ax.set_ylim((-lim, lim))
    
    def update(n):
        u_j_inter[1::3] = u_j[:,n]
        v_j_inter[1::3] = v_j[:,n]
        lines.set_data(u_j_inter, v_j_inter)
        lines.set_3d_properties(x_grid_inter)
        return [lines]
    
    ani = FuncAnimation(fig, update, frames=N_m, blit=True, interval=33)
    plt.show()   

def main():
    F_uj, F_vj = default_wind_field(category=4, zone=1)


if __name__ == '__main__':
    main()
