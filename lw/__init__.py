import numpy as np
from scipy.constants import pi, m_e, e, c, alpha, hbar, epsilon_0, mu_0
from numba import njit, prange
from warnings import warn

def get_lw_RE(x, y, z, ux, uy, uz, t, n):
    '''
    从坐标和动量计算推迟时间和电场，实际为计算R*E，忽略速度项

    参考doi.org/10.5281/zenodo.843510 2.32、2.36、2.39式

    x, y, z : ndarray
        坐标向量
    ux, uy, uz : ndarray
        归一化动量，u = p/mc
    t : ndarray
        时间矢量
    n : tuple | list | ndarray
        长度为1的方向矢量
    '''

    if np.isnan(x).any():
        warn("NaN values detected in input vector. Discarded continue.")
        nan_mask = ~np.isnan(x)
        x = x[nan_mask]
        y = y[nan_mask]
        z = z[nan_mask]
        ux = ux[nan_mask]
        uy = uy[nan_mask]
        uz = uz[nan_mask]
        t = t[nan_mask]

    nx = n[0]
    ny = n[1]
    nz = n[2]
    n_norm = np.sqrt(nx**2 + ny**2 + nz**2)
    if n_norm != 1:
        nx /= n_norm
        ny /= n_norm
        nz /= n_norm

    return _calc_lw(x, y, z, ux, uy, uz, t, nx, ny, nz)


@njit(parallel=True)
def _calc_lw(x, y, z, ux, uy, uz, t, nx, ny, nz):
    nt = len(t)
    t_ret = np.zeros(nt-2)
    REx = np.zeros(nt-2)
    REy = np.zeros(nt-2)
    REz = np.zeros(nt-2)
    for it in prange(1, nt-1):

        t_ret[it-1] = t[it] - (nx*x[it] + ny*y[it] + nz*z[it]) / c

        betax, betay, betaz = _calc_beta(ux[it], uy[it], uz[it])
        betax_prev, betay_prev, betaz_prev = _calc_beta(ux[it-1], uy[it-1], uz[it-1])
        betax_next, betay_next, betaz_next = _calc_beta(ux[it+1], uy[it+1], uz[it+1])

        dt = t[it+1] - t[it-1]
        # 加速度，假设首尾加速度为0
        ax = (betax_next - betax_prev) / dt
        ay = (betay_next - betay_prev) / dt
        az = (betaz_next - betaz_prev) / dt

        n_dot_a = nx*ax + ny*ay + nz*az
        n_dot_beta = nx*betax + ny*betay + nz*betaz
        factor = 1 / (1 - n_dot_beta)**3
        factor *= e / (4*pi*epsilon_0*c)

        REx[it-1] = (n_dot_a*(nx - betax) + (n_dot_beta - 1) * ax) * factor
        REy[it-1] = (n_dot_a*(ny - betay) + (n_dot_beta - 1) * ay) * factor
        REz[it-1] = (n_dot_a*(nz - betaz) + (n_dot_beta - 1) * az) * factor
    return t_ret, REx, REy, REz

@njit
def _calc_beta(ux, uy, uz):
    inv_gamma = 1 / np.sqrt(ux**2 + uy**2 + uz**2 + 1)
    betax = ux * inv_gamma
    betay = uy * inv_gamma
    betaz = uz * inv_gamma
    return betax, betay, betaz


@njit(parallel=True)
def get_RE_spectrum(RE, t_ret, omega_axis):
    '''
    计算lw谱，不使用FFT直接积分。慢但是方便。

    RE : ndarray
        推迟势电场矢量REx, REy or REz
    t_ret : ndarray
        推迟时间矢量
    omega_axis : ndarray
        频谱的频率轴
    '''
    nomega = len(omega_axis)
    RE_ft = np.zeros(nomega, dtype=np.complex128)
    norm = 1 / np.sqrt(c*mu_0) / np.sqrt(2*pi)
    
    # definition of Fourier transformation
    for i in prange(nomega):
        w = omega_axis[i]
        # trapzoid integral
        RE_ft[i] = np.trapz(RE * np.exp(1j*w*t_ret), t_ret) * norm

    # normalize
    return RE_ft


def get_lw_spectrum(x, y, z, ux, uy, uz, t, n, omega_axis):
    '''
    从坐标和动量计算LW场的频谱

    Prameters
    ===
    x, y, z : ndarray
        坐标向量
    ux, uy, uz : ndarray
        归一化动量，u = p/mc
    t : ndarray
        时间矢量
    n : tuple | list | ndarray
        长度为1的方向矢量
    omega_axis : ndarray
        频谱的频率轴
    
    Returns
    ===
    I: ndarray
        返回dI/dΩdω
    '''
    t_ret, REx, REy, REz = get_lw_RE(x, y, z, ux, uy, uz, t, n)
    

    I = np.zeros(len(omega_axis))
    for RE in (REx, REy, REz):
        RE_ft = get_RE_spectrum(RE, t_ret, omega_axis)
        I += RE_ft.real**2 + RE_ft.imag**2
    I *= 2
    return I