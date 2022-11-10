import numpy as np
from scipy.constants import pi, m_e, e, c, alpha, hbar, epsilon_0, mu_0
from numba import njit, prange

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

    n_norm = np.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
    if isinstance(n, tuple):
        n = list(n)
    if n_norm != 1:
        n[0] /= n_norm
        n[1] /= n_norm
        n[2] /= n_norm


    t_ret = t - (n[0]*x + n[1]*y + n[2]*z) / c
    inv_gamma = 1 / np.sqrt(ux**2 + uy**2 + uz**2 + 1)
    betax = ux * inv_gamma
    betay = uy * inv_gamma
    betaz = uz * inv_gamma
    # 加速度，假设首尾加速度为0
    ax = np.concatenate(([0], (betax[2:] - betax[:-2])/(t[2:] - t[:-2]), [0]))
    ay = np.concatenate(([0], (betay[2:] - betay[:-2])/(t[2:] - t[:-2]), [0]))
    az = np.concatenate(([0], (betaz[2:] - betaz[:-2])/(t[2:] - t[:-2]), [0]))

    n_dot_a = n[0]*ax + n[1]*ay + n[2]*az
    n_dot_beta = n[0]*betax + n[1]*betay + n[2]*betaz
    factor = 1 / (1 - n_dot_beta)**3
    factor *= e / (4*pi*epsilon_0*c)

    REx = (n_dot_a*(n[0] - betax) + (n_dot_beta - 1) * ax) * factor
    REy = (n_dot_a*(n[1] - betay) + (n_dot_beta - 1) * ay) * factor
    REz = (n_dot_a*(n[2] - betaz) + (n_dot_beta - 1) * az) * factor
    return t_ret, REx, REy, REz


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
    
    # definition of Fourier transformation
    for i in prange(nomega):
        w = omega_axis[i]
        # trapzoid integral
        RE_ft[i] = np.trapz(RE * np.exp(1j*w*t_ret), t_ret) / np.sqrt(c*mu_0) / np.sqrt(2*pi)

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