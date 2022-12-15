import numpy as np
from scipy.constants import pi, m_e, e, c, alpha, hbar, epsilon_0, mu_0
import cupy as cp

def get_RE_spectrum_d(RE, t_ret, omega_axis):
    '''
    计算lw谱，不使用FFT直接积分。慢但是方便。

    RE : ndarray
        推迟势电场矢量REx, REy or REz
    t_ret : ndarray
        推迟时间矢量
    omega_axis : ndarray
        频谱的频率轴
    '''
    norm = 1 / np.sqrt(c*mu_0) / np.sqrt(2*pi)
    
    RE_d = cp.asarray(RE)
    t_ret_d = cp.asarray(t_ret)
    omega_axis_d = cp.asarray(omega_axis)[:, None]
    RE_ft_d = cp.trapz(RE_d * cp.exp(1j*omega_axis_d*t_ret_d), t_ret_d, axis=1) * norm
    
    return RE_ft_d.get()

def get_RE_spectrum_2d_d(RE, t_ret, omega_axis):
    '''
    计算lw谱，不使用FFT直接积分。慢但是方便。

    RE : ndarray
        推迟势电场矢量REx, REy or REz
    t_ret : ndarray
        推迟时间矢量
    omega_axis : ndarray
        频谱的频率轴
    '''
    assert RE.ndim == 2 and t_ret.ndim == 2 and omega_axis.ndim == 1
    norm = 1 / np.sqrt(c*mu_0) / np.sqrt(2*pi)

    nomega = len(omega_axis)
    npart, nt = RE.shape

    RE_d = cp.asarray(RE)
    t_ret_d = cp.asarray(t_ret)
    omega_axis_d = cp.asarray(omega_axis)[:, None]

    RE_ft_d = cp.zeros((nomega, npart), dtype=np.complex128)
    for iw in range(nomega):
        w = omega_axis_d[iw]
        RE_ft_d[iw, :] = cp.trapz(RE_d * cp.exp(1j*w*t_ret_d), t_ret_d, axis=1) * norm
    
    return RE_ft_d.get()