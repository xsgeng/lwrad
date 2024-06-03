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
        warn("NaN values detected in input vector. NaNs will be discarded.")

    nx = n[0]
    ny = n[1]
    nz = n[2]
    n_norm = np.sqrt(nx**2 + ny**2 + nz**2)
    if n_norm != 1:
        nx /= n_norm
        ny /= n_norm
        nz /= n_norm

    return calc_lw(x, y, z, ux, uy, uz, t, nx, ny, nz)


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

calc_lw_mt = njit(_calc_lw, parallel=True)
calc_lw = njit(_calc_lw)


@njit(inline="always")
def _calc_beta(ux, uy, uz):
    inv_gamma = 1 / np.sqrt(ux**2 + uy**2 + uz**2 + 1)
    betax = ux * inv_gamma
    betay = uy * inv_gamma
    betaz = uz * inv_gamma
    return betax, betay, betaz

@njit(parallel=True)
def get_RE_spectrum_mt(RE, t_ret, omega_axis):
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

@njit
def get_RE_spectrum_st(RE, t_ret, omega_axis):
    '''
    计算lw谱，不使用FFT直接积分。慢但是方便。

    RE : ndarray
        推迟势电场矢量REx, REy or REz
    t_ret : ndarray
        推迟时间矢量
    omega_axis : ndarray
        频谱的频率轴
    '''
    assert RE.ndim == 1 and t_ret.ndim == 1 and omega_axis.ndim == 1
    
    nomega = len(omega_axis)
    RE_ft = np.zeros(nomega, dtype=np.complex128)
    norm = 1 / np.sqrt(c*mu_0) / np.sqrt(2*pi)
    
    nan_mask = ~np.isnan(RE)
    RE_ = RE[nan_mask]
    t_ret_ = t_ret[nan_mask]
    nt_ = len(t_ret_)

    buf = np.zeros(nt_, dtype='c16')
    # definition of Fourier transformation
    for iw in range(nomega):
        w = omega_axis[iw]

        for it in range(nt_):
            buf[it] = RE_[it] * np.exp(1j*w*t_ret_[it])

        # trapzoid integral
        for it in range(nt_-1):
            dt = t_ret[it+1] - t_ret[it]
            RE_ft[iw] += 0.5*dt*(buf[it]+buf[it+1])
        RE_ft[iw] *= norm

    return RE_ft

@njit(parallel=True)
def get_RE_spectrum_2d_mt(RE, t_ret, omega_axis):
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
    npart, nt = RE.shape

    RE_ft = np.zeros((nomega, npart), dtype=np.complex128)
    norm = 1 / np.sqrt(c*mu_0) / np.sqrt(2*pi)
    
    # definition of Fourier transformation
    for ip in prange(npart):
        # trapzoid integral
        RE_ft[:, ip] = get_RE_spectrum_st(RE[ip], t_ret[ip], omega_axis)

    # normalize
    return RE_ft

get_RE_spectrum_2d_st = njit(get_RE_spectrum_2d_mt.py_func)


def get_lw_spectrum_cpu(
    x, y, z, ux, uy, uz, t, n, omega_axis, 
    mt=True,
):
    nomega = len(omega_axis)
    t_ret, REx, REy, REz = get_lw_RE(x, y, z, ux, uy, uz, t, n)
    I = np.zeros(nomega)
    for RE in (REx, REy, REz):
        if mt:
            RE_ft = get_RE_spectrum_mt(RE, t_ret, omega_axis)
        else:
            RE_ft = get_RE_spectrum_st(RE, t_ret, omega_axis)
        I += RE_ft.real**2 + RE_ft.imag**2
    I *= 2
    return I
    

def get_lw_spectrum_2d_cpu(
    x, y, z, ux, uy, uz, t, n, omega_axis, 
    mt=True,
):
    nt, npart = x.shape
    nomega = len(omega_axis)

    # buffer
    t_ret = np.zeros((npart, nt-2))
    REx = np.zeros((npart, nt-2))
    REy = np.zeros((npart, nt-2))
    REz = np.zeros((npart, nt-2))

    I = np.zeros((nomega, npart))

    for i in range(npart):
        t_ret[i, :], REx[i, :], REy[i, :], REz[i, :] = get_lw_RE(x[:, i], y[:, i], z[:, i], ux[:, i], uy[:, i], uz[:, i], t, n)
        
    for RE in (REx, REy, REz):
        if mt:
            RE_ft = get_RE_spectrum_2d_mt(RE, t_ret, omega_axis)
        else:
            RE_ft = get_RE_spectrum_2d_st(RE, t_ret, omega_axis)
        I += RE_ft.real**2 + RE_ft.imag**2
    I *= 2
    return I.sum(axis=1)

    
