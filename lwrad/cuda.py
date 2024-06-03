import numpy as np
from scipy.constants import pi, m_e, e, c, alpha, hbar, epsilon_0, mu_0
import cupy as cp


from numba import njit, cuda, float64, complex128
import math, cmath
from scipy.constants import pi, epsilon_0, mu_0, c, e

@njit(inline="always")
def _calc_beta(ux, uy, uz):
    inv_gamma = 1 / math.sqrt(ux**2 + uy**2 + uz**2 + 1)
    betax = ux * inv_gamma
    betay = uy * inv_gamma
    betaz = uz * inv_gamma
    return betax, betay, betaz


@njit(inline="always")
def _calc_lw(
    t, x, y, z, 
    ux, uy, uz, 
    ux_prev, uy_prev, uz_prev, 
    ux_next, uy_next, uz_next, 
    nx, ny, nz, dt_next_prev
):
    betax, betay, betaz = _calc_beta(ux, uy, uz)
    betax_prev, betay_prev, betaz_prev = _calc_beta(ux_prev, uy_prev, uz_prev)
    betax_next, betay_next, betaz_next = _calc_beta(ux_next, uy_next, uz_next)

    # 加速度，假设首尾加速度为0
    ax = (betax_next - betax_prev) / dt_next_prev
    ay = (betay_next - betay_prev) / dt_next_prev
    az = (betaz_next - betaz_prev) / dt_next_prev

    n_dot_a = nx*ax + ny*ay + nz*az
    n_dot_beta = nx*betax + ny*betay + nz*betaz
    factor = 1 / (1 - n_dot_beta)**3
    factor *= e / (4*pi*epsilon_0*c)

    t_ret = t - (nx*x + ny*y + nz*z) / c
    REx = (n_dot_a*(nx - betax) + (n_dot_beta - 1) * ax) * factor
    REy = (n_dot_a*(ny - betay) + (n_dot_beta - 1) * ay) * factor
    REz = (n_dot_a*(nz - betaz) + (n_dot_beta - 1) * az) * factor
    return t_ret, REx, REy, REz


@cuda.jit
def kernel_lw_RE(
    t, x, y, z, ux, uy, uz, 
    nx, ny, nz,
    t_ret, REx, REy, REz,
):
    
    nt = len(t)
    it = cuda.grid(1)
    if it == 0 or it >= nt - 1:
        return
    
    t_ret[it-1], REx[it-1], REy[it-1], REz[it-1] = _calc_lw(
        t[it], x[it], y[it], z[it], 
        ux[it], uy[it], uz[it], 
        ux[it-1], uy[it-1], uz[it-1], 
        ux[it+1], uy[it+1], uz[it+1], 
        nx, ny, nz, t[it+1] - t[it-1]
    )
    
    
@cuda.jit
def kernel_lw_RE_2d(
    t, x, y, z, ux, uy, uz, 
    nx, ny, nz,
    t_ret, REx, REy, REz,
):
    
    nt = len(t)
    npart = x.shape[1]
    it, ip = cuda.grid(2)
    if it == 0 or it >= nt - 1 or ip >= npart:
        return
    
    t_ret[it-1, ip], REx[it-1, ip], REy[it-1, ip], REz[it-1, ip] = _calc_lw(
        t[it], x[it, ip], y[it, ip], z[it, ip], 
        ux[it, ip], uy[it, ip], uz[it, ip], 
        ux[it-1, ip], uy[it-1, ip], uz[it-1, ip], 
        ux[it+1, ip], uy[it+1, ip], uz[it+1, ip], 
        nx, ny, nz, t[it+1] - t[it-1]
    )

_TPB = _NBUF = 32
_NSHR = _TPB*3
@cuda.jit
def kernel_RE_spectrum(
    t_ret, REx, REy, REz,
    omega, 
    REx_ft_real, REy_ft_real, REz_ft_real,
    REx_ft_imag, REy_ft_imag, REz_ft_imag,
):
    
    nt = len(t_ret)
    nomega = len(omega)

    it, iomega = cuda.grid(2)
    tid = cuda.threadIdx.x

    buf = cuda.shared.array(_NSHR, complex128)
    buf[tid] = 0
    buf[tid+_NBUF] = 0
    buf[tid+2*_NBUF] = 0
    if it >= nt-1 or iomega >= nomega:
        return

    w = omega[iomega]

    next = it + 1
    if math.isnan(REx[it]) or math.isnan(REx[next]):
        return

    dt = t_ret[next] - t_ret[it]

    buf[tid]         = 0.5*dt*(REx[it]*cmath.exp(1j*w*t_ret[it]) + REx[next]*cmath.exp(1j*w*t_ret[next]))
    buf[tid+_NBUF]   = 0.5*dt*(REy[it]*cmath.exp(1j*w*t_ret[it]) + REy[next]*cmath.exp(1j*w*t_ret[next]))
    buf[tid+2*_NBUF] = 0.5*dt*(REz[it]*cmath.exp(1j*w*t_ret[it]) + REz[next]*cmath.exp(1j*w*t_ret[next]))

    cuda.syncwarp()
    s = 1
    while s < _NBUF:
        if tid % (2 * s) == 0:
            # Stride by `s` and add
            buf[tid] += buf[tid + s]
            buf[tid+_NBUF] += buf[tid+_NBUF + s]
            buf[tid+2*_NBUF] += buf[tid+2*_NBUF + s]
        s *= 2
        cuda.syncthreads()
    
    if tid == 0:
        norm = 1 / math.sqrt(c*mu_0) / math.sqrt(2*pi)

        REx_ft = buf[0] * norm
        REy_ft = buf[_NBUF] * norm
        REz_ft = buf[2*_NBUF] * norm
        cuda.atomic.add(REx_ft_real, iomega, REx_ft.real)
        cuda.atomic.add(REy_ft_real, iomega, REy_ft.real)
        cuda.atomic.add(REz_ft_real, iomega, REz_ft.real)
        cuda.atomic.add(REx_ft_imag, iomega, REx_ft.imag)
        cuda.atomic.add(REy_ft_imag, iomega, REy_ft.imag)
        cuda.atomic.add(REz_ft_imag, iomega, REz_ft.imag)


@cuda.jit
def kernel_RE_spectrum_2d(
    t_ret, REx, REy, REz,
    omega, 
    I
):
    nomega = len(omega)
    nt, npart = t_ret.shape

    iomega, ip = cuda.grid(2)

    if ip >= npart or iomega >= nomega:
        return

    w = omega[iomega]
    REx_ft = 0.0j
    REy_ft = 0.0j
    REz_ft = 0.0j
    for it in range(nt-1):
        if math.isnan(REx[it+1, ip]):
            continue

        dt = t_ret[it+1, ip] - t_ret[it, ip]
        # todo: check dt
        REx_ft += 0.5*dt*(REx[it, ip]*cmath.exp(1j*w*t_ret[it, ip])+REx[it+1, ip]*cmath.exp(1j*w*t_ret[it+1, ip]))
        REy_ft += 0.5*dt*(REy[it, ip]*cmath.exp(1j*w*t_ret[it, ip])+REy[it+1, ip]*cmath.exp(1j*w*t_ret[it+1, ip]))
        REz_ft += 0.5*dt*(REz[it, ip]*cmath.exp(1j*w*t_ret[it, ip])+REz[it+1, ip]*cmath.exp(1j*w*t_ret[it+1, ip]))


    norm = 1 / math.sqrt(c*mu_0) / math.sqrt(2*pi)

    REx_ft *= norm
    REy_ft *= norm
    REz_ft *= norm
    cuda.atomic.add(I, iomega, 2*(REz_ft.real**2 + REz_ft.imag**2 + REx_ft.real**2 + REx_ft.imag**2 + REy_ft.real**2 + REy_ft.imag**2))


def get_lw_spectrum_cuda(x, y, z, ux, uy, uz, t, n, omega_axis):
    nomega = len(omega_axis)
    REx_ft = cp.zeros(nomega, dtype='c16')
    REy_ft = cp.zeros(nomega, dtype='c16')
    REz_ft = cp.zeros(nomega, dtype='c16')

    nt = len(t)
    t_ret = cp.zeros(nt-2)
    REx = cp.zeros_like(t_ret)
    REy = cp.zeros_like(t_ret)
    REz = cp.zeros_like(t_ret)
    kernel_lw_RE[nt//_TPB+1, _TPB](
        t, x, y, z, ux, uy, uz, 
        n[0], n[1], n[2],
        t_ret, REx, REy, REz,
    )
    kernel_RE_spectrum[((nt-2)//_TPB + 1, nomega), (_TPB, 1)](
        t_ret, REx, REy, REz,
        cp.array(omega_axis),
        REx_ft.real, REy_ft.real, REz_ft.real,
        REx_ft.imag, REy_ft.imag, REz_ft.imag
    )
    I = 2*(REz_ft.real**2 + REz_ft.imag**2 + REx_ft.real**2 + REx_ft.imag**2 + REy_ft.real**2 + REy_ft.imag**2)
    return I.get()


def get_lw_spectrum_2d_cuda(x, y, z, ux, uy, uz, t, n, omega_axis, tpb=128):
    nomega = len(omega_axis)
    npart = x.shape[1]
    nt = len(t)
    
    t_ret = cp.zeros((nt-2, npart), order='F')
    REx = cp.zeros_like(t_ret)
    REy = cp.zeros_like(t_ret)
    REz = cp.zeros_like(t_ret)
    kernel_lw_RE_2d[(nt//tpb+1, npart), (tpb, 1)](
        t, x, y, z, ux, uy, uz,
        n[0], n[1], n[2],
        t_ret, REx, REy, REz,
    )

    I = cp.zeros(nomega)
    kernel_RE_spectrum_2d[(nomega, (nt-2)//tpb + 1), (1, tpb)](
        t_ret, REx, REy, REz,
        omega_axis,
        I
    )
    return I.get()