
import numpy as np

from .cpu import c, get_lw_spectrum_2d_cpu, get_lw_spectrum_cpu


def _check_velocity(t, x, y, z):
    ndim = x.ndim

    dt = t[1:] - t[:-1]
    if ndim == 2:
        dt = dt[:, None]
    vx = (x[1:, ...] - x[:-1, ...]) / dt
    vy = (y[1:, ...] - y[:-1, ...]) / dt
    vz = (z[1:, ...] - z[:-1, ...]) / dt
    beta = np.sqrt(vx**2 + vy**2 + vz**2) / c
    if (beta >= 1).any():
        raise ValueError("speed greater than c, check input trajectories. \
                            set `check_velocity` to False to disable check.")
    

def get_lw_spectrum(
    x, y, z, ux, uy, uz, t, n, omega_axis, 
    backend='mt', check_velocity=True,
):
    '''
    从坐标和动量计算LW场的频谱

    Prameters
    ===
    x, y, z : ndarray
        坐标向量或矩阵。其中矩阵的行为时间、列为不同粒子
    ux, uy, uz : ndarray
        归一化动量，u = p/mc。结构与xyz相同。
    t : ndarray
        时间矢量
    n : tuple | list | ndarray
        长度为1的方向矢量
    omega_axis : ndarray
        频谱的频率轴
    backend : str
        mp | cuda | None
    check_velocity : bool
        是否检查xyz的轨迹是否超过光速
    
    Returns
    ===
    I: ndarray
        返回dI/dΩdω
    '''

    assert backend in ['mt', 'cuda', None]
    assert omega_axis.ndim == 1
    
    if backend=='cuda':
        try:
            from .cuda import get_lw_spectrum_2d_cuda, get_lw_spectrum_cuda
        except ImportError as e:
            raise e
        
    if check_velocity:
        _check_velocity(t, x, y, z)
    
    args = (x, y, z, ux, uy, uz, t, n, omega_axis)
    argsT = (x.T, y.T, z.T, ux.T, uy.T, uz.T, t, n, omega_axis)
    if x.ndim == 1:
        if backend=='cuda':
            return get_lw_spectrum_cuda(*args)
        if backend == 'mt':
            return get_lw_spectrum_cpu(*args, mt=True)
        if backend is None:
            return get_lw_spectrum_cpu(*args, mt=False)
        
    elif x.ndim == 2:
        if x.shape[0] == x.shape[1]:
            print("input trajectory is a square matrix, assume the 2nd axis is time.")
        
        if len(t) == x.shape[1]:
            if backend=='cuda':
                return get_lw_spectrum_2d_cuda(*argsT)
            if backend == 'mt':
                return get_lw_spectrum_2d_cpu(*argsT, mt=True)
            if backend is None:
                return get_lw_spectrum_2d_cpu(*argsT, mt=False)
        elif len(t) == x.shape[0]:
            if backend=='cuda':
                return get_lw_spectrum_2d_cuda(*args)
            if backend == 'mt':
                return get_lw_spectrum_2d_cpu(*args, mt=True)
            if backend is None:
                return get_lw_spectrum_2d_cpu(*args, mt=False)
        else:
            raise TypeError("length of t does not match input 2d trajectory.")
    else:
        raise TypeError("input x, y, z, ux, uy, uz must 1d vector or 2d array.")

        
def get_lw_angular_spectrum(
    x, y, z, ux, uy, uz, t, omega_axis,
    theta_max, ntheta, theta_min=0.0,
    direction="x", theta_plane="xy",
    use_cuda=False, check_velocity=True,
):
    '''
    从坐标和动量计算LW场的频谱

    Prameters
    ===
    x, y, z : ndarray
        坐标向量或矩阵。其中矩阵的行为时间、列为不同粒子
    ux, uy, uz : ndarray
        归一化动量，u = p/mc。结构与xyz相同。
    t : ndarray
        时间矢量
    omega_axis : ndarray
        频谱的频率轴
    theta_max, theta_min : float
        角度范围参数
    ntheta : int
        角度数量
    direction : "x", "y" or "z"
        扫描角度的中心方向
    theta_plane : "xy", "yz" or "xz"
        角度范围所在的平面
    use_cuda : bool
        是否用gpu计算频谱，需要安装cupy
    check_velocity : bool
        是否检查xyz的轨迹是否超过光速
    
    Returns
    ===
    I: ndarray
        返回dI/dΩdω
    '''
    assert direction in ["x", "y", "z"]
    if direction == "x":
        assert theta_plane in ["xy", "xz"]
        if theta_plane == "xy":
            n = lambda theta : [np.cos(theta), np.sin(theta), 0.0]
        if theta_plane == "xz":
            n = lambda theta : [np.cos(theta), 0.0, np.sin(theta)]
    if direction == "y":
        assert theta_plane in ["xy", "yz"]
        if theta_plane == "xy":
            n = lambda theta : [np.sin(theta), np.cos(theta), 0.0]
        if theta_plane == "yz":
            n = lambda theta : [0.0, np.cos(theta), np.sin(theta)]
    if direction == "z":
        assert theta_plane in ["xz", "yz"]
        if theta_plane == "xz":
            n = lambda theta : [np.sin(theta), 0.0, np.cos(theta)]
        if theta_plane == "yz":
            n = lambda theta : [0.0, np.sin(theta), np.cos(theta)]
        

    nomega = len(omega_axis)
    theta_axis = np.linspace(theta_min, theta_max, ntheta)
    spectrum = np.zeros((ntheta, nomega))

    # TODO: 
    # 多个角度调用2d函数,
    # jit函数分段sum
    for itheta in range(ntheta):
        theta = theta_axis[itheta]
        spectrum[itheta, :] += get_lw_spectrum(x, y, z, ux, uy, uz, t, n(theta), omega_axis, use_cuda, check_velocity)
    
    return theta_axis, spectrum