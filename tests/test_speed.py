
import unittest
import numpy as np
from lw import c, pi, get_lw_spectrum
from time import perf_counter_ns

def dummy_thomson():
    import h5py

    l0 = 0.8e-6
    with h5py.File("tests/test.h5", "r") as f:
        x = f['x'][:, 0] / 2/pi * l0
        y = f['y'][:, 0] / 2/pi * l0
        z = f['z'][:, 0] / 2/pi * l0
        ux = f['px'][:, 0]
        uy = f['py'][:, 0]
        uz = f['pz'][:, 0]
        t = np.arange(len(ux)) * f.attrs['dt'] / 2/pi * l0/c

    return x, y, z, ux, uy, uz, t


class TestSpeed(unittest.TestCase):
    
    def setUp(self) -> None:
        l0 = 8e-7
        omega0 = 2*pi*c / l0
        self.args = dummy_thomson()
        self.args_multiple = [np.stack([arg]*1000, axis=1) for arg in self.args[:-1]]
        self.args_multiple += [self.args[-1]]
        self.n = [1, 0, 0]
        nomega = 256
        gamma0 = 5.0

        self.omega_axis = np.linspace(0, 3, nomega) * 4*gamma0**2*omega0

    def test_cpu(self):
        get_lw_spectrum(*self.args, self.n, self.omega_axis, backend=None)
        tic = perf_counter_ns()
        for _ in range(100):
            get_lw_spectrum(*self.args, self.n, self.omega_axis, backend=None)
        dt = perf_counter_ns() - tic
        print(f"get_lw_spectrum on cpu: {dt/100*1e-6:.0f} ms/call")


    def test_cpu_multiple(self):
        get_lw_spectrum(*self.args_multiple, self.n, self.omega_axis)
        tic = perf_counter_ns()
        for _ in range(100):
            get_lw_spectrum(*self.args_multiple, self.n, self.omega_axis)
        dt = perf_counter_ns() - tic
        print(f"get_lw_spectrum on cpu: {dt/100*1e-6:.0f} ms/call")