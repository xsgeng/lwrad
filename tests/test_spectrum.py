import unittest
import numpy as np
from lw import c, pi, get_lw_field, get_lw_spectrum

class TestSpectrum(unittest.TestCase):
    def test_thomson(self):
        import h5py

        l0 = 8e-7
        omega0 = 2*pi*c / l0

        with h5py.File("test.h5", "r") as f:
            x = f['x'][:, 0] / 2/pi * l0
            y = f['y'][:, 0] / 2/pi * l0
            z = f['z'][:, 0] / 2/pi * l0
            ux = f['px'][:, 0]
            uy = f['py'][:, 0]
            uz = f['pz'][:, 0]
            t = np.arange(len(ux)) * f.attrs['dt'] / 2/pi * l0/c

        nomega = 256
        gamma0 = 5.0

        omega_axis = np.linspace(0, 3, nomega) * 4*gamma0**2*omega0

        n = [1, 0, 0]
        t_ret, Ex, Ey, Ez = get_lw_field(x, y, z, ux, uy, uz, t, n)
        Ez_ft = get_lw_spectrum(Ez, t_ret, omega_axis)

        I = (Ez_ft.real**2 + Ez_ft.imag**2) 
        I *= 2
        
        self.assertAlmostEqual(I.max(), 1.7e-34, 2)