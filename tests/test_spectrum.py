import unittest
import numpy as np
from lw import c, pi, get_lw_RE, get_RE_spectrum, get_lw_spectrum

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
        
        I = get_lw_spectrum(x, y, z, ux, uy, uz, t, n, omega_axis)
        print(I.max())
        self.assertAlmostEqual(I.max(), 1.7e-34, 2)