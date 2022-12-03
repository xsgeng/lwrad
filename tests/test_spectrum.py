import unittest
import numpy as np
from lw import c, pi, get_lw_RE, get_RE_spectrum, get_lw_spectrum

class TestSpectrum(unittest.TestCase):
    l0 = 8e-7
    omega0 = 2*pi*c / l0

    def dummy_thomson(self):
        import h5py

        l0 = self.l0
        with h5py.File("tests/test.h5", "r") as f:
            x = f['x'][:, 0] / 2/pi * l0
            y = f['y'][:, 0] / 2/pi * l0
            z = f['z'][:, 0] / 2/pi * l0
            ux = f['px'][:, 0]
            uy = f['py'][:, 0]
            uz = f['pz'][:, 0]
            t = np.arange(len(ux)) * f.attrs['dt'] / 2/pi * l0/c

        return x, y, z, ux, uy, uz, t

    def test_thomson(self):
        omega0 = self.omega0
        nomega = 256
        gamma0 = 5.0

        omega_axis = np.linspace(0, 3, nomega) * 4*gamma0**2*omega0

        n = [1, 0, 0]
        
        I = get_lw_spectrum(*self.dummy_thomson(), n, omega_axis)
        self.assertAlmostEqual(I.max(), 1.7e-34, 2)


    def test_nan(self):
        args = [np.concatenate(([np.nan]*10, x, [np.nan]*10) ) for x in self.dummy_thomson()]

        omega0 = self.omega0
        nomega = 256
        gamma0 = 5.0
        omega_axis = np.linspace(0, 3, nomega) * 4*gamma0**2*omega0

        n = [1, 0, 0]
        # test warn once
        with self.assertWarns(UserWarning):
            I = get_lw_spectrum(*args, n, omega_axis)
        
        self.assertNotIn(np.nan, I)
        self.assertAlmostEqual(I.max(), 1.7e-34, 2)
