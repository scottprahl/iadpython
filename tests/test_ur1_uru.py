# pylint: disable=invalid-name

"""Tests for total flux calculations."""

import unittest
import numpy as np
import iadpython


class TestFinite(unittest.TestCase):
    """Finite layer in air."""

    def test_01_sandwich(self):
        """Anisotropic finite layer calculation."""
        s = iadpython.Sample(a=0.5, b=1, g=0.9, n=1.0, quad_pts=4)
        ur1, ut1, uru, utu = s.rt()
        ur1_true = 0.00585
        ut1_true = 0.59232
        uru_true = 0.01641
        utu_true = 0.42287
        np.testing.assert_allclose([ur1], [ur1_true], atol=1e-5)
        np.testing.assert_allclose([ut1], [ut1_true], atol=1e-5)
        np.testing.assert_allclose([uru], [uru_true], atol=1e-5)
        np.testing.assert_allclose([utu], [utu_true], atol=1e-5)

    def test_02_sandwich(self):
        """Anisotropic finite layer in air."""
        s = iadpython.Sample(a=0.5, b=1, g=0.9, n=1.4, n_above=1.0, n_below=1.0)
        ur1, ut1, uru, utu = s.rt()
        ur1_true = 0.03859
        ut1_true = 0.54038
        uru_true = 0.06527
        utu_true = 0.45887
        np.testing.assert_allclose([ur1], [ur1_true], atol=1e-5)
        np.testing.assert_allclose([ut1], [ut1_true], atol=1e-5)
        np.testing.assert_allclose([utu], [utu_true], atol=1e-5)
        np.testing.assert_allclose([uru], [uru_true], atol=1e-5)

    def test_03_sandwich(self):
        """Anisotropic finite layer with slides."""
        s = iadpython.Sample(a=0.5, b=1, g=0.9, n=1.4, n_above=1.5, n_below=1.5)
        ur1, ut1, uru, utu = s.rt()
        ur1_true = 0.05563
        ut1_true = 0.52571
        uru_true = 0.08472
        utu_true = 0.44368
        np.testing.assert_allclose([ur1], [ur1_true], atol=1e-5)
        np.testing.assert_allclose([uru], [uru_true], atol=1e-5)
        np.testing.assert_allclose([ut1], [ut1_true], atol=1e-5)
        np.testing.assert_allclose([utu], [utu_true], atol=1e-5)

    def test_04_semi_infinite(self):
        """Anisotropic infinite layer with slides."""
        s = iadpython.Sample(a=0.5, b=np.inf, g=0.9, n=1.4, n_above=1.5)
        ur1, ut1, uru, utu = s.rt()
        ur1_true = 0.04255
        ut1_true = 0.00000
        uru_true = 0.07001
        utu_true = 0.00000
        np.testing.assert_allclose([ur1], [ur1_true], atol=1e-5)
        np.testing.assert_allclose([uru], [uru_true], atol=1e-5)
        np.testing.assert_allclose([ut1], [ut1_true], atol=1e-5)
        np.testing.assert_allclose([utu], [utu_true], atol=1e-5)

    def test_05_semi_infinite(self):
        """Anisotropic infinite layer with slides and arrays."""
        s = iadpython.Sample(a=[0.0, 0.5], b=np.inf, g=0.9, n=1.4, n_above=1.5)
        ur1, ut1, uru, utu = s.rt()
        ur1_true = [0.04110, 0.04255]
        ut1_true = [0.00000, 0.00000]
        uru_true = [0.06735, 0.07001]
        utu_true = [0.00000, 0.00000]
        np.testing.assert_allclose(ur1, ur1_true, atol=1e-5)
        np.testing.assert_allclose(uru, uru_true, atol=1e-5)
        np.testing.assert_allclose(ut1, ut1_true, atol=1e-5)
        np.testing.assert_allclose(utu, utu_true, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
