# pylint: disable=invalid-name
# pylint: disable=bad-whitespace
# pylint: disable=no-self-use
# pylint: disable=too-many-statements

"""Tests for Boundary reflection."""

import unittest
import numpy as np
import iadpython.start
import iadpython.boundary

class boundary(unittest.TestCase):
    """Boundary layer calculations."""

    def test_01_boundary(self):
        """Test initialization."""
        n_glass = 1.5
        n_slab= 1.3
        s = iadpython.start.Slab(n=n_slab, n_above=n_glass, n_below=n_glass)
        m = iadpython.start.Method(s)
        r, t = iadpython.boundary.boundary_RT(n_slab, n_glass, 1.0, 0, m)
        rr = np.array([0.08628,0.32200,0.03502,0.00807])
        tt = np.array([0.00000,0.00000,0.91484,0.95530])
        np.testing.assert_allclose(r, rr, atol=1e-5)
        np.testing.assert_allclose(t, tt, atol=1e-5)

    def test_02_boundary(self):
        """Test initialization."""
        n_glass = 1.5
        n_slab= 1.3
        s = iadpython.start.Slab(n=n_slab, n_above=n_glass, n_below=n_glass)
        m = iadpython.start.Method(s)
        r, t = iadpython.boundary.boundary_RT(1.0, n_glass, n_slab, 0, m)
        rr = np.array([0.041160, 0.030208, 0.021036, 0.008069])
        tt = np.array([0.522944, 0.906186, 0.948843, 0.955297])
        np.testing.assert_allclose(r, rr, atol=1e-5)
        np.testing.assert_allclose(t, tt, atol=1e-5)

    def test_03_boundary(self):
        """Test initialization."""
        s = iadpython.start.Slab(n=1.3, n_above=1.5, n_below=1.5)
        m = iadpython.start.Method(s)
        r01, r10, t01, t10 = iadpython.boundary.init_boundary(s,m,top=True)
        rr01 = np.array([0.041160, 0.030208, 0.021036, 0.008069])
        rr10 = np.array([0.08628,0.32200,0.03502,0.00807])
        tt01 = np.array([0.522944, 0.906186, 0.948843, 0.955297])
        tt10 = np.array([0.00000,0.00000,0.91484,0.95530])
        np.testing.assert_allclose(r01, rr01, atol=1e-5)
        np.testing.assert_allclose(t01, tt01, atol=1e-5)
        np.testing.assert_allclose(r10, rr10, atol=1e-5)
        np.testing.assert_allclose(t10, tt10, atol=1e-5)

    def test_04_boundary(self):
        """Test initialization."""
        s = iadpython.start.Slab(n=1.3, n_above=1.5, n_below=1.5)
        m = iadpython.start.Method(s)
        r01, r10, t01, t10 = iadpython.boundary.init_boundary(s,m,top=False)
        rr10 = np.array([0.041160, 0.030208, 0.021036, 0.008069])
        rr01 = np.array([0.08628,0.32200,0.03502,0.00807])
        tt10 = np.array([0.522944, 0.906186, 0.948843, 0.955297])
        tt01 = np.array([0.00000,0.00000,0.91484,0.95530])
        np.testing.assert_allclose(r01, rr01, atol=1e-5)
        np.testing.assert_allclose(t01, tt01, atol=1e-5)
        np.testing.assert_allclose(r10, rr10, atol=1e-5)
        np.testing.assert_allclose(t10, tt10, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
