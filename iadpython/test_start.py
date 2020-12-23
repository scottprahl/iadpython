# pylint: disable=invalid-name
# pylint: disable=bad-whitespace
# pylint: disable=no-self-use

"""Tests for initial layer thickness Function."""

import unittest
import numpy as np
from nose.plugins.attrib import attr
import iadpython

def wip(f):
    """
    Only test functions with @wip decorator.

    Add the @wip decorator before functions that are works-in-progress.
    `nosetests -a wip test_combo.py` will test only those with @wip decorator.
    """
    return attr('wip')(f)

class A_thin(unittest.TestCase):
    """Starting layer calculations."""

    def test_01_thin(self):
        """Isotropic finite layer."""
        s = iadpython.ad.Sample(a=1, b=1, g=0.0, n=1, quad_pts=4)
        b_min = iadpython.starting_thickness(s)
        np.testing.assert_approx_equal(b_min, 0.0625)

    def test_02_thin(self):
        """Isotropic thick layer."""
        s = iadpython.ad.Sample(a=1, b=100, g=0.0, n=1, quad_pts=4)
        b_min = iadpython.starting_thickness(s)
        np.testing.assert_approx_equal(b_min, 0.048828125)

    def test_03_thin(self):
        """Isotropic semi-infinite layer."""
        s = iadpython.ad.Sample(a=1, b=np.inf, g=0.0, n=1, quad_pts=4)
        b_min = iadpython.starting_thickness(s)
        np.testing.assert_approx_equal(b_min, 0.04429397)

    def test_04_thin(self):
        """Isotropic semi-infinite layer."""
        s = iadpython.ad.Sample(a=1, b=0, g=0.0, n=1, quad_pts=4)
        b_min = iadpython.starting_thickness(s)
        np.testing.assert_approx_equal(b_min, 0.0)

    def test_05_thin(self):
        """Anisotropic finite layer."""
        s = iadpython.ad.Sample(a=1, b=1, g=0.9, n=1, quad_pts=4)
        b_min = iadpython.starting_thickness(s)
        np.testing.assert_approx_equal(b_min, 0.08597500)

    def test_06_thin(self):
        """Anisotropic thick layer."""
        s = iadpython.ad.Sample(a=1, b=100, g=0.9, n=1, quad_pts=4)
        b_min = iadpython.starting_thickness(s)
        np.testing.assert_approx_equal(b_min, 0.06716797)

    def test_07_thin(self):
        """Anisotropic semi-infinite layer."""
        s = iadpython.ad.Sample(a=1, b=np.inf, g=0.9, n=1, quad_pts=4)
        b_min = iadpython.starting_thickness(s)
        np.testing.assert_approx_equal(b_min, 0.04429397)

    def test_08_thin(self):
        """Anisotropic zero-thickness layer."""
        s = iadpython.ad.Sample(a=1, b=0, g=0.9, n=1, quad_pts=4)
        b_min = iadpython.starting_thickness(s)
        np.testing.assert_approx_equal(b_min, 0.0)

class B_igi(unittest.TestCase):
    """IGI layer initializations."""

    def test_01_igi(self):
        """IGI initialization with isotropic scattering."""
        s = iadpython.ad.Sample(a=1, b=100, g=0.0, n=1, quad_pts=4)
        rr, tt = iadpython.start.igi(s)
        np.testing.assert_approx_equal(s.b_thinnest, 0.048828125)

        r = np.array([[ 1.55547, 0.33652, 0.17494, 0.13780],
                      [ 0.33652, 0.07281, 0.03785, 0.02981],
                      [ 0.17494, 0.03785, 0.01968, 0.01550],
                      [ 0.13780, 0.02981, 0.01550, 0.01221]])

        t = np.array([[13.04576, 0.33652, 0.17494, 0.13780],
                      [ 0.33652, 2.84330, 0.03785, 0.02981],
                      [ 0.17494, 0.03785, 1.83038, 0.01550],
                      [ 0.13780, 0.02981, 0.01550, 7.62158]])

        np.testing.assert_allclose(r, rr, atol=1e-5)
        np.testing.assert_allclose(t, tt, atol=1e-5)

    def test_02_igi(self):
        """IGI initialization with anisotropic scattering."""
        s = iadpython.ad.Sample(a=1, b=100, g=0.9, quad_pts=4)
        rr, tt = iadpython.start.igi(s)
        np.testing.assert_approx_equal(s.b_thinnest, 0.06716797)

        r = np.array([[ 3.19060, 0.51300, 0.09360,-0.01636],
                      [ 0.51300, 0.04916, 0.00524, 0.00941],
                      [ 0.09360, 0.00524, 0.00250, 0.00486],
                      [-0.01636, 0.00941, 0.00486,-0.00628]])

        t = np.array([[ 9.56148, 0.66419, 0.16129,-0.01868],
                      [ 0.66419, 2.80843, 0.07395, 0.02700],
                      [ 0.16129, 0.07395, 1.83985, 0.07886],
                      [-0.01868, 0.02700, 0.07886, 7.57767]])

        np.testing.assert_allclose(r, rr, atol=1e-5)
        np.testing.assert_allclose(t, tt, atol=1e-5)

class C_diamond(unittest.TestCase):
    """Diamond layer initializations."""

    def test_0_diamond(self):
        """Diamond initialization with isotropic scattering."""
        s = iadpython.ad.Sample(a=1, b=100, g=0.0, quad_pts=4)
        rr, tt = iadpython.start.diamond(s)

        r = np.array([[1.04004,0.27087,0.14472,0.11473],
                      [0.27087,0.07055,0.03769,0.02988],
                      [0.14472,0.03769,0.02014,0.01596],
                      [0.11473,0.02988,0.01596,0.01266]])

        t = np.array([[15.57900,0.27087,0.14472,0.11473],
                      [0.27087,2.86214,0.03769,0.02988],
                      [0.14472,0.03769,1.83444,0.01596],
                      [0.11473,0.02988,0.01596,7.63134]])

        np.testing.assert_allclose(r, rr, atol=1e-5)
        np.testing.assert_allclose(t, tt, atol=1e-5)

    def test_02_diamond(self):
        """Diamond initialization with anisotropic scattering."""
        s = iadpython.ad.Sample(a=1, b=100, g=0.9, quad_pts=4)

        rr, tt = iadpython.start.diamond(s)

        r = np.array([[ 1.92637,0.40140,0.08092,-0.00869],
                      [ 0.40140,0.05438,0.00773,0.00888],
                      [ 0.08092,0.00773,0.00306,0.00473],
                      [-0.00869,0.00888,0.00473,-0.00569]])

        t = np.array([[13.55020,0.50578,0.13009,-0.00913],
                      [ 0.50578,2.83738,0.07117,0.02622],
                      [ 0.13009,0.07117,1.84366,0.07534],
                      [-0.00913,0.02622,0.07534,7.59016]])

        np.testing.assert_allclose(r, rr, atol=1e-5)
        np.testing.assert_allclose(t, tt, atol=1e-5)

    def test_03_diamond(self):
        """Diamond initialization with isotropic scattering and n=1.5."""
        s = iadpython.ad.Sample(a=1, b=100, g=0.0, n=1.5, quad_pts=4)

        rr, tt = iadpython.start.diamond(s)

        r = np.array([[0.65936,0.21369,0.15477,0.12972],
                      [0.21369,0.06926,0.05016,0.04204],
                      [0.15477,0.05016,0.03633,0.03045],
                      [0.12972,0.04204,0.03045,0.02552]])

        t = np.array([[5.14582,0.21369,0.15477,0.12972],
                      [0.21369,2.00149,0.05016,0.04204],
                      [0.15477,0.05016,2.83938,0.03045],
                      [0.12972,0.04204,0.03045,7.14833]])

        np.testing.assert_allclose(r, rr, atol=1e-5)
        np.testing.assert_allclose(t, tt, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
