# pylint: disable=invalid-name
# pylint: disable=too-many-locals
# pylint: disable=protected-access

"""Tests for sphere object."""

import unittest
import numpy as np
import iadpython


class SimpleSphere(unittest.TestCase):
    """Creation and setting of sphere parameters."""

    def test_01_object_creation(self):
        """Simple sphere creation."""
        s = iadpython.sphere.Sphere(200, 20)
        np.testing.assert_allclose(s.d, 200, atol=1e-5)
        np.testing.assert_allclose(s.sample.d, 20, atol=1e-5)
        np.testing.assert_allclose(s.empty.d, 0, atol=1e-5)
        np.testing.assert_allclose(s.detector.d, 0, atol=1e-5)

    def test_02_portsize(self):
        """Test setting values."""
        s = iadpython.sphere.Sphere(200, 20)
        s.empty.d = 10
        s.detector.d = 5
        s.sample.d = 18
        np.testing.assert_allclose(s.d, 200, atol=1e-5)
        np.testing.assert_allclose(s.sample.d, 18, atol=1e-5)
        np.testing.assert_allclose(s.empty.d, 10, atol=1e-5)
        np.testing.assert_allclose(s.detector.d, 5, atol=1e-5)

    def test_03_cap(self):
        """Spherical cap calculations."""
        R = 100
        r = 10
        s = iadpython.sphere.Sphere(2 * R, 2 * r)
        acap = s.sample.cap_area() / (4 * np.pi * R**2)
        np.testing.assert_allclose(acap, s.sample.a, atol=1e-5)

        acap1 = np.pi * r**2 / (4 * np.pi * R**2)
        a_cap1 = s.sample.approx_relative_cap_area()
        np.testing.assert_allclose(acap1, a_cap1, atol=1e-5)


# class SphereGain(unittest.TestCase):
#     """Basic tests of gain relative to black sphere."""
# 
#     def test_01_gain(self):
#         """Gain calculations, r_wall=0."""
#         s = iadpython.sphere.Sphere(200, 25)
#         s.r_wall = 0
#         g = s.gain(0)
#         np.testing.assert_allclose(g, 1, atol=1e-5)
# 
#     def test_02_gain(self):
#         """Gain calculations, r_wall=1."""
#         s = iadpython.sphere.Sphere(200, 25)
#         g = s.gain(0, r_wall=1)
#         Asphere = 4 * np.pi * (200/2)**2
#         Asample = np.pi * (25/2)**2
#         gg = Asphere/Asample
#         np.testing.assert_allclose(g, gg, atol=1e-4)
# 
#     def test_03_gain(self):
#         """Gain calculations, r_wall=0."""
#         s = iadpython.sphere.Sphere(200, 25, d_empty=5, d_detector=10)
#         s.r_wall = 0
#         g = s.gain(0)
#         np.testing.assert_allclose(g, 1, atol=1e-5)
# 
#     def test_04_gain(self):
#         """Gain calculations, r_wall=1."""
#         s = iadpython.sphere.Sphere(200, 25, d_empty=5, d_detector=10)
#         s.r_wall = 1
#         g = s.gain(0)
#         gg = 1/(s.a_detector+s.a_empty+s.a_sample)
#         np.testing.assert_allclose(g, gg, atol=1e-5)


class SphereMultiplierSamplePortOnly(unittest.TestCase):
    """Relative increase in radiative flux on sphere walls."""

    def setUp(self):
        """Set up test conditions for SphereMultiplier tests."""
        R = 100
        d_sphere = 2 * R
        d_sample = 20
        self.sphere = iadpython.Sphere(d_sphere, d_sample, r_wall=0)

    def test_01_multiplier(self):
        """Scalar calculations no reflection from sample, black walls."""
        M = self.sphere.multiplier(UX1=1, URU=0)
        np.testing.assert_allclose(self.sphere.a_wall, M, atol=1e-5)

    def test_02_multiplier(self):
        """Scalar calculation 80% transmission."""
        M = self.sphere.multiplier(UX1=0.8, URU=0)
        np.testing.assert_allclose(0.8 * self.sphere.a_wall, M, atol=1e-5)

    def test_03_multiplier(self):
        """Scalar calculations total transmission with total sample reflection"""
        M = self.sphere.multiplier(UX1=1, URU=1)
        M1 = self.sphere.a_wall / (1 - self.sphere.a_wall * self.sphere.r_wall - self.sphere.sample.a)
        np.testing.assert_allclose(M, M1, atol=1e-5)

    def test_04_multiplier(self):
        """Array of r_wall values."""
        self.sphere.r_wall = np.linspace(0, 1, 4)
        M = self.sphere.multiplier(UX1=1, URU=0)
        mm = self.sphere.a_wall / (1 - self.sphere.a_wall * self.sphere.r_wall)
        np.testing.assert_allclose(mm, M, atol=1e-5)

    def test_05_multiplier(self):
        """Array of r_wall values."""
        self.sphere.r_wall = np.linspace(0, 1, 4)
        M = self.sphere.multiplier(UX1=0.8, URU=0)
        mm = 0.8 * self.sphere.a_wall / (1 - self.sphere.a_wall * self.sphere.r_wall)
        np.testing.assert_allclose(mm, M, atol=1e-5)

    def test_06_multiplier(self):
        """Array of r_wall values."""
        self.sphere.r_wall = np.linspace(0, 1, 4)
        M = self.sphere.multiplier(UX1=1, URU=1)
        M1 = self.sphere.a_wall / (1 - self.sphere.a_wall * self.sphere.r_wall - self.sphere.sample.a)
        M1[3] = np.inf
        np.testing.assert_allclose(M, M1, atol=1e-5)

class SphereMultiplier(unittest.TestCase):
    """Relative increase in radiative flux on sphere walls."""

    def setUp(self):
        """Set up test conditions for SphereMultiplier tests."""
        R = 100
        d_sphere = 2 * R
        d_sample = 20
        d_empty = 20
        d_detector = 10
        self.sphere = iadpython.Sphere(d_sphere, d_sample, 
                                       d_empty=d_empty, d_detector=d_detector, 
                                       r_detector=0.5, r_wall=0.95)

    def test_01_multiplier(self):
        """Scalar calculations no reflection from sample, black walls."""
        URU = 0
        UX1 = 1
        M = self.sphere.multiplier(UX1=UX1, URU=URU)
        denom = 1
        denom -= self.sphere._a_wall * self.sphere.r_wall
        denom -= self.sphere.sample.a * URU
        denom -= self.sphere.detector.a * self.sphere.detector.uru
        M1 = UX1 * self.sphere.a_wall / denom
        np.testing.assert_allclose(M1, M, atol=1e-5)

    def test_02_multiplier(self):
        """Scalar calculation 80% transmission."""
        URU = 0
        UX1 = 0.8
        M = self.sphere.multiplier(UX1=UX1, URU=URU)
        denom = 1
        denom -= self.sphere._a_wall * self.sphere.r_wall
        denom -= self.sphere.sample.a * URU
        denom -= self.sphere.detector.a * self.sphere.detector.uru
        M1 = UX1 * self.sphere.a_wall / denom
        np.testing.assert_allclose(M1, M, atol=1e-5)

    def test_03_multiplier(self):
        """Scalar calculations total transmission with total sample reflection"""
        URU = 1
        UX1 = 1
        M = self.sphere.multiplier(UX1=UX1, URU=URU)
        denom = 1
        denom -= self.sphere._a_wall * self.sphere.r_wall
        denom -= self.sphere.sample.a * URU
        denom -= self.sphere.detector.a * self.sphere.detector.uru
        M1 = UX1 * self.sphere.a_wall / denom
        np.testing.assert_allclose(M1, M, atol=1e-5)

    def test_04_multiplier(self):
        """Array of r_wall values."""
        self.sphere.r_wall = np.linspace(0, 1, 4)
        URU = 0
        UX1 = 1
        M = self.sphere.multiplier(UX1=UX1, URU=URU)
        denom = 1
        denom -= self.sphere._a_wall * self.sphere.r_wall
        denom -= self.sphere.sample.a * URU
        denom -= self.sphere.detector.a * self.sphere.detector.uru
        M1 = UX1 * self.sphere.a_wall / denom
        np.testing.assert_allclose(M1, M, atol=1e-5)

    def test_05_multiplier(self):
        """Array of r_wall values."""
        self.sphere.r_wall = np.linspace(0, 1, 4)
        URU = 0
        UX1 = 0.8
        M = self.sphere.multiplier(UX1=UX1, URU=URU)
        denom = 1
        denom -= self.sphere._a_wall * self.sphere.r_wall
        denom -= self.sphere.sample.a * URU
        denom -= self.sphere.detector.a * self.sphere.detector.uru
        M1 = UX1 * self.sphere.a_wall / denom
        np.testing.assert_allclose(M1, M, atol=1e-5)

    def test_06_multiplier(self):
        """Array of r_wall values."""
        self.sphere.r_wall = np.linspace(0, 1, 4)
        URU = 1
        UX1 = 1
        M = self.sphere.multiplier(UX1=UX1, URU=URU)
        denom = 1
        denom -= self.sphere._a_wall * self.sphere.r_wall
        denom -= self.sphere.sample.a * URU
        denom -= self.sphere.detector.a * self.sphere.detector.uru
        M1 = UX1 * self.sphere.a_wall / denom
        np.testing.assert_allclose(M1, M, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
