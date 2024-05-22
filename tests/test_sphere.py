"""Tests for sphere object."""

import unittest
import numpy as np
import iadpython

module_path = iadpython.__file__
print("Module imported from:", module_path)


class SimpleSphere(unittest.TestCase):
    """Creation and setting of sphere parameters."""

    def test_01_object_creation(self):
        """Simple sphere creation."""
        s = iadpython.Sphere(200, 20)
        np.testing.assert_allclose(s.d, 200, atol=1e-5)
        np.testing.assert_allclose(s.sample.d, 20, atol=1e-5)
        np.testing.assert_allclose(s.third.d, 0, atol=1e-5)
        np.testing.assert_allclose(s.detector.d, 0, atol=1e-5)

    def test_02_portsize(self):
        """Test setting values."""
        s = iadpython.Sphere(200, 20)
        s.third.d = 10
        s.detector.d = 5
        s.sample.d = 18
        np.testing.assert_allclose(s.d, 200, atol=1e-5)
        np.testing.assert_allclose(s.sample.d, 18, atol=1e-5)
        np.testing.assert_allclose(s.third.d, 10, atol=1e-5)
        np.testing.assert_allclose(s.detector.d, 5, atol=1e-5)

    def test_03_cap(self):
        """Spherical cap calculations."""
        R = 100
        r = 10
        s = iadpython.Sphere(2 * R, 2 * r)
        acap = s.sample.cap_area_exact() / (4 * np.pi * R**2)
        np.testing.assert_allclose(acap, s.sample.a, atol=1e-5)

        acap1 = np.pi * r**2 / (4 * np.pi * R**2)
        a_cap1 = s.sample.relative_cap_area()
        np.testing.assert_allclose(acap1, a_cap1, atol=1e-5)


class SphereGain(unittest.TestCase):
    """Basic tests of gain relative to black sphere."""

    def test_01_gain(self):
        """Gain calculations, r_wall=0."""
        d_sphere = 200
        d_sample = 25
        s = iadpython.Sphere(d_sphere, d_sample, r_wall=0)
        g = s.gain(sample_uru=0)
        np.testing.assert_allclose(g, 1, atol=1e-5)

    def test_02_gain(self):
        """Gain calculations, r_wall=1."""
        d_sphere = 200
        d_sample = 25
        s = iadpython.Sphere(d_sphere, d_sample, r_wall=1)
        g = s.gain(sample_uru=0)
        p = iadpython.Port(s, d_sample)
        Asphere = 4 * np.pi * (d_sphere / 2)**2
        gg = Asphere / p.cap_area()
        np.testing.assert_allclose(g, gg, atol=1e-4)

    def test_03_gain(self):
        """Gain calculations with third and detector, r_wall=0."""
        d_sphere = 200
        d_sample = 25
        s = iadpython.Sphere(d_sphere, d_sample, d_third=5, d_detector=10, r_wall=0)
        g = s.gain(sample_uru=0)
        np.testing.assert_allclose(g, 1, atol=1e-5)

    def test_04_gain(self):
        """Gain calculations with third and detector, r_wall=1."""
        d_sphere = 200
        d_sample = 25
        s = iadpython.Sphere(d_sphere, d_sample, d_third=5, d_detector=10, r_wall=1)
        g = s.gain(sample_uru=0)
        gg = 1 / (s.detector.a + s.third.a + s.sample.a)
        np.testing.assert_allclose(g, gg, atol=1e-5)

    def test_05_gain(self):
        """Array of r_wall values."""
        r_wall = np.linspace(0, 1, 4)
        d_sphere = 200
        d_sample = 25
        s = iadpython.Sphere(d_sphere, d_sample, d_third=5, d_detector=10, r_wall=r_wall)
        g = s.gain(sample_uru=0)
        np.testing.assert_allclose(len(r_wall), len(g), atol=1e-5)

class DoubleSphere(unittest.TestCase):
    """Creation and setting of a single sphere used parameters."""

    def test_01_double_photon(self):
        """One photon test for double sphere configuration."""
        s = iadpython.Sphere(50, 10, r_wall=1)
        detected, transmitted, _ = s.do_one_photon(double=True)
        np.testing.assert_allclose(detected, 0, atol=1e-5)
        np.testing.assert_allclose(transmitted, 1, atol=1e-5)

    def test_02_double_photon(self):
        """One photon test for double sphere configuration."""
        s = iadpython.Sphere(50, 10, r_wall=1)
        detected, transmitted, _ = s.do_one_photon(double=True, weight=0.5)
        np.testing.assert_allclose(detected, 0, atol=1e-5)
        np.testing.assert_allclose(transmitted, 0.5, atol=1e-5)

    def test_01_double_N_photon(self):
        """Half the light should be detected."""
        N=10000
        d_sample = 10
        s = iadpython.Sphere(50, d_sample, r_wall=1, d_detector=d_sample)
        ave, stderr = s.do_N_photons(N, double=True)
        np.testing.assert_allclose(ave, 0.5, atol=0.03)


if __name__ == '__main__':
    unittest.main()
