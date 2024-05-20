# pylint: disable=protected-access

"""Tests for sphere object."""

import unittest
import iadpython

class TestDoubleSphere(unittest.TestCase):
    """Simple Double Sphere Tests."""

    def setUp(self):
        """Setup for all these tests."""
        r_sphere = iadpython.Sphere(100, 30, d_detector=10, r_wall=1)
        t_sphere = iadpython.Sphere(100, 30, d_detector=10, r_wall=1)
        self.double = iadpython.DoubleSphere(r_sphere, t_sphere)
        self.ur1 = 0
        self.uru = 0
        self.ut1 = 1
        self.utu = 1

    def test_init(self):
        """Make sure basics work."""
        self.assertEqual(self.double.ur1, 0.0)
        self.assertEqual(self.double.ut1, 1.0)
        self.assertEqual(self.double.uru, 0.0)
        self.assertEqual(self.double.utu, 1.0)

    def test_str(self):
        """Test string describing the object."""
        output = str(self.double)
        self.assertIn('Reflection Sphere', output)
        self.assertIn('Transmission Sphere', output)
        self.assertIn('ur1 =   0.000', output)
        self.assertIn('ut1 =   1.000', output)
        self.assertIn('uru =   0.000', output)
        self.assertIn('utu =   1.000', output)

    def test_no_sample(self):
        """Light passes unhindered between spheres."""
        self.double.ur1 = 0
        self.double.ut1 = 1
        self.double.uru = self.double.ur1
        self.double.utu = self.double.ut1
        N = 1000
        r_total = 0
        t_total = 0
#        print("no sample")
        for _ in range(N):
            r_detected, t_detected = self.double.do_one_photon()
            r_total += r_detected
            t_total += t_detected
#            print(r_total, t_total)

        self.assertAlmostEqual(r_total/N, 0.5, places=1)
        self.assertAlmostEqual(t_total/N, 0.5, places=1)

    def test_no_sample_N(self):
        """Light passes unhindered between spheres."""
        self.double.ur1 = 0
        self.double.ut1 = 1
        self.double.uru = self.double.ur1
        self.double.utu = self.double.ut1
        N = 1000
        r, _, t, _ = self.double.do_N_photons(N)
        self.assertAlmostEqual(r, 0.5, places=1)
        self.assertAlmostEqual(t, 0.5, places=1)

    def test_mirror_sample(self):
        """Light passes unhindered between spheres."""
        self.double.ur1 = 1
        self.double.ut1 = 0
        self.double.uru = self.double.ur1
        self.double.utu = self.double.ut1

        N = 10
        r_total = 0
        t_total = 0
        for _ in range(N):
            r_detected, t_detected = self.double.do_one_photon()
            r_total += r_detected
            t_total += t_detected
        self.assertAlmostEqual(r_total/N, 1.0, places=5)
        self.assertAlmostEqual(t_total/N, 0.0, places=5)

    def test_mirror_sample_N(self):
        """Light passes unhindered between spheres."""
        self.double.ur1 = 1
        self.double.ut1 = 0
        self.double.uru = self.double.ur1
        self.double.utu = self.double.ut1

        N = 10
        r, _, t, _ = self.double.do_N_photons(N)
        self.assertAlmostEqual(r, 1.0, places=5)
        self.assertAlmostEqual(t, 0.0, places=5)

if __name__ == '__main__':
    unittest.main()
