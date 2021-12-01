# pylint: disable=invalid-name
# pylint: disable=no-self-use
# pylint: disable=consider-using-f-string

"""Tests for C-library Inverse Adding-Doubling."""

import unittest
from iadpython import iadc


class basic_forward(unittest.TestCase):
    """Forward adding-doubling calculations."""

    def test_01_thick_non_scattering(self):
        """Thick non-scattering."""
        ur1, ut1, uru, utu = iadc.rt(1.0, 1.0, 0.0, 100000.0, 0.0)
        self.assertAlmostEqual(ur1, 0.00000, delta=0.0001)
        self.assertAlmostEqual(ut1, 0.00000, delta=0.0001)
        self.assertAlmostEqual(uru, 0.00000, delta=0.0001)
        self.assertAlmostEqual(utu, 0.00000, delta=0.0001)

    def test_02_thick(self):
        """Thick scattering."""
        ur1, ut1, uru, utu = iadc.rt(1.0, 1.0, 0.8, 100000.0, 0.0)
        self.assertAlmostEqual(ur1, 0.28525, delta=0.0001)
        self.assertAlmostEqual(ut1, 0.00000, delta=0.0001)
        self.assertAlmostEqual(uru, 0.34187, delta=0.0001)
        self.assertAlmostEqual(utu, 0.00000, delta=0.0001)

    def test_03_thick_non_absorbing(self):
        """Thick non-absorbing."""
        ur1, ut1, uru, utu = iadc.rt(1.0, 1.0, 1.0, 100000.0, 0.0)
        self.assertAlmostEqual(ur1, 1.0000, delta=0.0001)
        self.assertAlmostEqual(ut1, 0.0000, delta=0.0001)
        self.assertAlmostEqual(uru, 1.0000, delta=0.0001)
        self.assertAlmostEqual(utu, 0.0000, delta=0.0001)

    def test_04_finite(self):
        """Finite isotropic scattering."""
        ur1, ut1, uru, utu = iadc.rt(1.0, 1.0, 0.8, 1.0, 0.0)
        self.assertAlmostEqual(ur1, 0.21085, delta=0.0001)
        self.assertAlmostEqual(ut1, 0.54140, delta=0.0001)
        self.assertAlmostEqual(uru, 0.28015, delta=0.0001)
        self.assertAlmostEqual(utu, 0.41624, delta=0.0001)

    def test_05_finite_anisotropic(self):
        """Finite anisotropic scattering."""
        ur1, ut1, uru, utu = iadc.rt(1.0, 1.0, 0.8, 1.0, 0.8)
        self.assertAlmostEqual(ur1, 0.03041, delta=0.0001)
        self.assertAlmostEqual(ut1, 0.76388, delta=0.0001)
        self.assertAlmostEqual(uru, 0.08416, delta=0.0001)
        self.assertAlmostEqual(utu, 0.61111, delta=0.0001)


class matched_sample(unittest.TestCase):
    """Simple inverse adding-doubling calculations."""

    def test_01_matched_sample(self):
        """No scattering case."""
        UR1 = 0
        a, b, g, _ = iadc.basic_rt_inverse(1.0, 1.0, UR1, 0, 0)
        self.assertAlmostEqual(a, 0, delta=0.0001)
        ur1x, _, _, _ = iadc.rt(1.0, 1.0, a, b, g)
        self.assertAlmostEqual(UR1, ur1x, delta=0.002)

    def test_02_matched_sample(self):
        """No absorption case."""
        UR1 = 0.99999
        a, b, g, _ = iadc.basic_rt_inverse(1.0, 1.0, UR1, 0, 0)
        self.assertAlmostEqual(a, 1, delta=0.0001)
        ur1x, _, _, _ = iadc.rt(1.0, 1.0, a, b, g)
        self.assertAlmostEqual(UR1, ur1x, delta=0.002)

    def test_03_matched_sample(self):
        """Scattering and absorption case, no transmission."""
        UR1 = 0.4
        a, b, g, _ = iadc.basic_rt_inverse(1.0, 1.0, UR1, 0, 0)
        self.assertAlmostEqual(a, 0.8915, delta=0.0001)
        ur1x, _, _, _ = iadc.rt(1.0, 1.0, a, b, g)
        self.assertAlmostEqual(UR1, ur1x, delta=0.002)

    def test_04_matched_sample(self):
        """Scattering and absorption case with transmission."""
        UR1 = 0.4
        UT1 = 0.1
        a, b, g, _ = iadc.basic_rt_inverse(1.0, 1.0, UR1, UT1, 0)
        self.assertAlmostEqual(a, 0.8938, delta=0.0001)
        self.assertAlmostEqual(b, 4.3978, delta=0.0001)
        ur1x, ut1x, _, _ = iadc.rt(1.0, 1.0, a, b, g)
        self.assertAlmostEqual(UR1, ur1x, delta=0.002)
        self.assertAlmostEqual(UT1, ut1x, delta=0.002)

    def test_05_matched_sample(self):
        """Reflection, transmission, and unscattered transmission."""
        UR1 = 0.4
        UT1 = 0.1
        tc = 0.002
        a, b, g, _ = iadc.basic_rt_inverse(1.0, 1.0, UR1, UT1, tc)
        self.assertAlmostEqual(a, 0.9301, delta=0.0001)
        self.assertAlmostEqual(b, 6.2146, delta=0.0001)
        self.assertAlmostEqual(g, 0.3116, delta=0.0001)
        ur1x, ut1x, _, _ = iadc.rt(1.0, 1.0, a, b, g)
        self.assertAlmostEqual(UR1, ur1x, delta=0.002)
        self.assertAlmostEqual(UT1, ut1x, delta=0.002)

    def test_06_matched_sample(self):
        """Reflection, transmission, and unscattered transmission."""
        UR1 = 0.4
        UT1 = 0.1
        tc = 0.049787
        a, b, g, _ = iadc.basic_rt_inverse(1.0, 1.0, UR1, UT1, tc)
        self.assertAlmostEqual(a, 0.7926, delta=0.0001)
        self.assertAlmostEqual(b, 3.0000, delta=0.0001)
        self.assertAlmostEqual(g, -0.6694, delta=0.0001)
        ur1x, ut1x, _, _ = iadc.rt(1.0, 1.0, a, b, g)
        self.assertAlmostEqual(UR1, ur1x, delta=0.002)
        self.assertAlmostEqual(UT1, ut1x, delta=0.002)

class mismatched_sample_no_slides(unittest.TestCase):
    """Slab with index of refraction equal to 1.5."""

    def test_01_mismatched_sample_no_slides(self):
        """No scattering case."""
        UR1 = 0.04
        a, b, g, _ = iadc.rt_inverse(1.5, 1.0, UR1, 0, 0)
        print("%8.5f %8.5f %8.5f" % (a,b,g))
        ur1x, _, _, _ = iadc.rt(1.5, 1.0, a, b, g)
        self.assertAlmostEqual(UR1, ur1x, delta=0.002)

    def test_02_mismatched_sample_no_slides(self):
        """No absorption case."""
        UR1 = 0.99999
        a, b, g, _ = iadc.rt_inverse(1.5, 1.0, UR1, 0, 0)
        ur1x, _, _, _ = iadc.rt(1.5, 1.0, a, b, g)
        self.assertAlmostEqual(UR1, ur1x, delta=0.002)

    def test_03_mismatched_sample_no_slides(self):
        """Scattering and absorption case, no transmission."""
        UR1 = 0.4
        a, b, g, _ = iadc.rt_inverse(1.5, 1.0, UR1, 0, 0)
        ur1x, _, _, _ = iadc.rt(1.5, 1.0, a, b, g)
        self.assertAlmostEqual(UR1, ur1x, delta=0.002)

    def test_04_mismatched_sample_no_slides(self):
        """Scattering and absorption case with transmission."""
        UR1 = 0.4
        UT1 = 0.1
        a, b, g, error = iadc.rt_inverse(1.5, 1.0, UR1, UT1, 0)
        print("%8.5f %8.5f %8.5f" % (a,b,g))
        print(error)
        ur1x, ut1x, _, _ = iadc.rt(1.5, 1.0, a, b, g)
        self.assertAlmostEqual(UR1, ur1x, delta=0.002)
        self.assertAlmostEqual(UT1, ut1x, delta=0.002)

    def test_04a_mismatched_sample_no_slides(self):
        """Scattering and absorption case with transmission."""
        UR1 = 0.25312
        UT1 = 0.28882
        a, b, g, error = iadc.rt_inverse(1.5, 1.0, UR1, UT1, 0)
        print("%8.5f %8.5f %8.5f" % (a,b,g))
        print(error)
        ur1x, ut1x, _, _ = iadc.rt(1.5, 1.0, a, b, g)
        self.assertAlmostEqual(UR1, ur1x, delta=0.002)
        self.assertAlmostEqual(UT1, ut1x, delta=0.002)

    def test_05_mismatched_sample_no_slides(self):
        """Reflection, transmission, and unscattered transmission."""
        UR1 = 0.04617
        UT1 = 0.29960
        tc = 0.12473
        a, b, g, _ = iadc.rt_inverse(1.5, 1.0, UR1, UT1, tc)
        ur1x, ut1x, _, _ = iadc.rt(1.5, 1.0, a, b, g)
        _, tcx, _, _ = iadc.rt_unscattered(1.5, 1.0, a, b, g)
        self.assertAlmostEqual(UR1, ur1x, delta=0.003)
        self.assertAlmostEqual(UT1, ut1x, delta=0.003)
        self.assertAlmostEqual(tc, tcx, delta=0.003)


class sample_with_slides(unittest.TestCase):
    """Slab with n=1.4 and slides=1.5."""

    def test_01_mismatched_sample_and_slides(self):
        """No scattering case."""
        UR1 = 0.07894
        a, b, g, _ = iadc.rt_inverse(1.4, 1.5, UR1, 0, 0)
        ur1x, _, _, _ = iadc.rt(1.4, 1.5, a, b, g)
        self.assertAlmostEqual(UR1, ur1x, delta=0.002)

    def test_02_mismatched_sample_and_slides(self):
        """No absorption case."""
        UR1 = 0.99999
        a, b, g, _ = iadc.rt_inverse(1.4, 1.5, UR1, 0, 0)
        ur1x, _, _, _ = iadc.rt(1.4, 1.5, a, b, g)
        self.assertAlmostEqual(UR1, ur1x, delta=0.002)

    def test_03_mismatched_sample_and_slides(self):
        """Scattering and absorption case, no transmission."""
        UR1 = 0.4
        a, b, g, _ = iadc.rt_inverse(1.4, 1.5, UR1, 0, 0)
        ur1x, _, _, _ = iadc.rt(1.4, 1.5, a, b, g)
        self.assertAlmostEqual(UR1, ur1x, delta=0.002)

    def test_04_mismatched_sample_and_slides(self):
        """Scattering and absorption case with transmission."""
        UR1 = 0.4
        UT1 = 0.1
        a, b, g, error = iadc.rt_inverse(1.4, 1.5, UR1, UT1, 0)
        print("%8.5f %8.5f %8.5f" % (a,b,g))
        print(error)
        ur1x, ut1x, _, _ = iadc.rt(1.4, 1.5, a, b, g)
        self.assertAlmostEqual(UR1, ur1x, delta=0.002)
        self.assertAlmostEqual(UT1, ut1x, delta=0.002)

    def test_05_mismatched_sample_and_slides(self):
        """Reflection, transmission, and unscattered transmission."""
        UR1 = 0.04756
        UT1 = 0.30214
        tc = 0.12444
        a, b, g, _ = iadc.rt_inverse(1.4, 1.5, UR1, UT1, tc)
        ur1x, ut1x, _, _ = iadc.rt(1.4, 1.5, a, b, g)
        self.assertAlmostEqual(UR1, ur1x, delta=0.002)
        self.assertAlmostEqual(UT1, ut1x, delta=0.002)


# class top_slide(unittest.TestCase):
#     @echo "********* One slide on top ************"
#     ./iad -V 0 -r 0.4 -n 1.4 -N 1.5 -G t
#     ./iad -V 0 -r 0.4 -t 0.1 -n 1.4 -N 1.5 -G t
#     ./iad -V 0 -r 0.4 -t 0.1 -u 0.002 -n 1.4 -N 1.5 -G t
#     ./iad -V 0 -r 0.4 -t 0.1 -u 0.045884 -n 1.4 -N 1.5 -G t
#
# class bottom_slide(unittest.TestCase):
#     @echo "********* One slide on bottom ************"
#     ./iad -V 0 -r 0.4 -n 1.4 -N 1.5 -G b
#     ./iad -V 0 -r 0.4 -t 0.1 -n 1.4 -N 1.5 -G b
#     ./iad -V 0 -r 0.4 -t 0.1 -u 0.002 -n 1.4 -N 1.5 -G b
#     ./iad -V 0 -r 0.4 -t 0.1 -u 0.045884 -n 1.4 -N 1.5 -G b
#
# class absorbing_slide(unittest.TestCase):
#     @echo "********* Absorbing Slide Tests ***********"
#     ./iad -V 0 -r 0.0000000 -t 0.135335 -E 0.5
#     ./iad -V 0 -r 0.0249268 -t 0.155858 -E 0.5
#     ./iad -V 0 -r 0.0399334 -t 0.124727 -E 0.5 -n 1.5 -N 1.5
#     ./iad -V 0 -r 0.0520462 -t 0.134587 -E 0.5 -n 1.5 -N 1.5
#
# class fixed_anisotropy(unittest.TestCase):
#     @echo "********* Constrain g ************"
#     ./iad -V 0 -r 0.4        -g 0.9
#     ./iad -V 0 -r 0.4 -t 0.1 -g 0.9
#     ./iad -V 0 -r 0.4        -g 0.9 -n 1.5
#     ./iad -V 0 -r 0.4 -t 0.1 -g 0.9 -n 1.5
#     ./iad -V 0 -r 0.4        -g 0.9 -n 1.4 -N 1.5
#     ./iad -V 0 -r 0.4 -t 0.1 -g 0.9 -n 1.4 -N 1.5
#
# class fixed_albedo(unittest.TestCase):
#     @echo "********* Constrain a ************"
#     ./iad -V 0 -r 0.4        -a 0.9
#     ./iad -V 0 -r 0.4 -t 0.1 -a 0.9
#
# class fixed_optical_thickness(unittest.TestCase):
#     @echo "********* Constrain b ************"
#     ./iad -V 0 -r 0.4        -b 3
#     ./iad -V 0 -r 0.4 -t 0.1 -b 3
#     ./iad -V 0 -r 0.4 -t 0.1 -b 3 -n 1.5
#     ./iad -V 0 -r 0.4 -t 0.1 -b 3 -n 1.4 -N 1.5
#
# class fixed_mus(unittest.TestCase):
#     @echo "********* Constrain mu_s ************"
#     ./iad -V 0 -r 0.4        -F 30
#     ./iad -V 0 -r 0.4 -t 0.1 -F 30
#     ./iad -V 0 -r 0.4        -F 30 -n 1.5
#     ./iad -V 0 -r 0.4 -t 0.1 -F 30 -n 1.5
#     ./iad -V 0 -r 0.4        -F 30 -n 1.4 -N 1.5
#     ./iad -V 0 -r 0.4 -t 0.1 -F 30 -n 1.4 -N 1.5
#
# class fixed_mua(unittest.TestCase):
#     @echo "********* Constrain mu_a ************"
#     ./iad -V 0 -r 0.3        -A 0.6
#     ./iad -V 0 -r 0.3 -t 0.1 -A 0.6
#     ./iad -V 0 -r 0.3        -A 0.6 -n 1.5
#     ./iad -V 0 -r 0.3 -t 0.1 -A 0.6 -n 1.5
#     ./iad -V 0 -r 0.3        -A 0.6 -n 1.4 -N 1.5
#     ./iad -V 0 -r 0.3 -t 0.1 -A 0.6 -n 1.4 -N 1.5
#
# class fixed_mua_and_g(unittest.TestCase):
#     @echo "********* Constrain mu_a and g************"
#     ./iad -V 0 -r 0.3        -A 0.6 -g 0.6
#     ./iad -V 0 -r 0.3        -A 0.6 -g 0.6 -n 1.5
#     ./iad -V 0 -r 0.3        -A 0.6 -g 0.6 -n 1.4 -N 1.5
#
# class fixed_mus_and_g(unittest.TestCase):
#     @echo "********* Constrain mu_s and g************"
#     ./iad -V 0 -r 0.3        -F 2.0 -g 0.5
#     ./iad -V 0 -r 0.3        -F 2.0 -g 0.5 -n 1.5
#     ./iad -V 0 -r 0.3        -F 2.0 -g 0.5 -n 1.4 -N 1.5
#
# class one_sphere(unittest.TestCase):
#     @echo "********* Basic One Sphere tests ***********"
#     ./iad -V 0 -r 0.4                     -S 1
#     ./iad -V 0 -r 0.4 -t 0.1              -S 1
#     ./iad -V 0 -r 0.4 -t 0.1 -u 0.002     -S 1
#     ./iad -V 0 -r 0.2 -t 0.2 -u 0.0049787 -S 1
#     @echo "********* More One Sphere tests ***********"
#     ./iad -V 0 -r 0.4                     -S 1 -1 '200 13 13 0'
#     ./iad -V 0 -r 0.4 -t 0.1              -S 1 -1 '200 13 13 0'
#     ./iad -V 0 -r 0.4 -t 0.1 -u 0.002     -S 1 -1 '200 13 13 0'
#     ./iad -V 0 -r 0.2 -t 0.2 -u 0.0049787 -S 1 -1 '200 13 13 0'
#
# class monte_carlo(unittest.TestCase):
#     @echo "******** Basic 10,000 photon tests *********"
#     ./iad -V 0 -r 0.4                     -S 1 -p 10000
#     ./iad -V 0 -r 0.4 -t 0.1              -S 1 -p 10000
#     ./iad -V 0 -r 0.4 -t 0.1 -u 0.002     -S 1 -p 10000
#     ./iad -V 0 -r 0.2 -t 0.2 -u 0.0049787 -S 1 -p 10000
#
#     @echo "******** Basic timed photon tests *********"
#     ./iad -V 0 -r 0.4                     -S 1 -p -500
#     ./iad -V 0 -r 0.4 -t 0.1              -S 1 -p -500
#     ./iad -V 0 -r 0.4 -t 0.1 -u 0.002     -S 1 -p -500
#     ./iad -V 0 -r 0.2 -t 0.2 -u 0.0049787 -S 1 -p -500
#
# class two_spheres(unittest.TestCase):
#     @echo "********* Basic Two Sphere tests ***********"
#     ./iad -V 0 -r 0.4                     -S 2
#     ./iad -V 0 -r 0.4 -t 0.1              -S 2
#     ./iad -V 0 -r 0.4 -t 0.1 -u 0.002     -S 2
#     ./iad -V 0 -r 0.2 -t 0.1 -u 0.0049787 -S 2
#     ./iad -V 0 -r 0.4                     -S 2 -1 '200 13 13 0' -2 '200 13 13 0'
#     ./iad -V 0 -r 0.4 -t 0.1              -S 2 -1 '200 13 13 0' -2 '200 13 13 0'
#     ./iad -V 0 -r 0.4 -t 0.1 -u 0.002     -S 2 -1 '200 13 13 0' -2 '200 13 13 0'
#     ./iad -V 0 -r 0.2 -t 0.1 -u 0.0049787 -S 2 -1 '200 13 13 0' -2 '200 13 13 0'
#
# class oblique_incidence(unittest.TestCase):
#     @echo "********* Oblique tests ***********"
#     ./iad -V 0 -i 60 -r 0.00000 -t 0.13691
#     ./iad -V 0 -i 60 -r 0.14932 -t 0.23181
#     ./iad -V 0 -i 60 -r 0.61996 -t 0.30605
#
# class tstd_and_rstd(unittest.TestCase):
#     @echo "********* Different Tstd and Rstd ***********"
#     ./iad -V 0 -r 0.4 -t 0.005  -d 1 -M 0  -S 1 -1 '200 13 13 0' -T 0.5
#     ./iad -V 0 -r 0.4 -t 0.005  -d 1 -M 0  -S 1 -1 '200 13 13 0'
#     ./iad -V 0 -r 0.2 -t 0.01   -d 1 -M 0  -S 1 -1 '200 13 13 0' -R 0.5
#     ./iad -V 0 -r 0.2 -t 0.01   -d 1 -M 0  -S 1 -1 '200 13 13 0'


if __name__ == '__main__':
    unittest.main()
