# pylint: disable=protected-access

"""Tests for sphere object."""

import random
import unittest
import iadpython


class TestDoubleSphere(unittest.TestCase):
    """Simple Double Sphere Tests."""

    def setUp(self):
        """Setup for all these tests."""
        random.seed(0)
        r_sphere = iadpython.Sphere(100, 30, d_detector=10, r_wall=1)
        t_sphere = iadpython.Sphere(100, 30, d_detector=10, r_wall=1)
        self.double = iadpython.DoubleSphere(r_sphere, t_sphere)
        self.double.ur1 = 0
        self.double.uru = 0
        self.double.ut1 = 1
        self.double.utu = 1

    def test_init(self):
        """Make sure basics work."""
        self.assertEqual(self.double.ur1, 0.0)
        self.assertEqual(self.double.ut1, 1.0)
        self.assertEqual(self.double.uru, 0.0)
        self.assertEqual(self.double.utu, 1.0)

    def test_str(self):
        """Test string describing the object."""
        output = str(self.double)
        self.assertIn("Reflection Sphere", output)
        self.assertIn("Transmission Sphere", output)
        self.assertIn("ur1 =   0.000", output)
        self.assertIn("ut1 =   1.000", output)
        self.assertIn("uru =   0.000", output)
        self.assertIn("utu =   1.000", output)

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

        self.assertAlmostEqual(r_total / N, 0.5, delta=0.06)
        self.assertAlmostEqual(t_total / N, 0.5, delta=0.06)

    def test_no_sample_N(self):
        """Light passes unhindered between spheres."""
        self.double.ur1 = 0
        self.double.ut1 = 1
        self.double.uru = self.double.ur1
        self.double.utu = self.double.ut1
        N = 1000
        r, _, t, _ = self.double.do_N_photons(N)
        self.assertAlmostEqual(r, 0.5, delta=0.06)
        self.assertAlmostEqual(t, 0.5, delta=0.06)

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
        self.assertAlmostEqual(r_total / N, 1.0, places=5)
        self.assertAlmostEqual(t_total / N, 0.0, places=5)

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

    def test_cweb_two_sphere_formulas(self):
        """Two-sphere gain/power formulas should match CWEB algebra exactly."""
        r_sphere = iadpython.Sphere(180, 25, d_third=12, d_detector=8, r_detector=0.08, r_wall=0.95, r_std=0.98)
        t_sphere = iadpython.Sphere(170, 22, d_third=9, d_detector=7, r_detector=0.05, r_wall=0.94, r_std=0.97)
        r_sphere.baffle = True
        t_sphere.baffle = False
        ds = iadpython.DoubleSphere(r_sphere, t_sphere)
        ds.f_r = 0.23

        ur1 = 0.31
        uru = 0.42
        ut1 = 0.27
        utu = 0.35

        def gain_manual(sphere, uru_sample, uru_third):
            as_x = sphere.sample.a
            ad_x = sphere.detector.a
            at_x = sphere.third.a
            aw_x = sphere.a_wall
            rd_x = sphere.detector.uru
            rw_x = sphere.r_wall
            if sphere.baffle:
                inv_gain = rw_x + (at_x / aw_x) * uru_third
                inv_gain *= aw_x + (1 - at_x) * (ad_x * rd_x + as_x * uru_sample)
                inv_gain = 1.0 - inv_gain
            else:
                inv_gain = 1.0 - aw_x * rw_x - ad_x * rd_x - as_x * uru_sample - at_x * uru_third
            return 1.0 / inv_gain

        g = gain_manual(r_sphere, uru, 0.0)
        gp = gain_manual(t_sphere, uru, 0.0)
        gain_denom = (
            1
            - r_sphere.sample.a
            * t_sphere.sample.a
            * r_sphere.a_wall
            * t_sphere.a_wall
            * (1 - r_sphere.third.a)
            * (1 - t_sphere.third.a)
            * g
            * gp
            * utu
            * utu
        )
        g11 = g / gain_denom
        g22 = gp / gain_denom

        r_expected = r_sphere.detector.a * (1 - r_sphere.third.a) * r_sphere.r_wall * g11
        r_expected *= (
            (1 - ds.f_r) * ur1
            + r_sphere.r_wall * ds.f_r
            + (1 - ds.f_r) * t_sphere.sample.a * (1 - t_sphere.third.a) * t_sphere.r_wall * ut1 * utu * gp
        )

        t_expected = t_sphere.detector.a * (1 - t_sphere.third.a) * t_sphere.r_wall * g22
        t_expected *= (1 - ds.f_r) * ut1 + (1 - r_sphere.third.a) * r_sphere.r_wall * r_sphere.sample.a * utu * (
            ds.f_r * r_sphere.r_wall + (1 - ds.f_r) * ur1
        ) * g

        self.assertAlmostEqual(ds.gain_single(ds.REFLECTION_SPHERE, uru, 0.0), g, places=12)
        self.assertAlmostEqual(ds.gain_single(ds.TRANSMISSION_SPHERE, uru, 0.0), gp, places=12)
        self.assertAlmostEqual(ds.gain_11(uru, utu), g11, places=12)
        self.assertAlmostEqual(ds.gain_22(uru, utu), g22, places=12)
        self.assertAlmostEqual(ds.two_sphere_r(ur1, uru, ut1, utu), r_expected, places=12)
        self.assertAlmostEqual(ds.two_sphere_t(ur1, uru, ut1, utu), t_expected, places=12)

        r_0 = ds.two_sphere_r(0, 0, 0, 0)
        t_0 = ds.two_sphere_t(0, 0, 0, 0)
        mr_expected = (
            r_sphere.r_std * (r_expected - r_0) / (ds.two_sphere_r(r_sphere.r_std, r_sphere.r_std, 0, 0) - r_0)
        )
        mt_expected = (t_expected - t_0) / (ds.two_sphere_t(0, 0, 1, 1) - t_0)

        mr_calc, mt_calc = ds.measured_rt(ur1, uru, ut1, utu)
        self.assertAlmostEqual(mr_calc, mr_expected, places=12)
        self.assertAlmostEqual(mt_calc, mt_expected, places=12)

    def test_experiment_measured_rt_uses_two_sphere_path(self):
        """Experiment should use two-sphere equations when num_spheres==2."""
        sample = iadpython.Sample(a=0.7, b=1.2, g=0.5, n=1.4, n_above=1.5, n_below=1.5, quad_pts=16)
        r_sphere = iadpython.Sphere(150, 10, d_third=10, d_detector=5, r_detector=0.04, r_wall=0.98, r_std=0.99)
        t_sphere = iadpython.Sphere(150, 10, d_third=10, d_detector=5, r_detector=0.04, r_wall=0.98, r_std=0.99)
        exp = iadpython.Experiment(sample=sample, num_spheres=2, r_sphere=r_sphere, t_sphere=t_sphere)
        exp.f_r = 0.17
        exp.fraction_of_rc_in_mr = 0.91
        exp.fraction_of_tc_in_mt = 0.83
        exp.ur1_lost = 0.01
        exp.ut1_lost = 0.02
        exp.uru_lost = 0.03
        exp.utu_lost = 0.04

        m_r, m_t = exp.measured_rt()

        ur1, ut1, uru, utu = sample.rt()
        nu_inside = iadpython.cos_snell(1, sample.nu_0, sample.n)
        r_u, t_u = iadpython.specular_rt(sample.n_above, sample.n, sample.n_below, sample.b, nu_inside)
        ur1_calc = max(ur1 - (1 - exp.fraction_of_rc_in_mr) * r_u - exp.ur1_lost, 0.0)
        ut1_calc = max(ut1 - (1 - exp.fraction_of_tc_in_mt) * t_u - exp.ut1_lost, 0.0)
        uru_calc = max(uru - exp.uru_lost, 0.0)
        utu_calc = max(utu - exp.utu_lost, 0.0)

        ds = iadpython.DoubleSphere(r_sphere, t_sphere)
        ds.f_r = exp.f_r
        expected_r, expected_t = ds.measured_rt(ur1_calc, uru_calc, ut1_calc, utu_calc)
        self.assertAlmostEqual(float(m_r), float(expected_r), places=12)
        self.assertAlmostEqual(float(m_t), float(expected_t), places=12)

    def test_measured_rt_reference_normalization_points(self):
        """Two-sphere normalization should match CWEB reference points."""
        r_sphere = iadpython.Sphere(190, 22, d_third=11, d_detector=7, r_detector=0.03, r_wall=0.96, r_std=0.98)
        t_sphere = iadpython.Sphere(170, 20, d_third=9, d_detector=6, r_detector=0.02, r_wall=0.95, r_std=0.97)
        ds = iadpython.DoubleSphere(r_sphere, t_sphere)
        ds.f_r = 0.19

        mr0, mt0 = ds.measured_rt(0.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(mr0, 0.0, places=12)
        self.assertAlmostEqual(mt0, 0.0, places=12)

        mr_std, _ = ds.measured_rt(r_sphere.r_std, r_sphere.r_std, 0.0, 0.0)
        self.assertAlmostEqual(mr_std, r_sphere.r_std, places=12)

        _, mt_std = ds.measured_rt(0.0, 0.0, 1.0, 1.0)
        self.assertAlmostEqual(mt_std, 1.0, places=12)

    def test_mr_mt_wrappers_match_measured_rt(self):
        """`MR`/`MT` wrappers should return the same values as `measured_rt`."""
        r_sphere = iadpython.Sphere(160, 18, d_third=8, d_detector=6, r_detector=0.05, r_wall=0.97, r_std=0.99)
        t_sphere = iadpython.Sphere(155, 18, d_third=8, d_detector=6, r_detector=0.06, r_wall=0.96, r_std=0.98)
        ds = iadpython.DoubleSphere(r_sphere, t_sphere)
        ds.f_r = 0.11

        ur1 = 0.27
        uru = 0.31
        ut1 = 0.39
        utu = 0.43

        mr, mt = ds.measured_rt(ur1, uru, ut1, utu)
        self.assertAlmostEqual(ds.MR(ur1, uru, ut1, utu), mr, places=12)
        self.assertAlmostEqual(ds.MT(ur1, uru, ut1, utu), mt, places=12)

    def test_experiment_two_sphere_requires_both_spheres(self):
        """Two-sphere forward path should validate required sphere objects."""
        sample = iadpython.Sample(a=0.6, b=1.0, g=0.4, quad_pts=16)
        r_sphere = iadpython.Sphere(150, 10, d_third=10, d_detector=5, r_wall=0.98, r_std=0.99)
        exp = iadpython.Experiment(sample=sample, num_spheres=2, r_sphere=r_sphere, t_sphere=None)
        with self.assertRaises(ValueError):
            exp.measured_rt()

    def test_experiment_two_sphere_clips_negative_corrected_values(self):
        """Negative corrected UR/UT values should be clipped to zero before CWEB equations."""
        sample = iadpython.Sample(a=0.8, b=1.5, g=0.6, quad_pts=16)
        r_sphere = iadpython.Sphere(150, 10, d_third=10, d_detector=5, r_detector=0.02, r_wall=0.98, r_std=0.99)
        t_sphere = iadpython.Sphere(150, 10, d_third=10, d_detector=5, r_detector=0.02, r_wall=0.98, r_std=0.99)
        exp = iadpython.Experiment(sample=sample, num_spheres=2, r_sphere=r_sphere, t_sphere=t_sphere)
        exp.ur1_lost = 10.0
        exp.ut1_lost = 10.0
        exp.uru_lost = 10.0
        exp.utu_lost = 10.0
        exp.fraction_of_rc_in_mr = 1.0
        exp.fraction_of_tc_in_mt = 1.0

        m_r, m_t = exp.measured_rt()
        self.assertAlmostEqual(float(m_r), 0.0, places=12)
        self.assertAlmostEqual(float(m_t), 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
