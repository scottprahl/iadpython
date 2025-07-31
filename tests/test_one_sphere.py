# pylint: disable=invalid-name

"""
Tests for calculations with one integrating sphere.

Start by testing to make sure the measured values are close
to those when no sphere is present.

Then make sure that inversion returns the same starting
values.
"""

import unittest
import iadpython as iad


class ForwardOneBigSphere(unittest.TestCase):
    """Test forward single sphere calculations."""

    def test_forward_01(self):
        """Big black sphere."""
        s = iad.Sample(a=0.95, b=1)
        ur1, ut1, uru, utu = s.rt()

        d_sphere = 2500
        d_sample = 10
        rsph = iad.Sphere(d_sphere, d_sample, d_third=1, r_wall=0)
        ur1a = rsph.MR(ur1, uru)

        tsph = iad.Sphere(d_sphere, d_sample, d_third=10, r_wall=0)
        ut1a = tsph.MT(ut1, uru)
        self.assertAlmostEqual(ur1, ur1a, delta=1e-4)
        self.assertAlmostEqual(ut1, ut1a, delta=1e-4)

    def test_forward_02(self):
        """Big gray sphere."""
        s = iad.Sample(a=0.95, b=1)
        ur1, ut1, uru, utu = s.rt()

        d_sphere = 2500
        d_sample = 10
        rsph = iad.Sphere(d_sphere, d_sample, d_third=1, r_wall=0.9)
        ur1a = rsph.MR(ur1, uru)

        tsph = iad.Sphere(d_sphere, d_sample, d_third=10, r_wall=0.9)
        ut1a = tsph.MR(ut1, uru)
        self.assertAlmostEqual(ur1, ur1a, delta=1e-4)
        self.assertAlmostEqual(ut1, ut1a, delta=1e-4)

    def test_forward_03(self):
        """Big nearly white sphere."""
        s = iad.Sample(a=0.95, b=1)
        ur1, ut1, uru, utu = s.rt()

        d_sphere = 2500
        d_sample = 10
        rsph = iad.Sphere(d_sphere, d_sample, d_third=1, r_wall=0.98)
        ur1a = rsph.MR(ur1, uru)

        tsph = iad.Sphere(d_sphere, d_sample, d_third=10, r_wall=0.98)
        ut1a = tsph.MT(ut1, uru)
        self.assertAlmostEqual(ur1, ur1a, delta=1e-4)
        self.assertAlmostEqual(ut1, ut1a, delta=1e-4)

    def test_forward_04(self):
        """Big white sphere."""
        s = iad.Sample(a=0.95, b=1)
        ur1, ut1, uru, utu = s.rt()

        d_sphere = 2500
        d_sample = 10
        rsph = iad.Sphere(d_sphere, d_sample, d_third=1, r_wall=1.0)
        ur1a = rsph.MR(ur1, uru)

        tsph = iad.Sphere(d_sphere, d_sample, d_third=10, r_wall=1.0)
        ut1a = tsph.MT(ut1, uru)

        # actual result is complicated.
        self.assertGreater(ur1, ur1a)
        self.assertLess(ut1, ut1a)


class ForwardOneMediumSphere(unittest.TestCase):
    """Test medium size forward single sphere calculations."""

    def test_forward_01(self):
        """Big black sphere."""
        s = iad.Sample(a=0.95, b=1)
        ur1, ut1, uru, utu = s.rt()

        d_sphere = 200
        d_sample = 10
        rsph = iad.Sphere(d_sphere, d_sample, d_third=1, r_wall=0)
        ur1a = rsph.MR(ur1, uru)

        tsph = iad.Sphere(d_sphere, d_sample, d_third=10, r_wall=0)
        ut1a = tsph.MT(ut1, uru)
        self.assertAlmostEqual(ur1, ur1a, delta=2e-4)
        self.assertAlmostEqual(ut1, ut1a, delta=2e-4)

    def test_forward_02(self):
        """Big gray sphere."""
        s = iad.Sample(a=0.95, b=1)
        ur1, ut1, uru, utu = s.rt()

        d_sphere = 200
        d_sample = 10
        rsph = iad.Sphere(d_sphere, d_sample, d_third=1, r_wall=0.9)
        ur1a = rsph.MR(ur1, uru)

        tsph = iad.Sphere(d_sphere, d_sample, d_third=10, r_wall=0.9)
        ut1a = tsph.MT(ut1, uru)
        self.assertAlmostEqual(ur1, ur1a, delta=2e-3)
        self.assertAlmostEqual(ut1, ut1a, delta=2e-3)

    def test_forward_03(self):
        """Big nearly white sphere."""
        s = iad.Sample(a=0.95, b=1)
        ur1, ut1, uru, utu = s.rt()

        d_sphere = 200
        d_sample = 10
        rsph = iad.Sphere(d_sphere, d_sample, d_third=1, r_wall=0.98)
        ur1a = rsph.MR(ur1, uru)

        tsph = iad.Sphere(d_sphere, d_sample, d_third=10, r_wall=0.98)
        ut1a = tsph.MT(ut1, uru)
        self.assertAlmostEqual(ur1, ur1a, delta=2e-2)
        self.assertAlmostEqual(ut1, ut1a, delta=2e-2)

    def test_forward_04(self):
        """Big white sphere."""
        s = iad.Sample(a=0.95, b=1)
        ur1, ut1, uru, utu = s.rt()

        d_sphere = 200
        d_sample = 10
        rsph = iad.Sphere(d_sphere, d_sample, d_third=1, r_wall=1)
        ur1a = rsph.MR(ur1, uru)

        tsph = iad.Sphere(d_sphere, d_sample, d_third=10, r_wall=1)
        ut1a = tsph.MT(ut1, uru)

        # actual result is complicated.
        self.assertGreater(ur1, ur1a)
        self.assertLess(ut1, ut1a)


class ForwardOneSphereMR(unittest.TestCase):
    """Test forward single sphere calculations of MR."""

    def test_MR_00(self):
        """No sphere."""
        s = iad.Sample(a=0.95, b=1, quad_pts=16)
        ur1, ut1, uru, utu = s.rt()

        # iad -a 0.95 -b 1 -z
        # use ad -a 0.95 -b 1 to get uru and utu
        self.assertAlmostEqual(ut1, 0.6226, delta=1e-4)
        self.assertAlmostEqual(ur1, 0.3017, delta=1e-4)
        self.assertAlmostEqual(utu, 0.5110, delta=1e-4)
        self.assertAlmostEqual(uru, 0.3964, delta=1e-4)

    def test_MR_01(self):
        """No unscattered reflectance case, with baffle."""
        d_sphere = 100
        d_sample = 10
        d_third = 10
        d_detector = 2
        r_wall = 0.98
        r_std = 0.9

        s = iad.Sample(a=0.95, b=1, quad_pts=12)
        ur1, ut1, uru, utu = s.rt()

        rsph = iad.Sphere(
            d_sphere,
            d_sample,
            d_third=d_third,
            d_detector=d_detector,
            r_wall=r_wall,
            r_std=r_std,
        )
        rsph.sample.uru = uru
        rsph.baffle = True
        MR = rsph.MR(ur1, uru)

        # iad -z -a 0.95 -b 1 -M 0 -z -1 '100 10 10 2 0.98' -R 0.9
        self.assertAlmostEqual(MR, 0.2798, delta=1e-4)

    def test_MR_02(self):
        """No unscattered reflectance case, no baffle."""
        d_sphere = 100
        d_sample = 10
        d_third = 10
        d_detector = 2
        r_wall = 0.98
        r_std = 0.9

        s = iad.Sample(a=0.95, b=1, quad_pts=12)
        ur1, ut1, uru, utu = s.rt()

        rsph = iad.Sphere(
            d_sphere,
            d_sample,
            d_third=d_third,
            d_detector=d_detector,
            r_wall=r_wall,
            r_std=r_std,
        )
        rsph.sample.uru = uru
        rsph.baffle = False
        MR = rsph.MR(ur1, uru)

        # iad -z -a 0.95 -b 1 -M 0 -z -H 0 -1 '100 10 10 2 0.98' -R 0.9 -q 12
        self.assertAlmostEqual(MR, 0.2859, delta=1e-4)

    def test_MR_03(self):
        """Unscattered reflectance case, with baffle."""
        d_sphere = 100
        d_sample = 10
        d_third = 10
        d_detector = 2
        r_wall = 0.98
        r_std = 0.9

        s = iad.Sample(a=0.95, b=1, n=1.5, quad_pts=12)
        ur1, ut1, uru, utu = s.rt()
        s = iad.Sample(a=0, b=1, n=1.5, quad_pts=12)
        R_u, _, _, _ = s.rt()

        rsph = iad.Sphere(
            d_sphere,
            d_sample,
            d_third=d_third,
            d_detector=d_detector,
            r_wall=r_wall,
            r_std=r_std,
        )
        rsph.sample.uru = uru
        rsph.baffle = True
        MR = rsph.MR(ur1, uru, R_u=R_u)

        # iad -z -a 0.95 -b 1 -n 1.5 -M 0 -z -1 '100 10 10 2 0.98' -R 0.9 -q 12
        self.assertAlmostEqual(MR, 0.2532, delta=1e-4)

    def test_MR_04(self):
        """Unscattered reflectance case, no baffle."""
        d_sphere = 100
        d_sample = 10
        d_third = 10
        d_detector = 2
        r_wall = 0.98
        r_std = 0.9

        s = iad.Sample(a=0.95, b=1, n=1.5, quad_pts=12)
        ur1, ut1, uru, utu = s.rt()
        s = iad.Sample(a=0, b=1, n=1.5, quad_pts=12)
        R_u, _, _, _ = s.rt()

        rsph = iad.Sphere(
            d_sphere,
            d_sample,
            d_third=d_third,
            d_detector=d_detector,
            r_wall=r_wall,
            r_std=r_std,
        )
        rsph.sample.uru = uru
        rsph.baffle = False
        MR = rsph.MR(ur1, uru, R_u=R_u)

        # iad -z -a 0.95 -b 1 -n 1.5 -M 0 -z -1 '100 10 10 2 0.98' -R 0.9 -q 12 -H 0
        self.assertAlmostEqual(MR, 0.2577, delta=1e-4)

    def test_MR_05(self):
        """No unscattered reflectance case that is not collected."""
        d_sphere = 100
        d_sample = 10
        d_third = 10
        d_detector = 2
        r_wall = 0.98
        r_std = 0.9

        s = iad.Sample(a=0.95, b=1, n=1.5, quad_pts=12)
        ur1, ut1, uru, utu = s.rt()
        s = iad.Sample(a=0, b=1, n=1.5, quad_pts=12)
        R_u, _, _, _ = s.rt()

        rsph = iad.Sphere(
            d_sphere,
            d_sample,
            d_third=d_third,
            d_detector=d_detector,
            r_wall=r_wall,
            r_std=r_std,
        )
        rsph.sample.uru = uru
        rsph.baffle = True
        MR = rsph.MR(ur1, uru, R_u=R_u, f_u=0)

        # iad -z -a 0.95 -b 1 -n 1.5 -M 0 -z -1 '100 10 10 2 0.98' -R 0.9 -q 12 -c 0
        self.assertAlmostEqual(MR, 0.2117, delta=1e-4)

    def test_MR_06(self):
        """Some light hits wall first."""
        d_sphere = 100
        d_sample = 10
        d_third = 10
        d_detector = 2
        r_wall = 0.98
        r_std = 0.9

        s = iad.Sample(a=0.95, b=1, n=1.5, quad_pts=12)
        ur1, ut1, uru, utu = s.rt()
        s = iad.Sample(a=0, b=1, n=1.5, quad_pts=12)
        R_u, _, _, _ = s.rt()

        rsph = iad.Sphere(
            d_sphere,
            d_sample,
            d_third=d_third,
            d_detector=d_detector,
            r_wall=r_wall,
            r_std=r_std,
        )
        rsph.sample.uru = uru
        rsph.baffle = True
        MR = rsph.MR(ur1, uru, R_u=R_u, f_w=0.5)

        # iad -z -a 0.95 -b 1 -n 1.5 -M 0 -z -1 '100 10 10 2 0.98' -R 0.9 -q 12 -f 0.5
        self.assertAlmostEqual(MR, 0.2396, delta=1e-4)

    def test_MT_01(self):
        """Standard present for sample and calibration. Also baffle."""
        d_sphere = 100
        d_sample = 10
        d_third = 10
        d_detector = 2
        r_wall = 0.98
        r_std = 0.9

        s = iad.Sample(a=0.95, b=1, quad_pts=12)
        ur1, ut1, uru, utu = s.rt()
        s = iad.Sample(a=0, b=1, quad_pts=12)
        _, T_u, _, _ = s.rt()

        tsph = iad.Sphere(
            d_sphere,
            d_sample,
            d_third=d_third,
            d_detector=d_detector,
            r_wall=r_wall,
            r_std=r_std,
        )
        tsph.refl = False
        tsph.sample.uru = uru
        tsph.baffle = True
        print(tsph)
        MT = tsph.MT(ut1, uru, T_u=T_u)

        # iad -z -a 0.95 -b 1 -M 0 -z -S 1 -1 '100 10 10 2 0.98' -T 0.9 -V 0
        self.assertAlmostEqual(MT, 0.6065, delta=2e-4)

    def test_MT_02(self):
        """Third port closed for sample and calibration."""
        d_sphere = 100
        d_sample = 10
        d_third = 0
        d_detector = 2
        r_wall = 0.98
        r_std = 0.9

        s = iad.Sample(a=0.95, b=1, quad_pts=12)
        ur1, ut1, uru, utu = s.rt()
        s = iad.Sample(a=0, b=1, quad_pts=12)
        _, T_u, _, _ = s.rt()

        tsph = iad.Sphere(
            d_sphere,
            d_sample,
            d_third=d_third,
            d_detector=d_detector,
            r_wall=r_wall,
            r_std=r_std,
        )
        tsph.refl = False
        tsph.sample.uru = uru
        tsph.baffle = True
        print(tsph.gain(0))
        print(tsph)
        MT = tsph.MT(ut1, uru, T_u=T_u)

        # iad -z -a 0.95 -b 1 -M 0 -z -S 1 -1 '100 10 0 2 0.98' -T 0.9 -V 0
        self.assertAlmostEqual(MT, 0.6376, delta=2e-4)

    def test_MT_03(self):
        """Third port open for sample and standard for calibration."""
        d_sphere = 100
        d_sample = 10
        d_third = 10
        d_detector = 2
        r_wall = 0.98
        r_std = 0.9

        s = iad.Sample(a=0.95, b=1, quad_pts=12)
        ur1, ut1, uru, utu = s.rt()
        s = iad.Sample(a=0, b=1, quad_pts=12)
        _, T_u, _, _ = s.rt()

        tsph = iad.Sphere(
            d_sphere,
            d_sample,
            d_third=d_third,
            d_detector=d_detector,
            r_wall=r_wall,
            r_std=r_std,
        )
        tsph.refl = False
        tsph.sample.uru = uru
        tsph.baffle = True
        print(tsph)
        MT = tsph.MT(ut1, uru, T_u=T_u, f_u=0)

        # iad -z -a 0.95 -b 1 -M 0 -z -S 1 -1 '100 10 10 2 0.98' -T 0.9 -V 0 -C 0
        self.assertAlmostEqual(MT, 0.2357, delta=2e-4)


if __name__ == "__main__":
    unittest.main()
