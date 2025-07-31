# pylint: disable=invalid-name

"""Tests for Inverse Adding Doubling."""

import unittest
import numpy as np
import iadpython as iad


class IADTestAlbedo(unittest.TestCase):
    """Test inversion when solving only for albedo."""

    def test_albedo_01(self):
        """No data returns None for optical properties."""
        exp = iad.Experiment()
        a, b, g = exp.invert_rt()
        self.assertIsNone(a)
        self.assertIsNone(b)
        self.assertIsNone(g)

    def test_albedo_02(self):
        """Matched slab with albedo=0."""
        exp = iad.Experiment(r=0)
        a, b, g = exp.invert_rt()
        self.assertAlmostEqual(a, 0.0, delta=1e-4)
        self.assertAlmostEqual(b, np.inf)
        self.assertAlmostEqual(g, 0)

    def test_albedo_03(self):
        """Matched slab with albedo=0.3."""
        exp = iad.Experiment(r=0.05721)
        a, b, g = exp.invert_rt()
        self.assertAlmostEqual(a, 0.3, delta=1e-4)
        self.assertAlmostEqual(b, np.inf)
        self.assertAlmostEqual(g, 0)

    def test_albedo_04(self):
        """Matched slab with albedo=0.95."""
        exp = iad.Experiment(r=0.53554)
        a, b, g = exp.invert_rt()
        self.assertAlmostEqual(a, 0.95, delta=1e-4)
        self.assertAlmostEqual(b, np.inf)
        self.assertAlmostEqual(g, 0)

    def test_albedo_04a(self):
        """Matched slab with albedo=1."""
        exp = iad.Experiment(r=1)
        a, b, g = exp.invert_rt()
        self.assertAlmostEqual(a, 1.0, delta=1e-4)
        self.assertAlmostEqual(b, np.inf)
        self.assertAlmostEqual(g, 0)

    def test_albedo_05(self):
        """Matched slab with g=0.9."""
        exp = iad.Experiment(r=0.13865, default_g=0.9)
        a, b, g = exp.invert_rt()
        self.assertAlmostEqual(a, 0.95, delta=1e-3)
        self.assertAlmostEqual(b, np.inf)
        self.assertAlmostEqual(g, 0.9)

    def test_albedo_06(self):
        """Matched slab with b=1."""
        exp = iad.Experiment(r=0.30172, default_b=1)
        a, b, g = exp.invert_rt()
        self.assertAlmostEqual(a, 0.95, delta=1e-3)
        self.assertAlmostEqual(b, 1)
        self.assertAlmostEqual(g, 0.0)

    def test_albedo_07(self):
        """Mismatched slab with albedo=0.95."""
        s = iad.Sample(n=1.4)
        exp = iad.Experiment(r=0.38697, sample=s)
        a, b, g = exp.invert_rt()
        self.assertAlmostEqual(a, 0.95, delta=2e-2)
        self.assertAlmostEqual(b, np.inf)
        self.assertAlmostEqual(g, 0)

    def test_albedo_08(self):
        """Mismatched slab glass slide and albedo=0.95."""
        s = iad.Sample(n=1.4, n_above=1.5, n_below=1.5)
        exp = iad.Experiment(r=0.39152, sample=s)
        a, b, g = exp.invert_rt()
        self.assertAlmostEqual(a, 0.95, delta=2e-2)
        self.assertAlmostEqual(b, np.inf)
        self.assertAlmostEqual(g, 0)

    def test_albedo_09(self):
        """Matched slab with arrays."""
        exp = iad.Experiment(r=[0.05721, 0.11523, 0.53554])
        a, b, g = exp.invert_rt()
        aa = [0.3, 0.5, 0.95]
        bb = [np.inf, np.inf, np.inf]
        gg = [0, 0, 0]
        np.testing.assert_allclose(a, aa, atol=1e-4)
        np.testing.assert_allclose(b, bb)
        np.testing.assert_allclose(g, gg)

    def test_albedo_10(self):
        """Matched slab with arrays with b=1."""
        rr = [0.05125, 0.09912, 0.30172]
        exp = iad.Experiment(r=rr, default_b=1)
        a, b, g = exp.invert_rt()
        aa = [0.3, 0.5, 0.95]
        bb = [1, 1, 1]
        gg = [0, 0, 0]
        np.testing.assert_allclose(a, aa, atol=1e-4)
        np.testing.assert_allclose(b, bb)
        np.testing.assert_allclose(g, gg)

    def test_albedo_11(self):
        """Matched slab with arrays with b=1 and g=0.5."""
        s = iad.Sample(quad_pts=16)
        rr = [0.01786, 0.03824, 0.15098]
        exp = iad.Experiment(r=rr, sample=s, default_b=1, default_g=0.5)
        a, b, g = exp.invert_rt()
        aa = [0.3, 0.5, 0.95]
        bb = [1, 1, 1]
        gg = [0.5, 0.5, 0.5]
        np.testing.assert_allclose(a, aa, atol=1e-4)
        np.testing.assert_allclose(b, bb)
        np.testing.assert_allclose(g, gg)

    def test_albedo_12(self):
        """Mismatched slab with arrays with b=1 and g=0.5."""
        s = iad.Sample(n=1.4, n_above=1.5, n_below=1.5, quad_pts=16)
        rr = [0.05486, 0.06722, 0.20618]
        exp = iad.Experiment(r=rr, sample=s, default_b=1, default_g=0.5)
        a, b, g = exp.invert_rt()
        aa = [0.3, 0.5, 0.95]
        bb = [1, 1, 1]
        gg = [0.5, 0.5, 0.5]
        np.testing.assert_allclose(a, aa, atol=1e-4)
        np.testing.assert_allclose(b, bb)
        np.testing.assert_allclose(g, gg)

    def test_albedo_13(self):
        """Solve for albedo using transmission (matched boundaries)."""
        tt = [0.40736, 0.44606, 0.62257]
        exp = iad.Experiment(t=tt, default_b=1)
        a, b, g = exp.invert_rt()
        aa = [0.3, 0.5, 0.95]
        bb = [1, 1, 1]
        gg = [0, 0, 0]
        np.testing.assert_allclose(a, aa, atol=2e-3)
        np.testing.assert_allclose(b, bb)
        np.testing.assert_allclose(g, gg)

    def test_albedo_14(self):
        """Solve for albedo with only transmission."""
        s = iad.Sample(n=1.4, n_above=1.5, n_below=1.5, quad_pts=16)
        tt = [0.38924, 0.43336, 0.65527]
        exp = iad.Experiment(t=tt, sample=s, default_b=1, default_g=0.5)
        a, b, g = exp.invert_rt()
        aa = [0.3, 0.5, 0.95]
        bb = [1, 1, 1]
        gg = [0.5, 0.5, 0.5]
        np.testing.assert_allclose(a, aa, atol=1e-4)
        np.testing.assert_allclose(b, bb)
        np.testing.assert_allclose(g, gg)


class IADTestOpticalThickness(unittest.TestCase):
    """Test inversion when solving only for optical thickness."""

    def test_b_01(self):
        """Matched slab with albedo=0.5."""
        exp = iad.Experiment(r=0.11283, default_a=0.5)
        a, b, g = exp.invert_rt()
        self.assertAlmostEqual(a, 0.5)
        self.assertAlmostEqual(b, 2, delta=1e-3)
        self.assertAlmostEqual(g, 0)

    def test_b_02(self):
        """Matched slab with albedo=0.5."""
        exp = iad.Experiment(t=0.18932, default_a=0.5)
        a, b, g = exp.invert_rt()
        self.assertAlmostEqual(a, 0.5)
        self.assertAlmostEqual(b, 2, delta=1e-3)
        self.assertAlmostEqual(g, 0)

    def test_b_03(self):
        """Matched slab with albedo=0.5."""
        exp = iad.Experiment(r=0, default_a=0.5)
        a, b, g = exp.invert_rt()
        self.assertAlmostEqual(a, 0.5)
        self.assertAlmostEqual(b, 0, delta=1e-4)
        self.assertAlmostEqual(g, 0)

    def test_b_04(self):
        """Matched slab with albedo=0.5."""
        exp = iad.Experiment(t=1, default_a=0.5)
        a, b, g = exp.invert_rt()
        self.assertAlmostEqual(a, 0.5)
        self.assertAlmostEqual(b, 0, delta=1e-4)
        self.assertAlmostEqual(g, 0)

    def test_b_05(self):
        """Solve for optical thickness with only reflection."""
        # Convergence with is challenging
        s = iad.Sample(n=1.4, n_above=1.5, n_below=1.5, quad_pts=16)
        rr = [0.20285, 0.34590]
        exp = iad.Experiment(r=rr, sample=s, default_a=0.95, default_g=0.0)
        a, b, g = exp.invert_rt()
        aa = [0.95, 0.95]
        bb = [0.5, 2]
        gg = [0.0, 0.0]
        np.testing.assert_allclose(a, aa)
        np.testing.assert_allclose(b, bb, atol=0.2)
        np.testing.assert_allclose(g, gg)

    def test_b_06(self):
        """Solve for optical thickness with only transmission."""
        s = iad.Sample(n=1.4, n_above=1.5, n_below=1.5, quad_pts=16)
        tt = [0.64220, 0.20330, 0.00380]
        exp = iad.Experiment(t=tt, sample=s, default_a=0.5, default_g=0.5)
        a, b, g = exp.invert_rt()
        aa = [0.5, 0.5, 0.5]
        bb = [0.5, 2, 7]
        gg = [0.5, 0.5, 0.5]
        np.testing.assert_allclose(a, aa)
        np.testing.assert_allclose(b, bb, atol=2e-2)
        np.testing.assert_allclose(g, gg)


class IADAnisotropy(unittest.TestCase):
    """Test inversion when solving only for scattering anisotropy."""

    def test_g_01(self):
        """Matched slab with albedo=0.5."""
        exp = iad.Experiment(r=0.42872, default_b=2, default_a=0.95)
        a, b, g = exp.invert_rt()
        self.assertAlmostEqual(a, 0.95)
        self.assertAlmostEqual(b, 2)
        self.assertAlmostEqual(g, 0, delta=1e-3)

    def test_g_02(self):
        """Matched slab with albedo=0.5."""
        exp = iad.Experiment(t=0.40931, default_b=2, default_a=0.95)
        a, b, g = exp.invert_rt()
        self.assertAlmostEqual(a, 0.95)
        self.assertAlmostEqual(b, 2)
        self.assertAlmostEqual(g, 0, delta=1e-3)


class IADAB(unittest.TestCase):
    """Test inversion when solving only for a and b."""

    def test_ab_01(self):
        """Matched slab with albedo=0.5, b=2."""
        exp = iad.Experiment(r=0.42872, t=0.40931, default_g=0)
        a, b, g = exp.invert_rt()
        self.assertAlmostEqual(a, 0.95, delta=1e-4)
        self.assertAlmostEqual(b, 2, delta=1e-3)
        self.assertAlmostEqual(g, 0)

    def test_ab_02(self):
        """Matched slab with albedo=0.5, b=2 with defaults."""
        exp = iad.Experiment(r=0.42872, t=0.40931)
        a, b, g = exp.invert_rt()
        self.assertAlmostEqual(a, 0.95, delta=1e-4)
        self.assertAlmostEqual(b, 2, delta=1e-3)
        self.assertAlmostEqual(g, 0)

    def test_ab_03(self):
        """Matched slab with albedo=0.5, b=2."""
        exp = iad.Experiment(r=0.18825, t=0.67381, default_g=0.3)
        a, b, g = exp.invert_rt()
        self.assertAlmostEqual(a, 0.9, delta=1e-4)
        self.assertAlmostEqual(b, 1, delta=1e-3)
        self.assertAlmostEqual(g, 0.3)


class IADAG(unittest.TestCase):
    """Test inversion when solving only for a and g."""

    def test_ag_01(self):
        """Matched slab with albedo=0.5, b=2."""
        exp = iad.Experiment(r=0.42872, t=0.40931, default_b=2)
        a, b, g = exp.invert_rt()
        self.assertAlmostEqual(a, 0.95, delta=1e-4)
        self.assertAlmostEqual(b, 2)
        self.assertAlmostEqual(g, 0.0, delta=1e-3)

    def test_ag_02(self):
        """Matched slab with albedo=0.9, g=0.3."""
        exp = iad.Experiment(r=0.18825, t=0.67381, default_b=1)
        a, b, g = exp.invert_rt()
        self.assertAlmostEqual(a, 0.9, delta=1e-4)
        self.assertAlmostEqual(b, 1)
        self.assertAlmostEqual(g, 0.3, delta=1e-3)

    def test_ag_03(self):
        """Matched slab with albedo=0.9, g=0.3."""
        exp = iad.Experiment(r=0.18825, t=0.67381, u=0.36788)
        a, b, g = exp.invert_rt()
        self.assertAlmostEqual(a, 0.9, delta=1e-4)
        self.assertAlmostEqual(b, 1, delta=1e-4)
        self.assertAlmostEqual(g, 0.3, delta=1e-3)

    def test_ag_04a(self):
        """Matched slab with albedo=0.9, g=0.3."""
        aa = [0.95, 0.95]
        bb = [0.5, 2]
        gg = [0.7, 0.3]
        s = iad.Sample(a=aa, b=bb, g=gg, quad_pts=16)
        rr, tt, _, _ = s.rt()
        s.a = [0, 0]
        _, uu, _, _ = s.rt()
        s.quad_pts = 8
        exp = iad.Experiment(r=rr, t=tt, u=uu, sample=s)
        a, b, g = exp.invert_rt()
        np.testing.assert_allclose(a, aa, atol=2e-2)
        np.testing.assert_allclose(b, bb, atol=2e-2)
        np.testing.assert_allclose(g, gg, atol=2e-2)

    def test_ag_04b(self):
        """Matched slab with albedo=0.9, g=0.3."""
        aa = [0.95, 0.95]
        bb = [2, 5]
        gg = [0.3, 0.7]
        s = iad.Sample(a=aa, b=bb, g=gg, quad_pts=16)
        rr, tt, _, _ = s.rt()
        s.a = [0, 0]
        _, uu, _, _ = s.rt()
        s.quad_pts = 8
        exp = iad.Experiment(r=rr, t=tt, u=uu, sample=s)
        a, b, g = exp.invert_rt()
        np.testing.assert_allclose(a, aa, atol=2e-2)
        np.testing.assert_allclose(b, bb, atol=2e-2)
        np.testing.assert_allclose(g, gg, atol=2e-2)

    def test_ag_05(self):
        """Mismatched slab with albedo=0.9, g=0.3."""
        aa = [0.95, 0.95]
        bb = [0.5, 2]
        gg = [0.7, 0.3]
        s = iad.Sample(a=aa, b=bb, g=gg, n=1.4, n_above=1.5, n_below=1.5, quad_pts=16)
        rr, tt, _, _ = s.rt()
        s.a = [0, 0]
        _, uu, _, _ = s.rt()
        exp = iad.Experiment(r=rr, t=tt, u=uu, sample=s)
        a, b, g = exp.invert_rt()
        np.testing.assert_allclose(a, aa, atol=2e-2)
        np.testing.assert_allclose(b, bb, atol=2e-2)
        np.testing.assert_allclose(g, gg, atol=2e-2)


class IADBG(unittest.TestCase):
    """Test inversion when solving only for b and g."""

    def test_bg_01(self):
        """Matched slab with albedo=0.5, b=2."""
        exp = iad.Experiment(r=0.42872, t=0.40931, default_a=0.95)
        a, b, g = exp.invert_rt()
        self.assertAlmostEqual(a, 0.95)
        self.assertAlmostEqual(b, 2, delta=1e-3)
        self.assertAlmostEqual(g, 0.0, delta=1e-3)

    def test_bg_02(self):
        """Matched slab with albedo=0.5, b=1."""
        exp = iad.Experiment(r=0.18825, t=0.67381, default_a=0.9)
        a, b, g = exp.invert_rt()
        self.assertAlmostEqual(a, 0.9)
        self.assertAlmostEqual(b, 1, delta=2e-2)
        self.assertAlmostEqual(g, 0.3, delta=2e-2)


class WhatIsB(unittest.TestCase):
    """Test inversion of unscattered transmission."""

    def test_inverting_tu_01(self):
        """Matched slab with albedo=0.0, b=0.1."""
        b = 0.1
        s = iad.Sample(a=0, b=b, quad_pts=16)
        x = iad.Experiment(sample=s)
        _, x.m_u, _, _ = s.rt()
        bb = x.what_is_b()
        self.assertAlmostEqual(b, bb, delta=1e-5)

    def test_inverting_tu_02(self):
        """Matched slab with albedo=0.0, b=2."""
        b = 2
        s = iad.Sample(a=0, b=b, quad_pts=16)
        x = iad.Experiment(sample=s)
        _, x.m_u, _, _ = s.rt()
        bb = x.what_is_b()
        self.assertAlmostEqual(b, bb, delta=2e-5)

    def test_inverting_tu_03(self):
        """Matched slab with albedo=0.0, b=2."""
        b = 2
        s = iad.Sample(a=0, b=b, n=1.5, quad_pts=16)
        x = iad.Experiment(sample=s)
        _, x.m_u, _, _ = s.rt()
        bb = x.what_is_b()
        self.assertAlmostEqual(b, bb, delta=2e-5)

    def test_inverting_tu_04(self):
        """Matched slab with albedo=0.0, b=2."""
        b = 2
        s = iad.Sample(a=0, b=b, n=1.4, n_above=1.5, n_below=1.5, quad_pts=16)
        x = iad.Experiment(sample=s)
        _, x.m_u, _, _ = s.rt()
        bb = x.what_is_b()
        self.assertAlmostEqual(b, bb, delta=2e-5)


class InversionRTNoSphere(unittest.TestCase):
    """
    Inversion of measured values using just R and T.

    These tests check the round trip process

    a,b,g --> M_R, M_T --> a, b, g

    with knowledge of the correct g value.
    """

    def test_inversion_01(self):
        """Matched slab with albedo=0.5."""
        s = iad.Sample(a=0.95, b=2, g=0.0)
        ur1, ut1, _, _ = s.rt()
        exp = iad.Experiment(sample=s)
        exp.m_r, exp.m_t = exp.measured_rt()

        a, b, g = exp.invert_rt()
        self.assertAlmostEqual(a, 0.95, delta=1e-3)
        self.assertAlmostEqual(b, 2, delta=1e-3)
        self.assertAlmostEqual(g, 0, delta=1e-3)

    def test_inversion_02(self):
        """Sandwiched slab with albedo=0.95, g=0."""
        s = iad.Sample(a=0.95, b=2, g=0.0)
        s.n = 1.4
        s.n_above = 1.5
        s.n_below = 1.5
        exp = iad.Experiment(sample=s)
        exp.m_r, exp.m_t = exp.measured_rt()

        a, b, g = exp.invert_rt()
        self.assertAlmostEqual(a, 0.95, delta=1e-3)
        self.assertAlmostEqual(b, 2, delta=1e-3)
        self.assertAlmostEqual(g, 0, delta=1e-3)

    def test_inversion_03(self):
        """Sandwiched slab with albedo=0.95, g=0.9."""
        s = iad.Sample(a=0.95, b=2, g=0.9)
        s.n = 1.4
        s.n_above = 1.5
        s.n_below = 1.5
        exp = iad.Experiment(sample=s)
        exp.default_g = 0.9
        exp.m_r, exp.m_t = exp.measured_rt()

        a, b, g = exp.invert_rt()
        self.assertAlmostEqual(a, 0.95, delta=1e-3)
        self.assertAlmostEqual(b, 2, delta=1e-3)
        self.assertAlmostEqual(g, 0.9, delta=1e-3)


class InversionRTUNoSphere(unittest.TestCase):
    """
    Inversion of measured values using R, T, and U.

    These tests check the round trip process

    a,b,g --> M_R, M_T, M_U --> a, b, g
    """

    def test_01(self):
        """Moderate albedo but g=0.9."""
        s = iad.Sample(a=0.5, b=2, g=0.9)
        ur1, ut1, _, _ = s.rt()
        exp = iad.Experiment(sample=s)
        exp.m_r, exp.m_t = exp.measured_rt()
        exp.sample.a = 0
        _, exp.m_u = exp.measured_rt()

        a, b, g = exp.invert_rt()

        self.assertAlmostEqual(a, 0.5, delta=2e-3)
        self.assertAlmostEqual(b, 2, delta=1e-3)
        self.assertAlmostEqual(g, 0.9, delta=1e-3)

    def test_02(self):
        """High albedo and g=0."""
        s = iad.Sample(a=0.95, b=2, g=0.0)
        ur1, ut1, _, _ = s.rt()
        exp = iad.Experiment(sample=s)
        exp.m_r, exp.m_t = exp.measured_rt()
        exp.sample.a = 0
        _, exp.m_u = exp.measured_rt()

        a, b, g = exp.invert_rt()

        self.assertAlmostEqual(a, 0.95, delta=1e-3)
        self.assertAlmostEqual(b, 2, delta=1e-3)
        self.assertAlmostEqual(g, 0.0, delta=1e-3)

    def test_03(self):
        """High albedo and moderate g=0.5."""
        s = iad.Sample(a=0.95, b=2, g=0.5)
        ur1, ut1, _, _ = s.rt()
        exp = iad.Experiment(sample=s)
        exp.m_r, exp.m_t = exp.measured_rt()
        exp.sample.a = 0
        _, exp.m_u = exp.measured_rt()

        a, b, g = exp.invert_rt()

        self.assertAlmostEqual(a, 0.95, delta=2e-3)
        self.assertAlmostEqual(b, 2, delta=1e-3)
        self.assertAlmostEqual(g, 0.5, delta=1e-3)

    def test_04(self):
        """High albedo and high g."""
        s = iad.Sample(a=0.95, b=2, g=0.9)
        ur1, ut1, _, _ = s.rt()
        exp = iad.Experiment(sample=s)
        exp.m_r, exp.m_t = exp.measured_rt()
        exp.sample.a = 0
        _, exp.m_u = exp.measured_rt()
        #        print("M_R = %.4f" % exp.m_r)
        #        print("M_T = %.4f" % exp.m_t)
        #        print("M_U = %.4f %.4f" % (exp.m_u, np.exp(-2) ))

        a, b, g = exp.invert_rt()

        #        print("a = %.4f" % a)
        #        print("b = %.4f" % b)
        #        print("g = %.4f" % g)

        self.assertAlmostEqual(a, 0.95, delta=1e-3)
        self.assertAlmostEqual(b, 2, delta=1e-3)
        self.assertAlmostEqual(g, 0.9, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
