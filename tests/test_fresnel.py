# pylint: disable=invalid-name
# pylint: disable=too-many-statements

"""Tests for Fresnel reflection."""

import unittest
import numpy as np
import iadpython as iad


class Fresnel(unittest.TestCase):
    """Starting layer calculations."""

    def test_01_snell(self):
        """Snell's law."""
        n_i = 1
        n_t = 1
        nui = 0.5
        nut = iad.fresnel.cos_snell(n_i, nui, n_t)
        np.testing.assert_approx_equal(nut, nui)

        nui = 0
        nut = iad.fresnel.cos_snell(n_i, nui, n_t)
        np.testing.assert_approx_equal(nut, nui)

        nui = 1
        nut = iad.fresnel.cos_snell(n_i, nui, n_t)
        np.testing.assert_approx_equal(nut, nui)

        nui = np.array([0.0, 0.2, 0.5, 0.9, 1.0])
        nut = iad.fresnel.cos_snell(n_i, nui, n_t)
        np.testing.assert_allclose(nut, nui, atol=1e-5)

        n_i = 1
        n_t = 1.5
        nui = 1
        nut = iad.fresnel.cos_snell(n_i, nui, n_t)
        np.testing.assert_approx_equal(nut, nui)

        nui = 0.5
        nut = iad.fresnel.cos_snell(n_i, nui, n_t)
        np.testing.assert_approx_equal(nut, 0.81649658)

        nui = np.array([0.0, 0.2, 0.5, 0.9, 1.0])
        nut = iad.fresnel.cos_snell(n_i, nui, n_t)
        t = np.array([0.745356, 0.757188, 0.816496, 0.956847, 1.0])
        np.testing.assert_allclose(nut, t, atol=1e-5)

        n_i = 1.5
        n_t = 1
        nui = 1
        nut = iad.fresnel.cos_snell(n_i, nui, n_t)
        np.testing.assert_approx_equal(nut, nui)

        nui = 0.5
        nut = iad.fresnel.cos_snell(n_i, nui, n_t)
        np.testing.assert_approx_equal(nut, 0)

        nui = np.array([0.0, 0.2, 0.5, 0.9, 1.0])
        nut = iad.fresnel.cos_snell(n_i, nui, n_t)
        t = np.array([0, 0, 0, 0.756637, 1.0])
        np.testing.assert_allclose(nut, t, atol=1e-5)

    def test_02_critical(self):
        """Critical angle."""
        n_i = 1
        n_t = 1
        nu_c = iad.fresnel.cos_critical(n_i, n_t)
        np.testing.assert_approx_equal(nu_c, 0)

        n_i = 1
        n_t = 1.5
        nu_c = iad.fresnel.cos_critical(n_i, n_t)
        np.testing.assert_approx_equal(nu_c, 0)

        n_i = 1.5
        n_t = 1
        nu_c = iad.fresnel.cos_critical(n_i, n_t)
        np.testing.assert_approx_equal(nu_c, 0.7453559)

    def test_03_fresnel_matched(self):
        """Fresnel reflection with matched boundaries."""
        n_i = 1
        n_t = 1
        nu_i = 0.5
        r = iad.fresnel.fresnel_reflection(n_i, nu_i, n_t)
        np.testing.assert_approx_equal(r, 0)

        nu_i = 1
        r = iad.fresnel.fresnel_reflection(n_i, nu_i, n_t)
        np.testing.assert_approx_equal(r, 0)

        #       90° incident light is tricky
        nu_i = 0.0
        r = iad.fresnel.fresnel_reflection(n_i, nu_i, n_t)
        np.testing.assert_approx_equal(r, 0)

        nu_i = np.array([0.0, 0.2, 0.5, 1.0])
        r = iad.fresnel.fresnel_reflection(n_i, nu_i, n_t)
        rr = np.array([0, 0, 0, 0])
        np.testing.assert_allclose(r, rr, atol=1e-5)

    def test_04_fresnel_low_to_high(self):
        """Fresnel reflection with mismatched boundaries."""
        n_i = 1
        n_t = 1.5
        nu_i = 1
        r = iad.fresnel.fresnel_reflection(n_i, nu_i, n_t)
        np.testing.assert_approx_equal(r, 0.04)

        nu_i = 1
        r = iad.fresnel.fresnel_reflection(n_i, nu_i, n_t)
        np.testing.assert_approx_equal(r, 0.04)

        nu_i = 0.5
        r = iad.fresnel.fresnel_reflection(n_i, nu_i, n_t)
        np.testing.assert_approx_equal(r, 0.0891867128)

        nu_i = np.array([0, 0.2, 0.5, 1.0])
        r = iad.fresnel.fresnel_reflection(n_i, nu_i, n_t)
        rr = np.array([1, 0.338894, 0.0891867128, 0.04])
        np.testing.assert_allclose(r, rr, atol=1e-5)

    def test_05_fresnel_high_to_low(self):
        """Fresnel reflection with mismatched boundaries total."""
        n_i = 1.5
        n_t = 1
        nu_i = 0.5
        r = iad.fresnel.fresnel_reflection(n_i, nu_i, n_t)
        np.testing.assert_approx_equal(r, 1)

        nu_i = 0.8
        r = iad.fresnel.fresnel_reflection(n_i, nu_i, n_t)
        np.testing.assert_approx_equal(r, 0.11414110022)

        nu_i = 1.0
        r = iad.fresnel.fresnel_reflection(n_i, nu_i, n_t)
        np.testing.assert_approx_equal(r, 0.04)

        nu_i = 0.0
        r = iad.fresnel.fresnel_reflection(n_i, nu_i, n_t)
        np.testing.assert_approx_equal(r, 1)

        nu_i = np.array([0.2, 0.5, 0.8, 1.0])
        r = iad.fresnel.fresnel_reflection(n_i, nu_i, n_t)
        rr = np.array([1, 1, 0.11414110022, 0.04])
        np.testing.assert_allclose(r, rr, atol=1e-5)

    def test_06_glass_matched(self):
        """Glass layer reflection."""
        n_i = 1.5
        n_g = 1.5
        n_t = 1.5
        nu_i = 1
        r = iad.fresnel.glass(n_i, n_g, n_t, nu_i)
        np.testing.assert_approx_equal(r, 0)

        n_i = 1.5
        n_g = 1.5
        n_t = 1.5
        nu_i = np.array([0.0, 0.2, 0.5, 1.0])
        r = iad.fresnel.glass(n_i, n_g, n_t, nu_i)
        rr = np.zeros(4)
        np.testing.assert_allclose(r, rr, atol=1e-5)

    def test_07_glass_low_to_high(self):
        """Glass layer reflection mismatched."""
        n_i = 1
        n_g = 1
        n_t = 1.5
        nu_i = 1
        r = iad.fresnel.glass(n_i, n_g, n_t, nu_i)
        np.testing.assert_approx_equal(r, 0.04)

        n_i = 1
        n_g = 1.5
        n_t = 1.5
        nu_i = 1
        r = iad.fresnel.glass(n_i, n_g, n_t, nu_i)
        np.testing.assert_approx_equal(r, 0.04)

        n_i = 1.0
        n_g = 1.5
        n_t = 1.0
        nu_i = 1
        rr = 2 * 0.04 * 0.96 / (1 - 0.04**2)
        r = iad.fresnel.glass(n_i, n_g, n_t, nu_i)
        np.testing.assert_approx_equal(r, rr)

        n_i = 1.0
        n_g = 1.5
        n_t = 1.3
        nu_i = 1
        rr = 0.0447030
        r = iad.fresnel.glass(n_i, n_g, n_t, nu_i)
        np.testing.assert_approx_equal(r, rr)

        n_i = 1.0
        n_g = 1.5
        n_t = 1.3
        nu_i = 0
        rr = 1
        r = iad.fresnel.glass(n_i, n_g, n_t, nu_i)
        np.testing.assert_approx_equal(r, rr)

    def test_08_glass_low_to_high_arrays(self):
        """Glass layer reflection low to high index arrays."""
        n_i = 1.0
        n_g = 1.5
        n_t = 1.5
        nu_i = np.array([0.0, 0.2, 0.5, 1.0])
        r = iad.fresnel.glass(n_i, n_g, n_t, nu_i)
        rr = np.array([1.0, 0.338894, 0.0891867128, 0.04])
        np.testing.assert_allclose(r, rr, atol=1e-5)

        n_i = 1.0
        n_g = 1.5
        n_t = 1.0
        nu_i = np.array([0, 0.2, 0.5, 0.8, 1.0])
        r = iad.fresnel.glass(n_i, n_g, n_t, nu_i)
        rr = np.array([1, 0.50623, 0.163768, 0.084098, 0.076923])
        np.testing.assert_allclose(r, rr, atol=1e-5)

        n_i = 1.0
        n_g = 1.5
        n_t = 1.3
        nu_i = np.array([0, 0.2, 0.5, 0.8, 1.0])
        r = iad.fresnel.glass(n_i, n_g, n_t, nu_i)
        rr = np.array([1, 0.343278, 0.095087, 0.048798, 0.044703])
        np.testing.assert_allclose(r, rr, atol=1e-5)

    def test_09_glass_high_to_low(self):
        """Glass layer reflection low to high index arrays."""
        n_i = 1.5
        n_g = 1.5
        n_t = 1.0
        nu_i = 0
        r = iad.fresnel.glass(n_i, n_g, n_t, nu_i)
        rr = 1
        np.testing.assert_approx_equal(r, rr)

        n_i = 1.5
        n_g = 1.5
        n_t = 1.0
        nu_i = 0.2
        r = iad.fresnel.glass(n_i, n_g, n_t, nu_i)
        rr = 1
        np.testing.assert_approx_equal(r, rr)

        n_i = 1.5
        n_g = 1.5
        n_t = 1.0
        nu_i = 0.8
        r = iad.fresnel.glass(n_i, n_g, n_t, nu_i)
        rr = 0.11414110022
        np.testing.assert_approx_equal(r, rr)

        n_i = 1.5
        n_g = 1.5
        n_t = 1.0
        nu_i = 1.0
        r = iad.fresnel.glass(n_i, n_g, n_t, nu_i)
        rr = 0.04
        np.testing.assert_approx_equal(r, rr)

        n_i = 1.5
        n_g = 1.0
        n_t = 1.0
        nu_i = 0
        r = iad.fresnel.glass(n_i, n_g, n_t, nu_i)
        rr = 1
        np.testing.assert_approx_equal(r, rr)

        n_i = 1.5
        n_g = 1.0
        n_t = 1.0
        nu_i = 0.2
        r = iad.fresnel.glass(n_i, n_g, n_t, nu_i)
        rr = 1
        np.testing.assert_approx_equal(r, rr)

        n_i = 1.5
        n_g = 1.0
        n_t = 1.0
        nu_i = 0.8
        r = iad.fresnel.glass(n_i, n_g, n_t, nu_i)
        rr = 0.11414110022
        np.testing.assert_approx_equal(r, rr)

        n_i = 1.5
        n_g = 1.0
        n_t = 1.0
        nu_i = 1.0
        r = iad.fresnel.glass(n_i, n_g, n_t, nu_i)
        rr = 0.04
        np.testing.assert_approx_equal(r, rr)

    def test_10_glass_high_to_low_arrays(self):
        """Glass layer reflection low to high index arrays."""
        n_i = 1.5
        n_g = 1.0
        n_t = 1.0
        nu_i = np.array([0, 0.2, 0.5, 0.8, 1.0])
        r = iad.fresnel.glass(n_i, n_g, n_t, nu_i)
        rr = np.array([1, 1, 1, 0.11414110022, 0.04])
        np.testing.assert_allclose(r, rr, atol=1e-5)

        n_i = 1.5
        n_g = 1.5
        n_t = 1.0
        nu_i = np.array([0, 0.2, 0.5, 0.8, 1.0])
        r = iad.fresnel.glass(n_i, n_g, n_t, nu_i)
        rr = np.array([1, 1, 1, 0.11414110022, 0.04])
        np.testing.assert_allclose(r, rr, atol=1e-5)

    def test_11_absorbing_glass(self):
        """Absorbing glass layer reflection."""
        n_i = 1.0
        n_g = 1.5
        n_t = 1.3
        nu_i = np.array([0.0, 0.2, 0.5, 0.8, 1.0])

        r, _ = iad.fresnel.absorbing_glass_RT(n_i, n_g, n_t, nu_i, 0)
        rr = iad.fresnel.glass(n_i, n_g, n_t, nu_i)
        np.testing.assert_allclose(r, rr, atol=1e-5)

        n_i = 1.0
        n_g = 1.5
        n_t = 1.3
        nu_i = np.array([0.0, 0.2, 0.5, 0.8, 1.0])

        r, _ = iad.fresnel.absorbing_glass_RT(n_i, n_g, n_t, nu_i, np.inf)
        rr = iad.fresnel.fresnel_reflection(n_i, nu_i, n_g)
        np.testing.assert_allclose(r, rr, atol=1e-5)


class AbsorbingGlass(unittest.TestCase):
    """Absorbing glass calculations."""

    def test_absorbing_glass_01(self):
        """No boundries."""
        n_i = 1.0
        n_g = 1.0
        n_t = 1.0
        nu_in = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
        rr = np.zeros_like(nu_in)
        tt = 1 - rr
        r, t = iad.absorbing_glass_RT(n_i, n_g, n_t, nu_in, 0)
        np.testing.assert_allclose(r, rr, atol=1e-5)
        np.testing.assert_allclose(t, tt, atol=1e-5)

    def test_absorbing_glass_02(self):
        """No glass, but sample."""
        n_i = 1.0
        n_g = 1.0
        n_t = 1.5
        nu_in = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
        rr = iad.fresnel_reflection(n_i, nu_in, n_t)
        tt = 1 - rr
        r, t = iad.absorbing_glass_RT(n_i, n_g, n_t, nu_in, 0)
        np.testing.assert_allclose(r, rr, atol=1e-5)
        np.testing.assert_allclose(t, tt, atol=1e-5)
        np.testing.assert_allclose(r[-1], 0.04, atol=1e-5)
        np.testing.assert_allclose(t[-1], 0.96, atol=1e-5)

    def test_absorbing_glass_03(self):
        """No glass, but sample."""
        n_i = 1.0
        n_g = 1.5
        n_t = 1.5
        nu_in = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
        rr = iad.fresnel_reflection(n_i, nu_in, n_t)
        tt = 1 - rr
        r, t = iad.absorbing_glass_RT(n_i, n_g, n_t, nu_in, 0)
        np.testing.assert_allclose(r, rr, atol=1e-5)
        np.testing.assert_allclose(t, tt, atol=1e-5)
        np.testing.assert_allclose(r[-1], 0.04, atol=1e-5)
        np.testing.assert_allclose(t[-1], 0.96, atol=1e-5)

    def test_absorbing_glass_04(self):
        """Non-absorbing glass slide on sample."""
        b = 0
        n_i = 1.0
        n_g = 1.5
        n_t = 1.4
        nu_in = np.array([0.01, 0.2, 0.5, 0.8, 1.0])
        r1 = iad.fresnel_reflection(n_i, nu_in, n_g)
        nu_g = iad.cos_snell(n_i, nu_in, n_g)
        r2 = iad.fresnel_reflection(n_g, nu_g, n_t)
        t1 = 1 - r1
        t2 = 1 - r2
        rr = r1 + r2 * t1 * t1 / (1 - r1 * r2 * np.exp(-2 * b / nu_g))
        tt = t1 * t2 / (1 - r1 * r2 * np.exp(-2 * b / nu_g))
        r, t = iad.absorbing_glass_RT(n_i, n_g, n_t, nu_in, b)
        np.testing.assert_allclose(r, rr, atol=1e-5)
        np.testing.assert_allclose(t, tt, atol=1e-5)

    def test_absorbing_glass_05(self):
        """Absorbing glass slide on sample."""
        b = 1
        n_i = 1.0
        n_g = 1.5
        n_t = 1.4
        nu_in = np.array([0.01, 0.2, 0.5, 0.8, 1.0])
        r1 = iad.fresnel_reflection(n_i, nu_in, n_g)
        nu_g = iad.cos_snell(n_i, nu_in, n_g)
        r2 = iad.fresnel_reflection(n_g, nu_g, n_t)
        t1 = 1 - r1
        t2 = 1 - r2
        rr = r1 + r2 * t1**2 * np.exp(-2 * b / nu_g) / (1 - r1 * r2 * np.exp(-2 * b / nu_g))
        tt = t1 * t2 * np.exp(-b / nu_g) / (1 - r1 * r2 * np.exp(-2 * b / nu_g))
        r, t = iad.absorbing_glass_RT(n_i, n_g, n_t, nu_in, b)
        np.testing.assert_allclose(r, rr, atol=1e-5)
        np.testing.assert_allclose(t, tt, atol=1e-5)


class Specular(unittest.TestCase):
    """Tests for unscattered light."""

    def test_01_specular(self):
        """Matched boundaries no absorption."""
        n_top = 1.0
        n_slab = 1.0
        n_bot = 1.0
        b_slab = 0
        nu_in = np.array([0.01, 0.2, 0.5, 0.8, 1.0])
        r, t = iad.specular_rt(n_top, n_slab, n_bot, b_slab, nu_in)
        rr = np.zeros_like(nu_in)
        tt = 1 - rr
        np.testing.assert_allclose(r, rr, atol=1e-5)
        np.testing.assert_allclose(t, tt, atol=1e-5)

    def test_02_specular(self):
        """Mismatched boundaries some absorption in slab."""
        n_top = 1.0
        n_slab = 1.5
        n_bot = 1.0
        b_slab = 1
        nu_in = np.array([0.01, 0.2, 0.5, 0.8, 1.0])
        r, t = iad.specular_rt(n_top, n_slab, n_bot, b_slab, nu_in)
        rr, tt = iad.absorbing_glass_RT(1, n_slab, 1, nu_in, b_slab)
        np.testing.assert_allclose(r, rr, atol=1e-5)
        np.testing.assert_allclose(t, tt, atol=1e-5)

    def test_03_specular(self):
        """Mismatched boundaries some absorption in slab."""
        n_top = 1.5
        n_slab = 1.5
        n_bot = 1.5
        b_slab = 1
        nu_in = np.array([0.01, 0.2, 0.5, 0.8, 1.0])
        r, t = iad.specular_rt(n_top, n_slab, n_bot, b_slab, nu_in)
        rr, tt = iad.absorbing_glass_RT(1, n_slab, 1, nu_in, b_slab)
        np.testing.assert_allclose(r, rr, atol=1e-5)
        np.testing.assert_allclose(t, tt, atol=1e-5)

    def test_04_specular(self):
        """Mismatched boundaries no transmission."""
        n_top = 1.5
        n_slab = 1.5
        n_bot = 1.5
        b_slab = np.inf
        nu_in = np.array([0.01, 0.2, 0.5, 0.8, 1.0])
        r, t = iad.specular_rt(n_top, n_slab, n_bot, b_slab, nu_in)
        rr, tt = iad.absorbing_glass_RT(1, n_slab, 1, nu_in, b_slab)
        np.testing.assert_allclose(r, rr, atol=1e-5)
        np.testing.assert_allclose(t, tt, atol=1e-5)

    def test_05_specular(self):
        """Mismatched boundaries no transmission but flipped."""
        n_top = 1.5
        n_slab = 1.5
        n_bot = 1.5
        b_slab = np.inf
        nu_in = np.array([0.01, 0.2, 0.5, 0.8, 1.0])
        r, t = iad.specular_rt(n_top, n_slab, n_bot, b_slab, nu_in, flip=True)
        rr, tt = iad.absorbing_glass_RT(1, n_slab, 1, nu_in, b_slab)
        np.testing.assert_allclose(r, rr, atol=1e-5)
        np.testing.assert_allclose(t, tt, atol=1e-5)

    def test_06_specular(self):
        """No slide on bottom and no transmission."""
        n_top = 1.5
        n_slab = 1.4
        n_bot = 1.4
        b_slab = np.inf
        nu_i = np.array([0.01, 0.2, 0.5, 0.8, 1.0])  # in air
        nu_in = iad.cos_snell(1, nu_i, n_slab)  # in slab
        r, t = iad.specular_rt(n_top, n_slab, n_bot, b_slab, nu_in)
        rr, _ = iad.absorbing_glass_RT(1, n_top, n_slab, nu_i, 0)
        tt = np.zeros_like(nu_i)
        np.testing.assert_allclose(r, rr, atol=1e-5)
        np.testing.assert_allclose(t, tt, atol=1e-5)

    def test_07_specular(self):
        """No slide on bottom and no transmission and flipped."""
        n_top = 1.5
        n_slab = 1.4
        n_bot = 1.4
        b_slab = np.inf
        nu_i = np.array([0.01, 0.2, 0.5, 0.8, 1.0])  # in air
        nu_in = iad.cos_snell(1, nu_i, n_slab)  # in slab
        r, t = iad.specular_rt(n_top, n_slab, n_bot, b_slab, nu_in, flip=True)
        rr, _ = iad.absorbing_glass_RT(1, n_bot, n_slab, nu_i, 0)
        tt = np.zeros_like(nu_i)
        np.testing.assert_allclose(r, rr, atol=1e-5)
        np.testing.assert_allclose(t, tt, atol=1e-5)

    def test_08_specular(self):
        """Slide on bottom and top."""
        n_top = 1.5
        n_slab = 1.4
        n_bot = 1.5
        b_slab = 1
        nu_i = 1.0
        nu_in = iad.cos_snell(1, nu_i, n_slab)  # in slab
        r, t = iad.specular_rt(n_top, n_slab, n_bot, b_slab, nu_in)
        s = iad.Sample(b=b_slab, n=n_slab, n_above=n_top, n_below=n_bot, quad_pts=32)
        rr, tt, _, _ = s.rt()
        np.testing.assert_allclose(r, rr, atol=1e-4)
        np.testing.assert_allclose(t, tt, atol=1e-4)


#     def test_09_specular(self):
#         """Slide on bottom and top with oblique incidence."""
#         n_top = 1.5
#         n_slab = 1.4
#         n_bot = 1.5
#         b_slab = 1
#         nu_i = 0.5
#         nu_in = iad.cos_snell(1, nu_i, n_slab) # in slab
#         r, t = iad.specular_rt(n_top, n_slab, n_bot, b_slab, nu_in)
#         s = iad.Sample(b=b_slab, n=n_slab, n_above=n_top, n_below=n_bot, quad_pts=32)
#         s.nu_0 = nu_i
#         rr, tt, _, _ = s.rt()
#         np.testing.assert_allclose(r, rr, atol=1e-4)
#         np.testing.assert_allclose(t, tt, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
