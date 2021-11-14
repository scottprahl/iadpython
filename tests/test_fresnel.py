# pylint: disable=invalid-name
# pylint: disable=no-self-use
# pylint: disable=too-many-statements

"""Tests for Fresnel reflection."""

import unittest
import numpy as np
import iadpython.fresnel


class Fresnel(unittest.TestCase):
    """Starting layer calculations."""

    def test_01_snell(self):
        """Snell's law."""
        n_i = 1
        n_t = 1
        nui = 0.5
        nut = iadpython.fresnel.cos_snell(n_i, nui, n_t)
        np.testing.assert_approx_equal(nut, nui)

        nui = 0
        nut = iadpython.fresnel.cos_snell(n_i, nui, n_t)
        np.testing.assert_approx_equal(nut, nui)

        nui = 1
        nut = iadpython.fresnel.cos_snell(n_i, nui, n_t)
        np.testing.assert_approx_equal(nut, nui)

        nui = np.array([0.0, 0.2, 0.5, 0.9, 1.0])
        nut = iadpython.fresnel.cos_snell(n_i, nui, n_t)
        np.testing.assert_allclose(nut, nui, atol=1e-5)

        n_i = 1
        n_t = 1.5
        nui = 1
        nut = iadpython.fresnel.cos_snell(n_i, nui, n_t)
        np.testing.assert_approx_equal(nut, nui)

        nui = 0.5
        nut = iadpython.fresnel.cos_snell(n_i, nui, n_t)
        np.testing.assert_approx_equal(nut, 0.81649658)

        nui = np.array([0.0, 0.2, 0.5, 0.9, 1.0])
        nut = iadpython.fresnel.cos_snell(n_i, nui, n_t)
        t = np.array([0.745356, 0.757188, 0.816496, 0.956847, 1.0])
        np.testing.assert_allclose(nut, t, atol=1e-5)

        n_i = 1.5
        n_t = 1
        nui = 1
        nut = iadpython.fresnel.cos_snell(n_i, nui, n_t)
        np.testing.assert_approx_equal(nut, nui)

        nui = 0.5
        nut = iadpython.fresnel.cos_snell(n_i, nui, n_t)
        np.testing.assert_approx_equal(nut, 0)

        nui = np.array([0.0, 0.2, 0.5, 0.9, 1.0])
        nut = iadpython.fresnel.cos_snell(n_i, nui, n_t)
        t = np.array([0, 0, 0, 0.756637, 1.0])
        np.testing.assert_allclose(nut, t, atol=1e-5)

    def test_02_critical(self):
        """Critical angle."""
        n_i = 1
        n_t = 1
        nu_c = iadpython.fresnel.cos_critical(n_i, n_t)
        np.testing.assert_approx_equal(nu_c, 0)

        n_i = 1
        n_t = 1.5
        nu_c = iadpython.fresnel.cos_critical(n_i, n_t)
        np.testing.assert_approx_equal(nu_c, 0)

        n_i = 1.5
        n_t = 1
        nu_c = iadpython.fresnel.cos_critical(n_i, n_t)
        np.testing.assert_approx_equal(nu_c, 0.7453559)

    def test_03_fresnel_matched(self):
        """Fresnel reflection with matched boundaries."""
        n_i = 1
        n_t = 1
        nu_i = 0.5
        r = iadpython.fresnel.reflection(n_i, nu_i, n_t)
        np.testing.assert_approx_equal(r, 0)

        nu_i = 1
        r = iadpython.fresnel.reflection(n_i, nu_i, n_t)
        np.testing.assert_approx_equal(r, 0)

#       90° incident light is tricky
        nu_i = 0.0
        r = iadpython.fresnel.reflection(n_i, nu_i, n_t)
        np.testing.assert_approx_equal(r, 0)

        nu_i = np.array([0.0, 0.2, 0.5, 1.0])
        r = iadpython.fresnel.reflection(n_i, nu_i, n_t)
        rr = np.array([0, 0, 0, 0])
        np.testing.assert_allclose(r, rr, atol=1e-5)

    def test_04_fresnel_low_to_high(self):
        """Fresnel reflection with mismatched boundaries."""
        n_i = 1
        n_t = 1.5
        nu_i = 1
        r = iadpython.fresnel.reflection(n_i, nu_i, n_t)
        np.testing.assert_approx_equal(r, 0.04)

        nu_i = 1
        r = iadpython.fresnel.reflection(n_i, nu_i, n_t)
        np.testing.assert_approx_equal(r, 0.04)

        nu_i = 0.5
        r = iadpython.fresnel.reflection(n_i, nu_i, n_t)
        np.testing.assert_approx_equal(r, 0.0891867128)

        nu_i = np.array([0, 0.2, 0.5, 1.0])
        r = iadpython.fresnel.reflection(n_i, nu_i, n_t)
        rr = np.array([1, 0.338894, 0.0891867128, 0.04])
        np.testing.assert_allclose(r, rr, atol=1e-5)

    def test_05_fresnel_high_to_low(self):
        """Fresnel reflection with mismatched boundaries total."""
        n_i = 1.5
        n_t = 1
        nu_i = 0.5
        r = iadpython.fresnel.reflection(n_i, nu_i, n_t)
        np.testing.assert_approx_equal(r, 1)

        nu_i = 0.8
        r = iadpython.fresnel.reflection(n_i, nu_i, n_t)
        np.testing.assert_approx_equal(r, 0.11414110022)

        nu_i = 1.0
        r = iadpython.fresnel.reflection(n_i, nu_i, n_t)
        np.testing.assert_approx_equal(r, 0.04)

        nu_i = 0.0
        r = iadpython.fresnel.reflection(n_i, nu_i, n_t)
        np.testing.assert_approx_equal(r, 1)

        nu_i = np.array([0.2, 0.5, 0.8, 1.0])
        r = iadpython.fresnel.reflection(n_i, nu_i, n_t)
        rr = np.array([1, 1, 0.11414110022, 0.04])
        np.testing.assert_allclose(r, rr, atol=1e-5)

    def test_06_glass_matched(self):
        """Glass layer reflection."""
        n_i = 1.5
        n_g = 1.5
        n_t = 1.5
        nu_i = 1
        r = iadpython.fresnel.glass(n_i, n_g, n_t, nu_i)
        np.testing.assert_approx_equal(r, 0)

        n_i = 1.5
        n_g = 1.5
        n_t = 1.5
        nu_i = np.array([0.0, 0.2, 0.5, 1.0])
        r = iadpython.fresnel.glass(n_i, n_g, n_t, nu_i)
        rr = np.zeros(4)
        np.testing.assert_allclose(r, rr, atol=1e-5)

    def test_07_glass_low_to_high(self):
        """Glass layer reflection mismatched."""
        n_i = 1
        n_g = 1
        n_t = 1.5
        nu_i = 1
        r = iadpython.fresnel.glass(n_i, n_g, n_t, nu_i)
        np.testing.assert_approx_equal(r, 0.04)

        n_i = 1
        n_g = 1.5
        n_t = 1.5
        nu_i = 1
        r = iadpython.fresnel.glass(n_i, n_g, n_t, nu_i)
        np.testing.assert_approx_equal(r, 0.04)

        n_i = 1.0
        n_g = 1.5
        n_t = 1.0
        nu_i = 1
        rr = 2 * 0.04 * 0.96 / (1 - 0.04**2)
        r = iadpython.fresnel.glass(n_i, n_g, n_t, nu_i)
        np.testing.assert_approx_equal(r, rr)

        n_i = 1.0
        n_g = 1.5
        n_t = 1.3
        nu_i = 1
        rr = 0.0447030
        r = iadpython.fresnel.glass(n_i, n_g, n_t, nu_i)
        np.testing.assert_approx_equal(r, rr)

        n_i = 1.0
        n_g = 1.5
        n_t = 1.3
        nu_i = 0
        rr = 1
        r = iadpython.fresnel.glass(n_i, n_g, n_t, nu_i)
        np.testing.assert_approx_equal(r, rr)

    def test_08_glass_low_to_high_arrays(self):
        """Glass layer reflection low to high index arrays."""
        n_i = 1.0
        n_g = 1.5
        n_t = 1.5
        nu_i = np.array([0.0, 0.2, 0.5, 1.0])
        r = iadpython.fresnel.glass(n_i, n_g, n_t, nu_i)
        rr = np.array([1.0, 0.338894, 0.0891867128, 0.04])
        np.testing.assert_allclose(r, rr, atol=1e-5)

        n_i = 1.0
        n_g = 1.5
        n_t = 1.0
        nu_i = np.array([0, 0.2, 0.5, 0.8, 1.0])
        r = iadpython.fresnel.glass(n_i, n_g, n_t, nu_i)
        rr = np.array([1, 0.50623, 0.163768, 0.084098, 0.076923])
        np.testing.assert_allclose(r, rr, atol=1e-5)

        n_i = 1.0
        n_g = 1.5
        n_t = 1.3
        nu_i = np.array([0, 0.2, 0.5, 0.8, 1.0])
        r = iadpython.fresnel.glass(n_i, n_g, n_t, nu_i)
        rr = np.array([1, 0.343278, 0.095087, 0.048798, 0.044703])
        np.testing.assert_allclose(r, rr, atol=1e-5)

    def test_09_glass_high_to_low(self):
        """Glass layer reflection low to high index arrays."""
        n_i = 1.5
        n_g = 1.5
        n_t = 1.0
        nu_i = 0
        r = iadpython.fresnel.glass(n_i, n_g, n_t, nu_i)
        rr = 1
        np.testing.assert_approx_equal(r, rr)

        n_i = 1.5
        n_g = 1.5
        n_t = 1.0
        nu_i = 0.2
        r = iadpython.fresnel.glass(n_i, n_g, n_t, nu_i)
        rr = 1
        np.testing.assert_approx_equal(r, rr)

        n_i = 1.5
        n_g = 1.5
        n_t = 1.0
        nu_i = 0.8
        r = iadpython.fresnel.glass(n_i, n_g, n_t, nu_i)
        rr = 0.11414110022
        np.testing.assert_approx_equal(r, rr)

        n_i = 1.5
        n_g = 1.5
        n_t = 1.0
        nu_i = 1.0
        r = iadpython.fresnel.glass(n_i, n_g, n_t, nu_i)
        rr = 0.04
        np.testing.assert_approx_equal(r, rr)

        n_i = 1.5
        n_g = 1.0
        n_t = 1.0
        nu_i = 0
        r = iadpython.fresnel.glass(n_i, n_g, n_t, nu_i)
        rr = 1
        np.testing.assert_approx_equal(r, rr)

        n_i = 1.5
        n_g = 1.0
        n_t = 1.0
        nu_i = 0.2
        r = iadpython.fresnel.glass(n_i, n_g, n_t, nu_i)
        rr = 1
        np.testing.assert_approx_equal(r, rr)

        n_i = 1.5
        n_g = 1.0
        n_t = 1.0
        nu_i = 0.8
        r = iadpython.fresnel.glass(n_i, n_g, n_t, nu_i)
        rr = 0.11414110022
        np.testing.assert_approx_equal(r, rr)

        n_i = 1.5
        n_g = 1.0
        n_t = 1.0
        nu_i = 1.0
        r = iadpython.fresnel.glass(n_i, n_g, n_t, nu_i)
        rr = 0.04
        np.testing.assert_approx_equal(r, rr)

    def test_10_glass_high_to_low_arrays(self):
        """Glass layer reflection low to high index arrays."""
        n_i = 1.5
        n_g = 1.0
        n_t = 1.0
        nu_i = np.array([0, 0.2, 0.5, 0.8, 1.0])
        r = iadpython.fresnel.glass(n_i, n_g, n_t, nu_i)
        rr = np.array([1, 1, 1, 0.11414110022, 0.04])
        np.testing.assert_allclose(r, rr, atol=1e-5)

        n_i = 1.5
        n_g = 1.5
        n_t = 1.0
        nu_i = np.array([0, 0.2, 0.5, 0.8, 1.0])
        r = iadpython.fresnel.glass(n_i, n_g, n_t, nu_i)
        rr = np.array([1, 1, 1, 0.11414110022, 0.04])
        np.testing.assert_allclose(r, rr, atol=1e-5)

    def test_11_absorbing_glass(self):
        """Absorbing glass layer reflection."""
        n_i = 1.0
        n_g = 1.5
        n_t = 1.3
        nu_i = np.array([0.0, 0.2, 0.5, 0.8, 1.0])

        r, _ = iadpython.fresnel.absorbing_glass_RT(n_i, n_g, n_t, nu_i, 0)
        rr = iadpython.fresnel.glass(n_i, n_g, n_t, nu_i)
        np.testing.assert_allclose(r, rr, atol=1e-5)

        n_i = 1.0
        n_g = 1.5
        n_t = 1.3
        nu_i = np.array([0.0, 0.2, 0.5, 0.8, 1.0])

        r, _ = iadpython.fresnel.absorbing_glass_RT(n_i, n_g, n_t, nu_i, np.inf)
        rr = iadpython.fresnel.reflection(n_i, nu_i, n_g)
        np.testing.assert_allclose(r, rr, atol=1e-5)

#     def test_08_sandwich(self):
#     def specular_nu_RT(n_top, n_slab, n_bottom, b_slab, nu, b_top=0, b_bottom=0, flip=False):
#         n_top = 1.0
#         n_slab = 1.5
#         n_bot = 1.0
#         b_slab = 0
#         nu = np.array([0.2, 0.5, 0.8, 1.0])


if __name__ == '__main__':
    unittest.main()
