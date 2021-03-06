# pylint: disable=invalid-name
# pylint: disable=bad-whitespace
# pylint: disable=no-self-use
# pylint: disable=too-many-locals

"""Tests for slide-sample-slide combinations."""

import unittest
import numpy as np
import iadpython

class A_nothing_sandwich(unittest.TestCase):
    """Empty layer in air."""

    def test_01_nothing(self):
        """Empty layer calculation with no boundaries."""
        s = iadpython.Sample(a=0.5, b=0.0, g=0.0, n=1, quad_pts=4)
        rr, _, tt, _ = s.rt_matrices()
        R, T = iadpython.zero_layer(s)
        np.testing.assert_allclose(R, rr, atol=1e-5)
        np.testing.assert_allclose(T, tt, atol=1e-5)

    def test_02_nothing(self):
        """Empty layer calculation with boundary but no slides."""
        s = iadpython.Sample(a=0.5, b=0, g=0.0, n=1.5, quad_pts=4)
        rr, _, tt, _ = s.rt_matrices()

        R=np.array([[8.51769,0.00000,0.00000,0.00000],
                    [0.00000,2.28231,0.00000,0.00000],
                    [0.00000,0.00000,0.44244,0.00000],
                    [0.00000,0.00000,0.00000,0.60416]])

        T=np.array([[0.00000,0.00000,0.00000,0.00000],
                    [0.00000,0.00000,0.00000,0.00000],
                    [0.00000,0.00000,2.71091,0.00000],
                    [0.00000,0.00000,0.00000,7.24994]])

        np.testing.assert_allclose(R, rr, atol=1e-5)
        np.testing.assert_allclose(T, tt, atol=1e-5)

    def test_03_nothing(self):
        """Empty layer calculation with matching slides ."""
        s = iadpython.Sample(a=0.5, b=0, g=0.0, n=1.5, quad_pts=4, n_above=1.5, n_below=1.5)
        rr, _, tt, _ = s.rt_matrices()

        R=np.array([[8.51769,0.00000,0.00000,0.00000],
                    [0.00000,2.28231,0.00000,0.00000],
                    [0.00000,0.00000,0.44244,0.00000],
                    [0.00000,0.00000,0.00000,0.60416]])

        T=np.array([[0.00000,0.00000,0.00000,0.00000],
                    [0.00000,0.00000,0.00000,0.00000],
                    [0.00000,0.00000,2.71091,0.00000],
                    [0.00000,0.00000,0.00000,7.24994]])

        np.testing.assert_allclose(R, rr, atol=1e-5)
        np.testing.assert_allclose(T, tt, atol=1e-5)

    def test_04_nothing(self):
        """Empty layer calculation with mismatched slides ."""
        s = iadpython.Sample(a=0.5, b=0, g=0.0, n=1.4, quad_pts=4, n_above=1.5, n_below=1.5)
        rr, _, tt, _ = s.rt_matrices()

        R=np.array([[9.66127,0.00000,0.00000,0.00000],
                    [0.00000,2.58873,0.00000,0.00000],
                    [0.00000,0.00000,0.40318,0.00000],
                    [0.00000,0.00000,0.00000,0.52606]])

        T=np.array([[0.00000,0.00000,0.00000,0.00000],
                    [0.00000,0.00000,0.00000,0.00000],
                    [0.00000,0.00000,2.37358,0.00000],
                    [0.00000,0.00000,0.00000,6.13737]])

        np.testing.assert_allclose(R, rr, atol=1e-5)
        np.testing.assert_allclose(T, tt, atol=1e-5)

class B_finite_sandwich(unittest.TestCase):
    """Finite layer in air."""

    def test_01_sandwich(self):
        """Isotropic finite layer calculation."""
        s = iadpython.Sample(a=0.5, b=1, g=0.0, n=1.5, quad_pts=4)
        rr, _, tt, _ = s.rt_matrices()

        R=np.array([[8.51769,0.00000,0.00000,0.00000],
                    [0.00000,2.28231,0.00000,0.00000],
                    [0.00000,0.00000,0.35939,0.09497],
                    [0.00000,0.00000,0.09497,0.44074]])

        T=np.array([[0.00000,0.00000,0.00000,0.00000],
                    [0.00000,0.00000,0.00000,0.00000],
                    [0.00000,0.00000,0.89580,0.08370],
                    [0.00000,0.00000,0.08370,2.73853]])

        np.testing.assert_allclose(R, rr, atol=1e-5)
        np.testing.assert_allclose(T, tt, atol=1e-5)

    def test_02_sandwich(self):
        """Isotropic finite layer with slides."""
        s = iadpython.Sample(a=0.5, b=1, g=0.0, n=1.4, n_above=1.5, n_below=1.5)
        s.quad_pts = 4
        rr, _, tt, _ = s.rt_matrices()

        R=np.array([[9.66127,0.00000,0.00000,0.00000],
                    [0.00000,2.58873,0.00000,0.00000],
                    [0.00000,0.00000,0.33761,0.09481],
                    [0.00000,0.00000,0.09481,0.39353]])

        T=np.array([[0.00000,0.00000,0.00000,0.00000],
                    [0.00000,0.00000,0.00000,0.00000],
                    [0.00000,0.00000,0.76374,0.08298],
                    [0.00000,0.00000,0.08298,2.32842]])

        np.testing.assert_allclose(R, rr, atol=1e-5)
        np.testing.assert_allclose(T, tt, atol=1e-5)

    def test_03_sandwich(self):
        """Non-scattering finite layer with mismatched slides."""
        s = iadpython.Sample(a=0.0, b=1, g=0.0, n=1.4, n_above=1.5, n_below=1.6)
        s.quad_pts = 4
        rr03, rr30, tt03, tt30 = s.rt_matrices()

        R03=np.array([[9.66127,0.00000,0.00000,0.00000],
                      [0.00000,2.58873,0.00000,0.00000],
                      [0.00000,0.00000,0.23608,0.00000],
                      [0.00000,0.00000,0.00000,0.32119]])

        R30=np.array([[9.66127,0.00000,0.00000,0.00000],
                      [0.00000,2.58873,0.00000,0.00000],
                      [0.00000,0.00000,0.28404,0.00000],
                      [0.00000,0.00000,0.00000,0.41427]])

        T03=np.array([[0.00000,0.00000,0.00000,0.00000],
                      [0.00000,0.00000,0.00000,0.00000],
                      [0.00000,0.00000,0.66071,0.00000],
                      [0.00000,0.00000,0.00000,2.21386]])

        T30=np.array([[0.00000,0.00000,0.00000,0.00000],
                      [0.00000,0.00000,0.00000,0.00000],
                      [0.00000,0.00000,0.66071,0.00000],
                      [0.00000,0.00000,0.00000,2.21386]])

        np.testing.assert_allclose(R03, rr03, atol=1e-5)
        np.testing.assert_allclose(R30, rr30, atol=1e-5)
        np.testing.assert_allclose(T03, tt03, atol=1e-5)
        np.testing.assert_allclose(T30, tt30, atol=1e-5)

    def test_04_sandwich(self):
        """Isotropic finite layer with top slide only."""
        s = iadpython.Sample(a=0.5, b=1, g=0.0, n=1.4, n_above=1.5, n_below=1.0)
        s.quad_pts = 4
        rr03, rr30, tt03, tt30 = s.rt_matrices()

        R03=np.array([[9.66127,0.00000,0.00000,0.00000],
                      [0.00000,2.58873,0.00000,0.00000],
                      [0.00000,0.00000,0.33325,0.09392],
                      [0.00000,0.00000,0.09392,0.38170]])

        R30=np.array([[9.66127,0.00000,0.00000,0.00000],
                      [0.00000,2.58873,0.00000,0.00000],
                      [0.00000,0.00000,0.29330,0.09784],
                      [0.00000,0.00000,0.09784,0.30804]])

        T03=np.array([[0.00000,0.00000,0.00000,0.00000],
                      [0.00000,0.00000,0.00000,0.00000],
                      [0.00000,0.00000,0.77759,0.08400],
                      [0.00000,0.00000,0.08356,2.36008]])

        T30=np.array([[0.00000,0.00000,0.00000,0.00000],
                      [0.00000,0.00000,0.00000,0.00000],
                      [0.00000,0.00000,0.77759,0.08356],
                      [0.00000,0.00000,0.08400,2.36008]])

        np.testing.assert_allclose(R03, rr03, atol=1e-5)
        np.testing.assert_allclose(R30, rr30, atol=1e-5)
        np.testing.assert_allclose(T03, tt03, atol=1e-5)
        np.testing.assert_allclose(T30, tt30, atol=1e-5)

    def test_05_sandwich(self):
        """Isotropic finite layer with mismatched slides."""
        s = iadpython.Sample(a=0.5, b=1, g=0.0, n=1.4, n_above=1.5, n_below=1.6)
        s.quad_pts = 4
        rr03, rr30, tt03, tt30 = s.rt_matrices()
        R03=np.array([[9.66127,0.00000,0.00000,0.00000],
                      [0.00000,2.58873,0.00000,0.00000],
                      [0.00000,0.00000,0.34228,0.09582],
                      [0.00000,0.00000,0.09582,0.40788]])

        R30=np.array([[9.66127,0.00000,0.00000,0.00000],
                      [0.00000,2.58873,0.00000,0.00000],
                      [0.00000,0.00000,0.38508,0.09142],
                      [0.00000,0.00000,0.09142,0.49721]])

        T03=np.array([[0.00000,0.00000,0.00000,0.00000],
                      [0.00000,0.00000,0.00000,0.00000],
                      [0.00000,0.00000,0.74890,0.08193],
                      [0.00000,0.00000,0.08218,2.29000]])

        T30=np.array([[0.00000,0.00000,0.00000,0.00000],
                      [0.00000,0.00000,0.00000,0.00000],
                      [0.00000,0.00000,0.74890,0.08218],
                      [0.00000,0.00000,0.08193,2.29000]])

        np.testing.assert_allclose(R03, rr03, atol=1e-5)
        np.testing.assert_allclose(R30, rr30, atol=1e-5)
        np.testing.assert_allclose(T03, tt03, atol=1e-5)
        np.testing.assert_allclose(T30, tt30, atol=1e-5)

    def test_06_sandwich(self):
        """Isotropic finite layer with mismatched slides the hard way."""
        s = iadpython.Sample(a=0.5, b=1, g=0.0, n=1.4, n_above=1.5, n_below=1.6)
        s.quad_pts = 4
        R01, R10, T01, T10 = iadpython.boundary_matrices(s, top=True)
        R23, R32, T23, T32 = iadpython.boundary_matrices(s, top=False)
        R12, T12 = iadpython.simple_layer_matrices(s)
        R02, R20, T02, T20 = iadpython.add_layers(s, R01, R10, T01, T10, R12, R12, T12, T12)
        rr03, rr30, tt03, tt30 = iadpython.add_layers(s, R02, R20, T02, T20, R23, R32, T23, T32)

        R03=np.array([[9.66127,0.00000,0.00000,0.00000],
                      [0.00000,2.58873,0.00000,0.00000],
                      [0.00000,0.00000,0.34228,0.09582],
                      [0.00000,0.00000,0.09582,0.40788]])

        R30=np.array([[9.66127,0.00000,0.00000,0.00000],
                      [0.00000,2.58873,0.00000,0.00000],
                      [0.00000,0.00000,0.38508,0.09142],
                      [0.00000,0.00000,0.09142,0.49721]])

        T03=np.array([[0.00000,0.00000,0.00000,0.00000],
                      [0.00000,0.00000,0.00000,0.00000],
                      [0.00000,0.00000,0.74890,0.08193],
                      [0.00000,0.00000,0.08218,2.29000]])

        T30=np.array([[0.00000,0.00000,0.00000,0.00000],
                      [0.00000,0.00000,0.00000,0.00000],
                      [0.00000,0.00000,0.74890,0.08218],
                      [0.00000,0.00000,0.08193,2.29000]])

        np.testing.assert_allclose(R03, rr03, atol=1e-5)
        np.testing.assert_allclose(R30, rr30, atol=1e-5)
        np.testing.assert_allclose(T03, tt03, atol=1e-5)
        np.testing.assert_allclose(T30, tt30, atol=1e-5)

    def test_07_thick_non_scattering(self):
        """Thick non-scattering."""
        s = iadpython.Sample(a=0.0, b=100000.0, g=0.0, n=1.0, n_above=1.0, n_below=1.0)
        ur1, ut1, uru, utu = s.rt()
        self.assertAlmostEqual(ur1, 0.00000, delta=0.0001)
        self.assertAlmostEqual(ut1, 0.00000, delta=0.0001)
        self.assertAlmostEqual(uru, 0.00000, delta=0.0001)
        self.assertAlmostEqual(utu, 0.00000, delta=0.0001)

    def test_08_thick(self):
        """Thick scattering."""
        s = iadpython.Sample(a=0.8, b=100000.0, g=0.0, n=1.0, n_above=1.0, n_below=1.0)
        ur1, ut1, uru, utu = s.rt()
        self.assertAlmostEqual(ur1, 0.28525, delta=0.0001)
        self.assertAlmostEqual(ut1, 0.00000, delta=0.0001)
        self.assertAlmostEqual(uru, 0.34187, delta=0.0001)
        self.assertAlmostEqual(utu, 0.00000, delta=0.0001)

    def test_09_thick_non_absorbing(self):
        """Thick non-absorbing."""
        s = iadpython.Sample(a=1.0, b=100000.0, g=0.0, n=1.0, n_above=1.0, n_below=1.0)
        ur1, ut1, uru, utu = s.rt()
        self.assertAlmostEqual(ur1, 1.0000, delta=0.0001)
        self.assertAlmostEqual(ut1, 0.0000, delta=0.0001)
        self.assertAlmostEqual(uru, 1.0000, delta=0.0001)
        self.assertAlmostEqual(utu, 0.0000, delta=0.0001)

    def test_10_thick_non_scattering(self):
        """Thick non-scattering with slide."""
        ur1c, ut1c, uruc, utuc = iadpython.rt(1.4, 1.5, 0, 100000, 0)
        s = iadpython.Sample(a=0.0, b=100000.0, g=0.0, n=1.4, n_above=1.5, n_below=1.5, quad_pts=16)
        ur1, ut1, uru, utu = s.rt()
        self.assertAlmostEqual(ur1, ur1c, delta=0.0001)
        self.assertAlmostEqual(ut1, ut1c, delta=0.0001)
        self.assertAlmostEqual(uru, uruc, delta=0.0001)
        self.assertAlmostEqual(utu, utuc, delta=0.0001)

    def test_11_thick(self):
        """Thick scattering with slide."""
        ur1c, ut1c, uruc, utuc = iadpython.rt(1.4, 1.5, 0.8, 100000, 0)
        s = iadpython.Sample(a=0.8, b=100000.0, g=0.0, n=1.4, n_above=1.5, n_below=1.5, quad_pts=16)
        ur1, ut1, uru, utu = s.rt()
        self.assertAlmostEqual(ur1, ur1c, delta=0.0001)
        self.assertAlmostEqual(ut1, ut1c, delta=0.0001)
        self.assertAlmostEqual(uru, uruc, delta=0.0001)
        self.assertAlmostEqual(utu, utuc, delta=0.0001)

    def test_12_thick(self):
        """Thick anisotropic scattering with slide."""
        ur1c, ut1c, uruc, utuc = iadpython.rt(1.4, 1.5, 0.8, 100000, 0.9)
        s = iadpython.Sample(a=0.8, b=100000.0, g=0.9, n=1.4, n_above=1.5, n_below=1.5, quad_pts=16)
        ur1, ut1, uru, utu = s.rt()
        self.assertAlmostEqual(ur1, ur1c, delta=0.0001)
        self.assertAlmostEqual(ut1, ut1c, delta=0.0001)
        self.assertAlmostEqual(uru, uruc, delta=0.0001)
        self.assertAlmostEqual(utu, utuc, delta=0.0001)

    def test_13_thick_non_absorbing(self):
        """Thick non-absorbing with slide."""
        ur1c, ut1c, uruc, utuc = iadpython.rt(1.4, 1.5, 1.0, 100000, 0)
        s = iadpython.Sample(a=1.0, b=100000.0, g=0.0, n=1.4, n_above=1.5, n_below=1.5, quad_pts=16)
        ur1, ut1, uru, utu = s.rt()
        self.assertAlmostEqual(ur1, ur1c, delta=0.0001)
        self.assertAlmostEqual(ut1, ut1c, delta=0.0001)
        self.assertAlmostEqual(uru, uruc, delta=0.0001)
        self.assertAlmostEqual(utu, utuc, delta=0.0001)


if __name__ == '__main__':
    unittest.main()
