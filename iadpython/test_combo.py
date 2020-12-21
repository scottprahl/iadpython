# pylint: disable=invalid-name
# pylint: disable=bad-whitespace
# pylint: disable=no-self-use

"""Tests for slide-sample-slide combinations."""

import unittest
import numpy as np
import iadpython

class Air_sandwich(unittest.TestCase):
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
        s = iadpython.Sample(a=0.5, b=1, g=0.0, n=1.0, n_above=1.5, n_below=1.0)
        s.quad_pts = 4
        rr03, rr30, tt03, tt30 = s.rt_matrices()

        R03=np.array([[19.41024,0.06124,0.04285,0.03524],
                      [0.06124,0.83640,0.10265,0.08790],
                      [0.04285,0.10265,0.25297,0.07688],
                      [0.03524,0.08790,0.07688,0.68248]])

        R30=np.array([[0.80071,0.31300,0.18613,0.15223],
                      [0.31300,0.21395,0.14597,0.12471],
                      [0.18613,0.14597,0.12100,0.09369],
                      [0.15223,0.12471,0.09369,0.16481]])

        T03=np.array([[0.00501,0.04926,0.07167,0.07041],
                      [0.01783,0.29506,0.09238,0.08524],
                      [0.02114,0.07934,0.57611,0.07273],
                      [0.02056,0.07307,0.07259,2.78119]])

        T30=np.array([[0.00501,0.01783,0.02114,0.02056],
                      [0.04926,0.29506,0.07934,0.07307],
                      [0.07167,0.09238,0.57611,0.07259],
                      [0.07041,0.08524,0.07273,2.78119]])

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

if __name__ == '__main__':
    unittest.main()
