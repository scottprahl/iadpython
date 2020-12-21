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
        rr, tt = s.rt()

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
        rr, tt = s.rt()

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
        """Isotropic finite layer with mismatched slides."""
        s = iadpython.Sample(a=0.5, b=1, g=0.0, n=1.4, n_above=1.5, n_below=1.6)
        s.quad_pts = 4
        rr, tt = s.rt()

        R=np.array([[9.66127,0.00000,0.00000,0.00000],
                    [0.00000,2.58873,0.00000,0.00000],
                    [0.00000,0.00000,0.34228,0.09582],
                    [0.00000,0.00000,0.09582,0.40788]])

        T=np.array([[0.00000,0.00000,0.00000,0.00000],
                    [0.00000,0.00000,0.00000,0.00000],
                    [0.00000,0.00000,0.74890,0.08218],
                    [0.00000,0.00000,0.08193,2.29000]])

        np.testing.assert_allclose(R, rr, atol=1e-5)
        np.testing.assert_allclose(T, tt, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
