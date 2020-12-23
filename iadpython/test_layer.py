# pylint: disable=invalid-name
# pylint: disable=bad-whitespace
# pylint: disable=no-self-use

"""Tests for finite layer thicknesses."""

import unittest
import numpy as np
import iadpython

class layer(unittest.TestCase):
    """Starting layer calculations."""

    def test_00_double(self):
        """Starting layer thickness calculation."""
        s = iadpython.ad.Sample(a=0.5, b=1, g=0.0, n=1, quad_pts=4)
        b_min = iadpython.start.starting_thickness(s)
        np.testing.assert_approx_equal(b_min, 0.0625)

    def test_01_double(self):
        """Adding isotropic layers."""
        s = iadpython.ad.Sample(a=0.5, b=1, g=0.0, n=1, quad_pts=4)
        rr, tt = iadpython.start.thinnest_layer(s)

        b=0.12500
        rr, tt = iadpython.add_layers_basic(s, rr, tt, rr, rr, tt, tt)
        R=np.array([[0.72851,0.22898,0.12610,0.10068],
                    [0.22898,0.07678,0.04267,0.03414],
                    [0.12610,0.04267,0.02374,0.01900],
                    [0.10068,0.03414,0.01900,0.01521]])

        T=np.array([[6.44003,0.21795,0.12290,0.09867],
                    [0.21795,2.39264,0.04243,0.03399],
                    [0.12290,0.04243,1.67063,0.01896],
                    [0.09867,0.03399,0.01896,7.07487]])

        np.testing.assert_allclose(R, rr, atol=1e-5)
        np.testing.assert_allclose(T, tt, atol=1e-5)

        b=0.25000
        rr, tt = iadpython.add_layers_basic(s, rr, tt, rr, rr, tt, tt)
        R=np.array([[0.78844,0.28216,0.15973,0.12840],
                    [0.28216,0.12615,0.07411,0.06010],
                    [0.15973,0.07411,0.04379,0.03556],
                    [0.12840,0.06010,0.03556,0.02889]])

        T=np.array([[1.64804,0.22726,0.14271,0.11749],
                    [0.22726,1.82607,0.07217,0.05886],
                    [0.14271,0.07217,1.44835,0.03518],
                    [0.11749,0.05886,0.03518,6.25854]])

        np.testing.assert_allclose(R, rr, atol=1e-5)
        np.testing.assert_allclose(T, tt, atol=1e-5)

        b=0.50000
        rr, tt = iadpython.add_layers_basic(s, rr, tt, rr, rr, tt, tt)
        R=np.array([[0.79808,0.30346,0.17574,0.14215],
                    [0.30346,0.17598,0.11217,0.09291],
                    [0.17574,0.11217,0.07299,0.06076],
                    [0.14215,0.09291,0.06076,0.05064]])

        T=np.array([[0.13566,0.15475,0.12350,0.10763],
                    [0.15475,1.06976,0.10102,0.08554],
                    [0.12350,0.10102,1.09191,0.05817],
                    [0.10763,0.08554,0.05817,4.90038]])

        np.testing.assert_allclose(R, rr, atol=1e-5)
        np.testing.assert_allclose(T, tt, atol=1e-5)

        b=1.00000
        rr, tt = iadpython.add_layers_basic(s, rr, tt, rr, rr, tt, tt)
        R=np.array([[0.80010,0.31085,0.18343,0.14931],
                    [0.31085,0.20307,0.14031,0.11912],
                    [0.18343,0.14031,0.10265,0.08848],
                    [0.14931,0.11912,0.08848,0.07658]])

        T=np.array([[0.01792,0.06195,0.07718,0.07528],
                    [0.06195,0.37419,0.09621,0.08830],
                    [0.07718,0.09621,0.62536,0.07502],
                    [0.07528,0.08830,0.07502,3.00924]])

        np.testing.assert_allclose(R, rr, atol=1e-5)
        np.testing.assert_allclose(T, tt, atol=1e-5)

    def test_02_double(self):
        """Isotropic slab with matched boundaries."""
        s = iadpython.ad.Sample(a=0.5, b=1, g=0.0, n=1, quad_pts=4)
        rr, tt = iadpython.simple_layer_matrices(s)

        R=np.array([[0.80010,0.31085,0.18343,0.14931],
                    [0.31085,0.20307,0.14031,0.11912],
                    [0.18343,0.14031,0.10265,0.08848],
                    [0.14931,0.11912,0.08848,0.07658]])

        T=np.array([[0.01792,0.06195,0.07718,0.07528],
                    [0.06195,0.37419,0.09621,0.08830],
                    [0.07718,0.09621,0.62536,0.07502],
                    [0.07528,0.08830,0.07502,3.00924]])

        np.testing.assert_allclose(R, rr, atol=1e-5)
        np.testing.assert_allclose(T, tt, atol=1e-5)

    def test_03_double(self):
        """Anisotropic slab with matched boundaries."""
        s = iadpython.ad.Sample(a=0.5, b=1, g=0.9, n=1, quad_pts=4)
        rr, tt = iadpython.simple_layer_matrices(s)

        R=np.array([[0.57675,0.15818,0.03296,-0.00369],
                    [0.15818,0.04710,0.00836,0.01059],
                    [0.03296,0.00836,0.00434,0.00708],
                    [-0.00369,0.01059,0.00708,-0.00831]])

        T=np.array([[0.01817,0.06990,0.03529,0.00115],
                    [0.06990,0.71075,0.06555,0.02857],
                    [0.03529,0.06555,0.91100,0.10190],
                    [0.00115,0.02857,0.10190,4.24324]])

        np.testing.assert_allclose(R, rr, atol=1e-5)
        np.testing.assert_allclose(T, tt, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
