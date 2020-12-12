# pylint: disable=invalid-name
"""Tests for initial layer thickness Function."""

import unittest
import numpy as np
import iadpython.start
import iadpython.phase


class phi(unittest.TestCase):

    def test_01_thin(self):
        s = iadpython.start.Slab()
        s.a = 1.0
        s.b = 100
        s.g = 0.0

        m = iadpython.start.Method(s)
        m.update_quadrature(s)
        m.update_start_depth()
        np.testing.assert_approx_equal(m.b_thinnest, 0.048828125)
        

    def test_01_igi(self):
        s = iadpython.start.Slab()
        s.a = 1.0
        s.b = 100
        s.g = 0.0

        m = iadpython.start.Method(s)
        m.update_quadrature(s)
        m.update_start_depth()

        hp, hm = iadpython.phase.get_phi_legendre(s,m)
        rr, tt = iadpython.start.igi_start(m,hp,hm)
        
        r = np.array([[ 1.55547, 0.33652, 0.17494, 0.13780],
                      [ 0.33652, 0.07281, 0.03785, 0.02981],
                      [ 0.17494, 0.03785, 0.01968, 0.01550],
                      [ 0.13780, 0.02981, 0.01550, 0.01221]])

        t = np.array([[13.04576, 0.33652, 0.17494, 0.13780],
                      [ 0.33652, 2.84330, 0.03785, 0.02981],
                      [ 0.17494, 0.03785, 1.83038, 0.01550],
                      [ 0.13780, 0.02981, 0.01550, 7.62158]])

        np.testing.assert_allclose(r, rr, atol=1e-5)
        np.testing.assert_allclose(t, tt, atol=1e-5)

    def test_02_igi(self):
        s = iadpython.start.Slab()
        s.a = 1.0
        s.b = 100
        s.g = 0.9

        m = iadpython.start.Method(s)
        m.update_quadrature(s)
        m.update_start_depth()

        hp, hm = iadpython.phase.get_phi_legendre(s,m)
        rr, tt = iadpython.start.igi_start(m,hp,hm)
        
        r = np.array([[ 3.19060, 0.51300, 0.09360,-0.01636],
                      [ 0.51300, 0.04916, 0.00524, 0.00941],
                      [ 0.09360, 0.00524, 0.00250, 0.00486],
                      [-0.01636, 0.00941, 0.00486,-0.00628]])

        t = np.array([[ 9.56148, 0.66419, 0.16129,-0.01868],
                      [ 0.66419, 2.80843, 0.07395, 0.02700],
                      [ 0.16129, 0.07395, 1.83985, 0.07886],
                      [-0.01868, 0.02700, 0.07886, 7.57767]])
                      
        np.testing.assert_allclose(r, rr, atol=1e-5)
        np.testing.assert_allclose(t, tt, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
