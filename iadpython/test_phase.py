# pylint: disable=invalid-name
"""Tests for Redistribution Function."""

import unittest
import numpy as np
import iadpython.start
import iadpython.phase


class phi(unittest.TestCase):

    def test_01_phi(self):
        h = np.array([[ 6.84908,3.69902,0.65844,-0.09856,0.00000,-0.08633,0.22946,0.22803,-0.37395],
                      [ 3.69902,2.73731,1.42038, 0.67022,0.00000, 0.38894,0.10074,0.09250, 0.22803],
                      [ 0.65844,1.42038,1.78555, 1.43478,0.00000, 1.10818,0.49081,0.10074, 0.22946],
                      [-0.09856,0.67022,1.43478, 1.57558,0.00000, 1.49114,1.10818,0.38894,-0.08633],
                      [ 0.00000,0.00000,0.00000, 0.00000,0.00000, 0.00000,0.00000,0.00000, 0.00000],
                      [-0.08633,0.38894,1.10818, 1.49114,0.00000, 1.57558,1.43478,0.67022,-0.09856],
                      [ 0.22946,0.10074,0.49081, 1.10818,0.00000, 1.43478,1.78555,1.42038, 0.65844],
                      [ 0.22803,0.09250,0.10074, 0.38894,0.00000, 0.67022,1.42038,2.73731, 3.69902],
                      [-0.37395,0.22803,0.22946,-0.08633,0.00000,-0.09856,0.65844,3.69902, 6.84908]])
        
        s = iadpython.start.Slab()
        s.g = 0.9
        m = iadpython.start.Method(s)
        n = 4
        m.quad_pts = n
        m.update_quadrature(s)
        hp, hm = iadpython.phase.get_phi_legendre(s,m)

        hh =  h[n+1:,n+1:]
        np.testing.assert_allclose(hp, hh, rtol=1e-4)

        hh =  np.fliplr(h[n+1:,0:n])
        np.testing.assert_allclose(hm, hh, rtol=1e-4)

if __name__ == '__main__':
    unittest.main()
