# pylint: disable=invalid-name
# pylint: disable=no-self-use

"""Tests for Inverse Adding Doubling."""

import unittest
import numpy as np
import iadpython


class GridTest(unittest.TestCase):
    """Test grid construction."""

    def test_grid_01(self):
        """Grid for search_ag"""
        exp = iadpython.Experiment(r=0.1, t=0.5, default_b=4)
        exp.determine_search()
        grid = iadpython.Grid(N=5)
        grid.calc(exp)
        aa = [[ 0.000, 0.250, 0.500, 0.750, 1.000],
              [ 0.000, 0.250, 0.500, 0.750, 1.000],
              [ 0.000, 0.250, 0.500, 0.750, 1.000],
              [ 0.000, 0.250, 0.500, 0.750, 1.000],
              [ 0.000, 0.250, 0.500, 0.750, 1.000]]
        bb = [[ 4.000, 4.000, 4.000, 4.000, 4.000],
              [ 4.000, 4.000, 4.000, 4.000, 4.000],
              [ 4.000, 4.000, 4.000, 4.000, 4.000],
              [ 4.000, 4.000, 4.000, 4.000, 4.000],
              [ 4.000, 4.000, 4.000, 4.000, 4.000]]
        gg = [[-0.950,-0.950,-0.950,-0.950,-0.950],
              [-0.475,-0.475,-0.475,-0.475,-0.475],
              [ 0.000, 0.000, 0.000, 0.000, 0.000],
              [ 0.475, 0.475, 0.475, 0.475, 0.475],
              [ 0.950, 0.950, 0.950, 0.950, 0.950]]
        ur1 = [[ 0.000, 0.119, 0.252, 0.424, 0.793],
               [ 0.000, 0.084, 0.185, 0.332, 0.767],
               [ 0.000, 0.046, 0.115, 0.242, 0.691],
               [ 0.000, 0.017, 0.051, 0.134, 0.523],
               [ 0.000, 0.001, 0.004, 0.011, 0.051]]
        ut1 = [[ 0.018, 0.020, 0.025, 0.045, 0.207],
               [ 0.018, 0.020, 0.025, 0.042, 0.233],
               [ 0.018, 0.023, 0.032, 0.061, 0.309],
               [ 0.018, 0.029, 0.053, 0.120, 0.477],
               [ 0.018, 0.047, 0.123, 0.331, 0.949]]
    
        np.testing.assert_allclose(grid.a, aa, atol=1e-5)
        np.testing.assert_allclose(grid.b, bb, atol=1e-5)
        np.testing.assert_allclose(grid.g, gg, atol=1e-5)
        np.testing.assert_allclose(grid.ur1, ur1, atol=1e-2)
        np.testing.assert_allclose(grid.ut1, ut1, atol=1e-2)

    def test_grid_02(self):
        """Matched slab with search_bg."""
        exp = iadpython.Experiment(r=0.1, t=0.5, default_a=0.5)
        exp.determine_search()
        grid = iadpython.Grid(N=5)
        grid.calc(exp)

        aa = [[0.5, 0.5, 0.5, 0.5, 0.5],
              [0.5, 0.5, 0.5, 0.5, 0.5],
              [0.5, 0.5, 0.5, 0.5, 0.5],
              [0.5, 0.5, 0.5, 0.5, 0.5],
              [0.5, 0.5, 0.5, 0.5, 0.5]]
        bb = [[ 0.000, 2.500, 5.000, 7.500,10.000],
              [ 0.000, 2.500, 5.000, 7.500,10.000],
              [ 0.000, 2.500, 5.000, 7.500,10.000],
              [ 0.000, 2.500, 5.000, 7.500,10.000],
              [ 0.000, 2.500, 5.000, 7.500,10.000]]

        gg = [[-0.950,-0.950,-0.950,-0.950,-0.950],
              [-0.475,-0.475,-0.475,-0.475,-0.475],
              [ 0.000, 0.000, 0.000, 0.000, 0.000],
              [ 0.475, 0.475, 0.475, 0.475, 0.475],
              [ 0.950, 0.950, 0.950, 0.950, 0.950]]

        np.testing.assert_allclose(grid.a, aa, atol=1e-5)
        np.testing.assert_allclose(grid.b, bb, atol=1e-5)
        np.testing.assert_allclose(grid.g, gg, atol=1e-5)

        print(grid)

    def test_grid_03(self):
        """Matched slab with search_ab."""
        exp = iadpython.Experiment(r=0.1, t=0.5, default_g=0.9)
        exp.determine_search()
        grid = iadpython.Grid(N=5)
        grid.calc(exp)

        aa = [[ 0.000, 0.250, 0.500, 0.750, 1.000],
              [ 0.000, 0.250, 0.500, 0.750, 1.000],
              [ 0.000, 0.250, 0.500, 0.750, 1.000],
              [ 0.000, 0.250, 0.500, 0.750, 1.000],
              [ 0.000, 0.250, 0.500, 0.750, 1.000]]
        bb = [[ 0.000, 0.000, 0.000, 0.000, 0.000],
              [ 2.500, 2.500, 2.500, 2.500, 2.500],
              [ 5.000, 5.000, 5.000, 5.000, 5.000],
              [ 7.500, 7.500, 7.500, 7.500, 7.500],
              [10.000,10.000,10.000,10.000,10.000]]
        gg = [[ 0.900, 0.900, 0.900, 0.900, 0.900],
              [ 0.900, 0.900, 0.900, 0.900, 0.900],
              [ 0.900, 0.900, 0.900, 0.900, 0.900],
              [ 0.900, 0.900, 0.900, 0.900, 0.900],
              [ 0.900, 0.900, 0.900, 0.900, 0.900]]

        np.testing.assert_allclose(grid.a, aa, atol=1e-5)
        np.testing.assert_allclose(grid.b, bb, atol=1e-5)
        np.testing.assert_allclose(grid.g, gg, atol=1e-5)

    def test_grid_04(self):
        """Grid for search_ag"""
        exp = iadpython.Experiment(r=0.1, t=0.5, default_b=4)
        exp.determine_search()
        grid = iadpython.Grid(N=21)
        grid.calc(exp)
        a,b,g = grid.min_abg(0.1,0.5)
        self.assertAlmostEqual(a, 0.9, delta=1e-5)
        self.assertAlmostEqual(b, 4, delta=1e-5)
        self.assertAlmostEqual(g, 0.855, delta=1e-5)

if __name__ == '__main__':
    unittest.main()
