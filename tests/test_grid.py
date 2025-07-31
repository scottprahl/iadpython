# pylint: disable=invalid-name

"""Tests for Inverse Adding Doubling."""

import unittest
import numpy as np
import iadpython


class GridTest(unittest.TestCase):
    """Test grid construction."""

    def test_grid_01(self):
        """Grid for search_ag."""
        fixed_b = 4
        exp = iadpython.Experiment(r=0.1, t=0.5, default_b=fixed_b)
        exp.determine_search()
        grid = iadpython.Grid(N=5)
        grid.calc(exp, default=fixed_b)
        aa = [
            [0.000, 0.250, 0.500, 0.750, 1.000],
            [0.000, 0.250, 0.500, 0.750, 1.000],
            [0.000, 0.250, 0.500, 0.750, 1.000],
            [0.000, 0.250, 0.500, 0.750, 1.000],
            [0.000, 0.250, 0.500, 0.750, 1.000],
        ]
        bb = [
            [4.000, 4.000, 4.000, 4.000, 4.000],
            [4.000, 4.000, 4.000, 4.000, 4.000],
            [4.000, 4.000, 4.000, 4.000, 4.000],
            [4.000, 4.000, 4.000, 4.000, 4.000],
            [4.000, 4.000, 4.000, 4.000, 4.000],
        ]
        gg = [
            [-0.990, -0.990, -0.990, -0.990, -0.990],
            [-0.495, -0.495, -0.495, -0.495, -0.495],
            [0.000, 0.000, 0.000, 0.000, 0.000],
            [0.495, 0.495, 0.495, 0.495, 0.495],
            [0.990, 0.990, 0.990, 0.990, 0.990],
        ]
        ur1 = [
            [0.000, 0.121, 0.258, 0.437, 0.788],
            [0.000, 0.085, 0.187, 0.335, 0.769],
            [0.000, 0.046, 0.115, 0.242, 0.691],
            [0.000, 0.016, 0.048, 0.129, 0.512],
            [0.000, 0.000, 0.001, 0.002, 0.008],
        ]
        ut1 = [
            [0.018, 0.020, 0.025, 0.046, 0.212],
            [0.018, 0.020, 0.025, 0.041, 0.231],
            [0.018, 0.023, 0.032, 0.061, 0.309],
            [0.018, 0.030, 0.055, 0.124, 0.488],
            [0.018, 0.049, 0.133, 0.361, 0.992],
        ]

        np.testing.assert_allclose(grid.a, aa, atol=1e-5)
        np.testing.assert_allclose(grid.b, bb, atol=1e-5)
        np.testing.assert_allclose(grid.g, gg, atol=1e-5)
        np.testing.assert_allclose(grid.ur1, ur1, atol=1e-2)
        np.testing.assert_allclose(grid.ut1, ut1, atol=1e-2)

    def test_grid_02(self):
        """Matched slab with search_bg."""
        fixed_a = 0.5
        exp = iadpython.Experiment(r=0.1, t=0.5, default_a=fixed_a)
        exp.determine_search()
        grid = iadpython.Grid(N=5)
        grid.calc(exp, default=fixed_a)

        aa = [
            [0.5, 0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5, 0.5],
        ]
        bb = [
            [0.000, 2.500, 5.000, 7.500, 10.000],
            [0.000, 2.500, 5.000, 7.500, 10.000],
            [0.000, 2.500, 5.000, 7.500, 10.000],
            [0.000, 2.500, 5.000, 7.500, 10.000],
            [0.000, 2.500, 5.000, 7.500, 10.000],
        ]

        gg = [
            [-0.990, -0.990, -0.990, -0.990, -0.990],
            [-0.495, -0.495, -0.495, -0.495, -0.495],
            [0.000, 0.000, 0.000, 0.000, 0.000],
            [0.495, 0.495, 0.495, 0.495, 0.495],
            [0.990, 0.990, 0.990, 0.990, 0.990],
        ]

        np.testing.assert_allclose(grid.a, aa, atol=1e-5)
        np.testing.assert_allclose(grid.b, bb, atol=1e-5)
        np.testing.assert_allclose(grid.g, gg, atol=1e-5)

        print(grid)

    def test_grid_03(self):
        """Matched slab with search_ab."""
        fixed_g = 0.9
        exp = iadpython.Experiment(r=0.1, t=0.5, default_g=fixed_g)
        exp.determine_search()
        grid = iadpython.Grid(N=5)
        grid.calc(exp, default=fixed_g)

        aa = [
            [0.000, 0.250, 0.500, 0.750, 1.000],
            [0.000, 0.250, 0.500, 0.750, 1.000],
            [0.000, 0.250, 0.500, 0.750, 1.000],
            [0.000, 0.250, 0.500, 0.750, 1.000],
            [0.000, 0.250, 0.500, 0.750, 1.000],
        ]
        bb = [
            [0.000, 0.000, 0.000, 0.000, 0.000],
            [2.500, 2.500, 2.500, 2.500, 2.500],
            [5.000, 5.000, 5.000, 5.000, 5.000],
            [7.500, 7.500, 7.500, 7.500, 7.500],
            [10.00, 10.00, 10.00, 10.00, 10.00],
        ]
        gg = [
            [0.900, 0.900, 0.900, 0.900, 0.900],
            [0.900, 0.900, 0.900, 0.900, 0.900],
            [0.900, 0.900, 0.900, 0.900, 0.900],
            [0.900, 0.900, 0.900, 0.900, 0.900],
            [0.900, 0.900, 0.900, 0.900, 0.900],
        ]

        np.testing.assert_allclose(grid.a, aa, atol=1e-5)
        np.testing.assert_allclose(grid.b, bb, atol=1e-5)
        np.testing.assert_allclose(grid.g, gg, atol=1e-5)

    def test_grid_04(self):
        """Verify that minimum values are returned."""
        fixed_b = 4
        exp = iadpython.Experiment(r=0.1, t=0.5, default_b=fixed_b)
        exp.determine_search()
        grid = iadpython.Grid(N=21)
        grid.calc(exp, default=fixed_b)
        a, b, g = grid.min_abg(0.1, 0.5)

        self.assertAlmostEqual(a, 0.9, delta=1e-5)
        self.assertAlmostEqual(b, 4, delta=1e-5)
        self.assertAlmostEqual(g, 0.792, delta=1e-5)


if __name__ == "__main__":
    unittest.main()
