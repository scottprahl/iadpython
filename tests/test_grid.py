# pylint: disable=invalid-name

"""Tests for Grid construction and spacing."""

import unittest
import numpy as np
import iadpython

# CWEB constants mirrored here for white-box checks
_MAX_ABS_G = 0.999999
_MIN_LOG_B = -8.0
_MAX_LOG_B_AB = 8.0
_MAX_LOG_B_BG = 10.0


def _nonlinear_a(N):
    x = np.linspace(0, 1, N)
    return 1.0 - (1.0 - x) ** 2 * (1.0 + 2.0 * x)


def _nonlinear_g(N):
    x = np.linspace(0, 1, N)
    return (1.0 - 2.0 * (1.0 - x) ** 2 * (1.0 + 2.0 * x)) * _MAX_ABS_G


class GridTest(unittest.TestCase):
    """Test grid construction."""

    def test_grid_01_find_ag_axes(self):
        """find_ag: a and g vary with correct nonlinear spacing; b is fixed."""
        fixed_b = 4.0
        N = 5
        exp = iadpython.Experiment(r=0.1, t=0.5, default_b=fixed_b)
        exp.useful_measurements()
        exp.determine_search()
        grid = iadpython.Grid(N=N)
        grid.calc(exp, default=fixed_b)

        # b must be fixed everywhere
        np.testing.assert_allclose(grid.b, fixed_b, atol=1e-10)

        # a varies along columns (axis 1) — nonlinear spacing
        expected_a = _nonlinear_a(N)
        for row in range(N):
            np.testing.assert_allclose(grid.a[row, :], expected_a, atol=1e-10)

        # g varies along rows (axis 0) — nonlinear spacing
        expected_g = _nonlinear_g(N)
        for col in range(N):
            np.testing.assert_allclose(grid.g[:, col], expected_g, atol=1e-10)

        # a and g must lie within physical bounds
        self.assertAlmostEqual(grid.a.min(), 0.0, delta=1e-10)
        self.assertAlmostEqual(grid.a.max(), 1.0, delta=1e-10)
        self.assertAlmostEqual(grid.g.min(), -_MAX_ABS_G, delta=1e-10)
        self.assertAlmostEqual(grid.g.max(), _MAX_ABS_G, delta=1e-10)

    def test_grid_02_find_bg_axes(self):
        """find_bg: b and g vary; b is log-spaced to exp(+10); a is fixed."""
        fixed_a = 0.5
        N = 5
        exp = iadpython.Experiment(r=0.1, t=0.5, default_a=fixed_a)
        exp.useful_measurements()
        exp.determine_search()
        grid = iadpython.Grid(N=N)
        grid.calc(exp, default=fixed_a)

        # a must be fixed everywhere
        np.testing.assert_allclose(grid.a, fixed_a, atol=1e-10)

        # b varies along columns — log spacing with BG limit
        expected_b = np.exp(np.linspace(_MIN_LOG_B, _MAX_LOG_B_BG, N))
        for row in range(N):
            np.testing.assert_allclose(grid.b[row, :], expected_b, rtol=1e-6)

        # g varies along rows — nonlinear spacing
        expected_g = _nonlinear_g(N)
        for col in range(N):
            np.testing.assert_allclose(grid.g[:, col], expected_g, atol=1e-10)

        # range checks
        self.assertAlmostEqual(grid.b.min(), np.exp(_MIN_LOG_B), delta=1e-6)
        self.assertAlmostEqual(grid.b.max(), np.exp(_MAX_LOG_B_BG), delta=1.0)

    def test_grid_03_find_ab_axes(self):
        """find_ab: a and b vary; b is log-spaced to exp(+8); g is fixed."""
        fixed_g = 0.9
        N = 5
        exp = iadpython.Experiment(r=0.1, t=0.5, default_g=fixed_g)
        exp.useful_measurements()
        exp.determine_search()
        grid = iadpython.Grid(N=N)
        grid.calc(exp, default=fixed_g)

        # g must be fixed everywhere
        np.testing.assert_allclose(grid.g, fixed_g, atol=1e-10)

        # a varies along columns — nonlinear spacing
        expected_a = _nonlinear_a(N)
        for row in range(N):
            np.testing.assert_allclose(grid.a[row, :], expected_a, atol=1e-10)

        # b varies along rows — log spacing with AB limit
        expected_b = np.exp(np.linspace(_MIN_LOG_B, _MAX_LOG_B_AB, N))
        for col in range(N):
            np.testing.assert_allclose(grid.b[:, col], expected_b, rtol=1e-6)

        # range checks
        self.assertAlmostEqual(grid.b.min(), np.exp(_MIN_LOG_B), delta=1e-6)
        self.assertAlmostEqual(grid.b.max(), np.exp(_MAX_LOG_B_AB), delta=1.0)

    def test_grid_04_min_abg(self):
        """min_abg should return a point within the grid domain."""
        fixed_b = 4.0
        exp = iadpython.Experiment(r=0.1, t=0.5, default_b=fixed_b)
        exp.useful_measurements()
        exp.determine_search()
        grid = iadpython.Grid(N=21)
        grid.calc(exp, default=fixed_b)
        a, b, g = grid.min_abg(0.1, 0.5)

        # b must equal the fixed value
        self.assertAlmostEqual(b, fixed_b, delta=1e-10)

        # a and g must be within bounds
        self.assertGreaterEqual(a, 0.0)
        self.assertLessEqual(a, 1.0)
        self.assertGreaterEqual(g, -_MAX_ABS_G)
        self.assertLessEqual(g, _MAX_ABS_G)

    def test_grid_uru_utu_populated(self):
        """Grid.calc() must populate uru and utu alongside ur1/ut1."""
        fixed_g = 0.5
        exp = iadpython.Experiment(r=0.1, t=0.5, default_g=fixed_g)
        exp.useful_measurements()
        exp.determine_search()
        grid = iadpython.Grid(N=5)
        grid.calc(exp, default=fixed_g)

        # uru and utu must be present and non-trivially computed
        self.assertIsNotNone(grid.uru)
        self.assertIsNotNone(grid.utu)
        self.assertEqual(grid.uru.shape, (5, 5))
        self.assertEqual(grid.utu.shape, (5, 5))

        # Interior cells (avoiding a=1 corner) should have uru in [0, 1]
        for i in range(5):
            for j in range(4):  # skip last column (a=1)
                self.assertGreaterEqual(grid.uru[i, j], 0.0)
                self.assertLessEqual(grid.uru[i, j], 1.0)


if __name__ == "__main__":
    unittest.main()
