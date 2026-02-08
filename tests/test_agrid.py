# pylint: disable=invalid-name
"""Tests for adaptive grid helper."""

import unittest
import iadpython


class AGridTest(unittest.TestCase):
    """Validate adaptive grid behavior."""

    def test_agrid_calc_holds_fixed_axis(self):
        """`find_ag` grid should keep b fixed at the provided default."""
        fixed_b = 4.0
        exp = iadpython.Experiment(r=0.1, t=0.5, default_b=fixed_b)
        exp.determine_search()

        grid = iadpython.AGrid(tol=0.02, max_depth=3)
        grid.calc(exp, default=fixed_b, search=exp.search)

        self.assertGreater(len(grid), 0)
        for _a, b, _g, _ur1, _ut1, _uru, _utu in grid:
            self.assertAlmostEqual(b, fixed_b, delta=1e-12)

    def test_agrid_min_abg_returns_best_match(self):
        """`min_abg` should return the point with the smallest |ΔR|+|ΔT|."""
        grid = iadpython.AGrid()
        grid.cache.put(0.1, 1.0, 0.0, 0.20, 0.20, 0.0, 0.0)
        grid.cache.put(0.9, 4.0, 0.7, 0.10, 0.50, 0.0, 0.0)
        grid.cache.put(0.3, 2.0, 0.2, 0.11, 0.51, 0.0, 0.0)

        a, b, g = grid.min_abg(0.10, 0.50)

        self.assertAlmostEqual(a, 0.9, delta=1e-12)
        self.assertAlmostEqual(b, 4.0, delta=1e-12)
        self.assertAlmostEqual(g, 0.7, delta=1e-12)

    def test_experiment_uses_adaptive_grid_for_two_parameter_search(self):
        """Experiment should default to adaptive grid for 2-parameter searches."""
        exp = iadpython.Experiment(r=0.1, t=0.5, default_g=0.5)
        exp.invert_rt()
        self.assertIsInstance(exp.grid, iadpython.AGrid)


if __name__ == "__main__":
    unittest.main()
