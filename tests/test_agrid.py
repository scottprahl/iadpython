# pylint: disable=invalid-name
"""Tests for adaptive grid helper."""

import unittest
import numpy as np
import iadpython


class _DummyMeasuredSpaceExperiment:
    """Simple stub that makes corrected-space ranking differ from raw-space ranking."""

    def measurement_distance_from_raw(self, ur1, ut1, uru, utu, include_lost=True, a=None, b=None, g=None):
        del uru, utu, include_lost, a, g
        m_r = ur1 - 0.01 * b
        m_t = ut1
        delta = abs(m_r - 0.09) + abs(m_t - 0.50)
        return m_r, m_t, delta


def _make_find_ag_exp(include_u=True):
    """Return a configured scalar experiment for white-box stale-grid checks."""
    exp = iadpython.Experiment(r=0.1, t=0.5, u=0.2 if include_u else None, default_b=4.0)
    exp.sample.n = 1.4
    exp.sample.n_above = 1.0
    exp.sample.n_below = 1.0
    exp.sample.nu_0 = 1.0
    exp.useful_measurements()
    exp.determine_search()
    return exp


class AGridTest(unittest.TestCase):
    """Validate adaptive grid behavior."""

    def test_agrid_calc_holds_fixed_axis(self):
        """`find_ag` grid should keep b fixed at the provided default."""
        fixed_b = 4.0
        exp = iadpython.Experiment(r=0.1, t=0.5, default_b=fixed_b)
        exp.useful_measurements()
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

    def test_agrid_min_abg_uses_corrected_measurement_space(self):
        """When an experiment is provided, ranking should use corrected-space distance."""
        grid = iadpython.AGrid()
        grid.cache.put(0.1, 0.0, 0.0, 0.10, 0.50, 0.0, 0.0)
        grid.cache.put(0.9, 2.0, 0.7, 0.11, 0.50, 0.0, 0.0)

        a, b, g = grid.min_abg(0.10, 0.50, exp=_DummyMeasuredSpaceExperiment())

        self.assertAlmostEqual(a, 0.9, delta=1e-12)
        self.assertAlmostEqual(b, 2.0, delta=1e-12)
        self.assertAlmostEqual(g, 0.7, delta=1e-12)

    def test_experiment_auto_selects_agrid_for_find_ab(self):
        """Auto mode: find_ab uses AGrid (adaptive sampling handles edge cases)."""
        exp = iadpython.Experiment(r=0.1, t=0.5, default_g=0.5)
        exp.invert_rt()
        self.assertIsInstance(exp.grid, iadpython.AGrid)

    def test_experiment_auto_find_ab_uses_fine_agrid(self):
        """Auto mode: find_ab now uses the fine adaptive grid settings."""
        exp = iadpython.Experiment(r=0.1, t=0.5, default_g=0.5)
        exp.invert_rt()
        self.assertIsInstance(exp.grid, iadpython.AGrid)
        self.assertAlmostEqual(exp.grid.tol, 0.01)
        self.assertEqual(exp.grid.max_depth, 8)
        self.assertEqual(exp.grid.min_depth, 3)

    def test_experiment_auto_selects_agrid_for_find_bg(self):
        """Auto mode: find_bg uses AGrid(fine) (33% fewer optimizer evals)."""
        exp = iadpython.Experiment(r=0.19, t=0.06, default_a=0.9)
        exp.invert_rt()
        self.assertIsInstance(exp.grid, iadpython.AGrid)

    def test_experiment_auto_find_bg_uses_fine_agrid(self):
        """Auto mode: find_bg AGrid uses tol=0.01 and max_depth=8."""
        exp = iadpython.Experiment(r=0.19, t=0.06, default_a=0.9)
        exp.invert_rt()
        self.assertIsInstance(exp.grid, iadpython.AGrid)
        self.assertAlmostEqual(exp.grid.tol, 0.01)
        self.assertEqual(exp.grid.max_depth, 8)

    def test_experiment_override_forces_grid(self):
        """use_adaptive_grid=False forces Grid even for find_bg."""
        exp = iadpython.Experiment(r=0.19, t=0.06, default_a=0.9)
        exp.use_adaptive_grid = False
        exp.invert_rt()
        self.assertIsInstance(exp.grid, iadpython.Grid)

    def test_agrid_find_ab_uses_grid_nonlinear_a_axis(self):
        """Adaptive find_ab sampling should use the same nonlinear a axis as Grid."""
        exp = iadpython.Experiment(r=0.1, t=0.5, default_g=0.5)
        exp.useful_measurements()
        exp.determine_search()

        grid = iadpython.AGrid(tol=1.0, max_depth=1, min_depth=1)
        grid.calc(exp, default=exp.sample.g, search=exp.search)

        a_values = sorted({entry.a for entry in grid.cache._data})
        expected = [0.0, 0.15625, 0.5, 0.84375, 1.0]
        np.testing.assert_allclose(a_values, expected, atol=1e-12)

    def test_agrid_find_ag_uses_grid_nonlinear_g_axis(self):
        """Adaptive find_ag sampling should use the same nonlinear g axis as Grid."""
        exp = iadpython.Experiment(r=0.1, t=0.5, default_b=4.0)
        exp.useful_measurements()
        exp.determine_search()

        grid = iadpython.AGrid(tol=1.0, max_depth=1, min_depth=1)
        grid.calc(exp, default=exp.sample.b, search=exp.search)

        g_values = sorted({entry.g for entry in grid.cache._data})
        max_abs_g = 0.999999
        expected = [
            -max_abs_g,
            -0.6874993125,
            0.0,
            0.6874993125,
            max_abs_g,
        ]
        np.testing.assert_allclose(g_values, expected, atol=1e-12)

    def test_agrid_stale_invalidates_on_cweb_context_changes(self):
        """Adaptive grids should reuse only when the CWEB raw-RT context is unchanged."""
        exp = _make_find_ag_exp(include_u=True)
        grid = iadpython.AGrid(tol=0.02, max_depth=3)
        grid.calc(exp, default=exp.sample.b, search=exp.search)

        same = _make_find_ag_exp(include_u=True)
        self.assertFalse(grid.is_stale(same, same.sample.b, search=same.search))

        changed_mu = _make_find_ag_exp(include_u=True)
        changed_mu.m_u = 0.25
        changed_mu.useful_measurements()
        changed_mu.determine_search()
        self.assertTrue(grid.is_stale(changed_mu, changed_mu.sample.b, search=changed_mu.search))

        changed_angle = _make_find_ag_exp(include_u=True)
        changed_angle.sample.nu_0 = 0.7
        self.assertTrue(grid.is_stale(changed_angle, changed_angle.sample.b, search=changed_angle.search))


if __name__ == "__main__":
    unittest.main()
