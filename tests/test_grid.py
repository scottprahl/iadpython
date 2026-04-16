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


class _DummyMeasuredSpaceExperiment:
    """Simple stub that makes corrected-space ranking differ from raw-space ranking."""

    def measurement_distance_from_raw(self, ur1, ut1, uru, utu, include_lost=True, a=None, b=None, g=None):
        """Return corrected (m_r, m_t, delta) for ranking grid candidates."""
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


def _make_find_bg_exp(fixed_a=0.9):
    """Return a configured scalar experiment for `find_bg` staleness checks."""
    exp = iadpython.Experiment(r=0.19, t=0.06, default_a=fixed_a)
    exp.sample.n = 1.4
    exp.sample.n_above = 1.0
    exp.sample.n_below = 1.0
    exp.sample.nu_0 = 1.0
    exp.useful_measurements()
    exp.determine_search()
    return exp


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

    def test_grid_min_abg_uses_corrected_measurement_space(self):
        """When an experiment is provided, ranking should use corrected-space distance."""
        grid = iadpython.Grid(N=2)
        grid.a = np.array([[0.1, 0.9], [0.1, 0.9]])
        grid.b = np.array([[0.0, 2.0], [0.0, 2.0]])
        grid.g = np.zeros((2, 2))
        grid.ur1 = np.array([[0.10, 0.11], [0.30, 0.40]])
        grid.ut1 = np.array([[0.50, 0.50], [0.20, 0.10]])
        grid.uru = np.zeros((2, 2))
        grid.utu = np.zeros((2, 2))

        a, b, g = grid.min_abg(0.10, 0.50, exp=_DummyMeasuredSpaceExperiment())

        self.assertAlmostEqual(a, 0.9, delta=1e-12)
        self.assertAlmostEqual(b, 2.0, delta=1e-12)
        self.assertAlmostEqual(g, 0.0, delta=1e-12)

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

    def test_grid_stale_matches_cweb_for_three_measurement_mu(self):
        """Three-measurement grids should invalidate when unscattered light changes."""
        exp = _make_find_ag_exp(include_u=True)
        grid = iadpython.Grid(N=5)
        grid.calc(exp, default=exp.sample.b)

        same = _make_find_ag_exp(include_u=True)
        self.assertFalse(grid.is_stale(same, same.sample.b, search=same.search))

        changed = _make_find_ag_exp(include_u=True)
        changed.m_u = 0.25
        changed.useful_measurements()
        changed.determine_search()
        self.assertTrue(grid.is_stale(changed, changed.sample.b, search=changed.search))

        two_measure = _make_find_ag_exp(include_u=False)
        self.assertFalse(grid.is_stale(two_measure, two_measure.sample.b, search=two_measure.search))

    def test_grid_stale_invalidates_on_boundary_and_angle_changes(self):
        """Grid reuse should depend on refractive index, slides, and incidence angle."""
        exp = _make_find_ag_exp(include_u=True)
        grid = iadpython.Grid(N=5)
        grid.calc(exp, default=exp.sample.b)

        changed_n = _make_find_ag_exp(include_u=True)
        changed_n.sample.n = 1.45
        self.assertTrue(grid.is_stale(changed_n, changed_n.sample.b, search=changed_n.search))

        changed_top = _make_find_ag_exp(include_u=True)
        changed_top.sample.n_above = 1.4
        self.assertTrue(grid.is_stale(changed_top, changed_top.sample.b, search=changed_top.search))

        changed_bottom = _make_find_ag_exp(include_u=True)
        changed_bottom.sample.n_below = 1.4
        self.assertTrue(grid.is_stale(changed_bottom, changed_bottom.sample.b, search=changed_bottom.search))

        changed_angle = _make_find_ag_exp(include_u=True)
        changed_angle.sample.nu_0 = 0.7
        self.assertTrue(grid.is_stale(changed_angle, changed_angle.sample.b, search=changed_angle.search))

    def test_grid_stale_invalidates_when_find_bg_fixed_a_changes(self):
        """`find_bg` grids should invalidate when the fixed albedo changes."""
        exp = _make_find_bg_exp(fixed_a=0.9)
        grid = iadpython.Grid(N=5)
        grid.calc(exp, default=exp.default_a)

        same = _make_find_bg_exp(fixed_a=0.9)
        self.assertFalse(grid.is_stale(same, same.default_a, search=same.search))

        changed = _make_find_bg_exp(fixed_a=0.8)
        self.assertTrue(grid.is_stale(changed, changed.default_a, search=changed.search))


if __name__ == "__main__":
    unittest.main()
