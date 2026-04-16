"""Tests for the hot-start / warm-start behavior of the MC lost-light loop.

Corresponds to tests A–D in MC_PLAN.md:

A. hot_start bypasses grid rebuild on second inversion
B. 1 MC iteration (mocked) produces a finite result; first_pass_abg is set correctly
C. warm start for MC re-inversion is the previous (a, b, g)
D. MC iteration does NOT reset self.grid to None between calls

None of these tests require the mc_lost binary.  Lost-light values are either
mocked (set directly on the experiment) or left at zero so the no-MC path is
exercised.
"""

# pylint: disable=protected-access

import math
import unittest
from unittest.mock import patch

import iadpython

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SPHERE_CONFIG = {
    "d_sphere": 150.0,
    "d_sample": 10.0,
    "d_third": 10.0,
    "d_detector": 5.0,
    "r_detector": 0.04,
    "r_wall": 0.98,
    "r_std": 0.99,
}


def _make_exp(a=0.5, b=1.0, g=0.0, n=1.0, n_above=1.0, quad_pts=4):
    """Build a minimal 1-sphere find_ag experiment with known optical props."""
    sample = iadpython.Sample(a=a, b=b, g=g, n=n, n_above=n_above, quad_pts=quad_pts)
    r_sphere = iadpython.Sphere(**SPHERE_CONFIG, refl=True)
    t_sphere = iadpython.Sphere(**SPHERE_CONFIG, refl=False)
    exp = iadpython.Experiment(
        sample=sample,
        num_spheres=1,
        r_sphere=r_sphere,
        t_sphere=t_sphere,
    )
    exp.method = "substitution"
    exp.default_g = g
    # Compute the forward measurement so we have realistic m_r, m_t
    m_r, m_t = exp.measured_rt()
    exp.m_r = float(m_r)
    exp.m_t = float(m_t)
    return exp


# ---------------------------------------------------------------------------
# Test A: hot_start bypasses grid rebuild
# ---------------------------------------------------------------------------


class TestHotStartBypassesGrid(unittest.TestCase):
    """A. hot_start=... causes invert_scalar_rt to skip grid evaluation."""

    def test_hot_start_uses_zero_grid_evals(self):
        """When hot_start is passed, _grid_evals should be 0."""
        exp = _make_exp()
        # Run once to establish first-pass solution and build grid
        a0, b0, g0 = exp.invert_scalar_rt()

        # Second call with hot_start: should NOT rebuild the grid
        a1, b1, _g1 = exp.invert_scalar_rt(hot_start=(a0, b0, g0))

        self.assertEqual(exp._grid_evals, 0, "hot_start should produce _grid_evals == 0")
        # The result should be close to the first-pass solution
        self.assertAlmostEqual(a1, a0, delta=0.05, msg="hot_start a should be close to first-pass a")
        self.assertAlmostEqual(b1, b0, delta=0.2, msg="hot_start b should be close to first-pass b")

    def test_cold_start_uses_nonzero_grid_evals(self):
        """Without hot_start, grid is rebuilt and _grid_evals > 0."""
        exp = _make_exp()
        # Force the grid to None so a rebuild definitely happens
        exp.grid = None
        exp.invert_scalar_rt()
        self.assertGreater(exp._grid_evals, 0, "cold start should evaluate grid points")

    def test_hot_start_optimizer_evals_positive(self):
        """hot_start should still drive the optimizer to convergence."""
        exp = _make_exp()
        a0, b0, g0 = exp.invert_scalar_rt()
        exp.invert_scalar_rt(hot_start=(a0, b0, g0))
        # Optimizer should have taken at least a few evaluations
        self.assertGreater(exp._optimizer_evals, 0)


# ---------------------------------------------------------------------------
# Test B: first_pass_abg is set; MC re-inversion gives a finite result
# ---------------------------------------------------------------------------


class TestFirstPassAbg(unittest.TestCase):
    """B. first_pass_abg is set correctly; MC re-inversion with mocked lost light.

    Still produces a finite (a, b, g).
    """

    def test_first_pass_abg_matches_no_mc_result(self):
        """first_pass_abg should equal the no-MC inversion result."""
        exp = _make_exp()
        exp.mc_lost_path = None  # no MC
        exp.max_mc_iterations = 0
        a, b, g = exp.invert_rt()
        self.assertIsNotNone(exp.first_pass_abg)
        a0, b0, g0 = exp.first_pass_abg
        self.assertAlmostEqual(a0, a, delta=1e-6)
        self.assertAlmostEqual(b0, b, delta=1e-6)
        self.assertAlmostEqual(g0, g, delta=1e-6)

    def test_mocked_mc_iteration_produces_finite_result(self):
        """With mocked _update_lost_light that returns small fixed values.

        One MC iteration should produce a finite (a, b, g).
        """
        exp = _make_exp()

        # Pretend mc_lost_path is set so the MC loop executes
        exp.mc_lost_path = "/fake/mc_lost"
        exp.max_mc_iterations = 1
        exp.num_spheres = 1

        def _fake_update(_a, _b, _g):
            # Inject a tiny non-zero lost-light perturbation
            exp.ur1_lost = 0.002
            exp.ut1_lost = 0.002
            exp.uru_lost = 0.001
            exp.utu_lost = 0.001
            # Return (max_diff, diff_ur1, diff_ut1, diff_uru, diff_utu)
            return 0.002, 0.002, 0.002, 0.001, 0.001

        with patch.object(exp, "_update_lost_light", side_effect=_fake_update):
            a, b, g = exp.invert_rt()

        self.assertTrue(math.isfinite(a), f"a={a} is not finite")
        self.assertTrue(math.isfinite(b), f"b={b} is not finite")
        self.assertTrue(math.isfinite(g), f"g={g} is not finite")
        self.assertIsNotNone(exp.first_pass_abg)
        self.assertEqual(exp._mc_iterations, 1)


# ---------------------------------------------------------------------------
# Test C: warm start is the previous (a, b, g), not a grid lookup
# ---------------------------------------------------------------------------


class TestWarmStartIsPreviousSolution(unittest.TestCase):
    """C. After the first pass, invert_scalar_rt is called with hot_start == first_pass_abg."""

    def test_mc_reversion_warm_starts_from_first_pass(self):
        """Record calls to invert_scalar_rt; the second call must receive hot_start == first_pass_abg."""
        exp = _make_exp()
        exp.mc_lost_path = "/fake/mc_lost"
        exp.max_mc_iterations = 1
        exp.num_spheres = 1

        recorded_hot_starts = []
        original_isr = exp.invert_scalar_rt

        def _recording_isr(hot_start=None, initial_simplex=None):
            recorded_hot_starts.append(hot_start)
            return original_isr(hot_start=hot_start, initial_simplex=initial_simplex)

        def _fake_update(_a, _b, _g):
            exp.ur1_lost = 0.001
            exp.ut1_lost = 0.001
            exp.uru_lost = 0.0005
            exp.utu_lost = 0.0005
            return 0.001, 0.001, 0.001, 0.0005, 0.0005

        with patch.object(exp, "_update_lost_light", side_effect=_fake_update):
            with patch.object(exp, "invert_scalar_rt", side_effect=_recording_isr):
                exp.invert_rt()

        # First call should have no hot_start (grid path)
        self.assertIsNone(recorded_hot_starts[0], "First invert_scalar_rt call should have no hot_start")

        # Second call should have hot_start == first_pass_abg
        self.assertIsNotNone(recorded_hot_starts[1], "Second invert_scalar_rt call must have a hot_start")
        hs_a, hs_b, hs_g = recorded_hot_starts[1]
        fp_a, fp_b, fp_g = exp.first_pass_abg
        self.assertAlmostEqual(hs_a, fp_a, places=12)
        self.assertAlmostEqual(hs_b, fp_b, places=12)
        self.assertAlmostEqual(hs_g, fp_g, places=12)


# ---------------------------------------------------------------------------
# Test D: grid is NOT reset to None between MC iterations
# ---------------------------------------------------------------------------


class TestGridPreservedAcrossMCIterations(unittest.TestCase):
    """D. self.grid is never set to None during MC iteration."""

    def test_grid_not_reset_during_mc_loop(self):
        """After each MC iteration, self.grid must not be None."""
        exp = _make_exp()
        exp.mc_lost_path = "/fake/mc_lost"
        exp.max_mc_iterations = 2
        exp.num_spheres = 1

        grid_states_after_update = []
        call_count = 0

        def _fake_update(_a, _b, _g):
            nonlocal call_count
            call_count += 1
            exp.ur1_lost = 0.001 * call_count
            exp.ut1_lost = 0.001 * call_count
            exp.uru_lost = 0.0005 * call_count
            exp.utu_lost = 0.0005 * call_count
            # Snapshot grid right after lost-light update
            grid_states_after_update.append(exp.grid)
            # Return large diff so the loop doesn't terminate early
            return 0.01, 0.01, 0.01, 0.005, 0.005

        with patch.object(exp, "_update_lost_light", side_effect=_fake_update):
            exp.invert_rt()

        # The grid should have been built on the first pass (grid != None)
        # and should never have been set to None during the MC loop.
        # grid_states_after_update[i] is the grid BEFORE the subsequent
        # invert_scalar_rt(hot_start=...) call — it must not be None.
        for i, grid_state in enumerate(grid_states_after_update):
            self.assertIsNotNone(
                grid_state,
                f"grid was None after MC iteration {i + 1} lost-light update "
                f"(the 'self.grid = None' bug is present)",
            )


if __name__ == "__main__":
    unittest.main()
