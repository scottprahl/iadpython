"""Tests for MC lost-light convergence rules in `iadpython.iad`.

These tests are unit-level and do not require the external `mc_lost` binary.
They patch the inversion and MC-update hooks so the MC loop can be exercised
directly against the same stopping logic used by the CWEB implementation.
"""

import iadpython


def _make_experiment():
    exp = iadpython.Experiment(sample=iadpython.Sample(a=0.2, b=1.0, g=0.0))
    exp.num_spheres = 1
    exp.mc_lost_path = "/fake/mc_lost"
    exp.MC_tolerance = 0.01
    exp.max_mc_iterations = 5
    return exp


def test_mc_iteration_waits_for_mu_stability(monkeypatch):
    """MC loop should stop on `mu_a` / `mu_s'` stability, not lost-fraction stability."""
    exp = _make_experiment()

    states = iter(
        [
            (0.2, 1.000, 0.0),  # initial inversion
            (0.2, 1.120, 0.0),  # lost fractions already stable, but mu values still changing
            (0.2, 1.125, 0.0),  # now mu values are stable enough to stop
        ]
    )

    updates = iter(
        [
            (0.0005, 0.0005, 0.0005, 0.0, 0.0),
            (0.0004, 0.0004, 0.0004, 0.0, 0.0),
        ]
    )

    def fake_invert_scalar_rt(hot_start=None, initial_simplex=None):  # pylint: disable=unused-argument
        a, b, g = next(states)
        exp.sample.a = a
        exp.sample.b = b
        exp.sample.g = g
        return a, b, g

    def fake_update_lost_light(_a, _b, _g):
        return next(updates)

    monkeypatch.setattr(exp, "invert_scalar_rt", fake_invert_scalar_rt)
    monkeypatch.setattr(exp, "_update_lost_light", fake_update_lost_light)

    a, b, g = exp._invert_scalar_with_mc()  # pylint: disable=protected-access

    assert exp._mc_iterations == 2  # pylint: disable=protected-access
    assert (a, b, g) == (0.2, 1.125, 0.0)


def test_mc_iteration_honors_direct_loss_guard(monkeypatch):
    """MC loop should keep iterating while direct lost-light terms still move too much."""
    exp = _make_experiment()

    states = iter(
        [
            (0.2, 1.000, 0.0),  # initial inversion
            (0.2, 1.005, 0.0),  # mu values already stable enough
            (0.2, 1.006, 0.0),  # direct-loss guard clears on this iteration
        ]
    )

    updates = iter(
        [
            (0.0020, 0.0020, 0.0005, 0.0, 0.0),  # diff_ur1 > 0.001 -> continue
            (0.0005, 0.0005, 0.0005, 0.0, 0.0),  # direct-loss guard clears
        ]
    )

    def fake_invert_scalar_rt(hot_start=None, initial_simplex=None):  # pylint: disable=unused-argument
        a, b, g = next(states)
        exp.sample.a = a
        exp.sample.b = b
        exp.sample.g = g
        return a, b, g

    def fake_update_lost_light(_a, _b, _g):
        return next(updates)

    monkeypatch.setattr(exp, "invert_scalar_rt", fake_invert_scalar_rt)
    monkeypatch.setattr(exp, "_update_lost_light", fake_update_lost_light)

    a, b, g = exp._invert_scalar_with_mc()  # pylint: disable=protected-access

    assert exp._mc_iterations == 2  # pylint: disable=protected-access
    assert (a, b, g) == (0.2, 1.006, 0.0)
