"""MC sweep tests: run all sphere-containing BLI .rxt files through MC inversion.

Every valid BLI file with integrating spheres should invert with MC enabled
without raising an exception and produce finite optical properties.

These tests require the ``mc_lost`` binary on PATH or in ``iad/mc_lost``.
Large files (> 50 rows) are marked ``slow`` and excluded from the default run.

Run with::

    # Fast tests only (excluded from 'make test'):
    .venv/bin/pytest tests/test_bli_mc_sweep.py -v -m "not slow"

    # All tests including slow files:
    .venv/bin/pytest tests/test_bli_mc_sweep.py -v

Background on convergence:
  Python MC outperforms the CWEB C binary on hard cases — e.g., for
  Polystyrene-Spheres/d=1060nm_0.87%_1mm.rxt the C binary reports not-converged
  with a 1.7% M_T error, while Python gives a 0.014% error.  The ``found``
  flag may be False on some rows (tight AD tolerance vs MC-corrected measurements)
  but the physical result is correct.  Tests here do NOT require found=True.
"""

import os
import shutil

import numpy as np
import pytest

import iadpython

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_BLI_DIR = os.path.join(_TEST_DIR, "iad-problems", "BLI")

_EXCLUDED = {
    "dataOut_forSP_20240118.rxt",    # no IAD1 header
    "headerFile_forSP_20240118.rxt", # valid header but zero data rows
}

# Files with physically inconsistent per-row constraints — skip fit check.
_SKIP_FIT_CHECK = {
    "mua-bad.rxt",
    "r_and_t_and_mua.rxt",
}


def _find_mc_lost():
    repo_root = os.path.dirname(_TEST_DIR)
    for candidate in (
        os.path.join(repo_root, "iad", "mc_lost"),
        os.path.join(repo_root, "iad", "src", "mc_lost"),
    ):
        if os.path.exists(candidate):
            return candidate
    return shutil.which("mc_lost")


_MC_LOST_PATH = _find_mc_lost()

requires_mc_lost = pytest.mark.skipif(
    _MC_LOST_PATH is None,
    reason="mc_lost binary not found (build with: cd iad && make mc_lost)",
)


def _has_spheres(exp):
    """Return True if this experiment uses integrating spheres."""
    ns = np.atleast_1d(np.asarray(exp.num_spheres))
    return bool(np.any(ns > 0))


def _row_count(exp):
    for attr in ("m_r", "m_t", "m_u"):
        val = getattr(exp, attr)
        if val is not None and not np.isscalar(val):
            return len(np.atleast_1d(val))
    return 1


def _enable_mc(exp, n_photons=10_000, max_iters=8):
    exp.mc_lost_path = _MC_LOST_PATH
    if exp.max_mc_iterations == 0:
        exp.max_mc_iterations = max_iters
    if exp.n_photons == 0:
        exp.n_photons = n_photons
    return exp


def _collect_sphere_rxt_files():
    """Return parametrized sphere-containing .rxt files, partitioned by size."""
    fast = []   # ≤ 50 rows
    slow = []   # > 50 rows
    for root, _dirs, files in os.walk(_BLI_DIR):
        for fname in files:
            if not fname.endswith(".rxt") or fname in _EXCLUDED:
                continue
            full = os.path.join(root, fname)
            try:
                exp = iadpython.read_rxt(full)
            except Exception:
                continue
            if not _has_spheres(exp):
                continue
            n = _row_count(exp)
            label = os.path.relpath(full, _TEST_DIR)
            param = pytest.param(full, id=label)
            if n <= 50:
                fast.append(param)
            else:
                slow.append(param)
    fast.sort(key=lambda p: p.values[0])
    slow.sort(key=lambda p: p.values[0])
    return fast, slow


_FAST_FILES, _SLOW_FILES = _collect_sphere_rxt_files()

# ---------------------------------------------------------------------------
# Fast test — small files (≤ 50 rows)
# ---------------------------------------------------------------------------

@requires_mc_lost
@pytest.mark.parametrize("rxt_path", _FAST_FILES)
def test_bli_mc_fast(rxt_path):
    """MC inversion of small sphere files: no exception, finite results, fit ≤ 10%, found=True.

    Uses 10k photons and 8 MC iterations (fast but less accurate than default).
    The ``found`` flag IS checked on first and last row — the Bug A/B fixes
    ensure the MC fixed point is correctly detected even at low photon counts.
    """
    exp = iadpython.read_rxt(rxt_path)
    _enable_mc(exp, n_photons=10_000, max_iters=8)

    a, b, g = exp.invert_rt()

    a_arr = np.atleast_1d(np.asarray(a, dtype=float))
    b_arr = np.atleast_1d(np.asarray(b, dtype=float))
    g_arr = np.atleast_1d(np.asarray(g, dtype=float))

    assert np.all(np.isfinite(a_arr)), f"non-finite albedo: {a_arr}"
    assert np.all(np.isfinite(b_arr)), f"non-finite optical thickness: {b_arr}"
    assert np.all(np.isfinite(g_arr)), f"non-finite anisotropy: {g_arr}"

    # found=True and fit-residual check on first and last row
    if os.path.basename(rxt_path) in _SKIP_FIT_CHECK:
        return
    n = _row_count(exp)
    for i in [0] if n == 1 else [0, n - 1]:
        _check_row_found(exp, i)
        _check_row_fit(exp, a_arr, b_arr, g_arr, i, tol=0.10)


# ---------------------------------------------------------------------------
# Slow test — large files (> 50 rows)
# ---------------------------------------------------------------------------

@requires_mc_lost
@pytest.mark.slow
@pytest.mark.parametrize("rxt_path", _SLOW_FILES)
def test_bli_mc_slow(rxt_path):
    """MC inversion of large sphere files: no exception, finite results, fit ≤ 10%, found=True.

    Marked ``slow`` — excluded from the default test run.
    Uses 10k photons for speed; default is 100k.
    """
    exp = iadpython.read_rxt(rxt_path)
    _enable_mc(exp, n_photons=10_000, max_iters=8)

    a, b, g = exp.invert_rt()

    a_arr = np.atleast_1d(np.asarray(a, dtype=float))
    b_arr = np.atleast_1d(np.asarray(b, dtype=float))
    g_arr = np.atleast_1d(np.asarray(g, dtype=float))

    assert np.all(np.isfinite(a_arr)), f"non-finite albedo: {a_arr}"
    assert np.all(np.isfinite(b_arr)), f"non-finite optical thickness: {b_arr}"
    assert np.all(np.isfinite(g_arr)), f"non-finite anisotropy: {g_arr}"

    if os.path.basename(rxt_path) in _SKIP_FIT_CHECK:
        return
    n = _row_count(exp)
    for i in [0] if n == 1 else [0, n - 1]:
        _check_row_found(exp, i)
        _check_row_fit(exp, a_arr, b_arr, g_arr, i, tol=0.10)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_row_found(exp_base, index):
    """Assert that MC inversion reports found=True for one row."""
    point = exp_base.point_at(index)
    _enable_mc(point, n_photons=10_000, max_iters=8)
    point._invert_scalar_with_mc()  # pylint: disable=protected-access
    assert point.found, (
        f"row {index}: found=False after MC inversion "
        f"(final_distance={point.final_distance:.6f}, tolerance={point.tolerance:.6f})"
    )


def _check_row_fit(exp_base, a_arr, b_arr, g_arr, index, tol=0.10):
    """Assert that the MC fit for one row is within tol of the measurement."""
    point = exp_base.point_at(index)
    # Use no-MC for the fit check (measured_rt() doesn't run MC)
    point.mc_lost_path = None
    point.max_mc_iterations = 0
    point.n_photons = 0
    point.ur1_lost = 0.0
    point.ut1_lost = 0.0
    point.uru_lost = 0.0
    point.utu_lost = 0.0
    point.first_pass_abg = None
    point.sample.a = float(np.atleast_1d(a_arr)[index])
    point.sample.b = float(np.atleast_1d(b_arr)[index])
    point.sample.g = float(np.atleast_1d(g_arr)[index])

    fit_r, fit_t = point.measured_rt()

    if point.m_r is not None:
        assert abs(float(fit_r) - float(point.m_r)) < tol, (
            f"row {index}: fit_r={fit_r:.4f} vs m_r={float(point.m_r):.4f} "
            f"(diff={abs(float(fit_r) - float(point.m_r)):.4f} > {tol})"
        )
    if point.m_t is not None:
        assert abs(float(fit_t) - float(point.m_t)) < tol, (
            f"row {index}: fit_t={fit_t:.4f} vs m_t={float(point.m_t):.4f} "
            f"(diff={abs(float(fit_t) - float(point.m_t)):.4f} > {tol})"
        )
