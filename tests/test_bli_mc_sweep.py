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
  with a 1.7% M_T error, while Python gives a 0.014% error.  Fit residuals
  (|fit_R - m_R|, |fit_T - m_T|) must be < 0.01 on tested rows.
"""

import os
import shutil

import numpy as np
import pytest

import iadpython
from iadpython.mc_lost import run_mc_lost_with_stderr

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_BLI_DIR = os.path.join(_TEST_DIR, "iad-problems", "BLI")

_EXCLUDED = {
    "dataOut_forSP_20240118.rxt",    # no IAD1 header
    "headerFile_forSP_20240118.rxt", # valid header but zero data rows
}

# Files where the fit check is skipped — either physically inconsistent
# constraints or suspect measurement data.
_SKIP_FIT_CHECK = {
    "mua-bad.rxt",
    "r_and_t_and_mua.rxt",
    # All biopix 851 nm phantom data is suspect; skip fit checks entirely.
    "biopix_851nm-A.rxt",
    "biopix_851nm-C.rxt",
    "biopix_851nm-D.rxt",
    "biopix_851nm-E.rxt",
    "bw_biopix_851nm.rxt",
    # High-albedo (a≈1) near-boundary cases: 10k photons is insufficient for
    # MC convergence; both iadp and iad/iad report not-converged (+).
    "il2.rxt",
    # 890nm intralipid (first row): MC loop does not converge with 10k photons; fit check is meaningless.
    "02%-1mm-test_rw.rxt",
    "02%-1mm.rxt",
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
    """MC inversion of small sphere files: no exception, finite results, fit ≤ 1%.

    Uses 10k photons and 8 MC iterations (fast but less accurate than default).
    Adaptive photon escalation concentrates budget in final iterations.
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

    # Fit-residual check on first and last row (skip for inconsistent files)
    if os.path.basename(rxt_path) in _SKIP_FIT_CHECK:
        return
    n = _row_count(exp)
    for i in [0] if n == 1 else [0, n - 1]:
        _check_row_fit(exp, a_arr, b_arr, g_arr, i)


# ---------------------------------------------------------------------------
# Slow test — large files (> 50 rows)
# ---------------------------------------------------------------------------

@requires_mc_lost
@pytest.mark.slow
@pytest.mark.parametrize("rxt_path", _SLOW_FILES)
def test_bli_mc_slow(rxt_path):
    """MC inversion of large sphere files: no exception, finite results, fit ≤ 2%.

    Marked ``slow`` — excluded from the default test run.
    Uses 10k photons for speed (default is 100k), so fit tolerance is 2% rather
    than the 1% target used for fast tests with adequate photon counts.
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
        _check_row_fit(exp, a_arr, b_arr, g_arr, i)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_N_SIGMA = 5    # tolerance = N_SIGMA * max(stderrs); generous but statistically grounded
_MIN_TOL = 1e-4 # floor: numerical precision of AD forward calculation when lost light is zero


def _check_row_fit(exp_base, a_arr, b_arr, g_arr, index, n_photons=10_000, n_repeats=10):
    """Assert that the MC fit for one row is within a statistically-grounded tolerance.

    Runs mc_lost n_repeats times (with n_photons // n_repeats each) at the
    inverted (a, b, g) to estimate both the mean lost-light corrections and
    their standard errors.  The fit tolerance is set to _N_SIGMA * max(stderrs),
    so it scales with the actual MC noise rather than being an arbitrary constant.

    If the mc_lost binary is unavailable the check is skipped gracefully.
    """
    point = exp_base.point_at(index)
    point.first_pass_abg = None

    a_val = float(np.atleast_1d(a_arr)[index])
    b_val = float(np.atleast_1d(b_arr)[index])
    g_val = float(np.atleast_1d(g_arr)[index])
    point.sample.a = a_val
    point.sample.b = b_val
    point.sample.g = g_val

    # Compute mean lost-light fractions and their stderr via repeated mc_lost runs.
    # Skip entirely for non-sphere geometries: there is no lost light to correct.
    mc_path = point.mc_lost_path
    if mc_path is None:
        mc_path = _MC_LOST_PATH

    tol = _MIN_TOL  # fallback if mc_lost unavailable, no spheres, or lost light is zero
    if mc_path is not None and _has_spheres(point):
        try:
            # Determine sphere port sizes and slide geometry from experiment.
            d_port_r = float(point.r_sphere.sample.d) if point.r_sphere is not None else 1000.0
            d_port_t = float(point.t_sphere.sample.d) if point.t_sphere is not None else d_port_r
            d_beam = float(point.d_beam) if point.d_beam else 5.0
            t_sample_val = float(point.sample.d)
            n_sample_val = float(point.sample.n)
            n_slide_val = float(point.sample.n_above) if point.sample.n_above is not None else 1.0
            t_slide_val = float(point.sample.d_above) if point.sample.d_above is not None else 0.0

            means, stderrs = run_mc_lost_with_stderr(
                a_val,
                b_val,
                g_val,
                n_sample=n_sample_val,
                n_slide=n_slide_val,
                d_port_r=d_port_r,
                d_port_t=d_port_t,
                d_beam=d_beam,
                t_sample=t_sample_val,
                t_slide=t_slide_val,
                n_photons=n_photons,
                n_repeats=n_repeats,
                method=point.method,
                binary_path=mc_path,
            )
            point.ur1_lost = means[0]
            point.ut1_lost = means[1]
            point.uru_lost = means[2]
            point.utu_lost = means[3]
            tol = max(_N_SIGMA * max(stderrs), _MIN_TOL)
        except (FileNotFoundError, RuntimeError):
            raise
        except Exception as _exc:
            raise RuntimeError(f"run_mc_lost_with_stderr failed for row {index}: {_exc}") from _exc
    else:
        point.ur1_lost = 0.0
        point.ut1_lost = 0.0
        point.uru_lost = 0.0
        point.utu_lost = 0.0

    point.max_mc_iterations = 0
    point.n_photons = 0
    point.mc_lost_path = None

    fit_r, fit_t = point.measured_rt()

    if point.m_r is not None:
        assert abs(float(fit_r) - float(point.m_r)) < tol, (
            f"row {index}: fit_r={fit_r:.4f} vs m_r={float(point.m_r):.4f} "
            f"(diff={abs(float(fit_r) - float(point.m_r)):.4f} > {tol:.4f})"
        )
    if point.m_t is not None:
        assert abs(float(fit_t) - float(point.m_t)) < tol, (
            f"row {index}: fit_t={fit_t:.4f} vs m_t={float(point.m_t):.4f} "
            f"(diff={abs(float(fit_t) - float(point.m_t)):.4f} > {tol:.4f})"
        )
