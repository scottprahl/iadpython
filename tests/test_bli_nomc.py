"""No-MC baseline tests for all valid BLI .rxt files.

Every valid BLI input file should invert without raising an exception and
produce a fit whose residual is within 5% of the measured values.

Run with::

    .venv/bin/pytest tests/test_bli_nomc.py -v

"""

import os

import numpy as np
import pytest

import iadpython

# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_BLI_DIR = os.path.join(_TEST_DIR, "iad-problems", "BLI")

# Files that cannot be inverted: not valid .rxt files or contain no data rows.
_EXCLUDED = {
    "dataOut_forSP_20240118.rxt",    # no IAD1 header — raw data file
    "headerFile_forSP_20240118.rxt", # IAD1 header but zero data rows
}

# Files where the per-row mu_a constraint (column A) is physically inconsistent
# with the measured R/T at some rows, so fit residuals can be large.
# These still run (no exception, finite results) but the fit check is skipped.
_SKIP_FIT_CHECK = {
    "mua-bad.rxt",          # explicitly bad mu_a=1e-6 at first row
    "r_and_t_and_mua.rxt",  # mu_a=1e-6 at first row; inconsistent with sphere measurement
}


def _collect_bli_rxt_files():
    """Return sorted list of pytest.param for all valid BLI .rxt files."""
    paths = []
    for root, _dirs, files in os.walk(_BLI_DIR):
        for fname in files:
            if fname.endswith(".rxt") and fname not in _EXCLUDED:
                full = os.path.join(root, fname)
                label = os.path.relpath(full, _TEST_DIR)
                paths.append(pytest.param(full, id=label))
    return sorted(paths, key=lambda p: p.values[0])


_BLI_RXT_FILES = _collect_bli_rxt_files()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _disable_mc(exp):
    """Force experiment onto the no-MC code path."""
    exp.max_mc_iterations = 0
    exp.mc_lost_path = None
    exp.n_photons = 0
    exp.ur1_lost = 0.0
    exp.ut1_lost = 0.0
    exp.uru_lost = 0.0
    exp.utu_lost = 0.0
    exp.first_pass_abg = None
    return exp


def _row_count(exp):
    """Number of scalar inversions in the experiment."""
    for attr in ("m_r", "m_t", "m_u"):
        val = getattr(exp, attr)
        if val is not None and not np.isscalar(val):
            return len(np.atleast_1d(val))
    return 1


def _check_row_fit(exp_base, a_arr, b_arr, g_arr, index, tol=0.05):
    """Assert that the fit for one row is within *tol* of the measurement."""
    point = exp_base.point_at(index)
    _disable_mc(point)
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


# ---------------------------------------------------------------------------
# Single combined test — one inversion per file
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rxt_path", _BLI_RXT_FILES)
def test_bli_nomc(rxt_path):
    """Invert a BLI file without MC: no exception, finite results, fit within 5%.

    Files in _SKIP_FIT_CHECK have physically inconsistent per-row mu_a
    constraints at some rows; those are tested for no-exception and finite
    results but the fit-residual assertion is skipped.
    """
    exp = iadpython.read_rxt(rxt_path)
    _disable_mc(exp)

    # --- must not raise ---
    a, b, g = exp.invert_rt()

    # --- all results must be finite ---
    a_arr = np.atleast_1d(np.asarray(a, dtype=float))
    b_arr = np.atleast_1d(np.asarray(b, dtype=float))
    g_arr = np.atleast_1d(np.asarray(g, dtype=float))

    assert np.all(np.isfinite(a_arr)), f"non-finite albedo: {a_arr}"
    assert np.all(np.isfinite(b_arr)), f"non-finite optical thickness: {b_arr}"
    assert np.all(np.isfinite(g_arr)), f"non-finite anisotropy: {g_arr}"

    # --- fit residual for first and last row ---
    if os.path.basename(rxt_path) in _SKIP_FIT_CHECK:
        return

    n = _row_count(exp)
    for i in ([0] if n == 1 else [0, n - 1]):
        _check_row_fit(exp, a_arr, b_arr, g_arr, i, tol=0.05)
