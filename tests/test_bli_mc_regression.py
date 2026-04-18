"""MC regression tests: compare Python MC inversion against known C-binary references.

These tests require the ``mc_lost`` binary on PATH or in ``iad/mc_lost``.
They are skipped automatically when the binary is not found.

Run with::

    .venv/bin/pytest tests/test_bli_mc_regression.py -v

Reference values were produced by running::

    iad/iad -c 0 -H 1 tests/iad-problems/BLI/Polystyrene-Spheres/pairs/1060-d=1/1060-d=1.rxt

using IAD version 3-16-3 (18 Jul 2025).
"""

import os
import shutil

import numpy as np
import pytest

import iadpython

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_BLI_DIR = os.path.join(_TEST_DIR, "iad-problems", "BLI")


def _find_mc_lost():
    """Return path to mc_lost binary, or None if not found."""
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


def _enable_mc(exp):
    """Configure experiment to use MC with the discovered mc_lost binary."""
    exp.mc_lost_path = _MC_LOST_PATH
    if exp.max_mc_iterations == 0:
        exp.max_mc_iterations = 8
    if exp.n_photons == 0:
        exp.n_photons = 100_000
    return exp


def _mua_musp(a_arr, b_arr, g_arr, d_mm):
    """Compute mu_a and mu_sp arrays from (a, b, g) and sample thickness."""
    a = np.atleast_1d(np.asarray(a_arr, dtype=float))
    b = np.atleast_1d(np.asarray(b_arr, dtype=float))
    g = np.atleast_1d(np.asarray(g_arr, dtype=float))
    mu_a = (1.0 - a) * b / d_mm
    mu_sp = (1.0 - g) * a * b / d_mm
    return mu_a, mu_sp


# ---------------------------------------------------------------------------
# Phase 2.1 — 1060-d=1 vs C reference
# ---------------------------------------------------------------------------

# Reference produced by: iad/iad -c 0 -H 1 1060-d=1.rxt  (IAD 3-16-3, 18 Jul 2025)
# Columns: (mu_a, mu_s')  [1/mm].  Only the first 13 rows (0-indexed 0-12) are
# used; rows 13-15 in 1060-d=1.rxt have c>0 (column c = 0.05/0.10/0.15) and the
# C binary repeats row 12's answer verbatim for those rows, indicating it did not
# converge.  Python correctly finds different solutions for c>0 rows.
_1060D1_REF_MUA = np.array([
    3.9121, 3.9132, 3.8930, 3.8947, 3.8815, 3.8727, 3.8911, 3.9007,
    3.9087, 3.9048, 3.8995, 3.8984, 3.8727,
])
_1060D1_REF_MUSP = np.array([
    0.8330, 0.8321, 0.8302, 0.8317, 0.8279, 0.8258, 0.8698, 0.8442,
    0.8355, 0.8330, 0.8331, 0.8385, 0.8258,
])
# Number of rows to compare (only the c=0 rows where C converged)
_1060D1_N_COMPARE = 13

_1060D1_RXT = os.path.join(
    _BLI_DIR, "Polystyrene-Spheres", "pairs", "1060-d=1", "1060-d=1.rxt"
)


@requires_mc_lost
def test_1060d1_mc_matches_c_reference():
    """Python MC inversion of 1060-d=1 must agree with C reference within 0.05/mm.

    The C reference was generated with ``iad -c 0 -H 1`` (MC enabled by default).
    Tolerance of 0.05 accounts for stochastic MC noise in both implementations.
    """
    exp = iadpython.read_rxt(_1060D1_RXT)
    _enable_mc(exp)

    a, b, g = exp.invert_rt()

    d_mm = float(exp.sample.d)
    mu_a, mu_sp = _mua_musp(a, b, g, d_mm)

    assert len(mu_a) == 16, (
        f"row count mismatch: got {len(mu_a)}, expected 16"
    )

    # Only compare the first 13 rows (c=0); rows 13-15 have c>0 and the C
    # binary did not converge for those (repeats row 12 answer verbatim).
    tol = 0.05  # 1/mm
    for i in range(_1060D1_N_COMPARE):
        assert abs(mu_a[i] - _1060D1_REF_MUA[i]) < tol, (
            f"row {i}: mu_a={mu_a[i]:.4f} vs ref={_1060D1_REF_MUA[i]:.4f} "
            f"(diff={abs(mu_a[i] - _1060D1_REF_MUA[i]):.4f} > {tol})"
        )
        assert abs(mu_sp[i] - _1060D1_REF_MUSP[i]) < tol, (
            f"row {i}: mu_sp={mu_sp[i]:.4f} vs ref={_1060D1_REF_MUSP[i]:.4f} "
            f"(diff={abs(mu_sp[i] - _1060D1_REF_MUSP[i]):.4f} > {tol})"
        )


# ---------------------------------------------------------------------------
# Phase 2.2 — il-unconstrained-T vs C reference  (PENDING)
# ---------------------------------------------------------------------------
# The C reference ``il-unconstrained-T.txt`` was generated with:
#   iad -a 0 il-unconstrained-T
# The corresponding input file ``il-unconstrained-T.rxt`` is not in the repo.
# Using ``il.rxt`` with default_a=0 does not reproduce the reference values
# (Python gives mu_a ≈ 0.46 vs C reference ≈ 0.57 at row 0).
# TODO: locate or reconstruct ``il-unconstrained-T.rxt`` and add the test.
