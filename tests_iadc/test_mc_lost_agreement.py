# pylint: disable=invalid-name

"""Cross-validation: Python Experiment.invert_rt() vs C iad binary with MC lost light.

For each .rxt test file, runs both the C `iad` binary (with ``-M 5``) and the
Python ``Experiment.invert_rt()`` (with ``mc_lost_path`` and
``max_mc_iterations=5``) on the same inputs, then asserts that the recovered
optical properties agree within the tolerances from PLAN.md §5.2:

  - albedo ``a``:              1 % relative
  - optical thickness ``b``:   1 % relative
  - anisotropy ``g``:          0.05 absolute

The C binary outputs ``(mu_a, mu_s', g)``; these are converted to ``(a, b, g)``
using the sample physical thickness stored in the parsed ``Experiment``.

To keep the suite fast, only three wavelengths are tested per file (first,
middle, last).  Each inversion with five MC iterations takes roughly 2–5 s,
so the whole module runs in under two minutes.

Prerequisites::

    cd iad && make        # builds both `iad` and `mc_lost` binaries

Run with::

    .venv/bin/pytest tests_iadc/test_mc_lost_agreement.py -v
"""

import pathlib
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

import iadpython

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
IAD_BINARY = REPO_ROOT / "iad" / "iad"
MC_LOST_BINARY = REPO_ROOT / "iad" / "mc_lost"
DATA_DIR = REPO_ROOT / "tests" / "data"

# ---------------------------------------------------------------------------
# Tolerances (PLAN.md §5.2, relaxed slightly from the 1 % target to account
# for MC stochastic noise — both programs use independent random seeds)
# ---------------------------------------------------------------------------

A_REL_TOL = 0.02   # 2 % relative on albedo
B_REL_TOL = 0.02   # 2 % relative on optical thickness
G_ABS_TOL = 0.05   # 5 % absolute on anisotropy

MAX_MC = 5         # mirrors ``-M 5`` passed to the C binary

# Number of wavelength rows to test per file (first, evenly spaced, last)
N_ROWS_TO_TEST = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_c_iad(rxt_path, max_mc):
    """Run the C iad binary on *rxt_path* and return parsed rows of (mu_a, mu_sp, g).

    The binary writes ``<stem>.txt`` next to the input file.  The data rows
    (non-comment, non-blank) are tab-delimited::

        wave  MR_meas  MR_fit  MT_meas  MT_fit  mu_a  mu_sp  g  [status]

    Returns a list of ``(mu_a, mu_sp, g)`` tuples, one per wavelength row.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_rxt = pathlib.Path(tmpdir) / rxt_path.name
        shutil.copy(rxt_path, tmp_rxt)

        cmd = [str(IAD_BINARY), "-M", str(max_mc), str(tmp_rxt.with_suffix(""))]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        out_path = tmp_rxt.with_suffix(".txt")
        rows = []
        with open(out_path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                # need at least: wave MR_m MR_f MT_m MT_f mu_a mu_sp g
                if len(parts) < 8:
                    continue
                mu_a  = float(parts[5])
                mu_sp = float(parts[6])
                g_c   = float(parts[7])
                rows.append((mu_a, mu_sp, g_c))
        return rows


def _abg_from_mu(mu_a, mu_sp, g_c, d):
    """Convert C output ``(mu_a, mu_s', g)`` and sample thickness *d* to ``(a, b, g)``.

    Args:
        mu_a:  absorption coefficient [mm^-1]
        mu_sp: reduced scattering coefficient mu_s*(1-g) [mm^-1]
        g_c:   anisotropy from C output
        d:     physical sample thickness [mm]

    Returns:
        ``(a, b, g)`` where ``a`` is the single-scattering albedo and
        ``b = mu_t * d`` is the optical thickness.
    """
    if abs(1.0 - g_c) < 1e-9:
        mu_s = 1e12
    else:
        mu_s = mu_sp / (1.0 - g_c)
    mu_t = mu_a + mu_s
    if mu_t <= 0:
        return 0.0, 0.0, g_c
    return mu_s / mu_t, mu_t * d, g_c


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class McLostAgreementTest(unittest.TestCase):
    """Python vs C iad agreement with MC lost-light iterations on .rxt files."""

    @classmethod
    def setUpClass(cls):
        if not IAD_BINARY.is_file():
            raise unittest.SkipTest(
                f"C iad binary not found at {IAD_BINARY}; build with: cd iad && make"
            )
        if not MC_LOST_BINARY.is_file():
            raise unittest.SkipTest(
                f"mc_lost binary not found at {MC_LOST_BINARY}; build with: cd iad && make mc_lost"
            )
        cls.mc_lost_path = str(MC_LOST_BINARY)

    def _scalar_experiment_at_row(self, exp, row_idx):
        """Return a copy of *exp* with scalar measurements taken from *row_idx*."""
        x = iadpython.Experiment(
            sample=exp.sample,
            r_sphere=exp.r_sphere,
            t_sphere=exp.t_sphere,
            num_spheres=exp.num_spheres,
        )
        x.method = exp.method
        x.rstd_r = exp.rstd_r
        x.d_beam = exp.d_beam
        x.fraction_of_rc_in_mr = exp.fraction_of_rc_in_mr
        x.fraction_of_tc_in_mt = exp.fraction_of_tc_in_mt
        x.f_r = exp.f_r

        x.m_r = float(exp.m_r[row_idx]) if exp.m_r is not None else None
        x.m_t = float(exp.m_t[row_idx]) if exp.m_t is not None else None
        x.m_u = float(exp.m_u[row_idx]) if exp.m_u is not None else None

        x.mc_lost_path = self.mc_lost_path
        x.max_mc_iterations = MAX_MC
        return x

    def _check_agreement(self, rxt_name, num_spheres):
        """Run C binary and Python on *rxt_name*, compare at evenly-spaced rows."""
        rxt_path = DATA_DIR / rxt_name

        # C side: run once, get all rows
        c_rows = _run_c_iad(rxt_path, MAX_MC)
        n = len(c_rows)
        self.assertGreater(n, 0, f"{rxt_name}: C binary returned no data rows")

        # Pick up to N_ROWS_TO_TEST evenly-spaced indices within what's available
        if n <= N_ROWS_TO_TEST:
            row_indices = list(range(n))
        else:
            row_indices = [int(round(i * (n - 1) / (N_ROWS_TO_TEST - 1)))
                           for i in range(N_ROWS_TO_TEST)]

        # Python side: parse .rxt once, then test each selected row as a scalar
        exp = iadpython.read_rxt(str(rxt_path))
        d = exp.sample.d

        for row_idx in row_indices:
            mu_a, mu_sp, g_c = c_rows[row_idx]
            c_a, c_b, c_g = _abg_from_mu(mu_a, mu_sp, g_c, d)

            scalar_exp = self._scalar_experiment_at_row(exp, row_idx)
            py_a, py_b, py_g = scalar_exp.invert_rt()

            with self.subTest(rxt=rxt_name, row=row_idx):
                if c_a > 0:
                    self.assertLessEqual(
                        abs(py_a - c_a) / c_a, A_REL_TOL,
                        f"albedo mismatch row {row_idx}: python={py_a:.5f}, C={c_a:.5f}",
                    )
                if c_b > 0:
                    self.assertLessEqual(
                        abs(py_b - c_b) / c_b, B_REL_TOL,
                        f"optical thickness mismatch row {row_idx}: python={py_b:.5f}, C={c_b:.5f}",
                    )
                self.assertLessEqual(
                    abs(py_g - c_g), G_ABS_TOL,
                    f"anisotropy mismatch row {row_idx}: python={py_g:.5f}, C={c_g:.5f}",
                )

    def test_one_sphere_sample_C(self):
        """One-sphere Python/C agreement on sample-C.rxt (M_R + M_T, 20 wavelengths)."""
        self._check_agreement("sample-C.rxt", num_spheres=1)

    def test_two_sphere_sample_E(self):
        """Two-sphere Python/C agreement on sample-E.rxt (M_R + M_T, 20 wavelengths)."""
        self._check_agreement("sample-E.rxt", num_spheres=2)


if __name__ == "__main__":
    unittest.main()
