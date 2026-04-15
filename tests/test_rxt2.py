"""Tests for data2/*.rxt files — second batch of real-data validation.

Covers all files in tests/data2/ that are not already tested in test_rxt.py.
Files already covered by test_rxt.py (basic-A–E, sample-A–G) are skipped.

Run convergence report::

    .venv/bin/python tests/test_rxt2.py
"""

import os
import re
import sys
import unittest
import numpy as np
import iadpython

test_dir = os.path.dirname(os.path.abspath(__file__))
data2_dir = os.path.join(test_dir, "data2")

# Already covered by test_rxt.py — skip to avoid duplication.
_ALREADY_TESTED = frozenset(
    [
        "basic-A.rxt",
        "basic-B.rxt",
        "basic-C.rxt",
        "basic-D.rxt",
        "basic-E.rxt",
        "sample-A.rxt",
        "sample-B.rxt",
        "sample-C.rxt",
        "sample-D.rxt",
        "sample-E.rxt",
        "sample-F.rxt",
        "sample-G.rxt",
    ]
)

# New files that have matching .txt references and are expected to converge.
# x_bad_data is tested separately because some of its rows have invalid inputs.
# terse-A is tested separately because the C .txt has no data rows.
_SMOKE_FILES = [
    "510nm-phantom.rxt",
    "basic-F.rxt",
    "d=746nm_0.90%_2mm.rxt",
    "double.rxt",
    "example2.rxt",
    "fairway-A.rxt",
    "fairway-B.rxt",
    "fairway-C.rxt",
    "fairway-D.rxt",
    "fairway-E.rxt",
    "il-A.rxt",
    "il-B.rxt",
    "il-C.rxt",
    "ink-A.rxt",
    "ink-B.rxt",
    "ink-C.rxt",
    "kenlee-A.rxt",
    "kenlee-B.rxt",
    "kenlee-C.rxt",
    "mayna.rxt",
    "newton.rxt",
    "royston1.rxt",
    "royston2.rxt",
    "royston3-A.rxt",
    "royston3-B.rxt",
    "royston3-C.rxt",
    "royston3-D.rxt",
    "royston3-E.rxt",
    "royston9-A.rxt",
    "royston9-B.rxt",
    "royston9-C.rxt",
    "royston9-D.rxt",
    "terse-A.rxt",
    "terse-B.rxt",
    "tio2_vis.rxt",
    "ville1.rxt",
    "vio-A.rxt",
    "vio-B.rxt",
]

_PHOTON_RE = re.compile(r"Photons used to estimate lost light =\s*(\d+)")


def _reference_txt_path(fname):
    """Return the .txt path for an .rxt fixture in data2."""
    root, _ = os.path.splitext(fname)
    return os.path.join(data2_dir, root + ".txt")


def _reference_photons(fname):
    """Return the MC photon count from the .txt reference, or None if missing."""
    txt = _reference_txt_path(fname)
    if not os.path.exists(txt):
        return None
    with open(txt, encoding="utf-8") as fh:
        for line in fh:
            m = _PHOTON_RE.search(line)
            if m:
                return int(m.group(1))
    return None


def _reference_kind(fname):
    """Return 'no-mc', 'mc-derived', or 'unknown' for a fixture."""
    photons = _reference_photons(fname)
    if photons is None:
        return "unknown"
    return "no-mc" if photons == 0 else "mc-derived"


def _prepare_no_mc(exp):
    """Force an experiment onto the no-MC path."""
    exp.mc_lost_path = None
    exp.max_mc_iterations = 0
    exp.n_photons = 0
    exp.ur1_lost = 0
    exp.ut1_lost = 0
    exp.uru_lost = 0
    exp.utu_lost = 0
    exp.first_pass_abg = None
    return exp


def _scalar_exp_at(exp, i):
    """Return a scalar no-MC experiment for wavelength index `i`."""
    point = exp.point_at(i)
    point.verbosity = 0
    return _prepare_no_mc(point)


def _row_count(exp):
    """Return the number of measurement rows in an experiment."""
    for attr in ("m_r", "m_t", "m_u"):
        val = getattr(exp, attr, None)
        if val is not None:
            return len(np.atleast_1d(val))
    return 0


def _row_indices_for_smoke(exp):
    """Return first and last row indices (all rows if ≤ 3)."""
    n = _row_count(exp)
    if n <= 3:
        return list(range(n))
    return [0, n - 1]


def _parse_txt_coefficients(txt_path):
    """Parse mu_a and mu_s' from a .txt file by reading data rows directly.

    Works even when the header contains non-numeric calibration values
    (e.g. tio2_vis.txt uses '(varies with input row)').
    Returns two numpy arrays: (mua, musp).
    """
    mua, musp = [], []
    with open(txt_path, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) >= 7:
                try:
                    mua.append(float(parts[5]))
                    musp.append(float(parts[6]))
                except ValueError:
                    pass
    return np.array(mua), np.array(musp)


def run_convergence_report():
    """Print a no-MC convergence summary for all smoke files."""
    hdr = f"{'file':30s}  {'rows':5s}  {'ref':10s}  {'dist[0]':10s}  {'dist[-1]':10s}"
    print(hdr)
    print("-" * len(hdr))
    for fname in _SMOKE_FILES:
        fpath = os.path.join(data2_dir, fname)
        try:
            exp = iadpython.read_rxt(fpath)
        except Exception as exc:
            print(f"{fname:30s}  READ ERROR: {exc}")
            continue
        n = _row_count(exp)
        indices = _row_indices_for_smoke(exp)
        dists = []
        for i in indices:
            try:
                pt = _scalar_exp_at(exp, i)
                pt.invert_rt()
                dists.append(f"{pt.final_distance:.2e}")
            except Exception as exc:
                dists.append(f"ERR:{exc}")
        while len(dists) < 2:
            dists.append("—")
        print(f"{fname:30s}  {n:5d}  {_reference_kind(fname):10s}  {dists[0]:10s}  {dists[-1]:10s}")


class TestData2Convergence(unittest.TestCase):
    """No-MC convergence smoke tests for data2 fixtures."""

    def test_no_mc_convergence_smoke(self):
        """All data2 smoke files should converge through the no-MC path."""
        for fname in _SMOKE_FILES:
            fpath = os.path.join(data2_dir, fname)
            exp_base = iadpython.read_rxt(fpath)
            for index in _row_indices_for_smoke(exp_base):
                with self.subTest(file=fname, row=index):
                    point = _scalar_exp_at(exp_base, index)
                    point.invert_rt()
                    self.assertIsNotNone(point.first_pass_abg)
                    self.assertTrue(np.isfinite(point.final_distance))
                    self.assertLess(point.final_distance, 0.1)

    def test_x_bad_data_invalid_rows_raise_on_inversion(self):
        """x_bad_data rows with negative transmittance should raise on inversion."""
        exp_base = iadpython.read_rxt(os.path.join(data2_dir, "x_bad_data.rxt"))
        for index in [0, 1]:
            with self.subTest(row=index):
                point = _scalar_exp_at(exp_base, index)
                with self.assertRaises(Exception):
                    point.invert_rt()

    def test_x_bad_data_valid_row_converges(self):
        """x_bad_data row 2 (valid data) should converge."""
        exp_base = iadpython.read_rxt(os.path.join(data2_dir, "x_bad_data.rxt"))
        point = _scalar_exp_at(exp_base, 2)
        point.invert_rt()
        self.assertLess(point.final_distance, 5e-2)


class TestData2NoMCCoefficients(unittest.TestCase):
    """Coefficient comparison against no-MC C references for data2 files.

    Only tio2_vis.rxt is included here: it was generated with ``iad -M 0``
    and no other special flags, so Python and C should agree closely.

    fairway-A and fairway-E were generated with ``-c 0 -M 0``.  The ``-c 0``
    flag sets ``fraction_of_rc_in_mr = 0``, which is not encoded in the .rxt
    file and is not applied by ``read_rxt``, so coefficient comparison against
    those references is not meaningful.
    """

    def test_tio2_vis_no_mc_coefficients_match_c_reference(self):
        """tio2_vis no-MC Python inversion should match C iad -M 0 reference."""
        txt_path = _reference_txt_path("tio2_vis.rxt")
        mua_ref, musp_ref = _parse_txt_coefficients(txt_path)

        exp_base = iadpython.read_rxt(os.path.join(data2_dir, "tio2_vis.rxt"))
        for index in _row_indices_for_smoke(exp_base):
            with self.subTest(row=index):
                point = _scalar_exp_at(exp_base, index)
                point.invert_rt()
                self.assertAlmostEqual(point.sample.mu_a(), mua_ref[index], delta=0.008)
                self.assertAlmostEqual(point.sample.mu_sp(), musp_ref[index], delta=0.02)


if __name__ == "__main__":
    if "--report" in sys.argv or len(sys.argv) == 1:
        print("No-MC convergence report for data2 fixtures\n")
        run_convergence_report()
    else:
        unittest.main()
