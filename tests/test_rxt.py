"""Tests for RXT file reading and grid comparison on real data.

Run normal tests::

    .venv/bin/pytest tests/test_rxt.py -v

Run the grid comparison benchmark::

    .venv/bin/python tests/test_rxt.py
"""

import os
import re
import sys
import unittest
import numpy as np
import iadpython

# Paths
test_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(test_dir, "data")


# ---------------------------------------------------------------------------
# Original unit tests (unchanged)
# ---------------------------------------------------------------------------

class TestRXT(unittest.TestCase):
    """RXT files data."""

    def test_rxt_01(self):
        """Verify m_r measurements read correctly."""
        filename = os.path.join(data_dir, "basic-A.rxt")
        exp = iadpython.read_rxt(filename)
        print(exp.lambda0)
        self.assertAlmostEqual(exp.m_r[0], 0.299262, delta=1e-5)
        self.assertAlmostEqual(exp.m_r[-1], 0.09662, delta=1e-5)
        self.assertIsNone(exp.m_t)
        self.assertIsNone(exp.m_u)
        self.assertIsNone(exp.lambda0)

    def test_rxt_02(self):
        """Verify m_r and m_t measurements read correctly."""
        filename = os.path.join(data_dir, "basic-B.rxt")
        exp = iadpython.read_rxt(filename)
        self.assertAlmostEqual(exp.m_r[0], 0.51485, delta=1e-5)
        self.assertAlmostEqual(exp.m_t[0], 0.19596, delta=1e-5)
        self.assertAlmostEqual(exp.m_r[-1], 0.44875, delta=1e-5)
        self.assertAlmostEqual(exp.m_t[-1], 0.00019, delta=1e-5)
        self.assertIsNone(exp.m_u)
        self.assertIsNone(exp.lambda0)

    def test_rxt_03(self):
        """Verify m_r, m_t, and m_u measurements read correctly."""
        filename = os.path.join(data_dir, "basic-C.rxt")
        exp = iadpython.read_rxt(filename)
        self.assertAlmostEqual(exp.m_r[0], 0.18744, delta=1e-5)
        self.assertAlmostEqual(exp.m_t[0], 0.57620, delta=1e-5)
        self.assertAlmostEqual(exp.m_u[0], 0.00560, delta=1e-5)
        self.assertAlmostEqual(exp.m_r[-1], 0.16010, delta=1e-5)
        self.assertAlmostEqual(exp.m_t[-1], 0.06857, delta=1e-5)
        self.assertAlmostEqual(exp.m_u[-1], 6.341e-12, delta=1e-14)
        self.assertIsNone(exp.lambda0)

    def test_rxt_04(self):
        """Verify lambda, m_r, m_t read correctly."""
        filename = os.path.join(data_dir, "sample-A.rxt")
        exp = iadpython.read_rxt(filename)
        self.assertAlmostEqual(exp.lambda0[0], 800)
        self.assertAlmostEqual(exp.m_r[0], 0.16830, delta=1e-5)
        self.assertAlmostEqual(exp.m_t[0], 0.24974, delta=1e-5)
        self.assertAlmostEqual(exp.lambda0[-1], 1000)
        self.assertAlmostEqual(exp.m_r[-1], 0.28689, delta=1e-5)
        self.assertAlmostEqual(exp.m_t[-1], 0.42759, delta=1e-5)
        self.assertIsNone(exp.m_u)

    def test_rxt_05(self):
        """Verify lambda, m_r read correctly."""
        filename = os.path.join(data_dir, "sample-B.rxt")
        exp = iadpython.read_rxt(filename)
        self.assertAlmostEqual(exp.lambda0[0], 800)
        self.assertAlmostEqual(exp.m_r[0], 0.16830, delta=1e-5)
        self.assertAlmostEqual(exp.lambda0[-1], 1000)
        self.assertAlmostEqual(exp.m_r[-1], 0.28689, delta=1e-5)
        self.assertIsNone(exp.m_t)
        self.assertIsNone(exp.m_u)


# ---------------------------------------------------------------------------
# Stage A and grid-policy helpers
# ---------------------------------------------------------------------------

_GRID_CONFIGS = [
    ("Grid  N=21", False, {"grid_n": 21}),
#    ("Grid  N=51", False, {"grid_n": 51}),
    ("AGrid coarse", True, {"adaptive_grid_tol": 0.05, "adaptive_grid_max_depth": 4, "adaptive_grid_min_depth": 2}),
#    ("AGrid medium", True, {"adaptive_grid_tol": 0.03, "adaptive_grid_max_depth": 6, "adaptive_grid_min_depth": 2}),
#    ("AGrid fine  ", True, {"adaptive_grid_tol": 0.01, "adaptive_grid_max_depth": 8, "adaptive_grid_min_depth": 3}),
#    ("Auto         ", None, {}),
]

_GRID_POLICY_FILES = [
    "sample-D.rxt",
    "sample-E.rxt",
    "sample-G.rxt",
    "basic-B.rxt",
    "basic-C.rxt",
    "basic-D.rxt",
    "basic-E.rxt",
    "sample-A.rxt",
    "sample-C.rxt",
]

_STAGE_A_SAMPLE_FILES = [
    "sample-A.rxt",
    "sample-B.rxt",
    "sample-C.rxt",
    "sample-D.rxt",
    "sample-E.rxt",
    "sample-F.rxt",
    "sample-G.rxt",
]

# Files that have integrating-sphere configs (num_spheres > 0) and are
# therefore candidates for MC lost-light correction.
_MC_SAMPLE_FILES = [
    "sample-C.rxt",
    "sample-D.rxt",
    "sample-E.rxt",
    "sample-F.rxt",
    "sample-G.rxt",
]

_FIRST_MC_PHOTONS = 100_000

_PHOTON_RE = re.compile(r"Photons used to estimate lost light =\s*(\d+)")
_COARSE_FAIL_DISTANCE = 1e-3
_COARSE_FAIL_RATIO = 100.0


def _reference_txt_path(rxt_name):
    """Return the matching `.txt` reference path for an `.rxt` fixture."""
    root, _ext = os.path.splitext(rxt_name)
    return os.path.join(data_dir, root + ".txt")


def _reference_photons(rxt_name):
    """Return the MC photon count recorded in the reference `.txt`, if present."""
    txt_path = _reference_txt_path(rxt_name)
    if not os.path.exists(txt_path):
        return None

    with open(txt_path, encoding="utf-8") as handle:
        for line in handle:
            match = _PHOTON_RE.search(line)
            if match:
                return int(match.group(1))
    return None


def _reference_kind(rxt_name):
    """Classify whether the existing `.txt` is a no-MC or MC-derived reference."""
    photons = _reference_photons(rxt_name)
    if photons is None:
        return "unknown"
    if photons == 0:
        return "no-mc"
    return "mc-derived"


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


def _run_config(exp, label, use_adaptive, extra):
    """Run no-MC inversion on `exp` with the requested grid configuration."""
    _prepare_no_mc(exp)
    if use_adaptive is not None:
        exp.use_adaptive_grid = use_adaptive
    for key, value in extra.items():
        setattr(exp, key, value)

    exp.grid = None
    exp.sample.rt_evals = 0

    try:
        a, b, g = exp.invert_rt()
    except Exception as exc:  # noqa: BLE001
        return {"label": label, "error": str(exc)}

    fit_r, fit_t = exp.measured_rt()
    grid_pts = len(exp.grid) if isinstance(exp.grid, iadpython.AGrid) else (
        exp.grid.N ** 2 if exp.grid is not None else 0
    )
    grid_evals = getattr(exp, "_grid_evals", 0)
    opt_evals = getattr(exp, "_optimizer_evals", exp.sample.rt_evals - grid_evals)
    first_a, first_b, first_g = exp.first_pass_abg if exp.first_pass_abg is not None else (a, b, g)

    return {
        "label": label,
        "search": exp.search,
        "grid_kind": type(exp.grid).__name__ if exp.grid is not None else "None",
        "grid_pts": grid_pts,
        "grid_evals": grid_evals,
        "opt_evals": opt_evals,
        "iad_iterations": exp.iterations,
        "total": exp.sample.rt_evals,
        "dist": exp.final_distance,
        "first_a": first_a,
        "first_b": first_b,
        "first_g": first_g,
        "a": a,
        "b": b,
        "g": g,
        "fit_r": fit_r,
        "fit_t": fit_t,
        "mu_a": exp.sample.mu_a(),
        "mu_sp": exp.sample.mu_sp(),
    }


def _row_indices_for_smoke(exp):
    """Return a small but representative set of row indices for smoke tests."""
    if exp.m_r is not None:
        n_rows = len(np.atleast_1d(exp.m_r))
    elif exp.m_t is not None:
        n_rows = len(np.atleast_1d(exp.m_t))
    else:
        n_rows = len(np.atleast_1d(exp.m_u))

    if n_rows <= 3:
        return list(range(n_rows))
    return [0, n_rows - 1]


def _axis_tail_values(grid, axis, count=8):
    """Return the largest sampled values for one varying axis in an `AGrid`."""
    values = sorted({getattr(key, axis) for key in grid.cache._data})
    return values[-count:]


def _axis_head_values(grid, axis, count=8):
    """Return the smallest sampled values for one varying axis in an `AGrid`."""
    values = sorted({getattr(key, axis) for key in grid.cache._data})
    return values[:count]


def _finite_median(values):
    """Return the median of finite values, or None if there are none."""
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return None
    return float(np.median(finite))


def _format_or_na(value, fmt):
    """Format a numeric value or return a placeholder for non-finite data."""
    if value is None or not np.isfinite(value):
        return "---"
    return fmt.format(value)


def run_grid_comparison():
    """Print a per-wavelength no-MC grid comparison table for the policy files."""
    hdr = (
        f"{'label':14s}  {'pts':6s}  {'bld':5s}  {'opt':5s}  "
        f"{'tot':5s}  {'dist':8s}  {'first a':7s}  {'first b':7s}  {'first g':7s}"
    )
    sep = "-" * len(hdr)

    for fname in _GRID_POLICY_FILES:
        fpath = os.path.join(data_dir, fname)
        exp_base = iadpython.read_rxt(fpath)

        n_wl = len(np.atleast_1d(exp_base.m_r)) if exp_base.m_r is not None else 0
        print(f"\n{'=' * 78}")
        print(f"  {fname}  ({n_wl} wavelengths, reference={_reference_kind(fname)})")
        print(f"{'=' * 78}")

        totals = {label: {"opt": [], "grid_pts": [], "dist": []} for label, *_ in _GRID_CONFIGS}

        for i in range(n_wl):
            wl = float(np.atleast_1d(exp_base.lambda0)[i]) if exp_base.lambda0 is not None else i
            print(f"\n  wl={wl}")
            print(f"  {hdr}")
            print(f"  {sep}")

            for label, use_adaptive, extra in _GRID_CONFIGS:
                point = _scalar_exp_at(exp_base, i)
                result = _run_config(point, label, use_adaptive, extra)
                if "error" in result:
                    print(f"  {label:14s}  ERROR: {result['error']}")
                    continue

                print(
                    f"  {result['label']:14s}  {result['grid_pts']:6d}  {result['grid_evals']:5d}  "
                    f"{result['opt_evals']:5d}  {result['total']:5d}  {result['dist']:8.2e}  "
                    f"{result['first_a']:.4f}  {result['first_b']:.4f}  {result['first_g']:.4f}"
                )
                totals[label]["opt"].append(result["opt_evals"])
                totals[label]["grid_pts"].append(result["grid_pts"])
                totals[label]["dist"].append(result["dist"])

        print(f"\n  --- Summary for {fname} (median over {n_wl} wavelengths) ---")
        print(f"  {'label':14s}  {'pts_med':8s}  {'opt_med':8s}  {'dist_med':10s}")
        for label, *_ in _GRID_CONFIGS:
            data = totals[label]
            if data["opt"]:
                print(
                    f"  {label:14s}  {np.median(data['grid_pts']):8.0f}  "
                    f"{np.median(data['opt']):8.0f}  {np.median(data['dist']):10.2e}"
                )


def run_stage_a_report():
    """Print a no-MC Stage A report for `sample-A` through `sample-G`."""
    hdr = (
        f"{'row':>4s}  {'search':8s}  {'ref':10s}  {'grid':8s}  {'dist':8s}  "
        f"{'first a':7s}  {'first b':7s}  {'first g':7s}"
    )
    sep = "-" * len(hdr)

    for fname in _STAGE_A_SAMPLE_FILES:
        exp_base = iadpython.read_rxt(os.path.join(data_dir, fname))
        n_wl = len(np.atleast_1d(exp_base.m_r)) if exp_base.m_r is not None else 0
        print(f"\n{'=' * 78}")
        print(f"  {fname}  ({n_wl} wavelengths, reference={_reference_kind(fname)})")
        print(f"{'=' * 78}")
        print(f"  {hdr}")
        print(f"  {sep}")

        for i in range(n_wl):
            point = _scalar_exp_at(exp_base, i)
            result = _run_config(point, "Auto", None, {})
            if "error" in result:
                print(f"  {i:4d}  ERROR: {result['error']}")
                continue

            print(
                f"  {i:4d}  {result['search']:8s}  {_reference_kind(fname):10s}  "
                f"{result['grid_kind']:8s}  {result['dist']:8.2e}  "
                f"{result['first_a']:.4f}  {result['first_b']:.4f}  {result['first_g']:.4f}"
            )


def run_agrid_coarse_failure_report():
    """Print rows where coarse AGrid misses the no-MC solution basin badly."""
    print("AGrid coarse failure report\n")

    for fname in _GRID_POLICY_FILES:
        exp_base = iadpython.read_rxt(os.path.join(data_dir, fname))
        n_wl = len(np.atleast_1d(exp_base.m_r)) if exp_base.m_r is not None else 0
        any_failures = False

        for i in range(n_wl):
            grid_result = _run_config(_scalar_exp_at(exp_base, i), "Grid  N=21", False, {"grid_n": 21})
            coarse_exp = _scalar_exp_at(exp_base, i)
            coarse_result = _run_config(
                coarse_exp,
                "AGrid coarse",
                True,
                {
                    "adaptive_grid_tol": 0.05,
                    "adaptive_grid_max_depth": 4,
                    "adaptive_grid_min_depth": 2,
                },
            )

            if "error" in grid_result or "error" in coarse_result:
                continue

            ratio = coarse_result["dist"] / max(grid_result["dist"], 1e-15)
            if coarse_result["dist"] <= _COARSE_FAIL_DISTANCE and ratio <= _COARSE_FAIL_RATIO:
                continue

            if not any_failures:
                print(f"{fname}")
                any_failures = True

            print(
                f"  row {i:2d} search={coarse_result['search']} "
                f"grid_dist={grid_result['dist']:.3e} coarse_dist={coarse_result['dist']:.3e} "
                f"ratio={ratio:.1f}"
            )
            print(
                f"    grid first   = ({grid_result['first_a']:.6f}, {grid_result['first_b']:.6f}, {grid_result['first_g']:.6f})"
            )
            print(
                f"    coarse first = ({coarse_result['first_a']:.6f}, {coarse_result['first_b']:.6f}, {coarse_result['first_g']:.6f})"
            )

            if isinstance(coarse_exp.grid, iadpython.AGrid):
                if coarse_result["search"] == "find_ab":
                    print(f"    coarse a tail = {_axis_tail_values(coarse_exp.grid, 'a')}")
                elif coarse_result["search"] == "find_ag":
                    print(f"    coarse a tail = {_axis_tail_values(coarse_exp.grid, 'a')}")
                    print(f"    coarse g head = {_axis_head_values(coarse_exp.grid, 'g')}")
                    print(f"    coarse g tail = {_axis_tail_values(coarse_exp.grid, 'g')}")
                elif coarse_result["search"] == "find_bg":
                    print(f"    coarse g head = {_axis_head_values(coarse_exp.grid, 'g')}")
                    print(f"    coarse g tail = {_axis_tail_values(coarse_exp.grid, 'g')}")

        if any_failures:
            print()


class TestRXTStageA(unittest.TestCase):
    """Fixture-driven no-MC smoke tests for Stage A."""

    def test_stage_a_reference_classification(self):
        """All sample references with baselines should now be no-MC references."""
        self.assertEqual(_reference_kind("sample-C.rxt"), "no-mc")
        self.assertEqual(_reference_kind("sample-D.rxt"), "no-mc")
        self.assertEqual(_reference_kind("sample-E.rxt"), "no-mc")
        self.assertEqual(_reference_kind("sample-G.rxt"), "no-mc")

    def test_stage_a_sample_sweep_smoke_without_mc(self):
        """Sample-A through sample-G should converge through the no-MC path."""
        for fname in _STAGE_A_SAMPLE_FILES:
            exp_base = iadpython.read_rxt(os.path.join(data_dir, fname))
            for index in _row_indices_for_smoke(exp_base):
                with self.subTest(file=fname, row=index):
                    point = _scalar_exp_at(exp_base, index)
                    result = _run_config(point, "Auto", None, {})
                    self.assertNotIn("error", result, msg=result.get("error"))
                    self.assertIsNotNone(point.first_pass_abg)
                    self.assertTrue(np.isfinite(result["dist"]))
                    self.assertLess(result["dist"], 5e-2)

    def _run_no_mc_coeff_comparison(self, fname, delta_mu_a, delta_mu_sp):
        """Helper: compare Python no-MC mu_a/mu_s' against the C no-MC reference."""
        _, ref = iadpython.read_txt(_reference_txt_path(fname))
        exp_base = iadpython.read_rxt(os.path.join(data_dir, fname))
        n_rows = len(ref.mua)
        for i in range(n_rows):
            with self.subTest(file=fname, row=i):
                point = _scalar_exp_at(exp_base, i)
                point.invert_rt()
                self.assertAlmostEqual(point.sample.mu_a(), ref.mua[i], delta=delta_mu_a)
                self.assertAlmostEqual(point.sample.mu_sp(), ref.musp[i], delta=delta_mu_sp)

    def test_sample_d_no_mc_coefficients_match_c_reference(self):
        """Python no-MC inversion for sample-D should match C iad -M 0 within tolerance."""
        self._run_no_mc_coeff_comparison("sample-D.rxt", delta_mu_a=0.008, delta_mu_sp=0.02)

    def test_sample_e_no_mc_coefficients_match_c_reference(self):
        """Python no-MC inversion for sample-E should match C iad -M 0 within tolerance."""
        self._run_no_mc_coeff_comparison("sample-E.rxt", delta_mu_a=0.008, delta_mu_sp=0.02)

    def test_sample_g_no_mc_coefficients_match_c_reference(self):
        """Python no-MC inversion for sample-G should match C iad -M 0 within tolerance."""
        self._run_no_mc_coeff_comparison("sample-G.rxt", delta_mu_a=0.008, delta_mu_sp=0.02)

    def test_basic_c_row_3_coarse_agrid_closure_finds_the_interior_basin(self):
        """Coarse AGrid should fill the missing local cross-combinations for row 3."""
        exp_base = iadpython.read_rxt(os.path.join(data_dir, "basic-C.rxt"))
        point = _scalar_exp_at(exp_base, 3)
        result = _run_config(
            point,
            "AGrid coarse",
            True,
            {
                "adaptive_grid_tol": 0.05,
                "adaptive_grid_max_depth": 4,
                "adaptive_grid_min_depth": 2,
            },
        )

        self.assertNotIn("error", result, msg=result.get("error"))
        self.assertLess(result["dist"], 1e-3)
        self.assertLess(result["first_g"], 0.98)
        self.assertAlmostEqual(result["first_a"], 0.88, delta=0.03)


def _find_mc_lost_binary():
    """Return path to mc_lost binary, or None if not found."""
    import shutil
    repo_binary = os.path.join(test_dir, "..", "iad", "mc_lost")
    if os.path.isfile(repo_binary) and os.access(repo_binary, os.X_OK):
        return os.path.abspath(repo_binary)
    return shutil.which("mc_lost")


def _run_mc_step(exp, a0, b0, g0, mc_lost_path, n_photons):
    """Run one MC lost-light update then re-invert with hot_start."""
    exp.mc_lost_path = mc_lost_path
    exp.n_photons = n_photons

    try:
        exp._update_lost_light(a0, b0, g0)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}

    ur1_lost = exp.ur1_lost
    ut1_lost = exp.ut1_lost
    uru_lost = exp.uru_lost
    utu_lost = exp.utu_lost

    try:
        a1, b1, g1 = exp.invert_scalar_rt(hot_start=(a0, b0, g0))
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}

    return {
        "a": a1,
        "b": b1,
        "g": g1,
        "dist": exp.final_distance,
        "iad_iterations": exp.iterations,
        "mu_a": exp.sample.mu_a(),
        "mu_sp": exp.sample.mu_sp(),
        "ur1_lost": ur1_lost,
        "ut1_lost": ut1_lost,
        "uru_lost": uru_lost,
        "utu_lost": utu_lost,
    }


def run_mc_report(n_mc_iterations, n_photons=_FIRST_MC_PHOTONS, filenames=None):
    """Print a per-wavelength table comparing no-MC and manual MC updates.

    For each sample file with integrating spheres the table shows the initial
    no-MC solution and then `n_mc_iterations` explicit lost-light updates,
    using hot_start between iterations so the grid is not rebuilt.
    """
    mc_lost_path = _find_mc_lost_binary()
    if mc_lost_path is None:
        print("mc_lost binary not found — build with: cd iad && make mc_lost")
        return

    hdr = (
        f"{'stage':8s}  {'grid':8s}  {'iad it':6s}  {'dist':8s}  {'a':7s}  {'b':7s}  {'g':7s}  "
        f"{'mu_a':8s}  {'mu_sp':8s}  {'ur1_L':7s}  {'ut1_L':7s}"
    )
    sep = "-" * len(hdr)
    if filenames is None:
        filenames = _MC_SAMPLE_FILES

    for fname in filenames:
        fpath = os.path.join(data_dir, fname)
        exp_base = iadpython.read_rxt(fpath)
        n_wl = len(np.atleast_1d(exp_base.m_r)) if exp_base.m_r is not None else 0
        file_summary = {
            "initial_dist": [],
            "first_mc_dist": [],
            "delta_a": [],
            "delta_b": [],
            "delta_g": [],
            "ur1_lost": [],
            "ut1_lost": [],
            "uru_lost": [],
            "utu_lost": [],
        }

        print(f"\n{'=' * 78}")
        print(f"  {fname}  ({n_wl} wavelengths, reference={_reference_kind(fname)}, "
              f"num_spheres={int(exp_base.num_spheres)}, mc_iters={n_mc_iterations}, photons={n_photons:,})")
        print(f"{'=' * 78}")

        for i in range(n_wl):
            wl_arr = np.atleast_1d(exp_base.lambda0) if exp_base.lambda0 is not None else None
            wl = float(wl_arr[i]) if wl_arr is not None else i
            print(f"\n  wl={wl}")
            print(f"  {hdr}")
            print(f"  {sep}")

            # --- first pass: no MC ---
            point = _scalar_exp_at(exp_base, i)
            no_mc = _run_config(point, "no-MC", None, {})

            if "error" in no_mc:
                print(f"  {'no-MC':8s}  ERROR: {no_mc['error']}")
                continue

            a0, b0, g0 = no_mc["a"], no_mc["b"], no_mc["g"]
            file_summary["initial_dist"].append(no_mc["dist"])
            print(
                f"  {'initial':8s}  {no_mc['grid_kind']:8s}  {no_mc['iad_iterations']:6d}  {no_mc['dist']:8.2e}  "
                f"{a0:7.4f}  {b0:7.4f}  {g0:7.4f}  "
                f"{no_mc['mu_a'] or 0:8.4f}  {no_mc['mu_sp'] or 0:8.4f}  "
                f"{'---':>7s}  {'---':>7s}"
            )

            # --- explicit MC updates ---
            a_prev, b_prev, g_prev = a0, b0, g0
            for mc_index in range(1, n_mc_iterations + 1):
                mc_result = _run_mc_step(point, a_prev, b_prev, g_prev, mc_lost_path, n_photons)

                if "error" in mc_result:
                    print(f"  {f'MC {mc_index}':8s}  ERROR: {mc_result['error']}")
                    break

                delta_a = mc_result["a"] - a_prev
                delta_b = mc_result["b"] - b_prev
                delta_g = mc_result["g"] - g_prev
                if mc_index == 1:
                    file_summary["first_mc_dist"].append(mc_result["dist"])
                if np.isfinite(delta_a):
                    file_summary["delta_a"].append(abs(delta_a))
                if np.isfinite(delta_b):
                    file_summary["delta_b"].append(abs(delta_b))
                if np.isfinite(delta_g):
                    file_summary["delta_g"].append(abs(delta_g))
                for key in ("ur1_lost", "ut1_lost", "uru_lost", "utu_lost"):
                    file_summary[key].append(mc_result[key])
                print(
                    f"  {f'MC {mc_index}':8s}  {'hot':8s}  {mc_result['iad_iterations']:6d}  {mc_result['dist']:8.2e}  "
                    f"{mc_result['a']:7.4f}  {mc_result['b']:7.4f}  "
                    f"{mc_result['g']:7.4f}  {mc_result['mu_a'] or 0:8.4f}  "
                    f"{mc_result['mu_sp'] or 0:8.4f}  "
                    f"{mc_result['ur1_lost']:7.5f}  {mc_result['ut1_lost']:7.5f}"
                )
                print(
                    f"  {'Δ':8s}  {'':8s}  {'':6s}  {'':8s}  "
                    f"{_format_or_na(delta_a, '{:+7.4f}'):>7s}  "
                    f"{_format_or_na(delta_b, '{:+7.4f}'):>7s}  "
                    f"{_format_or_na(delta_g, '{:+7.4f}'):>7s}  "
                    f"{'':8s}  {'':8s}  "
                    f"{mc_result['uru_lost']:7.5f}  {mc_result['utu_lost']:7.5f}"
                )
                a_prev, b_prev, g_prev = mc_result["a"], mc_result["b"], mc_result["g"]

        if file_summary["initial_dist"]:
            delta_a_med = _finite_median(file_summary["delta_a"])
            delta_b_med = _finite_median(file_summary["delta_b"])
            delta_g_med = _finite_median(file_summary["delta_g"])
            print(f"\n  --- Summary for {fname} (median over {len(file_summary['initial_dist'])} wavelengths) ---")
            print(
                f"  initial dist={np.median(file_summary['initial_dist']):.2e}  "
                f"1st MC dist={np.median(file_summary['first_mc_dist']):.2e}  "
                f"|Δa|={_format_or_na(delta_a_med, '{:.4f}')}  "
                f"|Δb|={_format_or_na(delta_b_med, '{:.4f}')}  "
                f"|Δg|={_format_or_na(delta_g_med, '{:.4f}')}"
            )
            print(
                f"  lost medians: ur1={np.median(file_summary['ur1_lost']):.5f}  "
                f"ut1={np.median(file_summary['ut1_lost']):.5f}  "
                f"uru={np.median(file_summary['uru_lost']):.5f}  "
                f"utu={np.median(file_summary['utu_lost']):.5f}"
            )


def _cli_photons(default):
    """Return the requested photon count from the ad-hoc CLI args."""
    photons = default
    for arg in sys.argv:
        if arg.startswith("--photons="):
            photons = int(arg.split("=", 1)[1])
    return photons


def _cli_mc_iterations():
    """Return requested MC iteration count from `--mc N` or `--mc=N`."""
    for index, arg in enumerate(sys.argv):
        if arg == "--mc":
            if index + 1 >= len(sys.argv):
                raise SystemExit("expected an integer after --mc")
            return int(sys.argv[index + 1])
        if arg.startswith("--mc="):
            return int(arg.split("=", 1)[1])
    return None


def _cli_mc_filters():
    """Return positional file filters for the ad-hoc `--mc` report."""
    filters = []
    skip_next = False
    for index, arg in enumerate(sys.argv[1:], start=1):
        if skip_next:
            skip_next = False
            continue
        if arg == "--mc":
            skip_next = True
            continue
        if arg.startswith("--mc="):
            continue
        if arg.startswith("--photons="):
            continue
        if arg.startswith("-"):
            continue
        filters.append(arg)
    return filters


def _match_mc_files(filters):
    """Return the MC fixture files selected by CLI filters."""
    if not filters:
        return list(_MC_SAMPLE_FILES)

    selected = []
    unmatched = []
    for raw_filter in filters:
        token = os.path.basename(raw_filter).lower()
        if token.endswith(".rxt"):
            stem = token[:-4]
        else:
            stem = token

        matches = [
            fname for fname in _MC_SAMPLE_FILES
            if fname.lower() == token or fname.lower() == f"{stem}.rxt"
        ]
        if not matches:
            unmatched.append(raw_filter)
            continue
        for fname in matches:
            if fname not in selected:
                selected.append(fname)

    if unmatched:
        allowed = ", ".join(_MC_SAMPLE_FILES)
        missing = ", ".join(unmatched)
        raise SystemExit(f"unknown --mc file filter(s): {missing}. Allowed files: {allowed}")

    return selected


if __name__ == "__main__":
    if "--compare" in sys.argv:
        print("Grid vs AGrid comparison on no-MC policy fixtures\n")
        run_grid_comparison()
    elif _cli_mc_iterations() is not None:
        mc_iterations = _cli_mc_iterations()
        photons = _cli_photons(_FIRST_MC_PHOTONS)
        filenames = _match_mc_files(_cli_mc_filters())
        print(f"MC correction report (iterations={mc_iterations}, photons={photons:,})\n")
        run_mc_report(mc_iterations, n_photons=photons, filenames=filenames)
    elif "--stage-a" in sys.argv or len(sys.argv) == 1:
        print("Stage A no-MC sample sweep\n")
        run_stage_a_report()
    else:
        unittest.main()
