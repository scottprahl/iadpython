"""Benchmark whether custom Nelder-Mead simplices help MC re-inversions.

This script compares three ways of solving the *same* MC-corrected re-inversion
problem after a lost-light update:

1. the current production path: ``invert_scalar_rt(hot_start=...)``
2. the same path, but with a small fixed ``initial_simplex`` injected into
   SciPy's Nelder-Mead call
3. the same path, but with an adaptive simplex that shrinks using the previous
   MC stage's parameter change

The production code is not changed here.  This is just a measurement tool to
tell us whether the added simplex complexity actually buys us fewer optimizer
iterations and evaluations than the simpler fixed-small-simplex approach.

Example::

    uv run python tests/bench_mc_simplex.py --mc 2 sample-E.rxt --photons=2000
"""

from __future__ import annotations

import argparse
import copy
import os
import pathlib
from collections import defaultdict
from unittest.mock import patch

import numpy as np

import iadpython


TEST_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = TEST_DIR
MC_SAMPLE_FILES = [
    "data/sample-C.rxt",
    "data/sample-D.rxt",
    "data/sample-E.rxt",
    "data/sample-F.rxt",
    "data/sample-G.rxt",
    "data2/510nm-phantom.rxt",
    "data2/basic-F.rxt",
    "data2/d=746nm_0.90%_2mm.rxt",
    "data2/double.rxt",
    "data2/example2.rxt",
    "data2/fairway-A.rxt",
    "data2/fairway-B.rxt",
    "data2/fairway-C.rxt",
    "data2/fairway-D.rxt",
    "data2/fairway-E.rxt",
    "data2/il-A.rxt",
    "data2/il-B.rxt",
    "data2/il-C.rxt",
    "data2/ink-A.rxt",
    "data2/ink-B.rxt",
    "data2/ink-C.rxt",
    "data2/kenlee-A.rxt",
    "data2/kenlee-B.rxt",
    "data2/kenlee-C.rxt",
    "data2/mayna.rxt",
    "data2/newton.rxt",
    "data2/royston1.rxt",
    "data2/royston2.rxt",
    "data2/royston3-A.rxt",
    "data2/royston3-B.rxt",
    "data2/royston3-C.rxt",
    "data2/royston3-D.rxt",
    "data2/royston3-E.rxt",
    "data2/royston9-A.rxt",
    "data2/royston9-B.rxt",
    "data2/royston9-C.rxt",
    "data2/royston9-D.rxt",
    "data2/terse-A.rxt",
    "data2/terse-B.rxt",
    "data2/tio2_vis.rxt",
    "data2/ville1.rxt",
    "data2/vio-A.rxt",
    "data2/vio-B.rxt",
]
G_BOUND_EPS = iadpython.iad.G_BOUND_EPS


def _find_mc_lost_binary():
    """Return path to mc_lost binary, or None if not found."""
    import shutil

    repo_binary = TEST_DIR.parent / "iad" / "mc_lost"
    if repo_binary.is_file() and os.access(repo_binary, os.X_OK):
        return str(repo_binary.resolve())
    return shutil.which("mc_lost")


def _prepare_no_mc(exp):
    """Force an experiment onto the deterministic no-MC path."""
    exp.mc_lost_path = None
    exp.max_mc_iterations = 0
    exp.n_photons = 0
    exp.ur1_lost = 0
    exp.ut1_lost = 0
    exp.uru_lost = 0
    exp.utu_lost = 0
    exp.first_pass_abg = None
    exp.verbosity = 0
    return exp


def _scalar_exp_at(exp, index):
    """Return a scalar experiment for wavelength index `index`."""
    return _prepare_no_mc(exp.point_at(index))


def _num_rows(exp):
    """Return the number of scalar rows in an experiment."""
    if exp.m_r is not None:
        return len(np.atleast_1d(exp.m_r))
    if exp.m_t is not None:
        return len(np.atleast_1d(exp.m_t))
    if exp.m_u is not None:
        return len(np.atleast_1d(exp.m_u))
    return 0


def _match_files(filters):
    """Match optional CLI filters against the MC fixture list."""
    if not filters:
        return list(MC_SAMPLE_FILES)

    selected = []
    unmatched = []
    for raw_filter in filters:
        token = os.path.basename(raw_filter).lower()
        stem = token[:-4] if token.endswith(".rxt") else token
        matches = [
            fname
            for fname in MC_SAMPLE_FILES
            if fname.lower() == token or fname.lower() == f"{stem}.rxt"
        ]
        if not matches:
            unmatched.append(raw_filter)
            continue
        for fname in matches:
            if fname not in selected:
                selected.append(fname)

    if unmatched:
        allowed = ", ".join(MC_SAMPLE_FILES)
        missing = ", ".join(unmatched)
        raise SystemExit(f"unknown file filter(s): {missing}. Allowed files: {allowed}")
    return selected


def _finite_median(values):
    """Return the median of finite values, or None if there are none."""
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return None
    return float(np.median(finite))


def _format_or_na(value, fmt):
    """Format a finite value or return a placeholder."""
    if value is None or not np.isfinite(value):
        return "---"
    return fmt.format(value)


def _bounded_vertex(center, step, lower, upper):
    """Move one simplex vertex away from `center` while respecting bounds."""
    pos_room = np.inf if not np.isfinite(upper) else upper - center
    neg_room = np.inf if not np.isfinite(lower) else center - lower

    if pos_room >= step:
        return center + step
    if neg_room >= step:
        return center - step

    if pos_room > neg_room and pos_room > 0:
        return center + 0.5 * pos_room
    if neg_room > 0:
        return center - 0.5 * neg_room

    if np.isfinite(upper) and upper > center:
        return center + min(step, 0.5 * (upper - center))
    if np.isfinite(lower) and center > lower:
        return center - min(step, 0.5 * (center - lower))

    return center


def _simplex_geometry(exp, hot_start, a_step=1e-3, b_rel_step=0.01, g_step=1e-3):
    """Return search-specific simplex geometry around the current hot start."""
    if exp.search == "find_ab":
        x0 = np.array([hot_start[0], hot_start[1]], dtype=float)
        steps = np.array([a_step, max(1e-3, b_rel_step * max(abs(x0[1]), 1.0))], dtype=float)
        lower = np.array([0.0, 0.0], dtype=float)
        upper = np.array([1.0, np.inf], dtype=float)
    elif exp.search == "find_ag":
        x0 = np.array([hot_start[0], hot_start[2]], dtype=float)
        steps = np.array([a_step, g_step], dtype=float)
        lower = np.array([0.0, -1.0 + G_BOUND_EPS], dtype=float)
        upper = np.array([1.0, 1.0 - G_BOUND_EPS], dtype=float)
    elif exp.search == "find_bg":
        x0 = np.array([hot_start[1], hot_start[2]], dtype=float)
        steps = np.array([max(1e-3, b_rel_step * max(abs(x0[0]), 1.0)), g_step], dtype=float)
        lower = np.array([0.0, -1.0 + G_BOUND_EPS], dtype=float)
        upper = np.array([np.inf, 1.0 - G_BOUND_EPS], dtype=float)
    else:
        raise ValueError(f"custom simplex is only relevant for 2D Nelder-Mead searches, got {exp.search}")

    return x0, steps, lower, upper


def _build_initial_simplex(exp, hot_start, steps):
    """Build a search-specific simplex around the current hot start."""
    x0, _default_steps, lower, upper = _simplex_geometry(exp, hot_start)
    simplex = np.tile(x0, (len(x0) + 1, 1))
    for axis in range(len(x0)):
        simplex[axis + 1, axis] = _bounded_vertex(x0[axis], steps[axis], lower[axis], upper[axis])
    return simplex


def _fixed_simplex_steps(exp, hot_start, a_step=1e-3, b_rel_step=0.01, g_step=1e-3, **_ignored):
    """Return the fixed warm-start simplex steps for one search."""
    _x0, steps, _lower, _upper = _simplex_geometry(
        exp,
        hot_start,
        a_step=a_step,
        b_rel_step=b_rel_step,
        g_step=g_step,
    )
    return steps


def _adaptive_simplex_steps(
    exp,
    hot_start,
    previous_delta,
    a_step=1e-3,
    b_rel_step=0.01,
    g_step=1e-3,
    adaptive_scale=1.0,
    a_min_step=1e-5,
    b_min_step=1e-4,
    g_min_step=1e-5,
):
    """Return a simplex that shrinks with the previous MC stage's movement."""
    fixed_steps = _fixed_simplex_steps(
        exp,
        hot_start,
        a_step=a_step,
        b_rel_step=b_rel_step,
        g_step=g_step,
    )
    if previous_delta is None:
        return fixed_steps

    if exp.search == "find_ab":
        observed = np.array([previous_delta["a"], previous_delta["b"]], dtype=float)
        minimum = np.array([a_min_step, b_min_step], dtype=float)
    elif exp.search == "find_ag":
        observed = np.array([previous_delta["a"], previous_delta["g"]], dtype=float)
        minimum = np.array([a_min_step, g_min_step], dtype=float)
    elif exp.search == "find_bg":
        observed = np.array([previous_delta["b"], previous_delta["g"]], dtype=float)
        minimum = np.array([b_min_step, g_min_step], dtype=float)
    else:
        raise ValueError(f"adaptive simplex is only relevant for 2D Nelder-Mead searches, got {exp.search}")

    if not np.all(np.isfinite(observed)):
        return fixed_steps

    adaptive = adaptive_scale * np.abs(observed)
    return np.clip(adaptive, minimum, fixed_steps)


def _run_hot_start(exp, hot_start, simplex=None):
    """Run one hot-start re-inversion, optionally with a custom simplex."""
    if simplex is None:
        a, b, g = exp.invert_scalar_rt(hot_start=hot_start)
    else:
        original_minimize = iadpython.iad.scipy.optimize.minimize

        def _patched_minimize(fun, x0, *args, **kwargs):
            method = kwargs.get("method")
            if method == "Nelder-Mead":
                options = dict(kwargs.get("options") or {})
                options["initial_simplex"] = simplex
                kwargs["options"] = options
            return original_minimize(fun, x0, *args, **kwargs)

        with patch.object(iadpython.iad.scipy.optimize, "minimize", side_effect=_patched_minimize):
            a, b, g = exp.invert_scalar_rt(hot_start=hot_start)

    return {
        "a": a,
        "b": b,
        "g": g,
        "nit": exp.iterations,
        "nfev": getattr(exp, "_optimizer_evals", exp.sample.rt_evals),
        "dist": exp.final_distance,
        "mu_a": exp.sample.mu_a(),
        "mu_sp": exp.sample.mu_sp(),
    }


def _solution_delta(base, other):
    """Return a small scalar comparing two converged solutions."""
    return (
        abs(base["a"] - other["a"])
        + abs(np.log1p(base["b"]) - np.log1p(other["b"]))
        + abs(base["g"] - other["g"])
    )


def _compare_mc_stage(stage_exp, hot_start, previous_delta, simplex_kwargs):
    """Compare baseline, fixed simplex, and adaptive simplex on one MC stage."""
    baseline_exp = copy.deepcopy(stage_exp)
    fixed_exp = copy.deepcopy(stage_exp)
    adaptive_exp = copy.deepcopy(stage_exp)

    baseline = _run_hot_start(baseline_exp, hot_start, simplex=None)
    fixed_steps = _fixed_simplex_steps(fixed_exp, hot_start, **simplex_kwargs)
    fixed_simplex = _build_initial_simplex(fixed_exp, hot_start, fixed_steps)
    fixed = _run_hot_start(fixed_exp, hot_start, simplex=fixed_simplex)
    adaptive_steps = _adaptive_simplex_steps(adaptive_exp, hot_start, previous_delta, **simplex_kwargs)
    adaptive_simplex = _build_initial_simplex(adaptive_exp, hot_start, adaptive_steps)
    adaptive = _run_hot_start(adaptive_exp, hot_start, simplex=adaptive_simplex)

    return {
        "baseline": baseline,
        "fixed": fixed,
        "adaptive": adaptive,
        "fixed_steps": fixed_steps,
        "adaptive_steps": adaptive_steps,
        "fixed_solution_delta": _solution_delta(baseline, fixed),
        "adaptive_solution_delta": _solution_delta(baseline, adaptive),
        "adaptive_vs_fixed_solution_delta": _solution_delta(fixed, adaptive),
        "fixed_nit_delta": fixed["nit"] - baseline["nit"],
        "adaptive_nit_delta": adaptive["nit"] - baseline["nit"],
        "adaptive_minus_fixed_nit": adaptive["nit"] - fixed["nit"],
        "fixed_nfev_delta": fixed["nfev"] - baseline["nfev"],
        "adaptive_nfev_delta": adaptive["nfev"] - baseline["nfev"],
        "adaptive_minus_fixed_nfev": adaptive["nfev"] - fixed["nfev"],
    }


def run_benchmark(mc_iterations, photons, filenames, simplex_kwargs):
    """Run the benchmark and print per-stage tables plus summaries."""
    mc_lost_path = _find_mc_lost_binary()
    if mc_lost_path is None:
        print("mc_lost binary not found — build with: cd iad && make mc_lost")
        return

    print(
        "MC re-inversion simplex benchmark "
        f"(iterations={mc_iterations}, photons={photons:,}, "
        f"a_step={simplex_kwargs['a_step']:.4g}, "
        f"b_rel={simplex_kwargs['b_rel_step']:.4g}, "
        f"g_step={simplex_kwargs['g_step']:.4g}, "
        f"adaptive_scale={simplex_kwargs['adaptive_scale']:.4g})\n"
    )

    overall = defaultdict(list)
    hdr = (
        f"{'stage':8s}  {'base it':7s}  {'fixed':7s}  {'adapt':7s}  "
        f"{'base ev':7s}  {'fixed':7s}  {'adapt':7s}  "
        f"{'base dist':10s}  {'fixed':10s}  {'adapt':10s}"
    )
    sep = "-" * len(hdr)

    for fname in filenames:
        exp_base = iadpython.read_rxt(str(DATA_DIR / fname))
        n_rows = _num_rows(exp_base)
        file_stats = defaultdict(list)

        print(f"\n{'=' * 88}")
        print(f"  {fname}  ({n_rows} rows)")
        print(f"{'=' * 88}")

        for row in range(n_rows):
            driver = _scalar_exp_at(exp_base, row)
            previous_delta = None
            wl_arr = np.atleast_1d(exp_base.lambda0) if exp_base.lambda0 is not None else None
            wl = float(wl_arr[row]) if wl_arr is not None else row

            try:
                a_prev, b_prev, g_prev = driver.invert_scalar_rt()
            except Exception as exc:  # noqa: BLE001
                print(f"\n  wl={wl}  ERROR during initial inversion: {exc}")
                continue

            if driver.search not in {"find_ab", "find_ag", "find_bg"}:
                print(
                    f"\n  wl={wl}  search={driver.search}  "
                    "skipping: custom simplex only applies to 2D Nelder-Mead searches"
                )
                continue

            print(
                f"\n  wl={wl}  search={driver.search}  "
                f"initial it={driver.iterations}  dist={driver.final_distance:.2e}  "
                f"a={a_prev:.4f}  b={b_prev:.4f}  g={g_prev:.4f}"
            )
            print(f"  {hdr}")
            print(f"  {sep}")

            for stage in range(1, mc_iterations + 1):
                driver.mc_lost_path = mc_lost_path
                driver.n_photons = photons

                try:
                    driver._update_lost_light(a_prev, b_prev, g_prev)
                except Exception as exc:  # noqa: BLE001
                    print(f"  {f'MC {stage}':8s}  ERROR during mc_lost update: {exc}")
                    break

                comparison = _compare_mc_stage(
                    driver,
                    (a_prev, b_prev, g_prev),
                    previous_delta,
                    simplex_kwargs,
                )
                baseline = comparison["baseline"]
                fixed = comparison["fixed"]
                adaptive = comparison["adaptive"]

                print(
                    f"  {f'MC {stage}':8s}  {baseline['nit']:7d}  {fixed['nit']:7d}  {adaptive['nit']:7d}  "
                    f"{baseline['nfev']:7d}  {fixed['nfev']:7d}  {adaptive['nfev']:7d}  "
                    f"{baseline['dist']:10.2e}  {fixed['dist']:10.2e}  {adaptive['dist']:10.2e}"
                )
                print(
                    f"  {'Δit':8s}  {'':7s}  {comparison['fixed_nit_delta']:7d}  {comparison['adaptive_nit_delta']:7d}  "
                    f"{'Δev':7s}  {comparison['fixed_nfev_delta']:7d}  {comparison['adaptive_nfev_delta']:7d}  "
                    f"{'Δsol':10s}  {_format_or_na(comparison['fixed_solution_delta'], '{:.2e}'):>10s}  "
                    f"{_format_or_na(comparison['adaptive_solution_delta'], '{:.2e}'):>10s}"
                )
                print(
                    f"  {'steps':8s}  {'fixed':7s}  "
                    f"{comparison['fixed_steps'][0]:7.3g}  {comparison['fixed_steps'][1]:7.3g}  "
                    f"{'adapt':7s}  {comparison['adaptive_steps'][0]:7.3g}  {comparison['adaptive_steps'][1]:7.3g}"
                )

                file_stats["base_nit"].append(baseline["nit"])
                file_stats["fixed_nit"].append(fixed["nit"])
                file_stats["adaptive_nit"].append(adaptive["nit"])
                file_stats["fixed_nit_delta"].append(comparison["fixed_nit_delta"])
                file_stats["adaptive_nit_delta"].append(comparison["adaptive_nit_delta"])
                file_stats["adaptive_minus_fixed_nit"].append(comparison["adaptive_minus_fixed_nit"])
                file_stats["base_nfev"].append(baseline["nfev"])
                file_stats["fixed_nfev"].append(fixed["nfev"])
                file_stats["adaptive_nfev"].append(adaptive["nfev"])
                file_stats["fixed_nfev_delta"].append(comparison["fixed_nfev_delta"])
                file_stats["adaptive_nfev_delta"].append(comparison["adaptive_nfev_delta"])
                file_stats["adaptive_minus_fixed_nfev"].append(comparison["adaptive_minus_fixed_nfev"])
                file_stats["base_dist"].append(baseline["dist"])
                file_stats["fixed_dist"].append(fixed["dist"])
                file_stats["adaptive_dist"].append(adaptive["dist"])
                file_stats["fixed_solution_delta"].append(comparison["fixed_solution_delta"])
                file_stats["adaptive_solution_delta"].append(comparison["adaptive_solution_delta"])
                file_stats["adaptive_vs_fixed_solution_delta"].append(
                    comparison["adaptive_vs_fixed_solution_delta"]
                )
                overall["base_nit"].append(baseline["nit"])
                overall["fixed_nit"].append(fixed["nit"])
                overall["adaptive_nit"].append(adaptive["nit"])
                overall["fixed_nit_delta"].append(comparison["fixed_nit_delta"])
                overall["adaptive_nit_delta"].append(comparison["adaptive_nit_delta"])
                overall["adaptive_minus_fixed_nit"].append(comparison["adaptive_minus_fixed_nit"])
                overall["base_nfev"].append(baseline["nfev"])
                overall["fixed_nfev"].append(fixed["nfev"])
                overall["adaptive_nfev"].append(adaptive["nfev"])
                overall["fixed_nfev_delta"].append(comparison["fixed_nfev_delta"])
                overall["adaptive_nfev_delta"].append(comparison["adaptive_nfev_delta"])
                overall["adaptive_minus_fixed_nfev"].append(comparison["adaptive_minus_fixed_nfev"])
                overall["base_dist"].append(baseline["dist"])
                overall["fixed_dist"].append(fixed["dist"])
                overall["adaptive_dist"].append(adaptive["dist"])
                overall["fixed_solution_delta"].append(comparison["fixed_solution_delta"])
                overall["adaptive_solution_delta"].append(comparison["adaptive_solution_delta"])
                overall["adaptive_vs_fixed_solution_delta"].append(
                    comparison["adaptive_vs_fixed_solution_delta"]
                )

                previous_delta = {
                    "a": abs(baseline["a"] - a_prev),
                    "b": abs(baseline["b"] - b_prev),
                    "g": abs(baseline["g"] - g_prev),
                }

                # Advance the canonical sequence using the current production path.
                a_prev, b_prev, g_prev = baseline["a"], baseline["b"], baseline["g"]

        if file_stats["base_nit"]:
            count = len(file_stats["base_nit"])
            fixed_nit_wins = sum(delta < 0 for delta in file_stats["fixed_nit_delta"])
            adaptive_nit_wins = sum(delta < 0 for delta in file_stats["adaptive_nit_delta"])
            adaptive_vs_fixed_nit_wins = sum(delta < 0 for delta in file_stats["adaptive_minus_fixed_nit"])
            fixed_nfev_wins = sum(delta < 0 for delta in file_stats["fixed_nfev_delta"])
            adaptive_nfev_wins = sum(delta < 0 for delta in file_stats["adaptive_nfev_delta"])
            adaptive_vs_fixed_nfev_wins = sum(
                delta < 0 for delta in file_stats["adaptive_minus_fixed_nfev"]
            )
            print(f"\n  --- Summary for {fname} ({count} MC stage solves) ---")
            print(
                f"  nit median:  base={np.median(file_stats['base_nit']):.1f}  "
                f"fixed={np.median(file_stats['fixed_nit']):.1f}  "
                f"adaptive={np.median(file_stats['adaptive_nit']):.1f}"
            )
            print(
                f"  nit wins vs base: fixed={fixed_nit_wins}/{count}  "
                f"adaptive={adaptive_nit_wins}/{count}  "
                f"adaptive vs fixed={adaptive_vs_fixed_nit_wins}/{count}"
            )
            print(
                f"  nfev median: base={np.median(file_stats['base_nfev']):.1f}  "
                f"fixed={np.median(file_stats['fixed_nfev']):.1f}  "
                f"adaptive={np.median(file_stats['adaptive_nfev']):.1f}"
            )
            print(
                f"  nfev wins vs base: fixed={fixed_nfev_wins}/{count}  "
                f"adaptive={adaptive_nfev_wins}/{count}  "
                f"adaptive vs fixed={adaptive_vs_fixed_nfev_wins}/{count}"
            )
            print(
                f"  dist median: base={np.median(file_stats['base_dist']):.2e}  "
                f"fixed={np.median(file_stats['fixed_dist']):.2e}  "
                f"adaptive={np.median(file_stats['adaptive_dist']):.2e}"
            )
            print(
                f"  Δsol vs base: fixed={_format_or_na(_finite_median(file_stats['fixed_solution_delta']), '{:.2e}')}  "
                f"adaptive={_format_or_na(_finite_median(file_stats['adaptive_solution_delta']), '{:.2e}')}  "
                f"adaptive vs fixed={_format_or_na(_finite_median(file_stats['adaptive_vs_fixed_solution_delta']), '{:.2e}')}"
            )

    if overall["base_nit"]:
        count = len(overall["base_nit"])
        fixed_nit_wins = sum(delta < 0 for delta in overall["fixed_nit_delta"])
        adaptive_nit_wins = sum(delta < 0 for delta in overall["adaptive_nit_delta"])
        adaptive_vs_fixed_nit_wins = sum(delta < 0 for delta in overall["adaptive_minus_fixed_nit"])
        fixed_nfev_wins = sum(delta < 0 for delta in overall["fixed_nfev_delta"])
        adaptive_nfev_wins = sum(delta < 0 for delta in overall["adaptive_nfev_delta"])
        adaptive_vs_fixed_nfev_wins = sum(delta < 0 for delta in overall["adaptive_minus_fixed_nfev"])
        print(f"\n{'=' * 88}")
        print(f"Overall summary ({count} MC stage solves)")
        print(f"{'=' * 88}")
        print(
            f"nit median:  base={np.median(overall['base_nit']):.1f}  "
            f"fixed={np.median(overall['fixed_nit']):.1f}  "
            f"adaptive={np.median(overall['adaptive_nit']):.1f}"
        )
        print(
            f"nit wins vs base: fixed={fixed_nit_wins}/{count}  "
            f"adaptive={adaptive_nit_wins}/{count}  "
            f"adaptive vs fixed={adaptive_vs_fixed_nit_wins}/{count}"
        )
        print(
            f"nfev median: base={np.median(overall['base_nfev']):.1f}  "
            f"fixed={np.median(overall['fixed_nfev']):.1f}  "
            f"adaptive={np.median(overall['adaptive_nfev']):.1f}"
        )
        print(
            f"nfev wins vs base: fixed={fixed_nfev_wins}/{count}  "
            f"adaptive={adaptive_nfev_wins}/{count}  "
            f"adaptive vs fixed={adaptive_vs_fixed_nfev_wins}/{count}"
        )
        print(
            f"dist median: base={np.median(overall['base_dist']):.2e}  "
            f"fixed={np.median(overall['fixed_dist']):.2e}  "
            f"adaptive={np.median(overall['adaptive_dist']):.2e}"
        )
        print(
            f"Δsol vs base: fixed={_format_or_na(_finite_median(overall['fixed_solution_delta']), '{:.2e}')}  "
            f"adaptive={_format_or_na(_finite_median(overall['adaptive_solution_delta']), '{:.2e}')}  "
            f"adaptive vs fixed={_format_or_na(_finite_median(overall['adaptive_vs_fixed_solution_delta']), '{:.2e}')}"
        )


def parse_args():
    """Parse CLI arguments for the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mc",
        type=int,
        default=1,
        help="number of lost-light stages to compare after the initial no-MC inversion",
    )
    parser.add_argument(
        "--photons",
        type=int,
        default=2000,
        help="photons per MC lost-light update",
    )
    parser.add_argument(
        "--a-step",
        type=float,
        default=1e-3,
        help="absolute simplex step for a",
    )
    parser.add_argument(
        "--b-rel-step",
        type=float,
        default=1e-2,
        help="relative simplex step for b",
    )
    parser.add_argument(
        "--g-step",
        type=float,
        default=1e-3,
        help="absolute simplex step for g",
    )
    parser.add_argument(
        "--adaptive-scale",
        type=float,
        default=1.0,
        help="scale factor applied to the previous MC stage's |Δparameter|",
    )
    parser.add_argument(
        "--a-min-step",
        type=float,
        default=1e-5,
        help="minimum adaptive simplex step for a",
    )
    parser.add_argument(
        "--b-min-step",
        type=float,
        default=1e-4,
        help="minimum adaptive simplex step for b",
    )
    parser.add_argument(
        "--g-min-step",
        type=float,
        default=1e-5,
        help="minimum adaptive simplex step for g",
    )
    parser.add_argument(
        "filters",
        nargs="*",
        help="optional file filters such as sample-E or sample-E.rxt",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mc < 1:
        raise SystemExit("--mc must be at least 1")

    run_benchmark(
        mc_iterations=args.mc,
        photons=args.photons,
        filenames=_match_files(args.filters),
        simplex_kwargs={
            "a_step": args.a_step,
            "b_rel_step": args.b_rel_step,
            "g_step": args.g_step,
            "adaptive_scale": args.adaptive_scale,
            "a_min_step": args.a_min_step,
            "b_min_step": args.b_min_step,
            "g_min_step": args.g_min_step,
        },
    )
