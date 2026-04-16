"""Benchmark: Grid vs AGrid warm-start quality for two-parameter inversion.

For each combination of (search_mode, target_case, grid_type, density_param),
records:
  - grid_points   : number of RT evaluations spent building the grid
  - optimizer_evals: Nelder-Mead function evaluations
  - total_evals   : grid_points + optimizer_evals
  - warm_start_err: L1 distance in (a,b,g) space from warm start to truth
  - final_distance : residual at convergence

Run with::

    .venv/bin/python tests/bench_grid.py

Produces bench_grid_results.csv in the repo root.

KEY FINDINGS (from initial run 2026-04-14)
------------------------------------------
find_ab / find_ag
  Optimizer evals are 40-70 across all grid types and densities.
  The warm start barely affects Nelder-Mead convergence.
  Grid(N=21) and AGrid(medium) give essentially identical optimizer performance.
  Knee is reached at ~N=11 for Grid or AGrid(coarse, ~130 pts).

find_bg
  Grid(N=21) gives a terrible warm start (ws_err~0.47, 88 opt evals median).
  Grid(N=51) improves to 72 opt evals but costs 2601 build points.
  AGrid(fine, tol=0.01, max_depth=8) achieves 59 opt evals with only ~833 pts.
  AGrid(fine) reduces optimizer iterations by ~33% vs Grid(N=21) for find_bg.

IMPLEMENTED RULE (Option B from GRID_PLAN.md)
  find_bg  -> AGrid(tol=0.01, max_depth=8, min_depth=3)
  find_ab/find_ag -> Grid(N=21)
  Override with exp.use_adaptive_grid = True/False.
"""

import csv
import pathlib
import sys
import numpy as np

# Allow running from repo root without install
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import iadpython  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTPUT_CSV = REPO_ROOT / "bench_grid_results.csv"

# ---------------------------------------------------------------------------
# Benchmark cases
# ---------------------------------------------------------------------------

FIND_AB_CASES = [
    # (a_true, b_true, g_fixed)
    (0.3, 0.3, 0.0),
    (0.5, 1.0, 0.0),
    (0.7, 2.0, 0.0),
    (0.9, 5.0, 0.5),
    (0.95, 10.0, 0.9),
    (0.99, 100.0, 0.0),
]

FIND_AG_CASES = [
    # (a_true, b_fixed, g_true)
    (0.5, 2.0, 0.0),
    (0.7, 4.0, 0.5),
    (0.9, 4.0, 0.8),
    (0.95, 2.0, 0.95),
    (0.5, 4.0, -0.5),
]

FIND_BG_CASES = [
    # (a_fixed, b_true, g_true)
    (0.3, 0.5, 0.0),
    (0.7, 2.0, 0.3),
    (0.9, 10.0, 0.7),
    (0.9, 50.0, 0.9),
    (0.5, 0.1, -0.5),
]

# Grid(N) density levels
GRID_N_VALUES = [5, 11, 21, 31, 51, 101]

# AGrid density levels: (label, tol, max_depth, min_depth)
AGRID_CONFIGS = [
    ("coarse", 0.05, 4, 2),
    ("medium", 0.03, 6, 2),
    ("fine", 0.01, 8, 3),
    ("very_fine", 0.003, 10, 3),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _forward(a, b, g):
    """Compute (m_r, m_t) for a sample with matched boundaries."""
    s = iadpython.Sample(a=a, b=b, g=g)
    ur1, ut1, _, _ = s.rt()
    return float(ur1), float(ut1)


def _run_grid(search, target, grid_constant, mr_target, mt_target, N):
    """Run inversion with Grid(N); return result dict."""
    exp = _make_experiment(search, target, grid_constant, mr_target, mt_target)
    exp.use_adaptive_grid = False
    exp.grid = iadpython.Grid(N=N)

    exp.sample.rt_evals = 0
    a, b, g = exp.invert_rt()

    grid_evals = getattr(exp, "_grid_evals", 0)
    opt_evals = getattr(exp, "_optimizer_evals", exp.sample.rt_evals - grid_evals)

    ws_err = _warm_start_error(search, target, grid_constant, exp, mr_target, mt_target, N=N)

    return dict(
        grid_type="Grid",
        density_param=N,
        grid_points=N * N,
        grid_evals=grid_evals,
        optimizer_evals=opt_evals,
        total_evals=exp.sample.rt_evals,
        warm_start_err=ws_err,
        final_distance=exp.final_distance,
        a_out=a,
        b_out=b,
        g_out=g,
    )


def _run_agrid(search, target, grid_constant, mr_target, mt_target, label, tol, max_depth, min_depth):
    """Run inversion with AGrid; return result dict."""
    exp = _make_experiment(search, target, grid_constant, mr_target, mt_target)
    exp.use_adaptive_grid = True
    exp.adaptive_grid_tol = tol
    exp.adaptive_grid_max_depth = max_depth
    exp.grid = iadpython.AGrid(tol=tol, max_depth=max_depth, min_depth=min_depth)

    exp.sample.rt_evals = 0
    a, b, g = exp.invert_rt()

    grid_evals = getattr(exp, "_grid_evals", 0)
    opt_evals = getattr(exp, "_optimizer_evals", exp.sample.rt_evals - grid_evals)
    grid_pts = len(exp.grid) if exp.grid is not None else 0

    ws_err = _warm_start_error_agrid(
        search, target, grid_constant, exp, mr_target, mt_target, tol=tol, max_depth=max_depth, min_depth=min_depth
    )

    return dict(
        grid_type="AGrid",
        density_param=label,
        grid_points=grid_pts,
        grid_evals=grid_evals,
        optimizer_evals=opt_evals,
        total_evals=exp.sample.rt_evals,
        warm_start_err=ws_err,
        final_distance=exp.final_distance,
        a_out=a,
        b_out=b,
        g_out=g,
    )


def _make_experiment(search, target, grid_constant, mr_target, mt_target):
    """Build an Experiment for the given search mode and target."""
    if search == "find_ab":
        a_true, b_true, g_fixed = target
        exp = iadpython.Experiment(r=mr_target, t=mt_target, default_g=g_fixed)
    elif search == "find_ag":
        a_true, b_fixed, g_true = target
        exp = iadpython.Experiment(r=mr_target, t=mt_target, default_b=b_fixed)
    elif search == "find_bg":
        a_fixed, b_true, g_true = target
        exp = iadpython.Experiment(r=mr_target, t=mt_target, default_a=a_fixed)
    else:
        raise ValueError(f"Unknown search: {search}")
    exp.verbosity = 0
    return exp


def _warm_start_error(search, target, grid_constant, exp, mr_target, mt_target, N):
    """Re-build a fresh Grid(N) and get the warm-start point; return L1 error."""
    g = iadpython.Grid(N=N)
    g.calc(exp, default=grid_constant)
    a_ws, b_ws, g_ws = g.min_abg(mr_target, mt_target)
    return _abg_error(search, target, a_ws, b_ws, g_ws)


def _warm_start_error_agrid(search, target, grid_constant, exp, mr_target, mt_target, tol, max_depth, min_depth):
    """Re-build a fresh AGrid and get the warm-start point; return L1 error."""
    ag = iadpython.AGrid(tol=tol, max_depth=max_depth, min_depth=min_depth)
    ag.calc(exp, default=grid_constant, search=exp.search)
    a_ws, b_ws, g_ws = ag.min_abg(mr_target, mt_target)
    return _abg_error(search, target, a_ws, b_ws, g_ws)


def _abg_error(search, target, a_ws, b_ws, g_ws):
    """L1 distance in (a,b,g) from warm-start to truth (log b for stability)."""
    if search == "find_ab":
        a_true, b_true, _ = target
        return abs(a_ws - a_true) + abs(np.log1p(b_ws) - np.log1p(b_true))
    if search == "find_ag":
        a_true, _, g_true = target
        return abs(a_ws - a_true) + abs(g_ws - g_true)
    if search == "find_bg":
        _, b_true, g_true = target
        return abs(np.log1p(b_ws) - np.log1p(b_true)) + abs(g_ws - g_true)
    return float("inf")


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------


def run_benchmark():
    """Run all benchmark cases; write CSV; print summary."""
    fieldnames = [
        "search",
        "case_idx",
        "a_true",
        "b_true",
        "g_true",
        "mr_target",
        "mt_target",
        "grid_type",
        "density_param",
        "grid_points",
        "grid_evals",
        "optimizer_evals",
        "total_evals",
        "warm_start_err",
        "final_distance",
        "a_out",
        "b_out",
        "g_out",
    ]

    rows = []
    mode_cases = [
        ("find_ab", FIND_AB_CASES),
        ("find_ag", FIND_AG_CASES),
        ("find_bg", FIND_BG_CASES),
    ]

    for search, cases in mode_cases:
        for ci, target in enumerate(cases):
            if search == "find_ab":
                a_true, b_true, g_fixed = target
                g_true = g_fixed
                grid_constant = g_fixed
            elif search == "find_ag":
                a_true, b_fixed, g_true = target
                b_true = b_fixed
                grid_constant = b_fixed
            else:  # find_bg
                a_fixed, b_true, g_true = target
                a_true = a_fixed
                grid_constant = a_fixed

            mr_t, mt_t = _forward(a_true, b_true, g_true)
            base = dict(
                search=search, case_idx=ci, a_true=a_true, b_true=b_true, g_true=g_true, mr_target=mr_t, mt_target=mt_t
            )

            print(f"  {search} case {ci}: a={a_true} b={b_true} g={g_true} " f"mr={mr_t:.4f} mt={mt_t:.4f}", flush=True)

            for N in GRID_N_VALUES:
                try:
                    r = _run_grid(search, target, grid_constant, mr_t, mt_t, N)
                    rows.append({**base, **r})
                    print(
                        f"    Grid N={N:3d}: grid_pts={r['grid_points']:5d} "
                        f"grid_evals={r['grid_evals']:4d} "
                        f"opt={r['optimizer_evals']:4d} "
                        f"ws_err={r['warm_start_err']:.4f} "
                        f"dist={r['final_distance']:.2e}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"    Grid N={N}: ERROR {exc}", flush=True)

            for label, tol, max_depth, min_depth in AGRID_CONFIGS:
                try:
                    r = _run_agrid(search, target, grid_constant, mr_t, mt_t, label, tol, max_depth, min_depth)
                    rows.append({**base, **r})
                    print(
                        f"    AGrid {label:9s}: grid_pts={r['grid_points']:5d} "
                        f"grid_evals={r['grid_evals']:4d} "
                        f"opt={r['optimizer_evals']:4d} "
                        f"ws_err={r['warm_start_err']:.4f} "
                        f"dist={r['final_distance']:.2e}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"    AGrid {label}: ERROR {exc}", flush=True)

    with open(OUTPUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {OUTPUT_CSV}")
    _print_summary(rows)


def _print_summary(rows):
    """Print a compact summary table grouped by search mode and grid type."""
    from collections import defaultdict

    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["search"], r["grid_type"], str(r["density_param"]))].append(r)

    print("\n--- Summary: median total_evals and warm_start_err ---")
    print(
        f"{'search':8s} {'type':6s} {'density':10s} {'pts_med':8s} " f"{'opt_med':8s} {'tot_med':8s} {'ws_err_med':10s}"
    )
    print("-" * 70)

    order = [
        ("find_ab", "Grid"),
        ("find_ab", "AGrid"),
        ("find_ag", "Grid"),
        ("find_ag", "AGrid"),
        ("find_bg", "Grid"),
        ("find_bg", "AGrid"),
    ]
    density_order = [str(n) for n in GRID_N_VALUES] + [label for label, *_ in AGRID_CONFIGS]

    for search, gtype in order:
        keys = [(search, gtype, d) for d in density_order if (search, gtype, d) in grouped]
        for key in keys:
            rs = grouped[key]
            pts = np.median([r["grid_points"] for r in rs])
            opt = np.median([r["optimizer_evals"] for r in rs])
            tot = np.median([r["total_evals"] for r in rs])
            ws = np.median([r["warm_start_err"] for r in rs])
            print(f"{key[0]:8s} {key[1]:6s} {key[2]:10s} {pts:8.0f} " f"{opt:8.0f} {tot:8.0f} {ws:10.4f}")
        if keys:
            print()


if __name__ == "__main__":
    print("Running grid benchmark …")
    run_benchmark()
