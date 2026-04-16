"""Adaptive grid helper for inverse adding-doubling calculations.

`AGrid` builds an adaptive 2D grid for the two-parameter search modes:
`find_ab`, `find_ag`, and `find_bg`. Corner values are recursively sampled until
the local variation in corrected measured-space values falls below a tolerance,
which typically requires fewer RT evaluations than a dense uniform grid.
"""

from __future__ import annotations

import numpy as np

from .cache import Cache
from .grid import _grid_context_is_stale, _grid_stale_context, _nonlinear_a_coord, _nonlinear_g_coord

# Mirrors CWEB iad_type.w: @d MAX_ABS_G 0.999999
_MAX_ABS_G = 0.999999

# Log-b limits mirror CWEB iad_calc.w Fill_AB_Grid / Fill_BG_Grid
_MIN_LOG_B = -8.0
_MAX_LOG_B_AB = 8.0  # exp(+8)  ≈ 2981  (find_ab, find_ag)
_MAX_LOG_B_BG = 10.0  # exp(+10) ≈ 22026 (find_bg)


class AGrid:
    """Adaptive cache-backed grid for 2-parameter inversion warm starts."""

    _CLOSURE_AXIS_VALUE_LIMIT = 4

    # for each mode, which axis is held fixed (and which two vary)
    _modes = {
        "find_ab": ("g", ("a", "b")),
        "find_ag": ("b", ("a", "g")),
        "find_bg": ("a", ("b", "g")),
    }

    # overall coordinate ranges for each axis; b is stored in log space, while
    # a and g use unit coordinates that map through the same nonlinear spacing
    # functions used by the dense Grid warm-start.
    _ranges = {
        "a": (0.0, 1.0),
        "b": (_MIN_LOG_B, _MAX_LOG_B_AB),  # log-b; find_bg overrides upper limit in build()
        "g": (0.0, 1.0),
    }

    def __init__(
        self,
        exp=None,
        search: str | None = None,
        tol: float = 0.03,
        max_depth: int = 6,
        min_depth: int = 2,
        default: float | None = None,
    ):
        """Initialize adaptive grid.

        Args:
            exp: Optional experiment. If provided with `search`, the grid is built.
            search: Search mode (`find_ab`, `find_ag`, or `find_bg`).
            tol: Subdivision threshold for center interpolation error in corrected
                measured space.
            max_depth: Maximum recursion depth for quadtree subdivision.
            min_depth: Minimum enforced subdivision depth.
            default: Fixed-axis value for the selected search mode.
        """
        self.cache = Cache()
        self.exp = exp
        self.search = search
        self.default = default
        self._stale_context = None
        self.tol = tol
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.fixed_axis = None
        self.vary_axes = None

        if search is not None:
            self._set_mode(search)

        if exp is not None and search is not None:
            self.calc(exp, default=default, search=search)

    def _set_mode(self, search: str) -> None:
        """Set fixed and varying axes based on search mode."""
        try:
            self.fixed_axis, self.vary_axes = self._modes[search]
        except KeyError as exc:
            raise ValueError(f"Unknown search mode: {search!r}") from exc
        self.search = search

    def _default_for_search(self) -> float:
        """Return fixed-axis value for current search mode."""
        if self.exp is None:
            raise ValueError("AGrid.calc requires an experiment")
        if self.search == "find_ab":
            return self.exp.sample.g
        if self.search == "find_ag":
            return self.exp.sample.b
        if self.search == "find_bg":
            return self.exp.sample.a
        raise ValueError(f"Unsupported search mode: {self.search!r}")

    def calc(self, exp, default: float | None = None, search: str | None = None):
        """Build (or rebuild) adaptive grid for an experiment.

        Args:
            exp: Experiment instance used to evaluate RT.
            default: Fixed-axis value for the selected search mode.
            search: Optional search mode override.
        """
        self.exp = exp
        if search is None:
            search = self.search or exp.search
        self._set_mode(search)
        self.default = self._default_for_search() if default is None else default
        self.cache = Cache()
        self.build()
        self._stale_context = _grid_stale_context(self.exp, self.search, self.default)

    def __str__(self):
        """Return basic details as a string for printing."""
        s = "---------------- Grid ---------------\n"
        if self.search is None:
            s += "search  = None\n"
        else:
            s += "search = %s\n" % self.search

        if self.default is None:
            s += "default = None\n"
        else:
            s += "default = %.5f\n" % self.default

        s += "entries = %d" % len(self.cache)
        return s

    def eval_f(self, a: float, b: float, g: float) -> None:
        """Memoised evaluation that fills the cache if missing."""
        if self.cache.get(a, b, g) is None:
            self.exp.sample.a = a
            self.exp.sample.b = b
            self.exp.sample.g = g
            ur1, ut1, uru, utu = self.exp.sample.rt()
            self.cache.put(a, b, g, ur1, ut1, uru, utu)

    def _measured_val(self, a: float, b: float, g: float) -> tuple[float, float]:
        """Ensure cached, then return corrected measured `(M_R, M_T)`."""
        self.eval_f(a, b, g)
        ur1, ut1, uru, utu = self.cache.get(a, b, g)
        if self.exp is None:
            return ur1, ut1

        m_r, m_t, _delta = self.exp.measurement_distance_from_raw(
            ur1,
            ut1,
            uru,
            utu,
            include_lost=False,
            a=a,
            b=b,
            g=g,
        )
        return float(m_r), float(m_t)

    def _subdivide(self, x0, x1, y0, y1, depth, collect):
        """Recursively subdivide the rectangle [x0,x1]×[y0,y1]."""
        xm = 0.5 * (x0 + x1)
        ym = 0.5 * (y0 + y1)

        mr00, mt00 = self._measured_val(*self._make_args(x0, y0))
        mr10, mt10 = self._measured_val(*self._make_args(x1, y0))
        mr01, mt01 = self._measured_val(*self._make_args(x0, y1))
        mr11, mt11 = self._measured_val(*self._make_args(x1, y1))
        mrcc, mtcc = self._measured_val(*self._make_args(xm, ym))

        interp_r = 0.25 * (mr00 + mr10 + mr01 + mr11)
        interp_t = 0.25 * (mt00 + mt10 + mt01 + mt11)
        err = np.abs(mrcc - interp_r) + np.abs(mtcc - interp_t)
        need_split = (depth < self.min_depth) or (err > self.tol and depth < self.max_depth)

        if need_split:
            # recurse into four sub-quads
            for nx0, nx1, ny0, ny1 in [
                (x0, xm, y0, ym),
                (xm, x1, y0, ym),
                (x0, xm, ym, y1),
                (xm, x1, ym, y1),
            ]:
                self._subdivide(nx0, nx1, ny0, ny1, depth + 1, collect)
        else:
            # accept corners and centre for the leaf cell
            collect.update(
                {
                    (x0, y0),
                    (x1, y0),
                    (x0, y1),
                    (x1, y1),
                    (xm, ym),
                }
            )

    def _make_args(self, u, v):
        """Build (a,b,g) from the 2D coords (u,v) and the fixed axis.

        When b is a free axis its coordinate is stored in log space; exponentiate
        before passing to the RT solver. The free a and g coordinates use the
        same nonlinear transforms as Grid, so adaptive subdivision happens in
        the transformed coordinate space rather than in raw physical space.
        """
        if self.fixed_axis == "g":
            # u = a, v = log_b
            return (_nonlinear_a_coord(u), np.exp(v), self.default)
        if self.fixed_axis == "b":
            # u = a, v = g  (b is fixed — already a real value)
            return (_nonlinear_a_coord(u), self.default, _nonlinear_g_coord(v))
        if self.fixed_axis == "a":
            # u = log_b, v = g
            return (self.default, np.exp(u), _nonlinear_g_coord(v))
        raise ValueError(f"Bad fixed axis: {self.fixed_axis!r}")

    def build(self) -> None:
        """
        Build an adaptive quadtree.

        The quadtree is created over the two free axes determined by `search`,
        subdividing until center interpolation error in corrected measured space
        is below `tol` (after `min_depth`) or `max_depth` is reached.
        """
        if self.search is None or self.fixed_axis is None:
            raise ValueError("Search mode must be set before build()")
        if self.default is None:
            raise ValueError("Fixed-axis default must be set before build()")
        if self.exp is None:
            raise ValueError("Experiment must be set before build()")

        axis0, axis1 = self.vary_axes

        # grab the ranges for the two free axes; for find_bg use the wider b limit
        ranges = dict(self._ranges)
        if self.search == "find_bg":
            ranges["b"] = (_MIN_LOG_B, _MAX_LOG_B_BG)
        (u0, u1), (v0, v1) = ranges[axis0], ranges[axis1]

        # collect the (u,v) corner points that meet the tolerance
        uv_points: set[tuple[float, float]] = set()
        self._subdivide(u0, u1, v0, v1, depth=0, collect=uv_points)

        for u, v in uv_points:
            a, b, g = self._make_args(u, v)
            self.eval_f(a, b, g)

    def min_abg(self, mr, mt, exp=None):
        """Find closest a, b, g closest to mr and mt."""
        ranked = []
        for a, b, g, ur1, ut1, uru, utu in self.cache:
            ranked.append((self._distance(mr, mt, a, b, g, ur1, ut1, uru, utu, exp), a, b, g))

        if not ranked:
            raise RuntimeError("AGrid cache is empty. Call calc() first.")

        ranked.sort(key=lambda item: item[0])
        best = ranked[0]

        closure_best = self._local_cartesian_closure(mr, mt, ranked, exp=exp)
        if closure_best is not None and closure_best[0] < best[0]:
            best = closure_best

        _delta, a_min, b_min, g_min = best
        return a_min, b_min, g_min

    def _distance(self, mr, mt, a, b, g, ur1, ut1, uru, utu, exp=None):
        """Return candidate distance in raw or corrected measurement space."""
        if exp is None:
            return np.abs(mr - ur1) + np.abs(mt - ut1)

        _fit_r, _fit_t, delta = exp.measurement_distance_from_raw(
            ur1,
            ut1,
            uru,
            utu,
            include_lost=False,
            a=a,
            b=b,
            g=g,
        )
        return delta

    def _candidate_axis_values(self, ranked, axis):
        """Return a few distinct physical axis values from the best candidates."""
        values = []
        for _delta, a, b, g in ranked:
            value = {"a": a, "b": b, "g": g}[axis]
            if value not in values:
                values.append(value)
            if len(values) >= self._CLOSURE_AXIS_VALUE_LIMIT:
                break
        return values

    def _abg_from_axis_values(self, axis0, value0, axis1, value1):
        """Build `(a, b, g)` from physical values for the two varying axes."""
        values = {axis0: value0, axis1: value1, self.fixed_axis: self.default}
        return values["a"], values["b"], values["g"]

    def _local_cartesian_closure(self, mr, mt, ranked, exp=None):
        """Evaluate a small local Cartesian closure across the best coarse samples.

        Adaptive sampling can leave a good basin represented by diagonal corners
        without ever sampling the off-diagonal combination that is actually
        closest to the target.  Before committing to a warm start, enrich the
        best coarse neighborhood by crossing a few top-ranked axis values.
        """
        if self.exp is None or self.vary_axes is None or len(ranked) < 2:
            return None

        axis0, axis1 = self.vary_axes
        values0 = self._candidate_axis_values(ranked, axis0)
        values1 = self._candidate_axis_values(ranked, axis1)
        if len(values0) < 2 or len(values1) < 2:
            return None

        best = None
        for value0 in values0:
            for value1 in values1:
                a, b, g = self._abg_from_axis_values(axis0, value0, axis1, value1)
                self.eval_f(a, b, g)
                ur1, ut1, uru, utu = self.cache.get(a, b, g)
                candidate = (
                    self._distance(mr, mt, a, b, g, ur1, ut1, uru, utu, exp=exp),
                    a,
                    b,
                    g,
                )
                if best is None or candidate[0] < best[0]:  # pylint: disable=unsubscriptable-object
                    best = candidate

        return best

    def is_stale(self, exp, default_value, search: str | None = None):
        """Decide if current grid is still useful."""
        if self.default is None:
            return True
        if search is None:
            search = getattr(exp, "search", self.search)
        return _grid_context_is_stale(self._stale_context, exp, search, default_value)

    def square_grid(self, N=21):
        """Return a square grid."""
        aa = _nonlinear_a_coord(np.linspace(self._ranges["a"][0], self._ranges["a"][1], N))
        max_log_b = _MAX_LOG_B_BG if self.search == "find_bg" else _MAX_LOG_B_AB
        bb = np.exp(np.linspace(_MIN_LOG_B, max_log_b, N))
        gg = _nonlinear_g_coord(np.linspace(self._ranges["g"][0], self._ranges["g"][1], N))

        if self.search == "find_ab":
            ggg = np.full((N, N), self.default)
            aaa, bbb = np.meshgrid(aa, bb)
        elif self.search == "find_bg":
            aaa = np.full((N, N), self.default)
            ggg, bbb = np.meshgrid(gg, bb)
        elif self.search == "find_ag":
            bbb = np.full((N, N), self.default)
            aaa, ggg = np.meshgrid(aa, gg)
        else:
            raise ValueError(f"Unknown search mode: {self.search!r}")

        ur1 = np.zeros((N, N))
        ut1 = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                self.exp.sample.a = aaa[i, j]
                self.exp.sample.b = bbb[i, j]
                self.exp.sample.g = ggg[i, j]
                ur1[i, j], ut1[i, j], _, _ = self.exp.sample.rt()

        return aaa, bbb, ggg, ur1, ut1

    def plot(self):
        """Plot the grid."""
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

        # turn the cache into a (N×7) array
        data = np.array(list(self.cache))
        if data.size == 0:
            raise RuntimeError("Cache is empty – run calc() first")

        a, b, g, _ur1, _ut1, _uru, _utu = data.T

        A, B, G, R, T = self.square_grid()
        Z = R + T
        # for i in range(21):
        #     for j in range(21):
        #         print("%8.4f %8.4f %8.4f %8.4f" % (A[i,j], B[i,j], G[i,j], Z[i,j]))

        if self.search == "find_ab":
            plt.plot(b, a, "sb", markersize=1)
            plt.ylabel("albedo [-]")
            plt.xlabel("optical thickness [-]")
            plt.pcolormesh(B, A, Z, cmap="gray")

        if self.search == "find_bg":
            plt.plot(b, g, "sb", markersize=1)
            plt.ylabel("anisotropy [-]")
            plt.xlabel("optical thickness [-]")
            plt.pcolormesh(B, G, Z, cmap="gray")

        if self.search == "find_ag":
            plt.plot(a, g, "sb", markersize=1)
            plt.xlabel("albedo [-]")
            plt.ylabel("anisotropy [-]")
            plt.pcolormesh(A, G, Z, cmap="gray")

    def __len__(self):
        """Return number of sampled points in the adaptive cache."""
        return len(self.cache)

    def __iter__(self):
        """Iterate over cached adaptive grid entries."""
        return iter(self.cache)
