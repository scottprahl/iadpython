"""Adaptive grid helper for inverse adding-doubling calculations.

`AGrid` builds an adaptive 2D grid for the two-parameter search modes:
`find_ab`, `find_ag`, and `find_bg`. Corner values are recursively sampled until
the local variation in `(ur1 + ut1)` falls below a tolerance, which typically
requires fewer RT evaluations than a dense uniform grid.
"""

from __future__ import annotations

import numpy as np

from .cache import Cache


class AGrid:
    """Adaptive cache-backed grid for 2-parameter inversion warm starts."""

    # for each mode, which axis is held fixed (and which two vary)
    _modes = {
        "find_ab": ("g", ("a", "b")),
        "find_ag": ("b", ("a", "g")),
        "find_bg": ("a", ("b", "g")),
    }

    # overall ranges for each axis
    _ranges = {
        "a": (0.0, 1.0),
        "b": (0.0, 10.0),
        "g": (-0.99, 0.99),
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
            tol: Subdivision threshold for center interpolation error of `(ur1 + ut1)`.
            max_depth: Maximum recursion depth for quadtree subdivision.
            min_depth: Minimum enforced subdivision depth.
            default: Fixed-axis value for the selected search mode.
        """
        self.cache = Cache()
        self.exp = exp
        self.search = search
        self.default = default
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

    def _sum_val(self, a: float, b: float, g: float) -> float:
        """Ensure cached, then return ur1+ut1."""
        self.eval_f(a, b, g)
        ur1, ut1, _, _ = self.cache.get(a, b, g)
        return ur1 + ut1

    def _subdivide(self, x0, x1, y0, y1, depth, collect):
        """Recursively subdivide the rectangle [x0,x1]×[y0,y1]."""
        xm = 0.5 * (x0 + x1)
        ym = 0.5 * (y0 + y1)

        # corners
        v00 = self._sum_val(*self._make_args(x0, y0))
        v10 = self._sum_val(*self._make_args(x1, y0))
        v01 = self._sum_val(*self._make_args(x0, y1))
        v11 = self._sum_val(*self._make_args(x1, y1))
        vcc = self._sum_val(*self._make_args(xm, ym))

        interp = 0.25 * (v00 + v10 + v01 + v11)
        err = np.abs(vcc - interp)
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
        """Build (a,b,g) from the 2D coords (u,v) and the fixed axis."""
        if self.fixed_axis == "g":
            return (u, v, self.default)
        if self.fixed_axis == "b":
            return (u, self.default, v)
        if self.fixed_axis == "a":
            return (self.default, u, v)
        raise ValueError(f"Bad fixed axis: {self.fixed_axis!r}")

    def build(self) -> None:
        """
        Build an adaptive quadtree.

        The quadtree is created over the two free axes determined by `search`,
        subdividing until center interpolation error in `(ur1 + ut1)` is below
        `tol` (after `min_depth`) or `max_depth` is reached.
        """
        if self.search is None or self.fixed_axis is None:
            raise ValueError("Search mode must be set before build()")
        if self.default is None:
            raise ValueError("Fixed-axis default must be set before build()")
        if self.exp is None:
            raise ValueError("Experiment must be set before build()")

        axis0, axis1 = self.vary_axes

        # grab the ranges for the two free axes
        (u0, u1), (v0, v1) = self._ranges[axis0], self._ranges[axis1]

        # collect the (u,v) corner points that meet the tolerance
        uv_points: set[tuple[float, float]] = set()
        self._subdivide(u0, u1, v0, v1, depth=0, collect=uv_points)

        for u, v in uv_points:
            a, b, g = self._make_args(u, v)
            self.eval_f(a, b, g)

    def min_abg(self, mr, mt):
        """Find closest a, b, g closest to mr and mt."""
        minimum = float("inf")
        a_min = 0
        b_min = 0
        g_min = 0

        for a, b, g, ur1, ut1, _uru, _utu in self.cache:
            delta = np.abs(mr - ur1) + np.abs(mt - ut1)
            if delta < minimum:
                minimum = delta
                a_min, b_min, g_min = a, b, g

        if minimum == float("inf"):
            raise RuntimeError("AGrid cache is empty. Call calc() first.")
        return a_min, b_min, g_min

    def is_stale(self, default_value, search: str | None = None):
        """Decide if current grid is still useful."""
        if self.default is None:
            return True
        if search is not None and self.search != search:
            return True
        if self.default != default_value:
            return True
        return False

    def square_grid(self, N=21):
        """Return a square grid."""
        aa = np.linspace(self._ranges["a"][0], self._ranges["a"][1], N)
        bb = np.linspace(self._ranges["b"][0], self._ranges["b"][1], N)
        gg = np.linspace(self._ranges["g"][0], self._ranges["g"][1], N)

        if self.search == "find_ab":
            ggg = np.full((N, N), self.default)
            aaa, bbb = np.meshgrid(aa, bb)

        if self.search == "find_bg":
            aaa = np.full((N, N), self.default)
            ggg, bbb = np.meshgrid(gg, bb)

        if self.search == "find_ag":
            bbb = np.full((N, N), self.default)
            aaa, ggg = np.meshgrid(aa, gg)

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
        import matplotlib.pyplot as plt

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
