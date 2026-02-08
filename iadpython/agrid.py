"""Class for doing inverse adding-doubling calculations for a sample.

Example::

    >>> import iadpython

    >>> exp = iadpython.Experiment(0.5, 0.1, default_g=0.5)
    >>> grid = iadpython.AGrid(exp,'find_ab')
    >>> grid.calc(exp)
    >>> print(grid)
"""

import numpy as np
import matplotlib.pyplot as plt

from cache import Cache


class AGrid:
    """Class to track pre-calculated R & T values.

    There is a long story associated with these routines.  I spent a lot of time
    trying to find an empirical function to allow a guess at a starting value for
    the inversion routine.  Basically nothing worked very well.  There were
    too many special cases and what not.  So I decided to calculate a whole bunch
    of reflection and transmission values and keep their associated optical
    properties linked nearby.
    """

    # for each mode, which axis is held fixed (and which two vary)
    _modes = {
        "find_ab": ("g", ("a", "b")),
        "find_ag": ("b", ("a", "g")),
        "find_bg": ("a", ("b", "g")),
    }

    # overall ranges for each axis
    _ranges = {
        "a": (0.0, 1.0),
        "b": (-4, 4),
        "g": (-0.999, 0.999),
    }

    def __init__(self, exp, search: str, tol: float = 0.1, max_depth: int = 6):
        """Initialize grid."""
        self.cache = Cache()
        self.exp = exp
        self.search = search
        self.tol = tol
        self.max_depth = max_depth

        # one lookup, one error-check
        try:
            self.fixed_axis, self.vary_axes = self._modes[search]
        except KeyError:
            raise ValueError(f"Unknown search mode: {search!r}")

        # now build your quadtree
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

        cache_size = len(self.cache)
        s += "entries = %d" % cache_size
        return s

    def eval_f(self, a: float, b: float, g: float) -> None:
        """Memoised evaluation that fills the cache if missing."""
        if self.cache.get(a, b, g) is None:
            self.exp.sample.a = a
            self.exp.sample.b = 10**b
            self.exp.sample.g = g
            ur1, ut1, uru, utu = self.exp.sample.rt()
            self.cache.put(a, b, g, ur1, ut1, uru, utu)

    def _sum_val(self, a: float, b: float, g: float) -> float:
        """Ensure cached, then return ur1+ut1."""
        self.eval_f(a, b, g)
        ur1, ut1, _, _ = self.cache.get(a, b, g)
        return ur1 + ut1

    def _subdivide(self, x0, x1, y0, y1, fixed_axis, fixed_val, depth, collect):
        """Recursively subdivide the rectangle [x0,x1]×[y0,y1]."""
        # corners
        v00 = self._sum_val(*(self._make_args(x0, y0, fixed_axis, fixed_val)))
        v10 = self._sum_val(*(self._make_args(x1, y0, fixed_axis, fixed_val)))
        v01 = self._sum_val(*(self._make_args(x0, y1, fixed_axis, fixed_val)))
        v11 = self._sum_val(*(self._make_args(x1, y1, fixed_axis, fixed_val)))

        vmin, vmax = min(v00, v10, v01, v11), max(v00, v10, v01, v11)
        if vmax - vmin > self.tol and depth < self.max_depth:
            xm = 0.5 * (x0 + x1)
            ym = 0.5 * (y0 + y1)
            # recurse into four sub-quads
            for nx0, nx1, ny0, ny1 in [
                (x0, xm, y0, ym),
                (xm, x1, y0, ym),
                (x0, xm, ym, y1),
                (xm, x1, ym, y1),
            ]:
                self._subdivide(nx0, nx1, ny0, ny1, fixed_axis, fixed_val, depth + 1, collect)
        else:
            # accept the four corners
            collect.update(
                {
                    (x0, y0),
                    (x1, y0),
                    (x0, y1),
                    (x1, y1),
                }
            )

    def _make_args(self, u, v, fixed_axis, fixed_val):
        """Build (a,b,g) from the 2D coords (u,v) and the fixed axis."""
        if fixed_axis == "g":
            return (u, v, self.exp.default_g)
        if fixed_axis == "b":
            return (u, self.exp.default_b, v)
        if fixed_axis == "a":
            return (self.exp.default_a, u, v)
        raise ValueError(f"bad axis {fixed_axis}")

    def build(self) -> None:
        """
        Build an adaptive quadtree.

        The quadtree is created over the two free axes determined by `search`,
        subdividing until |Δ(ur1+ut1)| ≤ tol or max_depth is reached.
        """
        fixed_val = 0
        if self.search == "find_ab":
            fixed_val = self.exp.default_g
        if self.search == "find_ag":
            fixed_val = self.exp.default_b
        if self.search == "find_bg":
            fixed_val = self.exp.default_a

        axis0, axis1 = self.vary_axes

        # grab the ranges for the two free axes
        (u0, u1), (v0, v1) = self._ranges[axis0], self._ranges[axis1]

        # collect the (u,v) corner points that meet the tolerance
        uv_points: set[tuple[float, float]] = set()
        self._subdivide(u0, u1, v0, v1, self.fixed_axis, fixed_val, depth=0, collect=uv_points)

        for u, v in uv_points:
            a, b, g = self._make_args(u, v, self.fixed_axis, fixed_val)
            self.eval_f(a, b, g)

    def min_abg(self, mr, mt):
        """Find closest a, b, g closest to mr and mt."""
        minimum = float("inf")
        a_min = 0
        b_min = 0
        g_min = 0

        for a, b, g, ur1, ut1, uru, utu in self.cache:
            if np.abs(mr - ur1) + np.abs(mt - ut1) < minimum:
                a_min, b_min, g_min = a, b, g

        return a_min, b_min, g_min

    def is_stale(self, default_value):
        """Decide if current grid is still useful."""
        if self.default is None:
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
            ggg = np.full((N, N), self.exp.default_g)
            aaa, bbb = np.meshgrid(aa, bb)

        if self.search == "find_bg":
            aaa = np.full((N, N), self.exp.default_a)
            ggg, bbb = np.meshgrid(gg, bb)

        if self.search == "find_ag":
            bbb = np.full((N, N), self.exp.default_b)
            aaa, ggg = np.meshgrid(aa, gg)

        ur1 = np.zeros((N, N))
        ut1 = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                self.exp.sample.a = aaa[i, j]
                self.exp.sample.b = 10 ** bbb[i, j]
                self.exp.sample.g = ggg[i, j]
                ur1[i, j], ut1[i, j], _, _ = self.exp.sample.rt()

        return aaa, bbb, ggg, ur1, ut1

    def plot(self):
        """Plot the grid."""
        # turn the cache into a (N×7) array
        data = np.array(list(self.cache))
        if data.size == 0:
            raise RuntimeError("Cache is empty – run sample_adaptive() first")

        a, b, g, ur1, ut1, uru, utu = data.T

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
