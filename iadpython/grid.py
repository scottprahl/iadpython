"""Class for doing inverse adding-doubling calculations for a sample.

Example::

    >>> import iadpython

    >>> exp = iadpython.Experiment(0.5, 0.1, default_g=0.5)
    >>> grid = iadpython.Grid()
    >>> grid.calc(exp)
    >>> print(grid)
"""

import numpy as np

# Mirrors CWEB iad_type.w: @d MAX_ABS_G 0.999999
_MAX_ABS_G = 0.999999

# Log-b limits mirror CWEB iad_calc.w Fill_AB_Grid / Fill_BG_Grid
_MIN_LOG_B = -8.0   # exp(-8)  ≈ 0.000335
_MAX_LOG_B_AB = 8.0   # exp(+8)  ≈ 2981  (find_ab, find_ag)
_MAX_LOG_B_BG = 10.0  # exp(+10) ≈ 22026 (find_bg)


def _grid_stale_context(exp, search, default):
    """Capture the raw-RT state that determines whether a grid can be reused."""
    sample = exp.sample
    return {
        "search": search,
        "default": default,
        "m_u": exp.m_u,
        "slab_index": sample.n,
        "slab_cos_angle": sample.nu_0,
        "top_slide_index": sample.n_above,
        "bottom_slide_index": sample.n_below,
        "fixed_g": sample.g if search == "find_ab" else None,
        "fixed_b": sample.b if search == "find_ag" else None,
        "fixed_a": sample.a if search == "find_bg" else None,
        "default_ba": exp.default_ba if search == "find_bsg" else None,
        "default_bs": exp.default_bs if search == "find_bag" else None,
    }


def _grid_context_is_stale(context, exp, search, default):
    """Return whether the cached raw-RT grid is stale for the current experiment."""
    if context is None:
        return True

    if context["search"] != search:
        return True

    if getattr(exp, "num_measurements", 0) == 3 and exp.m_u != context["m_u"]:
        return True

    sample = exp.sample
    if sample.n != context["slab_index"]:
        return True
    if sample.nu_0 != context["slab_cos_angle"]:
        return True
    if sample.n_above != context["top_slide_index"]:
        return True
    if sample.n_below != context["bottom_slide_index"]:
        return True

    if search == "find_ab" and sample.g != context["fixed_g"]:
        return True
    if search == "find_ag" and sample.b != context["fixed_b"]:
        return True
    if search == "find_bg" and sample.a != context["fixed_a"]:
        return True
    if search == "find_bsg" and exp.default_ba != context["default_ba"]:
        return True
    if search == "find_bag" and exp.default_bs != context["default_bs"]:
        return True

    return context["default"] != default


def _nonlinear_a_coord(x):
    """Map a unit coordinate to Grid's nonlinear albedo spacing."""
    return 1.0 - (1.0 - x) ** 2 * (1.0 + 2.0 * x)


def _nonlinear_a(N):
    """Nonlinear albedo spacing denser at 0 and 1 (mirrors CWEB).

    a = 1 - (1-x)^2 (1+2x),  x = j/(N-1)

    Gives e.g. for N=11: 0.000, 0.028, 0.104, 0.216, 0.352, 0.500,
    0.648, 0.784, 0.896, 0.972, 1.000
    """
    x = np.linspace(0, 1, N)
    return _nonlinear_a_coord(x)


def _nonlinear_g_coord(x):
    """Map a unit coordinate to Grid's nonlinear anisotropy spacing."""
    return (1.0 - 2.0 * (1.0 - x) ** 2 * (1.0 + 2.0 * x)) * _MAX_ABS_G


def _nonlinear_g(N):
    """Nonlinear anisotropy spacing symmetric and denser near ±MAX_ABS_G (mirrors CWEB).

    g = (1 - 2*(1-x)^2*(1+2x)) * MAX_ABS_G,  x = i/(N-1)
    """
    x = np.linspace(0, 1, N)
    return _nonlinear_g_coord(x)


class Grid:
    """Class to track pre-calculated R & T values.

    There is a long story associated with these routines.  I spent a lot of time
    trying to find an empirical function to allow a guess at a starting value for
    the inversion routine.  Basically nothing worked very well.  There were
    too many special cases and what not.  So I decided to calculate a whole bunch
    of reflection and transmission values and keep their associated optical
    properties linked nearby.

    Spacing mirrors CWEB iad_calc.w:
    - ``b``: log-spaced from exp(-8) to exp(+8) [find_ab/find_ag] or exp(+10) [find_bg]
    - ``a``: nonlinear, denser near 0 and 1
    - ``g``: nonlinear, symmetric and denser near ±0.999999
    """

    def __init__(self, search=None, default=None, N=21):
        """Object initialization."""
        self.search = search
        self.default = default
        self._stale_context = None
        self.N = N
        self.a = np.zeros((N, N))
        self.b = np.zeros((N, N))
        self.g = np.zeros((N, N))
        self.ur1 = np.zeros((N, N))
        self.ut1 = np.zeros((N, N))
        self.uru = np.zeros((N, N))
        self.utu = np.zeros((N, N))

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

        s += matrix_as_string(self.a, "a")
        s += matrix_as_string(self.b, "b")
        s += matrix_as_string(self.g, "g")
        s += matrix_as_string(self.ur1, "ur1")
        s += matrix_as_string(self.ut1, "ut1")
        s += matrix_as_string(self.uru, "uru")
        s += matrix_as_string(self.utu, "utu")

        return s

    def calc(self, exp, default=None):
        """Precalculate a grid.

        Albedo uses nonlinear spacing denser near 0 and 1.
        Optical thickness uses log spacing over exp(-8) to exp(+8) [or exp(+10)
        for find_bg], matching CWEB Fill_AB_Grid / Fill_BG_Grid.
        Anisotropy uses nonlinear symmetric spacing denser near ±0.999999.
        """
        if default is not None:
            self.default = default

        N = self.N
        a = _nonlinear_a(N)
        g = _nonlinear_g(N)

        max_log_b = _MAX_LOG_B_BG if exp.search == "find_bg" else _MAX_LOG_B_AB
        b = np.exp(np.linspace(_MIN_LOG_B, max_log_b, N))

        self.search = exp.search
        self._stale_context = _grid_stale_context(exp, self.search, self.default)

        if self.search == "find_ab":
            self.g = np.full((N, N), self.default)
            self.a, self.b = np.meshgrid(a, b)

        if self.search == "find_ag":
            self.b = np.full((N, N), self.default)
            self.a, self.g = np.meshgrid(a, g)

        if self.search == "find_bg":
            self.a = np.full((N, N), self.default)
            self.b, self.g = np.meshgrid(b, g)

        for i in range(N):
            for j in range(N):
                exp.sample.a = self.a[i, j]
                exp.sample.b = self.b[i, j]
                exp.sample.g = self.g[i, j]
                self.ur1[i, j], self.ut1[i, j], self.uru[i, j], self.utu[i, j] = exp.sample.rt()

    def _distance(self, mr, mt, i, j, exp=None):
        """Return candidate distance in raw or corrected measurement space."""
        if exp is None:
            return abs(mr - self.ur1[i, j]) + abs(mt - self.ut1[i, j])

        _fit_r, _fit_t, delta = exp.measurement_distance_from_raw(
            self.ur1[i, j],
            self.ut1[i, j],
            self.uru[i, j],
            self.utu[i, j],
            include_lost=False,
            a=self.a[i, j],
            b=self.b[i, j],
            g=self.g[i, j],
        )
        return delta

    def min_abg(self, mr, mt, exp=None):
        """Find a, b, g closest to mr and mt.

        Locates the nearest grid cell in L1 measurement space, then checks the
        full 3×3 neighbourhood (mirrors CWEB Near_Grid_Points + Grid_ABG loop in
        iad_find.w).  Returns the single best candidate from that neighbourhood.
        """
        if self.ur1 is None:
            raise ValueError("Grid.calc(exp) must be called before Grid.min_abg")

        A = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                A[i, j] = self._distance(mr, mt, i, j, exp=exp)

        ii_flat = A.argmin()
        i_best = ii_flat // A.shape[1]
        j_best = ii_flat % A.shape[1]

        # Check 3×3 neighbourhood around the global best cell
        N = self.N
        best_dist = float("inf")
        a_best = self.a[i_best, j_best]
        b_best = self.b[i_best, j_best]
        g_best = self.g[i_best, j_best]

        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni, nj = i_best + di, j_best + dj
                if 0 <= ni < N and 0 <= nj < N:
                    d = self._distance(mr, mt, ni, nj, exp=exp)
                    if d < best_dist:
                        best_dist = d
                        a_best = self.a[ni, nj]
                        b_best = self.b[ni, nj]
                        g_best = self.g[ni, nj]

        return a_best, b_best, g_best

    def is_stale(self, exp, default, search=None):
        """Decide if current grid is still useful."""
        if self.default is None:
            return True
        if search is None:
            search = getattr(exp, "search", self.search)
        return _grid_context_is_stale(self._stale_context, exp, search, default)


def matrix_as_string(x, label=""):
    """Return matrix as a string."""
    if x is None:
        return ""

    n, m = x.shape

    ndashes = (80 - len(label) - 2) // 2
    s = "\n"
    s += "-" * ndashes + " " + label + " " + "-" * ndashes
    s += "\n[\n"
    for i in range(n):
        s += "["
        for j in range(m):
            s += "%6.3f, " % x[i, j]
        s += "], \n"
    s += "]\n"
    return s
