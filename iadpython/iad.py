"""Class for doing inverse adding-doubling calculations for a sample.

Example::

    >>> import iadpython as iad

    >>> exp = iad.Experiment(0.5,0.1,0.01)
    >>> a, b, g = exp.invert()
    >>> print("a = %7.3f" % a)
    >>> print("b = %7.3f" % b)
    >>> print("g = %7.3f" % g)

    >>> s = iad.Sample(0.5,0.1,0.01,n=1.4, n_top=1.5, n_bottom=1.5)
    >>> exp = iad.Experiment(s)
    >>> a, b, g = exp.invert()
    >>> print("a = %7.3f" % a)
    >>> print("b = %7.3f" % b)
    >>> print("g = %7.3f" % g)
"""

import sys
import copy
import numpy as np
import scipy.optimize
import iadpython as iad
from iadpython.mc_lost import run_mc_lost

G_BOUND_EPS = 1e-6
_TWO_PARAMETER_SEARCHES = {
    "find_ab",
    "find_ag",
    "find_bg",
    "find_ba",
    "find_bs",
    "find_bag",
    "find_bsg",
}

# ---------------------------------------------------------------------------
# Adaptive simplex helpers for MC re-inversion (used by _invert_scalar_with_mc)
# ---------------------------------------------------------------------------

_SIMPLEX_A_STEP = 1e-3
_SIMPLEX_B_REL_STEP = 1e-2
_SIMPLEX_G_STEP = 1e-3
_SIMPLEX_ADAPTIVE_SCALE = 1.0
_SIMPLEX_A_MIN_STEP = 1e-5
_SIMPLEX_B_MIN_STEP = 1e-4
_SIMPLEX_G_MIN_STEP = 1e-5
_MINIMIZER_BOUNDARY_NUDGE = 1e-4


def _nudge_bounded_start(search, a, b, g):
    """Move free bounded parameters just inside bounds before Nelder-Mead."""
    if search in ("find_ab", "find_ag"):
        a = float(np.clip(a, _MINIMIZER_BOUNDARY_NUDGE, 1.0 - _MINIMIZER_BOUNDARY_NUDGE))
    if search in ("find_ag", "find_bg"):
        g = float(
            np.clip(
                g,
                -1.0 + G_BOUND_EPS + _MINIMIZER_BOUNDARY_NUDGE,
                1.0 - G_BOUND_EPS - _MINIMIZER_BOUNDARY_NUDGE,
            )
        )
    return a, b, g


def _simplex_bounded_vertex(center, step, lower, upper):
    """Offset one simplex vertex from *center* by *step*, respecting bounds."""
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
    return center


def _simplex_geometry(search, hot_start):
    """Return (x0, lower, upper) for the 2-D Nelder-Mead search."""
    a, b, g = _nudge_bounded_start(search, *hot_start)
    if search == "find_ab":
        return (
            np.array([a, b], dtype=float),
            np.array([0.0, 0.0], dtype=float),
            np.array([1.0, np.inf], dtype=float),
        )
    if search == "find_ag":
        return (
            np.array([a, g], dtype=float),
            np.array([0.0, -1.0 + G_BOUND_EPS], dtype=float),
            np.array([1.0, 1.0 - G_BOUND_EPS], dtype=float),
        )
    if search == "find_bg":
        return (
            np.array([b, g], dtype=float),
            np.array([0.0, -1.0 + G_BOUND_EPS], dtype=float),
            np.array([np.inf, 1.0 - G_BOUND_EPS], dtype=float),
        )
    raise ValueError(f"adaptive simplex not applicable to search={search!r}")


def _adaptive_simplex_steps(search, hot_start, previous_delta):
    """Return per-axis simplex step sizes, shrunk by the previous MC stage movement.

    On the first MC iteration *previous_delta* is None and the fixed defaults
    are returned.  On subsequent iterations each axis step is
    ``adaptive_scale * |Δparam|`` clamped to ``[min_step, fixed_step]``.
    """
    _a, b, _g = hot_start
    if search == "find_ab":
        fixed = np.array(
            [_SIMPLEX_A_STEP, max(_SIMPLEX_B_MIN_STEP, _SIMPLEX_B_REL_STEP * max(abs(b), 1.0))], dtype=float
        )
        minimum = np.array([_SIMPLEX_A_MIN_STEP, _SIMPLEX_B_MIN_STEP], dtype=float)
    elif search == "find_ag":
        fixed = np.array([_SIMPLEX_A_STEP, _SIMPLEX_G_STEP], dtype=float)
        minimum = np.array([_SIMPLEX_A_MIN_STEP, _SIMPLEX_G_MIN_STEP], dtype=float)
    elif search == "find_bg":
        fixed = np.array(
            [max(_SIMPLEX_B_MIN_STEP, _SIMPLEX_B_REL_STEP * max(abs(b), 1.0)), _SIMPLEX_G_STEP], dtype=float
        )
        minimum = np.array([_SIMPLEX_B_MIN_STEP, _SIMPLEX_G_MIN_STEP], dtype=float)
    else:
        raise ValueError(f"adaptive simplex not applicable to search={search!r}")

    if previous_delta is None or not np.all(np.isfinite(previous_delta)):
        return fixed

    adaptive = _SIMPLEX_ADAPTIVE_SCALE * np.abs(previous_delta)
    return np.clip(adaptive, minimum, fixed)


def _build_adaptive_simplex(search, hot_start, previous_delta):
    """Build an N+1 × N initial simplex for a 2-D Nelder-Mead hot-start.

    The simplex is centred on *hot_start* with step sizes derived from the
    previous MC iteration's parameter change (or fixed defaults on the first
    iteration when *previous_delta* is None).
    """
    steps = _adaptive_simplex_steps(search, hot_start, previous_delta)
    x0, lower, upper = _simplex_geometry(search, hot_start)
    simplex = np.tile(x0, (len(x0) + 1, 1))
    for axis, (x0_ax, step, lo, hi) in enumerate(zip(x0, steps, lower, upper)):
        simplex[axis + 1, axis] = _simplex_bounded_vertex(x0_ax, step, lo, hi)
    return simplex


DEBUG_A_LITTLE = 1
DEBUG_GRID = 2
DEBUG_ITERATIONS = 4
DEBUG_LOST_LIGHT = 8
DEBUG_BEST_GUESS = 16
DEBUG_SEARCH = 32
DEBUG_GRID_CALC = 64
DEBUG_SPHERE_GAIN = 128
DEBUG_EVERY_CALC = 256
DEBUG_MC = 512
DEBUG_ANY = 0xFFFFFFFF
_LEGACY_DEBUG_GRID_SIZE = 101

_GRID_FILL_NAMES = {
    "find_ab": "AB",
    "find_ag": "AG",
    "find_bg": "BG",
    "find_bag": "BaG",
    "find_bsg": "BsG",
}

_SEARCH_DEBUG_NAMES = {
    "find_a": "FIND_A",
    "find_b": "FIND_B",
    "find_ab": "FIND_AB",
    "find_ag": "FIND_AG",
    "auto": "FIND_AUTO",
    "find_bg": "FIND_BG",
    "find_bag": "FIND_BaG",
    "find_bsg": "FIND_BsG",
    "find_ba": "FIND_Ba",
    "find_bs": "FIND_Bs",
    "find_g": "FIND_G",
    "find_b_no_absorption": "FIND_B_WITH_NO_ABSORPTION",
    "find_b_no_scattering": "FIND_B_WITH_NO_SCATTERING",
}

SEARCH_CODE_TO_NAME = {
    0: "find_a",
    1: "find_b",
    2: "find_ab",
    3: "find_ag",
    4: "auto",
    5: "find_bg",
    6: "find_bag",
    7: "find_bsg",
    8: "find_ba",
    9: "find_bs",
    10: "find_g",
    11: "find_b_no_absorption",
    12: "find_b_no_scattering",
}


def _point_value(value, index=None):
    """Return a scalar value for one row of array-valued input."""
    if value is None or np.isscalar(value) or np.ndim(value) == 0:
        return value
    return value[index]


def _as_scalar_float(value, label):
    """Return a Python float from a scalar-like value."""
    if value is None:
        raise ValueError(f"{label} cannot be None")

    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    if arr.size == 1:
        return float(arr.reshape(()))
    raise ValueError(f"{label} must be scalar, got shape {arr.shape}")


def _debug_values_differ(left, right):
    """Return True if two scalar-ish debug context values differ."""
    try:
        return bool(left != right)
    except ValueError:
        return bool(np.any(np.asarray(left) != np.asarray(right)))


def _grid_stale_debug_reason(grid, exp, search, default):
    """Return CWEB-style DEBUG_GRID reason text when a grid is stale."""
    if grid is None:
        return "GRID: Fill because NULL"

    context = getattr(grid, "_stale_context", None)
    if getattr(grid, "default", None) is None or context is None:
        return "GRID: Fill because not initialized"

    if context["search"] != search:
        return "GRID: Fill because search type changed"

    if getattr(exp, "num_measurements", 0) == 3 and _debug_values_differ(exp.m_u, context["m_u"]):
        return "GRID: Fill because unscattered light changed"

    sample = exp.sample
    if _debug_values_differ(sample.n, context["slab_index"]):
        return "GRID: Fill because slab refractive index changed"
    if _debug_values_differ(sample.nu_0, context["slab_cos_angle"]):
        return "GRID: Fill because light angle changed"
    if _debug_values_differ(sample.n_above, context["top_slide_index"]):
        return "GRID: Fill because top slide index changed"
    if _debug_values_differ(sample.n_below, context["bottom_slide_index"]):
        return "GRID: Fill because bottom slide index changed"

    if search == "find_ab" and _debug_values_differ(sample.g, context["fixed_g"]):
        return "GRID: Fill because anisotropy changed"
    if search == "find_ag" and _debug_values_differ(sample.b, context["fixed_b"]):
        return "GRID: Fill because optical depth changed"
    if search == "find_bsg" and _debug_values_differ(exp.default_ba, context["default_ba"]):
        return "GRID: Fill because mu_a changed"
    if search == "find_bag" and _debug_values_differ(exp.default_bs, context["default_bs"]):
        return "GRID: Fill because mu_s changed"
    if _debug_values_differ(context["default"], default):
        return "GRID: Fill because search type changed"

    return None


def _print_grid_fill_debug(search, default):
    """Emit CWEB-style DEBUG_GRID fill text for the current search."""
    name = _GRID_FILL_NAMES.get(search)
    if name is None:
        return

    if search == "find_ab":
        print(f"GRID: Filling AB grid (g={float(default):.5f})", file=sys.stderr)
        return

    print(f"GRID: Filling {name} grid", file=sys.stderr)
    if search == "find_ag":
        print("GRID: a        = %9.7f to %9.7f " % (0.0, 1.0), file=sys.stderr)
        print("GRID: b        = %9.5f " % float(default), file=sys.stderr)
        print("GRID: g  range = %9.6f to %9.6f " % (-0.999999, 0.999999), file=sys.stderr)
    elif search == "find_bg":
        print("GRID: a        = %9.7f " % float(default), file=sys.stderr)
        print("GRID: b  range = %9.5f to %9.3f " % (np.exp(-8), np.exp(10)), file=sys.stderr)
        print("GRID: g  range = %9.6f to %9.6f " % (-0.999999, 0.999999), file=sys.stderr)
    elif search == "find_bag":
        print("GRID:       bs = %9.5f" % float(default), file=sys.stderr)
        print("GRID: ba range = %9.6f to %9.3f " % (np.exp(-8), np.exp(10)), file=sys.stderr)
    elif search == "find_bsg":
        print("GRID:       ba = %9.5f" % float(default), file=sys.stderr)
        print("GRID: bs range = %9.6f to %9.3f " % (np.exp(-8), np.exp(10)), file=sys.stderr)


def _distance_for_abg(exp, a, b, g):
    """Return a CWEB-style corrected distance for one `(a,b,g)` guess."""
    original = (exp.sample.a, exp.sample.b, exp.sample.g)
    try:
        exp.sample.a = a
        exp.sample.b = b
        exp.sample.g = g
        ur1, ut1, uru, utu = exp.sample.rt()
        _m_r, _m_t, distance = exp.measurement_distance_from_raw(
            ur1,
            ut1,
            uru,
            utu,
            include_lost=True,
            a=a,
            b=b,
            g=g,
            debug_sphere=False,
        )
        return float(distance)
    finally:
        exp.sample.a, exp.sample.b, exp.sample.g = original


def _grid_guess(grid, exp, i, j):
    """Return one CWEB `Grid_ABG` candidate."""
    if i < 0 or i >= grid.N or j < 0 or j >= grid.N:
        return {"a": 0.5, "b": 0.5, "g": 0.5, "distance": 999.0}

    _m_r, _m_t, distance = exp.measurement_distance_from_raw(
        grid.ur1[i, j],
        grid.ut1[i, j],
        grid.uru[i, j],
        grid.utu[i, j],
        include_lost=True,
        a=grid.a[i, j],
        b=grid.b[i, j],
        g=grid.g[i, j],
        debug_sphere=False,
    )
    return {
        "a": float(grid.a[i, j]),
        "b": float(grid.b[i, j]),
        "g": float(grid.g[i, j]),
        "distance": float(distance),
    }


def _nearest_grid_index(grid, exp):
    """Return the nearest grid index using CWEB's corrected distance."""
    distances = np.zeros((grid.N, grid.N))
    for i in range(grid.N):
        for j in range(grid.N):
            distances[i, j] = _grid_guess(grid, exp, i, j)["distance"]
    flat_index = int(distances.argmin())
    return flat_index // distances.shape[1], flat_index % distances.shape[1]


def _ranked_grid_guesses(grid, exp, initial_abg):
    """Return sorted CWEB-style best-guess candidates."""
    a0, b0, g0 = initial_abg
    guesses = [
        {
            "a": float(a0),
            "b": float(b0),
            "g": float(g0),
            "distance": _distance_for_abg(exp, a0, b0, g0),
        }
    ]
    i_best, j_best = _nearest_grid_index(grid, exp)
    for di, dj in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
        guesses.append(_grid_guess(grid, exp, i_best + di, j_best + dj))
    return sorted(guesses, key=lambda guess: guess["distance"])


def _first_different_guess(guesses, axis, skip=None):
    """Return the first top guess whose axis differs from the best guess."""
    if skip is None:
        skip = set()
    for index in range(1, min(7, len(guesses))):
        if index in skip:
            continue
        if guesses[0][axis] != guesses[index][axis]:
            return index
    return min(7, len(guesses) - 1)


def _third_simplex_guess(guesses, second_index, axis):
    """Return the third CWEB simplex guess index for the secondary axis."""
    for index in range(1, min(7, len(guesses))):
        if index == second_index:
            continue
        if guesses[0][axis] != guesses[index][axis] or guesses[second_index][axis] != guesses[index][axis]:
            return index
    return min(7, len(guesses) - 1)


def _simplex_debug_guesses(search, guesses):
    """Return the three guesses CWEB prints as simplex start nodes."""
    if search == "find_ab":
        second = _first_different_guess(guesses, "a")
        third = _third_simplex_guess(guesses, second, "b")
    elif search == "find_ag":
        second = _first_different_guess(guesses, "a")
        third = _third_simplex_guess(guesses, second, "g")
    elif search == "find_bg":
        second = _first_different_guess(guesses, "b")
        third = _third_simplex_guess(guesses, second, "g")
    else:
        return []
    return [guesses[0], guesses[second], guesses[third]]


def _print_best_guess_debug(search, guesses):
    """Emit CWEB-style `-x 16` best grid point output."""
    print("BEST: GRID GUESSES", file=sys.stderr)
    print("BEST:  k      albedo          b          g   distance", file=sys.stderr)
    for index, guess in enumerate(guesses[:7]):
        print(
            "BEST:%3d  %10.5f %10.5f %10.5f %10.5f"
            % (index, guess["a"], guess["b"], guess["g"], guess["distance"]),
            file=sys.stderr,
        )

    simplex_guesses = _simplex_debug_guesses(search, guesses)
    if not simplex_guesses:
        return

    print("-----------------------------------------------------", file=sys.stderr)
    for index, guess in enumerate(simplex_guesses, start=1):
        print(
            "BEST: <%d> %10.5f %10.5f %10.5f %10.5f"
            % (index, guess["a"], guess["b"], guess["g"], guess["distance"]),
            file=sys.stderr,
        )
    print(file=sys.stderr)


def _print_grid_calc_fill_header():
    """Emit CWEB DEBUG_GRID_CALC grid-fill header."""
    print("+   i   j       a         b          g     |     M_R        grid  |     M_T        grid", file=sys.stderr)


def _grid_calc_distance(grid, exp, i, j, print_row=False):
    """Return a corrected grid distance and optionally emit its CWEB debug row."""
    fit_r, fit_t, distance = exp.measurement_distance_from_raw(
        grid.ur1[i, j],
        grid.ut1[i, j],
        grid.uru[i, j],
        grid.utu[i, j],
        include_lost=True,
        a=grid.a[i, j],
        b=grid.b[i, j],
        g=grid.g[i, j],
        debug_sphere=False,
    )
    if print_row:
        print(
            "g %3d %3d %10.5f %10.4f %10.5f | %10.5f %10.5f | %10.5f %10.5f |%10.3f"
            % (
                i,
                j,
                grid.a[i, j],
                grid.b[i, j],
                grid.g[i, j],
                exp._debug_measurement_value(exp.m_r),  # pylint: disable=protected-access
                fit_r,
                exp._debug_measurement_value(exp.m_t),  # pylint: disable=protected-access
                fit_t,
                distance,
            ),
            file=sys.stderr,
        )
    return float(distance)


def _debug_grid_calc_min_abg(grid, exp):
    """Find the nearest grid point while emitting CWEB `-x 64` rows."""
    print(
        "+   i   j       a         b          g     |     M_R        grid   |     M_T        grid   |  distance",
        file=sys.stderr,
    )
    distances = np.zeros((grid.N, grid.N))
    for i in range(grid.N):
        for j in range(grid.N):
            distances[i, j] = _grid_calc_distance(grid, exp, i, j, print_row=True)

    flat_index = int(distances.argmin())
    i_best = flat_index // distances.shape[1]
    j_best = flat_index % distances.shape[1]

    best_dist = float("inf")
    a_best = grid.a[i_best, j_best]
    b_best = grid.b[i_best, j_best]
    g_best = grid.g[i_best, j_best]
    for di, dj in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
        ni = i_best + di
        nj = j_best + dj
        if 0 <= ni < grid.N and 0 <= nj < grid.N:
            distance = _grid_calc_distance(grid, exp, ni, nj, print_row=True)
            if distance < best_dist:
                best_dist = distance
                a_best = grid.a[ni, nj]
                b_best = grid.b[ni, nj]
                g_best = grid.g[ni, nj]

    return a_best, b_best, g_best


class Experiment:
    """Container class for details of an experiment."""

    def __init__(
        self,
        r=None,
        t=None,
        u=None,
        sample=None,
        r_sphere=None,
        t_sphere=None,
        num_spheres=0,
        default_a=None,
        default_b=None,
        default_g=None,
    ):
        """Object initialization."""
        if sample is None:
            self.sample = iad.Sample()
        else:
            self.sample = sample

        self.r_sphere = r_sphere
        self.t_sphere = t_sphere
        self.num_spheres = num_spheres

        self.m_r = r
        self.m_t = t
        self.m_u = u
        self.lambda0 = None

        # these will have to be eventually supported
        self.d_beam = 1
        self.lambda0 = None

        self.flip_sample = False
        self.fraction_of_rc_in_mr = 1
        self.fraction_of_tc_in_mt = 1
        self.f_r = 0.0

        self.ur1_lost = 0
        self.uru_lost = 0
        self.ut1_lost = 0
        self.utu_lost = 0

        self.default_a = default_a
        self.default_b = default_b
        self.default_g = default_g
        self.default_ba = None
        self.default_bs = None
        self.default_mua = None
        self.default_mus = None

        self.search = "unknown"
        self.metric = 1
        self.tolerance = 0.0001
        self.MC_tolerance = 0.01
        self.final_distance = 1
        self.iterations = 1
        self.found = False
        self.error = 1
        self.num_measurements = 0
        self.grid = None
        self.method = "unknown"
        self.debug_level = 0
        self.verbosity = 2
        self.max_mc_iterations = 19
        self.n_photons = 100000
        self.mc_lost_path = None  # path to mc_lost binary; None disables MC iteration
        self.t_slide = 0.0  # physical thickness of cover glass slides [mm]
        self._mc_iterations = 0  # number of MC iterations used in last invert_rt() call
        self.use_adaptive_grid = None  # None = auto-select by search mode
        self.adaptive_grid_tol = 0.05
        self.adaptive_grid_max_depth = 4
        self.adaptive_grid_min_depth = 2
        self.grid_n = 21  # N for Grid when not using AGrid
        self.counter = 0
        self.include_measurements = True
        self.search_override = None
        self.search_code = 4
        self.first_pass_abg = None
        self._grid_evals = 0
        self._optimizer_evals = 0
        self._last_invert_status_valid = False
        self._debug_search_reported = False

    def __str__(self):
        """Return basic details as a string for printing."""
        s = "---------------- Sample ---------------\n"
        s += self.sample.__str__()
        s += "\n--------------- Spheres ---------------\n"
        if not np.isscalar(self.num_spheres):
            s += "number of spheres range (%s)\n" % iad.stringify("%d", self.num_spheres)
        elif self.num_spheres == 0:
            s += "No spheres used.\n"
        elif self.num_spheres == 1:
            s += "A single integrating sphere was used.\n"
        elif self.num_spheres == 2:
            s += "Double integrating spheres were used.\n"
        if self.r_sphere is not None:
            s += "Reflectance Sphere--------\n"
            s += self.r_sphere.__str__()
        if self.t_sphere is not None:
            s += "Transmission Sphere--------\n"
            s += self.t_sphere.__str__()

        if self.include_measurements:
            s += "\n------------- Measurements ------------\n"
            s += "   Reflection               = "
            s += iad.stringify("%.5f", self.m_r) + "\n"
            s += "   Transmission             = "
            s += iad.stringify("%.5f", self.m_t) + "\n"
            s += "   Unscattered Transmission = "
            s += iad.stringify("%.5f", self.m_u) + "\n"
        return s

    def check_measurements(self):
        """Make sure measurements are sane."""
        between = " Must be between 0 and 1."
        if (
            not (self.m_r is None or np.isscalar(self.m_r))
            or not (self.m_t is None or np.isscalar(self.m_t))
            or not (self.m_u is None or np.isscalar(self.m_u))
        ):
            raise ValueError("invert_scalar_rt() is only for scalar m_r, m_t, m_u")

        if self.m_r is not None:
            if self.m_r < 0 or self.m_r > 1:
                raise ValueError("Invalid refl. %.4f" % self.m_r + between)

        if self.m_t is not None:
            if self.m_t < 0 or self.m_t > 1:
                raise ValueError("Invalid trans. %.4f" % self.m_t + between)

        if self.m_u is not None:
            if self.m_u < 0 or self.m_u > 1:
                raise ValueError("Invalid unscattered trans. %.4f." % self.m_u + between)

    def useful_measurements(self):
        """Count the number of useful measurements."""
        self.num_measurements = 0
        if self.m_r is not None:
            self.num_measurements += 1
        if self.m_t is not None:
            self.num_measurements += 1
        if self.m_u is not None:
            self.num_measurements += 1

    def determine_one_parameter_search(self):
        """Establish proper search when only one measurement is available."""
        if self.default_a is not None:
            if np.isclose(self.default_a, 0.0):
                self.search = "find_b_no_scattering"
            elif np.isclose(self.default_a, 1.0):
                self.search = "find_b_no_absorption"
            elif self.default_b is not None:
                # both a and b fixed → only anisotropy is free
                self.search = "find_g"
            elif self.m_t == 0:
                self.search = "find_g"
            else:
                self.search = "find_b"
        elif self.default_b is not None:
            self.search = "find_a"
        elif self.default_bs is not None:
            self.search = "find_ba"
        elif self.default_ba is not None:
            self.search = "find_bs"
        elif self.m_r is not None:
            # only reflection given, no defaults → semi-infinite slab, find albedo
            self.search = "find_a"
        elif self.m_t == 0:
            self.search = "find_a"
        else:
            self.search = "find_b_no_absorption"

    @staticmethod
    def _debug_measurement_value(value):
        """Return CWEB's zero-filled scalar measurement value."""
        return 0.0 if value is None else float(value)

    def _debug_search_counts(self):
        """Return CWEB DEBUG_SEARCH measurement and constraint counts."""
        independent = 0
        if self._debug_measurement_value(self.m_r) > 0:
            independent += 1
        if self._debug_measurement_value(self.m_t) > 0:
            independent += 1
        if self._debug_measurement_value(self.m_u) > 0:
            independent += 1

        constraints = 0
        for value in (self.default_a, self.default_b, self.default_g, self.default_mua, self.default_mus):
            if value is not None:
                constraints += 1
        return independent, constraints

    def _debug_minimum_mr_mt(self):
        """Return CWEB-style minimum measured R/T estimate for DEBUG_SEARCH."""
        original_abg = (self.sample.a, self.sample.b, self.sample.g)
        try:
            self.sample.a = 0.0
            if self.default_b is not None:
                self.sample.b = self.default_b
            elif self._debug_measurement_value(self.m_u) > 0:
                self.sample.b = self.what_is_b()
            else:
                self.sample.b = np.inf
            self.sample.g = 0.0 if self.default_g is None else self.default_g
            ur1, ut1, uru, utu = self.sample.rt()
            return self.measured_rt_from_raw(ur1, ut1, uru, utu, include_lost=False)
        finally:
            self.sample.a, self.sample.b, self.sample.g = original_abg

    def _debug_estimate_rt(self):
        """Return CWEB `Estimate_RT` values for the search trace."""
        m_r = self._debug_measurement_value(self.m_r)
        m_t = self._debug_measurement_value(self.m_t)
        m_u = self._debug_measurement_value(self.m_u)
        rc, tc = self._debug_minimum_mr_mt()

        if self.fraction_of_rc_in_mr:
            rt = m_r
            rd = rt - self.fraction_of_rc_in_mr * rc
            if rd < 0:
                rd = 0.0
                rc = rt
        else:
            rd = m_r
            rt = rd + rc

        if m_u > 0:
            tc = m_u
        td = m_t - self.fraction_of_tc_in_mt * tc
        tt = td + tc
        return rt, tt, rd, rc, td, tc

    def _debug_search_decision_header(self):
        """Emit CWEB DEBUG_SEARCH input and estimate trace."""
        if not (self.debug_level & DEBUG_SEARCH) or self._debug_search_reported:
            return None

        independent, constraints = self._debug_search_counts()
        rt, tt, rd, rc, td, tc = self._debug_estimate_rt()
        print(f"SEARCH: starting with {independent:d} measurement(s)", file=sys.stderr)
        print(f"SEARCH:      and with {constraints:d} constraint(s)", file=sys.stderr)
        if self.default_a is not None:
            print("            albedo constrained", file=sys.stderr)
        if self.default_b is not None:
            print("            optical thickness constrained", file=sys.stderr)
        if self.default_g is not None:
            print("            anisotropy constrained", file=sys.stderr)
        if self.default_mua is not None:
            print("          mua constrained", file=sys.stderr)
        if self.default_mus is not None:
            print("          mus constrained", file=sys.stderr)
        print(
            "SEARCH: m_r = %8.5f m_t = %8.5f m_u = %8.5f"
            % (
                self._debug_measurement_value(self.m_r),
                self._debug_measurement_value(self.m_t),
                self._debug_measurement_value(self.m_u),
            ),
            file=sys.stderr,
        )
        print("SEARCH:  rt = %8.5f  rd = %8.5f  ru = %8.5f" % (rt, rd, rc), file=sys.stderr)
        print("SEARCH:  tt = %8.5f  td = %8.5f  tu = %8.5f" % (tt, td, tc), file=sys.stderr)

        if rd == 0 and independent >= 2:
            print("SEARCH: no information in rd", file=sys.stderr)
            independent -= 1
        if td == 0 and independent >= 2:
            print("SEARCH: no information in td", file=sys.stderr)
            independent -= 1
        return independent, constraints

    def _debug_search_decision_footer(self, context):
        """Emit CWEB DEBUG_SEARCH final search choice."""
        if context is None:
            return

        independent, constraints = context
        print(f"SEARCH: ending with {independent:d} measurement(s)", file=sys.stderr)
        print(f"SEARCH:    and with {constraints:d} constraint(s)", file=sys.stderr)
        search_name = _SEARCH_DEBUG_NAMES.get(self.search, self.search.upper())
        print(f"SEARCH: final choice for search = {search_name}", file=sys.stderr)
        self._debug_search_reported = True

    def reset_debug_search_reported(self):
        """Reset the CWEB DEBUG_SEARCH final search marker."""
        self._debug_search_reported = False

    def _debug_search_using(self):
        """Emit the CWEB DEBUG_SEARCH line for the selected inverse routine."""
        if not self.debug_level & DEBUG_SEARCH:
            return

        mu = float(self.sample.nu_0)
        if self.search == "find_ab":
            if self.default_g is not None:
                suffix = "  %7.3f (constrained g)" % float(self.default_g)
            else:
                suffix = "  %7.3f (default)" % 0.0
            print(f"SEARCH: Using U_Find_AB() mu={mu:4.2f}, g={suffix}", file=sys.stderr)
        elif self.search == "find_ag":
            if self._debug_measurement_value(self.m_u) > 0:
                suffix = " b= %6.3f  (M_U)" % float(self.what_is_b())
            elif self.default_b is not None:
                suffix = " b = %6.3f (constrained)" % float(self.default_b)
            else:
                suffix = " b = %6.3f (default)" % 1.0
            print(f"SEARCH: Using U_Find_AG() mu={mu:4.2f}, {suffix}", file=sys.stderr)
        elif self.search == "find_bg":
            message = "SEARCH: Using U_Find_BG() (mu=%6.4f)" % mu
            if self.default_a is not None:
                message += "  default_a = %8.5f" % float(self.default_a)
            print(message, file=sys.stderr)
        elif self.search == "find_a":
            message = "SEARCH: Using U_Find_A() (mu=%6.4f)" % mu
            if self.default_b is not None:
                message += "  default_b = %8.5f" % float(self.default_b)
            if self.default_g is not None:
                message += "  default_g = %8.5f" % float(self.default_g)
            print(message, file=sys.stderr)
        elif self.search in ("find_b", "find_b_no_absorption", "find_b_no_scattering"):
            message = "SEARCH: Using U_Find_B() (mu=%6.4f)" % mu
            if self.default_a is not None:
                message += "  default_a = %8.5f" % float(self.default_a)
            if self.default_g is not None:
                message += "  default_g = %8.5f" % float(self.default_g)
            print(message, file=sys.stderr)
        elif self.search == "find_g":
            message = "SEARCH: Using U_Find_G() (mu=%6.4f)" % mu
            if self.default_a is not None:
                message += "  default_a = %8.5f" % float(self.default_a)
            if self.default_b is not None:
                message += "  default_b = %8.5f" % float(self.default_b)
            print(message, file=sys.stderr)
        elif self.search == "find_bs":
            message = "SEARCH: Using U_Find_Bs() (mu=%6.4f)" % mu
            if self.default_ba is not None:
                message += "  default_ba = %8.5f" % float(self.default_ba)
            if self.default_g is not None:
                message += "  default_g = %8.5f" % float(self.default_g)
            print(message, file=sys.stderr)
        elif self.search == "find_ba":
            message = "SEARCH: Using U_Find_Bs() (mu=%6.4f)" % mu
            if self.default_bs is not None:
                message += "  default_bs = %8.5f" % float(self.default_bs)
            if self.default_g is not None:
                message += "  default_g = %8.5f" % float(self.default_g)
            print(message, file=sys.stderr)
        elif self.search == "find_bsg":
            message = "SEARCH: Using U_Find_BsG() (mu=%6.4f)" % mu
            if self.default_ba is not None:
                message += "  default_ba = %8.5f" % float(self.default_ba)
            print(message, file=sys.stderr)

    def determine_two_parameter_search(self):
        """Establish proper search when 2 or 3 measurements are available."""
        if self.default_a is not None:
            if np.isclose(self.default_a, 0.0) or self.default_g is not None:
                self.search = "find_b"
            elif self.default_b is not None:
                self.search = "find_g"
            else:
                self.search = "find_bg"
        elif self.default_b is not None:
            if self.default_g is not None:
                self.search = "find_a"
            else:
                self.search = "find_ag"
        elif self.default_g is not None:
            self.search = "find_ab"
        elif self.default_ba is not None:
            if self.default_g is not None:
                self.search = "find_bs"
            else:
                self.search = "find_bsg"
        elif self.default_bs is not None:
            if self.default_g is not None:
                self.search = "find_ba"
            else:
                self.search = "find_bag"
        elif self.m_r is None and self.m_t is not None and self.m_t > 0:
            self.search = "find_b_no_scattering"
        else:
            if self.m_r is not None and self.m_t is not None and self.m_u is not None and self.m_u > 0:
                self.search = "find_ag"
            else:
                self.search = "find_ab"

    def determine_search(self):
        """Determine type of search to do."""
        if self.search_override not in (None, "auto"):
            self.search = self.search_override
            self.search_code = next(
                (code for code, name in SEARCH_CODE_TO_NAME.items() if name == self.search),
                4,
            )
            return

        if self.num_measurements == 0:
            self.search = "unknown"
            self.search_code = -1
            return

        debug_context = self._debug_search_decision_header()
        if self.num_measurements == 1:
            self.determine_one_parameter_search()
        else:
            self.determine_two_parameter_search()

        self.search_code = next(
            (code for code, name in SEARCH_CODE_TO_NAME.items() if name == self.search),
            4,
        )
        self._debug_search_decision_footer(debug_context)

    def _debug(self, mask, message):
        """Emit debug output matching the enabled bitmask."""
        if self.debug_level & mask:
            print(message, file=sys.stderr)

    def _debug_search_header(self):
        """Emit the CWEB-style inverse-search header for ``-x 4``."""
        if not self.debug_level & DEBUG_ITERATIONS:
            return
        distance_name = "relative distance" if self.metric == 0 else "absolute distance"
        print("---------------- Beginning New Search -----------------", file=sys.stderr)
        print(
            "                a         b          g   |"
            "     M_R        calc   |     M_T        calc   | "
            f"{distance_name}",
            file=sys.stderr,
        )

    def _debug_iteration_row(self, m_r_calc=None, m_t_calc=None, distance=None):
        """Emit one CWEB-style objective row for ``-x 4``."""
        if not self.debug_level & DEBUG_ITERATIONS:
            return

        if m_r_calc is None or m_t_calc is None or distance is None:
            m_r_calc, m_t_calc = self.measured_rt()
            distance = 0.0
            if self.m_r is not None:
                distance += abs(float(m_r_calc) - float(self.m_r))
            if self.m_t is not None:
                distance += abs(float(m_t_calc) - float(self.m_t))

        measured_r = 0.0 if self.m_r is None else float(self.m_r)
        measured_t = 0.0 if self.m_t is None else float(self.m_t)
        print(
            f"        {float(self.sample.a):10.5f} {float(self.sample.b):10.4f}"
            f" {float(self.sample.g):10.5f} |"
            f" {measured_r:10.5f} {float(m_r_calc):10.5f} |"
            f" {measured_t:10.5f} {float(m_t_calc):10.5f} |"
            f"{float(distance):10.3f}",
            file=sys.stderr,
        )

    def _debug_lost_light_header(self):
        """Emit the CWEB-style lost-light table header for ``-x 8``."""
        if not self.debug_level & DEBUG_LOST_LIGHT:
            return
        print(
            "#      | Meas      M_R  | Meas      M_T  |  calc   calc   calc  |"
            "  Lost   Lost   Lost   Lost  | MC   IAD  Error",
            file=sys.stderr,
        )
        print(
            "# wave |  M_R      fit  |  M_T      fit  |  mu_a   mu_s'   g    |  "
            " UR1    URU    UT1    UTU  |  #    #   Type",
            file=sys.stderr,
        )
        print(
            "#  nm  |  ---      ---  |  ---      ---  |  1/mm   1/mm    ---  |"
            "   ---    ---    ---    ---  | ---  ---  ---",
            file=sys.stderr,
        )
        print("#" + "-" * 113, file=sys.stderr)

    def _debug_lost_light_row(self, line=0):
        """Emit one CWEB-style lost-light result row for ``-x 8``."""
        if not self.debug_level & DEBUG_LOST_LIGHT:
            return

        m_r_fit, m_t_fit = self.measured_rt()
        wavelength = self.lambda0
        label = f"{float(wavelength):6.1f}" if wavelength not in (None, 0) else f"{int(line):6d}"
        mu_a = min(float(self.sample.mu_a()), 199.9999)
        mu_sp = min(float(self.sample.mu_sp()), 999.9999)
        m_r = 0.0 if self.m_r is None else float(self.m_r)
        m_t = 0.0 if self.m_t is None else float(self.m_t)
        status = "*" if self.found else "+"
        print(
            f"{label}   {m_r:6.4f} {float(m_r_fit): 6.4f} | "
            f"{m_t:6.4f} {float(m_t_fit): 6.4f} | "
            f"{mu_a:6.3f} {mu_sp:6.3f} {float(self.sample.g):6.3f} |"
            f" {self.ur1_lost:6.4f} {self.uru_lost:6.4f} "
            f"{self.ut1_lost:6.4f} {self.utu_lost:6.4f} | "
            f"{self._mc_iterations:2d}  {self.iterations:3d}    {status} ",
            file=sys.stderr,
        )

    def _set_sample_from_ba_bs(self, ba, bs):
        """Set `(a, b)` from absorption/scattering optical depths."""
        ba = max(float(ba), 0.0)
        bs = max(float(bs), 0.0)
        total = ba + bs
        self.sample.b = total
        self.sample.a = 0.0 if np.isclose(total, 0.0) else bs / total

    def _positive_guess(self, value, floor=1e-6):
        """Return a usable non-negative optimizer starting point."""
        if value is None:
            return floor
        if np.isscalar(value):
            if np.isfinite(value):
                return max(float(value), floor)
            return 1.0
        arr = np.asarray(value)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return 1.0
        return max(float(np.max(finite)), floor)

    def point_at(self, index=None):
        """Return a scalar experiment for one row of vector-valued input."""
        point = copy.deepcopy(self)

        exp_attrs = [
            "lambda0",
            "m_r",
            "m_t",
            "m_u",
            "d_beam",
            "fraction_of_rc_in_mr",
            "fraction_of_tc_in_mt",
            "default_a",
            "default_b",
            "default_g",
            "default_mua",
            "default_mus",
            "default_ba",
            "default_bs",
            "num_spheres",
            "rstd_r",
            "error",
            "f_r",
        ]
        for attr in exp_attrs:
            setattr(point, attr, _point_value(getattr(self, attr, None), index))

        sample_attrs = [
            "a",
            "b",
            "g",
            "d",
            "n",
            "n_above",
            "n_below",
            "d_above",
            "d_below",
            "b_above",
            "b_below",
        ]
        for attr in sample_attrs:
            setattr(point.sample, attr, _point_value(getattr(self.sample, attr), index))

        for sphere_name in ["r_sphere", "t_sphere"]:
            sphere = getattr(point, sphere_name)
            source = getattr(self, sphere_name)
            if sphere is None or source is None:
                continue
            sphere.r_wall = _point_value(source.r_wall, index)
            sphere.r_std = _point_value(source.r_std, index)

        row_quad_pts = getattr(self, "_row_quad_pts", None)
        if row_quad_pts is not None:
            point.sample.quad_pts = int(_point_value(row_quad_pts, index))

        point.grid = None
        point.include_measurements = False
        point.first_pass_abg = None
        point.reset_debug_search_reported()
        return point

    def _debug_single_sphere_reflection(self, ur1, uru, ru, m_r, include_lost, stream=None):
        """Emit CWEB DEBUG_SPHERE_GAIN output for a reflectance sphere."""
        if stream is None:
            stream = sys.stderr

        sphere = self.r_sphere
        ur1 = _as_scalar_float(ur1, "UR1")
        uru = _as_scalar_float(uru, "URU")
        ru = _as_scalar_float(ru, "Ru")
        m_r = _as_scalar_float(m_r, "M_R")
        ur1_lost = self.ur1_lost if include_lost else 0.0
        uru_lost = self.uru_lost if include_lost else 0.0
        ur1_calc = max(ur1 - ru - ur1_lost, 0.0)
        uru_calc = max(uru - uru_lost, 0.0)

        r_first = 1.0
        if sphere.baffle:
            r_first = sphere.r_wall * (1 - sphere.third.a)

        gain_0 = _as_scalar_float(sphere.gain(0.0, 0.0), "G_0")
        gain = _as_scalar_float(sphere.gain(uru_calc, 0.0), "G")
        gain_cal = _as_scalar_float(sphere.gain(sphere.r_std, 0.0), "G_cal")
        p_cal = gain_cal * (sphere.r_std * (1 - self.f_r) + self.f_r * sphere.r_wall)
        p_0 = gain_0 * (self.f_r * sphere.r_wall)
        p_ss = r_first * (ur1_calc * (1 - self.f_r) + self.f_r * sphere.r_wall)
        p_su = sphere.r_wall * (1 - self.f_r) * self.fraction_of_rc_in_mr * ru
        p = gain * (p_ss + p_su)

        show_no_lost = include_lost and self.uru_lost > 0
        if show_no_lost:
            gain_none = _as_scalar_float(sphere.gain(uru, 0.0), "G_no_lost")
            p_none = gain_none * (p_ss + p_su)
            m_none = sphere.r_std * (p_none - p_0) / (p_cal - p_0)

        print("SPHERE: REFLECTION", file=stream)
        print("SPHERE:       baffle = %d" % int(bool(sphere.baffle)), file=stream)
        print("SPHERE:       R_u collected = %5.1f%%" % (self.fraction_of_rc_in_mr * 100), file=stream)
        print("SPHERE:       hits sphere wall first = %5.1f%%" % (self.f_r * 100), file=stream)
        print("SPHERE:       UR1 = %7.3f   UR1_calc = %7.3f" % (ur1, ur1_calc), file=stream)
        print("SPHERE:       URU = %7.3f   URU_calc = %7.3f" % (uru, uru_calc), file=stream)
        print("SPHERE:       R_u = %7.3f" % ru, file=stream)
        print("SPHERE:       G_0 = %7.3f        P_0 = %7.3f" % (gain_0, p_0), file=stream)
        print("SPHERE:         G = %7.3f          P = %7.3f" % (gain, p), file=stream)
        if show_no_lost:
            print("SPHERE: G_no_lost = %7.3f  P_no_lost = %7.3f" % (gain_none, p_none), file=stream)
        print("SPHERE:     G_cal = %7.3f      P_cal = %7.3f" % (gain_cal, p_cal), file=stream)
        if show_no_lost:
            print("SPHERE: M_no_lost = %7.3f" % m_none, file=stream)
        print("SPHERE:       M_R = %7.3f" % m_r, file=stream)

    def _debug_single_sphere_transmission(self, ur1, ut1, uru, ru, tu, m_t, include_lost, stream=None):
        """Emit CWEB DEBUG_SPHERE_GAIN output for a transmission sphere."""
        if stream is None:
            stream = sys.stderr

        sphere = self.t_sphere
        ur1 = _as_scalar_float(ur1, "UR1")
        ut1 = _as_scalar_float(ut1, "UT1")
        uru = _as_scalar_float(uru, "URU")
        ru = _as_scalar_float(ru, "Ru")
        tu = _as_scalar_float(tu, "Tu")
        m_t = _as_scalar_float(m_t, "M_T")
        ur1_lost = self.ur1_lost if include_lost else 0.0
        uru_lost = self.uru_lost if include_lost else 0.0
        ut1_lost = self.ut1_lost if include_lost else 0.0
        ur1_calc = max(ur1 - ru - ur1_lost, 0.0)
        uru_calc = max(uru - uru_lost, 0.0)
        ut1_calc = max(ut1 - tu - ut1_lost, 0.0)

        if sphere.third.a == 0:
            r_cal = sphere.r_wall
            r_third = sphere.r_wall
        elif self.fraction_of_tc_in_mt > 0:
            r_cal = sphere.r_std
            r_third = sphere.r_std
        else:
            r_cal = sphere.r_std
            r_third = 0.0

        r_first = 1.0
        if sphere.baffle:
            r_first = sphere.r_wall * (1 - sphere.third.a) + r_third * sphere.third.a

        gain = _as_scalar_float(sphere.gain(uru_calc, r_third), "G")
        gain_cal = _as_scalar_float(sphere.gain(0.0, r_cal), "G_cal")
        p_su = r_third * tu * self.fraction_of_tc_in_mt
        p_ss = r_first * ut1_calc

        show_no_lost = include_lost and self.uru_lost > 0
        if show_no_lost:
            gain_none = _as_scalar_float(sphere.gain(uru, 0.0), "G_no_lost")
            p_none = p_ss + p_su
            m_none = (p_su + p_ss) * gain_none / gain_cal

        print("SPHERE: TRANSMISSION", file=stream)
        print("SPHERE:       baffle = %d" % int(bool(sphere.baffle)), file=stream)
        print("SPHERE:       T_u collected = %5.1f%%" % (self.fraction_of_tc_in_mt * 100), file=stream)
        print("SPHERE:       UR1 = %7.3f   UR1_calc = %7.3f" % (ur1, ur1_calc), file=stream)
        print("SPHERE:       URU = %7.3f   URU_calc = %7.3f" % (uru, uru_calc), file=stream)
        print("SPHERE:       UT1 = %7.3f   UT1_calc = %7.3f" % (ut1, ut1_calc), file=stream)
        print("SPHERE:       T_u = %7.3f" % tu, file=stream)
        print("SPHERE:         G = %7.3f          P = %7.3f" % (gain, p_su + p_ss), file=stream)
        if show_no_lost:
            print("SPHERE: G_no_lost = %7.3f  P_no_lost = %7.3f" % (gain_none, p_none), file=stream)
        print("SPHERE:     G_cal = %7.3f      P_cal = %7.3f" % (gain_cal, 1.0), file=stream)
        print("SPHERE:   r_third = %7.3f      r_cal = %7.3f" % (r_third, r_cal), file=stream)
        print("SPHERE:   r_first = %7.3f" % r_first, file=stream)
        print("SPHERE:       Psu = %7.3f        Pss = %7.3f" % (p_su, p_ss), file=stream)
        if show_no_lost:
            print("SPHERE: M_no_lost = %7.3f" % m_none, file=stream)
        print("SPHERE:       M_T = %7.3f" % m_t, file=stream)
        print(file=stream)

    def measured_rt_from_raw(self, ur1, ut1, uru, utu, include_lost=True, a=None, b=None, g=None, debug_sphere=True):
        """Convert raw RT values into the measured `M_R/M_T` for this experiment."""
        s = self.sample
        original = None

        if a is not None or b is not None or g is not None:
            original = (s.a, s.b, s.g)
            if a is not None:
                s.a = a
            if b is not None:
                s.b = b
            if g is not None:
                s.g = g

        try:
            nu_inside = iad.cos_snell(1, s.nu_0, s.n)
            r_u, t_u = iad.specular_rt(s.n_above, s.n, s.n_below, s.b, nu_inside)

            if self.num_spheres == 0 or not include_lost:
                ur1_actual = ur1
                ut1_actual = ut1
                uru_actual = uru
                utu_actual = utu
            else:
                ur1_actual = ur1 - self.ur1_lost
                ut1_actual = ut1 - self.ut1_lost
                uru_actual = uru - self.uru_lost
                utu_actual = utu - self.utu_lost

            if self.debug_level & DEBUG_EVERY_CALC:
                print(
                    "measured_rt corrections: "
                    f"ur1={ur1:.5f}->{ur1_actual:.5f} "
                    f"ut1={ut1:.5f}->{ut1_actual:.5f} "
                    f"uru={uru:.5f}->{uru_actual:.5f} "
                    f"utu={utu:.5f}->{utu_actual:.5f}",
                    file=sys.stderr,
                )

            m_r = ur1_actual - (1.0 - self.fraction_of_rc_in_mr) * r_u
            m_t = ut1_actual - (1.0 - self.fraction_of_tc_in_mt) * t_u

            if self.num_spheres == 2:
                if self.r_sphere is None or self.t_sphere is None:
                    raise ValueError("Double sphere mode requires both reflection and transmission spheres.")

                lost_ur1 = self.ur1_lost if include_lost else 0.0
                lost_ut1 = self.ut1_lost if include_lost else 0.0
                uru_calc = max(uru_actual, 0.0)
                utu_calc = max(utu_actual, 0.0)
                ur1_calc = max(ur1 - (1.0 - self.fraction_of_rc_in_mr) * r_u - lost_ur1, 0.0)
                ut1_calc = max(ut1 - (1.0 - self.fraction_of_tc_in_mt) * t_u - lost_ut1, 0.0)

                d_spheres = iad.DoubleSphere(self.r_sphere, self.t_sphere)
                d_spheres.f_r = self.f_r
                m_r, m_t = d_spheres.measured_rt(ur1_calc, uru_calc, ut1_calc, utu_calc)
                return m_r, m_t

            if self.num_spheres == 1:
                if self.method in ("comparison", 1):
                    return m_r, m_t

                if self.r_sphere is not None:
                    f_u = self.fraction_of_rc_in_mr
                    m_r = self.r_sphere.MR(ur1_actual, uru_actual, R_u=r_u, f_u=f_u, f_w=self.f_r)
                    if debug_sphere and self.debug_level & DEBUG_SPHERE_GAIN:
                        self._debug_single_sphere_reflection(ur1, uru, r_u, m_r, include_lost)

                if self.t_sphere is not None:
                    m_t = self.t_sphere.MT(ut1_actual, uru_actual, T_u=t_u, f_u=self.fraction_of_tc_in_mt)
                    if debug_sphere and self.debug_level & DEBUG_SPHERE_GAIN:
                        self._debug_single_sphere_transmission(ur1, ut1, uru, r_u, t_u, m_t, include_lost)

            return m_r, m_t
        finally:
            if original is not None:
                s.a, s.b, s.g = original

    def measurement_distance(self, m_r, m_t):
        """Return scalar L1 distance between calculated and measured `M_R/M_T`.

        CWEB's two-parameter searches always compare transmission and, unless
        albedo is fixed at zero, reflectance.  Missing file columns therefore
        behave like zero-valued measurements once the search needs that axis.
        """
        if self.search in _TWO_PARAMETER_SEARCHES:
            measured_t = 0.0 if self.m_t is None else _as_scalar_float(self.m_t, "measured M_T")
            delta = abs(_as_scalar_float(m_t, "calculated M_T") - measured_t)
            if self.default_a is None or not np.isclose(self.default_a, 0.0):
                measured_r = 0.0 if self.m_r is None else _as_scalar_float(self.m_r, "measured M_R")
                delta += abs(_as_scalar_float(m_r, "calculated M_R") - measured_r)
            return delta

        delta = 0.0
        if self.m_r is not None:
            delta += abs(_as_scalar_float(m_r, "calculated M_R") - _as_scalar_float(self.m_r, "measured M_R"))
        if self.m_t is not None:
            delta += abs(_as_scalar_float(m_t, "calculated M_T") - _as_scalar_float(self.m_t, "measured M_T"))
        return delta

    def _debug_no_sphere_rt_from_raw(self, ur1, ut1, include_lost):
        """Return direct `M_R/M_T` with optional lost-light terms."""
        s = self.sample
        nu_inside = iad.cos_snell(1, s.nu_0, s.n)
        r_u, t_u = iad.specular_rt(s.n_above, s.n, s.n_below, s.b, nu_inside)
        ur1_lost = self.ur1_lost if include_lost else 0.0
        ut1_lost = self.ut1_lost if include_lost else 0.0
        m_r = ur1 - (1.0 - self.fraction_of_rc_in_mr) * r_u - ur1_lost
        m_t = ut1 - (1.0 - self.fraction_of_tc_in_mt) * t_u - ut1_lost
        return max(m_r, 0.0), max(m_t, 0.0)

    def _debug_sphere_rt_from_raw(self, ur1, ut1, uru, utu, include_lost):
        """Return sphere-corrected `M_R/M_T` without printing nested debug rows."""
        debug_level = self.debug_level
        self.debug_level &= ~(DEBUG_EVERY_CALC | DEBUG_SPHERE_GAIN)
        try:
            return self.measured_rt_from_raw(ur1, ut1, uru, utu, include_lost=include_lost)
        finally:
            self.debug_level = debug_level

    def debug_a_little_summary(self, stream=None):
        """Emit the compact CWEB `DEBUG_A_LITTLE` final summary."""
        if stream is None:
            stream = sys.stderr

        ur1, ut1, uru, utu = self.sample.rt()
        mr_no_corr, mt_no_corr = self._debug_no_sphere_rt_from_raw(ur1, ut1, include_lost=False)
        mr_mc, mt_mc = self._debug_no_sphere_rt_from_raw(ur1, ut1, include_lost=True)
        mr_sphere, mt_sphere = self._debug_sphere_rt_from_raw(ur1, ut1, uru, utu, include_lost=False)
        mr_sphere_mc, mt_sphere_mc = self._debug_sphere_rt_from_raw(ur1, ut1, uru, utu, include_lost=True)

        measured_r = 0.0 if self.m_r is None else float(self.m_r)
        measured_t = 0.0 if self.m_t is None else float(self.m_t)
        mu_a = 0.0 if self.sample.mu_a() is None else float(self.sample.mu_a())
        mu_s = 0.0 if self.sample.mu_s() is None else float(self.sample.mu_s())
        mu_sp = 0.0 if self.sample.mu_sp() is None else float(self.sample.mu_sp())

        print(
            f"AD iterations= {int(self.iterations):3d}   MC iterations={int(self._mc_iterations):3d}"
            f"            a={float(self.sample.a):6.4f}, b={float(self.sample.b):.4f},"
            f" g={float(self.sample.g):.4f}, mu_a={mu_a:6.4f}, mu_s={mu_s:6.4f}, mus'={mu_sp:6.4f}",
            file=stream,
        )
        print(
            f"    M_R loss           {float(self.ur1_lost):8.5f}  M_T loss           {float(self.ut1_lost):8.5f}",
            end="",
            file=stream,
        )
        if self._mc_iterations == 0:
            print(" (none yet)", file=stream)
        else:
            print(file=stream)
        print(
            f"    M_R no corrections {float(mr_no_corr):8.5f}  M_T no corrections {float(mt_no_corr):8.5f}",
            file=stream,
        )
        print(
            f"    M_R + sphere       {float(mr_sphere):8.5f}  M_T + sphere       {float(mt_sphere):8.5f}",
            file=stream,
        )
        print(f"    M_R + mc           {float(mr_mc):8.5f}  M_T + mc           {float(mt_mc):8.5f}", file=stream)
        print(
            f"    M_R + sphere + mc  {float(mr_sphere_mc):8.5f}  M_T + sphere + mc  {float(mt_sphere_mc):8.5f}",
            file=stream,
        )
        print(f"    M_R measured       {measured_r:8.5f}  M_T measured       {measured_t:8.5f}", file=stream)
        print(f"Final distance {float(self.final_distance):8.5f}\n", file=stream)

    def measurement_distance_from_raw(
        self, ur1, ut1, uru, utu, include_lost=True, a=None, b=None, g=None, debug_sphere=True
    ):
        """Return corrected `M_R/M_T` and scalar L1 distance to the measurements."""
        m_r, m_t = self.measured_rt_from_raw(
            ur1,
            ut1,
            uru,
            utu,
            include_lost=include_lost,
            a=a,
            b=b,
            g=g,
            debug_sphere=debug_sphere,
        )

        delta = self.measurement_distance(m_r, m_t)
        return m_r, m_t, delta

    def invert_scalar_rt(self, hot_start=None, initial_simplex=None):
        """Find a,b,g for a single experimental measurement.

        This routine assumes that `m_r`, `m_t`, and `m_u` are scalars.

        Args:
            hot_start: optional (a, b, g) tuple to use as the optimizer
                starting point for two-parameter searches, bypassing the grid
                lookup.  Pass the previous inversion result here when
                re-inverting after a lost-light update so the grid is not
                rebuilt and the previous solution seeds the optimizer.
            initial_simplex: optional (N+1) × N numpy array passed directly to
                SciPy's Nelder-Mead ``options['initial_simplex']``.  Only used
                for find_ab / find_ag / find_bg searches.  When None the
                default SciPy simplex is used.

        Returns:
            - `a` is the single scattering albedo of the slab
            - `b` is the optical thickness of the slab
            - `g` is the anisotropy of single scattering
        """
        self.check_measurements()
        self.useful_measurements()
        self.determine_search()

        if self.m_r is None and self.m_t is None and self.m_u is None:
            return None, None, None

        self.sample.rt_evals = 0
        self._last_invert_status_valid = False

        # assign default values
        self.sample.a = 0 if self.default_a is None else self.default_a
        self.sample.b = self.what_is_b() if self.default_b is None else self.default_b
        self.sample.g = 0 if self.default_g is None else self.default_g

        if self.default_bs is not None and self.default_ba is None:
            guess_ba = max(self._positive_guess(self.sample.b) - self.default_bs, 0.0)
            self._set_sample_from_ba_bs(guess_ba, self.default_bs)

        if self.default_ba is not None and self.default_bs is None:
            guess_bs = max(self._positive_guess(self.sample.b) - self.default_ba, 0.0)
            self._set_sample_from_ba_bs(self.default_ba, guess_bs)

        self._debug_search_using()

        #        print('search is', self.search)
        #        print('     a = ', self.sample.a)
        #        print('     b = ', self.sample.b)
        #        print('     g = ', self.sample.g)

        if self.debug_level & DEBUG_SPHERE_GAIN and not self.debug_level & DEBUG_ITERATIONS:
            self.measured_rt()
            self.measured_rt()

        self._debug_search_header()
        self._debug_iteration_row()

        result = None
        if self.search == "find_a":
            result = scipy.optimize.minimize_scalar(afun, args=(self), bounds=(0, 1), method="bounded")

        if self.search == "find_b":
            result = scipy.optimize.minimize_scalar(bfun, args=(self), method="brent")

        if self.search == "find_g":
            result = scipy.optimize.minimize_scalar(
                gfun,
                args=(self),
                bounds=(-1 + G_BOUND_EPS, 1 - G_BOUND_EPS),
                method="bounded",
            )

        if self.search in ["find_ab", "find_ag", "find_bg"]:
            debug_best_guess = bool(self.debug_level & DEBUG_BEST_GUESS)
            debug_grid_calc = bool(self.debug_level & DEBUG_GRID_CALC)
            debug_sphere_gain = bool(self.debug_level & DEBUG_SPHERE_GAIN)
            use_legacy_debug_grid = debug_best_guess or debug_grid_calc or debug_sphere_gain

            if hot_start is not None and not use_legacy_debug_grid:
                # Bypass the grid: use the caller-supplied starting point
                # directly.  The grid is not rebuilt; the raw-RT cache is
                # still valid because lost-light values do not affect it.
                a, b, g = hot_start
                self._grid_evals = 0
            else:
                if hot_start is not None:
                    self.sample.a, self.sample.b, self.sample.g = hot_start

                # Determine grid type for this search mode.
                # use_adaptive_grid=True : always AGrid with user-tunable tol/depth.
                # use_adaptive_grid=False: always Grid(N=grid_n).
                # use_adaptive_grid=None (default): auto-select.
                agrid_tol = self.adaptive_grid_tol
                agrid_max_depth = self.adaptive_grid_max_depth
                agrid_min_depth = getattr(self, "adaptive_grid_min_depth", 2)
                grid_n = _LEGACY_DEBUG_GRID_SIZE if use_legacy_debug_grid else self.grid_n
                if use_legacy_debug_grid:
                    want_agrid = False
                elif self.use_adaptive_grid is None:
                    want_agrid = True
                elif self.use_adaptive_grid:
                    want_agrid = True
                    agrid_tol = self.adaptive_grid_tol
                    agrid_max_depth = self.adaptive_grid_max_depth
                    agrid_min_depth = getattr(self, "adaptive_grid_min_depth", 2)
                else:
                    want_agrid = False

                # Replace grid if type changed
                if want_agrid and not isinstance(self.grid, iad.AGrid):
                    self.grid = None
                if not want_agrid and isinstance(self.grid, iad.AGrid):
                    self.grid = None
                if not want_agrid and isinstance(self.grid, iad.Grid) and getattr(self.grid, "N", None) != grid_n:
                    self.grid = None
                grid_was_none = self.grid is None
                if self.grid is None:
                    if want_agrid:
                        self.grid = iad.AGrid(tol=agrid_tol, max_depth=agrid_max_depth, min_depth=agrid_min_depth)
                    else:
                        self.grid = iad.Grid(N=grid_n)

                # the grids are two-dimensional, one value is held constant
                grid_constant = None
                if self.search == "find_ag":
                    grid_constant = self.sample.b
                if self.search == "find_bg":
                    grid_constant = self.sample.a
                if self.search == "find_ab":
                    grid_constant = self.sample.g

                initial_abg = (self.sample.a, self.sample.b, self.sample.g)
                stale_reason = None
                if self.debug_level & DEBUG_GRID:
                    if grid_was_none:
                        stale_reason = "GRID: Fill because NULL"
                    else:
                        stale_reason = _grid_stale_debug_reason(self.grid, self, self.search, grid_constant)

                if isinstance(self.grid, iad.AGrid):
                    if self.grid.is_stale(self, grid_constant, search=self.search):
                        if stale_reason is not None:
                            print(stale_reason, file=sys.stderr)
                        if self.debug_level & DEBUG_GRID:
                            _print_grid_fill_debug(self.search, grid_constant)
                        self.grid.calc(self, default=grid_constant, search=self.search)
                        if self.debug_level & DEBUG_LOST_LIGHT:
                            print(f"GRID: {self.sample.rt_evals} AD evaluations to fill the grid", file=sys.stderr)
                    else:
                        self.grid.exp = self
                else:
                    if self.grid.is_stale(self, grid_constant, search=self.search):
                        if stale_reason is not None:
                            print(stale_reason, file=sys.stderr)
                        if self.debug_level & DEBUG_GRID:
                            _print_grid_fill_debug(self.search, grid_constant)
                        if debug_grid_calc:
                            _print_grid_calc_fill_header()
                        self.grid.calc(self, grid_constant)
                        if self.debug_level & DEBUG_LOST_LIGHT:
                            print(f"GRID: {self.sample.rt_evals} AD evaluations to fill the grid", file=sys.stderr)

                if self.debug_level & DEBUG_GRID:
                    print("GRID: Finding best grid points", file=sys.stderr)
                best_guesses = None
                if self.debug_level & DEBUG_BEST_GUESS:
                    best_guesses = _ranked_grid_guesses(self.grid, self, initial_abg)
                    _print_best_guess_debug(self.search, best_guesses)
                if debug_grid_calc and isinstance(self.grid, iad.Grid):
                    a, b, g = _debug_grid_calc_min_abg(self.grid, self)
                else:
                    a, b, g = self.grid.min_abg(self.m_r, self.m_t, exp=self)
                self._grid_evals = self.sample.rt_evals
                if best_guesses is not None:
                    a = best_guesses[0]["a"]
                    b = best_guesses[0]["b"]
                    g = best_guesses[0]["g"]

            a, b, g = _nudge_bounded_start(self.search, a, b, g)

        if self.search == "find_ab":
            x = scipy.optimize.Bounds(np.array([0, 0]), np.array([1, np.inf]))
            nm_opts = {"initial_simplex": initial_simplex} if initial_simplex is not None else {}
            result = scipy.optimize.minimize(
                abfun, [a, b], args=(self), bounds=x, method="Nelder-Mead", options=nm_opts or None
            )

        if self.search == "find_ag":
            x = scipy.optimize.Bounds(
                np.array([0, -1 + G_BOUND_EPS]),
                np.array([1, 1 - G_BOUND_EPS]),
            )
            nm_opts = {"initial_simplex": initial_simplex} if initial_simplex is not None else {}
            result = scipy.optimize.minimize(
                agfun, [a, g], args=(self), bounds=x, method="Nelder-Mead", options=nm_opts or None
            )

        if self.search == "find_bg":
            x = scipy.optimize.Bounds(
                np.array([0, -1 + G_BOUND_EPS]),
                np.array([np.inf, 1 - G_BOUND_EPS]),
            )
            nm_opts = {"initial_simplex": initial_simplex} if initial_simplex is not None else {}
            result = scipy.optimize.minimize(
                bgfun, [b, g], args=(self), bounds=x, method="Nelder-Mead", options=nm_opts or None
            )

        if self.search == "find_ba":
            guess = max(self.sample.b - self.default_bs, 0.0)
            x = scipy.optimize.Bounds(np.array([0.0]), np.array([np.inf]))
            result = scipy.optimize.minimize(bafun, [guess], args=(self), bounds=x, method="Powell")

        if self.search == "find_bs":
            guess = max(self.sample.b - self.default_ba, 0.0)
            x = scipy.optimize.Bounds(np.array([0.0]), np.array([np.inf]))
            result = scipy.optimize.minimize(bsfun, [guess], args=(self), bounds=x, method="Powell")

        if self.search == "find_bag":
            guess_ba = max(self.sample.b - self.default_bs, 0.0)
            x = scipy.optimize.Bounds(
                np.array([0.0, -1 + G_BOUND_EPS]),
                np.array([np.inf, 1 - G_BOUND_EPS]),
            )
            result = scipy.optimize.minimize(
                bagfun,
                [guess_ba, self.sample.g],
                args=(self),
                bounds=x,
                method="Powell",
            )

        if self.search == "find_bsg":
            guess_bs = max(self.sample.b - self.default_ba, 0.0)
            x = scipy.optimize.Bounds(
                np.array([0.0, -1 + G_BOUND_EPS]),
                np.array([np.inf, 1 - G_BOUND_EPS]),
            )
            result = scipy.optimize.minimize(
                bsgfun,
                [guess_bs, self.sample.g],
                args=(self),
                bounds=x,
                method="Powell",
            )

        if self.search == "find_b_no_absorption":
            self.sample.a = 1.0
            result = scipy.optimize.minimize_scalar(bfun, args=(self), method="brent")

        if self.search == "find_b_no_scattering":
            self.sample.a = 0.0
            result = scipy.optimize.minimize_scalar(bfun, args=(self), method="brent")

        self._optimizer_evals = self.sample.rt_evals - self._grid_evals

        if result is not None:
            self.iterations = getattr(result, "nit", getattr(result, "nfev", 0))
            self.final_distance = float(getattr(result, "fun", np.nan))
            self.found = bool(np.isfinite(self.final_distance) and self.final_distance < self.tolerance)
            self._last_invert_status_valid = True
            if self.debug_level & DEBUG_ITERATIONS:
                print(f"Final amoeba/brent result after {self.iterations} iterations", file=sys.stderr)
                self._debug_iteration_row()
            if self.debug_level & DEBUG_A_LITTLE:
                self.debug_a_little_summary()

        return self.sample.a, self.sample.b, self.sample.g

    def print_dot(self):
        """Print a character for each datapoint during analysis."""
        if self.verbosity == 0 or self.debug_level:
            return
        self.counter += 1
        if self.counter % 50 == 0:
            print(file=sys.stderr)
        print(".", end="", file=sys.stderr)
        sys.stderr.flush()

    def _update_lost_light(self, a, b, g, n_photons_override=None):
        """Call mc_lost binary once; apply damped update to lost fractions.

        Uses the current sphere geometry and sample parameters to assemble the
        mc_lost command.  Applies a damped update (factor 0.3) to avoid oscillation.

        Args:
            a: albedo from the most recent invert_scalar_rt() call
            b: optical thickness from the most recent invert_scalar_rt() call
            g: anisotropy from the most recent invert_scalar_rt() call
            n_photons_override: if given, use this photon count instead of self.n_photons

        Returns:
            Tuple of:
              - maximum absolute change across all four lost fractions
              - raw change in direct reflected lost light
              - raw change in direct transmitted lost light
              - raw change in diffuse reflected lost light
              - raw change in diffuse transmitted lost light
        """
        n_sample = float(self.sample.n)
        n_slide = float(self.sample.n_above) if self.sample.n_above != 1.0 else 1.0
        t_sample = float(self.sample.d)
        t_slide_val = float(self.sample.d_above) if self.sample.d_above is not None else float(self.t_slide)

        d_port_r = float(self.r_sphere.sample.d) if self.r_sphere is not None else 1000.0
        d_port_t = float(self.t_sphere.sample.d) if self.t_sphere is not None else d_port_r

        n_ph = int(n_photons_override) if n_photons_override is not None else int(self.n_photons)
        new_ur1, new_ut1, new_uru, new_utu = run_mc_lost(
            a=float(a),
            b=float(b),
            g=float(g),
            n_sample=n_sample,
            n_slide=n_slide,
            d_port_r=d_port_r,
            d_port_t=d_port_t,
            d_beam=float(self.d_beam),
            t_sample=t_sample,
            t_slide=t_slide_val,
            n_photons=n_ph,
            method=self.method,
            binary_path=self.mc_lost_path,
        )

        if self.debug_level & DEBUG_MC:
            print(
                "mc_lost input: "
                f"a={a:.5f} b={b:.5f} g={g:.5f} "
                f"n_sample={n_sample:.5f} n_slide={n_slide:.5f} "
                f"d_port_r={d_port_r:.5f} d_port_t={d_port_t:.5f} "
                f"d_beam={float(self.d_beam):.5f} t_sample={t_sample:.5f} "
                f"t_slide={t_slide_val:.5f} n_photons={n_ph} "
                f"method={self.method}",
                file=sys.stderr,
            )

        if self.debug_level & DEBUG_MC:
            print(
                f"  MC iter {self._mc_iterations + 1}: "
                f"a={a:.5f} b={b:.5f} g={g:.5f} | "
                f"ur1_lost {self.ur1_lost:.5f}→{new_ur1:.5f}  "
                f"ut1_lost {self.ut1_lost:.5f}→{new_ut1:.5f}  "
                f"uru_lost {self.uru_lost:.5f}→{new_uru:.5f}  "
                f"utu_lost {self.utu_lost:.5f}→{new_utu:.5f}",
                file=sys.stderr,
            )

        FACTOR = 0.3
        diff_ur1 = new_ur1 - self.ur1_lost
        diff_ut1 = new_ut1 - self.ut1_lost
        diff_uru = new_uru - self.uru_lost
        diff_utu = new_utu - self.utu_lost

        self.ur1_lost += FACTOR * diff_ur1
        self.ut1_lost += FACTOR * diff_ut1
        self.uru_lost += FACTOR * diff_uru
        self.utu_lost += FACTOR * diff_utu

        max_abs_diff = max(abs(diff_ur1), abs(diff_ut1), abs(diff_uru), abs(diff_utu))
        return max_abs_diff, diff_ur1, diff_ut1, diff_uru, diff_utu

    def _current_optical_coefficients(self):
        """Return the current `(mu_a, mu_s')` implied by the sample state."""
        return self.sample.mu_a(), self.sample.mu_sp()

    def _invert_scalar_with_mc(self):
        """Run invert_scalar_rt() with optional MC lost light iteration.

        When mc_lost_path is set and num_spheres > 0, repeatedly calls the
        mc_lost binary to refine the four lost-light fractions until they
        converge, then re-inverts with the final lost values.

        The algorithm mirrors iad_main.w:998-1078:
          1. Invert ignoring lost light (ur1_lost etc. start at 0 or prior value)
          2. MC estimate → damped update of lost fractions
          3. Re-invert with updated lost fractions
          4. Repeat until `mu_a` and `mu_s'` stop changing, with the same
             direct-loss guard used by the C implementation

        Returns:
            (a, b, g) — optical properties of the slab
        """
        self._mc_iterations = 0
        a, b, g = self.invert_scalar_rt()
        self.first_pass_abg = (a, b, g)

        if self.num_spheres > 0 and self.debug_level & DEBUG_LOST_LIGHT:
            self._debug_lost_light_header()
            self._debug_lost_light_row()

        if self.mc_lost_path is None or self.num_spheres == 0:
            return a, b, g

        previous_delta = None  # per-axis |Δparam| from the last MC stage; None on first pass
        _prev_diff_ur1 = None  # diff_ur1 from the previous iteration, for photon escalation
        _mc_failed = False  # True if loop exited because AD failed to converge

        for _ in range(self.max_mc_iterations):
            last_mu_a, last_mu_sp = self._current_optical_coefficients()
            if self.debug_level & (DEBUG_ITERATIONS | DEBUG_A_LITTLE):
                print(
                    f"\n------------- Monte Carlo Iteration {self._mc_iterations + 1} -----------------",
                    file=sys.stderr,
                )
            # Adaptive photon escalation: use fewer photons when corrections are large
            # (early iterations) and more photons when corrections have nearly settled
            # (final iterations), concentrating the photon budget where it matters.
            if _prev_diff_ur1 is None or _prev_diff_ur1 > 0.01:
                # Early: corrections still large, cheap run is fine
                _n_ph_this = max(self.n_photons // 10, 10_000)
            elif _prev_diff_ur1 > 0.001:
                # Middle: corrections settling, use base photon count
                _n_ph_this = self.n_photons
            else:
                # Final: corrections nearly converged, use more photons for cleaner AD landscape
                _n_ph_this = min(self.n_photons * 5, 10_000_000)
            max_diff, diff_ur1, diff_ut1, _diff_uru, _diff_utu = self._update_lost_light(
                a, b, g, n_photons_override=_n_ph_this
            )
            _prev_diff_ur1 = diff_ur1
            # Build an adaptive initial simplex centred on the current solution.
            # Step sizes shrink proportionally to the previous iteration's parameter
            # movement, falling back to fixed defaults on the first MC stage.
            # Only applies to 2-D Nelder-Mead searches (find_ab / find_ag / find_bg).
            if self.search in ("find_ab", "find_ag", "find_bg"):
                simplex = _build_adaptive_simplex(self.search, (a, b, g), previous_delta)
            else:
                simplex = None
            self._debug(DEBUG_GRID_CALC, "warm-starting from previous (a,b,g) after lost-light update")
            self._mc_iterations += 1
            a_new, b_new, g_new = self.invert_scalar_rt(hot_start=(a, b, g), initial_simplex=simplex)
            # Record the per-axis movement for the next iteration's simplex sizing.
            if self.search == "find_ab":
                previous_delta = np.array([abs(a_new - a), abs(b_new - b)], dtype=float)
            elif self.search == "find_ag":
                previous_delta = np.array([abs(a_new - a), abs(g_new - g)], dtype=float)
            elif self.search == "find_bg":
                previous_delta = np.array([abs(b_new - b), abs(g_new - g)], dtype=float)
            a, b, g = a_new, b_new, g_new
            mu_a, mu_sp = self._current_optical_coefficients()
            self._debug_lost_light_row()

            # Determine whether the direct lost-light terms are still moving.
            # Must be computed before the convergence gate so both branches can use it.
            too_much_lost = diff_ur1 > 0.001 or diff_ut1 > 0.001

            # If the AD inversion did not converge, stop immediately.
            # More MC iterations will not fix a fundamentally non-convergent case.
            if self._last_invert_status_valid and not self.found:
                self._debug(DEBUG_ITERATIONS, "AD did not converge — stopping MC loop")
                _mc_failed = True
                break

            if mu_a is None or mu_sp is None or last_mu_a is None or last_mu_sp is None:
                if max_diff < self.MC_tolerance:
                    self._debug(DEBUG_ITERATIONS, "found!")
                    break
                continue

            if abs(last_mu_a - mu_a) > self.MC_tolerance:
                self._debug(DEBUG_ITERATIONS, "Repeat MC because mua is still changing")
                continue

            if abs(last_mu_sp - mu_sp) > self.MC_tolerance:
                self._debug(DEBUG_ITERATIONS, "Repeat MC because musp is still changing")
                continue

            if too_much_lost:
                self._debug(DEBUG_ITERATIONS, "Repeat MC because mua and musp are still changing")
                continue

            self._debug(DEBUG_ITERATIONS, "found!")
            break

        if _mc_failed:
            self.found = False

        return a, b, g

    def invert_rt(self):
        """Find a,b,g for experimental measurements.

        This method works if `m_r`, `m_t`, and `m_u` are scalars or arrays.
        When mc_lost_path is set and num_spheres > 0, an outer MC iteration
        loop refines the lost-light fractions before returning.

        Returns:
            - `a` is the single scattering albedo of the slab
            - `b` is the optical thickness of the slab
            - `g` is the anisotropy of single scattering
        """
        if self.m_r is None and self.m_t is None and self.m_u is None:
            return self._invert_scalar_with_mc()

        # any scalar measurement indicates a single data point
        if np.isscalar(self.m_r) or np.isscalar(self.m_t) or np.isscalar(self.m_u):
            return self._invert_scalar_with_mc()

        # figure out the number of points that we need to invert
        if self.m_r is not None:
            N = len(self.m_r)
        elif self.m_t is not None:
            N = len(self.m_t)
        else:
            N = len(self.m_u)

        a = np.zeros(N)
        b = np.zeros(N)
        g = np.zeros(N)

        for i in range(N):
            x = self.point_at(i)
            a[i], b[i], g[i] = x._invert_scalar_with_mc()  # pylint: disable=protected-access
            self.print_dot()

        print(file=sys.stderr)
        return a, b, g

    def what_is_b(self):
        """Find optical thickness using unscattered transmission."""
        s = self.sample
        t_u = self.m_u or 0

        r1, t1 = iad.absorbing_glass_RT(1.0, s.n_above, s.n, s.nu_0, s.b_above)

        mu = iad.cos_snell(1.0, s.nu_0, s.n)

        r2, t2 = iad.absorbing_glass_RT(s.n, s.n_below, 1.0, mu, s.b_below)

        if t_u <= 0:
            return np.inf

        if t_u >= t1 * t2 / (1 - r1 * r2):
            return 0.001

        tt = t1 * t2

        if r1 == 0 or r2 == 0:
            ratio = tt / t_u
        else:
            ratio = (tt + np.sqrt(tt**2 + 4 * t_u**2 * r1 * r2)) / (2 * t_u)

        return s.nu_0 * np.log(ratio)

    def measured_rt(self):
        r"""Calculate measured reflection and transmission.

        The direct incident power is :math:`(1-f_u)P`. The reflected power will
        be :math:`(1-f_u)R_{direct} P`.  Since baffles ensure that the light cannot
        reach the detector, we must bounce the light off the sphere walls to
        use to above gain formulas.  The contribution will then be

        .. math:: (1-f_u)R_{direct} (1-a_e) r_w P.

        The measured power will be

        .. math:: P_d = a_d (1-a_e) r_w [(1-f_u) r_{direct} + f_u r_w] P ⋅ G(r_s)

        Similarly the power falling on the detector measuring transmitted light is

        .. math:: P_d'= a_d' t_{direct} r_w' (1-a_e') P ⋅ G'(r_s)

        when the empty port in the transmission sphere is closed,
        :math:`a_e'=0`.

        The normalized sphere measurements are

        .. math:: M_R = r_{std}⋅\frac{R(r_{direct},r_s)-R(0,0)}{R(r_{std},r_{std})-R(0,0)}

        and

        .. math:: M_T = t_{std}⋅\frac{T(t_{direct},r_s)-T(0,0)}{T(t_{std},r_{std})-T(0,0)}

        Args:
            ur1: reflection for collimated incidence
            ut1: transmission for collimated incidence
            uru: reflection for diffuse incidence
            utu: transmission for diffuse incidence

        Returns:
            [float, float]: measured reflection and transmission
        """
        ur1, ut1, uru, utu = self.sample.rt()
        return self.measured_rt_from_raw(ur1, ut1, uru, utu, include_lost=True)


def afun(x, *args):
    """Vary the albedo."""
    exp = args[0]
    exp.sample.a = x
    m_r, m_t = exp.measured_rt()

    result = exp.measurement_distance(m_r, m_t)
    exp._debug_iteration_row(m_r, m_t, result)  # pylint: disable=protected-access
    if exp.debug_level & DEBUG_EVERY_CALC:
        print(f"calc find_a: a={x:.7f} mr={m_r:.7f} mt={m_t:.7f} delta={result:.7g}", file=sys.stderr)
    return result


def bfun(x, *args):
    """Vary the optical thickness."""
    exp = args[0]
    exp.sample.b = x
    m_r, m_t = exp.measured_rt()

    result = exp.measurement_distance(m_r, m_t)
    exp._debug_iteration_row(m_r, m_t, result)  # pylint: disable=protected-access
    if exp.debug_level & DEBUG_EVERY_CALC:
        print(f"calc find_b: b={x:.7f} mr={m_r:.7f} mt={m_t:.7f} delta={result:.7g}", file=sys.stderr)
    return result


def gfun(x, *args):
    """Vary the anisotropy."""
    exp = args[0]
    exp.sample.g = x
    m_r, m_t = exp.measured_rt()

    result = exp.measurement_distance(m_r, m_t)
    exp._debug_iteration_row(m_r, m_t, result)  # pylint: disable=protected-access
    if exp.debug_level & DEBUG_EVERY_CALC:
        print(f"calc find_g: g={x:.7f} mr={m_r:.7f} mt={m_t:.7f} delta={result:.7g}", file=sys.stderr)
    return result


def abfun(x, *args):
    """Vary the ab."""
    exp = args[0]
    exp.sample.a = x[0]
    exp.sample.b = x[1]
    m_r, m_t = exp.measured_rt()
    delta = exp.measurement_distance(m_r, m_t)
    exp._debug_iteration_row(m_r, m_t, delta)  # pylint: disable=protected-access
    if exp.debug_level & DEBUG_EVERY_CALC:
        print(
            f"calc find_ab: a={exp.sample.a:.7f} b={exp.sample.b:.7f} " f"mr={m_r:.7f} mt={m_t:.7f} delta={delta:.7g}",
            file=sys.stderr,
        )
    return delta


def bgfun(x, *args):
    """Vary the bg."""
    exp = args[0]
    exp.sample.b = x[0]
    exp.sample.g = x[1]
    m_r, m_t = exp.measured_rt()
    delta = exp.measurement_distance(m_r, m_t)
    exp._debug_iteration_row(m_r, m_t, delta)  # pylint: disable=protected-access
    if exp.debug_level & DEBUG_EVERY_CALC:
        print(
            f"calc find_bg: b={exp.sample.b:.7f} g={exp.sample.g:.7f} " f"mr={m_r:.7f} mt={m_t:.7f} delta={delta:.7g}",
            file=sys.stderr,
        )
    return delta


def agfun(x, *args):
    """Vary the ag."""
    exp = args[0]
    exp.sample.a = x[0]
    exp.sample.g = x[1]
    m_r, m_t = exp.measured_rt()
    delta = exp.measurement_distance(m_r, m_t)
    exp._debug_iteration_row(m_r, m_t, delta)  # pylint: disable=protected-access
    if exp.debug_level & DEBUG_EVERY_CALC:
        print(
            f"calc find_ag: a={exp.sample.a:.7f} g={exp.sample.g:.7f} " f"mr={m_r:.7f} mt={m_t:.7f} delta={delta:.7g}",
            file=sys.stderr,
        )
    return delta


def bafun(x, *args):
    """Vary the absorption optical depth with scattering optical depth fixed."""
    exp = args[0]
    ba = float(np.atleast_1d(x)[0])
    exp._set_sample_from_ba_bs(ba, exp.default_bs)  # pylint: disable=protected-access
    m_r, m_t = exp.measured_rt()
    delta = exp.measurement_distance(m_r, m_t)
    exp._debug_iteration_row(m_r, m_t, delta)  # pylint: disable=protected-access
    if exp.debug_level & DEBUG_EVERY_CALC:
        print(
            f"calc find_ba: ba={ba:.7f} a={exp.sample.a:.7f} b={exp.sample.b:.7f} "
            f"mr={m_r:.7f} mt={m_t:.7f} delta={delta:.7g}",
            file=sys.stderr,
        )
    return delta


def bsfun(x, *args):
    """Vary the scattering optical depth with absorption optical depth fixed."""
    exp = args[0]
    bs = float(np.atleast_1d(x)[0])
    exp._set_sample_from_ba_bs(exp.default_ba, bs)  # pylint: disable=protected-access
    m_r, m_t = exp.measured_rt()
    delta = exp.measurement_distance(m_r, m_t)
    exp._debug_iteration_row(m_r, m_t, delta)  # pylint: disable=protected-access
    if exp.debug_level & DEBUG_EVERY_CALC:
        print(
            f"calc find_bs: bs={bs:.7f} a={exp.sample.a:.7f} b={exp.sample.b:.7f} "
            f"mr={m_r:.7f} mt={m_t:.7f} delta={delta:.7g}",
            file=sys.stderr,
        )
    return delta


def bagfun(x, *args):
    """Vary absorption optical depth and anisotropy with scattering fixed."""
    exp = args[0]
    ba = float(np.atleast_1d(x)[0])
    g = float(np.atleast_1d(x)[1])
    exp._set_sample_from_ba_bs(ba, exp.default_bs)  # pylint: disable=protected-access
    exp.sample.g = g
    m_r, m_t = exp.measured_rt()
    delta = exp.measurement_distance(m_r, m_t)
    exp._debug_iteration_row(m_r, m_t, delta)  # pylint: disable=protected-access
    if exp.debug_level & DEBUG_EVERY_CALC:
        print(
            f"calc find_bag: ba={ba:.7f} g={g:.7f} a={exp.sample.a:.7f} b={exp.sample.b:.7f} "
            f"mr={m_r:.7f} mt={m_t:.7f} delta={delta:.7g}",
            file=sys.stderr,
        )
    return delta


def bsgfun(x, *args):
    """Vary scattering optical depth and anisotropy with absorption fixed."""
    exp = args[0]
    bs = float(np.atleast_1d(x)[0])
    g = float(np.atleast_1d(x)[1])
    exp._set_sample_from_ba_bs(exp.default_ba, bs)  # pylint: disable=protected-access
    exp.sample.g = g
    m_r, m_t = exp.measured_rt()
    delta = exp.measurement_distance(m_r, m_t)
    exp._debug_iteration_row(m_r, m_t, delta)  # pylint: disable=protected-access
    if exp.debug_level & DEBUG_EVERY_CALC:
        print(
            f"calc find_bsg: bs={bs:.7f} g={g:.7f} a={exp.sample.a:.7f} b={exp.sample.b:.7f} "
            f"mr={m_r:.7f} mt={m_t:.7f} delta={delta:.7g}",
            file=sys.stderr,
        )
    return delta
