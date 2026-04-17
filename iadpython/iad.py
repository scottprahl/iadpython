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
        else:
            if self.m_u is None or self.m_u <= 0:
                self.search = "find_ab"
            else:
                self.search = "find_ag"

    def determine_search(self):
        """Determine type of search to do."""
        if self.search_override not in (None, "auto"):
            self.search = self.search_override
            self.search_code = next(
                (code for code, name in SEARCH_CODE_TO_NAME.items() if name == self.search),
                4,
            )
            self._debug(DEBUG_SEARCH, f"search override -> {self.search} ({self.search_code})")
            return

        if self.num_measurements == 0:
            self.search = "unknown"
            self.search_code = -1
            return

        if self.num_measurements == 1:
            self.determine_one_parameter_search()
        else:
            self.determine_two_parameter_search()

        self.search_code = next(
            (code for code, name in SEARCH_CODE_TO_NAME.items() if name == self.search),
            4,
        )
        self._debug(DEBUG_SEARCH, f"automatic search -> {self.search} ({self.search_code})")

    def _debug(self, mask, message):
        """Emit debug output matching the enabled bitmask."""
        if self.debug_level & mask:
            print(message, file=sys.stderr)

    def _debug_search_header(self):
        """Emit the CWEB-style inverse-search header for ``-x 4``."""
        if not (self.debug_level & DEBUG_ITERATIONS):
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
        if not (self.debug_level & DEBUG_ITERATIONS):
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
        if not (self.debug_level & DEBUG_LOST_LIGHT):
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
        if not (self.debug_level & DEBUG_LOST_LIGHT):
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
        return point

    def measured_rt_from_raw(self, ur1, ut1, uru, utu, include_lost=True, a=None, b=None, g=None):
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

            if self.debug_level & DEBUG_A_LITTLE:
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
                if self.debug_level & DEBUG_SPHERE_GAIN:
                    print(
                        "double sphere: "
                        f"ur1={ur1_calc:.5f} uru={uru_calc:.5f} "
                        f"ut1={ut1_calc:.5f} utu={utu_calc:.5f} -> "
                        f"MR={m_r:.5f} MT={m_t:.5f}",
                        file=sys.stderr,
                    )
                return m_r, m_t

            if self.num_spheres == 1:
                if self.method in ("comparison", 1):
                    return m_r, m_t

                if self.r_sphere is not None:
                    f_u = self.fraction_of_rc_in_mr
                    m_r = self.r_sphere.MR(ur1_actual, uru_actual, R_u=r_u, f_u=f_u, f_w=self.f_r)
                    if self.debug_level & DEBUG_SPHERE_GAIN:
                        print(
                            f"reflectance sphere MR={m_r:.5f} from ur1={ur1_actual:.5f} uru={uru_actual:.5f}",
                            file=sys.stderr,
                        )

                if self.t_sphere is not None:
                    m_t = self.t_sphere.MT(ut1_actual, uru_actual, T_u=t_u, f_u=self.fraction_of_tc_in_mt)
                    if self.debug_level & DEBUG_SPHERE_GAIN:
                        print(
                            f"transmission sphere MT={m_t:.5f} from ut1={ut1_actual:.5f} uru={uru_actual:.5f}",
                            file=sys.stderr,
                        )

            return m_r, m_t
        finally:
            if original is not None:
                s.a, s.b, s.g = original

    def measurement_distance_from_raw(self, ur1, ut1, uru, utu, include_lost=True, a=None, b=None, g=None):
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
        )

        delta = 0.0
        if self.m_r is not None:
            delta += abs(_as_scalar_float(m_r, "calculated M_R") - _as_scalar_float(self.m_r, "measured M_R"))
        if self.m_t is not None:
            delta += abs(_as_scalar_float(m_t, "calculated M_T") - _as_scalar_float(self.m_t, "measured M_T"))
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

        #        print('search is', self.search)
        #        print('     a = ', self.sample.a)
        #        print('     b = ', self.sample.b)
        #        print('     g = ', self.sample.g)

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

            if hot_start is not None:
                # Bypass the grid: use the caller-supplied starting point
                # directly.  The grid is not rebuilt; the raw-RT cache is
                # still valid because lost-light values do not affect it.
                a, b, g = hot_start
                self._grid_evals = 0
                self._debug(DEBUG_BEST_GUESS, f"hot start a={a:.5f} b={b:.5f} g={g:.5f}")
            else:
                # Determine grid type for this search mode.
                # use_adaptive_grid=True : always AGrid with user-tunable tol/depth.
                # use_adaptive_grid=False: always Grid(N=grid_n).
                # use_adaptive_grid=None (default): auto-select.
                agrid_tol = self.adaptive_grid_tol
                agrid_max_depth = self.adaptive_grid_max_depth
                agrid_min_depth = getattr(self, "adaptive_grid_min_depth", 2)
                if self.use_adaptive_grid is None:
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
                if self.grid is None:
                    if want_agrid:
                        self.grid = iad.AGrid(tol=agrid_tol, max_depth=agrid_max_depth, min_depth=agrid_min_depth)
                    else:
                        self.grid = iad.Grid(N=self.grid_n)

                # the grids are two-dimensional, one value is held constant
                grid_constant = None
                if self.search == "find_ag":
                    grid_constant = self.sample.b
                if self.search == "find_bg":
                    grid_constant = self.sample.a
                if self.search == "find_ab":
                    grid_constant = self.sample.g

                if isinstance(self.grid, iad.AGrid):
                    if self.grid.is_stale(self, grid_constant, search=self.search):
                        self._debug(DEBUG_GRID_CALC, f"recomputing adaptive grid for {self.search}")
                        self.grid.calc(self, default=grid_constant, search=self.search)
                else:
                    if self.grid.is_stale(self, grid_constant, search=self.search):
                        self._debug(DEBUG_GRID_CALC, f"recomputing grid for {self.search}")
                        self.grid.calc(self, grid_constant)

                a, b, g = self.grid.min_abg(self.m_r, self.m_t, exp=self)
                self._grid_evals = self.sample.rt_evals
                if self.debug_level & DEBUG_BEST_GUESS:
                    print("grid constant %8.5f" % grid_constant)
                    print("grid start a=%8.5f" % a)
                    print("grid start b=%8.5f" % b)
                    print("grid start g=%8.5f" % g)
                elif self.debug_level & DEBUG_GRID:
                    print(f"grid constant {grid_constant:8.5f}", file=sys.stderr)

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

        if result is not None:
            self.iterations = getattr(result, "nit", getattr(result, "nfev", 0))
            self.final_distance = float(getattr(result, "fun", np.nan))
            self.found = bool(np.isfinite(self.final_distance) and self.final_distance < self.tolerance)
            self._last_invert_status_valid = True
            if self.debug_level & DEBUG_ITERATIONS:
                print(f"Final amoeba/brent result after {self.iterations} iterations", file=sys.stderr)
                self._debug_iteration_row()

        self._optimizer_evals = self.sample.rt_evals - self._grid_evals

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

    def _update_lost_light(self, a, b, g):
        """Call mc_lost binary once; apply damped update to lost fractions.

        Uses the current sphere geometry and sample parameters to assemble the
        mc_lost command.  Applies a damped update (factor 0.3, matching the C
        implementation in iad_main.w) to avoid oscillation.

        Args:
            a: albedo from the most recent invert_scalar_rt() call
            b: optical thickness from the most recent invert_scalar_rt() call
            g: anisotropy from the most recent invert_scalar_rt() call

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

        d_port_r = float(self.r_sphere.sample.d) if self.r_sphere is not None else 1000.0
        d_port_t = float(self.t_sphere.sample.d) if self.t_sphere is not None else d_port_r

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
            t_slide=float(self.t_slide),
            n_photons=int(self.n_photons),
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
                f"t_slide={float(self.t_slide):.5f} n_photons={int(self.n_photons)} "
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

        for _ in range(self.max_mc_iterations):
            last_mu_a, last_mu_sp = self._current_optical_coefficients()
            last_final_distance = self.final_distance
            if self.debug_level & (DEBUG_ITERATIONS | DEBUG_A_LITTLE):
                print(
                    f"\n------------- Monte Carlo Iteration {self._mc_iterations + 1} -----------------",
                    file=sys.stderr,
                )
            max_diff, diff_ur1, diff_ut1, _diff_uru, _diff_utu = self._update_lost_light(a, b, g)
            # Build an adaptive initial simplex centred on the current solution.
            # Step sizes shrink proportionally to the previous iteration's parameter
            # movement, falling back to fixed defaults on the first MC stage.
            # Only applies to 2-D Nelder-Mead searches (find_ab / find_ag / find_bg).
            if self.search in ("find_ab", "find_ag", "find_bg"):
                simplex = _build_adaptive_simplex(self.search, (a, b, g), previous_delta)
            else:
                simplex = None
            self._debug(DEBUG_GRID_CALC, "warm-starting from previous (a,b,g) after lost-light update")
            a_new, b_new, g_new = self.invert_scalar_rt(hot_start=(a, b, g), initial_simplex=simplex)
            # Record the per-axis movement for the next iteration's simplex sizing.
            if self.search == "find_ab":
                previous_delta = np.array([abs(a_new - a), abs(b_new - b)], dtype=float)
            elif self.search == "find_ag":
                previous_delta = np.array([abs(a_new - a), abs(g_new - g)], dtype=float)
            elif self.search == "find_bg":
                previous_delta = np.array([abs(b_new - b), abs(g_new - g)], dtype=float)
            a, b, g = a_new, b_new, g_new
            self._mc_iterations += 1
            mu_a, mu_sp = self._current_optical_coefficients()
            self._debug_lost_light_row()

            # Match the CWEB iad_main.w convergence gate:
            # 1. wait for optical properties to stabilize
            # 2. keep iterating while direct lost-light terms are still moving
            if self._last_invert_status_valid and not self.found:
                if last_final_distance - self.final_distance < self.MC_tolerance:
                    self._debug(DEBUG_ITERATIONS, "MC does not make things better")
                    break
                self._debug(DEBUG_ITERATIONS, "Repeat MC because distance is reduced")
                continue

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

            too_much_lost = diff_ur1 > 0.001 or diff_ut1 > 0.001
            if too_much_lost:
                self._debug(DEBUG_ITERATIONS, "Repeat MC because mua and musp are still changing")
                continue

            self._debug(DEBUG_ITERATIONS, "found!")
            break

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

    result = 0
    if exp.m_r is not None:
        result += np.abs(m_r - exp.m_r)
    if exp.m_t is not None:
        result += np.abs(m_t - exp.m_t)
    exp._debug_iteration_row(m_r, m_t, result)  # pylint: disable=protected-access
    if exp.debug_level & DEBUG_EVERY_CALC:
        print(f"calc find_a: a={x:.7f} mr={m_r:.7f} mt={m_t:.7f} delta={result:.7g}", file=sys.stderr)
    return result


def bfun(x, *args):
    """Vary the optical thickness."""
    exp = args[0]
    exp.sample.b = x
    m_r, m_t = exp.measured_rt()

    result = 0
    if exp.m_r is not None:
        result += np.abs(m_r - exp.m_r)
    if exp.m_t is not None:
        result += np.abs(m_t - exp.m_t)
    exp._debug_iteration_row(m_r, m_t, result)  # pylint: disable=protected-access
    if exp.debug_level & DEBUG_EVERY_CALC:
        print(f"calc find_b: b={x:.7f} mr={m_r:.7f} mt={m_t:.7f} delta={result:.7g}", file=sys.stderr)
    return result


def gfun(x, *args):
    """Vary the anisotropy."""
    exp = args[0]
    exp.sample.g = x
    m_r, m_t = exp.measured_rt()

    result = 0
    if exp.m_r is not None:
        result += np.abs(m_r - exp.m_r)
    if exp.m_t is not None:
        result += np.abs(m_t - exp.m_t)
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
    delta = np.abs(m_r - exp.m_r) + np.abs(m_t - exp.m_t)
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
    delta = np.abs(m_r - exp.m_r) + np.abs(m_t - exp.m_t)
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
    delta = np.abs(m_r - exp.m_r) + np.abs(m_t - exp.m_t)
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
    delta = np.abs(m_r - exp.m_r) + np.abs(m_t - exp.m_t)
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
    delta = np.abs(m_r - exp.m_r) + np.abs(m_t - exp.m_t)
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
    delta = np.abs(m_r - exp.m_r) + np.abs(m_t - exp.m_t)
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
    delta = np.abs(m_r - exp.m_r) + np.abs(m_t - exp.m_t)
    exp._debug_iteration_row(m_r, m_t, delta)  # pylint: disable=protected-access
    if exp.debug_level & DEBUG_EVERY_CALC:
        print(
            f"calc find_bsg: bs={bs:.7f} g={g:.7f} a={exp.sample.a:.7f} b={exp.sample.b:.7f} "
            f"mr={m_r:.7f} mt={m_t:.7f} delta={delta:.7g}",
            file=sys.stderr,
        )
    return delta
