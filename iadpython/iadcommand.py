#!/usr/bin/env python3
"""Command line for support for iadpython."""

import os
import copy
import shutil
import sys
from enum import Enum
import argparse
from contextlib import redirect_stdout
import time
import numpy as np
import iadpython

# Global COUNTER to mimic static behavior in C
COUNTER = 0
ANY_ERROR = False


class SlidePosition(Enum):
    """All the combinations of glass and sample."""

    NO_SLIDES = 0
    ONE_SLIDE_ON_TOP = 1
    TWO_IDENTICAL_SLIDES = 2
    ONE_SLIDE_ON_BOTTOM = 3
    ONE_SLIDE_NEAR_SPHERE = 4
    ONE_SLIDE_NOT_NEAR_SPHERE = 5


class LightCondition(Enum):
    """Options for interpreting illumination."""

    MR_IS_ONLY_RD = 1
    MT_IS_ONLY_TD = 2
    NO_UNSCATTERED_LIGHT = 3


class InputError(Enum):
    """Possible input errors."""

    NO_ERROR = 0
    TOO_MANY_ITERATIONS = 1
    MR_TOO_BIG = 2
    MR_TOO_SMALL = 3
    MT_TOO_BIG = 4
    MT_TOO_SMALL = 5
    MU_TOO_BIG = 6
    MU_TOO_SMALL = 7
    TOO_MUCH_LIGHT = 8


def print_error_legend():
    """Print error explanation and quit."""
    print("----------------- Sorry, but ... errors encountered ---------------")
    print("   *  ==> Success          ")
    print("  0-9 ==> Monte Carlo Iteration")
    print("   R  ==> M_R is too big   ")
    print("   r  ==> M_R is too small")
    print("   T  ==> M_T is too big   ")
    print("   t  ==> M_T is too small")
    print("   U  ==> M_U is too big   ")
    print("   u  ==> M_U is too small")
    print("   !  ==> M_R + M_T > 1    ")
    print("   +  ==> Did not converge\n")
    sys.exit(0)


def what_char(err):
    """Return appropriate character for analysis of current datapoint."""
    if err == InputError.NO_ERROR:
        return "*"
    if err == InputError.TOO_MANY_ITERATIONS:
        return "+"
    if err == InputError.MR_TOO_BIG:
        return "R"
    if err == InputError.MR_TOO_SMALL:
        return "r"
    if err == InputError.MT_TOO_BIG:
        return "T"
    if err == InputError.MT_TOO_SMALL:
        return "t"
    if err == InputError.MU_TOO_BIG:
        return "U"
    if err == InputError.MU_TOO_SMALL:
        return "u"
    if err == InputError.TOO_MUCH_LIGHT:
        return "!"
    return "?"


def print_dot(start_time, err, points, final, verbosity):
    """Print a character for each datapoint during analysis."""
    global COUNTER  # pylint: disable=global-statement
    global ANY_ERROR  # pylint: disable=global-statement
    COUNTER += 1

    if err != InputError.NO_ERROR:
        ANY_ERROR = err

    if verbosity == 0:
        return

    if final == 99:
        print(what_char(err), end="")
    else:
        COUNTER -= 1
        print(f"{final % 10}\b", end="")

    if final == 99:
        if COUNTER % 50 == 0:
            rate = (time.time() - start_time) / points
            print(f"  {points} done ({rate:.2f} s/pt)")
        elif COUNTER % 10 == 0:
            print(" ", end="")

    sys.stdout.flush()


def validator_01(value):
    """Is value between 0 and 1."""
    try:
        fvalue = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Commandline: {value} is not a valid number") from exc
    if not 0 <= fvalue <= 1:
        raise argparse.ArgumentTypeError(f"Commandline: {value} is not between 0 and 1")
    return fvalue


def validator_11(value):
    """Is value between -1 and 1."""
    try:
        fvalue = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value} is not a valid number") from exc
    if not -1 <= fvalue <= 1:
        raise argparse.ArgumentTypeError(f"Commandline: {value} is not between -1 and 1")
    if fvalue == -1:
        return -0.999999
    if fvalue == 1:
        return 0.999999
    return fvalue


def validator_positive(value):
    """Is value non-negative."""
    try:
        fvalue = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Commandline: {value} is not a valid number") from exc
    if fvalue < 0:
        raise argparse.ArgumentTypeError(f"{value} is not positive")
    return fvalue


def validator_mc_iterations(value):
    """Is number of Monte Carlo iterations between 0 and 50."""
    try:
        ivalue = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Commandline: {value} is not a valid integer") from exc
    if not 0 <= ivalue <= 50:
        raise argparse.ArgumentTypeError(f"Commandline: {value} must be between 0 and 50")
    return ivalue


def validator_search_code(value):
    """Is search override a valid CWEB search code."""
    try:
        ivalue = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Commandline: {value} is not a valid integer") from exc
    if ivalue not in iadpython.SEARCH_CODE_TO_NAME:
        raise argparse.ArgumentTypeError(f"Commandline: {value} must be between 0 and 12")
    return ivalue


def validator_wave_limits(value):
    """Parse a quoted wavelength-limit string like '500 600'."""
    parts = str(value).replace(",", " ").split()
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Commandline: -l expects two wavelength limits")
    try:
        wave_min = float(parts[0])
        wave_max = float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Commandline: wavelength limits must be numeric") from exc
    if wave_min > wave_max:
        raise argparse.ArgumentTypeError("Commandline: wavelength limits must be in ascending order")
    return (wave_min, wave_max)


def validator_scattering_constraint(value):
    """Accept a constant mus or a CWEB-style power-law scattering constraint."""
    text = str(value).strip()
    if text[:1].lower() != "p":
        return validator_positive(text)

    parts = text.split()
    if len(parts) != 4 or parts[0].lower() != "p":
        raise argparse.ArgumentTypeError("Commandline: bad -F option. Use '-F 1.0' or \"-F 'P 500 1.0 -1.3'\"")
    try:
        lambda0 = float(parts[1])
        mus0 = float(parts[2])
        gamma = float(parts[3])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Commandline: bad numeric value in -F power law") from exc
    if lambda0 <= 0 or mus0 < 0:
        raise argparse.ArgumentTypeError("Commandline: bad -F option. lambda0 must be positive and mus0 non-negative")
    return ("power", lambda0, mus0, gamma)


# Argument specifications
arg_specs = [
    {
        "flags": ["-1"],
        "dest": "r_sphere",
        "metavar": ("SPHERE_D", "SAMPLE_D", "ENTRANCE_D", "DETECTOR_D", "WALL_R"),
        "nargs": 5,
        "type": float,
        "help": "Reflection sphere parameters",
    },
    {
        "flags": ["-2"],
        "dest": "t_sphere",
        "metavar": ("SPHERE_D", "SAMPLE_D", "ENTRANCE_D", "DETECTOR_D", "WALL_R"),
        "nargs": 5,
        "type": float,
        "help": "Transmission sphere parameters",
    },
    {
        "flags": ["-a", "--albedo"],
        "dest": "albedo",
        "type": validator_01,
        "help": "Use this albedo",
    },
    {
        "flags": ["-A", "--mua"],
        "dest": "mua",
        "type": validator_positive,
        "help": "Use this absorption coefficient",
    },
    {
        "flags": ["-b"],
        "dest": "b",
        "type": validator_positive,
        "help": "Use this optical thickness",
    },
    {
        "flags": ["-B", "--diameter"],
        "dest": "diameter",
        "type": validator_positive,
        "help": "Beam diameter",
    },
    {
        "flags": ["-c"],
        "dest": "f_r",
        "type": validator_01,
        "help": "Fraction of unscattered refl in MR",
    },
    {
        "flags": ["-C"],
        "dest": "f_t",
        "type": validator_01,
        "help": "Fraction of unscattered trans in MT",
    },
    {
        "flags": ["-d"],
        "dest": "thickness",
        "type": validator_positive,
        "help": "Thickness of sample",
    },
    {
        "flags": ["-D"],
        "dest": "slide_thickness",
        "type": validator_positive,
        "help": "Thickness of slide",
    },
    {
        "flags": ["-e"],
        "dest": "error",
        "type": validator_positive,
        "help": "Error tolerance (default 0.0001)",
    },
    {
        "flags": ["-E"],
        "dest": "E",
        "type": validator_positive,
        "help": "Optical depth (=mua*D) for slides",
    },
    {
        "flags": ["-f"],
        "dest": "f_wall",
        "type": validator_01,
        "help": "Fraction 0.0-1.0 of light to hit sphere wall first",
    },
    {
        "flags": ["-F", "--mus"],
        "dest": "mus",
        "type": validator_scattering_constraint,
        "help": "Use this scattering coefficient",
    },
    {"flags": ["-g"], "type": validator_11, "help": "Scattering anisotropy"},
    {
        "flags": ["-G"],
        "type": str,
        "choices": ["0", "2", "t", "b", "n", "f"],
        "help": "Type of boundary ",
    },
    {"flags": ["-H"], "type": int, "choices": [0, 1, 2, 3], "help": "Sphere baffle configuration"},
    {"flags": ["-i"], "type": float, "help": "Light incident at this angle in degrees"},
    {"flags": ["-j"], "dest": "musp", "type": validator_positive, "help": "Use this reduced scattering coefficient"},
    {"flags": ["-J"], "action": "store_true", "help": "Generate grid after inverse calculation"},
    {"flags": ["-l"], "type": validator_wave_limits, "help": "Wavelength limits"},
    {"flags": ["-L"], "type": float, "help": "Specify the wavelength lambda"},
    {"flags": ["-M"], "type": validator_mc_iterations, "help": "Number of Monte Carlo iterations"},
    {
        "flags": ["-n", "--nslab"],
        "dest": "nslab",
        "type": validator_positive,
        "help": "Index of refraction of slab",
    },
    {
        "flags": ["-N", "--nslide"],
        "dest": "nslide",
        "type": validator_positive,
        "help": "Index of refraction of slides",
    },
    {"flags": ["-o"], "dest": "out_fname", "type": str, "help": "Filename for output"},
    {"flags": ["-p"], "type": int, "help": "# of Monte Carlo photons (negative means milliseconds)"},
    {
        "flags": ["-q"],
        "type": int,
        "default": 8,
        "help": "Number of quadrature points (default=8)",
    },
    {"flags": ["-r"], "type": validator_01, "help": "Total reflection measurement"},
    {
        "flags": ["-R"],
        "type": validator_01,
        "help": "Actual reflectance for 100%% measurement",
    },
    {"flags": ["-s"], "type": validator_search_code, "help": "Search override"},
    {
        "flags": ["-S"],
        "type": int,
        "choices": [0, 1, 2],
        "help": "Number of spheres used",
    },
    {"flags": ["-t"], "type": validator_01, "help": "Total transmission measurement"},
    {
        "flags": ["-T"],
        "type": validator_01,
        "help": "Actual transmission for 100%% measurement",
    },
    {
        "flags": ["-u"],
        "type": validator_01,
        "help": "Unscattered transmission measurement",
    },
    {
        "flags": ["-v", "--version"],
        "dest": "v",
        "action": "store_true",
        "help": "short version",
    },
    {"flags": ["-V"], "type": int, "choices": [0, 1, 2], "default": 2, "help": "verbosity level"},
    {
        "flags": ["--verbosity"],
        "dest": "V",
        "type": int,
        "choices": [0, 1, 2],
        "help": argparse.SUPPRESS,
    },
    {"flags": ["-w"], "type": validator_01, "help": "Reflection sphere wall reflectivity"},
    {"flags": ["-W"], "type": validator_01, "help": "Transmission sphere wall reflectivity"},
    {"flags": ["-x"], "type": int, "help": "Set debugging level"},
    {"flags": ["-X"], "action": "store_true", "help": "Dual beam configuration"},
    {"flags": ["-z"], "action": "store_true", "help": "Do forward calculation"},
    {
        "flags": ["filename"],
        "nargs": "?",
        "type": str,
        "default": None,
        "help": "Input filename",
    },
]


def print_version(verbosity):
    """Print version information using the CWEB `-v/-V` convention."""
    if verbosity == 0:
        print(f"iadp {iadpython.__version__}", end="")
        return

    s = ""
    s += "iadp " + iadpython.__version__ + "\n"
    s += "Copyright " + iadpython.__copyright__ + ", " + iadpython.__author__ + "\n"
    s += "\nForward and inverse adding-doubling radiative transport calculations.\n"
    s += "Extensive documentation is at <https://iadpython.readthedocs.io>\n"
    print(s, end="")


def example_text():
    """Return a string with some command-line examples."""
    s = ""
    s += "Examples:\n"
    s += "  iad file.rxt              Results will be put in file.txt\n"
    s += "  iad file                  Same as above\n"
    s += "  iad -c 0.9 file.rxt       Assume M_R includes 90%% of unscattered reflectance\n"
    s += "  iad -C 0.8 file.rxt       Assume M_T includes 80%% of unscattered transmittance\n"
    s += "  iad -e 0.0001 file.rxt    Better convergence to R & T values\n"
    s += "  iad -f 1.0 file.rxt       All light hits reflectance sphere wall first\n"
    s += "  iad -o out file.rxt       Calculated values in out\n"
    s += "  iad -r 0.3                R_total=0.\n"
    return s


def build_parser():
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="iad command line program",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=example_text(),
    )
    for spec in arg_specs:
        flags = spec["flags"]
        other_args = {k: v for k, v in spec.items() if k != "flags"}
        parser.add_argument(*flags, **other_args)
    return parser


def _resolve_scattering_constraint(constraint, wavelength):
    """Resolve a constant or power-law scattering constraint."""
    if constraint is None:
        return None

    if not (isinstance(constraint, tuple) and constraint and constraint[0] == "power"):
        return constraint

    if wavelength is None:
        raise argparse.ArgumentTypeError(
            "Commandline: power-law scattering (-F 'P ...') requires wavelength data or -L"
        )

    _, lambda0, mus0, gamma = constraint
    resolved = mus0 * np.power(np.asarray(wavelength, dtype=float) / lambda0, gamma)
    if np.ndim(resolved) == 0:
        return float(resolved)
    return resolved


def _slice_if_matching_length(value, mask):
    """Slice arrays that align with a wavelength mask; leave scalars unchanged."""
    if value is None or np.isscalar(value) or np.ndim(value) == 0:
        return value
    arr = np.asarray(value)
    if arr.shape[0] != mask.shape[0]:
        return value
    return arr[mask]


def _filter_experiment_by_wavelength(exp, limits):
    """Return a wavelength-filtered copy of an experiment."""
    if limits is None or exp.lambda0 is None or np.isscalar(exp.lambda0):
        return exp

    wave_min, wave_max = limits
    mask = (np.asarray(exp.lambda0) >= wave_min) & (np.asarray(exp.lambda0) <= wave_max)
    if not np.any(mask):
        raise argparse.ArgumentTypeError(f"Commandline: no wavelengths fall within {wave_min:g} to {wave_max:g}")

    filtered = copy.deepcopy(exp)
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
    ]
    for attr in exp_attrs:
        setattr(filtered, attr, _slice_if_matching_length(getattr(filtered, attr), mask))

    sample_attrs = ["d", "n", "n_above", "n_below", "d_above", "d_below", "b_above", "b_below"]
    for attr in sample_attrs:
        setattr(filtered.sample, attr, _slice_if_matching_length(getattr(filtered.sample, attr), mask))

    for sphere_name in ["r_sphere", "t_sphere"]:
        sphere = getattr(filtered, sphere_name)
        if sphere is None:
            continue
        sphere.r_wall = _slice_if_matching_length(sphere.r_wall, mask)
        sphere.r_std = _slice_if_matching_length(sphere.r_std, mask)

    return filtered


def _discover_mc_lost_binary():
    """Return a repo-local or PATH `mc_lost` binary if present."""
    repo_binary = os.path.join(os.path.dirname(os.path.dirname(__file__)), "iad", "mc_lost")
    if os.path.exists(repo_binary):
        return repo_binary
    src_binary = os.path.join(os.path.dirname(os.path.dirname(__file__)), "iad", "src", "mc_lost")
    if os.path.exists(src_binary):
        return src_binary
    return shutil.which("mc_lost")


def _point_value(value, index=None):
    """Return a scalar value for a single row."""
    if value is None or np.isscalar(value) or np.ndim(value) == 0:
        return value
    if index is None:
        arr = np.asarray(value)
        if arr.size == 1:
            return arr.reshape(-1)[0]
        return value
    return value[index]


def add_sample_constraints(exp, args):
    """Command-line constraints on sample."""
    if args.thickness is not None:
        exp.sample.d = args.thickness

    if args.slide_thickness is not None:
        exp.sample.d_above = args.slide_thickness
        exp.sample.d_below = args.slide_thickness

    if args.E is not None:
        exp.sample.b_above = args.E
        exp.sample.b_below = args.E

    if args.nslab is not None:
        exp.sample.n = args.nslab

    if args.nslide is not None:
        exp.sample.n_above = args.nslide
        exp.sample.n_below = args.nslide

    if args.G is not None:
        if args.nslide is None:
            raise argparse.ArgumentTypeError("Commandline: Cannot use -G without also specifing slide index with -N")
        if args.G == "0":
            exp.sample.n_above = 1
            exp.sample.n_below = 1
        elif args.G == "t":
            exp.sample.n_above = args.nslide
            exp.sample.n_below = 1
        elif args.G == "b":
            exp.sample.n_above = 1
            exp.sample.n_below = args.nslide
        elif args.G == "2":
            exp.sample.n_above = args.nslide
            exp.sample.n_below = args.nslide
        elif args.G == "n":
            exp.flip_sample = True
            exp.sample.n_above = args.nslide
            exp.sample.n_below = 1
        else:  # must be 'f'
            exp.flip_sample = True
            exp.sample.n_above = 1
            exp.sample.n_below = args.nslide

    if args.i is not None:
        if args.i < 0 or args.i > 90:
            raise argparse.ArgumentTypeError("Commandline: Bad argument to -i, value must be between 0 and 90")
        exp.sample.nu_0 = np.cos(np.radians(args.i))

    if args.q is not None:
        if args.q % 4:
            raise argparse.ArgumentTypeError("Commandline: Number of quadrature points must be a multiple of 4")
        if exp.sample.nu_0 != 1 and args.q % 12:
            raise argparse.ArgumentTypeError("Commandline: Quadrature must be 12, 24, 36,... for oblique incidence")
        exp.sample.quad_pts = args.q


def add_experiment_constraints(exp, args):
    """Command-line constraints on experiment."""

    def cli_sphere_to_object(values, refl):
        """Convert `-1`/`-2` CLI sphere values to a Sphere object."""
        sphere_d, sample_d, entrance_d, detector_d, wall_r = values
        return iadpython.Sphere(
            sphere_d,
            sample_d,
            d_third=entrance_d,
            d_detector=detector_d,
            r_std=1.0,
            r_wall=wall_r,
            refl=refl,
        )

    if args.S is not None:
        exp.num_spheres = args.S

    if args.r_sphere is not None:
        exp.r_sphere = cli_sphere_to_object(args.r_sphere, refl=True)
        if args.S is None:
            exp.num_spheres = 1

    if args.t_sphere is not None:
        exp.t_sphere = cli_sphere_to_object(args.t_sphere, refl=False)
        if args.S is None:
            exp.num_spheres = 2

    if getattr(args, "w", None) is not None and args.r_sphere is not None:
        raise argparse.ArgumentTypeError("Commandline: -w is overridden by -1 option. omit.")

    if getattr(args, "W", None) is not None and args.t_sphere is not None:
        raise argparse.ArgumentTypeError("Commandline: -W is overridden by -2 option. omit.")

    if args.diameter is not None:
        exp.d_beam = args.diameter

    if args.r is not None:
        exp.m_r = args.r

    if args.t is not None:
        exp.m_t = args.t

    if args.u is not None:
        exp.m_u = args.u

    rstd_r = getattr(args, "R", None)
    rstd_t = getattr(args, "T", None)
    if rstd_r is not None:
        if exp.r_sphere is not None:
            exp.r_sphere.r_std = rstd_r
        if exp.t_sphere is not None and rstd_t is None:
            exp.t_sphere.r_std = rstd_r

    if rstd_t is not None:
        if exp.t_sphere is not None:
            exp.t_sphere.r_std = rstd_t
        if exp.r_sphere is not None and rstd_r is None:
            exp.r_sphere.r_std = rstd_t

    if getattr(args, "w", None) is not None and exp.r_sphere is not None:
        exp.r_sphere.r_wall = args.w

    if getattr(args, "W", None) is not None and exp.t_sphere is not None:
        exp.t_sphere.r_wall = args.W

    if getattr(args, "H", None) is not None:
        r_baffle = args.H in (1, 3)
        t_baffle = args.H in (2, 3)
        if exp.r_sphere is not None:
            exp.r_sphere.baffle = r_baffle
        if exp.t_sphere is not None:
            exp.t_sphere.baffle = t_baffle

    if getattr(args, "L", None) is not None:
        exp.lambda0 = args.L


def add_analysis_constraints(exp, args):
    """Add command line constraints on analysis."""
    # constraints on analysis
    if args.albedo is not None:
        exp.default_a = args.albedo

    if args.b is not None:
        exp.default_b = args.b

    if args.g is not None:
        exp.default_g = args.g

    if args.mua is not None:
        exp.default_mua = args.mua
        exp.default_ba = args.mua * exp.sample.d

    if args.mus is not None:
        exp.default_mus = _resolve_scattering_constraint(args.mus, exp.lambda0)
        exp.default_bs = exp.default_mus * exp.sample.d

    if getattr(args, "musp", None) is not None:
        if exp.default_g is not None:
            exp.default_mus = args.musp / max(1.0 - exp.default_g, 1e-6)
        else:
            exp.default_mus = args.musp
        exp.default_bs = exp.default_mus * exp.sample.d

    if args.f_r is not None:
        exp.fraction_of_rc_in_mr = args.f_r

    if args.f_t is not None:
        exp.fraction_of_tc_in_mt = args.f_t

    if getattr(args, "f_wall", None) is not None:
        exp.f_r = args.f_wall

    if getattr(args, "error", None) is not None:
        exp.tolerance = args.error
        exp.MC_tolerance = args.error

    if args.M is not None:
        exp.max_mc_iterations = args.M

    if args.p is not None:
        exp.n_photons = args.p

    if getattr(args, "x", None) is not None:
        exp.debug_level = args.x

    if getattr(args, "s", None) is not None:
        exp.search_override = iadpython.SEARCH_CODE_TO_NAME[args.s]

    exp.verbosity = getattr(args, "V", 2)

    if getattr(args, "X", False):
        exp.method = "comparison"


def forward_calculation(exp):
    """Do a forward calculation."""
    # set albedo
    if exp.default_a is None:
        if exp.default_mus is None:
            exp.sample.a = 0
        elif exp.default_mua is None:
            exp.sample.a = 1
        else:
            exp.sample.a = exp.default_mus / (exp.default_mua + exp.default_mus)
    else:
        exp.sample.a = exp.default_a

    # set optical thickness
    if exp.default_b is None:
        if exp.sample.d is None:
            exp.sample.b = float("inf")
        elif exp.sample.a == 0:
            if exp.default_mua is None:
                exp.sample.b = float("inf")
            else:
                exp.sample.b = exp.default_mua * exp.sample.d
        elif exp.default_mus is None:
            exp.sample.b = float("inf")
        else:
            exp.sample.b = exp.default_mus / exp.sample.a * exp.sample.d
    else:
        exp.sample.b = exp.default_b

    # set scattering anisotropy
    if exp.default_g is None:
        exp.sample.g = 0
    else:
        exp.sample.g = exp.default_g

    print(exp.sample)

    ur1, ut1, _uru, _utu = exp.sample.rt()
    ru, tu = exp.sample.unscattered_rt()
    print("Calculated quantities")
    print("   R total         = %.3f" % ur1)
    print("   R scattered     = %.3f" % (ur1 - ru))
    print("   R unscattered   = %.3f" % ru)
    print("   T total         = %.3f" % ut1)
    print("   T scattered     = %.3f" % (ut1 - tu))
    print("   T unscattered   = %.3f" % tu)
    sys.exit(0)


def print_results_header(debug_lost_light=False):
    """Print the header for results to stdout."""
    print(
        "#     \tMeasured \t   M_R   \tMeasured \t   M_T   \tEstimated\tEstimated\tEstimated",
        end="",
    )
    if debug_lost_light:
        print(
            "\t  Lost   \t  Lost   \t  Lost   \t  Lost   \t   MC    \t   IAD   \t  Error  ",
            end="",
        )
    print()

    print(
        "##wave\t   M_R   \t   fit   \t   M_T   \t   fit   \t  mu_a   \t  mu_s'  \t    g    ",
        end="",
    )
    if debug_lost_light:
        print(
            "\t   UR1   \t   URU   \t   UT1   \t   UTU   \t    #    \t    #    \t  State  ",
            end="",
        )
    print()

    print(
        "# [nm]\t  [---]  \t  [---]  \t  [---]  \t  [---]  \t  1/mm   \t  1/mm   \t  [---]  ",
        end="",
    )
    if debug_lost_light:
        print(
            "\t  [---]  \t  [---]  \t  [---]  \t  [---]  \t  [---]  \t  [---]  \t  [---]  ",
            end="",
        )
    print()


def print_debug_data_point(exp):
    """Emit the CWEB-style data-point banner used by debug modes."""
    if not getattr(exp, "debug_level", 0):
        return

    print("\n-------------------NEXT DATA POINT---------------------", file=sys.stderr)
    if exp.lambda0 not in (None, 0):
        print(f"lambda={float(exp.lambda0):6.1f} ", end="", file=sys.stderr)
    m_r = 0.0 if exp.m_r is None else float(exp.m_r)
    m_t = 0.0 if exp.m_t is None else float(exp.m_t)
    print(f"MR={m_r:8.5f} MT={m_t:8.5f}\n", file=sys.stderr)


def print_debug_search_status(exp):
    """Emit a CWEB-style long search status for debug modes."""
    if getattr(exp, "found", False):
        print("Successful Search\n", file=sys.stderr)
    else:
        print("Failed Search, too many iterations\n", file=sys.stderr)


def _row_count(exp):
    """Return the number of scalar inversions represented by an experiment."""
    for value in [exp.lambda0, exp.m_r, exp.m_t, exp.m_u]:
        if value is not None and not np.isscalar(value) and np.ndim(value) > 0:
            return len(value)
    return 1


def _point_experiment(exp, index=None):
    """Create a scalar experiment for a single row of input data."""
    point = copy.deepcopy(exp)
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
    ]
    for attr in exp_attrs:
        setattr(point, attr, _point_value(getattr(exp, attr), index))

    sample_attrs = ["d", "n", "n_above", "n_below", "d_above", "d_below", "b_above", "b_below"]
    for attr in sample_attrs:
        setattr(point.sample, attr, _point_value(getattr(exp.sample, attr), index))

    for sphere_name in ["r_sphere", "t_sphere"]:
        sphere = getattr(point, sphere_name)
        source = getattr(exp, sphere_name)
        if sphere is None or source is None:
            continue
        sphere.r_wall = _point_value(source.r_wall, index)
        sphere.r_std = _point_value(source.r_std, index)

    point.grid = None
    point.include_measurements = False
    return point


def _grid_filename(args):
    """Return the output filename for `-J` grid generation."""
    if args.filename:
        root, _ext = os.path.splitext(args.filename)
        return root + ".grid"
    if args.out_fname:
        root, _ext = os.path.splitext(args.out_fname)
        return root + ".grid"
    return "iad.grid"


def _populate_grid_lost_light(exp):
    """Apply a direct MC lost-light estimate to a scalar grid point."""
    if exp.max_mc_iterations == 0 or exp.num_spheres == 0 or exp.mc_lost_path is None:
        exp.ur1_lost = 0
        exp.ut1_lost = 0
        exp.uru_lost = 0
        exp.utu_lost = 0
        return

    n_sample = float(exp.sample.n)
    n_slide = float(exp.sample.n_above) if exp.sample.n_above != 1.0 else 1.0
    d_port_r = float(exp.r_sphere.sample.d) if exp.r_sphere is not None else 1000.0
    d_port_t = float(exp.t_sphere.sample.d) if exp.t_sphere is not None else d_port_r

    exp.ur1_lost, exp.ut1_lost, exp.uru_lost, exp.utu_lost = iadpython.mc_lost.run_mc_lost(
        a=float(exp.sample.a),
        b=float(exp.sample.b),
        g=float(exp.sample.g),
        n_sample=n_sample,
        n_slide=n_slide,
        d_port_r=d_port_r,
        d_port_t=d_port_t,
        d_beam=float(exp.d_beam),
        t_sample=float(exp.sample.d),
        t_slide=float(exp.t_slide),
        n_photons=int(exp.n_photons),
        method=exp.method,
        binary_path=exp.mc_lost_path,
    )


def _write_grid_file(exp, args):
    """Write a CWEB-style `.grid` file for a scalar experiment."""
    grid_name = _grid_filename(args)
    aa = [0, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0]
    bb = [0, 0.2, 0.5, 1.0, 3.0, 10.0, 100]
    g_value = exp.default_g if exp.default_g is not None else exp.sample.g

    with open(grid_name, "w", encoding="utf-8") as grid:
        print(f"# g={g_value:6.4f}", file=grid)
        print("#    a'          b'          g           M_R         M_T", file=grid)
        for aprime in aa:
            for bprime in bb:
                point = copy.deepcopy(exp)
                point.ur1_lost = 0
                point.ut1_lost = 0
                point.uru_lost = 0
                point.utu_lost = 0
                point.sample.g = g_value
                point.sample.a = aprime / max(1 - g_value + aprime * g_value, 1e-12)
                point.sample.b = bprime / max(1 - point.sample.a * g_value, 1e-12)
                _populate_grid_lost_light(point)
                m_r, m_t = point.measured_rt()
                print(
                    f"{aprime:10.5f}, {bprime:10.5f}, {g_value:10.5f}, {m_r:10.5f}, {m_t:10.5f}",
                    file=grid,
                )


def print_result_row(exp, line=0, debug_lost_light=False):
    """Print one scalar result row."""
    m_r_fit, m_t_fit = exp.measured_rt()
    wavelength = exp.lambda0
    mu_a = min(float(exp.sample.mu_a()), 199.9999)
    mu_sp = min(float(exp.sample.mu_sp()), 999.9999)
    m_r = 0.0 if exp.m_r is None else float(exp.m_r)
    m_t = 0.0 if exp.m_t is None else float(exp.m_t)

    if wavelength not in (None, 0):
        print(f"{float(wavelength):6.1f}", end="\t")
    else:
        print(f"{line:6d}", end="\t")

    print(f"{m_r: 9.4f}\t{m_r_fit: 9.4f}\t{m_t: 9.4f}\t{m_t_fit: 9.4f}\t", end="")
    print(f"{mu_a: 9.4f}\t{mu_sp: 9.4f}\t{float(exp.sample.g): 9.4f}", end="")

    if debug_lost_light:
        print(
            f"\t{exp.ur1_lost: 8.4f}\t{exp.uru_lost: 8.4f}\t{exp.ut1_lost: 8.4f}\t{exp.utu_lost: 8.4f}"
            f"\t{exp._mc_iterations: 3d}\t{exp.iterations: 3d}\t  {what_char(InputError.NO_ERROR)}",  # pylint: disable=protected-access
            end="",
        )
    print()


def invert_file(exp, args):
    """Process an entire .rxt file."""
    # determine output file name
    if args.out_fname is None:
        root, ext = os.path.splitext(args.filename)
        if ext == ".rxt":
            args.out_fname = root + ".txt"
        else:
            args.out_fname = args.filename + ".txt"

    exp = _filter_experiment_by_wavelength(exp, getattr(args, "l", None))
    n_rows = _row_count(exp)
    debug_lost_light = bool(exp.debug_level & iadpython.DEBUG_LOST_LIGHT)
    last_point = None

    with open(args.out_fname, "w", encoding="utf-8") as output_stream, redirect_stdout(output_stream):
        if exp.verbosity > 0 and not exp.debug_level:
            print_results_header(debug_lost_light=debug_lost_light)

        for i in range(n_rows):
            point = _point_experiment(exp, i)
            print_debug_data_point(point)
            a, b, g = point.invert_rt()
            point.sample.a = a
            point.sample.b = b
            point.sample.g = g
            if not point.debug_level:
                print_result_row(point, line=i, debug_lost_light=debug_lost_light)
            else:
                print_debug_search_status(point)
            last_point = point

        if getattr(args, "J", False) and last_point is not None:
            _write_grid_file(last_point, args)
    sys.exit(0)


def main():
    """Main command-line interface."""
    parser = build_parser()

    try:
        args = parser.parse_args()

        if args.v:
            print_version(args.V)
            sys.exit(0)

        # If there is a file then read it, otherwise create a blank experiment
        if args.filename:
            exp = iadpython.read_rxt(args.filename)
        else:
            exp = iadpython.Experiment()

        # update the search to include the command line constraints
        add_sample_constraints(exp, args)
        add_experiment_constraints(exp, args)
        add_analysis_constraints(exp, args)

        # follow iad behavior: sphere measurements default to substitution mode
        if np.isscalar(exp.num_spheres) and exp.num_spheres > 0 and getattr(exp, "method", "unknown") == "unknown":
            exp.method = "substitution"

        if np.isscalar(exp.num_spheres) and exp.num_spheres > 0 and exp.max_mc_iterations > 0:
            exp.mc_lost_path = _discover_mc_lost_binary()
            if exp.mc_lost_path is None:
                raise RuntimeError("mc_lost binary not found. Build it with: cd iad && make mc_lost")

        if getattr(exp, "method", "unknown") in ("comparison", 1):
            if np.isscalar(exp.num_spheres) and exp.num_spheres == 2:
                raise argparse.ArgumentTypeError(
                    "Commandline: Dual beam (-X) cannot be used with double integrating spheres."
                )
            if getattr(exp, "f_r", 0.0) != 0:
                raise argparse.ArgumentTypeError("Commandline: Dual beam (-X) cannot be used when -f is non-zero.")

        if args.z:
            forward_calculation(exp)

        if args.filename:
            invert_file(exp, args)

        if (exp.m_r is None) and (exp.m_t is None) and (exp.m_u is None):
            raise argparse.ArgumentTypeError("Commandline: One measurement needed or use '-z' for forward calc.")

        point = _point_experiment(_filter_experiment_by_wavelength(exp, getattr(args, "l", None)))
        print_debug_data_point(point)
        a, b, g = point.invert_rt()
        point.sample.a = a
        point.sample.b = b
        point.sample.g = g
        debug_lost_light = bool(point.debug_level & iadpython.DEBUG_LOST_LIGHT)
        if point.verbosity > 0 and not point.debug_level:
            print_results_header(debug_lost_light=debug_lost_light)
        if point.debug_level:
            print_debug_search_status(point)
        else:
            print_result_row(point, debug_lost_light=debug_lost_light)
        if getattr(args, "J", False):
            _write_grid_file(point, args)
        sys.exit(0)

    except (argparse.ArgumentTypeError, OSError, ValueError, RuntimeError, TypeError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
