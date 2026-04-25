#!/usr/bin/env python3
"""Command line for support for iadpython."""

import os
import copy
import shutil
import sys
import datetime
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

    if final:
        print(what_char(err), end="", file=sys.stderr)
    else:
        COUNTER -= 1
        print(f"{points % 10}\b", end="", file=sys.stderr)

    if final:
        if COUNTER % 50 == 0:
            rate = (time.time() - start_time) / COUNTER
            print(f"  {COUNTER:3d} done ({rate:.2f} s/pt)", file=sys.stderr)
        elif COUNTER % 10 == 0:
            print(" ", end="", file=sys.stderr)

    sys.stderr.flush()


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
        "nargs": "*",
        "type": str,
        "default": [],
        "help": "Input filename(s)",
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
    s = "Examples:\n"
    s += "  iadp file.rxt              Results will be put in file.txt\n"
    s += "  iadp file                  Same as above\n"
    s += "  iadp -c 0.9 file.rxt       Assume M_R includes 90% of unscattered reflectance\n"
    s += "  iadp -C 0.8 file.rxt       Assume M_T includes 80% of unscattered transmittance\n"
    s += "  iadp -e 0.0001 file.rxt    Better convergence to R & T values\n"
    s += "  iadp -f 1.0 file.rxt       All light hits reflectance sphere wall first\n"
    s += "  iadp -l '500 600' file.rxt Only do wavelengths between 500 and 600\n"
    s += "  iadp -o out file.rxt       Calculated values in out\n"
    s += "  iadp -r 0.3                R_total=0.3, b=inf, find albedo\n"
    s += "  iadp -r 0.3 -t 0.4         R_total=0.3, T_total=0.4, find a,b,g\n"
    s += "  iadp -r 0.3 -t 0.4 -n 1.5  R_total=0.3, T_total=0.4, n=1.5, find a,b\n"
    s += "  iadp -p 1000 file.rxt      Only 1000 photons\n"
    s += "  iadp -p -100 file.rxt      Allow only 100ms per iteration\n"
    s += "  iadp -q 4 file.rxt         Four quadrature points\n"
    s += "  iadp -M 0 file.rxt         No MC    (iad)\n"
    s += "  iadp -M 1 file.rxt         MC once  (iad -> MC -> iad)\n"
    s += "  iadp -M 2 file.rxt         MC twice (iad -> MC -> iad -> MC -> iad)\n"
    s += "  iadp -M 0 -q 4 file.rxt    Fast and crude conversion\n"
    s += "  iadp -G t file.rxt         One top slide with properties from file.rxt\n"
    s += "  iadp -G b -N 1.5 -D 1 file Use 1 bottom slide with n=1.5 and thickness=1\n"
    s += "  iadp -x   1 file.rxt       Show sphere and MC effects\n"
    s += "  iadp -x   2 file.rxt       Show grid decisions\n"
    s += "  iadp -x   4 file.rxt       Show iterations\n"
    s += "  iadp -x   8 file.rxt       Show lost light effects\n"
    s += "  iadp -x  16 file.rxt       Show best grid points\n"
    s += "  iadp -x  32 file.rxt       Show decisions for type of search\n"
    s += "  iadp -x  64 file.rxt       Show all grid calculations\n"
    s += "  iadp -x 128 file.rxt       Show sphere calculations\n"
    s += "  iadp -x 511 file.rxt       Show all debugging output\n"
    s += "  iadp -X -i 8 file.rxt      Dual beam spectrometer with 8 degree incidence\n"
    s += "\n"
    s += "  iadp -z -a 0.9 -b 1 -i 45  Forward calc assuming 45 degree incidence\n"
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


def _input_filenames(args):
    """Return positional input filenames, preserving old string-style args."""
    filenames = args.filename
    if filenames is None:
        return []
    if isinstance(filenames, str):
        return [filenames]
    return list(filenames)


def _resolve_input_filename(filename):
    """Return an input file to process, or None when the argument should be ignored."""
    _root, ext = os.path.splitext(filename)
    if os.path.isfile(filename):
        if ext.lower() in ("", ".rxt"):
            return filename
        return None

    candidate = filename + ".rxt"
    if os.path.isfile(candidate):
        return candidate

    return None


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
        "num_spheres",
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


def _header_scalar(value, default=0.0):
    """Return a scalar value suitable for one-line output headers."""
    if value is None:
        return default
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return array.flat[0].item()


def _input_column_labels(exp):
    """Return compact `.rxt` input column labels, when available."""
    return getattr(exp, "input_column_labels", "") or ""


def _format_header_value(exp, label, fmt, value):
    """Format a header value or mark it as row-varying like CWEB iad."""
    if label and label in _input_column_labels(exp):
        return " (varies with input row)"
    return fmt % _header_scalar(value)


def _method_phrase(exp):
    """Return the CWEB wording for the measurement method."""
    method = getattr(exp, "method", "unknown")
    if method in ("substitution", 2):
        return " using the substitution (single-beam) method."
    if method in ("comparison", 1):
        return " using the comparison (dual-beam) method."
    return " using an unknown method."


def _sphere_attr(sphere, dotted_attr, default=0.0):
    """Return a possibly nested sphere attribute for header printing."""
    if sphere is None:
        return default
    value = sphere
    for attr in dotted_attr.split("."):
        value = getattr(value, attr)
    return value


def _print_sphere_header(exp, sphere_name, title, third_label):
    """Print one integrating-sphere block in the legacy output header."""
    sphere = getattr(exp, sphere_name)
    baffle = bool(_sphere_attr(sphere, "baffle", default=False))
    baffle_text = "has a baffle" if baffle else "has no baffle"
    ignored = " (ignored since no spheres used)" if int(_header_scalar(exp.num_spheres)) == 0 else ""
    print(f"# {title} sphere {baffle_text} between sample and detector{ignored}")
    print(f"#                      sphere diameter = {_header_scalar(_sphere_attr(sphere, 'd')):7.1f} mm")
    print(f"#                 sample port diameter = {_header_scalar(_sphere_attr(sphere, 'sample.d')):7.1f} mm")
    third_port_d = _header_scalar(_sphere_attr(sphere, "third.d"))
    if third_label == "entrance port diameter":
        print(f"#               entrance port diameter = {third_port_d:7.1f} mm")
    else:
        print(f"#                  third port diameter = {third_port_d:7.1f} mm")
    print(f"#               detector port diameter = {_header_scalar(_sphere_attr(sphere, 'detector.d')):7.1f} mm")
    detector_reflectance = _header_scalar(_sphere_attr(sphere, "detector.uru")) * 100
    wall_reflectance = _format_header_value(exp, "w", "%7.1f %%", _sphere_attr(sphere, "r_wall") * 100)
    print(f"#                 detector reflectance = {detector_reflectance:7.1f} %")
    print(f"#                     wall reflectance = {wall_reflectance}")
    if sphere_name == "r_sphere":
        standard = _format_header_value(exp, "R", "%7.1f %%", _sphere_attr(sphere, "r_std") * 100)
    else:
        standard = "%7.1f %%" % (_header_scalar(_sphere_attr(sphere, "r_std")) * 100)
    print(f"#                 calibration standard = {standard}")
    print("#")


def _fixed_input_description(exp):
    """Return the CWEB wording for fixed-column input files."""
    params = int(_header_scalar(getattr(exp, "num_measures", 0)))
    descriptions = {
        -1: "No M_R or M_T -- forward calculation",
        1: "Just M_R was measured",
        2: "M_R and M_T were measured",
        3: "M_R, M_T, and M_U were measured",
        4: "M_R, M_T, M_U, and r_w were measured",
        5: "M_R, M_T, M_U, r_w, and t_w were measured",
        6: "M_R, M_T, M_U, r_w, t_w, and r_std were measured",
        7: "M_R, M_T, M_U, r_w, t_w, r_std and t_std were measured",
    }
    return descriptions.get(params, "Something went wrong ... measures should be 1 to 7!")


def _print_input_description(exp):
    """Print the legacy input-column and measurement-method line."""
    labels = _input_column_labels(exp)
    if labels:
        label_text = "".join(f" {label} " for label in labels)
        print(f"# {len(labels)} input columns with LABELS:{label_text}{_method_phrase(exp)}")
    else:
        print(f"# {_fixed_input_description(exp)}{_method_phrase(exp)}")


def _print_sphere_correction_description(exp):
    """Print the legacy sphere-correction and incidence-angle line."""
    num_spheres = int(_header_scalar(exp.num_spheres))
    method = getattr(exp, "method", "unknown")
    if num_spheres == 0:
        description = "No sphere corrections were used"
    elif num_spheres == 1 and method in ("comparison", 1):
        description = "No sphere corrections were needed"
    elif num_spheres == 1:
        description = "Single sphere corrections were used"
    else:
        description = "Double sphere corrections were used"

    nu_0 = float(_header_scalar(getattr(exp.sample, "nu_0", 1.0), default=1.0))
    angle = int(np.degrees(np.arccos(np.clip(nu_0, -1.0, 1.0))))
    print(f"# {description} and light was incident at {angle} degrees from the normal.")


def _print_search_description(exp):
    """Print the active inverse-search description for the legacy header."""
    search = getattr(exp, "search_override", None)
    if search in (None, "auto"):
        search = getattr(exp, "search", "unknown")

    if search == "find_ab":
        print("# The inverse routine varied the albedo and optical depth.")
        print("# ")
        default_g = _header_scalar(exp.default_g, default=0.0)
        print(f"# Default single scattering anisotropy = {default_g:7.3f} ")
    elif search == "find_ag":
        print("# The inverse routine varied the albedo and anisotropy.")
        print("# ")
        if exp.default_b is not None:
            print(f"#                     Default (mu_t*d) = {_header_scalar(exp.default_b):7.3g}")
        else:
            print("# ")
    elif search == "find_a":
        print("# The inverse routine varied only the albedo.")
        print("# ")
        default_g = _header_scalar(exp.default_g, default=0.0)
        default_b = _header_scalar(exp.default_b, default=np.inf)
        print(f"# Default single scattering anisotropy is {default_g:7.3f}  and (mu_t*d) = {default_b:7.3g}")
    elif search in ("find_b", "find_b_no_absorption", "find_b_no_scattering"):
        print("# The inverse routine varied only the optical depth.")
        print("# ")
        default_g = _header_scalar(exp.default_g, default=0.0)
        if exp.default_a is not None:
            default_a = _header_scalar(exp.default_a)
            print(f"# Default single scattering anisotropy is {default_g:7.3f} and default albedo = {default_a:7.3g}")
        else:
            print(f"# Default single scattering anisotropy is {default_g:7.3f} ")
    elif search == "find_ba":
        print("# The inverse routine varied only the absorption.")
        print("# ")
        print(f"#                     Default (mu_s*d) = {_header_scalar(exp.default_bs, default=0.0):7.3g}")
    elif search == "find_bs":
        print("# The inverse routine varied only the scattering.")
        print("# ")
        print(f"#                     Default (mu_a*d) = {_header_scalar(exp.default_ba, default=0.0):7.3g}")
    else:
        print("# The inverse routine adapted to the input data.")
        print("# ")
        print("# ")


def _version_string():
    """Return version string for the file header.

    During development (version contains 'dev'), append the current date/time
    so successive runs can be distinguished.  For release versions use the
    plain version number, matching the iad/iad convention from version.pl.
    """
    ver = iadpython.__version__
    if "dev" in ver:
        stamp = datetime.datetime.now().strftime("%H:%M on %-d %b %Y")
        return f"{ver} ({stamp})"
    return ver


def print_file_header(exp, command_line=None):
    """Print the CWEB-style metadata header for result files."""
    command_line = command_line or " ".join(sys.argv)
    print(f"# Inverse Adding-Doubling iadp {_version_string()} ")
    print(f"# {command_line} ")
    print(f"#                        Beam diameter = {_format_header_value(exp, 'B', '%7.1f mm', exp.d_beam)}")
    print(f"#                     Sample thickness = {_format_header_value(exp, 'd', '%7.3f mm', exp.sample.d)}")
    print(f"#                  Top slide thickness = {_format_header_value(exp, 'D', '%7.3f mm', exp.sample.d_above)}")
    print(f"#               Bottom slide thickness = {_format_header_value(exp, 'D', '%7.3f mm', exp.sample.d_below)}")
    print(f"#           Sample index of refraction = {_format_header_value(exp, 'n', '%7.4f mm', exp.sample.n)}")
    print(f"#        Top slide index of refraction = {_format_header_value(exp, 'N', '%7.4f mm', exp.sample.n_above)}")
    print(f"#     Bottom slide index of refraction = {_format_header_value(exp, 'N', '%7.4f mm', exp.sample.n_below)}")
    print("# ")
    print(
        "#  Percentage unscattered refl. in M_R = "
        f"{_format_header_value(exp, 'c', '%7.1f %%', exp.fraction_of_rc_in_mr * 100)}"
    )
    print(
        "# Percentage unscattered trans. in M_T = "
        f"{_format_header_value(exp, 'C', '%7.1f %%', exp.fraction_of_tc_in_mt * 100)}"
    )
    print("# ")
    _print_sphere_header(exp, "r_sphere", "Reflection", "entrance port diameter")
    _print_sphere_header(exp, "t_sphere", "Transmission", "third port diameter")
    _print_input_description(exp)
    _print_sphere_correction_description(exp)
    _print_search_description(exp)
    print(f"#                 AD quadrature points = {int(_header_scalar(exp.sample.quad_pts)):3d}")
    print(f"#             AD tolerance for success = {float(_header_scalar(exp.tolerance)):9.5f}")
    print(f"#      MC tolerance for mu_a and mu_s' = {float(_header_scalar(exp.MC_tolerance)):7.3f} %")
    photons = int(_header_scalar(exp.n_photons)) if int(_header_scalar(exp.max_mc_iterations)) > 0 else 0
    print(f"#  Photons used to estimate lost light =   {photons:d}")
    print("#")


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


def _minimum_mr_mt(exp):
    """Return the CWEB minimum possible `M_R/M_T` estimate for a point."""
    original_abg = (exp.sample.a, exp.sample.b, exp.sample.g)
    original_debug = exp.debug_level
    try:
        exp.sample.a = 0.0
        if exp.default_b is not None:
            exp.sample.b = exp.default_b
        elif exp.m_u is not None and exp.m_u > 0:
            exp.sample.b = exp.what_is_b()
        else:
            exp.sample.b = np.inf
        exp.sample.g = 0.0 if exp.default_g is None else exp.default_g
        exp.debug_level &= ~iadpython.DEBUG_EVERY_CALC
        ur1, ut1, uru, utu = exp.sample.rt()
        return exp.measured_rt_from_raw(ur1, ut1, uru, utu, include_lost=False)
    finally:
        exp.sample.a, exp.sample.b, exp.sample.g = original_abg
        exp.debug_level = original_debug


def _debug_input_error(exp):
    """Approximate CWEB `measure_OK()` for long debug status output."""
    m_r = 0.0 if exp.m_r is None else float(exp.m_r)
    m_t = 0.0 if exp.m_t is None else float(exp.m_t)
    m_u = 0.0 if exp.m_u is None else float(exp.m_u)

    if m_t < 0:
        return InputError.MT_TOO_SMALL
    if m_t > 1:
        return InputError.MT_TOO_BIG
    if m_r < 0:
        return InputError.MR_TOO_SMALL
    if m_r > 1:
        return InputError.MR_TOO_BIG

    try:
        min_mr, _min_mt = _minimum_mr_mt(exp)
    except (ValueError, FloatingPointError, OverflowError):
        min_mr = None
    if min_mr is not None and m_r < float(min_mr) - 1e-8:
        return InputError.MR_TOO_SMALL

    if m_u < 0:
        return InputError.MU_TOO_SMALL
    if 0 < m_t < m_u:
        return InputError.MU_TOO_BIG
    if not getattr(exp, "found", False):
        return InputError.TOO_MANY_ITERATIONS
    return InputError.NO_ERROR


def _print_long_error(err):
    """Emit CWEB-style long search status text."""
    if err == InputError.TOO_MANY_ITERATIONS:
        print("Failed Search, too many iterations", file=sys.stderr)
    if err == InputError.MR_TOO_BIG:
        print("Failed Search, M_R is too big", file=sys.stderr)
    if err == InputError.MR_TOO_SMALL:
        print("Failed Search, M_R is too small", file=sys.stderr)
    if err == InputError.MT_TOO_BIG:
        print("Failed Search, M_T is too big", file=sys.stderr)
    if err == InputError.MT_TOO_SMALL:
        print("Failed Search, M_T is too small", file=sys.stderr)
    if err == InputError.MU_TOO_BIG:
        print("Failed Search, M_U is too big", file=sys.stderr)
    if err == InputError.MU_TOO_SMALL:
        print("Failed Search, M_U is too snall", file=sys.stderr)
    if err == InputError.TOO_MUCH_LIGHT:
        print("Failed Search, Total light bigger than 1", file=sys.stderr)
    if err == InputError.NO_ERROR:
        print("Successful Search", file=sys.stderr)
    print(file=sys.stderr)


def print_debug_search_status(exp):
    """Emit a CWEB-style long search status for debug modes."""
    err = _result_input_error(exp)
    _print_long_error(err)


def _result_input_error(exp):
    """Return and store the one-character result status for an inversion."""
    if getattr(exp, "debug_level", 0) and not exp.debug_level & iadpython.DEBUG_A_LITTLE:
        err = InputError.NO_ERROR if getattr(exp, "found", False) else InputError.TOO_MANY_ITERATIONS
    else:
        err = _debug_input_error(exp)
    exp.error = err.value
    return err


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
        "num_spheres",
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

    point.grid = exp.grid
    point.include_measurements = False
    return point


def _grid_filename(args):
    """Return the output filename for `-J` grid generation."""
    filenames = _input_filenames(args)
    if filenames:
        root, _ext = os.path.splitext(filenames[0])
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
    t_slide_val = float(exp.sample.d_above) if exp.sample.d_above is not None else float(exp.t_slide)
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
        t_slide=t_slide_val,
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


def print_result_row(exp, line=0, debug_lost_light=False, err=None):
    """Print one scalar result row."""
    if err is None:
        err = _result_input_error(exp)
    original_debug = getattr(exp, "debug_level", 0)
    try:
        exp.debug_level = 0
        m_r_fit, m_t_fit = exp.measured_rt()
    finally:
        exp.debug_level = original_debug
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
            f"\t{exp._mc_iterations: 3d}\t{exp.iterations: 3d}\t  {what_char(err)}",  # pylint: disable=protected-access
            end="",
        )
    else:
        print(f"\t {what_char(err)} ", end="")
    print()


def invert_file(exp, args):
    """Process an entire .rxt file."""
    # determine output file name
    if args.out_fname is None:
        filename = _input_filenames(args)[0]
        root, ext = os.path.splitext(filename)
        if ext == ".rxt":
            args.out_fname = root + ".txt"
        else:
            args.out_fname = filename + ".txt"

    exp = _filter_experiment_by_wavelength(exp, getattr(args, "l", None))
    n_rows = _row_count(exp)
    debug_lost_light = bool(exp.debug_level & iadpython.DEBUG_LOST_LIGHT)
    last_point = None
    start_time = time.time()
    global COUNTER  # pylint: disable=global-statement
    global ANY_ERROR  # pylint: disable=global-statement
    COUNTER = 0
    ANY_ERROR = False

    with open(args.out_fname, "w", encoding="utf-8") as output_stream, redirect_stdout(output_stream):
        if exp.verbosity > 0:
            print_file_header(exp)
            print_results_header(debug_lost_light=debug_lost_light)

        for i in range(n_rows):
            point = _point_experiment(exp, i)
            print_debug_data_point(point)
            a, b, g = point.invert_rt()
            exp.grid = point.grid
            point.sample.a = a
            point.sample.b = b
            point.sample.g = g
            err = _result_input_error(point)
            print_result_row(point, line=i + 1, debug_lost_light=debug_lost_light, err=err)
            if point.debug_level:
                print_debug_search_status(point)
            else:
                print_dot(start_time, err, n_rows, True, point.verbosity)
            last_point = point

        if getattr(args, "J", False) and last_point is not None:
            _write_grid_file(last_point, args)


def main():
    """Main command-line interface."""
    parser = build_parser()

    try:
        args = parser.parse_args()
        filenames = _input_filenames(args)

        if args.v:
            print_version(args.V)
            sys.exit(0)

        # If there is a file then read it, otherwise create a blank experiment
        if filenames:
            resolved_filenames = []
            for filename in filenames:
                resolved = _resolve_input_filename(filename)
                if resolved is not None:
                    resolved_filenames.append(resolved)
                    continue

                _root, ext = os.path.splitext(filename)
                if os.path.exists(filename) and ext != ".rxt":
                    continue

                raise argparse.ArgumentTypeError(f"Commandline: could not find {filename} or {filename}.rxt")

            if not resolved_filenames:
                sys.exit(0)

            if args.out_fname is not None and len(resolved_filenames) > 1:
                raise argparse.ArgumentTypeError("Commandline: -o cannot be used with multiple input files")

            for filename in resolved_filenames:
                file_args = copy.copy(args)
                file_args.filename = [filename]
                exp = iadpython.read_rxt(filename)

                # update the search to include the command line constraints
                add_sample_constraints(exp, file_args)
                add_experiment_constraints(exp, file_args)
                add_analysis_constraints(exp, file_args)

                # follow iad behavior: sphere measurements default to substitution mode
                if (
                    np.isscalar(exp.num_spheres)
                    and exp.num_spheres > 0
                    and getattr(exp, "method", "unknown") == "unknown"
                ):
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
                        raise argparse.ArgumentTypeError(
                            "Commandline: Dual beam (-X) cannot be used when -f is non-zero."
                        )

                if file_args.z:
                    forward_calculation(exp)

                invert_file(exp, file_args)

            sys.exit(0)

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

        if (exp.m_r is None) and (exp.m_t is None) and (exp.m_u is None):
            raise argparse.ArgumentTypeError("Commandline: One measurement needed or use '-z' for forward calc.")

        point = _point_experiment(_filter_experiment_by_wavelength(exp, getattr(args, "l", None)))
        print_debug_data_point(point)
        a, b, g = point.invert_rt()
        point.sample.a = a
        point.sample.b = b
        point.sample.g = g
        debug_lost_light = bool(point.debug_level & iadpython.DEBUG_LOST_LIGHT)
        if point.verbosity > 0:
            print_results_header(debug_lost_light=debug_lost_light)
        err = _result_input_error(point)
        print_result_row(point, debug_lost_light=debug_lost_light, err=err)
        if point.debug_level:
            print_debug_search_status(point)
        if getattr(args, "J", False):
            _write_grid_file(point, args)
        sys.exit(0)

    except (argparse.ArgumentTypeError, OSError, ValueError, RuntimeError, TypeError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
