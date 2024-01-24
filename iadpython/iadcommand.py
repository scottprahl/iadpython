#!/usr/bin/env python3

# pylint: disable=line-too-long
# pylint: disable=too-many-arguments
# pylint: disable=too-many-branches
# pylint: disable=too-many-return-statements
# pylint: disable=global-statement

import sys
from enum import Enum
import argparse
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
        return '*'
    if err == InputError.TOO_MANY_ITERATIONS:
        return '+'
    if err == InputError.MR_TOO_BIG:
        return 'R'
    if err == InputError.MR_TOO_SMALL:
        return 'r'
    if err == InputError.MT_TOO_BIG:
        return 'T'
    if err == InputError.MT_TOO_SMALL:
        return 't'
    if err == InputError.MU_TOO_BIG:
        return 'U'
    if err == InputError.MU_TOO_SMALL:
        return 'u'
    if err == InputError.TOO_MUCH_LIGHT:
        return '!'
    return '?'

def print_dot(start_time, err, points, final, verbosity):
    """Print a character for each datapoint during analysis."""
    global COUNTER
    global ANY_ERROR
    COUNTER += 1

    if err != InputError.NO_ERROR:
        ANY_ERROR = err

    if verbosity == 0:
        return

    if final == 99:
        print(what_char(err), end='')
    else:
        COUNTER -= 1
        print(f"{final % 10}\b", end='')

    if final == 99:
        if COUNTER % 50 == 0:
            rate = (time.time() - start_time) / points
            print(f"  {points} done ({rate:.2f} s/pt)")
        elif COUNTER % 10 == 0:
            print(" ", end='')

    sys.stdout.flush()


def validator_01(value):
    """Is value between 0 and 1."""
    fvalue = float(value)
    if not 0 <= fvalue <= 1:
        raise ValueError(argparse.ArgumentTypeError(f"{value} is not between 0 and 1"))
    return fvalue

def validator_11(value):
    """Is value between -1 and 1."""
    fvalue = float(value)
    if not -1 <= fvalue <= 1:
        raise ValueError(argparse.ArgumentTypeError(f"{value} is not between 0 and 1"))
    return fvalue

def validator_positive(value):
    """Is value non-negative."""
    fvalue = float(value)
    if fvalue < 0:
        raise ValueError(argparse.ArgumentTypeError(f"{value} is not positive"))
    return fvalue

# Argument specifications
arg_specs = [
    {"flags": ["-1"], "dest": "r_sphere", "nargs": 5, "type": float, "help": "Five numbers separated by spaces"},
    {"flags": ["-2"], "dest": "t_sphere", "nargs": 5, "type": float, "help": "Five numbers separated by spaces"},
    {"flags": ["-a", "--albedo"], "dest": "albedo", "type": validator_01, "help": "Use this albedo"},
    {"flags": ["-A", "--mua"], "dest": "mua", "type": validator_positive, "help": "Use this absorption coefficient"},
    {"flags": ["-b"], "dest": "b", "type": validator_positive, "help": "Use this optical thickness"},
    {"flags": ["-B", "--diameter"], "dest": "diameter", "type": validator_positive, "help": "Beam diameter"},
    {"flags": ["-c"], "dest": "f_r", "type": validator_01, "help": "Fraction of unscattered refl in MR"},
    {"flags": ["-C"], "dest": "f_t", "type": validator_01, "help": "Fraction of unscattered trans in MT"},
    {"flags": ["-d"], "dest": "thickness", "type": validator_positive, "help": "Thickness of sample"},
    {"flags": ["-D"], "dest": "slide_thickness", "type": validator_positive, "help": "Thickness of slide"},
    {"flags": ["-e"], "dest": "error", "type": float, "default": 0.0001, "help": "Error tolerance (default 0.0001)"},
    {"flags": ["-E"], "dest": "E", "type": validator_positive, "help": "Optical depth (=mua*D) for slides"},
    {"flags": ["-f"], "dest": "f_wall", "type": validator_01, "help": "Fraction 0.0-1.0 of light to hit sphere wall first"},
    {"flags": ["-F", "--mus"], "dest": "mus", "type": validator_positive, "help": "Use this scattering coefficient"},
    {"flags": ["-g"], "type": validator_11, "default": 0, "help": "Scattering anisotropy (default 0)"},
    {"flags": ["-G"], "type": str, "choices": ['0', '2', 't', 'b', 'n', 'f'], "help": "Type of boundary "},
    {"flags": ["-i"], "type": float, "help": "Light incident at this angle in degrees"},
    {"flags": ["-M"], "type": int, "help": "Number of Monte Carlo iterations"},
    {"flags": ["-n"], "type": validator_positive, "help": "Index of refraction of slab"},
    {"flags": ["-N"], "type": validator_positive, "help": "Index of refraction of slides"},
    {"flags": ["-o"], "type": str, "help": "Filename for output"},
    {"flags": ["-p"], "type": int, "help": "# of Monte Carlo photons (default 100000)"},
    {"flags": ["-q"], "type": int, "default": 8, "help": "Number of quadrature points (default=8)"},
    {"flags": ["-r"], "type": validator_01, "help": "Total reflection measurement"},
    {"flags": ["-R"], "type": validator_01, "help": "Actual reflectance for 100%% measurement"},
    {"flags": ["-S"], "type": int, "choices": [0, 1, 2], "help": "Number of spheres used"},
    {"flags": ["-t"], "type": validator_01, "help": "Total transmission measurement"},
    {"flags": ["-T"], "type": validator_01, "help": "Actual transmission for 100%% measurement"},
    {"flags": ["-u"], "type": validator_01, "help": "Unscattered transmission measurement"},
    {"flags": ["-v", "--version"], "action": 'version', 'version': "iadp " + iadpython.__version__, "help": "short version"},
    {"flags": ["-V"], "action": "store_true", "help": "long version"},
    {"flags": ["--verbosity"], "type": int, "choices": [0, 1, 2], "help": "Verbosity level"},
    {"flags": ["-x"], "type": int, "help": "Set debugging level"},
    {"flags": ["-X"], "action": "store_true", "help": "Dual beam configuration"},
    {"flags": ["-z"], "action": "store_true", "help": "Do forward calculation"},
    {"flags": ["filename"], "nargs": "?", "type": str, "default": None, "help": "Input filename"}
]

def print_long_version():
    """Print the version information and quit."""
    s = ''
    s += "    version: " + iadpython.__version__ + "\n"
    s += "    Author: " + iadpython.__author__ + "\n"
    s += "    Copyright: " + iadpython.__copyright__ + "\n"
    s += "    License: " + iadpython.__license__ + "\n"
    s += "    URL:" + iadpython.__url__ + "\n"
    s += "\n    Forward and inverse adding-doubling radiative transport calculations.\n"
    s += "    Extensive documentation is at <https://iadpython.readthedocs.io>\n"
    print(s)
    sys.exit(0)
    

def example_text():
    """return a string with some command-line examples."""
    s = ''
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

    if args.n is not None:
        exp.sample.n = args.n

    if args.N is not None:
        exp.sample.n_above = args.N
        exp.sample.n_below = args.N

    if args.G is not None:
        if args.N is None:
            raise ValueError('Cannot use -G without also specifing slide index with -N')
        if args.G == 0:
            exp.sample.n_above = 1
            exp.sample.n_below = 1
        elif args.G == 't':
            exp.sample.n_above = args.N
            exp.sample.n_below = 1
        elif args.G == 'b':
            exp.sample.n_above = 1
            exp.sample.n_below = args.N
        elif args.G == 2:
            exp.sample.n_above = args.N
            exp.sample.n_below = args.N
        elif args.G == 'n':
            exp.flip_sample = True
            exp.sample.n_above = args.N
            exp.sample.n_below = 1
        else:  # must be 'f'
            exp.flip_sample = True
            exp.sample.n_above = 1
            exp.sample.n_below = args.N

    if args.i is not None:
        if abs(args.i) > 90:
            raise ValueError('bad argument to -i, value must be between -90 and 90')
        exp.sample.nu_0 = np.cos(np.radians(args.i))

    if args.q is not None:
        if args.q % 4:
            raise ValueError('Number of quadrature points must be a multiple of 4')
        if exp.sample.nu_0 !=1 and args.q % 12:
            raise ValueError('Quadrature must be 12, 24, 36,... for oblique incidence')
        exp.sample.quad_pts = args.q

def add_experiment_constraints(exp, args):
    """Command-line constraints on experiment."""
    if args.S is not None:
        exp.num_spheres = args.S

    if args.r_sphere is not None:
        exp.r_sphere = args.r_sphere

    if args.t_sphere is not None:
        exp.r_sphere = args.t_sphere

    if args.diameter is not None:
        exp.d_beam = args.diameter

    if args.r is not None:
        exp.m_r = args.r

    if args.t is not None:
        exp.m_t = args.t

    if args.u is not None:
        exp.m_u = args.u

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

    if args.mus is not None:
        exp.default_mus = args.mus

    if args.f_r is not None:
        exp.fraction_of_rc_in_mr = args.f_r

    if args.f_t is not None:
        exp.fraction_of_tc_in_mt = args.f_t

    if args.M is not None:
        # MC iterations
        pass

    if args.p is not None:
        # photons
        pass

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
            exp.sample.b = Inf
        elif exp.sample.a == 0:
            if exp.default_mua is None:
                exp.sample.b = Inf
            else:
                exp.sample.b = exp.default_mua * exp.sample.d
        elif exp.default_mus is None:
            exp.sample.b = Inf
        else:
            exp.sample.b = exp.default_muas/exp.sample.a * exp.sample.d
    else:
        exp.sample.b = exp.default_b

    # set scattering anisotropy
    if exp.default_g is None:
        exp.sample.g = 0
    else:
        exp.sample.g = exp.default_g

    print(exp.sample)

    ur1, ut1, uru, utu = exp.sample.rt()
    ru, tu = exp.sample.unscattered_rt()
    print("Calculated quantities")
    print("   R total         = %.3f" % ur1)
    print("   R scattered     = %.3f" % (ur1 - ru))
    print("   R unscattered   = %.3f" % ru)
    print("   T total         = %.3f" % ut1)
    print("   R scattered     = %.3f" % (ut1 - tu))
    print("   T unscattered   = %.3f" % tu)

def main():
    """Main command-line interface."""

    # Initialize the parser and parse options
    parser = argparse.ArgumentParser(description='iad command line program',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=example_text())

    for spec in arg_specs:
        parser.add_argument(*spec.pop("flags"), **spec)
    args = parser.parse_args()

    if args.V:
        print_long_version()

    # If there is a file then read it, otherwise create a blank experiment
    if args.filename:
        exp = iadpython.read_rxt(args.filename)
    else:
        exp = iadpython.Experiment()

    # update the search to include the command line constraints
    add_sample_constraints(exp, args)
    add_experiment_constraints(exp, args)
    add_analysis_constraints(exp, args)

    if args.z:
        forward_calculation(exp)
    else:
        a, b, g = exp.invert_rt()
        print("   a = %.3f" % a)
        print("   b = %.3f" % b)
        print("   g = %.3f" % g)

if __name__ == "__main__":
    main()
