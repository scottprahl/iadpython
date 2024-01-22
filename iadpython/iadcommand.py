# pylint: disable=line-too-long

import sys
from enum import Enum
import argparse
import iadpython

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

# Argument specifications
arg_specs = [
    {"flags": ["-1"], "dest": "r_sphere", "nargs": 5, "type": float, "help": "Five numbers separated by spaces"},
    {"flags": ["-2"], "dest": "t_sphere", "nargs": 5, "type": float, "help": "Five numbers separated by spaces"},
    {"flags": ["-a"], "type": float, "help": "Use this albedo"},
    {"flags": ["-A"], "type": float, "help": "Use this absorption coefficient"},
    {"flags": ["-b"], "type": float, "help": "Use this optical thickness"},
    {"flags": ["-B"], "type": float, "help": "Beam diameter"},
    {"flags": ["-c"], "type": float, "help": "Fraction of unscattered refl in MR"},
    {"flags": ["-C"], "type": float, "help": "Fraction of unscattered trans in MT"},
    {"flags": ["-d"], "type": float, "help": "Thickness of sample"},
    {"flags": ["-D"], "type": float, "help": "Thickness of slide"},
    {"flags": ["-e"], "type": float, "default": 0.0001, "help": "Error tolerance (default 0.0001)"},
    {"flags": ["-E"], "type": float, "help": "Optical depth (=mua*D) for slides"},
    {"flags": ["-f"], "type": float, "help": "Fraction 0.0-1.0 of light to hit sphere wall first"},
    {"flags": ["-F"], "type": str, "help": "Use this scattering coefficient or formula"},
    {"flags": ["-g"], "type": float, "default": 0, "help": "Scattering anisotropy (default 0)"},
    {"flags": ["-G"], "type": str, "choices": ['0', '2', 't', 'b', 'n', 'f'], "help": "Type of boundary "},
    {"flags": ["-i"], "type": float, "help": "Light incident at this angle in degrees"},
    {"flags": ["-M"], "type": int, "help": "Number of Monte Carlo iterations"},
    {"flags": ["-n"], "type": float, "help": "Index of refraction of slab"},
    {"flags": ["-N"], "type": float, "help": "Index of refraction of slides"},
    {"flags": ["-o"], "type": str, "help": "Filename for output"},
    {"flags": ["-p"], "type": int, "help": "# of Monte Carlo photons (default 100000)"},
    {"flags": ["-q"], "type": int, "default": 8, "help": "Number of quadrature points (default=8)"},
    {"flags": ["-r"], "type": float, "help": "Total reflection measurement"},
    {"flags": ["-R"], "type": float, "help": "Actual reflectance for 100% measurement"},
    {"flags": ["-S"], "type": int, "choices": [0, 1, 2], "help": "Number of spheres used"},
    {"flags": ["-t"], "type": float, "help": "Total transmission measurement"},
    {"flags": ["-T"], "type": float, "help": "Actual transmission for 100% measurement"},
    {"flags": ["-u"], "type": float, "help": "Unscattered transmission measurement"},
    {"flags": ["-v"], "action": "store_true", "help": "Version information"},
    {"flags": ["-V"], "type": int, "choices": [0, 1, 2], "help": "Verbosity level"},
    {"flags": ["-x"], "type": int, "help": "Set debugging level"},
    {"flags": ["-X"], "action": "store_true", "help": "Dual beam configuration"},
    {"flags": ["-z"], "action": "store_true", "help": "Do forward calculation"},
    {"flags": ["input"], "nargs": "?", "help": "Input file"}
]

# Initialize the parser and add options
parser = argparse.ArgumentParser(description='iad command line program',
                                 formatter_class=argparse.RawTextHelpFormatter)
for spec in arg_specs:
    parser.add_argument(*spec.pop("flags"), **spec)

# Parse the arguments
args = parser.parse_args()

def print_version():
    """Print the version information and quit."""
    print("iadpython version:", iadpython.__version__)
    print("Author:", iadpython.__author__, "-", iadpython.__email__)
    print("Copyright:", iadpython.__copyright__)
    print("License:", iadpython.__license__)
    print("URL:", iadpython.__url__)
    print("\nForward and inverse adding-doubling radiative transport calculations.")
    print("Extensive documentation is at <https://iadpython.readthedocs.io>\n")
    sys.exit(0)

def print_usage():
    """Print usage information and quit."""
    usage_text = parser.format_help()
    print(usage_text)
    print("Examples:")
    print("  iad file.rxt              Results will be put in file.txt")
    print("  iad file                  Same as above")
    print("  iad -c 0.9 file.rxt       Assume M_R includes 90%% of unscattered reflectance")
    print("  iad -C 0.8 file.rxt       Assume M_T includes 80%% of unscattered transmittance")
    print("  iad -e 0.0001 file.rxt    Better convergence to R & T values")
    print("  iad -f 1.0 file.rxt       All light hits reflectance sphere wall first")
    print("  iad -o out file.rxt       Calculated values in out")
    print("  iad -r 0.3                R_total=0.")
    sys.exit(0)


# Main logic
def main():
    # Parse and use the arguments
    albedo = args.a
    # ... other arguments

    # Perform calculations
    # This would involve using functions from iadpython and additional logic
    # For example:
    # result = calculate_mr_mt(albedo, other_params)

    # Output the results
    # print(result)

if __name__ == "__main__":
    main()
