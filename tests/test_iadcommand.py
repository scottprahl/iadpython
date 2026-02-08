"""Tests for the command line wrapper."""

import sys
import os
import io
import tempfile
import unittest
import argparse
from unittest.mock import patch

# Calculate the path to the iadpython package so pytest works from any directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
iadpython_package_dir = os.path.join(parent_dir, "iadpython")
sys.path.append(iadpython_package_dir)

# Now we can import iadcommand
import iadcommand


class TestCommandLineArgs(unittest.TestCase):
    """Tests for validator functions."""

    def test_validator_01_valid(self):
        """Valid test that 01."""
        self.assertEqual(iadcommand.validator_01("0.5"), 0.5)

    def test_validator_01_invalid(self):
        """Invalid test for 11."""
        with self.assertRaises(argparse.ArgumentTypeError):
            iadcommand.validator_01("-1")

    def test_validator_11_valid(self):
        """Valid test that 11."""
        self.assertEqual(iadcommand.validator_11("-0.5"), -0.5)

    def test_validator_11_invalid(self):
        """Invalid test for 11."""
        with self.assertRaises(argparse.ArgumentTypeError):
            iadcommand.validator_11("2")

    def test_validator_11_invalid_message(self):
        """Invalid anisotropy message should report the correct range."""
        with self.assertRaises(argparse.ArgumentTypeError) as cm:
            iadcommand.validator_11("2")
        self.assertIn("between -1 and 1", str(cm.exception))

    def test_validator_positive_valid(self):
        """Valid test for positive."""
        self.assertEqual(iadcommand.validator_positive("10"), 10.0)

    def test_validator_positive_invalid(self):
        """Invalid test for positive."""
        with self.assertRaises(argparse.ArgumentTypeError):
            iadcommand.validator_positive("-5")


class TestIadFile(unittest.TestCase):
    """Tests for rxt files."""

    def test_valid_arguments(self):
        """Simple test."""
        test_args = ["iadcommand.py", "data/basic-A.rxt"]
        with self.assertRaises(SystemExit):
            with patch("sys.argv", test_args):
                iadcommand.main()

    def test_wavelength_rxt_file_runs(self):
        """Process wavelength-based .rxt data without lambda truth-value errors."""
        sample_file = os.path.join(current_dir, "data", "sample-A.rxt")
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as handle:
            out_file = handle.name
        test_args = ["iadcommand.py", sample_file, "-o", out_file]
        try:
            with self.assertRaises(SystemExit) as cm:
                with patch("sys.argv", test_args):
                    iadcommand.main()
            self.assertEqual(cm.exception.code, 0)
        finally:
            if os.path.exists(out_file):
                os.remove(out_file)

    def test_bad_output_path_exits_cleanly(self):
        """Invalid output path should report an error, not crash on closed stdout."""
        sample_file = os.path.join(current_dir, "data", "basic-A.rxt")
        with tempfile.NamedTemporaryFile() as handle:
            bad_out = os.path.join(handle.name, "output.txt")
            test_args = ["iadcommand.py", sample_file, "-o", bad_out]
            with patch("sys.argv", test_args):
                with patch("sys.stdout", new_callable=io.StringIO) as fake_stdout:
                    with self.assertRaises(SystemExit) as cm:
                        iadcommand.main()
            self.assertEqual(cm.exception.code, 1)
            self.assertIn("Error:", fake_stdout.getvalue())


class TestExperimentConstraints(unittest.TestCase):
    """Tests for command-line constraints on experiment setup."""

    def test_reflection_sphere_cli_values_create_sphere(self):
        """Convert `-1` values into a reflectance sphere object."""
        exp = iadcommand.iadpython.Experiment()
        args = argparse.Namespace(
            S=None,
            r_sphere=[250.0, 20.0, 10.0, 10.0, 0.99],
            t_sphere=None,
            diameter=None,
            r=None,
            t=None,
            u=None,
        )

        iadcommand.add_experiment_constraints(exp, args)

        self.assertIsInstance(exp.r_sphere, iadcommand.iadpython.Sphere)
        self.assertIsNone(exp.t_sphere)
        self.assertEqual(exp.r_sphere.d, 250.0)
        self.assertEqual(exp.r_sphere.sample.d, 20.0)
        self.assertEqual(exp.r_sphere.third.d, 10.0)
        self.assertEqual(exp.r_sphere.detector.d, 10.0)
        self.assertEqual(exp.r_sphere.r_wall, 0.99)
        self.assertTrue(exp.r_sphere.refl)

    def test_transmission_sphere_cli_values_assign_t_sphere(self):
        """Assign `-2` values to transmission sphere instead of reflection sphere."""
        exp = iadcommand.iadpython.Experiment()
        args = argparse.Namespace(
            S=None,
            r_sphere=None,
            t_sphere=[300.0, 25.0, 12.0, 8.0, 0.98],
            diameter=None,
            r=None,
            t=None,
            u=None,
        )

        iadcommand.add_experiment_constraints(exp, args)

        self.assertIsNone(exp.r_sphere)
        self.assertIsInstance(exp.t_sphere, iadcommand.iadpython.Sphere)
        self.assertEqual(exp.t_sphere.d, 300.0)
        self.assertEqual(exp.t_sphere.sample.d, 25.0)
        self.assertEqual(exp.t_sphere.third.d, 12.0)
        self.assertEqual(exp.t_sphere.detector.d, 8.0)
        self.assertEqual(exp.t_sphere.r_wall, 0.98)
        self.assertFalse(exp.t_sphere.refl)

    def test_transmission_sphere_inverse_cli_runs(self):
        """Ensure `-2` transmission sphere path does not crash in inverse mode."""
        test_args = ["iadcommand.py", "-S", "1", "-2", "250", "20", "10", "10", "0.99", "-t", "0.3"]
        with patch("sys.argv", test_args):
            with self.assertRaises(SystemExit) as cm:
                iadcommand.main()
        self.assertEqual(cm.exception.code, 0)

    def test_reflectance_standard_sets_both_when_t_missing(self):
        """`-R` should set transmission standard too when `-T` is absent."""
        exp = iadcommand.iadpython.Experiment(
            r_sphere=iadcommand.iadpython.Sphere(250, 20),
            t_sphere=iadcommand.iadpython.Sphere(250, 20, refl=False),
        )
        args = argparse.Namespace(
            S=None,
            r_sphere=None,
            t_sphere=None,
            diameter=None,
            r=None,
            t=None,
            u=None,
            R=0.88,
            T=None,
        )

        iadcommand.add_experiment_constraints(exp, args)

        self.assertEqual(exp.r_sphere.r_std, 0.88)
        self.assertEqual(exp.t_sphere.r_std, 0.88)

    def test_transmission_standard_sets_both_when_r_missing(self):
        """`-T` should set reflectance standard too when `-R` is absent."""
        exp = iadcommand.iadpython.Experiment(
            r_sphere=iadcommand.iadpython.Sphere(250, 20),
            t_sphere=iadcommand.iadpython.Sphere(250, 20, refl=False),
        )
        args = argparse.Namespace(
            S=None,
            r_sphere=None,
            t_sphere=None,
            diameter=None,
            r=None,
            t=None,
            u=None,
            R=None,
            T=0.91,
        )

        iadcommand.add_experiment_constraints(exp, args)

        self.assertEqual(exp.r_sphere.r_std, 0.91)
        self.assertEqual(exp.t_sphere.r_std, 0.91)


class TestAnalysisConstraints(unittest.TestCase):
    """Tests for command-line constraints on analysis setup."""

    @staticmethod
    def _base_analysis_args(**overrides):
        args = {
            "albedo": None,
            "b": None,
            "g": None,
            "mua": None,
            "mus": None,
            "f_r": None,
            "f_t": None,
            "f_wall": None,
            "error": None,
            "M": None,
            "p": None,
            "x": None,
            "X": False,
        }
        args.update(overrides)
        return argparse.Namespace(**args)

    def test_extra_analysis_flags_are_applied(self):
        """Apply -e, -f, -M, -p, -x and -X like iad command-line behavior."""
        exp = iadcommand.iadpython.Experiment()
        args = self._base_analysis_args(error=0.002, f_wall=0.3, M=4, p=12345, x=8, X=True)

        iadcommand.add_analysis_constraints(exp, args)

        self.assertEqual(exp.tolerance, 0.002)
        self.assertEqual(exp.MC_tolerance, 0.002)
        self.assertEqual(exp.f_r, 0.3)
        self.assertEqual(exp.max_mc_iterations, 4)
        self.assertEqual(exp.n_photons, 12345)
        self.assertEqual(exp.debug_level, 8)
        self.assertEqual(exp.method, "comparison")


class TestSampleConstraints(unittest.TestCase):
    """Tests for command-line constraints on sample setup."""

    @staticmethod
    def _base_sample_args(**overrides):
        args = {
            "thickness": None,
            "slide_thickness": None,
            "E": None,
            "nslab": None,
            "nslide": None,
            "G": None,
            "i": None,
            "q": None,
        }
        args.update(overrides)
        return argparse.Namespace(**args)

    def test_boundary_mode_zero_sets_both_sides_to_air(self):
        """`-G 0` should set both slide indices to 1."""
        exp = iadcommand.iadpython.Experiment()
        args = self._base_sample_args(nslide=1.5, G="0")

        iadcommand.add_sample_constraints(exp, args)

        self.assertEqual(exp.sample.n_above, 1)
        self.assertEqual(exp.sample.n_below, 1)

    def test_boundary_mode_two_sets_both_sides_to_slide_index(self):
        """`-G 2` should set both sides to the specified slide index."""
        exp = iadcommand.iadpython.Experiment()
        args = self._base_sample_args(nslide=1.5, G="2")

        iadcommand.add_sample_constraints(exp, args)

        self.assertEqual(exp.sample.n_above, 1.5)
        self.assertEqual(exp.sample.n_below, 1.5)


class TestIadCommandForward(unittest.TestCase):
    """Test forward calculation scenarios."""

    def test_only_albedo(self):
        """Test with only albedo provided."""
        test_args = ["iadcommand.py", "-z", "-a", "0.5"]
        with self.assertRaises(SystemExit):
            with patch("sys.argv", test_args):
                iadcommand.main()

    def test_only_optical_thickness(self):
        """Test with only optical thickness provided."""
        test_args = ["iadcommand.py", "-z", "-b", "1"]
        with self.assertRaises(SystemExit):
            with patch("sys.argv", test_args):
                iadcommand.main()

    def test_albedo_and_optical_thickness(self):
        """Test with albedo and optical thickness."""
        test_args = ["iadcommand.py", "-z", "-a", "0.1", "-b", "1"]
        with patch("sys.argv", test_args):
            with self.assertRaises(SystemExit):  # Expecting the program to exit
                iadcommand.main()

    def test_albedo_and_optical_thickness_and_g(self):
        """Test with albedo and optical thickness and g."""
        test_args = ["iadcommand.py", "-z", "-a", "0.1", "-b", "1", "-g", "0.9"]
        with self.assertRaises(SystemExit):
            with patch("sys.argv", test_args):
                iadcommand.main()

    def test_slab_index(self):
        """Test with slab index."""
        test_args = ["iadcommand.py", "-z", "-a", "0.1", "-n", "1.4"]
        with self.assertRaises(SystemExit):
            with patch("sys.argv", test_args):
                iadcommand.main()

    def test_slab_and_slide_index(self):
        """Test with slab and slide index."""
        test_args = ["iadcommand.py", "-z", "-a", "0.1", "-n", "1.4", "-n", "1.5"]
        with self.assertRaises(SystemExit):
            with patch("sys.argv", test_args):
                iadcommand.main()

    def test_only_absorption_coefficient(self):
        """Test with only absorption coefficient provided."""
        test_args = ["iadcommand.py", "-z", "--mua", "0.1"]
        with self.assertRaises(SystemExit):
            with patch("sys.argv", test_args):
                iadcommand.main()

    def test_only_scattering_coefficient(self):
        """Test with only scattering coefficient provided."""
        test_args = ["iadcommand.py", "-z", "--mus", "10"]
        with self.assertRaises(SystemExit):
            with patch("sys.argv", test_args):
                iadcommand.main()

    def test_only_anisotropy(self):
        """Test with only anisotropy provided."""
        test_args = ["iadcommand.py", "-z", "-g", "0.9"]
        with self.assertRaises(SystemExit):
            with patch("sys.argv", test_args):
                iadcommand.main()

    def test_scattering_and_absorption(self):
        """Test with scattering and absorption."""
        test_args = ["iadcommand.py", "-z", "--mua", "0.1", "--mus", "10"]
        with self.assertRaises(SystemExit):
            with patch("sys.argv", test_args):
                iadcommand.main()

    def test_scattering_and_absorption_and_g(self):
        """Test with scattering and absorption and g."""
        test_args = ["iadcommand.py", "-z", "--mua", "0.1", "--mus", "10", "-g", "0.9"]
        with self.assertRaises(SystemExit):
            with patch("sys.argv", test_args):
                iadcommand.main()

    def test_invalid_albedo(self):
        """Test with invalid albedo value."""
        test_args = ["iadcommand.py", "-z", "-a", "-1"]
        with patch("sys.argv", test_args):
            with self.assertRaises(SystemExit):
                iadcommand.main()

    def test_invalid_optical_thickness(self):
        """Test example with invalid argument values."""
        test_args = ["iadcommand.py", "-z", "-a", "0.5", "-b", "-1"]
        with patch("sys.argv", test_args):
            with self.assertRaises(SystemExit):  # Expecting the program to exit
                iadcommand.main()

    def test_forward_calculation_missing_albedo(self):
        """Test example when albedo is not provided."""
        test_args = ["iadcommand.py", "-z", "-b", "1", "-g", "0.9"]
        with patch("sys.argv", test_args):
            with self.assertRaises(SystemExit):  # Expecting the program to exit
                iadcommand.main()

    def test_invalid_scattering(self):
        """Test example with invalid argument values."""
        test_args = ["iadcommand.py", "-z", "--mus", "-0.5"]
        with patch("sys.argv", test_args):
            with self.assertRaises(SystemExit):  # Expecting the program to exit
                iadcommand.main()

    def test_invalid_absorption(self):
        """Test example with invalid argument values."""
        test_args = ["iadcommand.py", "-z", "--mua", "-0.5"]
        with patch("sys.argv", test_args):
            with self.assertRaises(SystemExit):  # Expecting the program to exit
                iadcommand.main()

    def test_invalid_anisotropy(self):
        """Test example with invalid argument values."""
        test_args = ["iadcommand.py", "-z", "-g", "1.1"]
        with patch("sys.argv", test_args):
            with self.assertRaises(SystemExit):  # Expecting the program to exit
                iadcommand.main()

    def test_invalid_mc_iterations_range(self):
        """`-M` outside 0..50 should fail like iad command-line behavior."""
        test_args = ["iadcommand.py", "-z", "-M", "55", "-a", "0.5"]
        with patch("sys.argv", test_args):
            with self.assertRaises(SystemExit) as cm:
                iadcommand.main()
        self.assertEqual(cm.exception.code, 2)

    def test_forward_output_uses_t_scattered_label(self):
        """Forward output should label transmission scattered term correctly."""
        test_args = ["iadcommand.py", "-z", "-a", "0.5", "-b", "1"]
        with patch("sys.argv", test_args):
            with patch("sys.stdout", new_callable=io.StringIO) as fake_stdout:
                with self.assertRaises(SystemExit):
                    iadcommand.main()
        self.assertIn("T scattered", fake_stdout.getvalue())


class TestIadSingle(unittest.TestCase):
    """Single sphere tests."""

    def test_inverse_missing(self):
        """Test with no arguments."""
        test_args = ["iadcommand.py"]
        with patch("sys.argv", test_args):
            with self.assertRaises(SystemExit):
                iadcommand.main()

    def test_inverse_reflection(self):
        """Test with invalid albedo value."""
        test_args = ["iadcommand.py", "-r", "0.5"]
        with patch("sys.argv", test_args):
            with self.assertRaises(SystemExit):
                iadcommand.main()

    def test_inverse_transmission(self):
        """Test with invalid albedo value."""
        test_args = ["iadcommand.py", "-t", "0.5"]
        with patch("sys.argv", test_args):
            with self.assertRaises(SystemExit):
                iadcommand.main()

    def test_inverse_unscattered(self):
        """Test with invalid albedo value."""
        test_args = ["iadcommand.py", "-u", "0.5"]
        with patch("sys.argv", test_args):
            with self.assertRaises(SystemExit):
                iadcommand.main()

    def test_dual_beam_rejects_two_spheres(self):
        """`-X` should not allow double integrating sphere mode."""
        test_args = ["iadcommand.py", "-X", "-S", "2", "-r", "0.3", "-t", "0.4"]
        with patch("sys.argv", test_args):
            with self.assertRaises(SystemExit) as cm:
                iadcommand.main()
        self.assertEqual(cm.exception.code, 1)

    def test_dual_beam_rejects_nonzero_wall_fraction(self):
        """`-X` should not allow non-zero `-f`, matching iad constraints."""
        test_args = ["iadcommand.py", "-X", "-f", "0.2", "-r", "0.3"]
        with patch("sys.argv", test_args):
            with self.assertRaises(SystemExit) as cm:
                iadcommand.main()
        self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
