"""Tests for the command line wrapper."""

import sys
import os
import io
import shutil
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


SINGLE_ROW_VARIABLE_RXT = """IAD1  # Must be first four characters
1.38   # Index of refraction of the sample
1.00   # Index of refraction of the top and bottom slides
1.62   # [mm] Thickness of sample
0.00   # [mm] Thickness of slides
2.00   # [mm] Diameter of illumination beam
0.99   # Reflectivity of the reflectance calibration standard

1      # Number of spheres used for the measurement

# Properties of sphere used for reflectance measurements
101.6  # [mm] Sphere Diameter
12.7   # [mm] Sample Port Diameter
12.7   # [mm] Entrance Port Diameter
0.6    # [mm] Detector Port Diameter
0.94   # Reflectivity of the sphere wall

# Properties of sphere for transmittance measurements
101.6  # [mm] Sphere Diameter
12.7   # [mm] Sample Port Diameter
12.7   # [mm] Third Port Diameter
0.6    # [mm] Detector Port Diameter
0.94   # Reflectivity of the sphere wall

    L         r       t        g
851.930000 0.315087 0.606721   0.9
"""

BIOPIX_851_RXT = SINGLE_ROW_VARIABLE_RXT.replace("0.9\n", "-0.5\n")

TWO_ROW_VARIABLE_RXT = SINGLE_ROW_VARIABLE_RXT.replace(
    "851.930000 0.315087 0.606721   0.9\n",
    "851.930000 0.315087 0.606721   0.9\n852.930000 0.314000 0.607000   0.9\n",
)

SINGLE_ROW_TU_RXT = SINGLE_ROW_VARIABLE_RXT.replace(
    "    L         r       t        g\n851.930000 0.315087 0.606721   0.9\n",
    "    L         t        u\n851.930000 0.606721 0.011859\n",
)

SINGLE_ROW_RU_RXT = SINGLE_ROW_VARIABLE_RXT.replace(
    "    L         r       t        g\n851.930000 0.315087 0.606721   0.9\n",
    "    L         r        u\n851.930000 0.315087 0.011859\n",
)


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

    def test_validator_scattering_constraint_power_law(self):
        """Power-law `-F` syntax should parse like the CWEB CLI."""
        parsed = iadcommand.validator_scattering_constraint("P 500 1.0 -1.3")
        self.assertEqual(parsed, ("power", 500.0, 1.0, -1.3))


class TestIadFile(unittest.TestCase):
    """Tests for rxt files."""

    def _run_single_row_rxt(self, rxt_text):
        """Run a generated single-row .rxt file and return its output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_file = os.path.join(tmpdir, "single-row.rxt")
            out_file = os.path.join(tmpdir, "single-row.txt")
            with open(sample_file, "w", encoding="utf-8") as fh:
                fh.write(rxt_text)

            test_args = ["iadcommand.py", sample_file, "-M", "0", "-o", out_file]
            with patch("sys.argv", test_args):
                with self.assertRaises(SystemExit) as cm:
                    iadcommand.main()

            self.assertEqual(cm.exception.code, 0)
            with open(out_file, encoding="utf-8") as fh:
                return fh.read()

    def _assert_result_file_has_header_and_rows(self, out_file, rows=1):
        """Assert iad-compatible result-file output and return the file contents."""
        with open(out_file, encoding="utf-8") as fh:
            output = fh.read()

        self.assertTrue(output.startswith("# Inverse Adding-Doubling "))
        self.assertIn("##wave", output)
        data_lines = [line for line in output.splitlines() if line.strip() and not line.startswith("#")]
        self.assertEqual(len(data_lines), rows)
        for line in data_lines:
            self.assertGreaterEqual(len(line.split()), 9)
        return output

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

    def test_single_row_variable_file_uses_scalar_values(self):
        """Single-row variable files should not pass length-1 arrays to scalar inversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_file = os.path.join(tmpdir, "single-row.rxt")
            out_file = os.path.join(tmpdir, "single-row.txt")
            with open(sample_file, "w", encoding="utf-8") as fh:
                fh.write(SINGLE_ROW_VARIABLE_RXT)

            test_args = ["iadcommand.py", sample_file, "-c", "0", "-M", "0", "-o", out_file]
            with patch("sys.argv", test_args):
                with self.assertRaises(SystemExit) as cm:
                    iadcommand.main()

            self.assertEqual(cm.exception.code, 0)
            with open(out_file, encoding="utf-8") as fh:
                output = fh.read()
            self.assertIn(" 851.9", output)

    def test_variable_file_with_transmission_and_unscattered_runs(self):
        """Variable `L t u` files should treat omitted reflectance as zero."""
        output = self._run_single_row_rxt(SINGLE_ROW_TU_RXT)

        self.assertIn(" 851.9", output)
        self.assertIn("   0.0000", output)

    def test_variable_file_with_reflectance_and_unscattered_runs(self):
        """Variable `L r u` files should treat omitted transmittance as zero."""
        output = self._run_single_row_rxt(SINGLE_ROW_RU_RXT)

        self.assertIn(" 851.9", output)
        self.assertIn("   0.0000", output)

    def test_file_output_includes_legacy_header_and_status_column(self):
        """Result files should include the iad metadata header and row status."""
        output = self._run_single_row_rxt(BIOPIX_851_RXT)

        self.assertTrue(output.startswith("# Inverse Adding-Doubling "))
        self.assertIn("#                        Beam diameter =     2.0 mm", output)
        self.assertIn("# Reflection sphere has a baffle between sample and detector", output)
        self.assertIn("# 4 input columns with LABELS: L  r  t  g  using the substitution (single-beam) method.", output)
        self.assertIn("#  Photons used to estimate lost light =   0", output)

        data_lines = [line for line in output.splitlines() if line.strip() and not line.startswith("#")]
        self.assertEqual(len(data_lines), 1)
        self.assertEqual(len(data_lines[0].split()), 9)
        self.assertRegex(data_lines[0], r"\s\*\s*$")

    def test_file_output_reports_progress_status_to_stderr(self):
        """Each non-debug file row should emit iad-style progress to stderr."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_file = os.path.join(tmpdir, "two-row.rxt")
            out_file = os.path.join(tmpdir, "two-row.txt")
            with open(sample_file, "w", encoding="utf-8") as fh:
                fh.write(TWO_ROW_VARIABLE_RXT)

            test_args = ["iadcommand.py", sample_file, "-M", "0", "-o", out_file]
            with patch("sys.argv", test_args):
                with patch("sys.stderr", new_callable=io.StringIO) as fake_stderr:
                    with self.assertRaises(SystemExit) as cm:
                        iadcommand.main()

            self.assertEqual(cm.exception.code, 0)
            self.assertEqual(fake_stderr.getvalue(), "**")

    def test_rxt_reflectance_standard_does_not_override_transmission_standard(self):
        """The .rxt header standard is reflection-only; CWEB leaves T standard at 1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_file = os.path.join(tmpdir, "single-row.rxt")
            with open(sample_file, "w", encoding="utf-8") as fh:
                fh.write(BIOPIX_851_RXT)

            exp = iadcommand.iadpython.read_rxt(sample_file)

            self.assertEqual(exp.sample.quad_pts, 8)
            self.assertEqual(exp.input_column_labels, "Lrtg")
            self.assertAlmostEqual(exp.r_sphere.r_std, 0.99, delta=1e-12)
            self.assertAlmostEqual(exp.t_sphere.r_std, 1.0, delta=1e-12)

    def test_biopix_single_row_no_mc_uses_cweb_quadrature_and_agrid_basin(self):
        """The `-c 0` no-MC path should find the same basin as CWEB iad."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_file = os.path.join(tmpdir, "single-row.rxt")
            with open(sample_file, "w", encoding="utf-8") as fh:
                fh.write(BIOPIX_851_RXT)

            args = iadcommand.build_parser().parse_args(["-c", "0", "-M", "0", sample_file])
            exp = iadcommand.iadpython.read_rxt(args.filename)
            iadcommand.add_sample_constraints(exp, args)
            iadcommand.add_experiment_constraints(exp, args)
            iadcommand.add_analysis_constraints(exp, args)
            point = iadcommand._point_experiment(exp, 0)  # pylint: disable=protected-access

            point.invert_rt()

            self.assertEqual(point.sample.quad_pts, 8)
            self.assertTrue(point.found)
            self.assertAlmostEqual(point.sample.mu_sp(), 0.764, delta=0.01)
            self.assertLess(point.final_distance, point.tolerance)

    def test_debug_iterations_use_cweb_trace_shape(self):
        """`-x 4` should emit the CWEB-style trace and keep result-file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_file = os.path.join(tmpdir, "single-row.rxt")
            out_file = os.path.join(tmpdir, "single-row.txt")
            with open(sample_file, "w", encoding="utf-8") as fh:
                fh.write(SINGLE_ROW_VARIABLE_RXT)

            test_args = ["iadcommand.py", sample_file, "-c", "0", "-M", "0", "-x", "4", "-o", out_file]
            with patch("sys.argv", test_args):
                with patch("sys.stderr", new_callable=io.StringIO) as fake_stderr:
                    with self.assertRaises(SystemExit) as cm:
                        iadcommand.main()

            self.assertEqual(cm.exception.code, 0)
            debug_output = fake_stderr.getvalue()
            self.assertIn("-------------------NEXT DATA POINT---------------------", debug_output)
            self.assertIn("---------------- Beginning New Search -----------------", debug_output)
            self.assertIn("Final amoeba/brent result after", debug_output)
            self.assertIn("Failed Search, too many iterations", debug_output)
            self._assert_result_file_has_header_and_rows(out_file)

    def test_debug_grid_uses_cweb_decision_text(self):
        """`-x 2` should emit CWEB-style grid decisions and reuse the grid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_file = os.path.join(tmpdir, "two-row.rxt")
            out_file = os.path.join(tmpdir, "two-row.txt")
            with open(sample_file, "w", encoding="utf-8") as fh:
                fh.write(TWO_ROW_VARIABLE_RXT)

            test_args = ["iadcommand.py", sample_file, "-M", "0", "-x", "2", "-o", out_file]
            with patch("sys.argv", test_args):
                with patch("sys.stderr", new_callable=io.StringIO) as fake_stderr:
                    with self.assertRaises(SystemExit) as cm:
                        iadcommand.main()

            self.assertEqual(cm.exception.code, 0)
            debug_output = fake_stderr.getvalue()
            self.assertIn("GRID: Fill because NULL", debug_output)
            self.assertIn("GRID: Filling AB grid (g=0.90000)", debug_output)
            self.assertEqual(debug_output.count("GRID: Fill because"), 1)
            self.assertEqual(debug_output.count("GRID: Filling AB grid"), 1)
            self.assertEqual(debug_output.count("GRID: Finding best grid points"), 2)
            self.assertNotIn("grid constant", debug_output)
            self._assert_result_file_has_header_and_rows(out_file, rows=2)

    def test_debug_best_guess_uses_cweb_table(self):
        """`-x 16` should emit the legacy best-grid simplex table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_file = os.path.join(tmpdir, "single-row.rxt")
            out_file = os.path.join(tmpdir, "single-row.txt")
            with open(sample_file, "w", encoding="utf-8") as fh:
                fh.write(BIOPIX_851_RXT)

            test_args = ["iadcommand.py", sample_file, "-M", "0", "-x", "16", "-o", out_file]
            with patch("sys.argv", test_args):
                with patch("sys.stderr", new_callable=io.StringIO) as fake_stderr:
                    with self.assertRaises(SystemExit) as cm:
                        iadcommand.main()

            self.assertEqual(cm.exception.code, 0)
            debug_output = fake_stderr.getvalue()
            self.assertIn("BEST: GRID GUESSES", debug_output)
            self.assertIn("BEST:  k      albedo          b          g   distance", debug_output)
            self.assertIn("BEST:  0     0.98963    0.72615   -0.50000    0.02724", debug_output)
            self.assertIn("BEST: <1>    0.98963    0.72615   -0.50000    0.02724", debug_output)
            self.assertIn("BEST: <2>    0.98599    0.72615   -0.50000    0.02744", debug_output)
            self.assertIn("BEST: <3>    0.98599    0.85214   -0.50000    0.03701", debug_output)
            self.assertIn("Successful Search", debug_output)
            self.assertNotIn("grid constant", debug_output)
            self.assertNotIn("GRID: Fill", debug_output)
            self._assert_result_file_has_header_and_rows(out_file)

    def test_debug_best_guess_repeats_for_mc_iterations(self):
        """`-x 16` should emit simplex points for every MC re-inversion."""

        def _fake_update_lost_light(exp, _a, _b, _g, **_kw):
            exp.ur1_lost += 0.002
            exp.ut1_lost += 0.002
            exp.uru_lost += 0.001
            exp.utu_lost += 0.001
            return 0.002, 0.002, 0.002, 0.001, 0.001

        with tempfile.TemporaryDirectory() as tmpdir:
            sample_file = os.path.join(tmpdir, "single-row.rxt")
            out_file = os.path.join(tmpdir, "single-row.txt")
            with open(sample_file, "w", encoding="utf-8") as fh:
                fh.write(BIOPIX_851_RXT)

            test_args = ["iadcommand.py", sample_file, "-M", "2", "-x", "16", "-o", out_file]
            with patch("sys.argv", test_args):
                with patch("iadcommand._discover_mc_lost_binary", return_value="/fake/mc_lost"):
                    with patch.object(iadcommand.iadpython.Experiment, "_update_lost_light", _fake_update_lost_light):
                        with patch("sys.stderr", new_callable=io.StringIO) as fake_stderr:
                            with self.assertRaises(SystemExit) as cm:
                                iadcommand.main()

            self.assertEqual(cm.exception.code, 0)
            debug_output = fake_stderr.getvalue()
            self.assertEqual(debug_output.count("BEST: GRID GUESSES"), 3)
            self.assertEqual(debug_output.count("BEST: <1>"), 3)
            self.assertEqual(debug_output.count("BEST: <2>"), 3)
            self.assertEqual(debug_output.count("BEST: <3>"), 3)
            self.assertNotIn("hot start", debug_output)
            self._assert_result_file_has_header_and_rows(out_file)

    def test_debug_search_uses_cweb_decision_trace(self):
        """`-x 32` should emit the CWEB search-selection trace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_file = os.path.join(tmpdir, "single-row.rxt")
            out_file = os.path.join(tmpdir, "single-row.txt")
            with open(sample_file, "w", encoding="utf-8") as fh:
                fh.write(BIOPIX_851_RXT)

            test_args = ["iadcommand.py", sample_file, "-M", "0", "-x", "32", "-o", out_file]
            with patch("sys.argv", test_args):
                with patch("sys.stderr", new_callable=io.StringIO) as fake_stderr:
                    with self.assertRaises(SystemExit) as cm:
                        iadcommand.main()

            self.assertEqual(cm.exception.code, 0)
            debug_output = fake_stderr.getvalue()
            self.assertIn("SEARCH: starting with 2 measurement(s)", debug_output)
            self.assertIn("SEARCH:      and with 1 constraint(s)", debug_output)
            self.assertIn("            anisotropy constrained", debug_output)
            self.assertIn("SEARCH: m_r =  0.31509 m_t =  0.60672 m_u =  0.00000", debug_output)
            self.assertIn("SEARCH:  rt =  0.31509  rd =  0.29233  ru =  0.02276", debug_output)
            self.assertIn("SEARCH:  tt =  0.60672  td =  0.60672  tu =  0.00000", debug_output)
            self.assertIn("SEARCH: ending with 2 measurement(s)", debug_output)
            self.assertIn("SEARCH:    and with 1 constraint(s)", debug_output)
            self.assertIn("SEARCH: final choice for search = FIND_AB", debug_output)
            self.assertIn("SEARCH: Using U_Find_AB() mu=1.00, g=   -0.500 (constrained g)", debug_output)
            self.assertNotIn("automatic search ->", debug_output)
            self.assertNotIn("search override ->", debug_output)
            self._assert_result_file_has_header_and_rows(out_file)

    def test_debug_search_repeats_routine_for_mc_iterations(self):
        """`-x 32 -M 2` should print one decision trace and three routine lines."""

        def _fake_update_lost_light(exp, _a, _b, _g, **_kw):
            exp.ur1_lost += 0.002
            exp.ut1_lost += 0.002
            exp.uru_lost += 0.001
            exp.utu_lost += 0.001
            return 0.002, 0.002, 0.002, 0.001, 0.001

        with tempfile.TemporaryDirectory() as tmpdir:
            sample_file = os.path.join(tmpdir, "single-row.rxt")
            out_file = os.path.join(tmpdir, "single-row.txt")
            with open(sample_file, "w", encoding="utf-8") as fh:
                fh.write(BIOPIX_851_RXT)

            test_args = ["iadcommand.py", sample_file, "-M", "2", "-x", "32", "-o", out_file]
            with patch("sys.argv", test_args):
                with patch("iadcommand._discover_mc_lost_binary", return_value="/fake/mc_lost"):
                    with patch.object(iadcommand.iadpython.Experiment, "_update_lost_light", _fake_update_lost_light):
                        with patch("sys.stderr", new_callable=io.StringIO) as fake_stderr:
                            with self.assertRaises(SystemExit) as cm:
                                iadcommand.main()

            self.assertEqual(cm.exception.code, 0)
            debug_output = fake_stderr.getvalue()
            self.assertEqual(debug_output.count("SEARCH: starting with"), 1)
            self.assertEqual(debug_output.count("SEARCH: final choice for search = FIND_AB"), 1)
            self.assertEqual(debug_output.count("SEARCH: Using U_Find_AB()"), 3)
            self.assertNotIn("automatic search ->", debug_output)
            self._assert_result_file_has_header_and_rows(out_file)

    def test_debug_grid_calc_uses_cweb_grid_dump(self):
        """`-x 64` should emit the legacy dense grid calculation table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_file = os.path.join(tmpdir, "single-row.rxt")
            out_file = os.path.join(tmpdir, "single-row.txt")
            with open(sample_file, "w", encoding="utf-8") as fh:
                fh.write(BIOPIX_851_RXT)

            test_args = ["iadcommand.py", sample_file, "-M", "0", "-x", "64", "-o", out_file]
            with patch("sys.argv", test_args):
                with patch("sys.stderr", new_callable=io.StringIO) as fake_stderr:
                    with self.assertRaises(SystemExit) as cm:
                        iadcommand.main()

            self.assertEqual(cm.exception.code, 0)
            debug_output = fake_stderr.getvalue()
            self.assertIn(
                "+   i   j       a         b          g     |"
                "     M_R        grid  |     M_T        grid",
                debug_output,
            )
            self.assertIn(
                "+   i   j       a         b          g     |"
                "     M_R        grid   |     M_T        grid   |  distance",
                debug_output,
            )
            self.assertIn(
                "g   0   0    0.00000     0.0003   -0.50000 |"
                "    0.31509    0.04450 |    0.60672    0.95663 |     0.621",
                debug_output,
            )
            self.assertIn(
                "g  48  94    0.98963     0.7261   -0.50000 |"
                "    0.31509    0.31476 |    0.60672    0.63364 |     0.027",
                debug_output,
            )
            grid_rows = [line for line in debug_output.splitlines() if line.startswith("g ")]
            self.assertEqual(len(grid_rows), 10210)
            self.assertNotIn("recomputing grid", debug_output)
            self.assertNotIn("recomputing adaptive grid", debug_output)
            self._assert_result_file_has_header_and_rows(out_file)

    def test_debug_sphere_gain_uses_cweb_blocks(self):
        """`-x 128` should emit the legacy sphere-gain algebra blocks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_file = os.path.join(tmpdir, "single-row.rxt")
            out_file = os.path.join(tmpdir, "single-row.txt")
            with open(sample_file, "w", encoding="utf-8") as fh:
                fh.write(BIOPIX_851_RXT)

            test_args = ["iadcommand.py", sample_file, "-M", "0", "-x", "128", "-o", out_file]
            with patch("sys.argv", test_args):
                with patch("sys.stderr", new_callable=io.StringIO) as fake_stderr:
                    with self.assertRaises(SystemExit) as cm:
                        iadcommand.main()

            self.assertEqual(cm.exception.code, 0)
            debug_output = fake_stderr.getvalue()
            self.assertIn("SPHERE: REFLECTION", debug_output)
            self.assertIn("SPHERE: TRANSMISSION", debug_output)
            self.assertIn("SPHERE:       baffle = 1", debug_output)
            self.assertIn("SPHERE:       R_u collected = 100.0%", debug_output)
            self.assertIn("SPHERE:       hits sphere wall first =   0.0%", debug_output)
            self.assertIn("SPHERE:       UR1 =   0.025   UR1_calc =   0.000", debug_output)
            self.assertIn("SPHERE:       URU =   0.068   URU_calc =   0.068", debug_output)
            self.assertIn("SPHERE:       G_0 =  14.847        P_0 =   0.000", debug_output)
            self.assertIn("SPHERE:       T_u collected = 100.0%", debug_output)
            self.assertIn("SPHERE:       UT1 =   0.000   UT1_calc =   0.000", debug_output)
            self.assertIn("SPHERE:       Psu =   0.000        Pss =   0.000", debug_output)
            self.assertEqual(debug_output.count("SPHERE: REFLECTION"), 46)
            self.assertEqual(debug_output.count("SPHERE: TRANSMISSION"), 46)
            self.assertNotIn("reflectance sphere MR=", debug_output)
            self.assertNotIn("transmission sphere MT=", debug_output)
            self.assertNotIn("double sphere:", debug_output)
            self._assert_result_file_has_header_and_rows(out_file)

    def test_debug_a_little_uses_cweb_summary_shape(self):
        """`-x 1` should emit the compact CWEB final summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_file = os.path.join(tmpdir, "single-row.rxt")
            out_file = os.path.join(tmpdir, "single-row.txt")
            with open(sample_file, "w", encoding="utf-8") as fh:
                fh.write(SINGLE_ROW_TU_RXT)

            test_args = ["iadcommand.py", sample_file, "-M", "0", "-x", "1", "-o", out_file]
            with patch("sys.argv", test_args):
                with patch("sys.stderr", new_callable=io.StringIO) as fake_stderr:
                    with self.assertRaises(SystemExit) as cm:
                        iadcommand.main()

            self.assertEqual(cm.exception.code, 0)
            debug_output = fake_stderr.getvalue()
            self.assertIn("-------------------NEXT DATA POINT---------------------", debug_output)
            self.assertIn("AD iterations=", debug_output)
            self.assertIn("MC iterations=", debug_output)
            self.assertIn("M_R no corrections", debug_output)
            self.assertIn("M_R + sphere", debug_output)
            self.assertIn("M_R + sphere + mc", debug_output)
            self.assertIn("M_R measured", debug_output)
            self.assertIn("Final distance", debug_output)
            self.assertIn("Failed Search, M_R is too small", debug_output)
            self.assertNotIn("measured_rt corrections", debug_output)
            self._assert_result_file_has_header_and_rows(out_file)

    def test_debug_a_little_shows_mc_iteration_zero_before_mc_loop(self):
        """`-x 1 -M 1` should show the initial CWEB-style MC iteration 0 summary."""

        def _fake_update_lost_light(exp, _a, _b, _g, **_kw):
            exp.ur1_lost = 0.002
            exp.ut1_lost = 0.002
            exp.uru_lost = 0.001
            exp.utu_lost = 0.001
            return 0.002, 0.002, 0.002, 0.001, 0.001

        with tempfile.TemporaryDirectory() as tmpdir:
            sample_file = os.path.join(tmpdir, "single-row.rxt")
            out_file = os.path.join(tmpdir, "single-row.txt")
            with open(sample_file, "w", encoding="utf-8") as fh:
                fh.write(SINGLE_ROW_VARIABLE_RXT)

            test_args = ["iadcommand.py", sample_file, "-M", "1", "-x", "1", "-o", out_file]
            with patch("sys.argv", test_args):
                with patch("iadcommand._discover_mc_lost_binary", return_value="/fake/mc_lost"):
                    with patch.object(iadcommand.iadpython.Experiment, "_update_lost_light", _fake_update_lost_light):
                        with patch("sys.stderr", new_callable=io.StringIO) as fake_stderr:
                            with self.assertRaises(SystemExit) as cm:
                                iadcommand.main()

            self.assertEqual(cm.exception.code, 0)
            debug_output = fake_stderr.getvalue()
            iteration_zero = debug_output.index("MC iterations=  0")
            mc_header = debug_output.index("------------- Monte Carlo Iteration 1 -----------------")
            iteration_one = debug_output.index("MC iterations=  1")
            self.assertLess(iteration_zero, mc_header)
            self.assertLess(mc_header, iteration_one)
            self._assert_result_file_has_header_and_rows(out_file)

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

    def test_baffle_wall_and_lambda_overrides_are_applied(self):
        """`-H`, `-w`, `-W`, and `-L` should update the experiment like iad."""
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
            T=None,
            H=3,
            w=0.88,
            W=0.77,
            L=632.8,
        )

        iadcommand.add_experiment_constraints(exp, args)

        self.assertTrue(exp.r_sphere.baffle)
        self.assertTrue(exp.t_sphere.baffle)
        self.assertEqual(exp.r_sphere.r_wall, 0.88)
        self.assertEqual(exp.t_sphere.r_wall, 0.77)
        self.assertEqual(exp.lambda0, 632.8)


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
            "musp": None,
            "f_r": None,
            "f_t": None,
            "f_wall": None,
            "error": None,
            "M": None,
            "p": None,
            "s": None,
            "x": None,
            "V": 2,
            "X": False,
        }
        args.update(overrides)
        return argparse.Namespace(**args)

    def test_extra_analysis_flags_are_applied(self):
        """Apply -e, -f, -M, -p, -x and -X like iad command-line behavior."""
        exp = iadcommand.iadpython.Experiment()
        args = self._base_analysis_args(error=0.002, f_wall=0.3, M=4, p=-12345, x=8, X=True, s=8, V=1)

        iadcommand.add_analysis_constraints(exp, args)

        self.assertEqual(exp.tolerance, 0.002)
        self.assertEqual(exp.MC_tolerance, 0.002)
        self.assertEqual(exp.f_r, 0.3)
        self.assertEqual(exp.max_mc_iterations, 4)
        self.assertEqual(exp.n_photons, -12345)
        self.assertEqual(exp.debug_level, 8)
        self.assertEqual(exp.method, "comparison")
        self.assertEqual(exp.search_override, "find_ba")
        self.assertEqual(exp.verbosity, 1)

    def test_mua_musp_constraints_fill_ba_and_bs(self):
        """`-A` and `-j` should populate optical-depth constraints."""
        exp = iadcommand.iadpython.Experiment()
        exp.sample.d = 2.0
        args = self._base_analysis_args(mua=0.4, musp=1.2, g=0.2)

        iadcommand.add_analysis_constraints(exp, args)

        self.assertAlmostEqual(exp.default_ba, 0.8)
        self.assertAlmostEqual(exp.default_mus, 1.5)
        self.assertAlmostEqual(exp.default_bs, 3.0)


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

    def test_oblique_incidence_requires_nonnegative_angle(self):
        """`-i` should reject negative angles like the CWEB CLI."""
        exp = iadcommand.iadpython.Experiment()
        args = self._base_sample_args(i=-5, q=12)

        with self.assertRaises(argparse.ArgumentTypeError):
            iadcommand.add_sample_constraints(exp, args)


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

    def test_version_uses_cweb_verbosity_convention(self):
        """`-v` should honor `-V` instead of acting like the old Python long-version flag."""
        test_args = ["iadcommand.py", "-v", "-V", "0"]
        with patch("sys.argv", test_args):
            with patch("sys.stdout", new_callable=io.StringIO) as fake_stdout:
                with self.assertRaises(SystemExit) as cm:
                    iadcommand.main()
        self.assertEqual(cm.exception.code, 0)
        self.assertEqual(fake_stdout.getvalue(), f"iadp {iadcommand.iadpython.__version__}")

    def test_wavelength_limits_filter_file_output(self):
        """`-l` should keep only wavelengths within the requested interval."""
        sample_file = os.path.join(current_dir, "data", "sample-A.rxt")
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as handle:
            out_file = handle.name
        test_args = ["iadcommand.py", sample_file, "-l", "850 900", "-o", out_file]
        try:
            with patch("sys.argv", test_args):
                with self.assertRaises(SystemExit) as cm:
                    iadcommand.main()
            self.assertEqual(cm.exception.code, 0)
            with open(out_file, encoding="utf-8") as fh:
                data_lines = [line for line in fh.readlines() if line.strip() and not line.startswith("#")]
            self.assertEqual(len(data_lines), 6)
        finally:
            if os.path.exists(out_file):
                os.remove(out_file)

    def test_grid_generation_writes_grid_file(self):
        """`-J` should emit a `.grid` file next to the input/output target."""
        source = os.path.join(current_dir, "data", "basic-A.rxt")
        with tempfile.TemporaryDirectory() as tmpdir:
            input_copy = os.path.join(tmpdir, "basic-A.rxt")
            shutil.copyfile(source, input_copy)
            test_args = ["iadcommand.py", input_copy, "-J"]
            with patch("sys.argv", test_args):
                with self.assertRaises(SystemExit) as cm:
                    iadcommand.main()
            self.assertEqual(cm.exception.code, 0)
            grid_file = os.path.join(tmpdir, "basic-A.grid")
            self.assertTrue(os.path.exists(grid_file))
            with open(grid_file, encoding="utf-8") as fh:
                contents = fh.read()
            self.assertIn("a'", contents)


if __name__ == "__main__":
    unittest.main()
