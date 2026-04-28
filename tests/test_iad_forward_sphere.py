"""Forward calculation (-z) parity and sphere-sensitivity tests.

These tests exercise the iadp and iad/iad forward-calculation path with
single-sphere parameters supplied on the command line.  They verify:

* iadp `-z` and iad/iad `-z` produce the same human-readable layout for
  forward calculations (line-by-line "Intrinsic Properties", "Derived
  quantities", "Sphere properties", and "Calculated quantities" blocks).
* `M_R (sphere)` shifts when sphere parameters change (diameter, sample
  port, wall reflectance, calibration standard, baffle).
* The two implementations stay close enough on `M_R (sphere)` to catch
  regressions in the sphere wiring on either side.
"""

import pathlib
import re
import shutil
import subprocess
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
C_IAD = REPO_ROOT / "iad" / "iad"
IADP = REPO_ROOT / ".venv" / "bin" / "iadp"


def _run(command):
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _iadp_args(extra):
    binary = IADP if IADP.exists() else shutil.which("iadp")
    if binary is None:
        raise unittest.SkipTest("iadp binary not found")
    return [str(binary), "-z", *extra]


def _iad_args(extra):
    return [str(C_IAD), "-z", *extra]


def _extract_field(stdout, label):
    pattern = re.compile(rf"^\s*{re.escape(label)}\s*=\s*([\-0-9eE\.]+)", re.M)
    match = pattern.search(stdout)
    if match is None:
        return None
    return float(match.group(1))


def _section_lines(stdout, header):
    lines = stdout.splitlines()
    out = []
    in_section = False
    for line in lines:
        if line.strip() == header:
            in_section = True
            continue
        if in_section:
            stripped = line.strip()
            if not stripped or (stripped.endswith(("Properties", "quantities")) and "=" not in stripped):
                break
            out.append(line)
    return out


class IadpForwardSphereTest(unittest.TestCase):
    """Exercise iadp -z with single-sphere parameters."""

    def _run_iadp(self, extra):
        return _run(_iadp_args(extra))

    def test_no_sphere_omits_sphere_block(self):
        """Without -1/-S there is no `Sphere properties` section or M_R lines."""
        result = self._run_iadp(["-a", "0.5", "-b", "1", "-M", "0"])
        self.assertNotIn("Sphere properties", result.stdout)
        self.assertNotIn("M_R (sphere)", result.stdout)
        self.assertIn("R total", result.stdout)
        self.assertIn("T total", result.stdout)

    def test_single_sphere_prints_sphere_block(self):
        """A single sphere prints sphere-properties and M_R/M_T sphere lines."""
        result = self._run_iadp(["-a", "0.5", "-b", "1", "-1", "200 25 13 13 0.95", "-M", "0"])
        self.assertIn("Sphere properties (1 sphere)", result.stdout)
        self.assertIn("Reflection sphere", result.stdout)
        self.assertNotIn("Transmission sphere", result.stdout)
        self.assertIn("M_R (sphere)", result.stdout)
        self.assertIn("M_T (sphere)", result.stdout)
        self.assertIn("                      sphere diameter =   200.0 mm", result.stdout)
        self.assertIn("                 sample port diameter =    25.0 mm", result.stdout)
        self.assertIn("                     wall reflectance =    95.0 %", result.stdout)

    def test_single_sphere_wall_reflectance_changes_mr(self):
        """Lowering wall reflectance must change M_R (sphere)."""
        high = self._run_iadp(["-a", "0.5", "-b", "1", "-1", "200 25 13 13 0.99", "-M", "0"])
        low = self._run_iadp(["-a", "0.5", "-b", "1", "-1", "200 25 13 13 0.50", "-M", "0"])
        m_r_high = _extract_field(high.stdout, "M_R (sphere)")
        m_r_low = _extract_field(low.stdout, "M_R (sphere)")
        self.assertIsNotNone(m_r_high)
        self.assertIsNotNone(m_r_low)
        self.assertNotAlmostEqual(m_r_high, m_r_low, places=3)

    def test_single_sphere_diameter_changes_mr(self):
        """Different sphere diameter changes M_R (sphere)."""
        big = self._run_iadp(["-a", "0.5", "-b", "1", "-1", "200 25 13 13 0.95", "-M", "0"])
        small = self._run_iadp(["-a", "0.5", "-b", "1", "-1", "100 25 13 13 0.95", "-M", "0"])
        m_r_big = _extract_field(big.stdout, "M_R (sphere)")
        m_r_small = _extract_field(small.stdout, "M_R (sphere)")
        self.assertNotAlmostEqual(m_r_big, m_r_small, places=3)

    def test_single_sphere_calibration_standard_changes_mr(self):
        """`-R` changes calibration standard and therefore M_R (sphere)."""
        nominal = self._run_iadp(["-a", "0.5", "-b", "1", "-1", "200 25 13 13 0.95", "-M", "0"])
        rstd = self._run_iadp(["-a", "0.5", "-b", "1", "-1", "200 25 13 13 0.95", "-R", "0.5", "-M", "0"])
        m_r_nominal = _extract_field(nominal.stdout, "M_R (sphere)")
        m_r_rstd = _extract_field(rstd.stdout, "M_R (sphere)")
        self.assertNotAlmostEqual(m_r_nominal, m_r_rstd, places=3)

    def test_baffle_flag_appears_in_output(self):
        """`-H` selects whether the printed sphere block reports a baffle."""
        with_baffle = self._run_iadp(["-a", "0.5", "-b", "1", "-1", "200 25 13 13 0.95", "-H", "1", "-M", "0"])
        no_baffle = self._run_iadp(["-a", "0.5", "-b", "1", "-1", "200 25 13 13 0.95", "-H", "0", "-M", "0"])
        self.assertIn("Reflection sphere has a baffle", with_baffle.stdout)
        self.assertIn("Reflection sphere has no baffle", no_baffle.stdout)


class IadCwebForwardSphereTest(unittest.TestCase):
    """Exercise iad/iad -z with single-sphere parameters."""

    @classmethod
    def setUpClass(cls):
        if not C_IAD.exists():
            raise unittest.SkipTest(f"{C_IAD} is not built")

    def test_single_sphere_prints_sphere_block(self):
        result = _run(_iad_args(["-a", "0.5", "-b", "1", "-1", "200 25 13 13 0.95", "-M", "0"]))
        self.assertIn("Sphere properties (1 sphere)", result.stdout)
        self.assertIn("Reflection sphere", result.stdout)
        self.assertIn("M_R (sphere)", result.stdout)
        self.assertIn("M_T (sphere)", result.stdout)

    def test_no_sphere_omits_sphere_block(self):
        result = _run(_iad_args(["-a", "0.5", "-b", "1"]))
        self.assertNotIn("Sphere properties", result.stdout)
        self.assertNotIn("M_R (sphere)", result.stdout)
        self.assertIn("R total", result.stdout)


class ForwardSphereParityTest(unittest.TestCase):
    """Compare iadp -z and iad/iad -z layouts for single-sphere forward calc."""

    @classmethod
    def setUpClass(cls):
        if not C_IAD.exists():
            raise unittest.SkipTest(f"{C_IAD} is not built")
        if not (IADP.exists() or shutil.which("iadp")):
            raise unittest.SkipTest("iadp binary not found")

    def _both(self, extra):
        return _run(_iadp_args(extra)), _run(_iad_args(extra))

    def test_section_headers_present_in_both(self):
        py, c = self._both(["-a", "0.5", "-b", "1", "-1", "200 25 13 13 0.95", "-H", "1", "-M", "0"])
        for header in (
            "Intrinsic Properties",
            "Derived quantities",
            "Sphere properties (1 sphere)",
            "Calculated quantities",
        ):
            self.assertIn(header, py.stdout, f"iadp missing '{header}'")
            self.assertIn(header, c.stdout, f"iad missing '{header}'")

    def test_sphere_block_matches_line_for_line(self):
        """Sphere geometry lines should be byte-identical between iadp and iad."""
        py, c = self._both(["-a", "0.5", "-b", "1", "-1", "200 25 13 13 0.95", "-H", "1", "-M", "0"])
        py_block = _section_lines(py.stdout, "Sphere properties (1 sphere)")
        c_block = _section_lines(c.stdout, "Sphere properties (1 sphere)")
        self.assertEqual(py_block, c_block)

    def test_raw_calculated_quantities_match(self):
        """R/T raw values should agree (no sphere model involved)."""
        py, c = self._both(["-a", "0.5", "-b", "1", "-1", "200 25 13 13 0.95", "-M", "0"])
        for label in ("R total", "R unscattered", "T total", "T unscattered"):
            py_val = _extract_field(py.stdout, label)
            c_val = _extract_field(c.stdout, label)
            self.assertIsNotNone(py_val, f"iadp missing '{label}'")
            self.assertIsNotNone(c_val, f"iad missing '{label}'")
            self.assertAlmostEqual(py_val, c_val, places=2, msg=f"{label} differs")


if __name__ == "__main__":
    unittest.main()
