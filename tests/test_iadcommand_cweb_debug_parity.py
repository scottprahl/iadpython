"""Fast iadp/iad debug-output parity checks for disabled Monte Carlo."""

import pathlib
import re
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
C_IAD = REPO_ROOT / "iad" / "iad"


def _single_data_row(source):
    """Return an RXT file body with only the first measurement row."""
    lines = source.read_text(encoding="utf-8").splitlines()
    result = []
    for line in lines:
        result.append(line)
        fields = line.split()
        if len(fields) >= 2:
            try:
                float(fields[0])
                float(fields[1])
            except ValueError:
                continue
            break
    return "\n".join(result) + "\n"


def _run(command):
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _run_iadp(sample_file):
    return _run([sys.executable, "-m", "iadpython.iadcommand", str(sample_file), "-M", "0", "-x", "8"])


def _run_iad(sample_file):
    return _run([str(C_IAD), "-M", "0", "-x", "8", str(sample_file)])


def _debug_rows(stderr):
    rows = []
    for line in stderr.splitlines():
        fields = [field for field in line.split() if field != "|"]
        if len(fields) >= 15 and fields[-1] in {"*", "+"}:
            rows.append(fields)
    return rows


class CwebDebugParityTest(unittest.TestCase):
    """Compare iadp and CWEB iad -x 8 debug tables when MC is disabled."""

    @classmethod
    def setUpClass(cls):
        if not C_IAD.exists():
            raise unittest.SkipTest(f"{C_IAD} is not built")

    def assertDebugTableParity(self, fixture_name):
        source = REPO_ROOT / "tests" / "data" / fixture_name
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_file = pathlib.Path(tmpdir) / fixture_name
            sample_file.write_text(_single_data_row(source), encoding="utf-8")

            py = _run_iadp(sample_file)
            c = _run_iad(sample_file)

        for completed in (py, c):
            self.assertIn("#      | Meas      M_R  | Meas      M_T", completed.stderr)
            self.assertIn("Lost   Lost   Lost   Lost  | MC   IAD  Error", completed.stderr)

        py_rows = _debug_rows(py.stderr)
        c_rows = _debug_rows(c.stderr)
        self.assertEqual(len(py_rows), 1)
        self.assertEqual(len(c_rows), 1)

        py_row = py_rows[0]
        c_row = c_rows[0]
        self.assertEqual(py_row[8:12], ["0.0000", "0.0000", "0.0000", "0.0000"])
        self.assertEqual(c_row[8:12], ["0.0000", "0.0000", "0.0000", "0.0000"])
        self.assertEqual(py_row[12], "0")
        self.assertEqual(c_row[12], "0")
        self.assertRegex(py_row[-1], re.compile(r"[*+]"))
        self.assertRegex(c_row[-1], re.compile(r"[*+]"))

    def test_zero_sphere_debug_table_matches_iad(self):
        self.assertDebugTableParity("basic-D.rxt")

    def test_one_sphere_debug_table_matches_iad(self):
        self.assertDebugTableParity("sample-C.rxt")

    def test_two_sphere_debug_table_matches_iad(self):
        self.assertDebugTableParity("sample-E.rxt")


if __name__ == "__main__":
    unittest.main()
