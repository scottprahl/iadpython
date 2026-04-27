"""Tests for the iadplus notebook generator."""

import os
import runpy
import subprocess
import tempfile
import unittest
from unittest.mock import patch


test_dir = os.path.dirname(os.path.abspath(__file__))
iadplus_path = os.path.join(os.path.dirname(test_dir), "iadpython", "iadplus")
iadplus = runpy.run_path(iadplus_path, run_name="__iadplus_test__")


class TestIADPlus(unittest.TestCase):
    """iadplus result parsing."""

    def test_parse_iadp_txt_uses_status_column(self):
        """Rows marked with `+` should parse as failed, not crash as floats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "status.txt")
            with open(filename, "w", encoding="utf-8") as handle:
                handle.write("# result table\n")
                handle.write(" 890.0\t0.3272\t0.3552\t0.5291\t0.5291\t0.0000\t1.3851\t0.5822\t + \n")
                handle.write(" 898.3\t0.3233\t0.3233\t0.5281\t0.5280\t0.0104\t1.2499\t0.5773\t * \n")

            lam, _mr, _cr, _mt, _ct, _mua, _musp, _g, success = iadplus["parse_iadp_txt"](filename)

        self.assertAlmostEqual(lam[0], 890.0)
        self.assertEqual(success.tolist(), [False, True])

    def test_run_iadp_preserves_live_stderr_progress(self):
        """Iadplus should let iadp stderr progress reach the terminal."""
        completed = subprocess.CompletedProcess(["iadp"], 0, stdout="")

        with patch.object(iadplus["subprocess"], "run", return_value=completed) as mock_run:
            ok = iadplus["run_iadp"]("sample.rxt", "sample.txt", "-c 0")

        self.assertTrue(ok)
        _cmd, kwargs = mock_run.call_args
        self.assertEqual(kwargs["stdout"], subprocess.PIPE)
        self.assertTrue(kwargs["text"])
        self.assertNotIn("stderr", kwargs)
        self.assertNotIn("capture_output", kwargs)


if __name__ == "__main__":
    unittest.main()
