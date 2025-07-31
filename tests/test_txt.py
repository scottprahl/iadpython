# pylint: disable=invalid-name

"""Tests for NIST reflectance data."""

import os
import unittest
import iadpython

# Get the directory of the current test file
test_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the data directory relative to the test file
data_dir = os.path.join(test_dir, "data")


class TestTXT(unittest.TestCase):
    """TXT files data."""

    def test_txt_01(self):
        """Verify m_r measurements read correctly."""
        filename = os.path.join(data_dir, "basic-A.txt")
        exp, data = iadpython.read_txt(filename)
        self.assertAlmostEqual(exp.m_r[0], 0.299262, delta=2e-4)
        self.assertAlmostEqual(exp.m_r[-1], 0.09662, delta=2e-4)
        self.assertAlmostEqual(exp.m_t[0], 0)
        self.assertIsNone(exp.m_u)
        self.assertAlmostEqual(data.mr[0], 0.299262, delta=2e-4)
        self.assertAlmostEqual(data.cr[0], 0.299262, delta=2e-4)
        self.assertAlmostEqual(data.mt[0], 0.000000, delta=2e-4)
        self.assertAlmostEqual(data.ct[0], 0.000000, delta=2e-4)
        self.assertAlmostEqual(data.mr[-1], 0.09662, delta=2e-4)
        self.assertAlmostEqual(data.cr[-1], 0.09662, delta=2e-4)
        self.assertAlmostEqual(data.mt[-1], 0.000000, delta=2e-4)
        self.assertAlmostEqual(data.ct[-1], 0.000000, delta=2e-4)

    def test_txt_02(self):
        """Verify m_r and m_t measurements read correctly."""
        filename = os.path.join(data_dir, "basic-B.txt")
        exp, data = iadpython.read_txt(filename)
        self.assertAlmostEqual(exp.m_r[0], 0.51485, delta=2e-4)
        self.assertAlmostEqual(exp.m_t[0], 0.19596, delta=2e-4)
        self.assertAlmostEqual(exp.m_r[-1], 0.44875, delta=2e-4)
        self.assertAlmostEqual(exp.m_t[-1], 0.00019, delta=2e-4)
        self.assertIsNone(exp.m_u)
        self.assertAlmostEqual(data.mr[0], 0.51485, delta=2e-4)
        self.assertAlmostEqual(data.cr[0], 0.51485, delta=2e-4)
        self.assertAlmostEqual(data.mt[0], 0.19596, delta=2e-4)
        self.assertAlmostEqual(data.ct[0], 0.19596, delta=2e-4)
        self.assertAlmostEqual(data.mr[-1], 0.44875, delta=2e-4)
        self.assertAlmostEqual(data.cr[-1], 0.44875, delta=2e-4)
        self.assertAlmostEqual(data.mt[-1], 0.00019, delta=2e-4)
        self.assertAlmostEqual(data.ct[-1], 0.00019, delta=2e-4)

    def test_txt_03(self):
        """Verify m_r, m_t, and m_u measurements read correctly."""
        filename = os.path.join(data_dir, "basic-C.txt")
        exp, data = iadpython.read_txt(filename)
        self.assertAlmostEqual(exp.m_r[0], 0.18744, delta=2e-4)
        self.assertAlmostEqual(exp.m_t[0], 0.57620, delta=2e-4)
        self.assertIsNone(exp.m_u)
        self.assertAlmostEqual(exp.m_r[-1], 0.16010, delta=2e-4)
        self.assertAlmostEqual(exp.m_t[-1], 0.06857, delta=2e-4)
        self.assertAlmostEqual(data.mr[0], 0.18744, delta=2e-4)
        self.assertAlmostEqual(data.cr[0], 0.18744, delta=2e-4)
        self.assertAlmostEqual(data.mt[0], 0.57620, delta=2e-4)
        self.assertAlmostEqual(data.ct[0], 0.57620, delta=2e-4)
        self.assertAlmostEqual(data.mr[-1], 0.16010, delta=2e-4)
        self.assertAlmostEqual(data.cr[-1], 0.16010, delta=2e-4)
        self.assertAlmostEqual(data.mt[-1], 0.06857, delta=2e-4)
        self.assertAlmostEqual(data.ct[-1], 0.06857, delta=2e-4)

    def test_txt_04(self):
        """Verify lambda, m_r, m_t read correctly."""
        filename = os.path.join(data_dir, "sample-A.txt")
        exp, data = iadpython.read_txt(filename)
        self.assertAlmostEqual(exp.lambda0[0], 800)
        self.assertAlmostEqual(exp.m_r[0], 0.16830, delta=2e-4)
        self.assertAlmostEqual(exp.m_t[0], 0.24974, delta=2e-4)
        self.assertAlmostEqual(exp.lambda0[-1], 1000)
        self.assertAlmostEqual(exp.m_r[-1], 0.28689, delta=2e-4)
        self.assertAlmostEqual(exp.m_t[-1], 0.42759, delta=2e-4)
        self.assertIsNone(exp.m_u)
        self.assertAlmostEqual(data.mr[0], 0.16830, delta=2e-4)
        self.assertAlmostEqual(data.cr[0], 0.16830, delta=2e-4)
        self.assertAlmostEqual(data.mt[0], 0.24974, delta=2e-4)
        self.assertAlmostEqual(data.ct[0], 0.24974, delta=2e-4)
        self.assertAlmostEqual(data.mr[-1], 0.28689, delta=2e-4)
        self.assertAlmostEqual(data.cr[-1], 0.28689, delta=2e-4)
        self.assertAlmostEqual(data.mt[-1], 0.42759, delta=2e-4)
        self.assertAlmostEqual(data.ct[-1], 0.42759, delta=2e-4)
        self.assertAlmostEqual(data.lam[-1], 1000)

    def test_txt_05(self):
        """Verify lambda, m_r read correctly."""
        filename = os.path.join(data_dir, "sample-B.txt")
        exp, data = iadpython.read_txt(filename)
        self.assertAlmostEqual(exp.lambda0[0], 800)
        self.assertAlmostEqual(exp.m_r[0], 0.16830, delta=2e-4)
        self.assertAlmostEqual(exp.lambda0[-1], 1000)
        self.assertAlmostEqual(exp.m_r[-1], 0.28689, delta=2e-4)
        self.assertAlmostEqual(exp.m_t[0], 0)
        self.assertIsNone(exp.m_u)
        self.assertAlmostEqual(data.lam[0], 800)
        self.assertAlmostEqual(data.mr[0], 0.16830, delta=2e-4)
        self.assertAlmostEqual(data.cr[0], 0.16830, delta=2e-4)
        self.assertAlmostEqual(data.mr[-1], 0.28689, delta=2e-4)
        self.assertAlmostEqual(data.cr[-1], 0.28689, delta=2e-4)
        self.assertAlmostEqual(data.lam[-1], 1000)
        self.assertAlmostEqual(data.mt[0], 0.000000, delta=2e-4)
        self.assertAlmostEqual(data.ct[0], 0.000000, delta=2e-4)
        self.assertAlmostEqual(data.mt[-1], 0.000000, delta=2e-4)
        self.assertAlmostEqual(data.ct[-1], 0.000000, delta=2e-4)


if __name__ == "__main__":
    unittest.main()
