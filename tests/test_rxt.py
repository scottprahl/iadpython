# pylint: disable=invalid-name
# pylint: disable=bad-whitespace

"""Tests for NIST reflectance data."""

import os
import unittest
import iadpython

# Get the directory of the current test file
test_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the data directory relative to the test file
data_dir = os.path.join(test_dir, 'data')


class TestRXT(unittest.TestCase):
    """RXT files data."""

    def test_rxt_01(self):
        """Verify m_r measurements read correctly."""
        filename = os.path.join(data_dir, 'basic-A.rxt')
        exp = iadpython.read_rxt(filename)
        print(exp.lambda0)
        self.assertAlmostEqual(exp.m_r[0], 0.299262, delta=1e-5)
        self.assertAlmostEqual(exp.m_r[-1], 0.09662, delta=1e-5)
        self.assertIsNone(exp.m_t)
        self.assertIsNone(exp.m_u)
        self.assertIsNone(exp.lambda0)

    def test_rxt_02(self):
        """Verify m_r and m_t measurements read correctly."""
        filename = os.path.join(data_dir, 'basic-B.rxt')
        exp = iadpython.read_rxt(filename)
        self.assertAlmostEqual(exp.m_r[0], 0.51485, delta=1e-5)
        self.assertAlmostEqual(exp.m_t[0], 0.19596, delta=1e-5)
        self.assertAlmostEqual(exp.m_r[-1], 0.44875, delta=1e-5)
        self.assertAlmostEqual(exp.m_t[-1], 0.00019, delta=1e-5)
        self.assertIsNone(exp.m_u)
        self.assertIsNone(exp.lambda0)

    def test_rxt_03(self):
        """Verify m_r, m_t, and m_u measurements read correctly."""
        filename = os.path.join(data_dir, 'basic-C.rxt')
        exp = iadpython.read_rxt(filename)
        self.assertAlmostEqual(exp.m_r[0], 0.18744, delta=1e-5)
        self.assertAlmostEqual(exp.m_t[0], 0.57620, delta=1e-5)
        self.assertAlmostEqual(exp.m_u[0], 0.00560, delta=1e-5)
        self.assertAlmostEqual(exp.m_r[-1], 0.16010, delta=1e-5)
        self.assertAlmostEqual(exp.m_t[-1], 0.06857, delta=1e-5)
        self.assertAlmostEqual(exp.m_u[-1], 6.341e-12, delta=1e-14)
        self.assertIsNone(exp.lambda0)

    def test_rxt_04(self):
        """Verify lambda, m_r, m_t read correctly."""
        filename = os.path.join(data_dir, 'sample-A.rxt')
        exp = iadpython.read_rxt(filename)
        self.assertAlmostEqual(exp.lambda0[0], 800)
        self.assertAlmostEqual(exp.m_r[0], 0.16830, delta=1e-5)
        self.assertAlmostEqual(exp.m_t[0], 0.24974, delta=1e-5)
        self.assertAlmostEqual(exp.lambda0[-1], 1000)
        self.assertAlmostEqual(exp.m_r[-1], 0.28689, delta=1e-5)
        self.assertAlmostEqual(exp.m_t[-1], 0.42759, delta=1e-5)
        self.assertIsNone(exp.m_u)

    def test_rxt_05(self):
        """Verify lambda, m_r read correctly."""
        filename = os.path.join(data_dir, 'sample-B.rxt')
        exp = iadpython.read_rxt(filename)
        self.assertAlmostEqual(exp.lambda0[0], 800)
        self.assertAlmostEqual(exp.m_r[0], 0.16830, delta=1e-5)
        self.assertAlmostEqual(exp.lambda0[-1], 1000)
        self.assertAlmostEqual(exp.m_r[-1], 0.28689, delta=1e-5)
        self.assertIsNone(exp.m_t)
        self.assertIsNone(exp.m_u)


if __name__ == '__main__':
    unittest.main()
