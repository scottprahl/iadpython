# pylint: disable=invalid-name
# pylint: disable=bad-whitespace
# pylint: disable=no-self-use

"""Tests for NIST reflectance data."""

import unittest
import numpy as np
import iadpython


class TestRXT(unittest.TestCase):
    """RXT files data."""

    def test_rxt_01(self):
        """Verify that subject 1 is read correctly."""
        exp = iadpython.read_rxt('./tests/basic-A.rxt')
        self.assertAlmostEqual(exp.m_r[0], 0.299262, delta=1e-5)
        self.assertAlmostEqual(exp.m_r[-1], 0.09662, delta=1e-5)
        self.assertIsNone(exp.m_t)
        self.assertIsNone(exp.m_u)
        self.assertIsNone(exp.lambda0)

    def test_rxt_02(self):
        """Verify that subject 1 is read correctly."""
        exp = iadpython.read_rxt('./tests/basic-B.rxt')
        self.assertAlmostEqual(exp.m_r[0], 0.51485, delta=1e-5)
        self.assertAlmostEqual(exp.m_t[0], 0.19596, delta=1e-5)
        self.assertAlmostEqual(exp.m_r[-1], 0.44875, delta=1e-5)
        self.assertAlmostEqual(exp.m_t[-1], 0.00019, delta=1e-5)
        self.assertIsNone(exp.m_u)
        self.assertIsNone(exp.lambda0)

    def test_rxt_03(self):
        """Verify that subject 1 is read correctly."""
        exp = iadpython.read_rxt('./tests/basic-C.rxt')
        self.assertAlmostEqual(exp.m_r[0], 0.18744, delta=1e-5)
        self.assertAlmostEqual(exp.m_t[0], 0.57620, delta=1e-5)
        self.assertAlmostEqual(exp.m_u[0], 0.00560, delta=1e-5)
        self.assertAlmostEqual(exp.m_r[-1], 0.16010, delta=1e-5)
        self.assertAlmostEqual(exp.m_t[-1], 0.06857, delta=1e-5)
        self.assertAlmostEqual(exp.m_u[-1], 6.341e-12, delta=1e-14)
        self.assertIsNone(exp.lambda0)


if __name__ == '__main__':
    unittest.main()
