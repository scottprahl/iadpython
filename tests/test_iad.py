# pylint: disable=invalid-name
# pylint: disable=no-self-use

"""Tests for Inverse Adding Doubling."""

import unittest
import numpy as np
import iadpython


class IADTest(unittest.TestCase):
    """IAD tests calculations."""

    def test_01(self):
        exp = iadpython.Experiment(0.5,0.1,0.01)
        a, b, g = exp.invert()
        self.assertIsNone(a)
        self.assertIsNone(b)
        self.assertIsNone(g)

    def test_02(self):
        exp = iadpython.Experiment(r=0.11523)
        a, b, g = exp.invert()
        self.assertAlmostEqual(a,0.5,delta=1e-4)
        self.assertAlmostEqual(b,np.inf)
        self.assertAlmostEqual(g,0,delta=1e-4)

    def test_03(self):
        exp = iadpython.Experiment(r=0)
        a, b, g = exp.invert()
        self.assertAlmostEqual(a,0.0,delta=1e-4)
        self.assertAlmostEqual(b,np.inf)
        self.assertAlmostEqual(g,0,delta=1e-4)

    def test_04(self):
        exp = iadpython.Experiment(r=1)
        a, b, g = exp.invert()
        self.assertAlmostEqual(a,1.0,delta=1e-4)
        self.assertAlmostEqual(b,np.inf)
        self.assertAlmostEqual(g,0,delta=1e-4)

if __name__ == '__main__':
    unittest.main()
