# pylint: disable=invalid-name
# pylint: disable=no-self-use

"""Tests for Inverse Adding Doubling."""

import unittest
import numpy as np
import iadpython


class layer(unittest.TestCase):
    """Starting layer calculations."""

    def test_01(self):
        exp = iadpython.Experiment(0.5,0.1,0.01)
        a, b, g = exp.invert()
        self.assertIsNone(a)
        self.assertIsNone(b)
        self.assertIsNone(g)

    def test_02(self):
        exp = iadpython.Experiment(r=0.11523)
        a, b, g = exp.invert()
        self.assert_approx_equal(a,0.5)
        self.assert_approx_equal(b,np.inf)
        self.assert_approx_equal(g,0)


if __name__ == '__main__':
    unittest.main()
