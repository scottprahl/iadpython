# pylint: disable=invalid-name
# pylint: disable=no-self-use

"""Tests for Inverse Adding Doubling."""

import unittest
import numpy as np
import iadpython


class IADTest(unittest.TestCase):
    """IAD tests calculations."""

    def test_01(self):
        """No data returns None for optical properties."""
        exp = iadpython.Experiment()
        a, b, g = exp.invert()
        self.assertIsNone(a)
        self.assertIsNone(b)
        self.assertIsNone(g)

    def test_02(self):
        """Matched slab with albedo=0.3."""
        exp = iadpython.Experiment(r= 0.05721)
        a, b, g = exp.invert()
        self.assertAlmostEqual(a,0.3,delta=1e-4)
        self.assertAlmostEqual(b,np.inf)
        self.assertAlmostEqual(g,0)

    def test_03(self):
        """Matched slab with albedo=0."""
        exp = iadpython.Experiment(r=0)
        a, b, g = exp.invert()
        self.assertAlmostEqual(a,0.0,delta=1e-4)
        self.assertAlmostEqual(b,np.inf)
        self.assertAlmostEqual(g,0)

#     def test_04(self):
#         """Matched slab with albedo=1."""
#         exp = iadpython.Experiment(r=1)
#         a, b, g = exp.invert()
#         self.assertAlmostEqual(a,1.0,delta=1e-4)
#         self.assertAlmostEqual(b,np.inf)
#         self.assertAlmostEqual(g,0,delta=1e-4)

    def test_05(self):
        """Matched slab with g=0.9."""
        exp = iadpython.Experiment(r=0.13865, default_g=0.9)
        a, b, g = exp.invert()
        self.assertAlmostEqual(a,0.95,delta=1e-3)
        self.assertAlmostEqual(b,np.inf)
        self.assertAlmostEqual(g,0.9)

    def test_06(self):
        """Matched slab with b=1."""
        exp = iadpython.Experiment(r=0.30172, default_b=1)
        a, b, g = exp.invert()
        self.assertAlmostEqual(a,0.95,delta=1e-3)
        self.assertAlmostEqual(b,1)
        self.assertAlmostEqual(g,0.0)

    def test_07(self):
        """Mismatched slab with albedo=0.95."""
        s = iadpython.Sample(n=1.4)
        exp = iadpython.Experiment(r= 0.38697, sample=s)
        a, b, g = exp.invert()
        self.assertAlmostEqual(a,0.95,delta=2e-2)
        self.assertAlmostEqual(b,np.inf)
        self.assertAlmostEqual(g,0)

    def test_08(self):
        """Mismatched slab glass slide and albedo=0.95."""
        s = iadpython.Sample(n=1.4, n_above=1.5, n_below=1.5)
        exp = iadpython.Experiment(r= 0.39152, sample=s)
        a, b, g = exp.invert()
        self.assertAlmostEqual(a,0.95,delta=2e-2)
        self.assertAlmostEqual(b,np.inf)
        self.assertAlmostEqual(g,0)

if __name__ == '__main__':
    unittest.main()
