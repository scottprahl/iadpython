# pylint: disable=invalid-name
# pylint: disable=no-self-use

"""
Tests for calculations with one integrating sphere.

Start by testing to make sure the measured values are close
to those when no sphere is present.

Then make sure that inversion returns the same starting
values.
"""

import unittest
import numpy as np
import iadpython as iad


class ForwardOneSphere(unittest.TestCase):
    """Test forward single sphere calculations."""

    def test_forward_01(self):
        """No reflection or transmission."""
        s = iad.Sample(a=0, b=np.inf)
        ur1, ut1, _, _ = s.rt()

        rsph = iad.Sphere(250,10)
        tsph = iad.Sphere(250,10)
        exp = iad.Experiment(sample=s, r_sphere=rsph, t_sphere=tsph, num_spheres=1)
        ur1a, ut1a = exp.measured_rt()
        self.assertAlmostEqual(ur1, ur1a)
        self.assertAlmostEqual(ut1, ut1a)

    def test_forward_02(self):
        """100% reflection and no transmission."""
        s = iad.Sample(a=1, b=np.inf)
        ur1, ut1, _, _ = s.rt()

        rsph = iad.Sphere(250,10)
        tsph = iad.Sphere(250,10)
        exp = iad.Experiment(sample=s, r_sphere=rsph, t_sphere=tsph, num_spheres=1)
        ur1a, ut1a = exp.measured_rt()
        self.assertAlmostEqual(ur1, ur1a, delta=1e-3)
        self.assertAlmostEqual(ut1, ut1a, delta=1e-5)

    def test_forward_03(self):
        """Some diffuse reflection and no transmission."""
        s = iad.Sample(a=0.5, b=np.inf)
        ur1, ut1, _, _ = s.rt()

        rsph = iad.Sphere(250,10)
        tsph = iad.Sphere(250,10)
        exp = iad.Experiment(sample=s, r_sphere=rsph, t_sphere=tsph, num_spheres=1)
        ur1a, ut1a = exp.measured_rt()
        self.assertAlmostEqual(ur1, ur1a, delta=1e-3)
        self.assertAlmostEqual(ut1, ut1a, delta=1e-5)

    def test_forward_04(self):
        """Normal transmission sphere."""
        s = iad.Sample(a=0.5, b=1, g=0.5)
        ur1, ut1, _, _ = s.rt()
        exp = iad.Experiment(sample=s)
        exp.num_spheres = 1
        exp.r_sphere = iad.Sphere(200,25)
        exp.t_sphere = iad.Sphere(200,25)
        ur1a, ut1a = exp.measured_rt()
        self.assertAlmostEqual(ur1, ur1a)
        self.assertAlmostEqual(ut1, ut1a)

class InversionOneSphere(unittest.TestCase):
    """Inversion of measured values."""

    def test_inversion_01(self):
        """Matched slab with albedo=0.5."""
        s = iad.Sample(a=0.95, b=2, g=0.0)
        ur1, ut1, _, _ = s.rt()
        exp = iad.Experiment(sample=s)
        exp.m_r, exp.m_t = exp.measured_rt()

        a, b, g = exp.invert_rt()
        self.assertAlmostEqual(a, 0.95)
        self.assertAlmostEqual(b, 2)
        self.assertAlmostEqual(g, 0, delta=1e-3)

    def test_inversion_02(self):
        """Matched slab with albedo=0.5."""
        s = iad.Sample(a=0.95, b=2, g=0.0)
        s.n = 1.4
        s.n_above = 1.5
        s.n_below = 1.5
        exp = iad.Experiment(sample=s)
        exp.m_r, exp.m_t = exp.measured_rt()

        a, b, g = exp.invert_rt()
        self.assertAlmostEqual(a, 0.95)
        self.assertAlmostEqual(b, 2)
        self.assertAlmostEqual(g, 0, delta=1e-3)

if __name__ == '__main__':
    unittest.main()
