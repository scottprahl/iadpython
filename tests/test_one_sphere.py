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


class ForwardOneBigSphere(unittest.TestCase):
    """Test forward single sphere calculations."""

    def test_forward_01(self):
        """Big black sphere."""
        s = iad.Sample(a=0.95, b=1)
        ur1, ut1, uru, utu = s.rt()

        r_wall = 0
        d_sphere = 2500
        d_sample = 10
        rsph = iad.Sphere(d_sphere, d_sample, d_entrance=1, r_wall=r_wall)
        M = rsph.multiplier(UR1=ur1, URU=uru, r_wall=r_wall)
        M100 = rsph.multiplier(UR1=1.0, URU=1.0, r_wall=r_wall)
        ur1a = M/M100

        tsph = iad.Sphere(d_sphere, d_sample, d_entrance=10, r_wall=r_wall)
        M = tsph.multiplier(UR1=ut1, URU=uru, r_wall=r_wall)
        M100 = tsph.multiplier(UR1=1, URU=0.0, r_wall=r_wall)
        ut1a = M/M100
        self.assertAlmostEqual(ur1, ur1a, delta=1e-5)
        self.assertAlmostEqual(ut1, ut1a, delta=1e-5)

    def test_forward_02(self):
        """Big gray sphere."""
        s = iad.Sample(a=0.95, b=1)
        ur1, ut1, uru, utu = s.rt()

        r_wall = 0.9
        d_sphere = 2500
        d_sample = 10
        rsph = iad.Sphere(d_sphere, d_sample, d_entrance=1, r_wall=r_wall)
        M = rsph.multiplier(UR1=ur1, URU=uru, r_wall=r_wall)
        M100 = rsph.multiplier(UR1=1.0, URU=1.0, r_wall=r_wall)
        ur1a = M/M100

        tsph = iad.Sphere(d_sphere, d_sample, d_entrance=10, r_wall=r_wall)
        M = tsph.multiplier(UR1=ut1, URU=uru, r_wall=r_wall)
        M100 = tsph.multiplier(UR1=1, URU=0.0, r_wall=r_wall)
        ut1a = M/M100
        self.assertAlmostEqual(ur1, ur1a, delta=1e-5)
        self.assertAlmostEqual(ut1, ut1a, delta=1e-5)

    def test_forward_03(self):
        """Big nearly white sphere."""
        s = iad.Sample(a=0.95, b=1)
        ur1, ut1, uru, utu = s.rt()

        r_wall = 0.98
        d_sphere = 2500
        d_sample = 10
        rsph = iad.Sphere(d_sphere, d_sample, d_entrance=1, r_wall=r_wall)
        M = rsph.multiplier(UR1=ur1, URU=uru, r_wall=r_wall)
        M100 = rsph.multiplier(UR1=1.0, URU=1.0, r_wall=r_wall)
        ur1a = M/M100

        tsph = iad.Sphere(d_sphere, d_sample, d_entrance=10, r_wall=r_wall)
        M = tsph.multiplier(UR1=ut1, URU=uru, r_wall=r_wall)
        M100 = tsph.multiplier(UR1=1, URU=0.0, r_wall=r_wall)
        ut1a = M/M100
        self.assertAlmostEqual(ur1, ur1a, delta=1e-4)
        self.assertAlmostEqual(ut1, ut1a, delta=1e-4)

    def test_forward_04(self):
        """Big white sphere."""
        s = iad.Sample(a=0.95, b=1)
        ur1, ut1, uru, utu = s.rt()

        r_wall = 1.0
        d_sphere = 2500
        d_sample = 10
        rsph = iad.Sphere(d_sphere, d_sample, d_entrance=1, r_wall=r_wall)
        M = rsph.multiplier(UR1=ur1, URU=uru, r_wall=r_wall)
        M100 = rsph.multiplier(UR1=1.0, URU=1.0, r_wall=r_wall)
        ur1a = M/M100

        tsph = iad.Sphere(d_sphere, d_sample, d_entrance=10, r_wall=r_wall)
        M = tsph.multiplier(UR1=ut1, URU=uru, r_wall=r_wall)
        M100 = tsph.multiplier(UR1=1, URU=0.0, r_wall=r_wall)
        ut1a = M/M100

        # actual result is complicated.  
        self.assertGreater(ur1, ur1a)
        self.assertLess(ut1, ut1a)


class ForwardOneMediumSphere(unittest.TestCase):
    """Test medium size forward single sphere calculations."""

    def test_forward_01(self):
        """Big black sphere."""
        s = iad.Sample(a=0.95, b=1)
        ur1, ut1, uru, utu = s.rt()

        r_wall = 0
        d_sphere = 200
        d_sample = 10
        rsph = iad.Sphere(d_sphere, d_sample, d_entrance=1, r_wall=r_wall)
        M = rsph.multiplier(UR1=ur1, URU=uru, r_wall=r_wall)
        M100 = rsph.multiplier(UR1=1.0, URU=1.0, r_wall=r_wall)
        ur1a = M/M100

        tsph = iad.Sphere(d_sphere, d_sample, d_entrance=10, r_wall=r_wall)
        M = tsph.multiplier(UR1=ut1, URU=uru, r_wall=r_wall)
        M100 = tsph.multiplier(UR1=1, URU=0.0, r_wall=r_wall)
        ut1a = M/M100
        self.assertAlmostEqual(ur1, ur1a, delta=2e-4)
        self.assertAlmostEqual(ut1, ut1a, delta=2e-4)

    def test_forward_02(self):
        """Big gray sphere."""
        s = iad.Sample(a=0.95, b=1)
        ur1, ut1, uru, utu = s.rt()

        r_wall = 0.9
        d_sphere = 200
        d_sample = 10
        rsph = iad.Sphere(d_sphere, d_sample, d_entrance=1, r_wall=r_wall)
        M = rsph.multiplier(UR1=ur1, URU=uru, r_wall=r_wall)
        M100 = rsph.multiplier(UR1=1.0, URU=1.0, r_wall=r_wall)
        ur1a = M/M100

        tsph = iad.Sphere(d_sphere, d_sample, d_entrance=10, r_wall=r_wall)
        M = tsph.multiplier(UR1=ut1, URU=uru, r_wall=r_wall)
        M100 = tsph.multiplier(UR1=1, URU=0.0, r_wall=r_wall)
        ut1a = M/M100
        self.assertAlmostEqual(ur1, ur1a, delta=2e-3)
        self.assertAlmostEqual(ut1, ut1a, delta=2e-3)

    def test_forward_03(self):
        """Big nearly white sphere."""
        s = iad.Sample(a=0.95, b=1)
        ur1, ut1, uru, utu = s.rt()

        r_wall = 0.98
        d_sphere = 200
        d_sample = 10
        rsph = iad.Sphere(d_sphere, d_sample, d_entrance=1, r_wall=r_wall)
        M = rsph.multiplier(UR1=ur1, URU=uru, r_wall=r_wall)
        M100 = rsph.multiplier(UR1=1.0, URU=1.0, r_wall=r_wall)
        ur1a = M/M100

        tsph = iad.Sphere(d_sphere, d_sample, d_entrance=10, r_wall=r_wall)
        M = tsph.multiplier(UR1=ut1, URU=uru, r_wall=r_wall)
        M100 = tsph.multiplier(UR1=1, URU=0.0, r_wall=r_wall)
        ut1a = M/M100
        self.assertAlmostEqual(ur1, ur1a, delta=2e-2)
        self.assertAlmostEqual(ut1, ut1a, delta=2e-2)

    def test_forward_04(self):
        """Big white sphere."""
        s = iad.Sample(a=0.95, b=1)
        ur1, ut1, uru, utu = s.rt()

        r_wall = 1.0
        d_sphere = 200
        d_sample = 10
        rsph = iad.Sphere(d_sphere, d_sample, d_entrance=1, r_wall=r_wall)
        M = rsph.multiplier(UR1=ur1, URU=uru, r_wall=r_wall)
        M100 = rsph.multiplier(UR1=1.0, URU=1.0, r_wall=r_wall)
        ur1a = M/M100

        tsph = iad.Sphere(d_sphere, d_sample, d_entrance=10, r_wall=r_wall)
        M = tsph.multiplier(UR1=ut1, URU=uru, r_wall=r_wall)
        M100 = tsph.multiplier(UR1=1, URU=0.0, r_wall=r_wall)
        ut1a = M/M100

        # actual result is complicated.  
        self.assertGreater(ur1, ur1a)
        self.assertLess(ut1, ut1a)


# class MeasureRT(unittest.TestCase):
#     """Test experiment object calculation of measured R and T."""
# 
#     def test_measure_rt_01(self):
#         """Big black sphere."""
#         s = iad.Sample(a=0.95, b=1)
#         ur1, ut1, uru, utu = s.rt()
# 
#         r_wall = 0
#         d_sphere = 200
#         d_sample = 10
#         rsph = iad.Sphere(d_sphere, d_sample, d_entrance=1, r_wall=r_wall)
#         M = rsph.multiplier(UR1=ur1, URU=uru, r_wall=r_wall)
#         M100 = rsph.multiplier(UR1=1.0, URU=1.0, r_wall=r_wall)
#         ur1a = M/M100
# 
#         tsph = iad.Sphere(d_sphere, d_sample, d_entrance=10, r_wall=r_wall)
#         M = tsph.multiplier(UR1=ut1, URU=uru, r_wall=r_wall)
#         M100 = tsph.multiplier(UR1=1, URU=0.0, r_wall=r_wall)
#         ut1a = M/M100
# 
#         exp = iad.Experiment(sample=s, r_sphere=rsph, t_sphere=tsph, num_spheres=1)
#         ur1b, ut1b = exp.measured_rt()
# 
#         self.assertAlmostEqual(ur1a, ur1b, delta=1e-7)
#         self.assertAlmostEqual(ut1a, ut1b, delta=1e-7)
# 
#     def test_measure_rt_02(self):
#         """Big gray sphere."""
#         s = iad.Sample(a=0.95, b=1)
#         ur1, ut1, uru, utu = s.rt()
# 
#         r_wall = 0.9
#         d_sphere = 200
#         d_sample = 10
#         rsph = iad.Sphere(d_sphere, d_sample, d_entrance=1, r_wall=r_wall)
#         M = rsph.multiplier(UR1=ur1, URU=uru, r_wall=r_wall)
#         M100 = rsph.multiplier(UR1=1.0, URU=1.0, r_wall=r_wall)
#         ur1a = M/M100
# 
#         tsph = iad.Sphere(d_sphere, d_sample, d_entrance=10, r_wall=r_wall)
#         M = tsph.multiplier(UR1=ut1, URU=uru, r_wall=r_wall)
#         M100 = tsph.multiplier(UR1=1, URU=0.0, r_wall=r_wall)
#         ut1a = M/M100
#         exp = iad.Experiment(sample=s, r_sphere=rsph, t_sphere=tsph, num_spheres=1)
#         ur1b, ut1b = exp.measured_rt()
# 
#         self.assertAlmostEqual(ur1a, ur1b, delta=1e-7)
#         self.assertAlmostEqual(ut1a, ut1b, delta=1e-7)
# 
#     def test_measure_rt_03(self):
#         """Big nearly white sphere."""
#         s = iad.Sample(a=0.95, b=1)
#         ur1, ut1, uru, utu = s.rt()
# 
#         r_wall = 0.98
#         d_sphere = 200
#         d_sample = 10
#         rsph = iad.Sphere(d_sphere, d_sample, d_entrance=1, r_wall=r_wall)
#         M = rsph.multiplier(UR1=ur1, URU=uru, r_wall=r_wall)
#         M100 = rsph.multiplier(UR1=1.0, URU=1.0, r_wall=r_wall)
#         ur1a = M/M100
# 
#         tsph = iad.Sphere(d_sphere, d_sample, d_entrance=10, r_wall=r_wall)
#         M = tsph.multiplier(UR1=ut1, URU=uru, r_wall=r_wall)
#         M100 = tsph.multiplier(UR1=1, URU=0.0, r_wall=r_wall)
#         ut1a = M/M100
#         exp = iad.Experiment(sample=s, r_sphere=rsph, t_sphere=tsph, num_spheres=1)
#         ur1b, ut1b = exp.measured_rt()
# 
#         self.assertAlmostEqual(ur1a, ur1b, delta=1e-7)
#         self.assertAlmostEqual(ut1a, ut1b, delta=1e-7)
# 
#     def test_measure_rt_04(self):
#         """Big white sphere."""
#         s = iad.Sample(a=0.95, b=1)
#         ur1, ut1, uru, utu = s.rt()
# 
#         r_wall = 1.0
#         d_sphere = 200
#         d_sample = 10
#         rsph = iad.Sphere(d_sphere, d_sample, d_entrance=1, r_wall=r_wall)
#         M = rsph.multiplier(UR1=ur1, URU=uru, r_wall=r_wall)
#         M100 = rsph.multiplier(UR1=1.0, URU=1.0, r_wall=r_wall)
#         ur1a = M/M100
# 
#         tsph = iad.Sphere(d_sphere, d_sample, d_entrance=10, r_wall=r_wall)
#         M = tsph.multiplier(UR1=ut1, URU=uru, r_wall=r_wall)
#         M100 = tsph.multiplier(UR1=1, URU=0.0, r_wall=r_wall)
#         ut1a = M/M100
# 
#         exp = iad.Experiment(sample=s, r_sphere=rsph, t_sphere=tsph, num_spheres=1)
#         ur1b, ut1b = exp.measured_rt()
# 
#         self.assertAlmostEqual(ur1a, ur1b, delta=1e-7)
#         self.assertAlmostEqual(ut1a, ut1b, delta=1e-7)

class InversionRTOneSphere(unittest.TestCase):
    """
    Inversion of measured values using just R and T.
    
    These tests check the round trip process
    
    a,b,g --> M_R, M_T --> a, b, g
    
    with knowledge of the correct g value.
    """

    def test_inversion_01(self):
        """Matched slab with albedo=0.5."""
        s = iad.Sample(a=0.95, b=2, g=0.0)
        ur1, ut1, _, _ = s.rt()
        exp = iad.Experiment(sample=s)
        exp.m_r, exp.m_t = exp.measured_rt()

        a, b, g = exp.invert_rt()
        self.assertAlmostEqual(a, 0.95, delta=1e-3)
        self.assertAlmostEqual(b, 2, delta=1e-3)
        self.assertAlmostEqual(g, 0, delta=1e-3)

    def test_inversion_02(self):
        """Sandwiched slab with albedo=0.95, g=0."""
        s = iad.Sample(a=0.95, b=2, g=0.0)
        s.n = 1.4
        s.n_above = 1.5
        s.n_below = 1.5
        exp = iad.Experiment(sample=s)
        exp.m_r, exp.m_t = exp.measured_rt()

        a, b, g = exp.invert_rt()
        self.assertAlmostEqual(a, 0.95, delta=1e-3)
        self.assertAlmostEqual(b, 2, delta=1e-3)
        self.assertAlmostEqual(g, 0, delta=1e-3)

    def test_inversion_03(self):
        """Sandwiched slab with albedo=0.95, g=0.9."""
        s = iad.Sample(a=0.95, b=2, g=0.9)
        s.n = 1.4
        s.n_above = 1.5
        s.n_below = 1.5
        exp = iad.Experiment(sample=s)
        exp.default_g = 0.9
        exp.m_r, exp.m_t = exp.measured_rt()

        a, b, g = exp.invert_rt()
        self.assertAlmostEqual(a, 0.95, delta=1e-3)
        self.assertAlmostEqual(b, 2, delta=1e-3)
        self.assertAlmostEqual(g, 0.9, delta=1e-3)

class InversionRTUOneSphere(unittest.TestCase):
    """
    Inversion of measured values using R, T, and U.
    
    These tests check the round trip process
    
    a,b,g --> M_R, M_T, M_U --> a, b, g
    """

    def test_01(self):
        """Moderate albedo but g=0.9."""
        s = iad.Sample(a=0.5, b=2, g=0.9)
        ur1, ut1, _, _ = s.rt()
        exp = iad.Experiment(sample=s)
        exp.m_r, exp.m_t = exp.measured_rt()
        exp.sample.a=0
        _, exp.m_u = exp.measured_rt()

        a, b, g = exp.invert_rt()

        self.assertAlmostEqual(a, 0.5, delta=2e-3)
        self.assertAlmostEqual(b, 2, delta=1e-3)
        self.assertAlmostEqual(g, 0.9, delta=1e-3)

    def test_02(self):
        """High albedo and g=0."""
        s = iad.Sample(a=0.95, b=2, g=0.0)
        ur1, ut1, _, _ = s.rt()
        exp = iad.Experiment(sample=s)
        exp.m_r, exp.m_t = exp.measured_rt()
        exp.sample.a=0
        _, exp.m_u = exp.measured_rt()

        a, b, g = exp.invert_rt()

        self.assertAlmostEqual(a, 0.95, delta=1e-3)
        self.assertAlmostEqual(b, 2, delta=1e-3)
        self.assertAlmostEqual(g, 0.0, delta=1e-3)

    def test_03(self):
        """High albedo and moderate g=0.5."""
        s = iad.Sample(a=0.95, b=2, g=0.5)
        ur1, ut1, _, _ = s.rt()
        exp = iad.Experiment(sample=s)
        exp.m_r, exp.m_t = exp.measured_rt()
        exp.sample.a=0
        _, exp.m_u = exp.measured_rt()

        a, b, g = exp.invert_rt()
        
        self.assertAlmostEqual(a, 0.95, delta=2e-3)
        self.assertAlmostEqual(b, 2, delta=1e-3)
        self.assertAlmostEqual(g, 0.5, delta=1e-3)

    def test_04(self):
        """High albedo and high g."""
        s = iad.Sample(a=0.95, b=2, g=0.9)
        ur1, ut1, _, _ = s.rt()
        exp = iad.Experiment(sample=s)
        exp.m_r, exp.m_t = exp.measured_rt()
        exp.sample.a=0
        _, exp.m_u = exp.measured_rt()
#        print("M_R = %.4f" % exp.m_r)
#        print("M_T = %.4f" % exp.m_t)
#        print("M_U = %.4f %.4f" % (exp.m_u, np.exp(-2) ))

        a, b, g = exp.invert_rt()
        
#        print("a = %.4f" % a)
#        print("b = %.4f" % b)
#        print("g = %.4f" % g)

        self.assertAlmostEqual(a, 0.95, delta=1e-3)
        self.assertAlmostEqual(b, 2, delta=1e-3)
        self.assertAlmostEqual(g, 0.9, delta=1e-3)

if __name__ == '__main__':
    unittest.main()
