# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments

"""
Class for doing inverse adding-doubling calculations for a sample.

    import iadpython

    exp = iadpython.Experiment(0.5,0.1,0.01)
    a, b, g = exp.invert()
    print("a = %7.3f" % a)
    print("b = %7.3f" % b)
    print("g = %7.3f" % g)

    s = iadpython.Sample(0.5,0.1,0.01,n=1.4, n_top=1.5, n_bottom=1.5)
    exp = iadpython.Experiment(s)
    a, b, g = exp.invert()
    print("a = %7.3f" % a)
    print("b = %7.3f" % b)
    print("g = %7.3f" % g)
"""

import copy
import numpy as np
import iadpython


class Experiment():
    """Container class for details of an experiment."""

    def __init__(self, r=None, t=None, u=None, sample=None, r_sphere=None, t_sphere=None):
        """Object initialization."""
        if sample == None:
            self.sample = iadpython.Sample()
        else:
            self.sample = sample

        self.r_sphere = r_sphere
        self.t_sphere = t_sphere
        self.num_spheres = 2
        if r_sphere == None:
            self.num_spheres -= 1
        if t_sphere == None:
            self.num_spheres -= 1

        self.m_r = r
        self.m_t = t
        self.m_u = u
        
        self.useful_measurements = 0
        
        # these will have to be eventually supported
        self.default_a = None
        self.default_b = None
        self.default_g = None
        self.default_ba = None
        self.default_bs = None
        self.default_mua = None
        self.default_mus = None

        self.d_beam = 1
        self.lambda0 = 633

        self.flip_sample = False
        self.fraction_of_rc_in_mr = 1
        self.fraction_of_tc_in_mt = 1

        self.ur1_lost = 0
        self.uru_lost = 0
        self.ut1_lost = 0
        self.utu_lost = 0

    def invert():
        method = iadpython.Analysis(self)
        method.check_measurements()
        method.useful_measurements()
        method.determine_search()
        method.initialize_grid()
        method.invert()
        return method.found_a, method.found_b, method.found_g


class Analysis():
    """Container class for how analysis is done."""

    def __init__(self, exp):
        """Object initialization."""
        self.exp = exp
        self.found_a = None
        self.found_b = None
        self.found_g = None

        self.found = False
        self.search = 'unknown'
        self.metric = 1
        self.tolerance = 1
        self.MC_tolerance = 1
        self.final_distance = 1
        self.iterations = 1
        self.error = 1

    def check_measurements():
        between = " Must be between 0 and 1."
        if self.exp.r is not None:
            if self.exp.r < 0 or self.exp.r > 1:
                raise "Invalid refl. %.4f" % self.exp.r + between

        if self.exp.t is not None:
            if self.exp.t < 0 or self.exp.t > 1:
                raise "Invalid trans. %.4f" % self.exp.t + between

        if self.exp.u is not None:
            if self.exp.u < 0 or self.exp.u > 1:
                raise "Invalid unscattered trans. %.4f." % self.exp.u + between

    def useful_measurements():
        self.useful_measurements = 0
        if self.exp.r is not None:
            self.useful_measurements += 1
        if self.exp.t is not None:
            self.useful_measurements += 1
        if self.exp.u is not None:
            self.useful_measurements += 1

    def determine_search():
        if self.useful_measurements == 0:
            raise "No useful measurements specified"
            return
        
        if useful_measurements == 1:
            if self.exp.r is not None:
                self.search = 'find_a'
                return
            if self.exp.t is not None:
                self.search = 'find_a'
                return
            if self.exp.u is not None:
                self.search = 'find_b'
                return
        
        

        return

    def initialize_grid():
        return

    def invert()
        return
