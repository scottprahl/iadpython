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

    def __init__(self, mr=None, mt=None, mu=None, sample=None, r_sphere=None, t_sphere=None):
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

        self.m_r = mr
        self.m_t = mt
        self.m_u = mu
        
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
        method = iadpython.Analyis(self)
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

        self.found = 1
        self.search = 1
        self.metric = 1
        self.tolerance = 1
        self.MC_tolerance = 1
        self.final_distance = 1
        self.iterations = 1
        self.error = 1

    def check_measurements()
        return

    def determine_measurements()
        return

    def determine_search()
        return

    def initialize_grid()
        return

    def invert()
        return
