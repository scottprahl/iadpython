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
import scipy.optimize

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
        self.d_beam = 1
        self.lambda0 = 633

        self.flip_sample = False
        self.fraction_of_rc_in_mr = 1
        self.fraction_of_tc_in_mt = 1

        self.ur1_lost = 0
        self.uru_lost = 0
        self.ut1_lost = 0
        self.utu_lost = 0

    def invert(self):
        m = iadpython.Analysis(self)
        m.check_measurements()
        m.useful_measurements()
        m.determine_search()
        m.initialize_grid()
        return m.invert()


class Analysis():
    """Container class for how analysis is done."""

    def __init__(self, exp):
        """Object initialization."""
        self.exp = exp

        self.default_a = None
        self.default_b = None
        self.default_g = None
        self.default_ba = None
        self.default_bs = None
        self.default_mua = None
        self.default_mus = None

        self.found = False
        self.search = 'unknown'
        self.metric = 1
        self.tolerance = 1
        self.MC_tolerance = 1
        self.final_distance = 1
        self.iterations = 1
        self.error = 1

    def check_measurements(self):
        between = " Must be between 0 and 1."
        if self.exp.m_r is not None:
            if self.exp.m_r < 0 or self.exp.m_r > 1:
                raise "Invalid refl. %.4f" % self.exp.m_r + between

        if self.exp.m_t is not None:
            if self.exp.m_t < 0 or self.exp.m_t > 1:
                raise "Invalid trans. %.4f" % self.exp.m_t + between

        if self.exp.m_u is not None:
            if self.exp.m_u < 0 or self.exp.m_u > 1:
                raise "Invalid unscattered trans. %.4f." % self.exp.m_u + between


    def useful_measurements(self):
        self.useful_measurements = 0
        if self.exp.m_r is not None:
            self.useful_measurements += 1
        if self.exp.m_t is not None:
            self.useful_measurements += 1
        if self.exp.m_u is not None:
            self.useful_measurements += 1


    def determine_one_parameter_search(self):
        """Establish proper search when only one measurement is available."""
        # default case
        self.search = 'find_a'

        # albedo is known
        if self.default_a is not None:
            if self.default_b is None:
                self.search = 'find_b'
            else:
                self.search = 'find_g'

        # optical thickness is known
        elif self.default_b is not None:
            self.search = 'find_a'

        # anisotropy is known
        elif self.default_g is not None:
            self.search = 'find_a'

        # scattering coefficient is known
        elif self.default_bs is not None:
            self.search = 'find_ba'

        # absorption coefficient is known
        elif self.default_ba is not None:
            self.search = 'find_bs'


    def determine_two_parameter_search(self):
        """Establish proper search when two measurements are available."""
        # default case
        self.search = 'find_ab'

        # albedo is known
        if self.default_a is not None:
            self.search = 'find_bg'

        # optical thickness is known
        elif self.default_b is not None:
            self.search = 'find_ag'

        # anisotropy is known
        elif self.default_g is not None:
            self.search = 'find_ab'

        # scattering coefficient is known
        elif self.default_bs is not None:
            self.search = 'find_bag'

        # absorption coefficient is known
        elif self.default_ba is not None:
            self.search = 'find_bsg'


    def determine_search(self):
        """Determine type of search to do."""
        self.search = 'unknown'

        if self.useful_measurements == 0:
            raise "No useful measurements specified"
            return

        elif self.useful_measurements == 1:
            self.determine_one_parameter_search()

        else:
            self.determine_two_parameter_search()


    def initialize_grid(self):
        """Precalculate a grid."""
        return

    def invert(self):
        """Do the inversion."""
        if self.search == 'find_a':
            if self.default_b:
                self.exp.sample.b = default_b
            else:
                self.exp.sample.b = np.inf

            if self.default_g:
                self.exp.sample.g = default_g
            else:
                self.exp.sample.g = 0

            bnds = scipy.optimize.Bounds(np.array([0]),np.array([1]))
            res = scipy.optimize.minimize(afun, 0.3, 
                                          args=[self], 
                                          method='Powell',
                                          bounds=bnds,
                                          tol=1e-5
                                          )
            print(res)
            return self.exp.sample.a, self.exp.sample.b, self.exp.sample.g

        return None, None, None

def afun(x, *args):
    """Vary the albedo."""
    analysis = args[0][0]
    analysis.exp.sample.a = x
    ur1, ut1, _, _ = analysis.exp.sample.rt()

    result = np.abs(ur1 - analysis.exp.m_r)
    if analysis.exp.m_t is not None:
        result += np.abs(ut1 - analysis.exp.m_t)

    return result
    