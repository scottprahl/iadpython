# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=consider-using-f-string

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

import numpy as np
import scipy.optimize
import iadpython

class Experiment():
    """Container class for details of an experiment."""

    def __init__(self,
                 r=None, t=None, u=None,
                 sample=None, r_sphere=None, t_sphere=None,
                 default_a=None, default_b=None, default_g=None):
        """Object initialization."""
        if sample is None:
            self.sample = iadpython.Sample()
        else:
            self.sample = sample

        self.r_sphere = r_sphere
        self.t_sphere = t_sphere
        self.num_spheres = 2
        if r_sphere is None:
            self.num_spheres -= 1
        if t_sphere is None:
            self.num_spheres -= 1

        self.m_r = r
        self.m_t = t
        self.m_u = u

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

        self.default_a = default_a
        self.default_b = default_b
        self.default_g = default_g
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
        self.num_measurements = 0
        self.grid = None

    def __str__(self):
        """Return basic details as a string for printing."""
        s = "---------------- Sample ---------------\n"
        s += self.sample.__str__()
        s += "\n--------------- Spheres ---------------\n"
        if self.num_spheres == 0:
            s += "No spheres used.\n"
        if self.r_sphere is not None:
            s += "Reflectance Sphere--------\n"
            s += self.r_sphere.__str__()
        if self.t_sphere is not None:
            s += "Transmission Sphere--------\n"
            s += self.t_sphere.__str__()
        s += "\n------------- Measurements ------------\n"

        s += "   Reflection               = %.5f\n" % self.m_r
        s += "   Transmission             = %.5f\n" % self.m_t
        s += "   Unscattered Transmission = %.5f\n" % self.m_r
        return s

    def check_measurements(self):
        """Make sure measurements are sane."""
        between = " Must be between 0 and 1."
        if self.m_r is not None:
            if self.m_r < 0 or self.m_r > 1:
                raise "Invalid refl. %.4f" % self.m_r + between

        if self.m_t is not None:
            if self.m_t < 0 or self.m_t > 1:
                raise "Invalid trans. %.4f" % self.m_t + between

        if self.m_u is not None:
            if self.m_u < 0 or self.m_u > 1:
                raise "Invalid unscattered trans. %.4f." % self.m_u + between


    def useful_measurements(self):
        """Count the number of useful measurements."""
        self.num_measurements = 0
        if self.m_r is not None:
            self.num_measurements += 1
        if self.m_t is not None:
            self.num_measurements += 1
        if self.m_u is not None:
            self.num_measurements += 1


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
        if self.num_measurements == 0:
            self.search = 'unknown'

        if self.num_measurements == 1:
            self.determine_one_parameter_search()

        else:
            self.determine_two_parameter_search()

    def initialize_grid(self):
        """Precalculate a grid."""
        self.grid = None

    def invert(self):
        """Find a,b,g for this experiment."""
        self.check_measurements()
        self.useful_measurements()
        self.determine_search()
        self.initialize_grid()


        if self.search == 'find_a':
            if self.default_b:
                self.sample.b = self.default_b
            else:
                self.sample.b = np.inf

            if self.default_g:
                self.sample.g = self.default_g
            else:
                self.sample.g = 0

            bnds = scipy.optimize.Bounds(np.array([0]),np.array([1]))
            res = scipy.optimize.minimize(afun, 0.5,
                                          args=[self],
                                          method='Powell',
                                          bounds=bnds,
                                          tol=1e-5
                                          )
            print(res)
            return self.sample.a, self.sample.b, self.sample.g

        return None, None, None

def afun(x, *args):
    """Vary the albedo."""
    analysis = args[0][0]
    analysis.sample.a = x
    ur1, ut1, _, _ = analysis.sample.rt()

    result = np.abs(ur1 - analysis.m_r)
    if analysis.m_t is not None:
        result += np.abs(ut1 - analysis.m_t)

    return result
    