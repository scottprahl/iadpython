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

import copy
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

        if self.m_r is None:
            s += "   Reflection               = Missing\n"
        else:
            s += "   Reflection               = %.5f\n" % self.m_r

        if self.m_t is None:
            s += "   Transmission             = Missing\n"
        else:
            s += "   Transmission             = %.5f\n" % self.m_t

        if self.m_u is None:
            s += "   Unscattered Transmission = Missing\n"
        else:
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

    def invert_one(self):
        """Find a,b,g for this experiment."""
        self.check_measurements()
        self.useful_measurements()
        self.determine_search()

        if self.m_r is None and self.m_t is None and self.m_u is None:
            return None, None, None

        if self.default_a:
            self.sample.a = self.default_a
        else:
            self.sample.a = 0

        if self.default_b:
            self.sample.b = self.default_b
        else:
            self.sample.b = np.inf

        if self.default_g:
            self.sample.g = self.default_g
        else:
            self.sample.g = 0

        if self.search == 'find_a':
            res = scipy.optimize.minimize_scalar(afun,
                                          args=(self),
                                          bounds=(0, 1),
                                          method='bounded',
                                          )
            return self.sample.a, self.sample.b, self.sample.g

        if self.search == 'find_b':
            res = scipy.optimize.minimize_scalar(bfun,
                                          args=(self),
                                          bounds=(1,5),
                                          method='brent',
                                          )
            return self.sample.a, self.sample.b, self.sample.g

        if self.search == 'find_g':
            res = scipy.optimize.minimize_scalar(gfun,
                                          args=(self),
                                          bounds=(-1, 1),
                                          method='bounded',
                                          )
            return self.sample.a, self.sample.b, self.sample.g

        if self.search == 'find_ab':
            min_error = np.inf
            for a in np.linspace(0,1,11):
                for b in np.linspace(0,10,11):
                    self.sample.a = a
                    self.sample.b = b
                    ur1, ut1, _, _ = self.sample.rt()
                    error = np.abs(ur1 - self.m_r) + np.abs(ut1 - self.m_t)
                    if error < min_error:
                        min_error=error
                        min_a = a
                        min_b = b

            bnds=scipy.optimize.Bounds(np.array([0,0]), np.array([1,np.inf]))
            res = scipy.optimize.minimize(abfun, [min_a,min_b], args=(self), bounds=bnds, method='Powell')
            return self.sample.a, self.sample.b, self.sample.g

        if self.search == 'find_ag':
            min_error = np.inf
            for a in np.linspace(0,1,11):
                for g in np.linspace(-0.95,0.95,10):
                    self.sample.a = a
                    self.sample.g = g
                    ur1, ut1, _, _ = self.sample.rt()
                    error = np.abs(ur1 - self.m_r) + np.abs(ut1 - self.m_t)
                    if error < min_error:
                        min_error=error
                        min_a = a
                        min_g = g
#                        print("new min %.5f a=%.5f g=%.5f" % (min_error, min_a, min_g))
                        
            bnds=scipy.optimize.Bounds(np.array([0,-1]), np.array([1,1]))
            res = scipy.optimize.minimize(agfun,
                                          [min_a,min_g],
                                          args=(self),
                                          bounds=bnds,
                                          method='Powell',
                                          )
            return self.sample.a, self.sample.b, self.sample.g

        if self.search == 'find_bg':
            min_error = np.inf
            for b in np.linspace(0,10,10):
                for g in np.linspace(-0.95,0.95,10):
                    self.sample.b = b
                    self.sample.g = g
                    ur1, ut1, _, _ = self.sample.rt()
                    error = np.abs(ur1 - self.m_r) + np.abs(ut1 - self.m_t)
                    if error < min_error:
                        min_error=error
                        min_b = b
                        min_g = g
#                        print("new min %.5f b=%.5f g=%.5f" % (min_error, min_b, min_g))

            bnds=scipy.optimize.Bounds(np.array([0,-1]), np.array([np.inf,1]))
            res = scipy.optimize.minimize(bgfun,
                                          [min_b,min_g],
                                          args=(self),
                                          bounds=bnds,
                                          method='Powell',
                                          )
            return self.sample.a, self.sample.b, self.sample.g

        return None, None, None

    def invert(self):
        """Find a,b,g for this experiment."""
        if self.m_r is None and self.m_t is None and self.m_u is None:
            return self.invert_one()

        # any scalar measurement indicates a single data point
        if np.isscalar(self.m_r) or np.isscalar(self.m_t) or np.isscalar(self.m_u):
            return self.invert_one()

        # figure out the number of points that we need to invert
        if self.m_r is not None:
            N = len(self.m_r)
        elif self.m_t is not None:
            N = len(self.m_t)
        else:
            N = len(self.m_u)

        a = np.zeros(N)
        b = np.zeros(N)
        g = np.zeros(N)

        x = copy.deepcopy(self)
        x.m_r = None
        x.m_t = None
        x.m_u = None

        for i in range(N):
            if self.m_r is not None:
                x.m_r = self.m_r[i]
            if self.m_t is not None:
                x.m_t = self.m_t[i]
            if self.m_u is not None:
                x.m_u = self.m_u[i]
            a[i], b[i], g[i] = x.invert_one()

        return a, b, g

def afun(x, *args):
    """Vary the albedo."""
    exp = args[0]
    exp.sample.a = x
    ur1, ut1, _, _ = exp.sample.rt()

    result = 0
    if exp.m_r is not None:
        result += np.abs(ur1 - exp.m_r)
    if exp.m_t is not None:
        result += np.abs(ut1 - exp.m_t)

    return result

def bfun(x, *args):
    """Vary the optical thickness."""
    exp = args[0]
    exp.sample.b = x
    ur1, ut1, _, _ = exp.sample.rt()

    result = 0
    if exp.m_r is not None:
        result += np.abs(ur1 - exp.m_r)
    if exp.m_t is not None:
        result += np.abs(ut1 - exp.m_t)

    return result

def gfun(x, *args):
    """Vary the anisotropy."""
    exp = args[0]
    exp.sample.g = x
    ur1, ut1, _, _ = exp.sample.rt()

    result = 0
    if exp.m_r is not None:
        result += np.abs(ur1 - exp.m_r)
    if exp.m_t is not None:
        result += np.abs(ut1 - exp.m_t)

    return result

def abfun(x, *args):
    """Vary the ab."""
    exp = args[0]
    exp.sample.a = x[0]
    exp.sample.b = x[1]
    ur1, ut1, _, _ = exp.sample.rt()
    return np.abs(ur1 - exp.m_r) + np.abs(ut1 - exp.m_t)

def bgfun(x, *args):
    """Vary the bg."""
    exp = args[0]
    exp.sample.b = x[0]
    exp.sample.g = x[1]
    ur1, ut1, _, _ = exp.sample.rt()
    return np.abs(ur1 - exp.m_r) + np.abs(ut1 - exp.m_t)

def agfun(x, *args):
    """Vary the ag."""
    exp = args[0]
    exp.sample.a = x[0]
    exp.sample.g = x[1]
    ur1, ut1, _, _ = exp.sample.rt()
    return np.abs(ur1 - exp.m_r) + np.abs(ut1 - exp.m_t)
