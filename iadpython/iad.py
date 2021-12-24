# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=consider-using-f-string
# pylint: disable=unused-argument

"""
Class for doing inverse adding-doubling calculations for a sample.

    Example:
        >>> import iadpython as iad

        >>> exp = iad.Experiment(0.5,0.1,0.01)
        >>> a, b, g = exp.invert()
        >>> print("a = %7.3f" % a)
        >>> print("b = %7.3f" % b)
        >>> print("g = %7.3f" % g)

        >>> s = iad.Sample(0.5,0.1,0.01,n=1.4, n_top=1.5, n_bottom=1.5)
        >>> exp = iad.Experiment(s)
        >>> a, b, g = exp.invert()
        >>> print("a = %7.3f" % a)
        >>> print("b = %7.3f" % b)
        >>> print("g = %7.3f" % g)
"""

import copy
import numpy as np
from scipy.optimize import minimize, minimize_scalar, Bounds
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
        self.lambda0 = None

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
        self.num_measures = 0
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

        s += "   Reflection               = "
        if self.m_r is None:
            s += "Missing\n"
        elif np.isscalar(self.m_r):
            s += "%.5f\n" % self.m_r
        else:
            s += "%s\n" % self.m_r.__str__()

        s += "   Transmission             = "
        if self.m_t is None:
            s += "Missing\n"
        elif np.isscalar(self.m_r):
            s += "%.5f\n" % self.m_t
        else:
            s += "%s\n" % self.m_t.__str__()

        s += "   Unscattered Transmission = "
        if self.m_u is None:
            s += "Missing\n"
        elif np.isscalar(self.m_r):
            s += "%.5f\n" % self.m_u
        else:
            s += "%s\n" % self.m_u.__str__()
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
        if self.m_u is not None:
            self.search = 'find_ag'
            self.default_b = self.what_is_b()

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

        self.sample.a = self.default_a or 0
        self.sample.b = self.default_b or np.inf
        self.sample.g = self.default_g or 0

        if self.search == 'find_a':
            res = minimize_scalar(afun, args=(self), bounds=(0, 1), method='bounded')

        if self.search == 'find_b':
            res = minimize_scalar(bfun, args=(self), bounds=(1, 5), method='brent')

        if self.search == 'find_g':
            res = minimize_scalar(gfun, args=(self), bounds=(-1, 1), method='bounded')

        if self.search in ['find_ab', 'find_ag', 'find_bg']:
            default = self.default_a or self.default_b or self.default_g
            if self.grid is None:
                self.grid = iadpython.Grid()
            if self.grid.is_stale(default):
                self.grid.calc(self)
            a, b, g = self.grid.min_abg(self.m_r, self.m_t)

        if self.search == 'find_ab':
            bnds = Bounds(np.array([0, 0]), np.array([1, np.inf]))
            res = minimize(abfun, [a, b], args=(self), bounds=bnds, method='Powell')

        if self.search == 'find_ag':
            bnds = Bounds(np.array([0, -1]), np.array([1, 1]))
            res = minimize(agfun, [a, g], args=(self), bounds=bnds, method='Powell')

        if self.search == 'find_bg':
            bnds = Bounds(np.array([0, -1]), np.array([np.inf, 1]))
            res = minimize(bgfun, [b, g], args=(self), bounds=bnds, method='Powell')

        return self.sample.a, self.sample.b, self.sample.g

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

    def what_is_b(self):
        """Find optical thickness using unscattered transmission."""
        s = self.sample
        t_un = self.m_u

        r1, t1 = iadpython.absorbing_glass_RT(1.0, s.n_above, s.n, s.nu_0, s.b_above)

        mu = iadpython.cos_snell(1.0, s.nu_0, s.n)

        r2, t2 = iadpython.absorbing_glass_RT(s.n, s.n_below, 1.0, mu, s.b_below)

        if t_un <= 0:
            return np.inf

        if t_un >= t1 * t2 / (1 - r1 * r2):
            return 0.001

        tt = t1 * t2

        if r1 == 0 or r2 == 0:
            ratio = tt / t_un
        else:
            ratio = (tt + np.sqrt(tt**2 + 4 * t_un**2 * r1 * r2))/(2 * t_un)

        return s.nu_0 * np.log(ratio)

    def measured_rt(self):
        """
        Calculate measured reflection and transmission.

        The direct incident power is :math:`(1-f)P`. The reflected power will
        be :math:`(1-f)R_{direct} P`.  Since baffles ensure that the light cannot
        reach the detector, we must bounce the light off the sphere walls to
        use to above gain formulas.  The contribution will then be

        .. math:: (1-f)R_{direct} (1-a_e) r_w P.

        The measured power will be

        .. math:: P_d = a_d (1-a_e) r_w [(1-f) r_{direct} + f r_w] P \\cdot G(r_s)

        Similarly the power falling on the detector measuring transmitted light is

        .. math:: P_d'= a_d' t_{direct} r_w' (1-a_e') P \\cdot G'(r_s)

        when the `entrance' port in the transmission sphere is closed,
        :math:`a_e'=0`.

        The normalized sphere measurements are

        .. math:: M_R = r_{std}\\cdot\\frac{R(r_{direct},r_s)-R(0,0)}{R(r_{std},r_{std})-R(0,0)}

        and

        .. math:: M_T = t_{std}\\cdot{T(t_{direct},r_s)-T(0,0) \\over T(t_{std},r_{std})-T(0,0)}

        Args:
            ur1: reflection for collimated incidence
            ut1: transmission for collimated incidence
            uru: reflection for diffuse incidence
            utu: transmission for diffuse incidence

        Returns:
            [float, float]: measured reflection and transmission
        """
        s = x.sample  
        ur1, ut1, uru, utu = s.rt()
    
        # find the unscattered reflection and transmission
        nu_inside = iad.cos_snell(1, s.nu_0, s.n)
        r_un, t_un = iad.specular_rt(s.n_above, s.n, s.n_below, s.b, nu_inside)

        # correct for lost light
        R_diffuse = uru - x.uru_lost
        T_diffuse = utu - x.utu_lost
        R_direct = ur1 - x.ur1_lost
        T_direct = ut1 - x.ut1_lost

        # correct for fraction not collected
        R_direct -= (1.0 - x.fraction_of_rc_in_mr) * r_un
        T_direct -= (1.0 - x.fraction_of_tc_in_mt) * t_un

        # Values when no spheres are used
        m_r = R_direct
        m_t = T_direct

        if exp.num_spheres == 1:
            return R_direct, T_direct
            r_gain_00 = self.r_sphere.multiplier(0, 0)
            r_gain_std = self.r_sphere.multiplier(self.r_sphere.r_std, self.r_sphere.r_std)
            r_gain_sample = self.r_sphere.multiplier(ur1, uru)

            f = self.fraction_of_rc_in_mr
            p_d = r_gain_sample * (ur1  * (1-f) + f*self.r_sphere.r_wall)
            p_std = r_gain_std * (self.r_sphere.r_std * (1-f) + f*self.r_sphere.r_wall)
            p_0 = r_gain_00 * f * self.r_sphere.r_wall
            m_r = self.r_sphere.r_std * (p_d - p_0)/(p_std - p_0)

            t_gain_00 = self.t_sphere.multiplier(0, 0)
            t_gain_std = self.t_sphere.multiplier(ur1, uru)
            m_t = ut1 * t_gain_00 / t_gain_std

        return m_r, m_t


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
