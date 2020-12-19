# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments

"""
Module for generating the starting layer for adding-doubling.

Two types of starting methods are possible.

    import iadpython.start

    n=4
    slab = iadpython.start.Slab(a=0.9, b=10, g=0.9, n=1.5)
    method = iadpython.start.Method(slab)
    r, t = iad.start.init_layer(slab, method)
    print(r)
    print(t)

"""

import numpy as np
import scipy.linalg
import iadpython.quadrature as quad
import iadpython.redistribution

__all__ = ('print_matrix',
           'zero_layer',
           'igi',
           'diamond',
           'init_layer'
           )


def print_matrix(a, method):
    """Print matrix and sums."""
    n = method.quad_pts

    #header line
    print("%9.5f" % 0.0, end='')
    for i in range(n):
        print("%9.5f" % method.nu[i], end='')
    print("     flux")

    #contents + row fluxes
    tflux = 0.0
    for i in range(n):
        print("%9.5f" % method.nu[i], end='')
        for j in range(n):
            if a[i, j] < -10 or a[i, j] > 10:
                print("    *****", end='')
            else:
                print("%9.5f" % a[i, j], end='')
        flux = 0.0
        for j in range(n):
            flux += a[i, j] * method.twonuw[j]
        print("%9.5f" % flux)
        tflux += flux * method.twonuw[i]

    #column fluxes
    print("%9s" % "flux   ", end='')
    for i in range(n):
        flux = 0.0
        for j in range(n):
            flux += a[j, i] * method.twonuw[j]
        print("%9.5f" % flux, end='')
    print("%9.5f\n" % tflux)

    #line of asterisks
    for i in range(n+1):
        print("*********", end='')
    print("\n")

def zero_layer(method):
    """
    Create R and T matrices for layer with zero thickness.

    Need to include quadrature normalization so R+T=1.
    """
    n = method.quad_pts
    r = np.zeros([n, n])
    t = np.identity(n) / method.twonuw

    return r, t

class Method():
    """Container class for tracking calculation details."""

    def __init__(self, slab, quad_pts=4):
        self.quad_pts = quad_pts
        self.b_thinnest = 0.0001
        self.nu = []
        self.weight = []
        self.twonuw = []
        self.update_quadrature(slab)

        af = slab.a * slab.g**self.quad_pts
        self.a_calc = (slab.a - af) / (1 - af)
        self.b_calc = (1 - af) * slab.b
        self.g_calc = slab.g
        self.update_start_depth()

    def __str__(self):
        """Return a reasonable string for printing."""
        s = ""
        s += "Quadrature Pts = %d\n" % self.quad_pts
        s += "Thinnest layer = %.5f\n" % self.b_thinnest
        s += "a_calc         = %.5f\n" % self.a_calc
        s += "b_calc         = %.5f\n" % self.b_calc
        s += "g_calc         = %.5f\n" % self.g_calc
        return s

    def update_quadrature(self, slab):
        """
        Calculate the correct set of quadrature points.

        This returns the quadrature angles using Radau quadrature over the
        interval 0 to 1 if there is no critical slab.nu for total internal reflection
        in the self.  If there is a critical slab.nu whose cosine is 'nu_c' then
        Radau quadrature points are chosen from 0 to 'nu_c' and Radau
        quadrature points over the interval 'nu_c' to 1.

        Now we need to include three slab.nus, the critical slab.nu, the cone
        slab.nu, and perpendicular.  Now the important slab.nus are the ones in
        the self.  So we calculate the cosine of the critical slab.nu in the
        slab and cosine of the cone slab.nu in the self.

        The critical slab.nu will always be greater than the cone slab.nu in the
        slab and therefore the cosine of the critical slab.nu will always be
        less than the cosine of the cone slab.nu.  Thus we will integrate from
        zero to the cosine of the critical slab.nu (using Gaussian quadrature
        to avoid either endpoint) then from the critical slab.nu to the cone
        slab.nu (using Radau quadrature so that the cosine slab.nu will be
        included) and finally from the cone slab.nu to 1 (again using Radau
        quadrature so that 1 will be included).
        """
        nu_c = iadpython.fresnel.cos_critical(slab.n, 1.0)
        nby2 = int(self.quad_pts / 2)

        if slab.nu_0 == 1:
            # case 1.  Normal incidence, no critical slab.nu
            if slab.n == 1:
                a1 = []
                w1 = []
                a2, w2 = quad.radau(self.quad_pts, a=0, b=1)

            # case 2.  Normal incidence, with critical slab.nu
            else:
                a1, w1 = quad.gauss(nby2, a=0, b=nu_c)
                a2, w2 = quad.radau(nby2, a=nu_c, b=1)
        else:
            # case 3.  Conical incidence.  Include nu_0
            if slab.n == 1.0:
                a1, w1 = quad.radau(nby2, a=0, b=slab.nu_0)
                a2, w2 = quad.radau(nby2, a=slab.nu_0, b=1)

            # case 4.  Conical incidence.  Include nu_c, nu_00, and 1
            else:
                nby3 = int(self.quad_pts / 3)
                # cosine of nu_0 in slab
                nu_00 = np.cos(np.arcsin(np.sin(np.arccos(slab.nu_0))/slab.n))
                a00, w00 = quad.gauss(nby3, a=0, b=nu_c)
                a01, w01 = quad.radau(nby3, a=nu_c, b=nu_00)
                a1 = np.append(a00, a01)
                w1 = np.append(w00, w01)
                a2, w2 = quad.radau(nby3, a=nu_00, b=1)

        self.nu = np.append(a1, a2)
        self.weight = np.append(w1, w2)
        self.twonuw = 2 * self.nu * self.weight

    def update_start_depth(self):
        """
        Find the best starting thickness to start the doubling process.

        The criterion is based on an assessment of the (1) round-off error,
        (2) the angular initialization error, and (3) the thickness
        initialization error.  Wiscombe concluded that an optimal
        starting thickness depends on the smallest quadrature slab.nu, and
        recommends that when either the infinitesimal generator or diamond
        initialization methods are used then the initial thickness is optimal
        when type 2 and 3 errors are comparable, or when d ≈ µ.

        Note that round-off is important when the starting thickness is less than
        |1e-4| for diamond initialization and less than
        |1e-8| for infinitesimal generator initialization assuming
        about 14 significant digits of accuracy.

        Since the final thickness is determined by repeated doubling, the
        starting thickness is found by dividing by 2 until the starting thickness is
        less than 'mu'.  Also we make checks for a layer with zero thickness
        and one that infinitely thick.
        """
        mu = self.nu[0]
        b = self.b_calc

        if b <= 0:
            return 0.0

        if b == np.inf:
            return mu / 2.0

        dd = b
        while dd > mu:
            dd /= 2

        self.b_thinnest = dd

class Slab():
    """Container class for details of a slab."""

    def __init__(self, a=0, b=1, g=0, n=1, n_above=1, n_below=1):
        self.a = a
        self.b = b
        self.g = g
        self.n = n
        self.n_above = n_above
        self.n_below = n_below
        self.d = 1.0
        self.nu_0 = 1.0
        self.b_above = 0
        self.b_below = 0

    def mu_a(self):
        """Absorption coefficient for the slab."""
        return (1-self.a) * self.b / self.d

    def mu_s(self):
        """Scattering coefficient for the slab."""
        return self.a * self.b / self.d

    def mu_sp(self):
        """Reduced scattering coefficient for the slab."""
        return (1 - self.g) * self.a * self.b / self.d

    def nu_c(self):
        """Cosine of critical angle in the slab."""
        return iadpython.fresnel.cos_critical(self.n, 1)

    def as_array(self):
        """Return details as an array."""
        return [self.a, self.b, self.g, self.d, self.n]

    def init_from_array(self, a):
        """Initialize basic details as an array."""
        self.a = a[0]
        self.b = a[1]
        self.g = a[2]
        self.d = a[3]
        self.n = a[4]

    def __str__(self):
        """Return basic details as a string for printing."""
        s = ""
        s += "albedo            = %.3f\n" % self.a
        s += "optical thickness = %.3f\n" % self.b
        s += "anisotropy        = %.3f\n" % self.g
        s += "\n"
        s += "  n slab          = %.4f\n" % self.n
        s += "  n top slide     = %.4f\n" % self.n_above
        s += "  n bottom slide  = %.4f\n" % self.n_below
        s += "\n"
        s += "d                 = %.3f mm\n" % self.d
        s += "mu_a              = %.3f /mm\n" % self.mu_a()
        s += "mu_s              = %.3f /mm\n" % self.mu_s()
        s += "mu_s*(1-g)        = %.3f /mm\n" % self.mu_sp()
        s += "Light angles\n"
        s += " cos(theta incident) = %.5f\n" % self.nu_0
        s += "      theta incident = %.1f°\n" % np.degrees(np.arccos(self.nu_0))
        s += " cos(theta critical) = %.5f\n" % self.nu_c()
        s += "      theta critical = %.1f°\n" % np.degrees(np.arccos(self.nu_c()))
        return s

def igi(slab, method):
    """
    Infinitesmal Generator Initialization.

    igi_start() generates the starting matrix with the inifinitesimal generator method.
    The accuracy is O(d) and assumes that the average irradiance upwards is
    equal to that travelling downwards at the top and the average radiance upwards
    equals that moving upwards from the bottom.

    Ultimately the formulas for the reflection matrix is

    R_ij = a*d/(4*nu_i*nu_j) hpm_ij

    and

    T_ij = a*d/(4*nu_i*nu_j) hpp_ij + delta_ij*(1-d/nu_i)/(2*nu_i w_i)
    """
    d = method.b_thinnest
    n = method.quad_pts

    hp, hm = iadpython.redistribution.hg_legendre(slab, method)

    temp = method.a_calc * d / 4 / method.nu
    R = temp * (hm / method.nu).T
    T = temp * (hp / method.nu).T
    T += (1 - d/method.nu)/method.twonuw * np.identity(n)

    return R, T

def diamond(slab, method):
    """
    Diamond initialization.

    The diamond method uses the same `r_hat` and `t_hat` as was used for infinitesimal
    generator method.  Division by `2*nu_j*w_j` is not needed until the
    final values for `R` and `T` are formed.
    """
    d = method.b_thinnest
    n = method.quad_pts
    I = np.identity(n)
    hp, hm = iadpython.redistribution.hg_legendre(slab, method)

    temp = method.a_calc * d * method.weight/ 4
    r_hat = temp * (hm / method.nu).T
    t_hat = d/(2*method.nu) * I - temp * (hp / method.nu).T

    C = np.linalg.solve((I+t_hat).T, r_hat.T)
    G = 0.5 * (I + t_hat - C.T @ r_hat).T
    lu, piv = scipy.linalg.lu_factor(G)

    D = 1/method.twonuw * (C * method.twonuw).T
    R = scipy.linalg.lu_solve((lu, piv), D).T/method.twonuw
    T = scipy.linalg.lu_solve((lu, piv), I).T/method.twonuw
    T -= I.T/method.twonuw
    return R, T

def init_layer(slab, method):
    """
    Reflection and transmission matrices for a thin layer.

    The optimal method for calculating a thin layer depends on the
    thinness of the layer and the quadrature angles.  This chooses
    the best method using Wiscombe's criteria.
    """
    if slab.b <= 0:
        return zero_layer(method)

    if method.b_thinnest < 1e-4 or method.b_thinnest < 0.09 * method.nu[0]:
        return igi(slab, method)

    return diamond(slab, method)
