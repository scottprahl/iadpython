# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes

"""
Module for generating the starting layer for adding-doubling.

Two types of starting methods are possible.

    import iad.start

    n=4
    r, t = iad.quadrature.start(a, b, g)
    print(r)
    print(t)

"""

import numpy as np
import iadpython.quadrature as quad
import iadpython.phase as phase

__all__ = ('cos_critical_angle',
           'print_matrix',
           'zero_layer',
           'igi_start',
           'diamond'
           )


def cos_critical_angle(ni, nt):
    """
    Calculates the cosine of the critical angle.

    If there is no critical angle then 0.0 is returned.
    """
    if nt >= ni:
        return 0.0

    return np.sqrt(1-(nt/ni)**2)

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
    """
    n = method.quad_pts
    r = np.zeros([n, n])
    t = np.zeros([n, n])

    for i in range(n):
        t[i, i] = 1/method.twonuw[i]

    return r, t

class Method():
    def __init__(self, slab):
        self.quad_pts = 4
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
        s = ""
        s += "Quadrature Pts    = %d\n" % self.quad_pts
        return s

    def update_quadrature(self, slab):
        """
        This returns the quadrature slab.nus using Radau quadrature over the
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
        nu_c = cos_critical_angle(slab.n, 1.0)
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
        The best starting thickness to start the doubling process.

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
    def __init__(self):
        self.a = 0.0
        self.b = 1.0
        self.g = 0.0
        self.d = 1.0    # mm
        self.n = 1.0
        self.n_above = 1.0
        self.n_below = 1.0
        self.nu_0 = 1.0
        self.nu_c = 0.0
        self.mu_a = 0.0
        self.mu_s = 1.0
        self.mu_sp = 1.0
        self.a_calc = self.a
        self.b_calc = self.b
        self.g_calc = self.g
        self.b_thinnest = self.b/100
        self.update_derived_optical_properties()

    def update_derived_optical_properties(self):
        self.nu_c = cos_critical_angle(self.n, 1)
        self.mu_a = (1-self.a) * self.b / self.d
        self.mu_s = self.a * self.b / self.d
        self.mu_sp = self.mu_s*(1-self.g)

    def as_array(self):
        return [self.a, self.b, self.g, self.d, self.n]

    def init_from_array(self, a):
        self.a = a[0]
        self.b = a[1]
        self.g = a[2]
        self.d = a[3]
        self.n = a[4]

    def __str__(self):
        s = ""
        s += "albedo            = %.3f\n" % self.a
        s += "optical thickness = %.3f\n" % self.b
        s += "anisotropy        = %.3f\n" % self.g
        s += "refractive index  = %.3f\n" % self.n
        s += "\n"
        s += "mu_a              = %.3f [1/mm]\n" % self.mu_a
        s += "mu_s              = %.3f [1/mm]\n" % self.mu_s
        s += "mu_s*(1-g)        = %.3f [1/mm]\n" % self.mu_sp
        return s

def igi_start(method, hp, hm):
    """
    Infinitesmal Generator Initialization.

    igi_start() generates the starting matrix with the inifinitesimal generator method.
    The accuracy is O(d) and assumes that the average irradiance upwards is
    equal to that travelling downwards at the top and the average radiance upwards
    equals that moving upwards from the bottom.

    Ultimately the formulas for the reflection matrix is

    R_ij = a*d/(4*nu_i*nu_j) hpm_{ij}

    and

    T_ij = a*d/(4*nu_i*nu_j) hpp_{ij} + delta_{ij}*(1-d/nu_i)/(2*\nu_i w_i)

    """
    d = method.b_thinnest
    n = method.quad_pts

    R = np.zeros([n, n])
    T = np.zeros([n, n])
    for j in range(n):
        temp = method.a_calc * d / 4 / method.nu[j]
        for i in range(n):
            c = temp/method.nu[i]
            R[i, j] = c * hm[i, j]
            T[i, j] = c * hp[i, j]

        T[j, j] += (1-d/method.nu[j])/method.twonuw[j]

    return R, T

def diamond(method, hp, hm):

    """
    Diamond initialization.

    The diamond method uses the same `R_hat` and `T_hat` as was used for infinitesimal
    generator method.  Division by `2*nu_j*w_j` is not needed until the
    final values for `R` and `T` are formed.
    """
    d = method.b_thinnest
    n = method.quad_pts

    R = np.zeros([n, n])
    T = np.zeros([n, n])
    for j in range(n):
        temp = method.a_calc * d * method.weight[j]/ 4
        for i in range(n):
            c = temp/method.nu[i]
            R[i, j] = c * hm[i, j]
            T[i, j] = -c * hp[i, j]
        T[j, j] += d/(2*method.nu[j])

    I = np.identity(n)

    C = R @ np.linalg.inv(I+T)
    G = 0.5 * (I + T - C @ R)
    G2 = 2*G
    Ginv = np.linalg.inv(G2)

    print("A from equation 5.55\n")
    print_matrix(T, method)

    print("B from equation 5.55\n")
    print_matrix(R, method)

    print("C\n")
    print_matrix(C, method)

    print("Inverse of G from equation 5.56\n")
    print_matrix(G2, method)

    print("G from equation 5.56\n")
    print_matrix(Ginv, method)

    return R, T



def init_layer(slab, method):
    """
    Reflection and transmission matrices for a thin layer.
    """

    n = method.quad_pts

    if slab.b <= 0:
        return zero_layer(method)

    hp, hm = phase.get_phi_legendre(slab, method)

    return zero_layer(method)

#         if (method.b_thinnest < 1e-4 || method.b_thinnest < 0.09 * angle[1])
#             Get_IGI_Layer(method, h, R, T)
#         else
#             Get_Diamond_Layer(method, h, R, T)
