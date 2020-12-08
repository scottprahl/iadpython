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

__all__ = ('cos_critical_angle',
           )


def cos_critical_angle(ni, nt):
    """
    Calculates the cosine of the critical angle.

    If there is no critical angle then 0.0 is returned (i.e., cos(pi/2)).
    """
    if nt >= ni:
        return 0.0

    return np.sqrt(1-(nt/ni)**2)

class slab():
    def __init__(self):
        self.quad_pts = 4
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
        self.nu = []
        self.weight = []
        self.twonuw = []
        self.update_derived_optical_properties()
        self.update_quadrature()

    def update_derived_optical_properties(self):
        self.nu_c = cos_critical_angle(self.n, 1)
        self.mu_a = (1-self.a) * self.b / self.d
        self.mu_s = self.a * self.b / self.d
        self.mu_sp = self.mu_s*(1-self.g)

        af = self.a * self.g**self.quad_pts
        self.a_calc = (self.a - af) / (1 - af)
        self.b_calc = (1 - af) * self.b
        self.g_calc = self.g

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

    def update_start_depth(self):
        """
        The best starting thickness to start the doubling process.

        The criterion is based on an assessment of the (1) round-off error,
        (2) the angular initialization error, and (3) the thickness
        initialization error.  Wiscombe concluded that an optimal
        starting thickness depends on the smallest quadrature angle, and
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
        if self.d <= 0:
            return 0.0

        if self.d == np.inf:
            return mu / 2.0

        dd = self.d
        while dd > mu:
            dd /= 2

        self.b_thinnest = dd

    def update_quadrature(self):
        """
        This returns the quadrature angles using Radau quadrature over the
        interval 0 to 1 if there is no critical angle for total internal reflection
        in the self.  If there is a critical angle whose cosine is 'nu_c' then
        Radau quadrature points are chosen from 0 to 'nu_c' and Radau
        quadrature points over the interval 'nu_c' to 1.

        Now we need to include three angles, the critical angle, the cone
        angle, and perpendicular.  Now the important angles are the ones in
        the self.  So we calculate the cosine of the critical angle in the
        slab and cosine of the cone angle in the self.

        The critical angle will always be greater than the cone angle in the
        slab and therefore the cosine of the critical angle will always be
        less than the cosine of the cone angle.  Thus we will integrate from
        zero to the cosine of the critical angle (using Gaussian quadrature
        to avoid either endpoint) then from the critical angle to the cone
        angle (using Radau quadrature so that the cosine angle will be
        included) and finally from the cone angle to 1 (again using Radau
        quadrature so that 1 will be included).
        """
        nu_c = cos_critical_angle(self.n, 1.0)
        nby2 = int(self.quad_pts / 2)

        if self.nu_0 == 1:
            # case 1.  Normal incidence, no critical angle
            if self.n == 1:
                a1 = []
                w1 = []
                a2, w2 = quad.radau(self.quad_pts, a=0, b=1)

            # case 2.  Normal incidence, with critical angle
            if self.nu_0 == 1.0:
                a1, w1 = quad.gauss(nby2, a=0, b=nu_c)
                a2, w2 = quad.radau(nby2, a=nu_c, b=1)
        else:
            # case 3.  Conical incidence.  Include nu_0
            if self.n == 1.0:
                a1, w1 = quad.radau(nby2, a=0, b=self.nu_0)
                a2, w2 = quad.radau(nby2, a=self.nu_0, b=1)

            # case 4.  Conical incidence.  Include nu_c, nu_00, and 1
            else:
                nby3 = int(self.quad_pts / 3)
                # cosine of nu_0 in slab
                nu_00 = np.cos(np.arcsin(np.sin(np.arccos(self.nu_0))/self.n))
                a00, w00 = quad.gauss(nby3, a=0, b=nu_c)
                a01, w01 = quad.radau(nby3, a=nu_c, b=nu_00)
                a1 = np.append(a00, a01)
                w1 = np.append(w00, w01)
                a2, w2 = quad.radau(nby3, a=nu_00, b=1)

        self.nu = np.append(a1, a2)
        self.weight = np.append(w1, w2)
        self.twonuw = 2 * self.nu * self.weight
        self.update_start_depth()
