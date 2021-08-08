# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments

"""
Class for doing adding-doubling calculations for a sample.

    import iadpython.ad as ad

    n=4
    sample = ad.Sample(a=0.9, b=10, g=0.9, n=1.5, quad_pts=4)
    r, t = sample.rt()
    print(r)
    print(t)
"""

import copy
import numpy as np
import iadpython.quadrature
import iadpython.combine
import iadpython.start

AD_MAX_THICKNESS = 1e6


class Sample():
    """Container class for details of a sample."""

    def __init__(self, a=0, b=1, g=0, n=1, n_above=1, n_below=1, quad_pts=4):
        """Object initialization."""
        self.a = a
        self.b = b
        self._g = g
        self._n = n
        self.n_above = n_above
        self.n_below = n_below
        self.d = 1.0
        self.nu_0 = 1.0
        self.b_above = 0
        self.b_below = 0
        self._quad_pts = quad_pts
        self.b_thinnest = None
        self.nu = None
        self.twonuw = None
        self.hp = None
        self.hm = None

    @property
    def n(self):
        """Getter property for index of refraction."""
        return self._n

    @n.setter
    def n(self, value):
        """When index is changed quadrature becomes invalid."""
        if value != self._n:
            self.nu = None
            self.twonuw = None
            self.hp = None
            self.hm = None
            self._n = value

    @property
    def g(self):
        """Getter property for anisotropy."""
        return self._g

    @g.setter
    def g(self, value):
        """When anisotropy is changed phi is invalid."""
        if value != self._g:
            self.hp = None
            self.hm = None
            self._g = value

    @property
    def quad_pts(self):
        """Getter property for number of quadrature points."""
        return self._quad_pts

    @quad_pts.setter
    def quad_pts(self, value):
        """When quadrature points are change everything is invalid."""
        if value != self._quad_pts:
            self.nu = None
            self.twonuw = None
            self.hp = None
            self.hm = None
            self._quad_pts = value

    def mu_a(self):
        """Absorption coefficient for the sample."""
        return (1 - self.a) * self.b / self.d

    def mu_s(self):
        """Scattering coefficient for the sample."""
        return self.a * self.b / self.d

    def mu_sp(self):
        """Reduced scattering coefficient for the sample."""
        return (1 - self.g) * self.a * self.b / self.d

    def nu_c(self):
        """Cosine of critical angle in the sample."""
        return iadpython.fresnel.cos_critical(self.n, 1)

    def a_delta_M(self):
        """Reduced albedo in delta-M approximation."""
        af = self.a * (self.g ** self.quad_pts)
        return (self.a - af) / (1 - af)

    def b_delta_M(self):
        """Reduced thickness in delta-M approximation."""
        af = self.a * (self.g ** self.quad_pts)
        return (1 - af) * self.b

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
        s += "  n sample          = %.4f\n" % self.n
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

    def wrmatrix(self, a, title=None):
        """Print matrix and sums."""
        n = self.quad_pts

        # header line
        if title is not None:
            print(title)
        print("cos_theta |", end='')
        for i in range(n):
            print("%9.5f" % self.nu[i], end='')
        print(" |     flux")
        print("----------+", end='')
        for i in range(n):
            print("---------", end='')
        print("-+---------")

        # contents + row fluxes
        for i in range(n):
            print("%9.5f |" % self.nu[i], end='')
            for j in range(n):
                if a[i, j] < -100 or a[i, j] > 100:
                    print("    *****", end='')
                else:
                    print("%9.5f" % a[i, j], end='')
            flux = 0.0
            for j in range(n):
                flux += a[i, j] * self.twonuw[j]
            print(" |%9.5f" % flux)

        # identify index of first quadrature angle greater than the critical angle
        nu_c = self.nu_c()
        k = np.min(np.where(self.nu > nu_c))
        UXx = np.dot(self.twonuw[k:], a[k:, k:])
        tflux = np.dot(self.twonuw[k:], UXx) * self.n**2

        # column fluxes
        print("----------+", end='')
        for i in range(n):
            print("---------", end='')
        print("-+---------")
        print("%9s |" % "flux   ", end='')
        for i in range(n):
            flux = 0.0
            for j in range(n):
                flux += a[j, i] * self.twonuw[j]
            print("%9.5f" % flux, end='')
        print(" |%9.5f\n" % tflux)

    def prmatrix(self, a, title=None):
        """Print matrix and sums."""
        if title is not None:
            print(title)
        n = self.quad_pts

        # first row
        print("[[", end='')
        for j in range(n - 1):
            print("%9.5f," % a[0, j], end='')
        print("%9.5f]," % a[0, -1])

        for i in range(1, n - 1):
            print(" [", end='')
            for j in range(n - 1):
                print("%9.5f," % a[i, j], end='')
            print("%9.5f]," % a[i, -1])

        # last row
        print(" [", end='')
        for j in range(n - 1):
            print("%9.5f," % a[-1, j], end='')
        print("%9.5f]]" % a[-1, -1])

    def update_quadrature(self):
        """
        Calculate the correct set of quadrature points.

        This returns the quadrature angles using Radau quadrature over the
        interval 0 to 1 if there is no critical angle for total internal reflection
        in the self.  If there is a critical angle whose cosine is 'nu_c' then
        Radau quadrature points are chosen from 0 to 'nu_c' and Radau
        quadrature points over the interval 'nu_c' to 1.

        Now we need to include three angles, the critical angle, the cone
        angle, and perpendicular.  Now the important angles are the ones in
        the self.  So we calculate the cosine of the critical angle in the
        sample and cosine of the cone angle in the self.

        The critical angle will always be greater than the cone angle in the
        sample and therefore the cosine of the critical angle will always be
        less than the cosine of the cone angle.  Thus we will integrate from
        zero to the cosine of the critical angle (using Gaussian quadrature
        to avoid either endpoint) then from the critical angle to the cone
        angle (using Radau quadrature so that the cosine angle will be
        included) and finally from the cone angle to 1 (again using Radau
        quadrature so that 1 will be included).
        """
        nby2 = self.quad_pts // 2

        if self.nu_0 == 1:
            # case 1.  Normal incidence, no critical angle
            if self._n == 1:
                a1 = []
                w1 = []
                a2, w2 = iadpython.quadrature.radau(self.quad_pts, a=0, b=1)

            # case 2.  Normal incidence, with critical angle
            else:
                nu_c = self.nu_c()
                a1, w1 = iadpython.quadrature.gauss(nby2, a=0, b=nu_c)
                a2, w2 = iadpython.quadrature.radau(nby2, a=nu_c, b=1)
        else:
            # case 3.  Conical incidence.  Include nu_0
            if self._n == 1.0:
                a1, w1 = iadpython.quadrature.radau(nby2, a=0, b=self.nu_0)
                a2, w2 = iadpython.quadrature.radau(nby2, a=self.nu_0, b=1)

            # case 4.  Conical incidence.  Include nu_c, nu_00, and 1
            else:
                nby3 = int(self.quad_pts / 3)
                nu_c = self.nu_c()

                # cosine of nu_0 in sample
                nu_00 = iadpython.fresnel.cos_snell(1.0, self.nu_0, self.n)
                a00, w00 = iadpython.quadrature.gauss(nby3, a=0, b=nu_c)
                a01, w01 = iadpython.quadrature.radau(nby3, a=nu_c, b=nu_00)
                a1 = np.append(a00, a01)
                w1 = np.append(w00, w01)
                a2, w2 = iadpython.quadrature.radau(nby3, a=nu_00, b=1)

        self.nu = np.append(a1, a2)
        self.twonuw = 2 * self.nu * np.append(w1, w2)

    def rt_matrices(self):
        """
        Total reflection and transmission.

        This is the top level routine for accessing the adding-doubling
        algorithm. By passing the optical paramters characteristic of the sample,
        this routine will do what it must to return the total reflection and
        transmission for collimated and diffuse irradiance.

        This routine has three different components based on if zero, one, or
        two boundary layers must be included.  If the index of refraction of the
        sample and the top and bottom slides are all one, then no boundaries need
        to be included.  If the top and bottom slides are identical, then some
        simplifications can be made and some time saved as a consequence. If the
        top and bottom slides are different, then the full red carpet treatment
        is required.

        Since the calculation time increases for each of these cases we test for
        matched boundaries first.  If the boundaries are matched then don't
        bother with boundaries for the top and bottom.  Just calculate the
        integrated reflection and transmission.   Similarly, if the top and
        bottom slides are similar, then quickly calculate these.
        """
        if self.b == 0:
            return iadpython.unscattered_rt(self)

        # cone not implemented yet
        if self.nu_0 != 1.0:
            #            RT_Cone(n,sample,OBLIQUE,UR1,UT1,URU,UTU);
            r, t = iadpython.start.zero_layer(self.quad_pts)
            return r, r, t, t

        R12, T12 = iadpython.simple_layer_matrices(self)

        # all done if boundaries are not an issue
        if self.n == 1 and self.n_above == 1 and self.n_below == 1 and \
                self.b_above == 0 and self.b_below == 0:
            return R12, R12, T12, T12

        # reflection/transmission arrays for top boundary
        R01, R10, T01, T10 = iadpython.start.boundary_layer(self, top=True)

        # same slide above and below.
        if self.n_above == self.n_below and self.b_above == self.b_below:
            R03, T03 = iadpython.add_same_slides(self, R01, R10, T01, T10, R12, T12)
            return R03, R03, T03, T03

        # reflection/transmission arrays for bottom boundary
        R23, R32, T23, T32 = iadpython.start.boundary_layer(self, top=False)

        # different boundaries on top and bottom
        R02, R20, T02, T20 = iadpython.add_slide_above(
            self, R01, R10, T01, T10, R12, R12, T12, T12)
        R03, R30, T03, T30 = iadpython.add_slide_below(
            self, R02, R20, T02, T20, R23, R32, T23, T32)

        return R03, R30, T03, T30

    def UX1_and_UXU(self, R, T):
        """
        Calculate total reflected and transmitted fluxes from matrices.

        The trick here is that the integration must be done over the fluxes
        that leave the sample.  This is not much of an issue for the transmitted
        fluxes because they are zero.  However, the internal reflected fluxes will
        not be zero and should be excluded from the sums.
        """
        nu_c = self.nu_c()
        # identify index of first quadrature angle greater than the critical angle
        k = np.min(np.where(self.nu > nu_c))

        URx = np.dot(self.twonuw[k:], R[k:, k:])
        UTx = np.dot(self.twonuw[k:], T[k:, k:])
        URU = np.dot(self.twonuw[k:], URx) * self.n**2
        UTU = np.dot(self.twonuw[k:], UTx) * self.n**2

        return URx[-1], UTx[-1], URU, UTU

    def rt(self):
        """
        Find the total reflected and transmitted flux for a sample.

        This is extended so that arrays can be handled.
        """
        len_a = 0
        len_b = 0
        len_g = 0
        if not np.isscalar(self.a):
            len_a = len(self.a)

        if not np.isscalar(self.b):
            len_b = len(self.b)

        if not np.isscalar(self.g):
            len_g = len(self.g)

        thelen = max(len_a, len_b, len_g)

        if thelen == 0:
            R, _, T, _ = self.rt_matrices()
            return self.UX1_and_UXU(R, T)

        if len_a and len_b and len_a != len_b:
            raise RuntimeError('rt: a and b arrays must be same length')

        if len_a and len_g and len_a != len_g:
            raise RuntimeError('rt: a and g arrays must be same length')

        if len_b and len_g and len_b != len_g:
            raise RuntimeError('rt: b and g arrays must be same length')

        ur1 = np.empty(thelen)
        ut1 = np.empty(thelen)
        uru = np.empty(thelen)
        utu = np.empty(thelen)

        if self.nu is None:
            self.update_quadrature()

        sample = copy.deepcopy(self)
        for i in range(thelen):
            if len_a > 0:
                sample.a = self.a[i]

            if len_b > 0:
                sample.b = self.b[i]

            if len_g > 0:
                sample.g = self.g[i]

            R, _, T, _ = sample.rt_matrices()
            ur1[i], ut1[i], uru[i], utu[i] = sample.UX1_and_UXU(R, T)

        return ur1, ut1, uru, utu
