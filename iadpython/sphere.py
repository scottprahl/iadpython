# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=consider-using-f-string
# pylint: disable=line-too-long

"""
Class for managing integrating spheres.

    Example:
        >>> import iadpython
        >>> s = iadpython.Sphere(250,20)
        >>> print(s)

"""

import numpy as np


class Sphere():
    """
    Container class for an integrating sphere.

    Attributes:
        - d_sphere: diameter of integrating sphere [mm]
        - d_sample: diameter of the sample port [mm]
        - d_entrance: diameter of the port that light enters the sphere [mm]
        - d_detector: diameter of the detector port [mm]
        - r_detector: reflectivity of the detector
        - r_wall: reflectivity of the wall
        - r_std: reflectivity of the standard used with the sphere

    Example:
        >>> import iadpython as iad
        >>> s = iad.Sphere(200, 20)
        >>> print(s)

    """

    def __init__(self, d_sphere, d_sample, d_entrance=0,
                       d_detector=0, r_detector=0, r_wall=1, r_std=1):
        """
        Object initialization.

        """
        self._d_sphere = d_sphere
        self._d_sample = d_sample
        self._d_entrance = d_entrance
        self._d_detector = d_detector

        self.a_sample = self.relative_cap_area(d_sample)
        self.a_detector = self.relative_cap_area(d_detector)
        self.a_entrance = self.relative_cap_area(d_entrance)
        self._a_wall = 1 - self.a_sample - self.a_entrance - self.a_detector
        self.r_detector = r_detector
        self._r_wall = r_wall
        self._r_std = r_std

    def cap_area(self, d_port):
        """Calculate area of spherical cap."""
        R = self.d_sphere / 2
        r = d_port / 2
        h = R - np.sqrt(R**2 - r**2)
        return 2 * np.pi * R * h

    def approx_relative_cap_area(self, d_port):
        """Calculate approx relative area of spherical cap."""
        R = self.d_sphere / 2
        r = d_port / 2
        return r**2 / (4 * R**2)

    def relative_cap_area(self, d_port):
        """Calculate relative area of spherical cap."""
#        R = self.d_sphere/2
#        r = d_port/2
#        h = R - np.sqrt(R**2-r**2)
#        return 2*np.pi*R*h / (4*np.pi*R**2)
        h = (self.d_sphere - np.sqrt(self.d_sphere**2 - d_port**2)) / 2
        return h / self.d_sphere

    def __str__(self):
        """Return basic details as a string for printing."""
        s = ""
        s += "Sphere diameter = %.1f mm\n" % self._d_sphere
        s += "Port diameters\n"
        s += "         sample = %.1f mm\n" % self._d_sample
        s += "       entrance = %.1f mm\n" % self._d_entrance
        s += "       detector = %.1f mm\n" % self._d_detector
        s += "Fractional areas of sphere\n"
        s += "          walls = %.5f\n" % self.a_wall
        s += "         sample = %.5f\n" % self.a_sample
        s += "       entrance = %.5f\n" % self.a_entrance
        s += "       detector = %.5f\n" % self.a_detector
        s += "Diffuse reflectivities\n"
        s += "          walls = %.1f%%\n" % (self.r_wall*100)
        s += "       detector = %.1f%%\n" % (self.r_detector*100)
        s += "       standard = %.1f%%\n" % (self.r_std*100)
        s += "Gain\n"
        s += "        nothing = %.1f\n" % self.multiplier(0,0)
        s += "       standard = %.1f\n" % self.multiplier(self.r_std, self.r_std)
        return s

    def gain(self, URU, r_wall=None):
        """
        Determine the gain relative to a black sphere.

        If the walls of the sphere are black then the light falling on the
        detector is the diffuse light entering the sphere divided by the
        surface area on the sphere (P/A).

        If the walls are perfectly white (and ports are perfectly absorbing)
        then the all entering light exits through the ports. (P/A_ports)

        The gain caused by 0% reflecting sphere walls (no port refl) is

        .. math:: \\mbox{gain} = \\frac{(P/A)}{(P/A)} = 1

        The gain caused by 100% reflecting sphere walls (no port refl) is

        .. math:: \\mbox{gain} = \\frac{(P/A_ports)}{(P/A)} = \\frac{A_total}{A_ports}

        Args:
            URU: reflectance from sample port for diffuse light
            r_wall: wall reflectance
        Returns:
            gain on detector caused by bounces inside sphere
        """
        if r_wall is None:
            r_wall = self.r_wall

        tmp = self.a_detector * self.r_detector + self.a_sample * URU
        tmp = r_wall * (self._a_wall + (1 - self.a_entrance) * tmp)
        if tmp == 1.0:
            G = np.inf
        else:
            G = 1.0 + tmp / (1.0 - tmp)
        return G

    def multiplier(self, UR1=None, URU=None, r_wall=None):
        """
        Determine the average reflectance of a sphere.

        The idea here is that UR1 is the reflection of the incident light
        for the first bounce.  Three cases come to mind

        1. If the light hits the sample first, then UR1 should be
        the sample reflectance for collimated illumination.

        2. If light hits the sphere wall first, then UR1 should be the wall
        reflectance.

        3. If light is enters the sphere completely diffuse then UR1=1

        As defined by LabSphere, "Technical Guide: integrating Sphere Theory
        and application" using equation 14

        Args:
            UR1: sample reflectance for normal collimated irradiance
            URU: sample reflectance for diffuse irradiance
            r_wall: wall reflectance
        Returns:
            sphere multiplier
        """
        if r_wall is None:
            r_wall = self._r_wall

        if UR1 is None:
            UR1 = r_wall

        if URU is None:
            URU = UR1

        denom = 1
        denom -= self._a_wall * r_wall
        denom -= self.a_sample * URU
        denom -= self.a_detector * self.r_detector

        if np.isscalar(denom):
            if denom < 1e-8:
                m = np.inf
            else:
                m = UR1/denom
        else:
            m = np.where(denom>1e-8, UR1/denom, np.inf)
        return m

    @property
    def d_sphere(self):
        """Getter property for sphere diameter."""
        return self._d_sphere

    @d_sphere.setter
    def d_sphere(self, value):
        """When size is changed ratios become invalid."""
        assert self.d_sample <= value, "sphere must be bigger than sample port"
        assert self.d_entrance <= value, "sphere must be bigger than entrance port"
        assert self.d_detector <= value, "sphere must be bigger than detector port"
        self._d_sphere = value
        self.a_sample = self.relative_cap_area(self._d_sample)
        self.a_detector = self.relative_cap_area(self._d_detector)
        self.a_entrance = self.relative_cap_area(self._d_entrance)
        self._a_wall = 1 - self.a_sample - self.a_entrance - self.a_detector

    @property
    def d_sample(self):
        """Getter property for sample port diameter."""
        return self._d_sample

    @d_sample.setter
    def d_sample(self, value):
        """When size is changed ratios become invalid."""
        assert 0 <= value <= self._d_sphere, "sample port must be between 0 and sphere diameter"
        self._d_sample = value
        self.a_sample = self.relative_cap_area(value)
        self._a_wall = 1 - self.a_sample - self.a_entrance - self.a_detector

    @property
    def d_entrance(self):
        """Getter property for entrance port diameter."""
        return self._d_entrance

    @d_entrance.setter
    def d_entrance(self, value):
        """When size is changed ratios become invalid."""
        assert 0 <= value <= self._d_sphere, "entrance port must be between 0 and sphere diameter"
        self._d_entrance = value
        self.a_entrance = self.relative_cap_area(value)
        self._a_wall = 1 - self.a_sample - self.a_entrance - self.a_detector

    @property
    def d_detector(self):
        """Getter property for detector port diameter."""
        return self._d_detector

    @d_detector.setter
    def d_detector(self, value):
        """When size is changed ratios become invalid."""
        assert 0 <= value <= self._d_sphere, "detector port must be between 0 and sphere diameter"
        self._d_detector = value
        self.a_detector = self.relative_cap_area(value)
        self._a_wall = 1 - self.a_sample - self.a_entrance - self.a_detector

    @property
    def a_wall(self):
        """Getter property for detector port diameter."""
        return self._a_wall

    @a_wall.setter
    def a_wall(self, value):
        """Change the relative wall area: a bit crazy."""
        assert 0 <= value <= 1, "relative wall are must be between 0 and 1"
        # Find the diameter of a spherical cap assuming all non-wall
        # port area is assigned to a single sample port
        self._d_sample = 2 * self._d_sphere * np.sqrt(value - value**2)
        self._d_entrance = 0
        self._d_detector = 0
        self.a_entrance = 0
        self.a_detector = 0
        self._a_wall = value
        self.a_sample = 1 - value

    @property
    def r_std(self):
        """Getter property for reflectance standard."""
        return self._r_std

    @r_std.setter
    def r_std(self, value):
        """Change the reflectance standard used."""
        if np.isscalar(value):
            assert 0 <= value <= 1, "Reflectivity of standard must be between 0 and 1"
        else:
            assert 0 <= value.all() <= 1, "Reflectivity of standard must be between 0 and 1"
        self._r_std = value
        self.gain_std = self.multiplier(self.r_std, self.r_std)

    @property
    def r_wall(self):
        """Getter property for wall reflectivity."""
        return self._r_wall

    @r_wall.setter
    def r_wall(self, value):
        """Change the wall reflectivity."""
        if np.isscalar(value):
            assert 0 <= value <= 1, "Reflectivity of standard must be between 0 and 1"
        else:
            assert 0 <= value.all() <= 1, "Reflectivity of standard must be between 0 and 1"
        self._r_wall = value

def Gain_11(RS, TS, URU, tdiffuse):
    """
    Net gain for on detector in reflection sphere for two sphere configuration.

    The light on the detector in the reflectance sphere is affected by interactions
    between the two spheres.  This function calculates the net gain on a detector
    in the reflection sphere for diffuse light starting in the reflectance sphere.

    .. math:: G_{11} = \\frac{(P_1/A_d)}{(P/A)}

    then the full expression for the gain is

    .. math:: \\frac{G(r_s)}{(1-a_s a_s' r_w r_w' (1-a_e)(1-a_e') G(r_s) G'(r_s)t_s^2)}

    """
    G = RS.gain(URU)
    GP = TS.gain(URU)

    areas = RS.a_sample * TS.a_sample * (1 - RS.a_entrance) * (1 - TS.a_entrance)
    G11 = G / (1 - areas * RS.r_wall * TS.r_wall * G * GP * tdiffuse**2)
    return G11


def Gain_22(RS, TS, URU, tdiffuse):
    """
    Two sphere gain in T sphere for light starting in T sphere.

    Similarly, when the light starts in the second sphere, the gain for light
    on the detector in the second sphere :math:`G_{22}` is found by switching
    all primed variables to unprimed.  Thus :math:`G_{21}(r_s,t_s)` is

    .. math:: G_{22}(r_s,t_s) = \\frac{G'(r_s)}{1-a_s a_s' r_w r_w'(1-a_e)(1-a_e') G(r_s) G'(r_s)t_s^2}

    """
    G = RS.gain(URU)
    GP = TS.gain(URU)

    areas = RS.a_sample * TS.a_sample * (1 - RS.a_entrance) * (1 - TS.a_entrance)
    G22 = GP / (1 - areas * RS.r_wall * TS.r_wall * G * GP * tdiffuse**2)
    return G22


def Two_Sphere_R(RS, TS, UR1, URU, UT1, UTU, f=0):
    """
    Total gain in R sphere for two sphere configuration.

    The light on the detector in the reflection sphere arises from three
    sources: the fraction of light directly reflected off the sphere wall

    .. math:: f r_w^2 (1-a_e) P,

    the fraction of light reflected by the sample

    .. math:: (1-f) r_{direct} r_w^2 (1-a_e) P,

    and the light transmitted through the sample

    .. math:: (1-f) t_{direct} r_w' (1-a_e') P,

    If we use the gain for each part then we add

    .. math:: G_{11}  a_d (1-a_e) r_w^2 f  P

    to

    .. math:: G_{11} a_d (1-a_e) r_w (1-f) r_{direct}  P

    and

    .. math:: G_{21} a_d (1-a_e') r_w' (1-f) t_{direct}  P

    which simplifies slightly to the formula used below
    """
    GP = TS.gain(URU)
    G11 = Gain_11(RS, TS, URU, UTU)

    x = RS.a_detector * (1 - RS.a_entrance) * RS.r_wall * G11
    p1 = (1 - f) * UR1
    p2 = RS.r_wall * f
    p3 = (1 - f) * TS.a_sample * (1 - TS.a_entrance) * TS.r_wall * UT1 * UTU * GP
    return x * (p1 + p2 + p3)


def Two_Sphere_T(RS, TS, UR1, URU, UT1, UTU, f=0):
    """
    Total gain in T sphere for two sphere configuration.

    For the power on the detector in the transmission (second) sphere we
    have the same three sources.  The only difference is that the subscripts
    on the gain terms now indicate that the light ends up in the second
    sphere

    .. math:: G_{12}  a_d' (1-a_e) r_w^2 f P

    plus

    .. math:: G_{12}  a_d' (1-a_e) r_w (1-f) r_{direct} P

    plus

    .. math:: G_{22} a_d' (1-a_e') r_w' (1-f) t_{direct}  P

    which simplifies slightly to the formula used below
    """
    G = RS.gain(URU)
    G22 = Gain_11(RS, TS, URU, UTU)

    x = TS.a_detector * (1 - TS.a_entrance) * TS.r_wall * G22
    x *= (1 - f) * UT1 + (1 - RS.a_entrance) * RS.r_wall * \
        RS.a_sample * UTU * (f * RS.r_wall + (1 - f) * UR1) * G
    return x
