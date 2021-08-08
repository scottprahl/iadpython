# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments

"""
Class for managing integrating spheres.

    import iadpython.sphere

    s = iadpython.sphere.Sphere(250,20)
    print(s)
"""

import numpy as np


class Sphere():
    """Container class for an integrating sphere."""

    def __init__(self, d_sphere, d_sample, d_entrance=0, d_detector=0, r_detector=0, r_wall=1):
        """Object initialization."""
        self._d_sphere = d_sphere
        self._d_sample = d_sample
        self._d_entrance = d_entrance
        self._d_detector = d_detector

        self.a_sample = self.relative_cap_area(d_sample)
        self.a_detector = self.relative_cap_area(d_detector)
        self.a_entrance = self.relative_cap_area(d_entrance)
        self._a_wall = 1 - self.a_sample - self.a_entrance - self.a_detector
        self.r_detector = r_detector
        self.r_wall = r_wall

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
        s += "Port diameters \n"
        s += "         sample = %.1f mm\n" % self._d_sample
        s += "       entrance = %.1f mm\n" % self._d_entrance
        s += "       detector = %.1f mm\n" % self._d_detector
        s += "Diffuse reflectivities \n"
        s += "           wall = %.1f%%\n" % self.r_wall
        s += "       detector = %.1f%%\n" % self.r_detector
        return s

    def gain(self, URU, r_wall=None):
        """
        Determine the gain for this integrating sphere.

        The gain G(URU) on the irradiance on the detector (relative
        to a perfectly black sphere) is

        G(URU) = (P_d /A_d) / (P/A)

        See sphere.ipynb for details on the derivation.

        Args:
            URU: total reflectance for diffuse illumination
            r_wall: fractional wall reflectivity
        Returns:
            gain on detector caused by multiple bounces in sphere
        """
        if r_wall is None:
            r_wall = self.r_wall

        tmp = self.a_detector * self.r_detector + self.a_sample * URU
        tmp = r_wall * (self._a_wall + (1 - self.a_entrance) * tmp)
        if tmp == 1.0:
            G = r_wall
        else:
            G = r_wall * (1.0 + tmp / (1.0 - tmp))
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
            UR1: sample reflectance for normal irradiance
            URU: sample reflectance for diffuse irradiance
            r_wall: wall reflectance
        Returns:
            sphere muliplier
        """
        if r_wall is None:
            r_wall = self.r_wall

        if UR1 is None:
            UR1 = r_wall

        if URU is None:
            URU = UR1

        denom = 1
        denom -= self._a_wall * r_wall
        denom -= self.a_sample * URU
        denom -= self.a_detector * self.r_detector
        return UR1 / denom

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
        """Changing the relative wall area is a bit crazy."""
        assert 0 <= value <= 1, "relative wall are must be between 0 and 1"
        # Find the diameter of a spherical cap assuming all non-wall
        # port area is assigned to a single sample port
        assert 0 < value < 1, "0 < relative wall area < 1"
        self._d_sample = 2 * self._d_sphere * np.sqrt(value - value**2)
        self._d_entrance = 0
        self._d_detector = 0
        self.a_entrance = 0
        self.a_detector = 0
        self._a_wall = value
        self.a_sample = 1 - value


def Gain_11(RS, TS, URU, tdiffuse):
    """
    Net gain for on detector in reflection sphere for two sphere configuration.

    The light on the detector in the reflectance sphere is affected by interactions
    between the two spheres.  This function calculates the net gain on a detector
    in the reflection sphere for diffuse light starting in the reflectance sphere.
    G₁₁ = (P₁/Ad) / (P/A)
    then the full expression for the gain is
    G(r_s)/(1-a_s a_s' r_w r_w' (1-a_e)(1-a_e') G(r_s) G'(r_s)t_s²)
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
    on the detector in the second sphere $G_{22}$ is found by switching
    all primed variables to unprimed.  Thus $G_{21}(r_s,t_s)$ is
    $$
    G_{22}(r_s,t_s) = {G'(r_s) over 1-a_s a_s' r_w r_w'
                                  (1-a_e)(1-a_e') G(r_s) G'(r_s)t_s²  }
    $$
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
    f r_w² (1-a_e) P,
    the fraction of light reflected by the sample
    (1-f) rdirect r_w² (1-a_e) P,
    and the light transmitted through the sample
    (1-f) tdirect r_w' (1-a_e') P,
    If we use the gain for each part then we add
    G₁₁ * a_d (1-a_e) r_w² f  P
    to
    G₁₁ * a_d (1-a_e) r_w (1-f) rdirect  P
    and
    G₂₁ * a_d (1-a_e') r_w' (1-f) tdirect  P
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
    G₁₂ * a_d' (1-a_e) r_w² f P
    plus
    G₁₂ * a_d' (1-a_e) r_w (1-f) rdirect P
    plus
    G₂₂ * a_d' (1-a_e') r_w' (1-f) tdirect  P
    which simplifies slightly to the formula used below
    """
    G = RS.gain(URU)
    G22 = Gain_11(RS, TS, URU, UTU)

    x = TS.a_detector * (1 - TS.a_entrance) * TS.r_wall * G22
    x *= (1 - f) * UT1 + (1 - RS.a_entrance) * RS.r_wall * \
        RS.a_sample * UTU * (f * RS.r_wall + (1 - f) * UR1) * G
    return x
