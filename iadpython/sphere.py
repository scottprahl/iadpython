# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=consider-using-f-string
# pylint: disable=line-too-long

"""
Class for managing integrating spheres.

This module contains the Sphere class, designed to simulate and analyze the
behavior of light within an integrating sphere. An integrating sphere is a
device used in optical measurements, which allows for the uniform scattering
of light. It is commonly used for reflection and transmission measurements
of materials.

The Sphere class models the geometrical and optical properties of an
integrating sphere, enabling the calculation of various parameters such as
the areas of spherical caps given port diameters, relative port areas,
and the gain caused by reflections within the sphere. It supports different
measurements scenarios by adjusting port diameters, detector reflectivity,
wall reflectivity, and using a reflectance standard.

Attributes:
    d_sphere (float): Diameter of the integrating sphere in millimeters.
    d_sample (float): Diameter of the port that holds the sample.
    d_empty (float): Diameter of the empty port.
    d_detector (float): Diameter of the port with the detector.
    r_detector (float): Reflectivity of the detector.
    r_wall (float): Reflectivity of the sphere's internal wall.
    r_std (float): Reflectivity of the standard used for calibration.

Methods:
    cap_area: actual area of a spherical cap for a port
    relative_cap_area: relative area of spherical cap to the sphere's area.
    gain: sphere gain relative to a black sphere
    multiplier: multiplier for wall power due to sphere

Example usage:
    >>> import iadpython
    >>> s = iadpython.Sphere(250,20)
    >>> print(s)

    >>> s = iadpython.Sphere(200, 20, d_empty=10, d_detector=10, r_detector=0.8, r_wall=0.99, r_std=0.99)
    >>> print(sphere)
    >>> area_sample = sphere.cap_area(sphere.d_sample)
    >>> print(f"Sample port area: {area_sample:.2f} mm²")
"""

import random
import numpy as np
import iadpython

def uniform_disk():
    """
    Generate a point uniformly distributed on a unit disk.

    This function generates and returns a point (x, y) that is uniformly distributed
    over the area of a unit disk centered at the origin (0, 0). The unit disk is
    defined as the set of all points (x, y) such that x^2 + y^2 <= 1. The method
    uses rejection sampling, where random points are generated within the square
    that bounds the unit disk, and only points that fall within the disk are accepted.

    Returns:
        tuple of (float, float, float): A point (x, y) where x and y are the
        coordinates of the point uniformly distributed within the unit disk,
        and s is the square of the distance from the origin to this point.
    """
    s = 2
    while s > 1:
        x = 2 * random.random() - 1
        y = 2 * random.random() - 1
        s = x*x + y*y
    return x, y, s


class Sphere():
    """Container class for an integrating sphere.
    
    For a reflection measurement, the empty port is the diameter of through
    which the light enters to hit the sample.  For a transmission measurement
    this is the port that might allow unscattered transmission to leave. In
    either case, the reflectance from this port is assumed to be zero.
    
    By default, the sample port is on top (z=-R), the empty port is on the bottom (z=R)
    and the detector is on the side (x=R)

    Attributes:
        - d: diameter of integrating sphere [mm]
        - r_wall: reflectivity of the wall
        - a_wall: fraction of the sphere that is walls (relative
        - r_std: reflectivity of the standard used with the sphere
        - sample: port object representing sample port
        - empty: port object representing empty port
        - detector: port object representing detector port
        - x: x-coordinate of photon on wall
        - y: y-coordinate of photon on wall
        - z: z-coordinate of photon on wall

    Example::

        >>> import iadpython as iad
        >>> s = iad.Sphere(200, 20)
        >>> print(s)
    """

    def __init__(self, d_sphere, d_sample, d_empty=0,
                 d_detector=0, r_detector=0, r_wall=0.99, r_std=0.99):
        self._d = d_sphere
        self._r_wall = r_wall
        self._r_std = r_std
        R = d_sphere/2
        self.sample = iadpython.Port(self, d_sample, x=0, y=0, z=-R)
        self.detector = iadpython.Port(self, d_detector, uru=r_detector, x=R, y=0, z=0)
        self.empty = iadpython.Port(self, d_empty, uru=0, x=0, y=0, z=R)
        self._a_wall = 1 - self.sample.a - self.empty.a - self.detector.a
        self.x = 0
        self.y = 0
        self.z = 0
        self.baffle = False
        self.weight = 0

    def __str__(self):
        """Return basic details as a string for printing."""
        s = ""
        s += "Sphere\n"
        s += "        diameter = %7.2f mm\n" % self.d
        s += "          radius = %7.2f mm\n" % (self.d/2)
        s += "   relative area = %7.4f\n" % self.a_wall
        s += "       uru walls = %7.1f%%\n" % (self.r_wall * 100)
        s += "    uru standard = %7.1f%%\n" % (self.r_std * 100)
        s += "          baffle = %s\n" % self.baffle
        s += "Sample Port\n" + str(self.sample)
        s += "Detector Port\n" + str(self.detector)
        s += "Empty Port\n" + str(self.empty)
        s += "Multipliers\n"
        s += "        nothing = %7.3f\n" % self.multiplier(0, 0)
        s += "       standard = %7.3f\n" % self.multiplier(self.r_std, self.r_std)
        s += "Monte Carlo\n"
        s += "       location = (%6.1f, %6.1f, %6.1f) mm\n" % (self.x, self.y, self.z)
        s += "         weight = %7.3f\n" % self.weight
        return s

    def multiplier(self, UX1=None, URU=None):
        """
        Determine detected light in a sphere.

        The idea here is that UX1 is the diffuse light entering the sphere.  This 
        might be diffuse light reflected by the sample or diffuse light transmitted
        through the sample.  

        We assume that the sample is characterized by the reflection and tranmission
        for collimated normally incident light (UR1 and UT1) and by the reflection 
        and transmission for diffuse incident light (URU and UTU).

        There are four common cases

        1. For a reflection experiment, `UX1=UR1` and `URU=URU`

        2. For a transmission experiment, `UX1=UT1` and `URU=URU`

        3. If light hits the sphere wall first, then `UX1=r_wall`

        4. If light is enters the sphere completely diffuse then `UX1=1`

        This matches LabSphere, "Technical Guide: integrating Sphere Theory
        and application" using equation 14 for case 3

        Args:
            UX1: diffuse light entering the sphere
            URU: sample reflectance for diffuse irradiance
        Returns:
            sphere multiplier
        """
        if UX1 is None:
            UX1 = self.r_wall

        if URU is None:
            URU = UX1

        denom = 1
        denom -= self._a_wall * self.r_wall
        denom -= self.sample.a * URU
        denom -= self.detector.a * self.detector.uru

        if np.isscalar(denom):
            if denom > 1e-8:
                m = UX1 * self._a_wall / denom
            else:
                m = np.inf
        else:
            m = np.where(denom > 1e-8, UX1 * self._a_wall / denom, np.inf)
        return m

    @property
    def d(self):
        """Getter property for sphere diameter."""
        return self._d

    @d.setter
    def d(self, value):
        """When size is changed ratios become invalid."""
        assert self.sample.d <= value, "sphere must be bigger than sample port"
        assert self.empty.d <= value, "sphere must be bigger than empty port"
        assert self.detector.d <= value, "sphere must be bigger than detector port"
        self._d = value
        self._a_wall = 1 - self.sample.a - self.empty.a - self.detector.a

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
        self.sample.d = 2 * self._d * np.sqrt(value - value**2)
        self.empty.d = 0
        self.detector.d = 0
        self._a_wall = value

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

    def uniform(self):
        """
        Generate a point uniformly distributed on the surface of a sphere.
    
        This function returns a point that is uniformly distributed over the surface
        of a sphere with a specified radius. It utilizes the method described in
        "Choosing a Point from the Surface of a Sphere" by George Marsaglia (1972),
        which is an efficient algorithm for generating such points using a transformation
        from points uniformly distributed on a unit disk.
    
        Returns:
            A numpy array of the coordinates (x, y, z) of a random point on the sphere's surface.
        """
        ux, uy, s = uniform_disk()
        root = 2 * np.sqrt(1-s)
        return np.array([ux * root, uy * root, 1 - 2 * s]) * self.d/2

    def do_one_photon(self):
        """Bounce photon inside sphere until it leaves."""
        bounces = 0
        detected = 0

        # assume photon launched form sample
        weight = 1
        last_location = PortType.SAMPLE

        while weight > 1e-4:

            self.x, self.y, self.z = self.uniform()

            if self.detector.hit():
                if last_location == PortType.DETECTOR:   # avoid hitting self
                    continue
                if last_location == PortType.SAMPLE and self.baffle: # sample --> detector prohibited
                    continue

                # record detected light and update weight
                transmitted = weight * (1-self.r_detector)
                detected += transmitted
                weight -= transmitted
                last_location = PortType.DETECTOR

            elif self.sample.hit():
                if last_location == PortType.SAMPLE:    # avoid hitting self
                    continue
                if last_location == PortType.DETECTOR and self.baffle: # detector --> sample prohibited
                    continue
                weight *= self.r_sample
                last_location = PortType.SAMPLE

            elif self.empty.hit():
                weight = 0
                last_location = PortType.EMPTY

            else:
                # must have hit wall
                weight *= self.r_wall
                last_location = PortType.WALL

            bounces +=1

        return detected, bounces
        
    def do_N_photons(self):
        
        total_detected = 0
        total_bounces = 0
        
        for i in range(N):
            detected, bounces = self.do_one_photon()
            total_detected += detected
            total_bounces += bounces
        
        print("average detected = %.5f" % (total_detected/N))
        print("average bounces  = %.5f" % (total_bounces/N))
                

def Gain_11(RS, TS, URU, tdiffuse):
    r"""Net gain for on detector in reflection sphere for two sphere configuration.

    The light on the detector in the reflectance sphere is affected by interactions
    between the two spheres.  This function calculates the net gain on a detector
    in the reflection sphere for diffuse light starting in the reflectance sphere.

    .. math:: G_{11} = \frac{(P_1/A_d)}{(P/A)}

    then the full expression for the gain is

    .. math:: \frac{G(r_s)}{(1-a_s a_s' r_w r_w' (1-a_e)(1-a_e') G(r_s) G'(r_s)t_s^2)}

    """
    G = RS.gain(URU)
    GP = TS.gain(URU)

    areas = RS.a_sample * TS.a_sample * (1 - RS.a_empty) * (1 - TS.a_empty)
    G11 = G / (1 - areas * RS.r_wall * TS.r_wall * G * GP * tdiffuse**2)
    return G11


def Gain_22(RS, TS, URU, tdiffuse):
    r"""Two sphere gain in T sphere for light starting in T sphere.

    Similarly, when the light starts in the second sphere, the gain for light
    on the detector in the second sphere :math:`G_{22}` is found by switching
    all primed variables to unprimed.  Thus :math:`G_{21}(r_s,t_s)` is

    .. math:: G_{22}(r_s,t_s) =\frac{G'(r_s)}{1-a_s a_s'r_w r_w'(1-a_e)(1-a_e')G(r_s)G'(r_s)t_s^2}

    """
    G = RS.gain(URU)
    GP = TS.gain(URU)

    areas = RS.a_sample * TS.a_sample * (1 - RS.a_empty) * (1 - TS.a_empty)
    G22 = GP / (1 - areas * RS.r_wall * TS.r_wall * G * GP * tdiffuse**2)
    return G22


def Two_Sphere_R(RS, TS, UR1, URU, UT1, UTU, f=0):
    """Total gain in R sphere for two sphere configuration.

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

    x = RS.a_detector * (1 - RS.a_empty) * RS.r_wall * G11
    p1 = (1 - f) * UR1
    p2 = RS.r_wall * f
    p3 = (1 - f) * TS.a_sample * (1 - TS.a_empty) * TS.r_wall * UT1 * UTU * GP
    return x * (p1 + p2 + p3)


def Two_Sphere_T(RS, TS, UR1, URU, UT1, UTU, f=0):
    """Total gain in T sphere for two sphere configuration.

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

    x = TS.a_detector * (1 - TS.a_empty) * TS.r_wall * G22
    x *= (1 - f) * UT1 + (1 - RS.a_empty) * RS.r_wall * \
        RS.a_sample * UTU * (f * RS.r_wall + (1 - f) * UR1) * G
    return x
