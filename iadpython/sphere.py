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
    gain: sphere gain relative to isotropic diffuse source in center of black sphere

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
from enum import Enum	
import iadpython

class PortType(Enum):	
    """Possible sphere wall locations."""	
    EMPTY = 0	
    WALL = 1	
    SAMPLE = 2	
    DETECTOR = 3
    FIRST = 4

def stringify(form, x):
    if x is None:
        s = 'None'
    elif np.isscalar(x):
        s = form % x
    else:
        mn = min(x)
        mx = max(x)
        s = form % mn
        s += ' to '
        s += form % mx
    return s

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
        s += "        diameter = %s mm\n" % stringify("%7.2f", self.d)
        s += "          radius = %s mm\n" % stringify("%7.2f", self.d/2)
        s += "   relative area = %7.4f\n" % self.a_wall
        s += "       uru walls = %s\n" % stringify("%7.1f%%", self.r_wall * 100)
        s += "    uru standard = %s\n" % stringify("%7.1f%%", self.r_std * 100)
        s += "          baffle = %s\n" % self.baffle
        s += "Sample Port\n" + str(self.sample)
        s += "Detector Port\n" + str(self.detector)
        s += "Empty Port\n" + str(self.empty)
        s += "Gain range\n"
        s += "         nothing = %s\n" % stringify("%7.3f", self.gain(0.0))
        s += "        standard = %s\n" % stringify("%7.3f", self.gain(self.r_std))
        s += "            100%% = %s\n" % stringify("%7.3f", self.gain(1.0))
        return s

    def gain(self, sample_uru=None):
        """
        Determine gain due to multiple bounces in the sphere.
        
        If UX1 is the power passing through the sample UT1 or reflected by the
        sample UR1, then power falling on the detector will be
        
        P_detector = a_detector * gain * UX1 * P_0

        and power detected will be
        
        P_detected = (1-r_detector) * a_detector * gain * UX1 * P_0
        """
        if sample_uru is not None:
            original_uru = self.sample.uru
            self.sample.uru = sample_uru
        
        tmp = self.detector.a * self.detector.uru + self.sample.a * self.sample.uru

        if self.baffle:
            denom = 1 - self.r_wall * (self.a_wall + (1 - self.empty.a) * tmp)
        else:
            denom = 1 - self._a_wall * self.r_wall - tmp
        
        g = 1 / denom
        
        # restore value
        if sample_uru is not None:
            self.sample.uru = original_uru
        return g

    def pdetector(self):
        P = (1-self.empty.a) * self.r_wall
        N = 1000
        pd = np.zeros(N)
        pw = np.zeros(N)
        ps = np.zeros(N)
        
        pd[0] = self.detector.a * P
        ps[0] = self.sample.a * P
        pw[0] = self.a_wall * P

        for j in range(N-1):
            pd[j+1] = self.detector.a * self.r_wall * pw[j]
            ps[j+1] = self.sample.a * self.r_wall * pw[j]
            pw[j+1] = self.a_wall * self.r_wall * pw[j] 
            pw[j+1] += (1 - self.empty.a) * self.detector.uru * pd[j]
            pw[j+1] += (1 - self.empty.a) * self.sample.uru * ps[j]
            
        sumw = np.cumsum(pw)
        sumw -= sumw[1]
        sumd = np.cumsum(pd)
        print(' k    P_d^k   P_w^k    sum(P_d^k)   sum(P_w^k)')
        for j in range(10):
            print("%3d %9.5f %9.5f %9.5f %9.5f" % (j+1, pd[j], pw[j], sumd[j], sumw[j]))
        print("%3d %9.5f %9.5f %9.5f %9.5f" % (N-1, pd[N-1], pw[N-1],  sumd[N-1], sumw[N-1]))
        print()
        
        pd[0] = self.detector.a * P
        ps[0] = self.sample.a * P
        pw[0] = self.a_wall * P

        pw[1] = self.a_wall * self.r_wall * pw[0] 
        pw[1] += (1 - self.empty.a) * self.detector.uru * pd[0]
        pw[1] += (1 - self.empty.a) * self.sample.uru * ps[0]
        pd[1] = self.detector.a * self.r_wall * pw[0]

        beta = (1 - self.empty.a) 
        beta *= self.detector.a * self.detector.uru + self.sample.a * self.sample.uru
        for j in range(1,N-1):
            pw[j+1] = self.r_wall * (self.a_wall * pw[j] + beta * pw[j-1])
        
        sumw = np.cumsum(pw)
        sumw -= sumw[1]
        sumd = np.cumsum(pd)
        print(' k    P_d^k   P_w^k    sum(P_d^k)   sum(P_w^k)')
        for j in range(10):
            print("%3d %9.5f %9.5f %9.5f %9.5f" % (j+1, pd[j], pw[j], sumd[j], sumw[j]))
        print("%3d %9.5f %9.5f %9.5f %9.5f" % (N-1, pd[N-1], pw[N-1],  sumd[N-1], sumw[N-1]))
        print()

        beta = (1 - self.empty.a) 
        beta *= self.detector.a * self.detector.uru + self.sample.a * self.sample.uru
        numer = self.a_wall**3*self.r_wall+ beta*(2*self.a_wall + self.a_wall**2*self.r_wall + beta)
        denom = 1-self.r_wall *(self.a_wall +  beta)
        sum3 = self.r_wall * numer/denom * P
        
        pdx = self.detector.a * P
        pdx += self.detector.a * self.r_wall * self.a_wall  * P
        pdx += self.detector.a * self.r_wall * (self.a_wall**2 * self.r_wall + beta) * P
        pdx += self.detector.a * self.r_wall * sum3
        
        print("%9.5f" % pdx)
        
        beta = (1 - self.empty.a) 
        beta *= self.detector.a * self.detector.uru + self.sample.a * self.sample.uru
        pdx = self.detector.a * P/(1-self.r_wall*(self.a_wall+beta))
        print("%9.5f" % pdx)
        
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
        self.gain_std = self.gain(self.r_std)

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
    
        Using Gaussian distribution for all three coordinates ensures a
        uniform distribution on the surface of the sphere. See
        
        https://math.stackexchange.com/questions/1585975
        
        Returns:
            (x, y, z) for a random point on the sphere's surface.
        """
        while True:
            x = random.gauss()
            y = random.gauss()
            z = random.gauss()
            r = np.sqrt(x**2 + y**2 + z**2)
            if r > 0:
                return np.array([x, y, z]) * (self.d / 2) / r

    def do_one_photon(self):
        """Bounce photon inside sphere until it leaves."""
        bounces = 0
        detected = 0

        # assume photon launched form sample
        weight = 1
        last_location = PortType.SAMPLE
#        R = self.d/2
        while weight > 1e-4:

            lastx = self.x
            lasty = self.y
            lastz = self.z
            self.x, self.y, self.z = self.uniform()

            if self.detector.hit():
                if last_location == PortType.DETECTOR:   # avoid hitting self
                    continue
                if last_location == PortType.SAMPLE and self.baffle: # sample --> detector prohibited
                    continue

                # record detected light and update weight
                vx=self.x-lastx
                vy=self.y-lasty
                vz=self.z-lastz
#                RR = np.sqrt(vx*vx+vy*vy+vz*vz)
#                print("n = [%5.2f, %5.2f, %5.2f]" % (self.x/R, self.y/R, self.z/R))
#                print("v = [%5.2f, %5.2f, %5.2f]" % (vx/RR, vy/RR, vz/RR))
#                costheta = abs((vx * self.x) + (vy * self.y) + (vz * self.z))/R/RR
#                print(costheta)
                transmitted = weight * (1-self.detector.uru) 
                detected += transmitted
                weight -= transmitted
                last_location = PortType.DETECTOR

            elif self.sample.hit():
                if last_location == PortType.SAMPLE:    # avoid hitting self
                    continue
                if last_location == PortType.DETECTOR and self.baffle: # detector --> sample prohibited
                    continue
                weight *= self.sample.uru
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
        
    def do_N_photons(self, N):
        
        num_trials = 10
        total_detected = np.zeros(num_trials)
        total_bounces = np.zeros(num_trials)
        
        N_per_trial = N // num_trials
        
        total = 0
        for j in range(num_trials):
            for i in range(N_per_trial):
                detected, bounces = self.do_one_photon()
    #            print("%d %8.3f %d" % (i,detected, bounces))
                total_detected[j] += detected
                total_bounces[j] += bounces
                total += 1
        
        ave = np.mean(total_detected)/N_per_trial
        std = np.std(total_detected)/N_per_trial
        stderr = std / np.sqrt(num_trials)
        print("average detected   = %.3f ± %.3f" % (ave,stderr))
        ave /= self.detector.a * (1-self.detector.uru)
        std /= self.detector.a * (1-self.detector.uru)
        stderr = std / np.sqrt(num_trials)
        print("average gain       = %.3f ± %.3f" % (ave,stderr))
        print("calculated gain    = %.3f" % self.gain())
        
        ave = np.mean(total_bounces)/N_per_trial
        std = np.std(total_bounces)/N_per_trial
        stderr = std / np.sqrt(num_trials)
        print("average bounces    = %.3f ± %.3f" % (ave,stderr))
                

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
