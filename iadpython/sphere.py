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
    d_third (float): Diameter of the third port.
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

    >>> s = iadpython.Sphere(200, 20, d_third=10, d_detector=10, r_detector=0.8, r_wall=0.99, r_std=0.99)
    >>> print(sphere)
    >>> area_sample = sphere.cap_area(sphere.d_sample)
    >>> print(f"Sample port area: {area_sample:.2f} mm²")
"""

import random
import time
from enum import Enum
import numpy as np
import iadpython as iad


class PortType(Enum):
    """Possible sphere wall locations."""

    WALL = 0
    SAMPLE = 1
    DETECTOR = 2
    THIRD = 3


class Sphere:
    """Container class for a three-port integrating sphere.

    For a reflection measurement, the third port is the diameter of through
    which the light enters to hit the sample.  For a transmission measurement
    this is the port that might allow unscattered transmission to leave. In
    either case, the reflectance from this port is assumed to be zero.

    By default, the sample port is on top (z=-R), the third port is on the bottom (z=R)
    and the detector is on the side (x=R)

    Attributes:
        - d: diameter of integrating sphere [mm]
        - r_wall: reflectivity of the wall
        - a_wall: fraction of the sphere that is walls (relative
        - r_std: reflectivity of the standard used with the sphere
        - sample: port object representing sample port
        - third: port object for empty port in reflection sphere
                or the standard port in transmission sphere
        - detector: port object representing detector port
        - x: x-coordinate of photon on wall
        - y: y-coordinate of photon on wall
        - z: z-coordinate of photon on wall

    Example::

        >>> import iadpython as iad
        >>> s = iad.Sphere(200, 20)
        >>> print(s)
    """

    def __init__(
        self,
        d_sphere,
        d_sample,
        d_third=0,
        r_third=0,
        d_detector=0,
        r_detector=0,
        r_std=0.99,
        r_wall=0.99,
        refl=True,
    ):
        """Initialization."""
        self._d = d_sphere
        self._r_wall = r_wall
        self._r_std = r_std
        self.refl = refl
        R = d_sphere / 2
        self.sample = iad.Port(self, d_sample, uru=0, x=0, y=0, z=-R)
        self.detector = iad.Port(self, d_detector, uru=r_detector, x=R, y=0, z=0)
        self.third = iad.Port(self, d_third, uru=r_third, x=0, y=0, z=R)
        self._a_wall = 1 - self.sample.a - self.third.a - self.detector.a
        self.x = 0
        self.y = 0
        self.z = 0
        self.baffle = False
        self.weight = 0

    def __repr__(self):
        """Return basic details as a string for printing."""
        s = ""
        s += "Sphere: d=%s, " % iad.stringify("%5.2f", self.d)
        s += "r_wall=%s, " % iad.stringify("%5.1f%%", self.r_wall * 100)
        s += "r_std=%s, " % iad.stringify("%5.1f%%", self.r_std * 100)
        s += "baffle= %s\n" % self.baffle
        s += "    Sample: " + repr(self.sample)
        s += "     Third: " + repr(self.third)
        s += "  Detector: " + repr(self.detector)
        return s

    def __str__(self):
        """Return full details as a string for printing."""
        if self.refl:
            s = "Reflectance "
        else:
            s = "Transmittance "
        s += "Sphere\n"
        s += "        diameter = %s mm\n" % iad.stringify("%7.2f", self.d)
        s += "          radius = %s mm\n" % iad.stringify("%7.2f", self.d / 2)
        s += "   relative area = %s\n" % iad.stringify("%7.1f%%", self.a_wall * 100)
        s += "       uru walls = %s\n" % iad.stringify("%7.1f%%", self.r_wall * 100)
        s += "    uru standard = %s\n" % iad.stringify("%7.1f%%", self.r_std * 100)
        s += "          baffle = %s\n" % self.baffle
        s += "Sample Port\n" + str(self.sample)
        if self.refl:
            s += "Entrance Port\n" + str(self.third)
        else:
            s += "Third Port\n" + str(self.third)
        s += "Detector Port\n" + str(self.detector)
        s += "Gain range\n"
        s += "    (sample uru =   0%%)  gain = %s\n" % iad.stringify(
            "%7.3f", self.gain(0.0)
        )
        ss = iad.stringify("%7.3f", self.gain(self.sample.uru))
        s += "    (sample uru = %3.0f%%)  gain = %s\n" % (self.sample.uru * 100, ss)
        s += "    (sample uru =  std)  gain = %s\n" % iad.stringify(
            "%7.3f", self.gain(self.r_std)
        )
        s += "    (sample uru = 100%%)  gain = %s\n" % iad.stringify(
            "%7.3f", self.gain(1.0)
        )
        return s

    def gain(self, sample_uru=None, third_uru=None):
        """
        Determine gain on detector due to multiple bounces in the sphere.

        If UX1 is the power passing through the sample UT1 or the power reflected by the
        sample UR1, then power falling on the detector port will be

        P_detector = a_detector * gain * UX1 * P_0

        where P_0 is the incident light.  The power detected will be reduced by the
        reflectivity of the detector

        P_detected = (1-r_detector) * P_detector
        """
        if sample_uru is None:
            sample_uru = self.sample.uru

        if third_uru is None:
            third_uru = self.third.uru

        if self.baffle:
            tmp = self.detector.a * self.detector.uru + self.sample.a * sample_uru
            r = self.r_wall + (self.third.a / self._a_wall) * third_uru
            denom = 1 - r * (self._a_wall + (1 - self.third.a) * tmp)
        else:
            denom = 1 - self._a_wall * self.r_wall
            denom -= self.detector.a * self.detector.uru
            denom -= self.sample.a * sample_uru
            denom -= self.third.a * third_uru

        denom = np.asarray(denom)
        return np.where(denom == 0, np.inf, 1 / denom)

    def MR(self, sample_ur1, sample_uru=None, R_u=0, f_u=1, f_w=0):
        """
        Determine the value of MR due to multiple bounces in the sphere.

        Args:
            sample_ur1: The reflectance of the sample for normal illumination.
            sample_uru: The reflectance of the sample for diffuse illumination.
            R_u (optional): The unscattered reflectance from the sample.
            f_u (optional): The fraction of unscattered reflected light collected.
            f_w (optional): The fraction of light that hits the sphere wall first.

        Returns:
            float: The calibrated measured reflection
        """
        if sample_uru is None:
            # use collimated total reflectance as approximate value for uru
            sample_uru = sample_ur1

        r_diffuse = sample_ur1 - R_u

        r_first = 1
        if self.baffle:
            # light cannot hit detector and must bounce once
            # however, some light can exit the entrance port
            r_first = self.r_wall * (1 - self.third.a)

        # nothing in sample port or third (entrance) port
        gain_0 = self.gain(0, 0)

        # sample in sample port, third (entrance) port is empty
        gain = self.gain(sample_uru, 0)

        # sample port has known standard, third (entrance) port is empty
        gain_cal = self.gain(self.r_std, 0)

        P_cal = gain_cal * (self.r_std * (1 - f_w) + f_w * self.r_wall)
        P_0 = gain_0 * (f_w * self.r_wall)

        P_ss = r_first * (r_diffuse * (1 - f_w) + f_w * self.r_wall)
        P_su = self.r_wall * (1 - f_w) * f_u * R_u
        P = gain * (P_ss + P_su)

        MR = self.r_std * (P - P_0) / (P_cal - P_0)

        #         print("UR1   =  %6.3f   r_diffuse = %6.3f" % (sample_ur1, r_diffuse))
        #         print("URU   =  %6.3f   R_u       = %6.3f" % (sample_uru, R_u))
        #         print("P_ss  =  %6.3f   P_su      = %6.3f" % (P_ss, P_su))
        #         print("G_0   =  %6.3f   P_0       = %6.3f" % (gain_0, P_0))
        #         print("G     =  %6.3f   P         = %6.3f" % (gain, P))
        #         print("G_cal =  %6.3f   P_cal     = %6.3f" % (gain_cal, P_cal))
        #         print("MR    =  %6.3f" % (MR))

        return MR

    def MT(self, sample_ut1, sample_uru, T_u=0, f_u=1):
        """
        Determine the measured transmission (MT) due to multiple bounces in the sphere.

        The f_u variable describes the fraction of unscattered transmission
        that is collected by the sphere.  It is equivalent to the -C option for the iad
        program and the default is that all the unscattered transmission is collected
        (because the third port is blocked).  If the third port allows unscattered light
        to leave the sphere then it should be set to zero.

        Args:
            sample_ut1: The transmission of the sample for normal illumination.
            sample_uru: The reflectance of the sample for diffuse illumination.
            T_u (optional): The unscattered transmission of the sample.
            f_u (optional): The fraction of unscattered transmission collected.

        Returns:
            float: The calculated measured transmission (MT) value.
        """
        if self.third.a == 0:
            # sample in sample port, third port is always sphere wall
            r_cal = self.r_wall
            r_third = self.r_wall

        elif f_u == 0:
            # sample in sample port, third port is empty except for calibration
            r_cal = self.r_std
            r_third = 0

        else:
            # sample in sample port, third port always has known standard
            r_cal = self.r_std
            r_third = self.r_std

        r_first = 1
        if self.baffle:
            r_first = self.r_wall * (1 - self.third.a) + r_third * self.third.a

        ut1_diffuse = sample_ut1 - T_u

        gain = self.gain(sample_uru, r_third)
        gain_cal = self.gain(0, r_cal)

        P_ss = r_first * ut1_diffuse
        P_su = r_third * T_u * f_u
        P = (P_su + P_ss) * gain
        P_cal = r_cal * gain_cal

        MT = r_cal * P / P_cal

        #         print("UT1     =  %6.3f   URU   = %6.3f" % (sample_ut1, sample_uru))
        #         print("T_u     =  %6.3f   f_u   = %5.2f" % (T_u,f_u))
        #         print("P_ss    =  %6.3f   P_su  = %6.3f" % (P_ss, P_su))
        #         print("G       =  %6.3f   P     = %6.3f" % (gain, P))
        #         print("G_cal   =  %6.3f   P_cal = %6.3f" % (gain_cal, P_cal))
        #         print("r_first =  %6.3f" % (r_first))
        #         print("r_cal   =  %6.3f" % (r_cal))
        #         print("r_third =  %6.3f" % (r_third))
        #         print("MT      =  %6.3f" % (MT))

        return MT

    def pdetector(self):
        """Print the detector power."""
        P = (1 - self.third.a) * self.r_wall
        N = 1000
        pd = np.zeros(N)
        pw = np.zeros(N)
        ps = np.zeros(N)

        pd[0] = self.detector.a * P
        ps[0] = self.sample.a * P
        pw[0] = self.a_wall * P

        for j in range(N - 1):
            pd[j + 1] = self.detector.a * self.r_wall * pw[j]
            ps[j + 1] = self.sample.a * self.r_wall * pw[j]
            pw[j + 1] = self.a_wall * self.r_wall * pw[j]
            pw[j + 1] += (1 - self.third.a) * self.detector.uru * pd[j]
            pw[j + 1] += (1 - self.third.a) * self.sample.uru * ps[j]

        sumw = np.cumsum(pw)
        sumw -= sumw[1]
        sumd = np.cumsum(pd)
        print(" k    P_d^k   P_w^k    sum(P_d^k)   sum(P_w^k)")
        for j in range(10):
            print(
                "%3d %9.5f %9.5f %9.5f %9.5f" % (j + 1, pd[j], pw[j], sumd[j], sumw[j])
            )
        print(
            "%3d %9.5f %9.5f %9.5f %9.5f"
            % (N - 1, pd[N - 1], pw[N - 1], sumd[N - 1], sumw[N - 1])
        )
        print()

        pd[0] = self.detector.a * P
        ps[0] = self.sample.a * P
        pw[0] = self.a_wall * P

        pw[1] = self.a_wall * self.r_wall * pw[0]
        pw[1] += (1 - self.third.a) * self.detector.uru * pd[0]
        pw[1] += (1 - self.third.a) * self.sample.uru * ps[0]
        pd[1] = self.detector.a * self.r_wall * pw[0]

        beta = 1 - self.third.a
        beta *= self.detector.a * self.detector.uru + self.sample.a * self.sample.uru
        for j in range(1, N - 1):
            pw[j + 1] = self.r_wall * (self.a_wall * pw[j] + beta * pw[j - 1])

        sumw = np.cumsum(pw)
        sumw -= sumw[1]
        sumd = np.cumsum(pd)
        print(" k    P_d^k   P_w^k    sum(P_d^k)   sum(P_w^k)")
        for j in range(10):
            print(
                "%3d %9.5f %9.5f %9.5f %9.5f" % (j + 1, pd[j], pw[j], sumd[j], sumw[j])
            )
        print(
            "%3d %9.5f %9.5f %9.5f %9.5f"
            % (N - 1, pd[N - 1], pw[N - 1], sumd[N - 1], sumw[N - 1])
        )
        print()

        beta = 1 - self.third.a
        beta *= self.detector.a * self.detector.uru + self.sample.a * self.sample.uru
        numer = self.a_wall**3 * self.r_wall + beta * (
            2 * self.a_wall + self.a_wall**2 * self.r_wall + beta
        )
        denom = 1 - self.r_wall * (self.a_wall + beta)
        sum3 = self.r_wall * numer / denom * P

        pdx = self.detector.a * P
        pdx += self.detector.a * self.r_wall * self.a_wall * P
        pdx += self.detector.a * self.r_wall * (self.a_wall**2 * self.r_wall + beta) * P
        pdx += self.detector.a * self.r_wall * sum3

        print("%9.5f" % pdx)

        beta = 1 - self.third.a
        beta *= self.detector.a * self.detector.uru + self.sample.a * self.sample.uru
        pdx = self.detector.a * P / (1 - self.r_wall * (self.a_wall + beta))
        print("%9.5f" % pdx)

    @property
    def d(self):
        """Getter property for sphere diameter."""
        return self._d

    @d.setter
    def d(self, value):
        """When size is changed ratios become invalid."""
        assert self.sample.d <= value, "sphere must be bigger than sample port"
        assert self.third.d <= value, "sphere must be bigger than third port"
        assert self.detector.d <= value, "sphere must be bigger than detector port"
        self._d = value
        self._a_wall = 1 - self.sample.a - self.third.a - self.detector.a

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
        self.third.d = 0
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
            assert (
                0 <= value.all() <= 1
            ), "Reflectivity of standard must be between 0 and 1"
        self._r_std = value
        self.gain_cal = self.gain(self.r_std)

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
            assert (
                0 <= value.all() <= 1
            ), "Reflectivity of standard must be between 0 and 1"
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

    def do_one_photon(self, double=False, weight=1):
        """
        Bounce photon inside sphere until it leaves.

        The photon can leave through the third port, the detector port,
        the sample port (transmitted/absorbed by the sample), or be absorbed
        by the sphere walls.

        This routine keeps propagating a photon until the weight drops to zero
        unless double is True

        If double is True then
        """
        bounces = 0
        detected = 0
        transmitted = 0

        # photon is launched from sample
        last_location = iad.PortType.SAMPLE
        #        R = self.d/2
        while weight > 0:

            # lastx = self.x
            # lasty = self.y
            # lastz = self.z
            self.x, self.y, self.z = self.uniform()

            if self.detector.hit():
                # avoid hitting self
                if last_location == iad.PortType.DETECTOR:
                    continue

                # sample --> detector prohibited
                if last_location == iad.PortType.SAMPLE and self.baffle:
                    continue

                #                vx=self.x-lastx
                #                vy=self.y-lasty
                #                vz=self.z-lastz
                #                RR = np.sqrt(vx*vx+vy*vy+vz*vz)
                #                print("n = [%5.2f, %5.2f, %5.2f]" % (self.x/R, self.y/R, self.z/R))
                #                print("v = [%5.2f, %5.2f, %5.2f]" % (vx/RR, vy/RR, vz/RR))
                #                costheta = abs((vx * self.x) + (vy * self.y) + (vz * self.z))/R/RR
                #                print(costheta)

                # record detected light and update weight
                d_transmitted = weight * (1 - self.detector.uru)
                detected += d_transmitted
                weight -= d_transmitted
                last_location = iad.PortType.DETECTOR

            elif self.sample.hit():
                # avoid hitting self
                if last_location == iad.PortType.SAMPLE:
                    continue

                # detector --> sample prohibited
                if last_location == iad.PortType.DETECTOR and self.baffle:
                    continue

                last_location = iad.PortType.SAMPLE
                if not double:
                    weight *= self.sample.uru  # photon stays in sphere
                else:
                    # in a double sphere setup, the photon may pass into the second sphere
                    # the photon continues with equal weight if it is reflected
                    # otherwise the photon wil be absorbed or transmitted.
                    if random.random() > self.sample.uru:
                        transmitted = weight
                        weight = 0

            elif self.third.hit():
                weight *= self.third.uru
                last_location = iad.PortType.THIRD

            else:
                # must have hit wall
                weight *= self.r_wall
                last_location = iad.PortType.WALL

            if 0 < weight < 1e-4:
                if random.random() < 0.1:
                    weight *= 10
                else:
                    weight = 0

            bounces += 1

        return detected, transmitted, bounces

    def do_N_photons_raw_array(self, N, num_trials=10, double=False):
        """Do a Monte Carlo simulation with N photons."""
        random.seed(time.time())  # Use current time as seed

        total_detected = np.zeros(num_trials)
        total_bounces = np.zeros(num_trials)

        N_per_trial = N // num_trials

        for j in range(num_trials):
            for _i in range(N_per_trial):
                detected, _, bounces = self.do_one_photon(double=double)
                total_detected[j] += detected
                total_bounces[j] += bounces

        detected = total_detected / N_per_trial
        bounces = total_bounces / N_per_trial
        return detected, bounces

    def do_N_photons_gain_array(self, N, num_trials=10, double=False):
        """Do a Monte Carlo simulation with N photons."""
        detected, bounces = self.do_N_photons_raw_array(N, num_trials, double)

        # convert from detected to gain
        scale = self.detector.a * (1 - self.detector.uru)
        if self.baffle:
            scale *= (1 - self.third.a) * self.r_wall + self.third.a * self.third.uru

        gains = detected / scale
        return gains, bounces

    def do_N_photons_gain(self, N, double=False):
        """Do a Monte Carlo simulation with N photons."""
        num_trials = 20
        gains, _ = self.do_N_photons_gain_array(N, num_trials, double)

        gain_ave = np.mean(gains)
        gain_stderr = np.std(gains) / np.sqrt(len(gains))

        return gain_ave, gain_stderr
