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

import numpy as np
import random
from enum import Enum
from iadpython import Sphere

class PortType(Enum):
    """Possible sphere wall locations."""
    EMPTY = 0
    WALL = 1
    SAMPLE = 2
    DETECTOR = 3

class MCSphere(Sphere):
    """Class for an Monte Carlo integrating sphere calcs.

    The center of the sphere is at (0,0,0)
    For a reflection measurement, the empty port is the diameter of through
    which the light enters to hit the sample.  For a transmission measurement
    this is the port that might allow unscattered transmission to leave. In
    either case, the reflectance from this port is assumed to be zero.

    Attributes:
        - d_sphere: diameter of integrating sphere [mm]
        - d_sample: diameter of the port that has the sample [mm]
        - d_empty: diameter of the empty port [mm]
        - d_detector: diameter of port with detector [mm]
        - r_detector: reflectivity of the detector
        - r_wall: reflectivity of the wall
        - r_std: reflectivity of the standard used with the sphere

    Example::

        >>> import iadpython as iad
        >>> s = iad.Sphere(200, 20)
        >>> print(s)
    """

    def __init__(self, d_sphere, d_sample, d_empty=0,
                 d_detector=0, r_detector=0, r_wall=0.99, r_std=0.99):

        super().__init__(d_sphere, d_sample, d_empty, d_detector,
                         r_detector, r_wall, r_std)
        self.weight = 1

    def __str__(self):
        """Return basic details as a string for printing."""
        s = super().__str__()
        s = "\n"
        s += "Sample Port\n"
        s += "          center = (%.1f, %.1f %.1f) mm\n" % (self.sample_x, self.sample_y, self.sample_z)
        s += "           chord = %.1f mm\n" % np.sqrt(self.sample_chord_sqr)
        s += "          radius = %.1f mm\n" % (np.sample_d/2)
        s += "         sagitta = %.1f mm\n" % (np.sample_sagitta)
        s += "Detector Port\n"
        s += "          center = (%.1f, %.1f %.1f) mm\n" % (self.detector_x, self.detector_y, self.detector_z)
        s += "           chord = %.1f mm\n" % np.sqrt(self.detector_chord_sqr)
        s += "          radius = %.1f mm\n" % (np.detector_d/2)
        s += "         sagitta = %.1f mm\n" % (np.detector_sagitta)
        s += "Empty Port\n"
        s += "          center = (%.1f, %.1f %.1f) mm\n" % (self.empty_x, self.empty_y, self.empty_z)
        s += "           chord = %.1f mm\n" % np.sqrt(self.empty_chord_sqr)
        s += "          radius = %.1f mm\n" % (np.empty_d/2)
        s += "         sagitta = %.1f mm\n" % (np.empty_sagitta)
        return s

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
                
