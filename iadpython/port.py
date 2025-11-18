# pylint: disable=protected-access)
"""
Implementation of container class for an integrating sphere port.

This module defines the Port class, which is integral to the analysis of light properties
within an integrating sphere. The Port class specifically models a port on the sphere's
surface, calculating and storing its geometrical properties and its interaction with light.

Classes:
    Port: Represents a port in an integrating sphere. It holds information about
    the port's dimensions, position, and optical properties. The class provides
    methods to calculate the port's geometrical characteristics, such as the
    area of the spherical cap it forms, its sagitta, and the chord distance from
    the port's center to its edge. Additionally, it includes functionality to
    assess whether a given point on the sphere's surface falls within the port's
    boundaries.

Functions:
    - cap_area_exact: Calculates the area of the spherical cap formed by the port.
    - relative_cap_area: Calculates the approximate relative area of the spherical cap.
    - calculate_sagitta: Determines the sagitta (or height) of the spherical cap.
    - max_center_chord: Computes the maximum distance from the port's center to its edge.
    - cap_area_exact: Calculates the exact relative area of the spherical cap.
    - hit: Checks if a point on the sphere falls within the port's boundaries.

Example:
    >>> import iadpython as iad
    >>> sphere = iad.Sphere(200, 20)  # Example of creating a sphere object
    >>> port = Port(sphere, 20, 0.5, 0, 0, 0)
    >>> print(port)

See Also:
    Sphere: Another class/module within the iadpython package that models the entire integrating sphere.
"""

import random
import numpy as np
import iadpython as iad


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
        s = x * x + y * y
    return x, y, s


class Port:
    """
    Container class for a port on an integrating sphere.

    A container class for a port in an integrating sphere, which is a structure
    used to analyze light properties. The class calculates and stores various
    geometrical properties of the port, such as its diameter, position, and the
    relative area of the spherical cap it forms on the sphere's surface.

    Attributes:
        sphere (object): Reference to the sphere the port belongs to.
        d (float): Diameter of the port in millimeters (mm).
        uru (float): Diffuse reflectance of diffuse light on the port.
        x (float): X-coordinate of the port's center on the sphere's surface.
        y (float): Y-coordinate of the port's center on the sphere's surface.
        z (float): Z-coordinate of the port's center on the sphere's surface.
        a (float): Relative area of the port's spherical cap to the sphere's surface.
        chord2 (float): Square of the distance from the port center to the port edge.
        sagitta (float): The sagitta (height) of the spherical cap formed by the port.

    Examples:
        Importing the module and creating a sphere with a port:

        ```python
        import iadpython as iad
        s = iad.Sphere(200, 20)
        print(s.sample)
    """

    def __init__(self, sphere, d, uru=0, x=0, y=0, z=0):
        """
        Initializes a Port object with specified dimensions and position within a sphere.

        The coordinates (x,y,z) indicate the center of the port and must be located
        on the sphere surface.  They default to the center of the sphere (0,0,0).  The
        cap location is only used by the `MCSphere` class which uses the `hit()`
        method.

        Args:
            sphere: The sphere object to which the port belongs.
            d: Diameter of the port in millimeters.
            uru: Diffuse reflectance of diffuse light on the port.
            x: X-coordinate of the port's center on the sphere's surface.
            y: Y-coordinate of the port's center on the sphere's surface.
            z: Z-coordinate of the port's center on the sphere's surface.
        """
        self.sphere = sphere
        self._d = d
        self.uru = uru

        self.x = x
        self.y = y
        self.z = z

        self.a = self.relative_cap_area()
        self.sagitta = self.calculate_sagitta()
        self.chord2 = self.max_center_chord() ** 2

    def __repr__(self):
        """Short string for port."""
        return "d=%5.1fmm, uru=%s\n" % (
            self.d,
            iad.stringify("%5.1f%%", self.uru * 100),
        )

    def __str__(self):
        """Return basic details as a string for printing."""
        s = ""
        s += "        diameter = %7.2f mm\n" % self.d
        s += "          radius = %7.2f mm\n" % (self.d / 2)
        s += "           chord = %7.2f mm\n" % np.sqrt(self.chord2)
        s += "         sagitta = %7.2f mm\n" % self.sagitta
        s += "          center = (%6.1f, %6.1f, %6.1f) mm\n" % (self.x, self.y, self.z)
        s += "   relative area = %7.2f%%\n" % (self.a * 100)
        s += "             uru = %s\n" % iad.stringify("%5.1f%%", self.uru * 100)
        return s

    @property
    def d(self):
        """float: Gets the diameter of the port."""
        return self._d

    @d.setter
    def d(self, value):
        """
        Sets the diameter of the port and recalculates geometrical properties.

        Args:
            value (float): The new diameter of the port.

        Raises:
            AssertionError: If the new diameter exceeds the diameter of the sphere.
        """
        assert self.sphere.d >= value, "Sphere must be bigger than the port."
        self._d = value
        self.a = self.relative_cap_area()
        self.sagitta = self.calculate_sagitta()
        self.chord2 = self.max_center_chord()
        self.sphere._a_wall = 1 - self.sphere.sample.a - self.sphere.third.a - self.sphere.detector.a

    def cap_area(self):
        """Approximate area of spherical cap."""
        return np.pi * self.d**2 / 4

    def cap_area_exact(self):
        """Exact area of spherical cap."""
        R = self.sphere.d / 2
        r = self.d / 2
        h = R - np.sqrt(R**2 - r**2)
        return 2 * np.pi * R * h

    def relative_cap_area(self):
        """Calculate approx relative area of spherical cap."""
        R = self.sphere.d / 2
        r = self.d / 2
        return r**2 / (4 * R**2)

    def calculate_sagitta(self):
        """Calculate sagitta of spherical cap."""
        R = self.sphere.d / 2
        r = self.d / 2
        return R - np.sqrt(R**2 - r**2)

    def max_center_chord(self):
        """Distance of chord from port center (on sphere) to port edge."""
        r = self.d / 2
        s = self.sagitta
        return np.sqrt(r**2 + s**2)

    def relative_cap_area_exact(self):
        """Calculate relative area of spherical cap."""
        h = (self.sphere.d - np.sqrt(self.sphere.d**2 - self.d**2)) / 2
        return h / self.sphere.d

    def hit(self):
        """Determine if point on the sphere within the port."""
        r2 = (self.x - self.sphere.x) ** 2
        r2 += (self.y - self.sphere.y) ** 2
        r2 += (self.z - self.sphere.z) ** 2
        #        print('cap center (%7.2f, %7.2f, %7.2f)' % (self.x,self.y,self.z))
        #        print('pt on sph  (%7.2f, %7.2f, %7.2f)' % (self.sphere.x,self.sphere.y,self.sphere.z))
        #        print("cap distance %7.2f %7.2f"% (r2, self.chord2))
        return r2 < self.chord2

    def set_center(self, x, y, z):
        """Centers the cap at x,y,z."""
        assert x**2 + y**2 + z**2 - (self.sphere.d / 2) ** 2 < 1e-6, "center not on sphere."
        self.x = x
        self.y = y
        self.z = z

    def uniform(self):
        """
        Generate a point uniformly distributed on a spherical cap.

        This function generates points uniformly distributed over a spherical cap,
        defined by a specified sagitta (height of the cap from its base to the top)
        and sphere radius. The spherical cap can be positioned either at the top or
        bottom of the sphere. The method utilizes principles from uniform distribution
        on a disk and transforms these points to the spherical cap geometry.

        WARNING: The cap assumed to be at the top of a sphere. Nothing is done to rotate
        the points so they align with the center of the cap!

        The algorithm to generate a random point on a spherical cap is adapted from
        http://marc-b-reynolds.github.io/distribution/2016/11/28/Uniform.html

        Args:
            sagitta: The height of the spherical cap from its base to the top.
            sphere_radius: The radius of the sphere with the cap.

        Returns:
            A numpy array of a random point on the spherical cap surface.
        """
        h = self.sagitta / (self.sphere.d / 2)
        ux, uy, s = uniform_disk()
        root = np.sqrt(h * (2 - h * s))
        point = np.array([ux * root, uy * root, 1 - h * s]) * (self.sphere.d / 2)
        return point
