"""Tests for the Port class."""

import unittest
import numpy as np
from iadpython import Sphere, Port


class TestPort(unittest.TestCase):
    """Unit tests for the Port class."""

    def setUp(self):
        """Set up test conditions for Port tests."""
        d_sample = 25  # 20 mm sample port
        d_sphere = 200  # 200 mm diameter sphere
        R = d_sphere / 2
        d_port = 20
        uru_port = 0.5
        self.sphere = Sphere(d_sphere, d_sample)
        self.sphere.z = R
        self.port = Port(self.sphere, d_port, uru=uru_port, z=R)  # Example port

    def test_cap_area(self):
        """Test the cap_area method."""
        area = self.port.cap_area_exact()
        self.assertTrue(np.isclose(area, 314.94861522998946), "Cap area calculation is incorrect")

    def test_approx_relative_cap_area(self):
        """Test the approx_relative_cap_area method."""
        approx_area = self.port.relative_cap_area()
        self.assertTrue(
            np.isclose(approx_area, 0.0025),
            "Approximate relative cap area calculation is incorrect",
        )

    def test_calculate_sagitta(self):
        """Test the calculate_sagitta method."""
        sagitta = self.port.calculate_sagitta()
        self.assertTrue(np.isclose(sagitta, 0.5012562893380021), "Sagitta calculation is incorrect")

    def test_max_center_chord(self):
        """Test the max_center_chord method."""
        max_chord = self.port.max_center_chord()
        self.assertTrue(
            np.isclose(max_chord, 10.012555011963775),
            "Max center chord calculation is incorrect",
        )

    def test_relative_cap_area(self):
        """Test the relative_cap_area method."""
        rel_area = self.port.relative_cap_area_exact()
        self.assertTrue(
            np.isclose(rel_area, 0.002506281446690011),
            "Relative cap area calculation is incorrect",
        )

    def test_hit(self):
        """Test the hit method."""
        self.assertTrue(self.port.hit(), "Hit detection is incorrect")

    def test_uniform_01(self):
        """Test the generating points on the top of the sphere."""
        for i in range(20):
            x, y, z = self.port.uniform()
            self.assertTrue(self.port.hit(), "Hit detection is incorrect")


if __name__ == "__main__":
    unittest.main()
