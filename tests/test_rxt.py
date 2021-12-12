# pylint: disable=invalid-name
# pylint: disable=bad-whitespace
# pylint: disable=no-self-use

"""Tests for NIST reflectance data."""

import unittest
import numpy as np
import iadpython


class TestRXT(unittest.TestCase):
    """RXT files data."""

    def test_rxt_01(self):
        """Verify that subject 1 is read correctly."""
        s = iadpython.read_rxt('./tests/basic-A.rxt')
        print(s)


if __name__ == '__main__':
    unittest.main()
