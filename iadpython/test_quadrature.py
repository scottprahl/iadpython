# pylint: disable=invalid-name
# pylint: disable=no-self-use

"""Tests for Quadrature."""

import unittest
import numpy as np
from nose.plugins.attrib import attr
import iadpython.quadrature as quad

def wip(f):
    """
    Only test functions with @wip decorator.

    Add the @wip decorator before functions that are works-in-progress.
    `nosetests -a wip test_combo.py` will test only those with @wip decorator.
    """
    return attr('wip')(f)

class gauss(unittest.TestCase):
    """Tests for Gaussian-Legendre quadrature."""

    def test_01_gaussian(self):
        """Gaussian quadrature with default endpoints."""
        n = 8
        x, w = quad.gauss(n)
        xx = np.empty(n)
        ww = np.empty(n)
        xx[4] = -0.1834346424956498
        xx[3] = 0.1834346424956498
        xx[5] = -0.5255324099163290
        xx[2] = 0.5255324099163290
        xx[6] = -0.7966664774136267
        xx[1] = 0.7966664774136267
        xx[7] = -0.9602898564975363
        xx[0] = 0.9602898564975363

        ww[4] = 0.3626837833783620
        ww[3] = 0.3626837833783620
        ww[5] = 0.3137066458778873
        ww[2] = 0.3137066458778873
        ww[6] = 0.2223810344533745
        ww[1] = 0.2223810344533745
        ww[7] = 0.1012285362903763
        ww[0] = 0.1012285362903763
        xx = np.flip(xx)
        ww = np.flip(ww)
        np.testing.assert_allclose(x, xx)
        np.testing.assert_allclose(w, ww)

    def test_02_gaussian(self):
        """Gaussian quadrature with specific endpoints."""
        n = 8
        a = -7
        b = 2
        x, w = quad.gauss(n, a, b)

        # integral of x^6 from 0 to 2 should be (x^7)/7
        quad_int = np.sum(x**6 * w)
        anal_int = (b)**7 / 7 - (a)**7 / 7
        np.testing.assert_approx_equal(quad_int, anal_int)

    def test_03_gaussian(self):
        """Gaussian quadrature with endpoint test."""
        n = 5
        x, _ = quad.gauss(n)
        self.assertLess(-1, x[0])
        self.assertLess(x[-1], 1)

    def test_04_gaussian(self):
        """Gaussian quadrature with endpoint test and specified endpoints."""
        n = 5
        a = -7
        b = 2
        x, _ = quad.gauss(n, a, b)
        self.assertLess(a, x[0])
        self.assertLess(x[-1], b)


class radau(unittest.TestCase):
    """Tests for Radau-Legendre quadrature."""

    def test_01_radau(self):
        """Radau quadrature with default endpoints."""
        x, w = quad.radau(8)
        xx = np.empty(8)
        ww = np.empty(8)
        xx[7] = -1
        xx[6] = -0.8874748789261557
        xx[5] = -0.6395186165262152
        xx[4] = -0.2947505657736607
        xx[3] = 0.0943072526611108
        xx[2] = 0.4684203544308211
        xx[1] = 0.7706418936781916
        xx[0] = 0.9550412271225750

        ww[7] = 2/(8*8)
        ww[6] = 0.1853581548029793
        ww[5] = 0.3041306206467856
        ww[4] = 0.3765175453891186
        ww[3] = 0.3915721674524935
        ww[2] = 0.3470147956345014
        ww[1] = 0.2496479013298649
        ww[0] = 0.1145088147442572

        # the provided solutions must be adapted because the lower endpoint is assumed fixed
        xx *= -1
        np.testing.assert_allclose(x, xx)
        np.testing.assert_allclose(w, ww)

    def test_02_radau(self):
        """Radau quadrature with specific endpoints."""
        n = 8
        a = -7
        b = 2
        x, w = quad.radau(n, a, b)

        # integral of x^6 from 0 to 2 should be (x^7)/7
        quad_int = np.sum(x**6 * w)
        anal_int = (b)**7 / 7 - (a)**7 / 7
        np.testing.assert_approx_equal(quad_int, anal_int)

    def test_03_radau(self):
        """Radau quadrature with endpoint test."""
        n = 5
        x, _ = quad.radau(n)
        np.testing.assert_equal(x[-1], 1)
        self.assertLess(-1, x[0])

    def test_04_radau(self):
        """Radau quadrature with endpoint test and specified endpoints."""
        n = 5
        a = -7
        b = 2
        x, _ = quad.radau(n, a, b)
        np.testing.assert_equal(x[-1], b)
        self.assertLess(a, x[0])


class lobatto(unittest.TestCase):
    """Tests for Lobatto-Legendre quadrature."""

    def test_01_lobatto(self):
        """Lobatto quadrature with default endpoints."""
        x, w = quad.lobatto(8)
        xx = np.empty(8)
        ww = np.empty(8)
        xx[7] = -1
        xx[6] = -0.8717401485096066153375
        xx[5] = -0.5917001814331423021445
        xx[4] = -0.2092992179024788687687
        xx[3] = 0.2092992179024788687687
        xx[2] = 0.5917001814331423021445
        xx[1] = 0.8717401485096066153375
        xx[0] = 1

        ww[7] = 0.03571428571428571428571
        ww[6] = 0.210704227143506039383
        ww[5] = 0.3411226924835043647642
        ww[4] = 0.4124587946587038815671
        ww[3] = 0.412458794658703881567
        ww[2] = 0.341122692483504364764
        ww[1] = 0.210704227143506039383
        ww[0] = 0.0357142857142857142857

        xx = np.flip(xx)
        ww = np.flip(ww)
        np.testing.assert_allclose(x, xx)
        np.testing.assert_allclose(w, ww)

    def test_02_lobatto(self):
        """Lobatto quadrature with specific endpoints."""
        n = 8
        a = -7
        b = 2
        x, w = quad.lobatto(n, a, b)

        # integral of x^6 from 0 to 2 should be (x^7)/7
        quad_int = np.sum(x**6 * w)
        anal_int = (b)**7 / 7 - (a)**7 / 7
        np.testing.assert_approx_equal(quad_int, anal_int)

    def test_03_lobatto(self):
        """Lobatto quadrature with endpoint test."""
        n = 5
        x, _ = quad.lobatto(n)
        np.testing.assert_equal(x[-1], 1)
        np.testing.assert_equal(x[0], -1)

    def test_04_lobatto(self):
        """Lobatto quadrature with endpoint test and specified endpoints."""
        n = 5
        a = -7
        b = 2
        x, _ = quad.lobatto(n, a, b)

        np.testing.assert_equal(x[-1], b)
        np.testing.assert_equal(x[0], a)


if __name__ == '__main__':
    unittest.main()
