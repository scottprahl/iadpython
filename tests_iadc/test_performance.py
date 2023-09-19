# pylint: disable=invalid-name
# pylint: disable=too-many-locals
# pylint: disable=consider-using-f-string

"""Compare speed of pure python implementation with c-library."""

import time
import unittest
import scipy.optimize
import numpy as np
import iadpython
from iadpython import iadc


class speed(unittest.TestCase):
    """Performance testing."""

    def test_speed_01(self):
        """First Speed Test."""
        n_slab = 1.5                  # slab with boundary reflection
        n_slide = 1.0                 # no glass slides above and below the sample
        b = 1                         # this is pretty much infinite
        g = 0.9                       # isotropic scattering is fine
        a = np.linspace(0, 1, 5000)   # albedo varies between 0 and 1

        start_time = time.perf_counter()
        _, _, _, _ = iadc.rt(n_slab, n_slide, a, b, g)
        deltaC = time.perf_counter() - start_time

        start_time = time.perf_counter()
        s = iadpython.Sample(a=a, b=b, g=g, n=n_slab, n_above=n_slide,
                             n_below=n_slide, quad_pts=16)
        _, _, _, _ = s.rt()
        deltaP = time.perf_counter() - start_time
        print("#     C    python  ratio")
        print("1 %7.2f %7.2f %5.0f%%" % (deltaC, deltaP, 100 * deltaP / deltaC))

    def test_speed_02(self):
        """Second Speed Test."""
        n_slab = 1.0                  # ignore boundary reflection
        n_slide = 1.0                 # no glass slides above and below the sample
        b = 10000                     # this is pretty much infinite
        g = 0.0                       # isotropic scattering is fine
        a = np.linspace(0, 1, 2000)   # albedo varies between 0 and 1

        start_time = time.perf_counter()
        _, _, _, _ = iadc.rt(n_slab, n_slide, a, b, g)
        deltaC = time.perf_counter() - start_time

        start_time = time.perf_counter()
        s = iadpython.Sample(a=a, b=b, g=g, n=n_slab, n_above=n_slide,
                             n_below=n_slide, quad_pts=16)
        _, _, _, _ = s.rt()
        deltaP = time.perf_counter() - start_time
        print("2 %7.2f %7.2f %5.0f%%" % (deltaC, deltaP, 100 * deltaP / deltaC))

    def test_speed_03(self):
        """Third Speed Test."""
        n_slab = 1.4                  # sample has refractive index
        n_slide = 1.5                 # glass slides above and below the sample
        g = 0.0                       # isotropic scattering is fine
        a = 0.9                       # scattering is 9X absorption
        b = np.linspace(0, 10, 4000)  # opstart_timeal thickness

        start_time = time.perf_counter()
        _, _, _, _ = iadc.rt(n_slab, n_slide, a, b, g)
        deltaC = time.perf_counter() - start_time

        start_time = time.perf_counter()
        s = iadpython.Sample(a=a, b=b, g=g, n=n_slab,
                             n_above=n_slide, n_below=n_slide, quad_pts=16)
        _, _, _, _ = s.rt()
        deltaP = time.perf_counter() - start_time
        print("3 %7.2f %7.2f %5.0f%%" % (deltaC, deltaP, 100 * deltaP / deltaC))

    def test_speed_04(self):
        """Fourth Speed Test."""
        N = 201
        n_slab = 1.0                    # ignore boundary reflection
        n_slide = 1.0                   # no glass slides above and below the sample
        g = 0.0                         # isotropic scattering is fine
        a = np.linspace(0.01, 0.99, N)  # avoid extremes because
        bmin = np.empty(N)
        b = 100000

        def f(bb):
            """C Function to find 99 percent."""
            ur1c, _, _, _ = iadc.rt(n_slab, n_slide, aa, bb, g)
            return (ur1c - ur1_inf * 0.99)**2

        def ff(bb):
            """Python Function to find 99 percent."""
            s.b = bb
            ur1p, _, _, _ = s.rt()
            return (ur1p - ur1_inf * 0.99)**2

        start_time = time.perf_counter()
        for i in range(N):
            aa = a[i]
            ur1_inf, _, _, _ = iadc.rt(n_slab, n_slide, aa, b, g)
            bmin[i] = scipy.optimize.brent(f)
        deltaC = time.perf_counter() - start_time

        start_time = time.perf_counter()
        s = iadpython.Sample(a=a, b=b, g=g, n=n_slab,
                             n_above=n_slide, n_below=n_slide, quad_pts=16)

        for i in range(N):
            s.a = a[i]
            s.b = b
            ur1_inf, _, _, _ = s.rt()
            bmin[i] = scipy.optimize.brent(ff)
        deltaP = time.perf_counter() - start_time
        print("4 %7.2f %7.2f %5.0f%%" % (deltaC, deltaP, 100 * deltaP / deltaC))

    def test_speed_05(self):
        """Fifth Speed Test."""
        n_slab = 1.0                 # ignore boundaries
        n_slide = 1.0                # no glass slides above and below the sample
        b = 1000                     # relatively thin sample
        ap = np.linspace(0, 1, 800)  # albedo varies between 0 and 1
        g = [0, 0.5, 0.95]
        ur1 = np.empty_like(g, dtype=list)

        start_time = time.perf_counter()
        for i in range(3):
            a = ap / (1 - g[i] + ap * g[i])
            ur1[i], _, _, _ = iadc.rt(n_slab, n_slide, a, b, g[i])
        deltaC = time.perf_counter() - start_time

        start_time = time.perf_counter()
        s = iadpython.Sample(b=b, n=n_slab,
                             n_above=n_slide, n_below=n_slide, quad_pts=16)
        for i in range(3):
            s.a = ap / (1 - g[i] + ap * g[i])
            s.g = g[i]
            ur1[i], _, _, _ = s.rt()
        deltaP = time.perf_counter() - start_time
        print("5 %7.2f %7.2f %5.0f%%" % (deltaC, deltaP, 100 * deltaP / deltaC))
