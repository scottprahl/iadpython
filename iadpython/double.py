"""
Class for managing double integrating spheres.

This module contains the DoubleSphere class, designed to simulate and analyze the
behavior of light with a sample sandwiched between a reflection and transmission
sphere.

Attributes:
    r_sphere: Reflectance sphere
    t_sphere: Transmittance sphere
    ur1: total reflection of sample for normal incidence
    ut1: total transmission of sample for normal incidence
    uru: total reflection of sample for diffuse incidence
    utu: total transmission of sample for diffuse incidence

Example usage:
    >>> import iadpython
    >>> rs = iadpython.Sphere(250,20)
    >>> ts = iadpython.Sphere(250,20)
    >>> d = iadpython.DoubleSphere(rs, ts)
    >>> print(d)
"""

import random
import time
import numpy as np


class DoubleSphere():
    """Container class for two  three-port integrating sphere.

    Attributes:
        - r_sphere: Reflectance sphere
        - t_sphere: Transmittance sphere
        - current: Current sphere photon is in
        - ur1: total reflection of sample for normal incidence
        - ut1: total transmission of sample for normal incidence
        - uru: total reflection of sample for diffuse incidence
        - utu: total transmission of sample for diffuse incidence

    Example::

        >>> import iadpython as iad
        >>> s = iad.Sphere(200, 20)
        >>> print(s)
    """

    def __init__(self, r_sphere, t_sphere):
        """Initialization."""
        self.r_sphere = r_sphere
        self.t_sphere = t_sphere
        self.current = self.r_sphere
        self.ur1 = 0
        self.ut1 = 1
        self._uru = self.ur1
        self._utu = self.ut1
        self.fraction_absorbed = 1
        self.absorbed = 1 - self._uru - self._utu

    def __str__(self):
        """Return full details as a string for printing."""
        s = ''
        s += "Reflection Sphere\n"
        s += str(self.r_sphere) + '\n'
        s += "Transmission Sphere\n"
        s += str(self.t_sphere) + '\n'
        s += "Sample Properties\n"
        s += "   ur1 = %7.3f\n" % self.ur1
        s += "   ut1 = %7.3f\n" % self.ut1
        s += "   uru = %7.3f\n" % self.uru
        s += "   utu = %7.3f\n" % self.utu
        return s

    def update_fraction_abs(self):
        if np.isscalar(self.absorbed):
            if self._utu + self.absorbed > 0:
                self.fraction_absorbed = self.absorbed / (self.utu + self.absorbed)
        else:
            for i in range(len(self.absorbed)):
                if np.isscalar(self._utu):
                    if self._utu + self.absorbed[i] > 0:
                        self.fraction_absorbed[i] = self.absorbed[i] / (self.utu + self.absorbed[i])
                else:
                    if self._utu[i] + self.absorbed[i] > 0:
                        self.fraction_absorbed[i] = self.absorbed[i] / (self.utu[i] + self.absorbed[i])

    @property
    def uru(self):
        """Getter property for sample uru."""
        return self._uru

    @uru.setter
    def uru(self, value):
        """Need to update sample uru."""
        self._uru = value
        self.r_sphere.sample.uru = value
        self.t_sphere.sample.uru = value
        self.absorbed = 1 - self._uru - self._utu
        self.fraction_absorbed = 0 * self.absorbed
        self.update_fraction_abs()

    @property
    def utu(self):
        """Need to update sample utu."""
        return self._utu

    @utu.setter
    def utu(self, value):
        """When size is changed ratios become invalid."""
        self._utu = value
        self.absorbed = 1 - self._uru - self._utu
        self.fraction_absorbed = 0 * self.absorbed
        self.update_fraction_abs()

    def do_one_photon(self):
        """Bounce photon in double spheres until it is detected or lost."""
        weight = 1
        passes = 0
        r_detected = 0
        t_detected = 0

        # photon normally incident on sample
        x = random.random()
        if x < self.ur1:  # reflected by sample
            self.current = self.r_sphere

        elif x < self.ur1 + self.ut1:  # transmitted through sample
            self.current = self.t_sphere
            passes = 1

        else:  # absorbed by sample
            weight = 0

        while weight > 0:
            detected, transmitted, _ = self.current.do_one_photon(weight=weight, double=True)

            if transmitted > 0:  # hit sample
                if random.random() < self.utu:  # passed through sample, switch spheres
                    passes += 1
                    if self.current == self.r_sphere:
                        self.current = self.t_sphere
                    else:
                        self.current = self.r_sphere
                    weight = transmitted
                else: # absorbed by sample
                    weight = 0
            else:
                weight = 0
                if self.current == self.r_sphere:
                    assert passes % 2 == 0, "reflection sphere should have even number of passes"
                    r_detected += detected
                else:
                    assert passes % 2 == 1, "reflection sphere should have odd number of passes"
                    t_detected += detected

        return r_detected, t_detected, passes

    def do_N_photons(self, N):
        """Do a Monte Carlo simulation with N photons."""
        # Use current time as seed
        random.seed(time.time())

        num_trials = 10
        total_r_detected = np.zeros(num_trials)
        total_t_detected = np.zeros(num_trials)
#        total_bounces = np.zeros(num_trials)

        N_per_trial = N // num_trials

        total = 0
        for j in range(num_trials):
            for _i in range(N_per_trial):
                r_detected, t_detected, _ = self.do_one_photon()
    #            print("%d %8.3f %8.3f" % (i, r_detected, t_detected))
                total_r_detected[j] += r_detected
                total_t_detected[j] += t_detected
                total += 1

        ave_r = np.mean(total_r_detected) / N_per_trial
        std_r = np.std(total_r_detected) / N_per_trial
        stderr_r = std_r / np.sqrt(num_trials)

        ave_t = np.mean(total_t_detected) / N_per_trial
        std_t = np.std(total_t_detected) / N_per_trial
        stderr_t = std_t / np.sqrt(num_trials)

        gr, gt = self.gain()
        print("r_average detected   = %.3f ± %.3f" % (ave_r, stderr_r))
        scale = self.r_sphere.detector.a * (1 - self.r_sphere.detector.uru)
        if self.r_sphere.baffle:
            scale *= (1 - self.r_sphere.third.a) * self.r_sphere.r_wall + self.r_sphere.third.a * self.r_sphere.third.uru
        ave_g = ave_r / scale
        std_g = std_r / scale
        stderr_g = std_g / np.sqrt(num_trials)
        print("average gain       = %.3f ± %.3f" % (ave_g, stderr_g))
        print("calculated gain    = %.3f" % gr)
        print()

        print("t_average detected   = %.3f ± %.3f" % (ave_t, stderr_t))
        scale = self.t_sphere.detector.a * (1 - self.t_sphere.detector.uru)
        if self.t_sphere.baffle:
            scale *= (1 - self.t_sphere.third.a) * self.t_sphere.r_wall + self.t_sphere.third.a * self.t_sphere.third.uru
        ave_g = ave_t / scale
        std_g = std_t / scale
        stderr_g = std_g / np.sqrt(num_trials)
        print("average gain       = %.3f ± %.3f" % (ave_g, stderr_g))
        print("calculated gain    = %.3f" % gt)

        return ave_r, stderr_r, ave_t, stderr_t


    def gain(self):
        """
        Wall power gain relative to two black spheres.

        The light on the detector in the both spheres is affected by interactions
        between the two spheres.  This function calculates the gain in power on
        the detectors due to the exchange of light between spheres as well as the
        usual integrating sphere effects.
        """
        Gr = self.r_sphere.gain(sample_uru=self.uru)
        Gt = self.t_sphere.gain(sample_uru=self.uru)
        a_s = self.r_sphere.sample.a
        alpha = a_s**2 * Gr * Gt * self.utu**2

        P_0 = Gr * self.ur1
        P_1 = Gt * self.ut1 + Gr * self.utu * a_s * P_0
        P_2 = Gr * self.utu * a_s * P_1

        Gr = P_0 + P_2 / (1 - alpha)
        Gt = P_1 / (1 - alpha)
        return Gr, Gt


#     def MR(self, sample_ur1, sample_uru=None):
#         """Determine MR due to multiple bounces in the sphere."""
#         if sample_uru is None:
#             sample_uru = sample_ur1
#
#         # sample in sample port, third (entrance) port is empty
#         gain = self.gain(sample_uru, 0)
#
#         # sample port has known standard, third (entrance) port is empty
#         gain100 = self.gain(self.r_std, 0)
#
#         P = sample_ur1 * gain
#         P100 = self.r_std * gain100
#
#         # this is the definition of MR
#         MR = self.r_std * P / P100
#         return MR
#
#     def MT(self, sample_ut1, sample_uru):
#         """Determine MT due to multiple bounces in the sphere."""
#         # sample in sample port, third port has known standard
#         gain = self.gain(sample_uru, self.r_std)
#
#         # sample port is empty, third port has known standard
#         gain100 = self.gain(0, self.r_std)
#
#         P = sample_ut1 * gain
#         P100 = self.r_std * gain100
#
#         # this is the definition of MT
#         MT = self.r_std * P / P100
#         return MT
#
#     def pdetector(self):
#         """Print the detector power."""
#         P = (1 - self.third.a) * self.r_wall
#         N = 1000
#         pd = np.zeros(N)
#         pw = np.zeros(N)
#         ps = np.zeros(N)
#
#         pd[0] = self.detector.a * P
#         ps[0] = self.sample.a * P
#         pw[0] = self.a_wall * P
#
#         for j in range(N - 1):
#             pd[j + 1] = self.detector.a * self.r_wall * pw[j]
#             ps[j + 1] = self.sample.a * self.r_wall * pw[j]
#             pw[j + 1] = self.a_wall * self.r_wall * pw[j]
#             pw[j + 1] += (1 - self.third.a) * self.detector.uru * pd[j]
#             pw[j + 1] += (1 - self.third.a) * self.sample.uru * ps[j]
#
#         sumw = np.cumsum(pw)
#         sumw -= sumw[1]
#         sumd = np.cumsum(pd)
#         print(' k    P_d^k   P_w^k    sum(P_d^k)   sum(P_w^k)')
#         for j in range(10):
#             print("%3d %9.5f %9.5f %9.5f %9.5f" % (j + 1, pd[j], pw[j], sumd[j], sumw[j]))
#         print("%3d %9.5f %9.5f %9.5f %9.5f" % (N - 1, pd[N - 1], pw[N - 1], sumd[N - 1], sumw[N - 1]))
#         print()
#
#         pd[0] = self.detector.a * P
#         ps[0] = self.sample.a * P
#         pw[0] = self.a_wall * P
#
#         pw[1] = self.a_wall * self.r_wall * pw[0]
#         pw[1] += (1 - self.third.a) * self.detector.uru * pd[0]
#         pw[1] += (1 - self.third.a) * self.sample.uru * ps[0]
#         pd[1] = self.detector.a * self.r_wall * pw[0]
#
#         beta = 1 - self.third.a
#         beta *= self.detector.a * self.detector.uru + self.sample.a * self.sample.uru
#         for j in range(1, N - 1):
#             pw[j + 1] = self.r_wall * (self.a_wall * pw[j] + beta * pw[j - 1])
#
#         sumw = np.cumsum(pw)
#         sumw -= sumw[1]
#         sumd = np.cumsum(pd)
#         print(' k    P_d^k   P_w^k    sum(P_d^k)   sum(P_w^k)')
#         for j in range(10):
#             print("%3d %9.5f %9.5f %9.5f %9.5f" % (j + 1, pd[j], pw[j], sumd[j], sumw[j]))
#         print("%3d %9.5f %9.5f %9.5f %9.5f" % (N - 1, pd[N - 1], pw[N - 1], sumd[N - 1], sumw[N - 1]))
#         print()
#
#         beta = 1 - self.third.a
#         beta *= self.detector.a * self.detector.uru + self.sample.a * self.sample.uru
#         numer = self.a_wall**3 * self.r_wall + beta * (2 * self.a_wall + self.a_wall**2 * self.r_wall + beta)
#         denom = 1 - self.r_wall * (self.a_wall + beta)
#         sum3 = self.r_wall * numer / denom * P
#
#         pdx = self.detector.a * P
#         pdx += self.detector.a * self.r_wall * self.a_wall * P
#         pdx += self.detector.a * self.r_wall * (self.a_wall**2 * self.r_wall + beta) * P
#         pdx += self.detector.a * self.r_wall * sum3
#
#         print("%9.5f" % pdx)
#
#         beta = 1 - self.third.a
#         beta *= self.detector.a * self.detector.uru + self.sample.a * self.sample.uru
#         pdx = self.detector.a * P / (1 - self.r_wall * (self.a_wall + beta))
#         print("%9.5f" % pdx)
