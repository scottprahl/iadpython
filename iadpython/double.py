# pylint: disable=too-many-statements
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
        self.fraction_absorbed = 0
        if self._utu + self.absorbed > 0:
            fraction_absorbed = self.absorbed / (self.utu + self.absorbed)

    @property
    def utu(self):
        """Need to update sample utu."""
        return self._utu

    @utu.setter
    def utu(self, value):
        """When size is changed ratios become invalid."""
        self._utu = value
        self.absorbed = 1 - self._uru - self._utu
        denom = self._utu + self.absorbed
        self.fraction_absorbed = 0
        if self._utu + self.absorbed > 0:
            fraction_absorbed = self.absorbed / (self.utu + self.absorbed)

    def do_one_photon(self):
        """Bounce photon in double spheres until it is detected or lost."""
        r_detected = 0
        t_detected = 0

        count=0

        # assume photon directed normally onto sample
        x = random.random()
        if x < self.ur1:
            self.current = self.r_sphere
        elif x < self.ur1 + self.ut1:
            self.current = self.t_sphere
        else:  # absorbed
            return 0, 0

        weight = 1
        while weight > 0:
            count+=1
            detected, weight, _ = self.current.do_one_photon(weight=weight, double=True)
#             if self.current == self.r_sphere:
#                 print("x refl count=", count, "detected=", detected, "weight=", weight)
#             else:
#                 print("x tran count=", count, "detected=", detected, "weight=", weight)

            weight *= (1- self.fraction_absorbed) # the rest is absorbed

            if self.current == self.r_sphere:
                r_detected += detected
                if weight > 0:
                    self.current = self.t_sphere
#                    print("y tran count=", count, "detected=", detected, "weight=", weight)
            else:
                t_detected += detected
                if weight > 0:
                    self.current = self.r_sphere
#                    print("y refl count=", count, "detected=", detected, "weight=", weight)

        return r_detected, t_detected

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
                r_detected, t_detected = self.do_one_photon()
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

#        print("r_average detected   = %.3f ± %.3f" % (ave_r, stderr_r))
#        print("t_average detected   = %.3f ± %.3f" % (ave_t, stderr_t))
        return ave_r, stderr_r, ave_t, stderr_t


    def Gain_11(self):
        r"""Net gain for on detector in reflection sphere for two sphere configuration.

        The light on the detector in the reflectance sphere is affected by interactions
        between the two spheres.  This function calculates the net gain on a detector
        in the reflection sphere for diffuse light starting in the reflectance sphere.

        .. math:: G_{11} = \frac{(P_1/A_d)}{(P/A)}

        then the full expression for the gain is

        .. math:: \frac{G(r_s)}{(1-a_s a_s' r_w r_w' (1-a_e)(1-a_e') G(r_s) G'(r_s)t_s^2)}

        """
        RS = self.r_sphere
        TS = self.t_sphere
        URU = self.uru
        tdiffuse = self.utu

        G = RS.gain(URU)
        GP = TS.gain(URU)

        areas = RS.a_sample * TS.a_sample * (1 - RS.a_third) * (1 - TS.a_third)
        G11 = G / (1 - areas * RS.r_wall * TS.r_wall * G * GP * tdiffuse**2)
        return G11


    def Gain_22(self):
        r"""Two sphere gain in T sphere for light starting in T sphere.

        Similarly, when the light starts in the second sphere, the gain for light
        on the detector in the second sphere :math:`G_{22}` is found by switching
        all primed variables to unprimed.  Thus :math:`G_{21}(r_s,t_s)` is

        .. math:: G_{22}(r_s,t_s) =\frac{G'(r_s)}{1-a_s a_s'r_w r_w'(1-a_e)(1-a_e')G(r_s)G'(r_s)t_s^2}

        """
        RS = self.r_sphere
        TS = self.t_sphere
        URU = self.uru
        tdiffuse = self.utu

        G = RS.gain(URU)
        GP = TS.gain(URU)

        areas = RS.a_sample * TS.a_sample * (1 - RS.a_third) * (1 - TS.a_third)
        G22 = GP / (1 - areas * RS.r_wall * TS.r_wall * G * GP * tdiffuse**2)
        return G22


    def Two_Sphere_R(self, f=0):
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
        RS = self.r_sphere
        TS = self.t_sphere
        UR1 = self.ur1
        UT1 = self.ut1
        URU = self.uru
        UTU = self.utu
        GP = TS.gain(URU)
        G11 = self.Gain_11()

        x = RS.a_detector * (1 - RS.a_third) * RS.r_wall * G11
        p1 = (1 - f) * UR1
        p2 = RS.r_wall * f
        p3 = (1 - f) * TS.a_sample * (1 - TS.a_third) * TS.r_wall * UT1 * UTU * GP
        return x * (p1 + p2 + p3)


    def Two_Sphere_T(self, f=0):
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
        RS = self.r_sphere
        TS = self.t_sphere
        UR1 = self.ur1
        UT1 = self.ut1
        URU = self.uru
        UTU = self.utu
        G = RS.gain(URU)
        G22 = self.Gain_11()

        x = TS.a_detector * (1 - TS.a_third) * TS.r_wall * G22
        x *= (1 - f) * UT1 + (1 - RS.a_third) * RS.r_wall * \
            RS.a_sample * UTU * (f * RS.r_wall + (1 - f) * UR1) * G
        return x


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
