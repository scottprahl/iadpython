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
import iadpython as iad


class DoubleSphere:
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

    REFLECTION_SPHERE = 1
    TRANSMISSION_SPHERE = 2

    def __init__(self, r_sphere, t_sphere):
        """Initialization."""
        self.r_sphere = r_sphere
        self.r_sphere.refl = True
        self.t_sphere = t_sphere
        self.t_sphere.refl = False
        self.current = self.r_sphere
        self.ur1 = 0
        self.ut1 = 1
        self._uru = self.ur1
        self._utu = self.ut1
        self.f_r = 0.0

    def __repr__(self):
        """Return basic details as a string for printing."""
        s = "Double sphere experiment\n"
        s += "    ur1 = %s\n" % iad.stringify("%5.1f%%", self.ur1 * 100)
        s += "    uru = %s\n" % iad.stringify("%5.1f%%", self.uru * 100)
        s += "    ut1 = %s\n" % iad.stringify("%5.1f%%", self.ut1 * 100)
        s += "    utu = %s\n" % iad.stringify("%5.1f%%", self.utu * 100)
        s += "R " + self.r_sphere.__repr__()
        s += "T " + self.t_sphere.__repr__()
        return s

    def __str__(self):
        """Return full details as a string for printing."""
        s = ""
        s += "Reflection Sphere\n"
        s += str(self.r_sphere) + "\n"
        s += "Transmission Sphere\n"
        s += str(self.t_sphere) + "\n"
        s += "Sample Properties\n"
        s += "   ur1 = %7.3f\n" % self.ur1
        s += "   ut1 = %7.3f\n" % self.ut1
        s += "   uru = %7.3f\n" % self.uru
        s += "   utu = %7.3f\n" % self.utu
        s += "Gain Properties\n"
        gr, gt = self.gain()
        s += "   r gain = %7.3f\n" % gr
        s += "   t gain = %7.3f\n" % gt
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

    @property
    def utu(self):
        """Need to update sample utu."""
        return self._utu

    @utu.setter
    def utu(self, value):
        """When size is changed ratios become invalid."""
        self._utu = value

    def do_one_photon(self, return_passes=False):
        """Bounce photon in double spheres until it is detected or lost.

        Args:
            return_passes: When True, include sample pass count in return value.

        Returns:
            (r_detected, t_detected) or (r_detected, t_detected, passes)
        """
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
                else:  # absorbed by sample
                    weight = 0
            else:
                weight = 0
                if self.current == self.r_sphere:
                    assert passes % 2 == 0, "reflection sphere should have even number of passes"
                    r_detected += detected
                else:
                    assert passes % 2 == 1, "reflection sphere should have odd number of passes"
                    t_detected += detected

        if return_passes:
            return r_detected, t_detected, passes
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

        gr, gt = self.gain()
        print("r_average detected   = %.3f ± %.3f" % (ave_r, stderr_r))
        scale = self.r_sphere.detector.a * (1 - self.r_sphere.detector.uru)
        if self.r_sphere.baffle:
            scale *= (
                1 - self.r_sphere.third.a
            ) * self.r_sphere.r_wall + self.r_sphere.third.a * self.r_sphere.third.uru
        ave_g = ave_r / scale
        std_g = std_r / scale
        stderr_g = std_g / np.sqrt(num_trials)
        print("average gain       = %.3f ± %.3f" % (ave_g, stderr_g))
        print("calculated gain    = %.3f" % gr)
        print()

        print("t_average detected   = %.3f ± %.3f" % (ave_t, stderr_t))
        scale = self.t_sphere.detector.a * (1 - self.t_sphere.detector.uru)
        if self.t_sphere.baffle:
            scale *= (
                1 - self.t_sphere.third.a
            ) * self.t_sphere.r_wall + self.t_sphere.third.a * self.t_sphere.third.uru
        ave_g = ave_t / scale
        std_g = std_t / scale
        stderr_g = std_g / np.sqrt(num_trials)
        print("average gain       = %.3f ± %.3f" % (ave_g, stderr_g))
        print("calculated gain    = %.3f" % gt)

        return ave_r, stderr_r, ave_t, stderr_t

    @staticmethod
    def _safe_inverse(x):
        """Return 1/x with infinities for zero denominators."""
        x = np.asarray(x, dtype=float)
        result = np.where(x == 0, np.inf, 1.0 / x)
        if np.ndim(result) == 0:
            return float(result)
        return result

    def _sphere_terms(self, sphere):
        """Return CWEB gain terms for reflection or transmission sphere."""
        if sphere == self.REFLECTION_SPHERE:
            s = self.r_sphere
        elif sphere == self.TRANSMISSION_SPHERE:
            s = self.t_sphere
        else:
            raise ValueError(f"Unknown sphere selector: {sphere}")

        return (
            s.sample.a,
            s.detector.a,
            s.third.a,
            s.a_wall,
            s.detector.uru,
            s.r_wall,
            s.baffle,
        )

    def gain_single(self, sphere, uru_sample, uru_third):
        """
        Single-sphere gain from CWEB `Gain()`.

        This is the exact algebra used in `iad/src/iad_calc.c`.
        """
        as_x, ad_x, at_x, aw_x, rd_x, rw_x, baffle_x = self._sphere_terms(sphere)

        if baffle_x:
            inv_gain = rw_x + (at_x / aw_x) * uru_third
            inv_gain *= aw_x + (1 - at_x) * (ad_x * rd_x + as_x * uru_sample)
            inv_gain = 1.0 - inv_gain
        else:
            inv_gain = 1.0 - aw_x * rw_x - ad_x * rd_x - as_x * uru_sample - at_x * uru_third

        return self._safe_inverse(inv_gain)

    def gain_11(self, uru, tdiffuse):
        """Gain for light starting/ending in the reflection sphere (`Gain_11`)."""
        g = self.gain_single(self.REFLECTION_SPHERE, uru, 0.0)
        gp = self.gain_single(self.TRANSMISSION_SPHERE, uru, 0.0)

        coupling = (
            self.r_sphere.sample.a
            * self.t_sphere.sample.a
            * self.r_sphere.a_wall
            * self.t_sphere.a_wall
            * (1 - self.r_sphere.third.a)
            * (1 - self.t_sphere.third.a)
            * g
            * gp
            * tdiffuse
            * tdiffuse
        )
        return g * self._safe_inverse(1.0 - coupling)

    def gain_22(self, uru, tdiffuse):
        """Gain for light starting/ending in the transmission sphere (`Gain_22`)."""
        g = self.gain_single(self.REFLECTION_SPHERE, uru, 0.0)
        gp = self.gain_single(self.TRANSMISSION_SPHERE, uru, 0.0)

        coupling = (
            self.r_sphere.sample.a
            * self.t_sphere.sample.a
            * self.r_sphere.a_wall
            * self.t_sphere.a_wall
            * (1 - self.r_sphere.third.a)
            * (1 - self.t_sphere.third.a)
            * g
            * gp
            * tdiffuse
            * tdiffuse
        )
        return gp * self._safe_inverse(1.0 - coupling)

    def gain(self, uru=None, utu=None):
        """
        Return `(Gain_11, Gain_22)` for the current sample diffuse properties.

        This preserves the historical `DoubleSphere.gain()` API while using
        the exact CWEB algebra.
        """
        if uru is None:
            uru = self.uru
        if utu is None:
            utu = self.utu
        return self.gain_11(uru, utu), self.gain_22(uru, utu)

    def two_sphere_r(self, ur1, uru, ut1, utu):
        """Exact CWEB `Two_Sphere_R()` power expression."""
        gp = self.gain_single(self.TRANSMISSION_SPHERE, uru, 0.0)
        x = self.r_sphere.detector.a * (1 - self.r_sphere.third.a) * self.r_sphere.r_wall
        x *= self.gain_11(uru, utu)
        x *= (
            (1 - self.f_r) * ur1
            + self.r_sphere.r_wall * self.f_r
            + (1 - self.f_r)
            * self.t_sphere.sample.a
            * (1 - self.t_sphere.third.a)
            * self.t_sphere.r_wall
            * ut1
            * utu
            * gp
        )
        return x

    def two_sphere_t(self, ur1, uru, ut1, utu):
        """Exact CWEB `Two_Sphere_T()` power expression."""
        g = self.gain_single(self.REFLECTION_SPHERE, uru, 0.0)
        x = self.t_sphere.detector.a * (1 - self.t_sphere.third.a) * self.t_sphere.r_wall
        x *= self.gain_22(uru, utu)
        x *= (1 - self.f_r) * ut1 + (
            1 - self.r_sphere.third.a
        ) * self.r_sphere.r_wall * self.r_sphere.sample.a * utu * (
            self.f_r * self.r_sphere.r_wall + (1 - self.f_r) * ur1
        ) * g
        return x

    def measured_rt(self, ur1, uru, ut1, utu):
        """
        Exact CWEB two-sphere normalization for measured `M_R` and `M_T`.

        Inputs must already include any unscattered/lost-light corrections.
        """
        r_0 = self.two_sphere_r(0.0, 0.0, 0.0, 0.0)
        t_0 = self.two_sphere_t(0.0, 0.0, 0.0, 0.0)

        m_r = self.r_sphere.r_std * (self.two_sphere_r(ur1, uru, ut1, utu) - r_0)
        m_r *= self._safe_inverse(self.two_sphere_r(self.r_sphere.r_std, self.r_sphere.r_std, 0.0, 0.0) - r_0)

        m_t = self.two_sphere_t(ur1, uru, ut1, utu) - t_0
        m_t *= self._safe_inverse(self.two_sphere_t(0.0, 0.0, 1.0, 1.0) - t_0)

        return m_r, m_t

    def MR(self, ur1, uru, ut1, utu):
        """Convenience wrapper for the two-sphere measured reflection."""
        m_r, _ = self.measured_rt(ur1, uru, ut1, utu)
        return m_r

    def MT(self, ur1, uru, ut1, utu):
        """Convenience wrapper for the two-sphere measured transmission."""
        _, m_t = self.measured_rt(ur1, uru, ut1, utu)
        return m_t
