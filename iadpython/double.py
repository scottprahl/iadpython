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
        s += str(self.r_sphere) + "\n"
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
            detected, transmitted, _ = self.current.do_one_photon(
                weight=weight, double=True
            )

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
                    assert (
                        passes % 2 == 0
                    ), "reflection sphere should have even number of passes"
                    r_detected += detected
                else:
                    assert (
                        passes % 2 == 1
                    ), "reflection sphere should have odd number of passes"
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
            scale *= (
                (1 - self.r_sphere.third.a) * self.r_sphere.r_wall
                + self.r_sphere.third.a * self.r_sphere.third.uru
            )
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
                (1 - self.t_sphere.third.a) * self.t_sphere.r_wall
                + self.t_sphere.third.a * self.t_sphere.third.uru
            )
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

    def MT(self, sample_ut1, sample_uru, Tu=0, f_u=1):
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
            Tu (optional): The unscattered transmission of the sample.
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

        ut1_calc = sample_ut1 - Tu

        gain = self.gain(sample_uru, r_third)
        gain_cal = self.gain(0, r_cal)

        P_ss = r_first * ut1_calc
        P_su = r_third * Tu * f_u
        P = (P_su + P_ss) * gain
        P_cal = r_cal * gain_cal

        MT = r_cal * P / P_cal

        #         print("UT1     =  %6.3f   URU   = %6.3f" % (sample_ut1, sample_uru))
        #         print("Tu      =  %6.3f   f_uns = %5.2f" % (Tu,f_u))
        #         print("P_ss    =  %6.3f   P_su  = %6.3f" % (P_ss, P_su))
        #         print("G       =  %6.3f   P     = %6.3f" % (gain, P))
        #         print("G_cal   =  %6.3f   P_cal = %6.3f" % (gain_cal, P_cal))
        #         print("r_first =  %6.3f" % (r_first))
        #         print("r_cal   =  %6.3f" % (r_cal))
        #         print("r_third =  %6.3f" % (r_third))
        #         print("MT      =  %6.3f" % (MT))

        return MT
