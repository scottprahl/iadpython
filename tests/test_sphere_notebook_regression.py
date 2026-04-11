"""Regression tests for the validated single-sphere notebook geometry."""

import unittest

import iadpython


class SingleSphereNotebookRegressionTest(unittest.TestCase):
    """Lock in the notebook-derived single-sphere behavior."""

    GAIN_TOLERANCE = 1e-12

    NOTEBOOK_GEOMETRY = {
        "d_sphere": 60.0,
        "d_sample": 20.0,
        "d_third": 15.0,
        "d_detector": 10.0,
        "r_detector": 0.5,
        "r_wall": 0.75,
        "r_std": 0.8,
    }

    GAIN_CASES = (
        (False, 0.00, 0.00, 3.5175572519083973),
        (False, 0.50, 0.00, 3.6982343499197436),
        (False, 0.50, 0.95, 3.91304347826087),
        (True, 0.00, 0.00, 3.506348979883008),
        (True, 0.50, 0.00, 3.637117063785704),
        (True, 0.50, 0.95, 3.8486383285335597),
        (True, 0.95, 0.25, 3.8213538714487862),
    )

    MR_CASES = (
        (False, 0.42, 0.33, 0.09, 1.00, 0.00, 0.37863712374581937),
        (False, 0.58, 0.41, 0.12, 0.55, 0.20, 0.4875552430699848),
        (True, 0.58, 0.44, 0.11, 0.65, 0.18, 0.3489107922617832),
        (True, 0.67, 0.52, 0.07, 1.00, 0.35, 0.3852201354702446),
    )

    MT_CASES = (
        (False, 0.61, 0.27, 0.14, 0.00, 0.46150975349683243),
        (False, 0.61, 0.27, 0.14, 1.00, 0.5985160612231492),
        (True, 0.55, 0.40, 0.10, 1.00, 0.431047859745533),
        (True, 0.48, 0.35, 0.18, 0.00, 0.2172085437972961),
    )

    def _default_sphere(self, refl=True):
        return iadpython.Sphere(refl=refl, **self.NOTEBOOK_GEOMETRY)

    def test_gain_matches_notebook_cases(self):
        """The seven notebook gain cases should stay numerically stable."""
        for baffle, sample_uru, third_uru, expected in self.GAIN_CASES:
            with self.subTest(baffle=baffle, sample_uru=sample_uru, third_uru=third_uru):
                sphere = self._default_sphere(refl=True)
                sphere.baffle = baffle
                sphere.sample.uru = sample_uru
                sphere.third.uru = third_uru

                self.assertAlmostEqual(float(sphere.gain()), expected, delta=self.GAIN_TOLERANCE)

    def test_measured_reflection_formula_regression(self):
        """Representative reflection normalization cases should stay stable."""
        for baffle, ur1, uru, ru, f_u, f_w, expected in self.MR_CASES:
            with self.subTest(baffle=baffle, ur1=ur1, uru=uru, ru=ru, f_u=f_u, f_w=f_w):
                sphere = self._default_sphere(refl=True)
                sphere.baffle = baffle
                measured = sphere.MR(ur1, sample_uru=uru, R_u=ru, f_u=f_u, f_w=f_w)

                self.assertAlmostEqual(float(measured), expected, delta=self.GAIN_TOLERANCE)

    def test_measured_transmission_formula_regression(self):
        """Representative transmission normalization cases should stay stable."""
        for baffle, ut1, uru, tu, f_u, expected in self.MT_CASES:
            with self.subTest(baffle=baffle, ut1=ut1, uru=uru, tu=tu, f_u=f_u):
                sphere = self._default_sphere(refl=False)
                sphere.baffle = baffle
                measured = sphere.MT(ut1, sample_uru=uru, T_u=tu, f_u=f_u)

                self.assertAlmostEqual(float(measured), expected, delta=self.GAIN_TOLERANCE)


if __name__ == "__main__":
    unittest.main()
