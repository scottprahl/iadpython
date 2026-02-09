"""Tests for NIST reflectance data."""

import unittest
import numpy as np
import iadpython


class TestNIST(unittest.TestCase):
    """NIST Reflectance data."""

    def test_01_subj_refl(self):
        """Verify that subject 1 is read correctly."""
        subject_number = 1
        λ, r1, r2, r3, rave = iadpython.nist.subject_reflectances(subject_number)
        np.testing.assert_allclose([λ[0]], [250.0], atol=1e-5)
        np.testing.assert_allclose([λ[-1]], [2500.0], atol=1e-5)
        np.testing.assert_allclose([r1[0]], [0.0611], atol=1e-5)
        np.testing.assert_allclose([r2[0]], [0.0563], atol=1e-5)
        np.testing.assert_allclose([r3[0]], [0.0546], atol=1e-5)
        np.testing.assert_allclose([rave[0]], [0.0573], atol=1e-5)
        np.testing.assert_allclose([r1[-1]], [0.0362], atol=1e-5)
        np.testing.assert_allclose([r2[-1]], [0.0311], atol=1e-5)
        np.testing.assert_allclose([r3[-1]], [0.0303], atol=1e-5)
        np.testing.assert_allclose([rave[-1]], [0.0325], atol=1e-5)

    def test_02_subj_refl(self):
        """Verify that subject 100 is read correctly."""
        subject_number = 100
        λ, r1, r2, r3, rave = iadpython.nist.subject_reflectances(subject_number)
        np.testing.assert_allclose([λ[0]], [250.0], atol=1e-5)
        np.testing.assert_allclose([λ[-1]], [2500.0], atol=1e-5)
        np.testing.assert_allclose([r1[0]], [0.0565], atol=1e-5)
        np.testing.assert_allclose([r2[0]], [0.0543], atol=1e-5)
        np.testing.assert_allclose([r3[0]], [0.0533], atol=1e-5)
        np.testing.assert_allclose([rave[0]], [0.0547], atol=1e-5)
        np.testing.assert_allclose([r1[-1]], [0.0404], atol=1e-5)
        np.testing.assert_allclose([r2[-1]], [0.0433], atol=1e-5)
        np.testing.assert_allclose([r3[-1]], [0.0405], atol=1e-5)
        np.testing.assert_allclose([rave[-1]], [0.0414], atol=1e-5)

    def test_03_subj_refl(self):
        """Verify that subject 5 is read correctly."""
        subject_number = 5
        λ, r1, r2, r3, rave = iadpython.nist.subject_reflectances(subject_number)
        np.testing.assert_allclose([λ[0]], [250.0], atol=1e-5)
        np.testing.assert_allclose([λ[-1]], [2500.0], atol=1e-5)
        np.testing.assert_allclose([r1[0]], [0.0580], atol=1e-5)
        np.testing.assert_allclose([r2[0]], [0.0565], atol=1e-5)
        np.testing.assert_allclose([r3[0]], [0.0573], atol=1e-5)
        np.testing.assert_allclose([rave[0]], [0.0573], atol=1e-5)
        np.testing.assert_allclose([r1[-1]], [0.0460], atol=1e-5)
        np.testing.assert_allclose([r2[-1]], [0.0440], atol=1e-5)
        np.testing.assert_allclose([r3[-1]], [0.0393], atol=1e-5)
        np.testing.assert_allclose([rave[-1]], [0.0431], atol=1e-5)

    def test_01_subj_ave_refl(self):
        """Verify that average reflectance of subject 1 read correctly."""
        subject_number = 1
        λ, rave = iadpython.nist.subject_average_reflectance(subject_number)
        np.testing.assert_allclose([λ[0]], [250.0], atol=1e-5)
        np.testing.assert_allclose([λ[-1]], [2500.0], atol=1e-5)
        np.testing.assert_allclose([rave[0]], [0.0573], atol=1e-5)
        np.testing.assert_allclose([rave[-1]], [0.0325], atol=1e-5)

    def test_02_subj_ave_refl(self):
        """Verify that average reflectance of subject 100 read correctly."""
        subject_number = 100
        λ, rave = iadpython.nist.subject_average_reflectance(subject_number)
        np.testing.assert_allclose([λ[0]], [250.0], atol=1e-5)
        np.testing.assert_allclose([λ[-1]], [2500.0], atol=1e-5)
        np.testing.assert_allclose([rave[0]], [0.0547], atol=1e-5)
        np.testing.assert_allclose([rave[-1]], [0.0414], atol=1e-5)

    def test_03_subj_ave_refl(self):
        """Verify that average reflectance of subject 100 read correctly."""
        subject_number = 5
        λ, rave = iadpython.nist.subject_average_reflectance(subject_number)
        np.testing.assert_allclose([λ[0]], [250.0], atol=1e-5)
        np.testing.assert_allclose([λ[-1]], [2500.0], atol=1e-5)
        np.testing.assert_allclose([rave[0]], [0.0573], atol=1e-5)
        np.testing.assert_allclose([rave[-1]], [0.0431], atol=1e-5)

    def test_01_all_ave_refl(self):
        """Verify that all average spectra read correctly."""
        λ, R = iadpython.nist.all_average_reflectances()
        np.testing.assert_allclose([λ[0]], [250.0], atol=1e-5)
        np.testing.assert_allclose([λ[-1]], [2500.0], atol=1e-5)

    def test_02_all_ave_refl(self):
        """Verify entries for all average spectra read correctly."""
        _λ, R = iadpython.nist.all_average_reflectances()
        # subj 1 reflectances
        np.testing.assert_allclose([R[0, 0]], [0.0573], atol=1e-5)
        np.testing.assert_allclose([R[-1, 0]], [0.0325], atol=1e-5)

        # subj 100 reflectances
        np.testing.assert_allclose([R[0, 99]], [0.0547], atol=1e-5)
        np.testing.assert_allclose([R[-1, 99]], [0.0414], atol=1e-5)

        # subj 5 reflectances
        np.testing.assert_allclose([R[0, 4]], [0.0573], atol=1e-5)
        np.testing.assert_allclose([R[-1, 4]], [0.0431], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
