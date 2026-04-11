"""Single-sphere parity tests between iadpython and the C implementation."""

import pathlib
import unittest

import iadpython


class SingleSphereCParityTest(unittest.TestCase):
    """Use the repo-local libiad build as the CWEB/C parity reference."""

    DIRECT_TOLERANCE = 1e-10
    RT_TOLERANCE = 5e-2
    U_TOLERANCE = 2e-3

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
        (False, 0.00, 0.00),
        (False, 0.50, 0.00),
        (False, 0.50, 0.95),
        (True, 0.00, 0.00),
        (True, 0.50, 0.00),
        (True, 0.50, 0.95),
        (True, 0.95, 0.25),
    )

    FORWARD_CASES = (
        {
            "name": "matched_substitution_unbaffled",
            "optics": {"a": 0.73, "b": 1.60, "g": 0.25},
            "sample": {"n": 1.0, "n_above": 1.0, "n_below": 1.0, "quad_pts": 16},
            "config": {
                "method": "substitution",
                "r_baffle": False,
                "t_baffle": False,
                "fraction_of_rc_in_mr": 1.0,
                "fraction_of_tc_in_mt": 1.0,
                "f_r": 0.0,
            },
        },
        {
            "name": "slab_in_air_substitution_mixed_collection",
            "optics": {"a": 0.82, "b": 3.40, "g": 0.72},
            "sample": {"n": 1.4, "n_above": 1.0, "n_below": 1.0, "quad_pts": 16},
            "config": {
                "method": "substitution",
                "r_baffle": True,
                "t_baffle": False,
                "fraction_of_rc_in_mr": 0.78,
                "fraction_of_tc_in_mt": 1.0,
                "f_r": 0.15,
            },
        },
        {
            "name": "glass_slides_substitution_baffled",
            "optics": {"a": 0.66, "b": 2.10, "g": 0.38},
            "sample": {"n": 1.4, "n_above": 1.5, "n_below": 1.5, "quad_pts": 16},
            "config": {
                "method": "substitution",
                "r_baffle": True,
                "t_baffle": True,
                "fraction_of_rc_in_mr": 0.64,
                "fraction_of_tc_in_mt": 0.0,
                "f_r": 0.22,
            },
        },
        {
            "name": "comparison_mode_with_partial_collection",
            "optics": {"a": 0.77, "b": 1.10, "g": 0.10},
            "sample": {"n": 1.4, "n_above": 1.0, "n_below": 1.0, "quad_pts": 16},
            "config": {
                "method": "comparison",
                "r_baffle": True,
                "t_baffle": True,
                "fraction_of_rc_in_mr": 0.83,
                "fraction_of_tc_in_mt": 0.40,
                "f_r": 0.05,
            },
        },
    )

    INVERSE_CASES = (
        {
            "name": "matched_boundaries",
            "optics": {"a": 0.68, "b": 1.70, "g": 0.25},
            "sample": {"n": 1.0, "n_above": 1.0, "n_below": 1.0, "quad_pts": 16},
            "config": {
                "method": "substitution",
                "r_baffle": False,
                "t_baffle": False,
                "fraction_of_rc_in_mr": 1.0,
                "fraction_of_tc_in_mt": 1.0,
                "f_r": 0.0,
            },
            "default_g": 0.25,
            "use_u": False,
        },
        {
            "name": "slab_in_air",
            "optics": {"a": 0.82, "b": 4.10, "g": 0.72},
            "sample": {"n": 1.4, "n_above": 1.0, "n_below": 1.0, "quad_pts": 16},
            "config": {
                "method": "substitution",
                "r_baffle": True,
                "t_baffle": False,
                "fraction_of_rc_in_mr": 0.78,
                "fraction_of_tc_in_mt": 1.0,
                "f_r": 0.15,
            },
            "default_g": 0.72,
            "use_u": False,
        },
        {
            "name": "glass_slides_with_unscattered",
            "optics": {"a": 0.66, "b": 2.30, "g": 0.38},
            "sample": {"n": 1.4, "n_above": 1.5, "n_below": 1.5, "quad_pts": 16},
            "config": {
                "method": "substitution",
                "r_baffle": True,
                "t_baffle": True,
                "fraction_of_rc_in_mr": 0.64,
                "fraction_of_tc_in_mt": 1.0,
                "f_r": 0.22,
            },
            "default_g": None,
            "use_u": True,
        },
    )

    @classmethod
    def setUpClass(cls):
        import iadpython.iadc as iadc

        cls.iadc = iadc
        repo_lib_dir = pathlib.Path(__file__).resolve().parents[1] / "iad" / "src"
        lib_path = pathlib.Path(str(iadc.libiad_path))

        if not lib_path.exists() or lib_path.resolve().parent != repo_lib_dir.resolve():
            raise unittest.SkipTest(
                "build repo-local libiad in /Users/prahl/Documents/Code/git/iadpython/iad/src "
                "to run single-sphere C parity tests"
            )

    def _make_spheres(self, r_baffle=False, t_baffle=False):
        r_sphere = iadpython.Sphere(refl=True, **self.NOTEBOOK_GEOMETRY)
        t_sphere = iadpython.Sphere(refl=False, **self.NOTEBOOK_GEOMETRY)
        r_sphere.baffle = r_baffle
        t_sphere.baffle = t_baffle
        return r_sphere, t_sphere

    def _make_sample(self, optics, sample_kwargs):
        kwargs = dict(sample_kwargs)
        kwargs.update(optics)
        return iadpython.Sample(**kwargs)

    def _make_experiment(self, sample, config, measured=None, default_g=None, use_u=False):
        r_sphere, t_sphere = self._make_spheres(
            r_baffle=config["r_baffle"],
            t_baffle=config["t_baffle"],
        )
        exp = iadpython.Experiment(
            sample=sample,
            num_spheres=1,
            r_sphere=r_sphere,
            t_sphere=t_sphere,
        )
        exp.method = config["method"]
        exp.fraction_of_rc_in_mr = config["fraction_of_rc_in_mr"]
        exp.fraction_of_tc_in_mt = config["fraction_of_tc_in_mt"]
        exp.f_r = config["f_r"]

        if measured is not None:
            exp.m_r = measured[0]
            exp.m_t = measured[1]
            exp.m_u = measured[2] if use_u else None
            exp.default_g = default_g

        return exp

    def _measurement_inputs(self, sample):
        ur1, ut1, uru, utu = sample.rt()
        nu_inside = iadpython.cos_snell(1.0, sample.nu_0, sample.n)
        ru, tu = iadpython.specular_rt(sample.n_above, sample.n, sample.n_below, sample.b, nu_inside)
        return ur1, ut1, uru, utu, ru, tu

    def test_gain_matches_c_for_notebook_cases(self):
        """Reflection-sphere gain should match libiad for the notebook cases."""
        for baffle, sample_uru, third_uru in self.GAIN_CASES:
            with self.subTest(baffle=baffle, sample_uru=sample_uru, third_uru=third_uru):
                sphere = iadpython.Sphere(refl=True, **self.NOTEBOOK_GEOMETRY)
                sphere.baffle = baffle
                sphere.sample.uru = sample_uru
                sphere.third.uru = third_uru

                python_gain = float(sphere.gain())
                c_gain = float(self.iadc._c_gain(sphere))
                self.assertAlmostEqual(python_gain, c_gain, delta=self.DIRECT_TOLERANCE)

    def test_direct_measured_rt_matches_c_with_shared_rt_inputs(self):
        """One-sphere normalization should match exactly when RT inputs are shared."""
        for case in self.FORWARD_CASES:
            with self.subTest(case=case["name"]):
                sample = self._make_sample(case["optics"], case["sample"])
                experiment = self._make_experiment(sample, case["config"])
                ur1, ut1, uru, utu, ru, tu = self._measurement_inputs(sample)

                python_mr, python_mt = experiment.measured_rt()
                c_mr, c_mt = self.iadc._c_calculate_measured_rt_from_rt(
                    experiment,
                    ur1,
                    ut1,
                    uru,
                    utu,
                    ru,
                    tu,
                )

                self.assertAlmostEqual(float(python_mr), float(c_mr), delta=self.DIRECT_TOLERANCE)
                self.assertAlmostEqual(float(python_mt), float(c_mt), delta=self.DIRECT_TOLERANCE)

    def test_inverse_matches_in_measurement_space(self):
        """Python and C inversions should both re-forward to the same measurements."""
        for case in self.INVERSE_CASES:
            with self.subTest(case=case["name"]):
                forward_sample = self._make_sample(case["optics"], case["sample"])
                forward_experiment = self._make_experiment(forward_sample, case["config"])
                measured_r, measured_t = forward_experiment.measured_rt()
                measured_u = None
                if case["use_u"]:
                    _, measured_u = forward_sample.unscattered_rt()

                inverse_sample = iadpython.Sample(**case["sample"])
                inverse_experiment = self._make_experiment(
                    inverse_sample,
                    case["config"],
                    measured=(float(measured_r), float(measured_t), None if measured_u is None else float(measured_u)),
                    default_g=case["default_g"],
                    use_u=case["use_u"],
                )

                py_a, py_b, py_g = inverse_experiment.invert_rt()
                c_a, c_b, c_g, c_error = self.iadc._c_invert_experiment(inverse_experiment)
                self.assertEqual(c_error, 0)

                py_sample = self._make_sample({"a": py_a, "b": py_b, "g": py_g}, case["sample"])
                c_sample = self._make_sample({"a": c_a, "b": c_b, "g": c_g}, case["sample"])

                py_forward = self._make_experiment(py_sample, case["config"]).measured_rt()
                c_forward = self._make_experiment(c_sample, case["config"]).measured_rt()

                self.assertLessEqual(abs(float(py_forward[0]) - float(measured_r)), self.RT_TOLERANCE)
                self.assertLessEqual(abs(float(py_forward[1]) - float(measured_t)), self.RT_TOLERANCE)
                self.assertLessEqual(abs(float(c_forward[0]) - float(measured_r)), self.RT_TOLERANCE)
                self.assertLessEqual(abs(float(c_forward[1]) - float(measured_t)), self.RT_TOLERANCE)

                if case["use_u"]:
                    _, py_u = py_sample.unscattered_rt()
                    _, c_u = c_sample.unscattered_rt()
                    self.assertLessEqual(abs(float(py_u) - float(measured_u)), self.U_TOLERANCE)
                    self.assertLessEqual(abs(float(c_u) - float(measured_u)), self.U_TOLERANCE)


if __name__ == "__main__":
    unittest.main()
