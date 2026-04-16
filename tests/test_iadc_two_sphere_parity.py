"""Two-sphere parity tests between iadpython and the C implementation."""

import pathlib
import unittest

import iadpython


class TwoSphereCParityTest(unittest.TestCase):
    """Use the repo-local libiad build as the CWEB/C parity reference."""

    DIRECT_TOLERANCE = 1e-10
    RT_TOLERANCE = 6e-2

    SPHERE_CONFIG = {
        "d_sphere": 150.0,
        "d_sample": 10.0,
        "d_third": 10.0,
        "d_detector": 5.0,
        "r_detector": 0.04,
        "r_wall": 0.98,
        "r_std": 0.99,
    }

    FORWARD_CASES = (
        {
            "name": "matched_unbaffled",
            "optics": {"a": 0.68, "b": 1.70, "g": 0.25},
            "sample": {"n": 1.0, "n_above": 1.0, "n_below": 1.0, "quad_pts": 16},
            "config": {
                "r_baffle": False,
                "t_baffle": False,
                "fraction_of_rc_in_mr": 1.0,
                "fraction_of_tc_in_mt": 1.0,
                "f_r": 0.0,
            },
        },
        {
            "name": "slab_in_air_mixed_collection",
            "optics": {"a": 0.82, "b": 4.10, "g": 0.72},
            "sample": {"n": 1.4, "n_above": 1.0, "n_below": 1.0, "quad_pts": 16},
            "config": {
                "r_baffle": True,
                "t_baffle": False,
                "fraction_of_rc_in_mr": 0.78,
                "fraction_of_tc_in_mt": 0.83,
                "f_r": 0.15,
            },
        },
        {
            "name": "glass_slides_baffled",
            "optics": {"a": 0.66, "b": 2.30, "g": 0.38},
            "sample": {"n": 1.4, "n_above": 1.5, "n_below": 1.5, "quad_pts": 16},
            "config": {
                "r_baffle": True,
                "t_baffle": True,
                "fraction_of_rc_in_mr": 0.64,
                "fraction_of_tc_in_mt": 0.71,
                "f_r": 0.22,
            },
        },
        {
            "name": "matched_baffled_partial_collection",
            "optics": {"a": 0.74, "b": 2.60, "g": 0.55},
            "sample": {"n": 1.0, "n_above": 1.0, "n_below": 1.0, "quad_pts": 16},
            "config": {
                "r_baffle": True,
                "t_baffle": True,
                "fraction_of_rc_in_mr": 0.62,
                "fraction_of_tc_in_mt": 0.47,
                "f_r": 0.10,
            },
        },
    )

    INVERSE_CASES = (
        {
            "name": "matched_unbaffled",
            "optics": {"a": 0.68, "b": 1.70, "g": 0.25},
            "sample": {"n": 1.0, "n_above": 1.0, "n_below": 1.0, "quad_pts": 16},
            "config": {
                "r_baffle": False,
                "t_baffle": False,
                "fraction_of_rc_in_mr": 1.0,
                "fraction_of_tc_in_mt": 1.0,
                "f_r": 0.0,
            },
            "default_g": 0.25,
        },
        {
            "name": "slab_in_air_mixed_collection",
            "optics": {"a": 0.82, "b": 4.10, "g": 0.72},
            "sample": {"n": 1.4, "n_above": 1.0, "n_below": 1.0, "quad_pts": 16},
            "config": {
                "r_baffle": True,
                "t_baffle": False,
                "fraction_of_rc_in_mr": 0.78,
                "fraction_of_tc_in_mt": 0.83,
                "f_r": 0.15,
            },
            "default_g": 0.72,
        },
        {
            "name": "glass_slides_baffled",
            "optics": {"a": 0.66, "b": 2.30, "g": 0.38},
            "sample": {"n": 1.4, "n_above": 1.5, "n_below": 1.5, "quad_pts": 16},
            "config": {
                "r_baffle": True,
                "t_baffle": True,
                "fraction_of_rc_in_mr": 0.64,
                "fraction_of_tc_in_mt": 0.71,
                "f_r": 0.22,
            },
            "default_g": 0.38,
        },
    )

    @classmethod
    def setUpClass(cls):
        """Import iadc and skip if the C extension is not built."""
        import iadpython.iadc as iadc

        cls.iadc = iadc
        repo_lib_dir = pathlib.Path(__file__).resolve().parents[1] / "iad" / "src"
        lib_path = pathlib.Path(str(iadc.libiad_path))

        if not lib_path.exists() or lib_path.resolve().parent != repo_lib_dir.resolve():
            raise unittest.SkipTest(
                "build repo-local libiad in /Users/prahl/Documents/Code/git/iadpython/iad/src "
                "to run two-sphere C parity tests"
            )

    def _make_spheres(self, config):
        r_sphere = iadpython.Sphere(refl=True, **self.SPHERE_CONFIG)
        t_sphere = iadpython.Sphere(refl=False, **self.SPHERE_CONFIG)
        r_sphere.baffle = config["r_baffle"]
        t_sphere.baffle = config["t_baffle"]
        return r_sphere, t_sphere

    def _make_sample(self, optics, sample_kwargs):
        kwargs = dict(sample_kwargs)
        kwargs.update(optics)
        return iadpython.Sample(**kwargs)

    def _make_experiment(self, sample, config, measured=None, default_g=None):
        r_sphere, t_sphere = self._make_spheres(config)
        exp = iadpython.Experiment(
            sample=sample,
            num_spheres=2,
            r_sphere=r_sphere,
            t_sphere=t_sphere,
        )
        exp.fraction_of_rc_in_mr = config["fraction_of_rc_in_mr"]
        exp.fraction_of_tc_in_mt = config["fraction_of_tc_in_mt"]
        exp.f_r = config["f_r"]

        if measured is not None:
            exp.m_r, exp.m_t = measured
            exp.default_g = default_g

        return exp

    def _measurement_inputs(self, sample):
        ur1, ut1, uru, utu = sample.rt()
        nu_inside = iadpython.cos_snell(1.0, sample.nu_0, sample.n)
        ru, tu = iadpython.specular_rt(sample.n_above, sample.n, sample.n_below, sample.b, nu_inside)
        return ur1, ut1, uru, utu, ru, tu

    def test_direct_measured_rt_matches_c_with_shared_rt_inputs(self):
        """Two-sphere normalization should match exactly when RT inputs are shared."""
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
        """Python and C two-sphere inversions should re-forward to the same measurements."""
        for case in self.INVERSE_CASES:
            with self.subTest(case=case["name"]):
                forward_sample = self._make_sample(case["optics"], case["sample"])
                forward_experiment = self._make_experiment(forward_sample, case["config"])
                measured_r, measured_t = forward_experiment.measured_rt()

                inverse_sample = iadpython.Sample(**case["sample"])
                inverse_experiment = self._make_experiment(
                    inverse_sample,
                    case["config"],
                    measured=(float(measured_r), float(measured_t)),
                    default_g=case["default_g"],
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


if __name__ == "__main__":
    unittest.main()
