# pylint: disable=invalid-name

"""Round-trip tests for forward and inverse R,T without spheres."""

import itertools
import unittest
import warnings

import iadpython


class RTRoundTripNoSphereTest(unittest.TestCase):
    """Validate R,T forward/inverse round trips for slabs without spheres."""

    A_VALUES = (0.0, 0.5, 0.99)
    B_VALUES = (0.1, 1.0, 10.0)
    G_VALUES = (-0.9, 0.0, 0.9)
    RT_TOLERANCE = 5e-2
    U_RT_TOLERANCE = 2.5e-2
    U_TOLERANCE = 2e-3

    UNSTABLE_U_CASES = {
        "matched": set(),
        "slab_in_air": set(),
        "glass_slides_in_air": set(),
    }

    def _max_round_trip_error(self, sample_kwargs):
        max_dr = 0.0
        max_dt = 0.0
        worst_case = None

        for a, b, g in itertools.product(self.A_VALUES, self.B_VALUES, self.G_VALUES):
            forward_sample = iadpython.Sample(a=a, b=b, g=g, **sample_kwargs)
            r_forward, t_forward, _, _ = forward_sample.rt()

            inverse_sample = iadpython.Sample(**sample_kwargs)
            exp = iadpython.Experiment(
                r=float(r_forward),
                t=float(t_forward),
                sample=inverse_sample,
                default_g=g,
            )
            self.assertEqual(exp.num_spheres, 0)

            a_inv, b_inv, g_inv = exp.invert_rt()

            inverse_forward_sample = iadpython.Sample(a=a_inv, b=b_inv, g=g_inv, **sample_kwargs)
            r_inverse, t_inverse, _, _ = inverse_forward_sample.rt()

            dr = abs(float(r_forward) - float(r_inverse))
            dt = abs(float(t_forward) - float(t_inverse))

            if dr > max_dr or dt > max_dt:
                worst_case = (a, b, g, float(a_inv), float(b_inv), float(g_inv), dr, dt)

            max_dr = max(max_dr, dr)
            max_dt = max(max_dt, dt)

        return max_dr, max_dt, worst_case

    def _assert_boundary_round_trip(self, sample_kwargs):
        max_dr, max_dt, worst_case = self._max_round_trip_error(sample_kwargs)
        msg = f"worst_case={worst_case}"
        self.assertLessEqual(max_dr, self.RT_TOLERANCE, msg=msg)
        self.assertLessEqual(max_dt, self.RT_TOLERANCE, msg=msg)

    def _max_round_trip_error_with_unscattered(
        self,
        boundary_name,
        sample_kwargs,
    ):
        max_dr = 0.0
        max_dt = 0.0
        max_du = 0.0
        solved_cases = 0
        worst_case = None
        unstable_cases = self.UNSTABLE_U_CASES[boundary_name]

        for a, b, g in itertools.product(self.A_VALUES, self.B_VALUES, self.G_VALUES):
            forward_sample = iadpython.Sample(a=a, b=b, g=g, **sample_kwargs)
            r_forward, t_forward, _, _ = forward_sample.rt()
            _, u_forward = forward_sample.unscattered_rt()

            inverse_sample = iadpython.Sample(**sample_kwargs)
            exp = iadpython.Experiment(
                r=float(r_forward),
                t=float(t_forward),
                u=float(u_forward),
                sample=inverse_sample,
            )
            self.assertIsNone(exp.default_g)
            self.assertEqual(exp.num_spheres, 0)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    a_inv, b_inv, g_inv = exp.invert_rt()
            except ValueError as exc:
                msg = f"unexpected inversion failure for {(a, b, g)}: {exc}"
                self.assertIn((a, b, g), unstable_cases, msg=msg)
                continue

            solved_cases += 1
            inverse_forward_sample = iadpython.Sample(a=a_inv, b=b_inv, g=g_inv, **sample_kwargs)
            r_inverse, t_inverse, _, _ = inverse_forward_sample.rt()
            _, u_inverse = inverse_forward_sample.unscattered_rt()

            dr = abs(float(r_forward) - float(r_inverse))
            dt = abs(float(t_forward) - float(t_inverse))
            du = abs(float(u_forward) - float(u_inverse))

            if dr > max_dr or dt > max_dt or du > max_du:
                worst_case = (a, b, g, float(a_inv), float(b_inv), float(g_inv), dr, dt, du)

            max_dr = max(max_dr, dr)
            max_dt = max(max_dt, dt)
            max_du = max(max_du, du)

        self.assertGreater(solved_cases, 0)
        return max_dr, max_dt, max_du, worst_case

    def _assert_boundary_round_trip_with_unscattered(self, boundary_name, sample_kwargs):
        max_dr, max_dt, max_du, worst_case = self._max_round_trip_error_with_unscattered(
            boundary_name,
            sample_kwargs,
        )
        msg = f"worst_case={worst_case}"
        self.assertLessEqual(max_dr, self.U_RT_TOLERANCE, msg=msg)
        self.assertLessEqual(max_dt, self.U_RT_TOLERANCE, msg=msg)
        self.assertLessEqual(max_du, self.U_TOLERANCE, msg=msg)

    def test_round_trip_matched_boundaries(self):
        """Matched boundaries: 1/1/1."""
        self._assert_boundary_round_trip(dict(n=1.0, n_above=1.0, n_below=1.0, quad_pts=16))

    def test_round_trip_slab_in_air(self):
        """Slab in air: 1/1.4/1."""
        self._assert_boundary_round_trip(dict(n=1.4, n_above=1.0, n_below=1.0, quad_pts=16))

    def test_round_trip_between_glass_slides_in_air(self):
        """Between slides in air: 1/1.5/1.4/1.5/1."""
        self._assert_boundary_round_trip(dict(n=1.4, n_above=1.5, n_below=1.5, quad_pts=16))

    def test_round_trip_with_unscattered_matched_boundaries(self):
        """Matched boundaries using R,T and unscattered transmission."""
        self._assert_boundary_round_trip_with_unscattered(
            "matched",
            dict(n=1.0, n_above=1.0, n_below=1.0, quad_pts=16),
        )

    def test_round_trip_with_unscattered_slab_in_air(self):
        """Slab in air using R,T and unscattered transmission."""
        self._assert_boundary_round_trip_with_unscattered(
            "slab_in_air",
            dict(n=1.4, n_above=1.0, n_below=1.0, quad_pts=16),
        )

    def test_round_trip_with_unscattered_between_glass_slides_in_air(self):
        """Between slides in air using R,T and unscattered transmission."""
        self._assert_boundary_round_trip_with_unscattered(
            "glass_slides_in_air",
            dict(n=1.4, n_above=1.5, n_below=1.5, quad_pts=16),
        )


if __name__ == "__main__":
    unittest.main()
