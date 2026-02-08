# pylint: disable=invalid-name

"""Round-trip tests for forward and inverse R,T with one integrating sphere."""

import itertools
import unittest

import iadpython


class RTRoundTripOneSphereTest(unittest.TestCase):
    """Validate R,T forward/inverse round trips with one-sphere corrections."""

    A_VALUES = (0.0, 0.5, 0.99)
    B_VALUES = (0.1, 1.0, 10.0)
    G_VALUES = (-0.9, 0.0, 0.9)
    RT_TOLERANCE = 5e-2
    U_RT_TOLERANCE = 5e-2
    U_TOLERANCE = 2e-3

    SPHERE_CONFIG = dict(
        d_sphere=150.0,
        d_sample=10.0,
        d_third=10.0,
        d_detector=5.0,
        r_detector=0.04,
        r_wall=0.98,
        r_std=0.99,
    )

    def _make_spheres(self):
        r_sphere = iadpython.Sphere(**self.SPHERE_CONFIG, refl=True)
        t_sphere = iadpython.Sphere(**self.SPHERE_CONFIG, refl=False)
        return r_sphere, t_sphere

    def _measured_rt(self, sample):
        r_sphere, t_sphere = self._make_spheres()
        exp = iadpython.Experiment(
            sample=sample,
            num_spheres=1,
            r_sphere=r_sphere,
            t_sphere=t_sphere,
        )
        self.assertEqual(exp.num_spheres, 1)
        m_r, m_t = exp.measured_rt()
        return float(m_r), float(m_t)

    def _max_round_trip_error(self, sample_kwargs):
        max_dr = 0.0
        max_dt = 0.0
        worst_case = None

        for a, b, g in itertools.product(self.A_VALUES, self.B_VALUES, self.G_VALUES):
            forward_sample = iadpython.Sample(a=a, b=b, g=g, **sample_kwargs)
            m_r_forward, m_t_forward = self._measured_rt(forward_sample)

            r_sphere, t_sphere = self._make_spheres()
            inv_exp = iadpython.Experiment(
                r=m_r_forward,
                t=m_t_forward,
                sample=iadpython.Sample(**sample_kwargs),
                default_g=g,
                num_spheres=1,
                r_sphere=r_sphere,
                t_sphere=t_sphere,
            )
            a_inv, b_inv, g_inv = inv_exp.invert_rt()

            inverse_sample = iadpython.Sample(a=a_inv, b=b_inv, g=g_inv, **sample_kwargs)
            m_r_inverse, m_t_inverse = self._measured_rt(inverse_sample)

            dr = abs(m_r_forward - m_r_inverse)
            dt = abs(m_t_forward - m_t_inverse)

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

    def _max_round_trip_error_with_unscattered(self, sample_kwargs):
        max_dr = 0.0
        max_dt = 0.0
        max_du = 0.0
        worst_case = None

        for a, b, g in itertools.product(self.A_VALUES, self.B_VALUES, self.G_VALUES):
            forward_sample = iadpython.Sample(a=a, b=b, g=g, **sample_kwargs)
            m_r_forward, m_t_forward = self._measured_rt(forward_sample)
            _, u_forward = forward_sample.unscattered_rt()

            r_sphere, t_sphere = self._make_spheres()
            inv_exp = iadpython.Experiment(
                r=m_r_forward,
                t=m_t_forward,
                u=float(u_forward),
                sample=iadpython.Sample(**sample_kwargs),
                num_spheres=1,
                r_sphere=r_sphere,
                t_sphere=t_sphere,
            )
            self.assertIsNone(inv_exp.default_g)
            a_inv, b_inv, g_inv = inv_exp.invert_rt()

            inverse_sample = iadpython.Sample(a=a_inv, b=b_inv, g=g_inv, **sample_kwargs)
            m_r_inverse, m_t_inverse = self._measured_rt(inverse_sample)
            _, u_inverse = inverse_sample.unscattered_rt()

            dr = abs(m_r_forward - m_r_inverse)
            dt = abs(m_t_forward - m_t_inverse)
            du = abs(float(u_forward) - float(u_inverse))

            if dr > max_dr or dt > max_dt or du > max_du:
                worst_case = (
                    a,
                    b,
                    g,
                    float(a_inv),
                    float(b_inv),
                    float(g_inv),
                    dr,
                    dt,
                    du,
                )

            max_dr = max(max_dr, dr)
            max_dt = max(max_dt, dt)
            max_du = max(max_du, du)

        return max_dr, max_dt, max_du, worst_case

    def _assert_boundary_round_trip_with_unscattered(self, sample_kwargs):
        max_dr, max_dt, max_du, worst_case = self._max_round_trip_error_with_unscattered(sample_kwargs)
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
            dict(n=1.0, n_above=1.0, n_below=1.0, quad_pts=16)
        )

    def test_round_trip_with_unscattered_slab_in_air(self):
        """Slab in air using R,T and unscattered transmission."""
        self._assert_boundary_round_trip_with_unscattered(
            dict(n=1.4, n_above=1.0, n_below=1.0, quad_pts=16)
        )

    def test_round_trip_with_unscattered_between_glass_slides_in_air(self):
        """Between slides in air using R,T and unscattered transmission."""
        self._assert_boundary_round_trip_with_unscattered(
            dict(n=1.4, n_above=1.5, n_below=1.5, quad_pts=16)
        )


if __name__ == "__main__":
    unittest.main()
