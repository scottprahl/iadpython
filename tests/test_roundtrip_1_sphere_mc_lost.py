# pylint: disable=invalid-name

"""Round-trip tests for forward and inverse R,T with one integrating sphere and
MC-estimated lost light.

The round-trip is self-consistent: both the forward measurement generation and
the inverse verification step use MC-estimated lost light, so the same physical
model is applied throughout.  The algorithm is:

  1. For true (a, b, g): run mc_lost to get lost fractions, then compute
     measured_rt() with those fractions → (m_r_fwd, m_t_fwd).
  2. Inverse: run invert_rt() with mc_lost_path set; the outer MC loop
     iterates until the lost-light estimates converge.
  3. Re-forward: for inverted (a_inv, b_inv, g_inv) use the converged lost
     fractions from the inversion and compute measured_rt() → (m_r_inv, m_t_inv).
  4. Assert |m_r_fwd - m_r_inv| and |m_t_fwd - m_t_inv| are within tolerance.

All boundary configurations pass within the 5 % tolerance.

Skip condition: mc_lost binary absent (build with: cd iad && make mc_lost).
"""

import itertools

import pytest

import iadpython
from iadpython.mc_lost import run_mc_lost


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

A_VALUES = (0.0, 0.5, 0.99)
B_VALUES = (0.1, 1.0, 10.0)
G_VALUES = (-0.9, 0.0, 0.9)

RT_TOLERANCE = 5e-2   # 5 %, matching the no-lost-light round-trip tests

SPHERE_CONFIG = {
    "d_sphere": 150.0,
    "d_sample": 10.0,
    "d_third": 10.0,
    "d_detector": 5.0,
    "r_detector": 0.04,
    "r_wall": 0.98,
    "r_std": 0.99,
}

# Photons per MC call.  100 k gives ~0.3 % noise on the lost fractions, which
# is well within the 5 % round-trip tolerance for moderate lost values.
MC_PHOTONS = 100_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spheres():
    r_sphere = iadpython.Sphere(**SPHERE_CONFIG, refl=True)
    t_sphere = iadpython.Sphere(**SPHERE_CONFIG, refl=False)
    return r_sphere, t_sphere


def _measured_rt_with_mc_lost(sample, mc_lost_path):
    """Compute sphere-corrected R and T with MC-estimated lost fractions.

    Calls mc_lost once for the given sample, sets the four lost components on
    the Experiment, then returns measured_rt().
    """
    r_sphere, t_sphere = _make_spheres()
    exp = iadpython.Experiment(
        sample=sample,
        num_spheres=1,
        r_sphere=r_sphere,
        t_sphere=t_sphere,
    )
    exp.method = "substitution"

    ur1_lost, ut1_lost, uru_lost, utu_lost = run_mc_lost(
        a=float(sample.a),
        b=float(sample.b),
        g=float(sample.g),
        n_sample=float(sample.n),
        n_slide=float(sample.n_above) if sample.n_above != 1.0 else 1.0,
        d_port_r=SPHERE_CONFIG["d_sample"],
        d_port_t=SPHERE_CONFIG["d_sample"],
        d_beam=5.0,
        t_sample=float(sample.d),
        n_photons=MC_PHOTONS,
        method="substitution",
        binary_path=mc_lost_path,
    )
    exp.ur1_lost = ur1_lost
    exp.ut1_lost = ut1_lost
    exp.uru_lost = uru_lost
    exp.utu_lost = utu_lost

    m_r, m_t = exp.measured_rt()
    return float(m_r), float(m_t)


def _round_trip_error(sample_kwargs, mc_lost_path):
    """Return (max_dr, max_dt, worst_case) over the full (a, b, g) grid."""
    max_dr = 0.0
    max_dt = 0.0
    worst_case = None

    for a, b, g in itertools.product(A_VALUES, B_VALUES, G_VALUES):
        fwd_sample = iadpython.Sample(a=a, b=b, g=g, **sample_kwargs)

        # Step 1: forward measurement WITH MC-estimated lost light
        m_r_fwd, m_t_fwd = _measured_rt_with_mc_lost(fwd_sample, mc_lost_path)

        # Step 2: inverse WITH MC iteration
        r_sphere, t_sphere = _make_spheres()
        inv_exp = iadpython.Experiment(
            r=m_r_fwd,
            t=m_t_fwd,
            sample=iadpython.Sample(**sample_kwargs),
            default_g=g,
            num_spheres=1,
            r_sphere=r_sphere,
            t_sphere=t_sphere,
        )
        inv_exp.method = "substitution"
        inv_exp.mc_lost_path = mc_lost_path
        inv_exp.n_photons = MC_PHOTONS
        a_inv, b_inv, g_inv = inv_exp.invert_rt()

        # Step 3: re-forward with the converged lost fractions from the inversion
        r_sphere2, t_sphere2 = _make_spheres()
        inv_sample = iadpython.Sample(a=a_inv, b=b_inv, g=g_inv, **sample_kwargs)
        chk_exp = iadpython.Experiment(
            sample=inv_sample,
            num_spheres=1,
            r_sphere=r_sphere2,
            t_sphere=t_sphere2,
        )
        chk_exp.method = "substitution"
        chk_exp.ur1_lost = inv_exp.ur1_lost
        chk_exp.ut1_lost = inv_exp.ut1_lost
        chk_exp.uru_lost = inv_exp.uru_lost
        chk_exp.utu_lost = inv_exp.utu_lost
        m_r_inv, m_t_inv = chk_exp.measured_rt()
        m_r_inv = float(m_r_inv)
        m_t_inv = float(m_t_inv)

        dr = abs(m_r_fwd - m_r_inv)
        dt = abs(m_t_fwd - m_t_inv)

        if dr > max_dr or dt > max_dt:
            worst_case = (a, b, g, float(a_inv), float(b_inv), float(g_inv), dr, dt)

        max_dr = max(max_dr, dr)
        max_dt = max(max_dt, dt)

    return max_dr, max_dt, worst_case


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_round_trip_mc_lost_matched_boundaries(mc_lost_path):
    """1-sphere MC lost light round-trip: matched boundaries (n=1/1/1)."""
    sample_kwargs = {"n": 1.0, "n_above": 1.0, "n_below": 1.0, "quad_pts": 16}
    max_dr, max_dt, worst_case = _round_trip_error(sample_kwargs, mc_lost_path)
    assert max_dr <= RT_TOLERANCE, f"max_dr={max_dr:.4f} > {RT_TOLERANCE}, worst={worst_case}"
    assert max_dt <= RT_TOLERANCE, f"max_dt={max_dt:.4f} > {RT_TOLERANCE}, worst={worst_case}"


def test_round_trip_mc_lost_slab_in_air(mc_lost_path):
    """1-sphere MC lost light round-trip: slab in air (n=1/1.4/1)."""
    sample_kwargs = {"n": 1.4, "n_above": 1.0, "n_below": 1.0, "quad_pts": 16}
    max_dr, max_dt, worst_case = _round_trip_error(sample_kwargs, mc_lost_path)
    assert max_dr <= RT_TOLERANCE, f"max_dr={max_dr:.4f} > {RT_TOLERANCE}, worst={worst_case}"
    assert max_dt <= RT_TOLERANCE, f"max_dt={max_dt:.4f} > {RT_TOLERANCE}, worst={worst_case}"


def test_round_trip_mc_lost_between_glass_slides(mc_lost_path):
    """1-sphere MC lost light round-trip: between glass slides (n=1/1.5/1.4/1.5/1)."""
    sample_kwargs = {"n": 1.4, "n_above": 1.5, "n_below": 1.5, "quad_pts": 16}
    max_dr, max_dt, worst_case = _round_trip_error(sample_kwargs, mc_lost_path)
    assert max_dr <= RT_TOLERANCE, f"max_dr={max_dr:.4f} > {RT_TOLERANCE}, worst={worst_case}"
    assert max_dt <= RT_TOLERANCE, f"max_dt={max_dt:.4f} > {RT_TOLERANCE}, worst={worst_case}"
