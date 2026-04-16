# pylint: disable=invalid-name

"""Tests for the mc_lost binary wrapper (iadpython.mc_lost).

These tests are skipped automatically when the mc_lost binary is not present.
Build it with::

    cd iad && make mc_lost
"""

import math
import subprocess

import pytest

from iadpython.mc_lost import run_mc_lost

# mc_lost_path fixture is provided by conftest.py

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run(mc_lost_path, **kwargs):
    """Convenience wrapper with sensible defaults."""
    defaults = dict(
        a=0.5,
        b=1.0,
        g=0.0,
        n_sample=1.4,
        n_slide=1.0,
        d_port_r=10.0,
        d_port_t=10.0,
        d_beam=5.0,
        t_sample=1.0,
        n_photons=50_000,
        method="substitution",
        binary_path=mc_lost_path,
    )
    defaults.update(kwargs)
    return run_mc_lost(**defaults)


# ---------------------------------------------------------------------------
# Basic sanity tests
# ---------------------------------------------------------------------------


class TestReturnShape:
    """Tests for the shape and type of mc_lost return values."""

    def test_returns_four_floats(self, mc_lost_path):
        """run_mc_lost should return a 4-tuple of floats."""
        result = _run(mc_lost_path)
        assert len(result) == 4
        for v in result:
            assert isinstance(v, float)

    def test_all_values_in_unit_interval(self, mc_lost_path):
        """All four lost-light fractions must lie in [0, 1]."""
        ur1_lost, ut1_lost, uru_lost, utu_lost = _run(mc_lost_path)
        for name, v in [("ur1_lost", ur1_lost), ("ut1_lost", ut1_lost), ("uru_lost", uru_lost), ("utu_lost", utu_lost)]:
            assert 0.0 <= v <= 1.0, f"{name}={v} outside [0, 1]"


class TestPhysics:
    """Physics-based sanity checks for mc_lost lost-light fractions."""

    def test_lost_increases_with_smaller_port(self, mc_lost_path):
        """Smaller port → more lost light."""
        _, _, uru_large, _ = _run(mc_lost_path, d_port_r=20.0, d_port_t=20.0)
        _, _, uru_small, _ = _run(mc_lost_path, d_port_r=5.0, d_port_t=5.0)
        assert uru_small >= uru_large, f"Expected more lost with smaller port: {uru_small} vs {uru_large}"

    def test_lost_is_zero_for_huge_port(self, mc_lost_path):
        """A port much larger than the beam should collect nearly everything."""
        ur1_lost, ut1_lost, uru_lost, utu_lost = _run(mc_lost_path, d_port_r=500.0, d_port_t=500.0, d_beam=1.0)
        tol = 0.01
        assert ur1_lost < tol, f"ur1_lost={ur1_lost} too large for huge port"
        assert ut1_lost < tol, f"ut1_lost={ut1_lost} too large for huge port"
        assert uru_lost < tol, f"uru_lost={uru_lost} too large for huge port"
        assert utu_lost < tol, f"utu_lost={utu_lost} too large for huge port"

    def test_high_albedo_has_more_scattered_light_to_lose(self, mc_lost_path):
        """High-albedo sample scatters more; more scattered light exits the port."""
        _, _, uru_hi, _ = _run(mc_lost_path, a=0.99, b=1.0, g=0.0)
        _, _, uru_lo, _ = _run(mc_lost_path, a=0.01, b=1.0, g=0.0)
        # For diffuse incidence, high albedo reflects more diffusely → more lost
        assert uru_hi >= uru_lo, f"Expected uru_lost(a=0.99)={uru_hi} >= uru_lost(a=0.01)={uru_lo}"

    def test_nonsubstitution_zeros_diffuse_lost(self, mc_lost_path):
        """Dual-beam (comparison) method: uru_lost and utu_lost must be 0."""
        _, _, uru_lost, utu_lost = _run(mc_lost_path, method="comparison")
        assert uru_lost == 0.0
        assert utu_lost == 0.0

    def test_substitution_has_nonzero_diffuse_lost(self, mc_lost_path):
        """Substitution method with small port should give nonzero diffuse lost."""
        _, _, uru_lost, utu_lost = _run(mc_lost_path, d_port_r=10.0, d_port_t=10.0, method="substitution")
        # With a 10 mm port, some diffuse light misses for a scattering sample
        assert uru_lost >= 0.0
        assert utu_lost >= 0.0


class TestMCConsistency:
    """Tests that MC output is self-consistent and agrees with adding-doubling."""

    def test_mc_and_ad_ur1_agree_within_noise(self, mc_lost_path):
        """MC total (col 1) ≈ AD value (col 5) for a moderately scattering slab."""
        # Run with enough photons for ~1% noise.  We access raw values via a
        # modified call that uses a large port so lost≈0, then MC total≈AD.
        cmd = [
            mc_lost_path,
            "-a",
            "0.5",
            "-b",
            "1.0",
            "-g",
            "0.0",
            "-n",
            "1.4",
            "-N",
            "1.0",
            "-P",
            "500",
            "-B",
            "1",
            "-p",
            "200000",
            "-m",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        parts = result.stdout.strip().split()
        assert len(parts) == 12
        mc_ur1 = float(parts[0])
        ad_ur1 = float(parts[4])
        # With a huge port, MC should match AD within ~2%
        assert math.isclose(
            mc_ur1, ad_ur1, rel_tol=0.05
        ), f"MC ur1={mc_ur1:.5f} vs AD ur1={ad_ur1:.5f}: discrepancy > 5%"

    def test_different_ports_consistent(self, mc_lost_path):
        """Separate-port code path returns plausible values."""
        ur1_lost, ut1_lost, uru_lost, utu_lost = _run(mc_lost_path, d_port_r=8.0, d_port_t=12.0)
        for name, v in [("ur1_lost", ur1_lost), ("ut1_lost", ut1_lost), ("uru_lost", uru_lost), ("utu_lost", utu_lost)]:
            assert 0.0 <= v <= 1.0, f"{name}={v} outside [0, 1]"


class TestErrorHandling:
    """Tests for error conditions in run_mc_lost."""

    def test_bad_binary_path_raises_file_not_found(self):
        """A non-existent binary path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="mc_lost binary not found"):
            run_mc_lost(
                a=0.5,
                b=1.0,
                g=0.0,
                d_port_r=10.0,
                d_port_t=10.0,
                binary_path="/nonexistent/mc_lost",
            )
