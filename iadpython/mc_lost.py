# pylint: disable=invalid-name

"""
Wrapper for the mc_lost Monte Carlo binary that estimates lost light fractions.

The binary must be built before use::

    cd iad && make mc_lost

The ``mc_lost`` program simulates photon transport through a slab and integrating
sphere ports using Monte Carlo to determine how much light escapes without being
collected (lost light).  The four returned quantities are fractions of the
incident power:

- ``ur1_lost`` — lost collimated-incidence reflected light
- ``ut1_lost`` — lost collimated-incidence transmitted light
- ``uru_lost`` — lost diffuse-incidence reflected light  (0 when method != 'substitution')
- ``utu_lost`` — lost diffuse-incidence transmitted light (0 when method != 'substitution')

Typical usage::

    import iadpython

    lost = iadpython.mc_lost.run_mc_lost(
        a=0.5, b=1.0, g=0.0,
        n_sample=1.4, n_slide=1.0,
        d_port_r=10.0, d_port_t=10.0,
        d_beam=5.0, t_sample=1.0,
        n_photons=100_000,
        binary_path="/path/to/iad/mc_lost",
    )
    ur1_lost, ut1_lost, uru_lost, utu_lost = lost
"""

import shlex
import subprocess


def run_mc_lost(
    a,
    b,
    g,
    *,
    n_sample=1.0,
    n_slide=1.0,
    d_port_r,
    d_port_t,
    d_beam=5.0,
    t_sample=1.0,
    t_slide=0.0,
    n_photons=100_000,
    method="substitution",
    binary_path="mc_lost",
):
    """Call the mc_lost binary and return the four lost-light fractions.

    The binary must be built first::

        cd iad && make mc_lost

    Parameters
    ----------
    a : float
        Single-scattering albedo [0, 1].
    b : float
        Optical thickness (> 0).
    g : float
        Scattering anisotropy [-1, 1].
    n_sample : float
        Index of refraction of the slab (default 1.0).
    n_slide : float
        Index of refraction of the cover glass slides (default 1.0, i.e. no slide).
        The binary applies the same index to both the top and bottom slides.
    d_port_r : float
        Diameter of the reflection-sphere sample port in mm.
    d_port_t : float
        Diameter of the transmission-sphere sample port in mm.
        The mc_lost binary accepts a single port diameter; when the two values
        differ the collimated lost fractions (ur1_lost, ut1_lost) are computed
        using d_port_r and the diffuse fractions (uru_lost, utu_lost) using
        d_port_t.  If both are equal a single call is made.
    d_beam : float
        Beam diameter in mm (default 5.0).
    t_sample : float
        Physical thickness of the slab in mm (default 1.0).
    t_slide : float
        Physical thickness of the cover glass slides in mm (default 0.0,
        i.e. no slide).  Only meaningful when n_slide != 1.0.
    n_photons : int
        Number of photons per MC run (default 100 000).
    method : str
        'substitution' or 'comparison'.  When not 'substitution', uru_lost
        and utu_lost are forced to 0.0, mirroring the C implementation.
    binary_path : str
        Path to the mc_lost binary.  If only a name is given the system PATH
        is searched.

    Returns
    -------
    tuple[float, float, float, float]
        ``(ur1_lost, ut1_lost, uru_lost, utu_lost)``

    Raises
    ------
    FileNotFoundError
        If the binary cannot be found.
    RuntimeError
        If the binary exits with a non-zero status or produces unexpected output.
    """
    def _call(d_port, collimated_only):
        """Build and run one mc_lost command; return the 12-float machine line."""
        cmd = [
            binary_path,
            "-a", f"{float(a):.8g}",
            "-b", f"{float(b):.8g}",
            "-g", f"{float(g):.8g}",
            "-n", f"{float(n_sample):.8g}",
            "-N", f"{float(n_slide):.8g}",
            "-P", f"{float(d_port):.8g}",
            "-B", f"{float(d_beam):.8g}",
            "-t", f"{float(t_sample):.8g}",
            "-p", str(int(n_photons)),
            "-m",
        ]
        if n_slide != 1.0 and t_slide > 0.0:
            cmd += ["-T", f"{float(t_slide):.8g}"]
        if collimated_only:
            cmd += ["-C"]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"mc_lost binary not found at {shlex.quote(binary_path)!r}. "
                "Build it with: cd iad && make mc_lost"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"mc_lost exited with status {exc.returncode}.\n"
                f"stderr: {exc.stderr.strip()}"
            ) from exc

        line = result.stdout.strip()
        parts = line.split()
        if len(parts) != 12:
            raise RuntimeError(
                f"Expected 12 values from mc_lost -m, got {len(parts)}: {line!r}"
            )
        return [float(p) for p in parts]

    if d_port_r == d_port_t:
        # Single call: both ports identical, binary gives all four lost values.
        values = _call(d_port_r, collimated_only=False)
        ur1_lost = values[8]
        ut1_lost = values[9]
        uru_lost = values[10]
        utu_lost = values[11]
    else:
        # Ports differ: separate calls for collimated (r-port) and diffuse (t-port).
        c_values = _call(d_port_r, collimated_only=True)
        ur1_lost = c_values[8]
        ut1_lost = c_values[9]

        d_values = _call(d_port_t, collimated_only=False)
        uru_lost = d_values[10]
        utu_lost = d_values[11]

    # Diffuse lost light does not apply for dual-beam (comparison) measurements.
    if method.lower() != "substitution":
        uru_lost = 0.0
        utu_lost = 0.0

    return ur1_lost, ut1_lost, uru_lost, utu_lost
