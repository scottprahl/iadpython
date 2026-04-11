"""
Module for direct and inverse adding-doubling calculations.

Example:
        >>> import iadpython as iad

        >>> a = 0.9
        >>> b = 10
        >>> g = 0.9
        >>> n_top = 1.5
        >>> n_bot = 1.5
        >>> ur1, ut1, uru, utu = iad.rt(n_top, n_bot, a, b, g)
        >>> print("total reflected   light for normal incidence %.5f" % UR1)
        >>> print("total transmitted light for normal incidence %.5f" % UT1)
        >>> print("total reflected   light for diffuse incidence %.5f" % URU)
        >>> print("total transmitted light for diffuse incidence %.5f" % UTU)
"""

import os
import sys
import ctypes
import ctypes.util
import pathlib
import numpy as np

__all__ = (
    "basic_rt",
    "basic_rt_unscattered",
    "basic_rt_cone",
    "basic_rt_oblique",
    "basic_rt_inverse",
    "rt",
    "rt_unscattered",
    "rt_cone",
    "rt_oblique",
    "rt_inverse",
)

_UNKNOWN = 0
_COMPARISON = 1
_SUBSTITUTION = 2
_MC_NONE = 0
_MC_USE_EXISTING = 1
_UNINITIALIZED = -99
_HENYEY_GREENSTEIN = 1
_REFLECTION_SPHERE = 1
_TRANSMISSION_SPHERE = 0


def _libiad_candidates():
    """Yield candidate shared-library paths in preference order."""
    override = os.environ.get("IADPYTHON_LIBIAD")
    if override:
        yield override

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    for name in ("libiad.dylib", "libiad.so", "libiad.dll"):
        candidate = repo_root / "iad" / "src" / name
        if candidate.exists():
            yield str(candidate)

    found = ctypes.util.find_library("libiad")
    if found:
        yield found


libiad_path = next(iter(_libiad_candidates()), None)

if not libiad_path:
    print("Unable to find the libiad library.")
    print("     macOS   = 'libiad.dylib'")
    print("     unix    = 'libiad.so'")
    print("     Windows = 'libiad.dll'")
    print("Paths searched:")
    print("     env var = IADPYTHON_LIBIAD")
    for p in sys.path:
        print("    ", p)
    sys.exit()

try:
    libiad = ctypes.CDLL(libiad_path)
except OSError:
    print("Unable to load the libiad library")
    print("Sorry")
    sys.exit()


class ADSlabType(ctypes.Structure):
    """ctypes mirror of the C `struct AD_slab_type`."""

    _fields_ = (
        ("a", ctypes.c_double),
        ("b", ctypes.c_double),
        ("g", ctypes.c_double),
        ("phase_function", ctypes.c_int),
        ("n_slab", ctypes.c_double),
        ("n_top_slide", ctypes.c_double),
        ("n_bottom_slide", ctypes.c_double),
        ("b_top_slide", ctypes.c_double),
        ("b_bottom_slide", ctypes.c_double),
        ("cos_angle", ctypes.c_double),
    )


class ADMethodType(ctypes.Structure):
    """ctypes mirror of the C `struct AD_method_type`."""

    _fields_ = (
        ("quad_pts", ctypes.c_int),
        ("a_calc", ctypes.c_double),
        ("b_calc", ctypes.c_double),
        ("g_calc", ctypes.c_double),
        ("b_thinnest", ctypes.c_double),
    )


class MeasureType(ctypes.Structure):
    """ctypes mirror of the C `struct measure_type`."""

    _fields_ = (
        ("slab_index", ctypes.c_double),
        ("slab_thickness", ctypes.c_double),
        ("slab_top_slide_index", ctypes.c_double),
        ("slab_top_slide_b", ctypes.c_double),
        ("slab_top_slide_thickness", ctypes.c_double),
        ("slab_bottom_slide_index", ctypes.c_double),
        ("slab_bottom_slide_b", ctypes.c_double),
        ("slab_bottom_slide_thickness", ctypes.c_double),
        ("slab_cos_angle", ctypes.c_double),
        ("num_spheres", ctypes.c_int),
        ("num_measures", ctypes.c_int),
        ("method", ctypes.c_int),
        ("flip_sample", ctypes.c_int),
        ("baffle_r", ctypes.c_int),
        ("baffle_t", ctypes.c_int),
        ("d_beam", ctypes.c_double),
        ("fraction_of_ru_in_mr", ctypes.c_double),
        ("fraction_of_tu_in_mt", ctypes.c_double),
        ("m_r", ctypes.c_double),
        ("m_t", ctypes.c_double),
        ("m_u", ctypes.c_double),
        ("lambda_", ctypes.c_double),
        ("as_r", ctypes.c_double),
        ("ad_r", ctypes.c_double),
        ("at_r", ctypes.c_double),
        ("aw_r", ctypes.c_double),
        ("rd_r", ctypes.c_double),
        ("rw_r", ctypes.c_double),
        ("rstd_r", ctypes.c_double),
        ("f_r", ctypes.c_double),
        ("as_t", ctypes.c_double),
        ("ad_t", ctypes.c_double),
        ("at_t", ctypes.c_double),
        ("aw_t", ctypes.c_double),
        ("rd_t", ctypes.c_double),
        ("rw_t", ctypes.c_double),
        ("rstd_t", ctypes.c_double),
        ("ur1_lost", ctypes.c_double),
        ("uru_lost", ctypes.c_double),
        ("ut1_lost", ctypes.c_double),
        ("utu_lost", ctypes.c_double),
        ("d_sphere_r", ctypes.c_double),
        ("d_sphere_t", ctypes.c_double),
    )


class InvertType(ctypes.Structure):
    """ctypes mirror of the C `struct invert_type`."""

    _fields_ = (
        ("a", ctypes.c_double),
        ("b", ctypes.c_double),
        ("g", ctypes.c_double),
        ("found", ctypes.c_int),
        ("search", ctypes.c_int),
        ("metric", ctypes.c_int),
        ("tolerance", ctypes.c_double),
        ("MC_tolerance", ctypes.c_double),
        ("final_distance", ctypes.c_double),
        ("error", ctypes.c_int),
        ("slab", ADSlabType),
        ("method", ADMethodType),
        ("AD_iterations", ctypes.c_int),
        ("MC_iterations", ctypes.c_int),
        ("default_a", ctypes.c_double),
        ("default_b", ctypes.c_double),
        ("default_g", ctypes.c_double),
        ("default_ba", ctypes.c_double),
        ("default_bs", ctypes.c_double),
        ("default_mua", ctypes.c_double),
        ("default_mus", ctypes.c_double),
    )

libiad.ez_RT.argtypes = (
    ctypes.c_int,  # n quadrature points
    ctypes.c_double,  # slab index of refraction
    ctypes.c_double,  # top slide index of refraction
    ctypes.c_double,  # bottom slide index of refraction
    ctypes.c_double,  # albedo mus/(mus+mua)
    ctypes.c_double,  # optical thickness d*(mua+mus)
    ctypes.c_double,  # scattering anisotropy g
    ctypes.POINTER(ctypes.c_double),  # UR1
    ctypes.POINTER(ctypes.c_double),  # UT1
    ctypes.POINTER(ctypes.c_double),  # URU
    ctypes.POINTER(ctypes.c_double),  # UTU
)


def basic_rt(n, nslab, ntop, nbot, a, b, g):
    """
    Calculate the total reflection and transmission for a turbid slab.

    `n` is the number of quadrature angles (more angles give better accuracy but slow
    the calculation by `n**3`, `n` must be a multiple of 4, typically 16 is good for forward
    calculations and 4 is appropriate for inverse calculations).

    The semi-infinite slab may be bounded by glass slides.
    The top glass slides have an index of refraction `ntop` and the bottom slide has
    an index `nbot`.  If there are no glass slides, set `ntop` and `nbottom` to 1.

    All input parameters must be scalars.

    Args:
        n: number of points in quadrature (multiple of 4)
        nslab: index of refraction of the slab
        ntop: index of refraction of the top slide
        nbot: index of refraction of the bottom slide
        a: single scattering albedo of the slab
        b: optical thickness of the slab
        g: anisotropy of single scattering
    Returns:
        - `UR1` is the total reflection for normally incident collimated light.
        - `UT1` is the total transmission for normally incident collimated light.
        - `URU` is the total reflection for diffuse incident light.
        - `UTU` is the total transmission for diffuse incident light.
    """
    ur1 = ctypes.c_double()
    ut1 = ctypes.c_double()
    uru = ctypes.c_double()
    utu = ctypes.c_double()
    libiad.ez_RT(n, nslab, ntop, nbot, a, b, g, ur1, ut1, uru, utu)
    return ur1.value, ut1.value, uru.value, utu.value


libiad.ez_RT_unscattered.argtypes = (
    ctypes.c_int,  # n quadrature points
    ctypes.c_double,  # slab index of refraction
    ctypes.c_double,  # top slide index of refraction
    ctypes.c_double,  # bottom slide index of refraction
    ctypes.c_double,  # albedo mus/(mus+mua)
    ctypes.c_double,  # optical thickness d*(mua+mus)
    ctypes.c_double,  # scattering anisotropy g
    ctypes.POINTER(ctypes.c_double),  # UR1
    ctypes.POINTER(ctypes.c_double),  # UT1
    ctypes.POINTER(ctypes.c_double),  # URU
    ctypes.POINTER(ctypes.c_double),  # UTU
)


def basic_rt_unscattered(n, nslab, ntop, nbot, a, b, g):
    """
    Calculate the unscattered reflection and transmission for a turbid slab.

    `n` is the number of quadrature angles (more angles give better accuracy but slow
    the calculation by `n**3`, `n` must be a multiple of 4, typically 16 is good for forward
    calculations and 4 is appropriate for inverse calculations).

    The semi-infinite slab may be bounded by glass slides.
    The top glass slides have an index of refraction `ntop` and the bottom slide has
    an index `nbot`.  If there are no glass slides, set `ntop` and `nbottom` to 1.

    All input parameters must be scalars.

    Args:
        n: number of points in quadrature (multiple of 4)
        nslab: index of refraction of the slab
        ntop: index of refraction of the top slide
        nbot: index of refraction of the bottom slide
        a: single scattering albedo of the slab
        b: optical thickness of the slab
        g: anisotropy of single scattering
    Returns:
        - `UR1` is the total reflection for normally incident collimated light.
        - `UT1` is the total transmission for normally incident collimated light.
        - `URU` is the total reflection for diffuse incident light.
        - `UTU` is the total transmission for diffuse incident light.
    """
    ur1 = ctypes.c_double()
    ut1 = ctypes.c_double()
    uru = ctypes.c_double()
    utu = ctypes.c_double()
    libiad.ez_RT_unscattered(n, nslab, ntop, nbot, a, b, g, ur1, ut1, uru, utu)
    return ur1.value, ut1.value, uru.value, utu.value


libiad.ez_RT_Cone.argtypes = (
    ctypes.c_int,  # n quadrature points
    ctypes.c_double,  # slab index of refraction
    ctypes.c_double,  # top slide index of refraction
    ctypes.c_double,  # bottom slide index of refraction
    ctypes.c_double,  # albedo mus/(mus+mua)
    ctypes.c_double,  # optical thickness d*(mua+mus)
    ctypes.c_double,  # scattering anisotropy g
    ctypes.c_double,  # cosine of cone angle
    ctypes.POINTER(ctypes.c_double),  # UR1
    ctypes.POINTER(ctypes.c_double),  # UT1
    ctypes.POINTER(ctypes.c_double),  # URU
    ctypes.POINTER(ctypes.c_double),  # UTU
)


def basic_rt_cone(n, nslab, ntop, nbot, a, b, g, cos_cone_angle):
    """
    Calculate reflection and transmission for a turbid slab exiting within a cone.

    This routine assumes normally incident or uniformly diffuse incident light
    and returns the total reflected or transmitted light that exits within a cone.
    The cosine of the cone angle is given by `cos_cone_angle`.

    `n` is the number of quadrature angles (more angles give better accuracy but slow
    the calculation by `n**3`, `n` must be a multiple of 4, typically 16 is good for forward
    calculations and 4 is appropriate for inverse calculations).

    The semi-infinite slab may be bounded by glass slides.
    The top glass slides have an index of refraction `ntop` and the bottom slide has
    an index `nbot`.  If there are no glass slides, set `ntop` and `nbottom` to 1.

    All input parameters must be scalars.

    Args:
        n: number of points in quadrature (multiple of 4)
        nslab: index of refraction of the slab
        ntop: index of refraction of the top slide
        nbot: index of refraction of the bottom slide
        a: single scattering albedo of the slab
        b: optical thickness of the slab
        g: anisotropy of single scattering
        cos_cone_angle: cosine of cone of exiting light
    Returns:
        - `UR1` is the reflection within cone for normally incident collimated light.
        - `UT1` is the transmission within cone for normally incident collimated light.
        - `URU` is the reflection within cone for diffuse incident light.
        - `UTU` is the transmission within cone for diffuse incident light.
    """
    ur1 = ctypes.c_double()
    ut1 = ctypes.c_double()
    uru = ctypes.c_double()
    utu = ctypes.c_double()
    libiad.ez_RT_Cone(n, nslab, ntop, nbot, a, b, g, cos_cone_angle, ur1, ut1, uru, utu)
    return ur1.value, ut1.value, uru.value, utu.value


libiad.ez_RT_Oblique.argtypes = (
    ctypes.c_int,  # n quadrature points
    ctypes.c_double,  # slab index of refraction
    ctypes.c_double,  # top slide index of refraction
    ctypes.c_double,  # bottom slide index of refraction
    ctypes.c_double,  # albedo mus/(mus+mua)
    ctypes.c_double,  # optical thickness d*(mua+mus)
    ctypes.c_double,  # scattering anisotropy g
    ctypes.c_double,  # cosine of oblique angle
    ctypes.POINTER(ctypes.c_double),  # UR1
    ctypes.POINTER(ctypes.c_double),  # UT1
    ctypes.POINTER(ctypes.c_double),  # URU
    ctypes.POINTER(ctypes.c_double),  # UTU
)


def basic_rt_oblique(n, nslab, ntop, nbot, a, b, g, cos_oblique):
    """
    Calculate reflection and transmission for light incident at a oblique angle.

    This returns the total R and T for light incident at an oblique angle or
    incident within a cone.  The cosine of the oblique angle (or the cone) is
    `cos_oblique`.

    n is the number of quadrature angles (more angles give better accuracy but slow
    the calculation by n**3, n must be a multiple of 4, typically 16 is good for forward
    calculations and 4 is appropriate for inverse calculations).

    The semi-infinite slab may be bounded by glass slides.
    The top glass slides have an index of refraction `ntop` and the bottom slide has
    an index `nbot`.  If there are no glass slides, set `ntop` and `nbottom` to 1.

    All input parameters must be scalars.

    Args:
        n: number of points in quadrature (multiple of 4)
        nslab: index of refraction of the slab
        ntop: index of refraction of the top slide
        nbot: index of refraction of the bottom slide
        a: single scattering albedo of the slab
        b: optical thickness of the slab
        g: anisotropy of single scattering
        cos_oblique: cosine of cone of incident light
    Returns:
        - `URx` is the total reflection for obliquely incident collimated light.
        - `UTx` is the total transmission for obliquely incident collimated light.
        - `URU` is the total reflection for diffuse light incident within a cone.
        - `UTU` is the total transmission for diffuse light incident within a cone.
    """
    ur1 = ctypes.c_double()
    ut1 = ctypes.c_double()
    uru = ctypes.c_double()
    utu = ctypes.c_double()
    libiad.ez_RT_Oblique(n, nslab, ntop, nbot, a, b, g, cos_oblique, ur1, ut1, uru, utu)
    return ur1.value, ut1.value, uru.value, utu.value


libiad.ez_Inverse_RT.argtypes = (
    ctypes.c_double,  # slab index of refraction
    ctypes.c_double,  # slide index of refraction
    ctypes.c_double,  # UR1
    ctypes.c_double,  # UT1
    ctypes.c_double,  # unscattered transmission
    ctypes.POINTER(ctypes.c_double),  # a
    ctypes.POINTER(ctypes.c_double),  # b
    ctypes.POINTER(ctypes.c_double),  # g
    ctypes.POINTER(ctypes.c_int),  # error
)


def basic_rt_inverse(nslab, nslide, ur1, ut1, tc):
    """
    Calculate optical properties given reflection and transmission values.

    Finds the optical properties `[a,b,g]` for a slab
    with total reflectance `ur1`, total transmission `ut1`, unscattered transmission `Tc`.
    The index of refraction of the slab is `nslab`, the index of refraction of the
    top and bottom slides is `nslide`.

    The semi-infinite slab may be bounded by glass slides.
    The top glass slides have an index of refraction `ntop` and the bottom slide has
    an index `nbot`.  If there are no glass slides, set `ntop` and `nbottom` to 1.

    All input parameters must be scalars.

    Args:
        nslab: index of refraction of the slab
        nslide: index of refraction of the slides
        ur1: is total reflection for normally incident collimated light.
        ut1: is total transmission for normally incident collimated light.
        tc: unscattered transmission through sample
    Returns:
        - `a` is the single scattering albedo of the slab
        - `b` is the optical thickness of the slab
        - `g` is the anisotropy of single scattering
    """
    a = ctypes.c_double()
    b = ctypes.c_double()
    g = ctypes.c_double()
    error = ctypes.c_int()
    libiad.ez_Inverse_RT(nslab, nslide, ur1, ut1, tc, a, b, g, error)
    return a.value, b.value, g.value, error.value


libiad.Initialize_Measure.argtypes = (ctypes.POINTER(MeasureType),)
libiad.Initialize_Measure.restype = None

libiad.Initialize_Result.argtypes = (MeasureType, ctypes.POINTER(InvertType), ctypes.c_int)
libiad.Initialize_Result.restype = None

libiad.Inverse_RT.argtypes = (MeasureType, ctypes.POINTER(InvertType))
libiad.Inverse_RT.restype = None

libiad.Set_Calc_State.argtypes = (MeasureType, InvertType)
libiad.Set_Calc_State.restype = None

libiad.Get_Calc_State.argtypes = (ctypes.POINTER(MeasureType), ctypes.POINTER(InvertType))
libiad.Get_Calc_State.restype = None

libiad.Calculate_MR_MT.argtypes = (
    MeasureType,
    InvertType,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
)
libiad.Calculate_MR_MT.restype = None

libiad.Calculate_Distance_With_Corrections.argtypes = (
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
)
libiad.Calculate_Distance_With_Corrections.restype = None

libiad.Gain.argtypes = (ctypes.c_int, MeasureType, ctypes.c_double, ctypes.c_double)
libiad.Gain.restype = ctypes.c_double


def _init_measure():
    """Return a C measurement structure populated with library defaults."""
    measure = MeasureType()
    libiad.Initialize_Measure(ctypes.byref(measure))
    return measure


def _c_method(method):
    """Translate Python method names to the C enum values."""
    if method in ("comparison", _COMPARISON):
        return _COMPARISON
    if method in ("substitution", _SUBSTITUTION):
        return _SUBSTITUTION
    return _UNKNOWN


def _apply_sphere_to_measure(measure, sphere, reflection):
    """Copy a Python `Sphere` into the corresponding C measurement fields."""
    if sphere is None:
        return

    prefix = "r" if reflection else "t"
    setattr(measure, f"d_sphere_{prefix}", float(sphere.d))
    setattr(measure, f"as_{prefix}", float(sphere.sample.a))
    setattr(measure, f"at_{prefix}", float(sphere.third.a))
    setattr(measure, f"ad_{prefix}", float(sphere.detector.a))
    setattr(measure, f"aw_{prefix}", float(sphere.a_wall))
    setattr(measure, f"rd_{prefix}", float(sphere.detector.uru))
    setattr(measure, f"rw_{prefix}", float(sphere.r_wall))
    setattr(measure, f"rstd_{prefix}", float(sphere.r_std))
    setattr(measure, f"baffle_{prefix}", int(bool(sphere.baffle)))


def _measure_from_experiment(experiment):
    """Create a C measurement struct from a Python `iadpython.Experiment`."""
    sample = experiment.sample
    measure = _init_measure()

    measure.slab_index = float(sample.n)
    measure.slab_thickness = float(sample.d)
    measure.slab_top_slide_index = float(sample.n_above)
    measure.slab_top_slide_b = float(sample.b_above)
    measure.slab_top_slide_thickness = float(sample.d_above)
    measure.slab_bottom_slide_index = float(sample.n_below)
    measure.slab_bottom_slide_b = float(sample.b_below)
    measure.slab_bottom_slide_thickness = float(sample.d_below)
    measure.slab_cos_angle = float(sample.nu_0)

    measure.num_spheres = int(experiment.num_spheres or 0)
    measure.num_measures = sum(value is not None for value in (experiment.m_r, experiment.m_t, experiment.m_u))
    measure.method = _c_method(experiment.method)
    measure.flip_sample = int(bool(experiment.flip_sample))

    measure.d_beam = float(experiment.d_beam)
    measure.fraction_of_ru_in_mr = float(experiment.fraction_of_rc_in_mr)
    measure.fraction_of_tu_in_mt = float(experiment.fraction_of_tc_in_mt)

    measure.m_r = 0.0 if experiment.m_r is None else float(experiment.m_r)
    measure.m_t = 0.0 if experiment.m_t is None else float(experiment.m_t)
    measure.m_u = 0.0 if experiment.m_u is None else float(experiment.m_u)
    measure.lambda_ = 0.0 if experiment.lambda0 is None else float(experiment.lambda0)
    measure.f_r = float(experiment.f_r)

    _apply_sphere_to_measure(measure, experiment.r_sphere, reflection=True)
    _apply_sphere_to_measure(measure, experiment.t_sphere, reflection=False)

    if measure.num_spheres == 0:
        measure.ur1_lost = 0.0
        measure.ut1_lost = 0.0
        measure.uru_lost = 0.0
        measure.utu_lost = 0.0
    else:
        measure.ur1_lost = float(experiment.ur1_lost)
        measure.ut1_lost = float(experiment.ut1_lost)
        measure.uru_lost = float(experiment.uru_lost)
        measure.utu_lost = float(experiment.utu_lost)

    return measure


def _result_from_experiment(experiment, measure):
    """Create a C inversion/result struct from a Python `iadpython.Experiment`."""
    sample = experiment.sample
    result = InvertType()
    libiad.Initialize_Result(measure, ctypes.byref(result), 1)

    result.slab.a = float(sample.a)
    result.slab.b = float(sample.b)
    result.slab.g = float(sample.g)
    result.slab.phase_function = _HENYEY_GREENSTEIN
    result.slab.n_slab = float(sample.n)
    result.slab.n_top_slide = float(sample.n_above)
    result.slab.n_bottom_slide = float(sample.n_below)
    result.slab.b_top_slide = float(sample.b_above)
    result.slab.b_bottom_slide = float(sample.b_below)
    result.slab.cos_angle = float(sample.nu_0)

    result.tolerance = float(experiment.tolerance)
    result.MC_tolerance = float(experiment.MC_tolerance)
    result.metric = int(experiment.metric)
    result.method.quad_pts = int(sample.quad_pts)
    if sample.b_thinnest is not None:
        result.method.b_thinnest = float(sample.b_thinnest)

    for attr in (
        "default_a",
        "default_b",
        "default_g",
        "default_ba",
        "default_bs",
        "default_mua",
        "default_mus",
    ):
        value = getattr(experiment, attr)
        if value is not None:
            setattr(result, attr, float(value))
        else:
            setattr(result, attr, float(_UNINITIALIZED))

    return result


def _c_gain(sphere, sample_uru=None, third_uru=None):
    """Developer-facing single-sphere gain calculation through libiad."""
    measure = _init_measure()
    _apply_sphere_to_measure(measure, sphere, reflection=bool(sphere.refl))

    if sample_uru is None:
        sample_uru = sphere.sample.uru
    if third_uru is None:
        third_uru = sphere.third.uru

    sphere_kind = _REFLECTION_SPHERE if sphere.refl else _TRANSMISSION_SPHERE
    return libiad.Gain(sphere_kind, measure, float(sample_uru), float(third_uru))


def _c_calculate_measured_rt(experiment, include_mc=False, include_spheres=True):
    """Developer-facing direct measured `M_R/M_T` calculation through libiad."""
    measure = _measure_from_experiment(experiment)
    result = _result_from_experiment(experiment, measure)

    measured_r = ctypes.c_double()
    measured_t = ctypes.c_double()
    include_mc_flag = _MC_USE_EXISTING if include_mc else _MC_NONE
    libiad.Calculate_MR_MT(
        measure,
        result,
        include_mc_flag,
        int(bool(include_spheres)),
        ctypes.byref(measured_r),
        ctypes.byref(measured_t),
    )
    return measured_r.value, measured_t.value


def _c_calculate_measured_rt_from_rt(experiment, ur1, ut1, uru, utu, ru, tu):
    """Developer-facing one-sphere normalization using explicit RT inputs."""
    measure = _measure_from_experiment(experiment)
    result = _result_from_experiment(experiment, measure)

    old_measure = MeasureType()
    old_result = InvertType()
    measured_r = ctypes.c_double()
    measured_t = ctypes.c_double()
    deviation = ctypes.c_double()

    libiad.Get_Calc_State(ctypes.byref(old_measure), ctypes.byref(old_result))
    try:
        libiad.Set_Calc_State(measure, result)
        libiad.Calculate_Distance_With_Corrections(
            float(ur1),
            float(ut1),
            float(ru),
            float(tu),
            float(uru),
            float(utu),
            ctypes.byref(measured_r),
            ctypes.byref(measured_t),
            ctypes.byref(deviation),
        )
    finally:
        libiad.Set_Calc_State(old_measure, old_result)

    return measured_r.value, measured_t.value


def _c_invert_experiment(experiment):
    """Developer-facing inverse calculation through libiad."""
    measure = _measure_from_experiment(experiment)
    result = _result_from_experiment(experiment, measure)
    libiad.Inverse_RT(measure, ctypes.byref(result))
    return result.a, result.b, result.g, result.error


def rt(nslab, nslide, a, b, g):
    """
    Calculate the total reflection and transmission for a turbid slab.

    This routine should be the primary entry point because the optical properties
    can be either scalars or arrays.

    Args:
        nslab: index of refraction of the slab
        nslide: index of refraction of the slides
        a: single scattering albedo of the slab
        b: optical thickness of the slab
        g: anisotropy of single scattering
    Returns:
        - `UR1` is the total reflection for normally incident collimated light.
        - `UT1` is the total transmission for normally incident collimated light.
        - `URU` is the total reflection for diffuse incident light.
        - `UTU` is the total transmission for diffuse incident light.
    """
    N_QUADRATURE = 16  # should be a multiple of 16

    if np.isscalar(a):
        len_a = 0
        aa = a
    else:
        len_a = len(a)

    if np.isscalar(b):
        len_b = 0
        bb = b
    else:
        len_b = len(b)

    if np.isscalar(g):
        len_g = 0
        gg = g
    else:
        len_g = len(g)

    thelen = max(len_a, len_b, len_g)

    if thelen == 0:
        return basic_rt(N_QUADRATURE, nslab, nslide, nslide, aa, bb, gg)

    if len_a and len_b and len_a != len_b:
        raise RuntimeError("rt: a and b arrays must be same length")

    if len_a and len_g and len_a != len_g:
        raise RuntimeError("rt: a and g arrays must be same length")

    if len_b and len_g and len_b != len_g:
        raise RuntimeError("rt: b and g arrays must be same length")

    ur1 = np.empty(thelen)
    ut1 = np.empty(thelen)
    uru = np.empty(thelen)
    utu = np.empty(thelen)

    for i in range(thelen):
        if len_a > 0:
            aa = a[i]

        if len_b > 0:
            bb = b[i]

        if len_g > 0:
            gg = g[i]

        ur1[i], ut1[i], uru[i], utu[i] = basic_rt(N_QUADRATURE, nslab, nslide, nslide, aa, bb, gg)

    return ur1, ut1, uru, utu


def rt_unscattered(nslab, nslide, a, b, g):
    """
    Calculate the unscattered reflection and transmission for a turbid slab.

    This routine should be the primary entry point because the optical properties
    can be either scalars or arrays.

    Args:
        nslab: index of refraction of the slab
        nslide: index of refraction of the slides
        a: single scattering albedo of the slab
        b: optical thickness of the slab
        g: anisotropy of single scattering
    Returns:
        - `UR1` is the unscattered reflection for normally incident collimated light.
        - `UT1` is the unscattered transmission for normally incident collimated light.
        - `URU` is the unscattered reflection for diffuse incident light.
        - `UTU` is the unscattered transmission for diffuse incident light.
    """
    N_QUADRATURE = 16  # should be a multiple of 16

    if np.isscalar(a):
        len_a = 0
        aa = a
    else:
        len_a = len(a)

    if np.isscalar(b):
        len_b = 0
        bb = b
    else:
        len_b = len(b)

    if np.isscalar(g):
        len_g = 0
        gg = g
    else:
        len_g = len(g)

    thelen = max(len_a, len_b, len_g)

    if thelen == 0:
        return basic_rt_unscattered(N_QUADRATURE, nslab, nslide, nslide, aa, bb, gg)

    if len_a and len_b and len_a != len_b:
        raise RuntimeError("rt_unscattered: a and b arrays must be same length")

    if len_a and len_g and len_a != len_g:
        raise RuntimeError("rt_unscattered: a and g arrays must be same length")

    if len_b and len_g and len_b != len_g:
        raise RuntimeError("rt_unscattered: b and g arrays must be same length")

    ur1 = np.empty(thelen)
    ut1 = np.empty(thelen)
    uru = np.empty(thelen)
    utu = np.empty(thelen)

    for i in range(thelen):
        if len_a > 0:
            aa = a[i]

        if len_b > 0:
            bb = b[i]

        if len_g > 0:
            gg = g[i]

        ur1[i], ut1[i], uru[i], utu[i] = basic_rt_unscattered(N_QUADRATURE, nslab, nslide, nslide, aa, bb, gg)

    return ur1, ut1, uru, utu


def rt_cone(nslab, nslide, a, b, g, cos_cone):
    """
    Calculate reflection and transmission for a turbid slab exiting within a cone.

    This routine should be the primary entry point because the optical properties
    can be either scalars or arrays.

    Args:
        n: number of points in quadrature (multiple of 4)
        nslab: index of refraction of the slab
        nslide: index of refraction of the top and bottom slides
        a: single scattering albedo of the slab
        b: optical thickness of the slab
        g: anisotropy of single scattering
        cos_cone: cosine of cone of exiting light
    Returns:
        - `UR1` is the reflection within cone for normally incident collimated light.
        - `UT1` is the transmission within cone for normally incident collimated light.
        - `URU` is the reflection within cone for diffuse incident light.
        - `UTU` is the transmission within cone for diffuse incident light.
    """
    N_QUADRATURE = 16  # should be a multiple of 16

    if np.isscalar(a):
        len_a = 0
        aa = a
    else:
        len_a = len(a)

    if np.isscalar(b):
        len_b = 0
        bb = b
    else:
        len_b = len(b)

    if np.isscalar(g):
        len_g = 0
        gg = g
    else:
        len_g = len(g)

    if np.isscalar(cos_cone):
        len_mu = 0
        mu = cos_cone
    else:
        len_mu = len(cos_cone)

    thelen = max(len_a, len_b, len_g, len_mu)

    if thelen == 0:
        return basic_rt_cone(N_QUADRATURE, nslab, nslide, nslide, aa, bb, gg, mu)

    if len_a and len_b and len_a != len_b:
        raise RuntimeError("rt_cone: a and b arrays must be same length")

    if len_a and len_g and len_a != len_g:
        raise RuntimeError("rt_cone: a and g arrays must be same length")

    if len_a and len_mu and len_a != len_mu:
        raise RuntimeError("rt_cone: a and mu arrays must be same length")

    if len_b and len_g and len_b != len_g:
        raise RuntimeError("rt_cone: b and g arrays must be same length")

    if len_b and len_mu and len_b != len_mu:
        raise RuntimeError("rt_cone: b and mu arrays must be same length")

    if len_g and len_mu and len_g != len_mu:
        raise RuntimeError("rt_cone: g and mu arrays must be same length")

    ur1 = np.empty(thelen)
    ut1 = np.empty(thelen)
    uru = np.empty(thelen)
    utu = np.empty(thelen)

    for i in range(thelen):
        if len_a > 0:
            aa = a[i]

        if len_b > 0:
            bb = b[i]

        if len_g > 0:
            gg = g[i]

        if len_mu > 0:
            mu = cos_cone[i]

        ur1[i], ut1[i], uru[i], utu[i] = basic_rt_cone(N_QUADRATURE, nslab, nslide, nslide, aa, bb, gg, mu)

    return ur1, ut1, uru, utu


def rt_oblique(nslab, nslide, a, b, g, cos_oblique):
    """
    Calculate reflection and transmission for light incident at a oblique angle.

    This routine should be the primary entry point because the optical properties
    can be either scalars or arrays.

    All input parameters must be scalars.

    Args:
        nslab: index of refraction of the slab
        nslide: index of refraction of the top and bottom slides
        a: single scattering albedo of the slab
        b: optical thickness of the slab
        g: anisotropy of single scattering
        cos_oblique: cosine of cone of incident light
    Returns:
        - `URx` is the total reflection for obliquely incident collimated light.
        - `UTx` is the total transmission for obliquely incident collimated light.
        - `URU` is the total reflection for diffuse light incident within a cone.
        - `UTU` is the total transmission for diffuse light incident within a cone.
    """
    N_QUADRATURE = 16  # should be a multiple of 16

    if np.isscalar(a):
        len_a = 0
        aa = a
    else:
        len_a = len(a)

    if np.isscalar(b):
        len_b = 0
        bb = b
    else:
        len_b = len(b)

    if np.isscalar(g):
        len_g = 0
        gg = g
    else:
        len_g = len(g)

    if np.isscalar(cos_oblique):
        len_mu = 0
        mu = cos_oblique
    else:
        len_mu = len(cos_oblique)

    thelen = max(len_a, len_b, len_g, len_mu)

    if thelen == 0:
        return basic_rt_oblique(N_QUADRATURE, nslab, nslide, nslide, aa, bb, gg, mu)

    if len_a and len_b and len_a != len_b:
        raise RuntimeError("rt_oblique: a and b arrays must be same length")

    if len_a and len_g and len_a != len_g:
        raise RuntimeError("rt_oblique: a and g arrays must be same length")

    if len_a and len_mu and len_a != len_mu:
        raise RuntimeError("rt_oblique: a and mu arrays must be same length")

    if len_b and len_g and len_b != len_g:
        raise RuntimeError("rt_oblique: b and g arrays must be same length")

    if len_b and len_mu and len_b != len_mu:
        raise RuntimeError("rt_oblique: b and mu arrays must be same length")

    if len_g and len_mu and len_g != len_mu:
        raise RuntimeError("rt_oblique: g and mu arrays must be same length")

    ur1 = np.empty(thelen)
    ut1 = np.empty(thelen)
    uru = np.empty(thelen)
    utu = np.empty(thelen)

    for i in range(thelen):
        if len_a > 0:
            aa = a[i]

        if len_b > 0:
            bb = b[i]

        if len_g > 0:
            gg = g[i]

        if len_mu > 0:
            mu = cos_oblique[i]

        ur1[i], ut1[i], uru[i], utu[i] = basic_rt_oblique(N_QUADRATURE, nslab, nslide, nslide, aa, bb, gg, mu)

    return ur1, ut1, uru, utu


def rt_inverse(nslab, nslide, ur1, ut1, t_unscattered):
    """
    Calculate [a,b,g] for a slab.

    This routine should be the primary entry point because the reflection
    and transmission can be either scalars or arrays.

    All input parameters must be scalars.

    Args:
        nslab: index of refraction of the slab
        nslide: index of refraction of the slides
        ur1: is total reflection for normally incident collimated light.
        ut1: is total transmission for normally incident collimated light.
        t_unscattered: unscattered transmission through sample
    Returns:
        - `a` is the single scattering albedo of the slab
        - `b` is the optical thickness of the slab
        - `g` is the anisotropy of single scattering
    """
    if np.isscalar(ur1):
        len_r1 = 0
    else:
        len_r1 = len(ur1)

    if np.isscalar(ut1):
        len_t1 = 0
    else:
        len_t1 = len(ut1)

    if np.isscalar(t_unscattered):
        len_tc = 0
        tc = t_unscattered
    else:
        len_tc = len(t_unscattered)

    thelen = max(len_r1, len_t1, len_tc)

    if thelen == 0:
        return basic_rt_inverse(nslab, nslide, ur1, ut1, tc)

    if len_r1 and len_t1 and len_r1 != len_t1:
        raise RuntimeError("inverse_rt: ur1 and ut1 arrays must be same length")

    if len_r1 and len_tc and len_r1 != len_tc:
        raise RuntimeError("inverse_rt: ur1 and tc arrays must be same length")

    if len_t1 and len_tc and len_t1 != len_tc:
        raise RuntimeError("inverse_rt: t1 and tc arrays must be same length")

    a = np.empty(thelen)
    b = np.empty(thelen)
    g = np.empty(thelen)
    error = np.empty(thelen)

    for i in range(thelen):
        if len_r1 > 0:
            r1 = ur1[i]

        if len_t1 > 0:
            t1 = ut1[i]

        if len_tc > 0:
            tc = t_unscattered[i]

        a[i], b[i], g[i], error[i] = basic_rt_inverse(nslab, nslide, r1, t1, tc)

    return a, b, g, error


class CSample:
    """Class for samples."""

    def __init__(self):
        """Initialize class."""
        self.sample_index = 1.4
        self.top_slide_index = 1.5
        self.bot_slide_index = 1.5
        self.sample_thickness = 1  # mm
        self.top_slide_thickness = 1  # mm
        self.bot_slide_thickness = 1  # mm

    def as_array(self):
        """Representation class as an array."""
        return [
            self.sample_index,
            self.top_slide_index,
            self.bot_slide_index,
            self.sample_thickness,
            self.top_slide_thickness,
            self.bot_slide_thickness,
        ]

    def init_from_array(self, a):
        """Initialize with an array."""
        self.sample_index = a[0]
        self.top_slide_index = a[1]
        self.bot_slide_index = a[2]
        self.sample_thickness = a[3]
        self.top_slide_thickness = a[4]
        self.bot_slide_thickness = a[5]

    def as_c_array(self):
        """Represent class as C array."""
        pyarr = self.as_array()
        return (ctypes.c_double * len(pyarr))(*pyarr)

    def __str__(self):
        """Represent class as string."""
        s = ""
        s += "sample index           = %.3f\n" % self.sample_index
        s += "top slide index        = %.3f\n" % self.top_slide_index
        s += "bottom slide index     = %.3f\n" % self.bot_slide_index

        s += "sample thickness       = %.1f mm\n" % self.sample_thickness
        s += "top slide thickness    = %.1f mm\n" % self.top_slide_thickness
        s += "bottom slide thickness = %.1f mm" % self.bot_slide_thickness
        return s


class CSphere:
    """Class for spheres."""

    def __init__(self):
        """Initialize class."""
        self.d_sphere = 8.0 * 25.4
        self.d_sample = 1.0 * 25.4
        self.d_empty = 1.0 * 25.4
        self.d_detector = 0.5 * 25.4
        self.refl_wall = 0.98
        self.refl_detector = 0.05

    def init_from_array(self, a):
        """Initialize with an array."""
        self.d_sphere = a[0]
        self.d_sample = a[1]
        self.d_empty = a[2]
        self.d_detector = a[3]
        self.refl_wall = a[4]
        self.refl_detector = a[5]

    def as_array(self):
        """Representation class as an array."""
        return [
            self.d_sphere,
            self.d_sample,
            self.d_empty,
            self.d_detector,
            self.refl_wall,
            self.refl_detector,
        ]

    def as_c_array(self):
        """Represent class as C array."""
        pyarr = self.as_array()
        return (ctypes.c_double * len(pyarr))(*pyarr)

    def __str__(self):
        """Represent class as string."""
        s = ""
        s += "sphere diameter        = %.1f mm\n" % self.d_sphere
        s += "sample port diameter   = %.1f mm\n" % self.d_sample
        s += "empty port diameter = %.1f mm\n" % self.d_empty
        s += "detector port diameter = %.1f mm\n" % self.d_detector
        s += "wall reflectivity      = %.3f\n" % self.refl_wall
        s += "detector reflectivity  = %.3f" % self.refl_detector
        return s


class CIllumination:
    """Class for illumination."""

    def __init__(self):
        """Initialize class."""
        self.beam_diameter = 5  # mm
        self.specular_reflection_excluded = 0
        self.direct_transmission_excluded = 0
        self.diffuse_illumination = 0
        self.num_spheres = 0
        self.lambda0 = 632.8  # nm      (unused, tracks wavelength)

    def init_from_array(self, a):
        """Initialize with an array."""
        self.beam_diameter = a[0]
        self.specular_reflection_excluded = a[1]
        self.direct_transmission_excluded = a[2]
        self.diffuse_illumination = a[3]
        self.lambda0 = a[4]
        self.num_spheres = a[5]

    def as_array(self):
        """Representation class as an array."""
        return [
            self.beam_diameter,
            self.specular_reflection_excluded,
            self.direct_transmission_excluded,
            self.diffuse_illumination,
            self.lambda0,
            self.num_spheres,
        ]

    def as_c_array(self):
        """Represent class as C array."""
        pyarr = self.as_array()
        return (ctypes.c_double * len(pyarr))(*pyarr)

    def __str__(self):
        """Represent class as string."""
        s = ""
        s += "diameter of beam       = %.1f mm\n" % self.beam_diameter
        s += "wavelength             = %.1f nm\n" % self.lambda0
        s += "exclude specular refl? = %s\n" % (bool(self.specular_reflection_excluded))
        s += "exclude direct trans?  = %s\n" % (bool(self.direct_transmission_excluded))
        s += "illumination diffuse?  = %s\n" % (bool(self.diffuse_illumination))
        s += "number of spheres      = %.0f" % self.num_spheres
        return s


class CAnalysis:
    """Class for analysis."""

    def __init__(self):
        """Initialize class."""
        self.quadrature_points = 8
        self.monte_carlo_runs = 0
        self.num_photons = 10000

    def init_from_array(self, a):
        """Initialize with an array."""
        self.quadrature_points = a[0]
        self.monte_carlo_runs = a[1]
        self.num_photons = a[2]

    def as_array(self):
        """Representation class as an array."""
        return [self.quadrature_points, self.monte_carlo_runs, self.num_photons]

    def as_c_array(self):
        """Represent class as C array."""
        pyarr = self.as_array()
        return (ctypes.c_double * len(pyarr))(*pyarr)

    def __str__(self):
        """Represent class as string."""
        s = ""
        s += "quadrature points      = %.0f\n" % self.quadrature_points
        s += "monte carlo runs       = %.0f\n" % self.monte_carlo_runs
        s += "number of photons      = %.0f" % self.num_photons
        return s


class CMeasurement:
    """Class for a measurement."""

    def __init__(self):
        """Initialize class."""
        self.standard_reflectance = 0.98
        self.reflectance = 0.5
        self.transmittance = 0.1
        self.unscattered_transmittance = 0

    def init_from_array(self, a):
        """Initialize with an array."""
        self.standard_reflectance = a[0]
        self.reflectance = a[1]
        self.transmittance = a[2]
        self.unscattered_transmittance = a[3]

    def as_array(self):
        """Representation class as an array."""
        return [
            self.standard_reflectance,
            self.reflectance,
            self.transmittance,
            self.unscattered_transmittance,
        ]

    def as_c_array(self):
        """Represent class as C array."""
        pyarr = self.as_array()
        return (ctypes.c_double * len(pyarr))(*pyarr)

    def __str__(self):
        """Represent class as string."""
        s = ""
        s += "standard reflectance   = %8.5f\n" % self.standard_reflectance
        s += "reflectance            = %8.5f\n" % self.reflectance
        s += "transmittance          = %8.5f\n" % self.transmittance
        s += "unscattered trans      = %8.5f\n" % self.unscattered_transmittance
        return s


libiad.Spheres_Inverse_RT2.argtypes = (
    ctypes.POINTER(ctypes.c_double),  # sample
    ctypes.POINTER(ctypes.c_double),  # illumination
    ctypes.POINTER(ctypes.c_double),  # Refl Sphere
    ctypes.POINTER(ctypes.c_double),  # Trans Sphere
    ctypes.POINTER(ctypes.c_double),  # analysis
    ctypes.POINTER(ctypes.c_double),  # measurement
    ctypes.POINTER(ctypes.c_double),  # a
    ctypes.POINTER(ctypes.c_double),  # b
    ctypes.POINTER(ctypes.c_double),  # g
)


def _indent(s):
    return "    " + s.replace("\n", "\n    ")


class Experiment:
    """Class for experiments."""

    def __init__(self):
        """Initialize class."""
        self.illumination = CIllumination()
        self.sample = CSample()
        self.r_sphere = CSphere()
        self.t_sphere = CSphere()
        self.analysis = CAnalysis()
        self.measurement = CMeasurement()

    def __str__(self):
        """Represent class as string."""
        s = "e.sample.\n"
        s += _indent(self.sample.__str__())
        s += "\n\ne.illumination.\n"
        s += _indent(self.illumination.__str__())
        s += "\n\ne.r_sphere.\n"
        s += _indent(self.r_sphere.__str__())
        s += "\n\ne.t_sphere.\n"
        s += _indent(self.t_sphere.__str__())
        s += "\n\ne.analysis.\n"
        s += _indent(self.analysis.__str__())
        s += "\n\ne.measurement.\n"
        s += _indent(self.measurement.__str__())
        return s

    def invert(self):
        """Calculate optical properties than match experiment."""
        a_s = self.sample.as_c_array()
        a_i = self.illumination.as_c_array()
        a_r = self.r_sphere.as_c_array()
        a_t = self.t_sphere.as_c_array()
        a_a = self.analysis.as_c_array()
        a_m = self.measurement.as_c_array()

        a = ctypes.c_double()
        b = ctypes.c_double()
        g = ctypes.c_double()

        libiad.Spheres_Inverse_RT2(a_s, a_i, a_r, a_t, a_a, a_m, a, b, g)
        return [a.value, b.value, g.value]
