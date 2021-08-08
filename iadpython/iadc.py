# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=bare-except

"""
Module for direct and inverse adding-doubling calculations.

    import iadpython as iad

    a = 0.9
    b = 10
    g = 0.9
    n_top = 1.5
    n_bot = 1.5
    ur1, ut1, uru, utu = iad.rt(n_top, n_bot, a, b, g)
    print("total reflected   light for normal incidence %.5f" % UR1)
    print("total transmitted light for normal incidence %.5f" % UT1)
    print("total reflected   light for diffuse incidence %.5f" % URU)
    print("total transmitted light for diffuse incidence %.5f" % UTU)
"""

import sys
import ctypes
import ctypes.util
import numpy as np

__all__ = ('basic_rt',
           'basic_rt_unscattered',
           'basic_rt_cone',
           'basic_rt_oblique',
           'basic_rt_inverse',
           'rt',
           'rt_unscattered',
           'rt_cone',
           'rt_oblique',
           'rt_inverse'
           )

libiad_path = ctypes.util.find_library("libiad")

if not libiad_path:
    print("Unable to find the libiad library.")
    print("     macOS   = 'libiad.dylib'")
    print("     unix    = 'libiad.so'")
    print("     Windows = 'libiad.dll'")
    print("Paths searched:")
    for p in sys.path:
        print("    ", p)
    sys.exit()

try:
    libiad = ctypes.CDLL(libiad_path)
except OSError:
    print("Unable to load the libiad library")
    print("Sorry")
    sys.exit()

libiad.ez_RT.argtypes = (ctypes.c_int,         # n quadrature points
                         ctypes.c_double,      # slab index of refraction
                         ctypes.c_double,      # top slide index of refraction
                         ctypes.c_double,      # bottom slide index of refraction
                         ctypes.c_double,      # albedo mus/(mus+mua)
                         ctypes.c_double,      # optical thickness d*(mua+mus)
                         ctypes.c_double,      # scattering anisotropy g
                         ctypes.POINTER(ctypes.c_double),   # UR1
                         ctypes.POINTER(ctypes.c_double),   # UT1
                         ctypes.POINTER(ctypes.c_double),   # URU
                         ctypes.POINTER(ctypes.c_double)    # UTU
                         )


def basic_rt(n, nslab, ntop, nbot, a, b, g):
    """
    Calculate the total reflection and transmission for a turbid slab.

    basic_rt() returns [UR1,UT1,URU,UTU] for a slab optionally
    bounded by glass slides.  The slab is characterized by an albedo a, an
    optical thickness b, a scattering anisotropy g, and an index of refraction nslab.

    The top glass slides have an index of refraction ntop and the bottom slide has
    an index nbot.  If there are no glass slides, set ntop and nbottom to 1.

    n is the number of quadrature angles (more angles give better accuracy but slow
    the calculation by n**3, n must be a multiple of 4, typically 16 is good for forward
    calculations and 4 is appropriate for inverse calculations).

    All input parameters must be scalars.

    UR1 is the total reflection for normally incident collimated light.
    UT1 is the total transmission for normally incident collimated light.
    URU is the total reflection for diffuse incident light.
    UTU is the total transmission for diffuse incident light.

    Args:
        n: number of points in quadrature (multiple of 4)
        nslab: index of refraction of the slab
        ntop: index of refraction of the top slide
        nbot: index of refraction of the bottom slide
        a: single scattering albedo of the slab
        b: optical thickness of the slab
        g: anisotropy of single scattering
    Returns
        [UR1, UT1, URU, UTU]
    """
    ur1 = ctypes.c_double()
    ut1 = ctypes.c_double()
    uru = ctypes.c_double()
    utu = ctypes.c_double()
    libiad.ez_RT(n, nslab, ntop, nbot, a, b, g, ur1, ut1, uru, utu)
    return ur1.value, ut1.value, uru.value, utu.value


libiad.ez_RT_unscattered.argtypes = (
    ctypes.c_int,         # n quadrature points
    ctypes.c_double,      # slab index of refraction
    ctypes.c_double,      # top slide index of refraction
    ctypes.c_double,      # bottom slide index of refraction
    ctypes.c_double,      # albedo mus/(mus+mua)
    ctypes.c_double,      # optical thickness d*(mua+mus)
    ctypes.c_double,      # scattering anisotropy g
    ctypes.POINTER(ctypes.c_double),   # UR1
    ctypes.POINTER(ctypes.c_double),   # UT1
    ctypes.POINTER(ctypes.c_double),   # URU
    ctypes.POINTER(ctypes.c_double)    # UTU
)


def basic_rt_unscattered(n, nslab, ntop, nbot, a, b, g):
    """
    Calculate the unscattered reflection and transmission for a turbid slab.

    basic_rt_unscattered(n,nslab,ntop,nbot,a,b,g) returns the unscattered
    light for normal and diffuse incidence [UR1,UT1,URU,UTU] for a slab
    bounded by glass slides.  The slab is characterized by an albedo a, an
    optical thickness b, a scattering anisotropy g, and an index of refraction nslab.
    The top glass slides have an index of refraction ntop and the bottom slide has
    an index nbot.  If there are no glass slides, set ntop and nbottom to 1.

    n is the number of quadrature angles (more angles give better accuracy but slow
    the calculation by n**3, n must be a multiple of 4, typically 16 is good for forward
    calculations and 4 is appropriate for inverse calculations).

    All parameters must be scalars.

    UR1 is the unscattered reflection for normally incident collimated light.
    UT1 is the unscattered transmission for normally incident collimated light.
    URU is the unscattered reflection for diffuse incident light.
    UTU is the unscattered transmission for diffuse incident light.
    """
    ur1 = ctypes.c_double()
    ut1 = ctypes.c_double()
    uru = ctypes.c_double()
    utu = ctypes.c_double()
    libiad.ez_RT_unscattered(n, nslab, ntop, nbot, a, b, g, ur1, ut1, uru, utu)
    return ur1.value, ut1.value, uru.value, utu.value


libiad.ez_RT_Cone.argtypes = (
    ctypes.c_int,         # n quadrature points
    ctypes.c_double,      # slab index of refraction
    ctypes.c_double,      # top slide index of refraction
    ctypes.c_double,      # bottom slide index of refraction
    ctypes.c_double,      # albedo mus/(mus+mua)
    ctypes.c_double,      # optical thickness d*(mua+mus)
    ctypes.c_double,      # scattering anisotropy g
    ctypes.c_double,      # cosine of cone angle
    ctypes.POINTER(ctypes.c_double),   # UR1
    ctypes.POINTER(ctypes.c_double),   # UT1
    ctypes.POINTER(ctypes.c_double),   # URU
    ctypes.POINTER(ctypes.c_double)    # UTU
)


def basic_rt_cone(n, nslab, ntop, nbot, a, b, g, cos_cone_angle):
    """
    Calculate reflection and transmission for a turbid slab exiting within a cone.

    basic_rt_cone(n,nslab,ntop,nbot,a,b,g,cos_cone_angle) assumes normally
    incident or uniformly diffuse incident light and returns the total reflected
    or transmitted light that exits within a cone. The cosine of the cone angle
    is given by cos_cone.  The returned values are [UR1,UT1,URU,UTU].

    The slab is characterized by an albedo a, an
    optical thickness b, a scattering anisotropy g, and an index of refraction nslab.
    The top glass slides have an index of refraction ntop and the bottom slide has
    an index nbot.  If there are no glass slides, set ntop and nbottom to 1.

    n is the number of quadrature angles (more angles give better accuracy but slow
    the calculation by n**3, n must be a multiple of 4, typically 16 is good for forward
    calculations and 4 is appropriate for inverse calculations).

    All parameters must be scalars.

    UR1 is the total reflection within a cone for normally incident collimated light.
    UT1 is the total transmission within a cone for normally incident collimated light.
    URU is the total reflection within a cone for diffuse incident light.
    UTU is the total transmission within a cone for diffuse incident light.
    """
    ur1 = ctypes.c_double()
    ut1 = ctypes.c_double()
    uru = ctypes.c_double()
    utu = ctypes.c_double()
    libiad.ez_RT_Cone(n, nslab, ntop, nbot, a, b, g,
                      cos_cone_angle, ur1, ut1, uru, utu)
    return ur1.value, ut1.value, uru.value, utu.value


libiad.ez_RT_Oblique.argtypes = (
    ctypes.c_int,         # n quadrature points
    ctypes.c_double,      # slab index of refraction
    ctypes.c_double,      # top slide index of refraction
    ctypes.c_double,      # bottom slide index of refraction
    ctypes.c_double,      # albedo mus/(mus+mua)
    ctypes.c_double,      # optical thickness d*(mua+mus)
    ctypes.c_double,      # scattering anisotropy g
    ctypes.c_double,      # cosine of oblique angle
    ctypes.POINTER(ctypes.c_double),   # UR1
    ctypes.POINTER(ctypes.c_double),   # UT1
    ctypes.POINTER(ctypes.c_double),   # URU
    ctypes.POINTER(ctypes.c_double)    # UTU
)


def basic_rt_oblique(n, nslab, ntop, nbot, a, b, g, cos_oblique):
    """
    Calculate reflection and transmission for light incident at a oblique angle.

    basic_rt_oblique(n,nslab,ntop,nbot,a,b,g,cos_oblique) returns the total R and T
    for light incident at an oblique angle or incident withn a cone.  The cosine
    of the oblique angle (or the cone) is cos_oblique. The returned values are
    [URx,UTx,URU,UTU].

    The slab is characterized by an albedo a, an optical thickness b, a scattering
    anisotropy g, and an index of refraction nslab. The top glass slide have an
    index of refraction of ntop and the bottom slide has an index nbot.  If there
    are no glass slides, set ntop and nbottom to 1.

    n is the number of quadrature angles (more angles give better accuracy but slow
    the calculation by n**3, n must be a multiple of 4, typically 16 is good for forward
    calculations and 4 is appropriate for inverse calculations).

    All parameters must be scalars.

    URx is the total reflection for obliquely incident collimated light.
    UTx is the total transmission for obliquely incident collimated light.
    URU is the total reflection for diffuse light incident within a cone.
    UTU is the total transmission for diffuse light incident within a cone.
    """
    ur1 = ctypes.c_double()
    ut1 = ctypes.c_double()
    uru = ctypes.c_double()
    utu = ctypes.c_double()
    libiad.ez_RT_Oblique(n, nslab, ntop, nbot, a, b, g,
                         cos_oblique, ur1, ut1, uru, utu)
    return ur1.value, ut1.value, uru.value, utu.value


libiad.ez_Inverse_RT.argtypes = (
    ctypes.c_double,      # slab index of refraction
    ctypes.c_double,      # slide index of refraction
    ctypes.c_double,      # UR1
    ctypes.c_double,      # UT1
    ctypes.c_double,      # unscattered transmission
    ctypes.POINTER(ctypes.c_double),   # a
    ctypes.POINTER(ctypes.c_double),   # b
    ctypes.POINTER(ctypes.c_double),   # g
    ctypes.POINTER(ctypes.c_int)       # error
)


def basic_rt_inverse(nslab, nslide, ur1, ut1, tc):
    """
    Calculate optical properties given reflection and transmission values.

    basic_rt_inverse(nslab, nslide, ur1, ut1, tc) finds [a,b,g] for a slab
    with total reflectance ur1, total transmission ut1, unscattered transmission Tc.
    The index of refraction of the slab is nslab, the index of refraction of the
    top and bottom slides is nslide.

    All parameters must be scalars
    """
    a = ctypes.c_double()
    b = ctypes.c_double()
    g = ctypes.c_double()
    error = ctypes.c_int()
    libiad.ez_Inverse_RT(nslab, nslide, ur1, ut1, tc, a, b, g, error)
    return a.value, b.value, g.value, error.value


def rt(nslab, nslide, a, b, g):
    """
    Calculate the total reflection and transmission for a turbid slab.

    rt(nslab,ntop,nbot,a,b,g) returns [UR1,UT1,URU,UTU] for a slab optionally
    bounded by glass slides.  The slab is characterized by an albedo a, an
    optical thickness b, a scattering anisotropy g, and an index of refraction nslab.
    The top glass slides have an index of refraction ntop and the bottom slide has
    an index nbot.  If there are no glass slides, set ntop and nbottom to 1.

    a, b, and g can be scalars or arrays.

    UR1 is the total reflection for normally incident collimated light.
    UT1 is the total reflection for normally incident collimated light.
    URU is the total reflection for diffuse incident light.
    UTU is the total reflection for diffuse incident light.
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
        raise RuntimeError('rt: a and b arrays must be same length')

    if len_a and len_g and len_a != len_g:
        raise RuntimeError('rt: a and g arrays must be same length')

    if len_b and len_g and len_b != len_g:
        raise RuntimeError('rt: b and g arrays must be same length')

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

        ur1[i], ut1[i], uru[i], utu[i] = basic_rt(
            N_QUADRATURE, nslab, nslide, nslide, aa, bb, gg)

    return ur1, ut1, uru, utu


def rt_unscattered(nslab, nslide, a, b, g):
    """
    Calculate the unscattered reflection and transmission for a turbid slab.

    rt_unscattered(nslab,nslide,a,b,g) returns the unscattered portion of
    light for normal and diffuse incidence [UR1,UT1,URU,UTU] for a slab
    bounded by glass slides.  The slab is characterized by an albedo a, an
    optical thickness b, a scattering anisotropy g, and an index of refraction nslab.
    The top and bottom glass slides have index nslide. If there are no glass slides,
    set ntop and nbottom to 1.

    a,b,g may be scalars or arrays.

    UR1 is the unscattered reflection for normally incident collimated light
    UT1 is the unscattered reflection for normally incident collimated light
    URU is the unscattered reflection for diffuse incident light
    UTU is the unscattered reflection for diffuse incident light
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
        raise RuntimeError(
            'rt_unscattered: a and b arrays must be same length')

    if len_a and len_g and len_a != len_g:
        raise RuntimeError(
            'rt_unscattered: a and g arrays must be same length')

    if len_b and len_g and len_b != len_g:
        raise RuntimeError(
            'rt_unscattered: b and g arrays must be same length')

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

        ur1[i], ut1[i], uru[i], utu[i] = basic_rt_unscattered(
            N_QUADRATURE, nslab, nslide, nslide, aa, bb, gg)

    return ur1, ut1, uru, utu


def rt_cone(nslab, nslide, a, b, g, cos_cone):
    """
    Calculate reflection and transmission for a turbid slab exiting within a cone.

    rt_cone(nslab,nslide,a,b,g,cos_cone) assumes normally
    incident or uniformly diffuse incident light and returns the total reflected
    or transmitted light that exits within a cone. The cosine of the cone angle
    is cos_cone.  The returned values are [UR1,UT1,URU,UTU].

    The slab is characterized by an albedo a, an
    optical thickness b, a scattering anisotropy g, and an index of refraction
    nslab. The top and bottom glass slides have an index of refraction nslide.
    If there are no glass slides, set nslide to 1.

    a, b, g, and cos_cone may be scalars or arrays.

    UR1 is the total reflection within a cone for normally incident collimated light.
    UT1 is the total transmission within a cone for normally incident collimated light.
    URU is the total reflection within a cone for diffuse incident light.
    UTU is the total transmission within a cone for diffuse incident light.
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
        raise RuntimeError('rt_cone: a and b arrays must be same length')

    if len_a and len_g and len_a != len_g:
        raise RuntimeError('rt_cone: a and g arrays must be same length')

    if len_a and len_mu and len_a != len_mu:
        raise RuntimeError('rt_cone: a and mu arrays must be same length')

    if len_b and len_g and len_b != len_g:
        raise RuntimeError('rt_cone: b and g arrays must be same length')

    if len_b and len_mu and len_b != len_mu:
        raise RuntimeError('rt_cone: b and mu arrays must be same length')

    if len_g and len_mu and len_g != len_mu:
        raise RuntimeError('rt_cone: g and mu arrays must be same length')

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

        ur1[i], ut1[i], uru[i], utu[i] = basic_rt_cone(
            N_QUADRATURE, nslab, nslide, nslide, aa, bb, gg, mu)

    return ur1, ut1, uru, utu


def rt_oblique(nslab, nslide, a, b, g, cos_oblique):
    """
    Calculate reflection and transmission for light incident at a oblique angle.

    rt_oblique(n,nslide,a,b,g,cos_oblique) returns the total R and T
    for light incident at an oblique angle or incident withn a cone.  The cosine
    of the oblique angle (or the cone) is cos_oblique. The returned values are
    [URx,UTx,URU,UTU].

    The slab is characterized by an albedo a, an optical thickness b, a scattering
    anisotropy g, and an index of refraction nslab. The top and bottom glass slides
    have an index of refraction nslide. If there are no glass slides, set nslide to 1.

    a, b, g, and cos_oblique may be scalars or arrays.

    URx is the total reflection for obliquely incident collimated light.
    UTx is the total transmission for obliquely incident collimated light.
    URU is the total reflection for diffuse light incident within a cone.
    UTU is the total transmission for diffuse light incident within a cone.
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
        raise RuntimeError('rt_oblique: a and b arrays must be same length')

    if len_a and len_g and len_a != len_g:
        raise RuntimeError('rt_oblique: a and g arrays must be same length')

    if len_a and len_mu and len_a != len_mu:
        raise RuntimeError('rt_oblique: a and mu arrays must be same length')

    if len_b and len_g and len_b != len_g:
        raise RuntimeError('rt_oblique: b and g arrays must be same length')

    if len_b and len_mu and len_b != len_mu:
        raise RuntimeError('rt_oblique: b and mu arrays must be same length')

    if len_g and len_mu and len_g != len_mu:
        raise RuntimeError('rt_oblique: g and mu arrays must be same length')

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

        ur1[i], ut1[i], uru[i], utu[i] = basic_rt_oblique(
            N_QUADRATURE, nslab, nslide, nslide, aa, bb, gg, mu)

    return ur1, ut1, uru, utu


def rt_inverse(nslab, nslide, ur1, ut1, t_unscattered):
    """
    Calculate [a,b,g] for a slab.

    with total reflectance ur1, total transmission ut1, unscattered transmission t_unscattered.
    The index of refraction of the slab is nslab, the index of refraction of the
    top and bottom slides is nslide.

    ur1, ut1, and t_unscattered may be scalars or arrays.
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
        raise RuntimeError(
            'inverse_rt: ur1 and ut1 arrays must be same length')

    if len_r1 and len_tc and len_r1 != len_tc:
        raise RuntimeError('inverse_rt: ur1 and tc arrays must be same length')

    if len_t1 and len_tc and len_t1 != len_tc:
        raise RuntimeError('inverse_rt: t1 and tc arrays must be same length')

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

        a[i], b[i], g[i], error[i] = basic_rt_inverse(
            nslab, nslide, r1, t1, tc)

    return a, b, g, error


class CSample():
    def __init__(self):
        self.sample_index = 1.4
        self.top_slide_index = 1.5
        self.bot_slide_index = 1.5
        self.sample_thickness = 1    # mm
        self.top_slide_thickness = 1    # mm
        self.bot_slide_thickness = 1    # mm

    def as_array(self):
        return [self.sample_index, self.top_slide_index, self.bot_slide_index,
                self.sample_thickness, self.top_slide_thickness, self.bot_slide_thickness]

    def init_from_array(self, a):
        self.sample_index = a[0]
        self.top_slide_index = a[1]
        self.bot_slide_index = a[2]
        self.sample_thickness = a[3]
        self.top_slide_thickness = a[4]
        self.bot_slide_thickness = a[5]

    def as_c_array(self):
        pyarr = self.as_array()
        return (ctypes.c_double * len(pyarr))(*pyarr)

    def __str__(self):
        s = ""
        s += "sample index           = %.3f\n" % self.sample_index
        s += "top slide index        = %.3f\n" % self.top_slide_index
        s += "bottom slide index     = %.3f\n" % self.bot_slide_index

        s += "sample thickness       = %.1f mm\n" % self.sample_thickness
        s += "top slide thickness    = %.1f mm\n" % self.top_slide_thickness
        s += "bottom slide thickness = %.1f mm" % self.bot_slide_thickness
        return s


class CSphere():
    def __init__(self):
        self.d_sphere = 8.0 * 25.4
        self.d_sample = 1.0 * 25.4
        self.d_entrance = 1.0 * 25.4
        self.d_detector = 0.5 * 25.4
        self.refl_wall = 0.98
        self.refl_detector = 0.05

    def init_from_array(self, a):
        self.d_sphere = a[0]
        self.d_sample = a[1]
        self.d_entrance = a[2]
        self.d_detector = a[3]
        self.refl_wall = a[4]
        self.refl_detector = a[5]

    def as_array(self):
        return [self.d_sphere, self.d_sample, self.d_entrance,
                self.d_detector, self.refl_wall, self.refl_detector]

    def as_c_array(self):
        pyarr = self.as_array()
        return (ctypes.c_double * len(pyarr))(*pyarr)

    def __str__(self):
        s = ""
        s += "sphere diameter        = %.1f mm\n" % self.d_sphere
        s += "sample port diameter   = %.1f mm\n" % self.d_sample
        s += "entrance port diameter = %.1f mm\n" % self.d_entrance
        s += "detector port diameter = %.1f mm\n" % self.d_detector
        s += "wall reflectivity      = %.3f\n" % self.refl_wall
        s += "detector reflectivity  = %.3f" % self.refl_detector
        return s


class CIllumination():
    def __init__(self):
        self.beam_diameter = 5  # mm
        self.specular_reflection_excluded = 0
        self.direct_transmission_excluded = 0
        self.diffuse_illumination = 0
        self.num_spheres = 0
        self.lambda0 = 632.8  # nm      (unused, tracks wavelength)

    def init_from_array(self, a):
        self.beam_diameter = a[0]
        self.specular_reflection_excluded = a[1]
        self.direct_transmission_excluded = a[2]
        self.diffuse_illumination = a[3]
        self.lambda0 = a[4]
        self.num_spheres = a[5]

    def as_array(self):
        return [self.beam_diameter, self.specular_reflection_excluded,
                self.direct_transmission_excluded, self.diffuse_illumination,
                self.lambda0, self.num_spheres]

    def as_c_array(self):
        pyarr = self.as_array()
        return (ctypes.c_double * len(pyarr))(*pyarr)

    def __str__(self):
        s = ""
        s += "diameter of beam       = %.1f mm\n" % self.beam_diameter
        s += "wavelength             = %.1f nm\n" % self.lambda0
        s += "exclude specular refl? = %s\n" % (
            bool(self.specular_reflection_excluded))
        s += "exclude direct trans?  = %s\n" % (
            bool(self.direct_transmission_excluded))
        s += "illumination diffuse?  = %s\n" % (
            bool(self.diffuse_illumination))
        s += "number of spheres      = %.0f" % self.num_spheres
        return s


class CAnalysis():
    def __init__(self):
        self.quadrature_points = 8
        self.monte_carlo_runs = 0
        self.num_photons = 10000

    def init_from_array(self, a):
        self.quadrature_points = a[0]
        self.monte_carlo_runs = a[1]
        self.num_photons = a[2]

    def as_array(self):
        return [self.quadrature_points, self.monte_carlo_runs, self.num_photons]

    def as_c_array(self):
        pyarr = self.as_array()
        return (ctypes.c_double * len(pyarr))(*pyarr)

    def __str__(self):
        s = ""
        s += "quadrature points      = %.0f\n" % self.quadrature_points
        s += "monte carlo runs       = %.0f\n" % self.monte_carlo_runs
        s += "number of photons      = %.0f" % self.num_photons
        return s


class CMeasurement():
    def __init__(self):
        self.standard_reflectance = 0.98
        self.reflectance = 0.5
        self.transmittance = 0.1
        self.unscattered_transmittance = 0

    def init_from_array(self, a):
        self.standard_reflectance = a[0]
        self.reflectance = a[1]
        self.transmittance = a[2]
        self.unscattered_transmittance = a[3]

    def as_array(self):
        return [self.standard_reflectance, self.reflectance, self.transmittance,
                self.unscattered_transmittance]

    def as_c_array(self):
        pyarr = self.as_array()
        return (ctypes.c_double * len(pyarr))(*pyarr)

    def __str__(self):
        s = ""
        s += "standard reflectance   = %8.5f\n" % self.standard_reflectance
        s += "reflectance            = %8.5f\n" % self.reflectance
        s += "transmittance          = %8.5f\n" % self.transmittance
        s += "unscattered trans      = %8.5f\n" % self.unscattered_transmittance
        return s

# void Spheres_Inverse_RT2(double*sample,
# double*illumination,
# double*sphere_r,
# double*sphere_t,
# double*analysis,
# double*measurement,
# double*a,
# double*b,
# double*g)


libiad.Spheres_Inverse_RT2.argtypes = (
    ctypes.POINTER(ctypes.c_double),   # sample
    ctypes.POINTER(ctypes.c_double),   # illumination
    ctypes.POINTER(ctypes.c_double),   # Refl Sphere
    ctypes.POINTER(ctypes.c_double),   # Trans Sphere
    ctypes.POINTER(ctypes.c_double),   # analysis
    ctypes.POINTER(ctypes.c_double),   # measurement
    ctypes.POINTER(ctypes.c_double),   # a
    ctypes.POINTER(ctypes.c_double),   # b
    ctypes.POINTER(ctypes.c_double)    # g
)


def _indent(s):
    return '    ' + s.replace('\n', '\n    ')


class Experiment():
    def __init__(self):
        self.illumination = CIllumination()
        self.sample = CSample()
        self.r_sphere = CSphere()
        self.t_sphere = CSphere()
        self.analysis = CAnalysis()
        self.measurement = CMeasurement()

    def __str__(self):
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
