# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments

"""
Module for generating the starting thinnest_layer for adding-doubling.

The next two routines 'A_Add_Slide' and 'B_Add_Slide' are modifications of
the full addition algorithms for dissimilar layers.  They are optimized to
take advantage of the diagonal nature of the boundary_layer matrices.  There are
two algorithms below to facilitate adding slides below and above the sample.

The important point that must be remembered is that all the angles in
this program assume that the angles are those actually in the sample.
This allows angles greater that the critical angle to be used.
Everything is fine as long as the index of refraction of the incident
medium is 1.0.  If this is not the case then the angle inside the medium
must be figured out.
"
Two types of starting methods are possible.

    import iadpython as iad

    s = iad.Sample(a=0.9, b=10, g=0.9, n=1.5)
    r, t = iad.start.init_layer(s)
    print(r)
    print(t)

"""

import numpy as np
import scipy.linalg
import iadpython.redistribution

__all__ = ('zero_layer',
           'starting_thickness',
           'igi',
           'diamond',
           'thinnest_layer',
           'boundary_layer',
           'boundary_matrices',
           'unscattered_rt'
           )


def zero_layer(sample):
    """
    Create R and T matrices for thinnest_layer with zero thickness.

    Need to include quadrature normalization so R+T=1.
    """
    if sample.twonuw is None:
        sample.update_quadrature()
    t = np.diagflat(1 / sample.twonuw)
    r = np.zeros_like(t)
    return r, t


def starting_thickness(sample):
    """
    Find the best starting thickness to start the doubling process.

    The criterion is based on an assessment of the (1) round-off error,
    (2) the angular initialization error, and (3) the thickness
    initialization error.  Wiscombe concluded that an optimal
    starting thickness depends on the smallest quadrature slab.nu, and
    recommends that when either the infinitesimal generator or diamond
    initialization methods are used then the initial thickness is optimal
    when type 2 and 3 errors are comparable, or when d ≈ µ.

    Note that round-off is important when the starting thickness is less than
    1e-4 for diamond initialization and less than 1e-8 for infinitesimal
    generator initialization assuming about 14 significant digits of accuracy.

    Since the final thickness is determined by repeated doubling, the
    starting thickness is found by dividing by 2 until the starting thickness is
    less than 'mu'.  Also we make checks for a thinnest_layer with zero thickness
    and one that infinitely thick.
    """
    if sample.nu is None:
        sample.update_quadrature()

    if sample.b <= 0:
        return 0.0

    if sample.b == np.inf:
        return sample.nu[0] / 2.0

    dd = sample.b_delta_M()
    while dd > sample.nu[0]:
        dd /= 2

    return dd


def igi(sample):
    """
    Infinitesmal Generator Initialization.

    igi_start() generates the starting matrix with the inifinitesimal generator method.
    The accuracy is O(d) and assumes that the average irradiance upwards is
    equal to that travelling downwards at the top and the average radiance upwards
    equals that moving upwards from the bottom.

    Ultimately the formulas for the reflection matrix is

    R_ij = a*d/(4*nu_i*nu_j) hpm_ij

    and

    T_ij = a*d/(4*nu_i*nu_j) hpp_ij + delta_ij*(1-d/nu_i)/(2*nu_i w_i)
    """
    if sample.nu is None:
        sample.update_quadrature()

    if sample.b_thinnest is None:
        sample.b_thinnest = starting_thickness(sample)

    n = sample.quad_pts
    d = sample.b_thinnest

    if sample.hp is None:
        sample.hp, sample.hm = iadpython.redistribution.hg_legendre(sample)

    temp = sample.a_delta_M() * d / 4 / sample.nu
    R = temp * (sample.hm / sample.nu).T
    T = temp * (sample.hp / sample.nu).T
    T += (1 - d / sample.nu) / sample.twonuw * np.identity(n)

    return R, T


def diamond(sample):
    """
    Diamond initialization.

    The diamond method uses the same `r_hat` and `t_hat` as was used for infinitesimal
    generator method.  Division by `2*nu_j*w_j` is not needed until the
    final values for `R` and `T` are formed.
    """
    if sample.nu is None:
        sample.update_quadrature()

    if sample.b_thinnest is None:
        sample.b_thinnest = starting_thickness(sample)

    n = sample.quad_pts
    d = sample.b_thinnest
    II = np.identity(n)

    if sample.hp is None:
        sample.hp, sample.hm = iadpython.redistribution.hg_legendre(sample)

    w = sample.twonuw / sample.nu / 2
    temp = sample.a_delta_M() * d * w / 4
    r_hat = temp * (sample.hm / sample.nu).T
    t_hat = d / (2 * sample.nu) * II - temp * (sample.hp / sample.nu).T

    C = np.linalg.solve((II + t_hat).T, r_hat.T)
    G = 0.5 * (II + t_hat - C.T @ r_hat).T
    lu, piv = scipy.linalg.lu_factor(G)

    D = 1 / sample.twonuw * (C * sample.twonuw).T
    R = scipy.linalg.lu_solve((lu, piv), D).T / sample.twonuw
    T = scipy.linalg.lu_solve((lu, piv), II).T / sample.twonuw
    T -= II.T / sample.twonuw
    return R, T


def thinnest_layer(sample):
    """
    Reflection and transmission matrices for a thin thinnest_layer.

    The optimal method for calculating a thin thinnest_layer depends on the
    thinness of the thinnest_layer and the quadrature angles.  This chooses
    the best method using Wiscombe's criteria.
    """
    if sample.b <= 0:
        return zero_layer(sample)

    sample.b_thinnest = starting_thickness(sample)

    if sample.b_thinnest < 1e-4 or sample.b_thinnest < 0.09 * sample.nu[0]:
        return igi(sample)

    return diamond(sample)


def _boundary(sample, n_i, n_g, n_t, b):
    """
    Find matrix for R and T for air/glass/slab interface.

    The resulting matrix is a diagonal matrix that is represented as an
    array.

    The reflection matrix is the same entering or exiting the slab. The
    transmission matrices should differ by a factor of
    (n_slab/n_outside)**4, due to n**2 law of radiance, but there is some
    inconsistency in the program and if I use this principle then regular
    calculations for R and T don't work and the fluence calculations still
    don't work.  So punted and took all that code out.

    """
    if sample.nu is None:
        sample.update_quadrature()

    if n_i == 1.0:
        nu = iadpython.fresnel.cos_snell(n_t, sample.nu, n_i)
    else:
        nu = sample.nu
    r, t = iadpython.fresnel.absorbing_glass_RT(n_i, n_g, n_t, nu, b)
    r *= sample.twonuw
    return r, t


def boundary_layer(s, top=True):
    """
    Create reflection and transmission arrays for a boundary_layer.

    If 'boundary_layer=="top"' then the arrays returned are for the top
    surface and the labels are as expected i.e. 'T01' is the reflection for
    light from air passing to the slab. Otherwise the calculations are made
    for the bottom surface and the labels are backwards i.e. 'T01 == T32'
    and 'T10 == T23', where 0 is the first air slide surface, 1 is the
    slide/slab surface, 2 is the second slide/slab surface, and 3 is the
    bottom slide/air surface.

    Args:
        s: slab
        top: True if this is the top slide
    Returns:
        R01: reflection array from air to slab
        R10: reflection array from slab to air
        T01: transmission array from air to slab
        T10: transmission array from slab to air
    """
    if top:
        R01, T01 = _boundary(s, 1.0, s.n_above, s.n, s.b_above)
        R10, T10 = _boundary(s, s.n, s.n_above, 1.0, s.b_above)
    else:
        R10, T10 = _boundary(s, 1.0, s.n_below, s.n, s.b_below)
        R01, T01 = _boundary(s, s.n, s.n_below, 1.0, s.b_below)
    return R01, R10, T01, T10


def boundary_matrices(s, top=True):
    """
    Create reflection and transmission matrices for a boundary_layer.

    These can be treated like any other layer in the adding-doubling
    formalism.

    Args:
        s: slab
        top: True if this is the top boundary
    Returns:
        R01: reflection array from air to slab
        R10: reflection array from slab to air
        T01: transmission array from air to slab
        T10: transmission array from slab to air
    """
    R01, R10, T01, T10 = boundary_layer(s, top=top)
    rr01 = np.diagflat(R01 / s.twonuw**2)
    rr10 = np.diagflat(R10 / s.twonuw**2)
    tt01 = np.diagflat(T01 / s.twonuw)
    tt10 = np.diagflat(T10 / s.twonuw)

    return rr01, rr10, tt01, tt10


def unscattered_rt(s):
    """
    Unscattered reflection and transmission for a slide-slab-slide sandwich.

    The sample is characterized by the record 'slab'.  The total unscattered_layer
    reflection and transmission for oblique irradiance ('urx' and 'utx')
    together with their companions 'uru' and 'utu' for diffuse irradiance.
    The cosine of the incident angle is specified by 'slab.cos_angle'.

    The way that this routine calculates the diffuse unscattered_layer quantities
    based on the global quadrature angles previously set-up.  Consequently,
    these estimates are not exact.  In fact if 'n=4' then only two
    quadrature points will actually be used to figure out the diffuse
    reflection and transmission (assuming mismatched boundaries).

    This algorithm is pretty simple.  Since the quadrature angles are all
    chosen assuming points inside the medium, I must calculate the
    corresponding angle for light entering from the outside.  If the the
    cosine of this angle is greater than zero then the angle does not
    correspond to a direction in which light is totally internally
    reflected. For this ray, I find the unscattered_layer that would be reflected
    or transmitted from the slab.  I multiply this by the quadrature angle
    and weight 'twoaw[i]' to get the total diffuse reflectance and
    transmittance.

    Oh, yes.  The mysterious multiplication by a factor of 'n_slab*n_slab'
    is required to account for the n**2-law of radiance.
    """
    r01, t01 = iadpython.zero_layer(s)
    r10, t10 = iadpython.zero_layer(s)

    r01, t01 = iadpython.specular_nu_RT(s.n_above, s.n, s.n_below, s.b, s.nu,
                                        s.b_above, s.b_below)
    r10, t10 = iadpython.specular_nu_RT(s.n_below, s.n, s.n_above, s.b, s.nu,
                                        s.b_below, s.b_above)

    rr01 = np.diagflat(r01) / s.twonuw
    rr10 = np.diagflat(r10) / s.twonuw
    tt01 = np.diagflat(t01) / s.twonuw
    tt10 = np.diagflat(t10) / s.twonuw

    return rr01, rr10, tt01, tt10


def unscattered(s):
    """
    Unscattered reflection and transmission for a slide-slab-slide sandwich.

    The sample is characterized by the record 'slab'.  The total unscattered_layer
    reflection and transmission for oblique irradiance ('urx' and 'utx')
    together with their companions 'uru' and 'utu' for diffuse irradiance.
    The cosine of the incident angle is specified by 'slab.cos_angle'.

    The way that this routine calculates the diffuse unscattered_layer quantities
    based on the global quadrature angles previously set-up.  Consequently,
    these estimates are not exact.  In fact if 'n=4' then only two
    quadrature points will actually be used to figure out the diffuse
    reflection and transmission (assuming mismatched boundaries).

    This algorithm is pretty simple.  Since the quadrature angles are all
    chosen assuming points inside the medium, I must calculate the
    corresponding angle for light entering from the outside.  If the the
    cosine of this angle is greater than zero then the angle does not
    correspond to a direction in which light is totally internally
    reflected. For this ray, I find the unscattered_layer that would be reflected
    or transmitted from the slab.  I multiply this by the quadrature angle
    and weight 'twoaw[i]' to get the total diffuse reflectance and
    transmittance.

    Oh, yes.  The mysterious multiplication by a factor of 'n_slab*n_slab'
    is required to account for the n**2-law of radiance.
    """
    n = s.quad_pts

    uru = 0
    utu = 0

    for i in range(n):
        nu_outside = iadpython.fresnel.cos_snell(s.n, s.nu[i], 1.0)
        if nu_outside == 0:
            r, t = iadpython.fresnel.specular_nu_RT(s.n_above, s.n, s.n_below,
                                                    s.b_above, s.b, s.b_below, nu_outside)
            uru += s.twonuw[i] * r[i, i]
            utu += s.twonuw[i] * t[i, i]

    ur1, ut1 = iadpython.fresnel.specular_nu_RT(s.n_above, s.n, s.n_below,
                                                s.b_above, s.b, s.b_below, s.nu_0)

    uru *= s.n**2
    utu *= s.n**2
    return ur1, ut1, uru, utu
