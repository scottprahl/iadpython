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

    import iadpython.start

    n=4
    sample = iadpython.start.Slab(a=0.9, b=10, g=0.9, n=1.5)
    method = iadpython.start.Method(slab)
    r, t = iad.start.init_layer(slab, method)
    print(r)
    print(t)

"""

import numpy as np
import scipy.linalg
import iadpython.redistribution

__all__ = ('print_matrix',
           'zero_layer',
           'starting_thickness',
           'igi',
           'diamond',
           'thinnest_layer',
           'boundary_layer',
           )


def print_matrix(a, sample):
    """Print matrix and sums."""
    n = sample.quad_pts

    #header line
    print("%9.5f" % 0.0, end='')
    for i in range(n):
        print("%9.5f" % sample.nu[i], end='')
    print("     flux")

    #contents + row fluxes
    tflux = 0.0
    for i in range(n):
        print("%9.5f" % sample.nu[i], end='')
        for j in range(n):
            if a[i, j] < -10 or a[i, j] > 10:
                print("    *****", end='')
            else:
                print("%9.5f" % a[i, j], end='')
        flux = 0.0
        for j in range(n):
            flux += a[i, j] * sample.twonuw[j]
        print("%9.5f" % flux)
        tflux += flux * sample.twonuw[i]

    #column fluxes
    print("%9s" % "flux   ", end='')
    for i in range(n):
        flux = 0.0
        for j in range(n):
            flux += a[j, i] * sample.twonuw[j]
        print("%9.5f" % flux, end='')
    print("%9.5f\n" % tflux)

    #line of asterisks
    for i in range(n+1):
        print("*********", end='')
    print("\n")


def zero_layer(sample):
    """
    Create R and T matrices for thinnest_layer with zero thickness.

    Need to include quadrature normalization so R+T=1.
    """
    n = sample.quad_pts
    r = np.zeros([n, n])
    t = np.identity(n) / sample.twonuw

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
    |1e-4| for diamond initialization and less than
    |1e-8| for infinitesimal generator initialization assuming
    about 14 significant digits of accuracy.

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
    if sample.b_thinnest is None:
        sample.b_thinnest = starting_thickness(sample)

    n = sample.quad_pts
    d = sample.b_thinnest

    hp, hm = iadpython.redistribution.hg_legendre(sample)

    temp = sample.a_delta_M() * d / 4 / sample.nu
    R = temp * (hm / sample.nu).T
    T = temp * (hp / sample.nu).T
    T += (1 - d/sample.nu)/sample.twonuw * np.identity(n)

    return R, T

def diamond(sample):
    """
    Diamond initialization.

    The diamond method uses the same `r_hat` and `t_hat` as was used for infinitesimal
    generator method.  Division by `2*nu_j*w_j` is not needed until the
    final values for `R` and `T` are formed.
    """
    if sample.b_thinnest is None:
        sample.b_thinnest = starting_thickness(sample)

    n = sample.quad_pts
    d = sample.b_thinnest
    I = np.identity(n)

    hp, hm = iadpython.redistribution.hg_legendre(sample)

    temp = sample.a_delta_M() * d * sample.weight/ 4
    r_hat = temp * (hm / sample.nu).T
    t_hat = d/(2*sample.nu) * I - temp * (hp / sample.nu).T

    C = np.linalg.solve((I+t_hat).T, r_hat.T)
    G = 0.5 * (I + t_hat - C.T @ r_hat).T
    lu, piv = scipy.linalg.lu_factor(G)

    D = 1/sample.twonuw * (C * sample.twonuw).T
    R = scipy.linalg.lu_solve((lu, piv), D).T/sample.twonuw
    T = scipy.linalg.lu_solve((lu, piv), I).T/sample.twonuw
    T -= I.T/sample.twonuw
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
    Create reflection and transmission matrices for a boundary_layer.

    If 'boundary_layer=="top"' then the arrays returned are for the top
    surface and the labels are as expected i.e. 'T01' is the reflection for
    light from air passing to the slab. Otherwise the calculations are made
    for the bottom surface and the labels are backwards i.e. 'T01 == T32'
    and 'T10 == T23', where 0 is the first air slide surface, 1 is the
    slide/slab surface, 2 is the second slide/slab surface, and 3 is the
    bottom slide/air surface
    Args:
        s: slab
        method: method
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
        if nu_outside != 0:
            r, t = iadpython.fresnel.specular_nu_RT(s.n_above, s.n, s.n_below,
                                                    s.b_above, s.b, s.b_below, nu_outside)
            uru += s.twonuw[i] * r
            utu += s.twonuw[i] * t

    ur1, ut1 = iadpython.fresnel.specular_nu_RT(s.n_above, s.n, s.n_below,
                                                s.b_above, s.b, s.b_below, s.nu_0)

    uru *= s.n**2
    utu *= s.n**2
    return ur1, ut1, uru, utu
