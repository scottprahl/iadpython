# pylint: disable = invalid-name
# pylint: disable = too-many-arguments

"""
Boundary incorporation algorithms.

The next two routines 'A_Add_Slide' and 'B_Add_Slide' are modifications of
the full addition algorithms for dissimilar layers.  They are optimized to
take advantage of the diagonal nature of the boundary matrices.  There are
two algorithms below to facilitate adding slides below and above the sample.

The important point that must be remembered is that all the angles in
this program assume that the angles are those actually in the sample.
This allows angles greater that the critical angle to be used.
Everything is fine as long as the index of refraction of the incident
medium is 1.0.  If this is not the case then the angle inside the medium
must be figured out.
"""
import numpy as np
import iadpython.fresnel

__all__ = ('init_boundary',
           'boundary_RT',
           'add_top',
           'add_bottom',
           'add_slides',
           'sp_RT',
           )

def init_boundary(s, method, top=True):
    """
    Create reflection and transmission matrices for a boundary.

    If 'boundary=="top"' then the arrays returned are for the top
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
        R01, T01 = boundary_RT(1.0, s.n_above, s.n, s.b_above, method)
        R10, T10 = boundary_RT(s.n, s.n_above, 1.0, s.b_above, method)
    else:
        R10, T10 = boundary_RT(1.0, s.n_below, s.n, s.b_below, method)
        R01, T01 = boundary_RT(s.n, s.n_below, 1.0, s.b_below, method)
    return R01, R10, T01, T10


def boundary_RT(n_i, n_g, n_t, b, method):
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
    if n_i == 1.0:
        nu = iadpython.fresnel.cos_snell(n_t, method.nu, n_i)
    else:
        nu = method.nu
    r, t = iadpython.fresnel.absorbing_glass_RT(n_i, n_g, n_t, nu, b)
    r *= method.twonuw
    return r, t


def A_Add_Slide(R12, R21, T12, T21, R10, T01, method):
    """
    Find two matrices when slide is added to top of slab.

    Compute the resulting 'R20' and 'T02' matrices for a glass slide
    on top of an inhomogeneous layer characterized by 'R12', 'R21', 'T12',
    'T21' using:

    T_02=T_12 (E-R_10*R_12 )**-1 T_01

    R_20=T_12 (E-R_10*R_12)**-1 R_10 T_21 + R_21
    Args:
        R12: reflection matrix for light moving downwards 1->2
        R21: reflection matrix for light moving upwards 2->1
        T12: transmission matrix for light moving downwards 1->2
        T21: transmission matrix for light moving upwards 2->1
        R10: reflection array for light moving upwards 0->1
        T01: transmission array for light moving downwards 1->2
    Returns:
        R20, T02
    """
    print("A_Add_Slide")
    print("R12=")
    iadpython.start.print_matrix(R12, method)
    print("R21=")
    iadpython.start.print_matrix(R21, method)
    print("T12=")
    iadpython.start.print_matrix(T12, method)
    print("T21=")
    iadpython.start.print_matrix(T21, method)
    print("R10=")
    print(R10)
    print("T01=")
    print(T01)
    X = np.identity(len(R10)) - R10*R12
    temp = np.linalg.solve(X.T, T12.T).T
    T02 = temp * T01
    R20 = (temp * R10) @ T21 + R21

    print("T02=")
    iadpython.start.print_matrix(T02, method)
    print("R20=")
    iadpython.start.print_matrix(R20, method)
    return R20, T02


def B_Add_Slide(R12, T21, R01, R10, T01, T10, method):
    """
    Find two other matrices when slide is added to top of slab.

    Compute the resulting 'R02' and 'T20' matrices for a glass slide
    on top of an inhomogeneous layer characterized by 'R12', 'R21', 'T12',
    'T21' using:

    T_20=T_10 (E-R_12R_10 )**-1 T_21

    R_02=T_10 (E-R_12R_10)**-1 R_12 T_01 + R_01
    """
    print("R12=")
    iadpython.start.print_matrix(R12, method)
    print("T21=")
    iadpython.start.print_matrix(T21, method)
    print("R01=")
    print(R01)
    print("R10=")
    print(R10)
    print("T01=")
    print(T01)
    print("T10=")
    print(T10)
    X = np.identity(len(R10)) - R12*R10
    temp = np.linalg.solve(X.T, T10.T).T
    T20 = temp * T21
    R02 = (temp @ R12) * T01 + R01

    print("T20=")
    print(T20)
    print("R02=")
    print(R02)
    return R02, T20


def add_top(R01, R10, T01, T10, R12, R21, T12, T21, method):
    """
    Calculate matrices for a slab with a boundary placed on top.

    Args:
        'R01', 'R10', 'T01', 'T10' :R, T for slide assuming 0=air and 1=slab
        'R12', 'R21', 'T12', 'T21' :R, T for slab  assuming 1=slide and 2=?
    Returns:
        'R02', 'R20', 'T02', 'T20' & calc R, T for both  assuming 0=air and 2=?
    """
    R20, T02 = A_Add_Slide(R12, R21, T12, T21, R10, T01, method)
    R02, T20 = B_Add_Slide(R12, T21, R01, R10, T01, T10, method)
    return R02, R20, T02, T20


def add_bottom(R01, R10, T01, T10, R12, R21, T12, T21, method):
    """
    Calculate matrices for a slab with a boundary placed on bottom.

    Args:
        'R01', 'R10', 'T01', 'T10' :R, T for slide assuming 0=air and 1=slab
        'R12', 'R21', 'T12', 'T21' :R, T for slab  assuming 1=slide and 2=?
    Returns:
        'R02', 'R20', 'T02', 'T20' & calc R, T for both  assuming 0=air and 2=?
    """
    R02, T20 = A_Add_Slide(R10, R01, T10, T01, R12, T21, method)
    R20, T02 = B_Add_Slide(R10, T01, R21, R12, T21, T12, method)
    return R02, R20, T02, T20


def add_slides(R01, R10, T01, T10, R, T, method):
    """
    Find matrix when slab is sandwiched between identical slides.

    This routine is optimized for a slab with equal boundaries on each side.
    It is assumed that the slab is homogeneous and therefore the 'R' and 'T'
    matrices are identical for upward or downward light directions.

    If equal boundary conditions exist on both sides of the slab then, by
    symmetry, the transmission and reflection operator for light travelling
    from the top to the bottom are equal to those for light propagating from
    the bottom to the top. Consequently only one set need be calculated.
    This leads to a faster method for calculating the reflection and
    transmission for a slab with equal boundary conditions on each side.
    Let the top boundary be layer 01, the medium layer 12, and the bottom
    layer 23.  The boundary conditions on each side are equal:  R_01=R_32,
    R_10=R_23, T_01=T_32, and T_10=T_23.

    For example the light reflected from layer 01 (travelling from boundary
    0 to boundary 1) will equal the amount of light reflected from layer 32,
    since there is no physical difference between the two cases.  The switch
    in the numbering arises from the fact that light passes from the medium
    to the outside at the top surface by going from 1 to 0, and from 2 to 3
    on the bottom surface.  The reflection and transmission for the slab
    with boundary conditions are R_30 and  T_03 respectively.  These are
    given by

    T_02 = T_12(E-R_10R_12)**-1T_01

    R_20 = T_12(E-R_10R_12)**-1 R_10T_21+R_21

    T_03 = T_10(E-R_20R_10)**-1T_02

    R_30 = T_10(E-R_20R_10)**-1 R_20T_01+R_01

    Args:
        'R01', 'R10', 'T01', 'T10' : R, T for slide assuming 0=air and 1=slab
        'R', 'T'                   : R12=R21, T12=T21 for homogeneous slab
    Returns:
        'T30', 'T03'       : R, T for all 3 with top = bottom boundary
    """
    n = method.quad_pts
    X = np.identity(n) - R10*R
    temp = np.linalg.solve(X.T, T.T).T
    T02 = temp * T01
    R20 = (temp * R10) @ T + R

    X = np.identity(n) - R20*R10
    temp = np.linalg.solve(X.T, T10.T).T
    T03 = temp * T02
    R30 = (temp @ R20) * T01 + R01/method.twonuw**2

    return R30, T03


def sp_RT(s, method):
    """
    Specular reflection and transmission for a slide-slab-slide sandwich.

    The sample is characterized by the record 'slab'.  The total unscattered
    reflection and transmission for oblique irradiance ('urx' and 'utx')
    together with their companions 'uru' and 'utu' for diffuse irradiance.
    The cosine of the incident angle is specified by 'slab.cos_angle'.

    The way that this routine calculates the diffuse unscattered quantities
    based on the global quadrature angles previously set-up.  Consequently,
    these estimates are not exact.  In fact if 'n=4' then only two
    quadrature points will actually be used to figure out the diffuse
    reflection and transmission (assuming mismatched boundaries).

    This algorithm is pretty simple.  Since the quadrature angles are all
    chosen assuming points inside the medium, I must calculate the
    corresponding angle for light entering from the outside.  If the the
    cosine of this angle is greater than zero then the angle does not
    correspond to a direction in which light is totally internally
    reflected. For this ray, I find the unscattered that would be reflected
    or transmitted from the slab.  I multiply this by the quadrature angle
    and weight 'twoaw[i]' to get the total diffuse reflectance and
    transmittance.

    Oh, yes.  The mysterious multiplication by a factor of 'n_slab*n_slab'
    is required to account for the n**2-law of radiance.
    """
    n = method.quad_pts
    uru = 0
    utu = 0

    for i in range(n):
        nu_outside = iadpython.fresnel.cos_snell(s.n, method.nu[i], 1.0)
        if nu_outside != 0:
            r, t = iadpython.fresnel.specular_nu_RT(s.n_above, s.n, s.n_below,
                                                    s.b_above, s.b, s.b_below, nu_outside)
            uru += method.twonuw[i] * r
            utu += method.twonuw[i] * t

    ur1, ut1 = iadpython.fresnel.specular_nu_RT(s.n_above, s.n, s.n_below,
                                                s.b_above, s.b, s.b_below, s.nu_0)

    uru *= s.n**2
    utu *= s.n**2
    return ur1, ut1, uru, utu
