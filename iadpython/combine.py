# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals

"""
Module for adding layers together.

Two types of starting methods are possible.

    import iadpython.start
    import iadpython.layer

    n=4
    slab = iadpython.start.Slab(a=0.9, b=10, g=0.9, n=1.5)
    method = iadpython.start.Method(slab)
    r_init, t_init = iad.start.init_layer(slab, method)
    r, t = iad.layer.double_until(r_init, t_init, method.b_thinnest, method.b)
    print(r)
    print(t)

"""

import scipy
import numpy as np
import iadpython

__all__ = ('add_layers',
           'add_layers_basic',
           'simple_layer_matrices',
           'add_slide_above',
           'add_slide_below',
           'add_same_slides'
           )


def add_layers_basic(sample, R10, T01, R12, R21, T12, T21):
    """
    Add two layers together.

    The basic equations for the adding-doubling sample (neglecting sources) are

    T_02  = T_12 (E - R_10 R_12)⁻¹ T_01

    R_20  = T_12 (E - R_10 R_12)⁻¹ R_10 T_21 +R_21

    T_20  = T_10 (E - R_12 R_10)⁻¹ T_21

    R_02  = T_10 (E - R_12 R_10)⁻¹ R_12 T_01 +R_01

    Upon examination it is clear that the two sets of equations have
    the same form.  These equations assume some of the multiplications are
    star multiplications. Explicitly,

    T_02  = T_12 (E - R_10 C R_12 )⁻¹ T_01

    R_20  = T_12 (E - R_10 C R_12 )⁻¹ R_10 C T_21 +R_21

    where the diagonal matrices C and E are

    E_ij= 1/(2*nu_i*w_i) delta_ij
    C_ij= 2*nu_i*w_i delta_ij
    """
    C = np.diagflat(sample.twonuw)
    E = np.diagflat(1 / sample.twonuw)

    A = E - R10 @ C @ R12
    B = np.linalg.solve(A.T, T12.T).T
    R20 = B @ R10 @ C @ T21 + R21
    T02 = B @ T01
    return R20, T02


def add_layers(sample, R01, R10, T01, T10, R12, R21, T12, T21):
    """
    Add two layers together.

    Use this when the combined system is asymmetric R02!=R20 and T02!=T20.
    """
    R20, T02 = add_layers_basic(sample, R10, T01, R12, R21, T12, T21)
    R02, T20 = add_layers_basic(sample, R12, T21, R10, R01, T10, T01)
    return R02, R20, T02, T20


def double_until(sample, r_start, t_start, b_start, b_end):
    """Double until proper thickness is reached."""
    r = r_start
    t = t_start
    if b_end == 0 or b_end <= b_start:
        return r, t

    if b_end > iadpython.AD_MAX_THICKNESS:
        old_utu = 100
        utu = 10
        while abs(utu - old_utu) > 1e-6:
            old_utu = utu
            r, t = add_layers_basic(sample, r, t, r, r, t, t)
            _, _, _, utu = sample.UX1_and_UXU(r, t)
        return r, t

    while abs(b_end - b_start) > 0.00001 and b_end > b_start:
        r, t = add_layers_basic(sample, r, t, r, r, t, t)
        b_start *= 2
    return r, t


def simple_layer_matrices(sample):
    """Create R and T matrices for layer without boundaries."""
    r_start, t_start = iadpython.start.thinnest_layer(sample)
    b_start = sample.b_thinnest
    b_end = sample.b_delta_M()
    r, t = double_until(sample, r_start, t_start, b_start, b_end)
    return r, t


def _add_boundary_config_a(sample, R12, R21, T12, T21, R10, T01):
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
    n = sample.quad_pts
    X = (np.identity(n) - R10 * R12.T).T
    temp = np.linalg.solve(X.T, T12.T).T
    T02 = temp * T01
    R20 = (temp * R10) @ T21 + R21

    return R20, T02


def _add_boundary_config_b(sample, R12, T21, R01, R10, T01, T10):
    """
    Find two other matrices when slide is added to top of slab.

    Compute the resulting 'R02' and 'T20' matrices for a glass slide
    on top of an inhomogeneous layer characterized by 'R12', 'R21', 'T12',
    'T21' using:

    T_20=T_10 (E-R_12R_10 )**-1 T_21

    R_02=T_10 (E-R_12R_10)**-1 R_12 T_01 + R_01

    Args:
        R12: reflection matrix for light moving downwards 1->2
        T21: transmission matrix for light moving upwards 2->1
        R01: reflection matrix for light moving downwards 0->1
        R10: reflection matrix for light moving upwards 1->0
        T01: transmission array for light moving downwards 1->2
        T10: transmission array for light moving upwards 0->1

    Returns:
        R02, T20
    """
    n = sample.quad_pts
    X = np.identity(n) - R12 * R10
    temp = np.linalg.solve(X.T, np.diagflat(T10)).T
    T20 = temp @ T21
    R02 = (temp @ R12) * T01
    R02 += np.diagflat(R01 / sample.twonuw**2)

    return R02, T20


def add_slide_above(sample, R01, R10, T01, T10, R12, R21, T12, T21):
    """
    Calculate matrices for a slab with a boundary placed above.

    This routine should be used before the slide is been added below!

    Here 0 is the air/top-of-slide, 1 is the bottom-of-slide/top-of-slab boundary,
    and 2 is the is the bottom-of-slab boundary.

    Args:
        R01: reflection arrays for slide 0->1
        R10: reflection arrays for slide 1->0
        T01: transmission arrays for slide 0->1
        T10: transmission arrays for slide 1->0
        R12: reflection matrices for slab 1->2
        R21: reflection matrices for slab 2->1
        T12: transmission matrices for slab 1->2
        T21: transmission matrices for slab 2->1

    Returns:
        R02, R20, T02, T20: matrices for slide+slab combination
    """
    R20, T02 = _add_boundary_config_a(sample, R12, R21, T12, T21, R10, T01)
    R02, T20 = _add_boundary_config_b(sample, R12, T21, R01, R10, T01, T10)
    return R02, R20, T02, T20


def add_slide_below(sample, R01, R10, T01, T10, R12, R21, T12, T21):
    """
    Calculate matrices for a slab with a boundary placed below.

    This routine should be used after the slide has been added to the top.

    Here 0 is the top of slab, 1 is the bottom-of-slab/top-of-slide boundary,
    and 2 is the is the bottom-of-slide/air boundary.

    Args:
        R01, R10: reflection matrices for slab
        T01, T10: transmission matrices for slab
        R12, R21: reflection arrays for slide
        T12, T21: transmission arrays for slide

    Returns:
        R02, R20, T02, T20: matrices for slab+slide combination
    """
    R02, T20 = _add_boundary_config_a(sample, R10, R01, T10, T01, R12, T21)
    R20, T02 = _add_boundary_config_b(sample, R10, T01, R21, R12, T21, T12)
    return R02, R20, T02, T20


def add_same_slides(sample, R01, R10, T01, T10, R, T):
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

    A_XX = T_12(E-R_10R_12)**-1

    R_20 = A_XX R_10T_21 + R_21

    B_XX = T_10(E-R_20R_10)**-1

    T_03 = B_XX A_XX T_01

    R_30 = B_XX R_20 T_01 + R_01/(2nuw)**2

    Args:
        R01: R, T for slide assuming 0=air and 1=slab
        R10, T01, T10: R, T for slide assuming 0=air and 1=slab
        T01, T10: R, T for slide assuming 0=air and 1=slab
        T10: R, T for slide assuming 0=air and 1=slab
        R, T: R12=R21, T12=T21 for homogeneous slab

    Returns:
        T30, T03: R, T for all 3 with top = bottom boundary
    """
    n = sample.quad_pts
    X = np.identity(n) - R10 * R
    AXX = np.linalg.solve(X, T.T).T
    R20 = (AXX * R10) @ T + R

    X = np.identity(n) - R20 * R10
    BXX = scipy.linalg.solve(X.T, np.diagflat(T10)).T
    T03 = BXX @ AXX * T01
    R30 = BXX @ R20 * T01
    R30 += np.diagflat(R01 / sample.twonuw**2)

    return R30, T03
