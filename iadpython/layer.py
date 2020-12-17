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

import numpy as np

__all__ = ('add_layers',
           'double_until',
           )

def add_layers(R10, T01, R12, R21, T12, T21, twonuw):
    """
    Basic Routine to Add Layers Without Sources.

    The basic equations for the adding-doubling method (neglecting sources) are

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
    n = len(twonuw)
    C = np.identity(n) * twonuw
    E = np.identity(n) / twonuw

    A = E - R10 @ C @ R12
    B = np.linalg.solve(A.T, T12.T).T
    R20 = B @ R10 @ C @ T21 + R21
    T02 = B @ T01
    return R20, T02

def double_until(r, t, b_start, b_end, twonuw):
    """Add until proper thickness is reached."""
    while abs(b_end-b_start) > 0.00001 and b_end > b_start:
        r, t = add_layers(r, t, r, r, t, t, twonuw)
        b_start *= 2

    return r, t
