# pylint: disable=invalid-name
# pylint: disable=no-member

import scipy.special
import numpy as np
import numpy.polynomial.legendre

__all__ = ('get_phi_elliptic',
           'get_phi_legendre',
           )

"""
Calculation of the redistribution function.

The single scattering phase function p(nu) for a tissue determines the
amount of light scattered at an angle nu=cos(theta) from the direction of
incidence.  The subtended angle nu is the dot product incident and exiting
of the unit vectors.

The redistribution function `h[i,j]` determines the fraction of light
scattered from an incidence cone with angle `nu_i` into a cone with angle
`nu_j`.  The redistribution function is calculated by averaging the phase
function over all possible azimuthal angles for fixed angles `nu_i` and
`nu_j`,

Note that the angles `nu_i` and `nu_j` may also be negative (light
travelling in the opposite direction).

When the cosine of the angle of incidence or exitance is unity (`nu_i=1` or
`nu_j=1`), then the redistribution function is equivalent to the phase
function p(nu_j).
"""

def get_phi_legendre(slab, method):
    """
    Calculate the redistribution matrix using Legendre polynomials.

    This is a straightforward implementation of Wiscombe's delta-M
    method for calculating the redistribution function as given in

    Wiscombe, "The Delta-M Method : Rapid Yet Accurate Radiative Flux
    Calculations for Strongly Asymmetric Phase Functions,"
    J. Atmos. Sci., 34, 1978.

    Probably should generate all the Legendre polynomials one
    time and then calculate.
    """

#     if (fabs(slab.g) >= 1)
# 	    throw("Get_Phi was called with a bad g_calc value")

    n = method.quad_pts

    # Isotropic phase function is constant
    g = slab.g
    if g == 0:
        h = np.ones([n, n])
        return h, h, h, h

    hp = np.ones([n, n])
    hm = np.ones([n, n])
    for k in range(1, n):
        c = np.append(np.zeros(k), [1])
        chik = (2*k + 1) * (g**k - g**n) /(1 - g**n)
        pk = numpy.polynomial.legendre.legval(method.nu, c)
        for i in range(n):
            for j in range(i+1):
                temp = chik * pk[i] * pk[j]
                hp[i, j] += temp
                hm[i, j] += (-1)**k * temp

# 	fill symmetric entries
    for i in range(n):
        for j in range(i+1, n):
            hp[i, j] = hp[j, i]
            hm[i, j] = hm[j, i]

    return hp, hm

def get_phi_elliptic(slab, method):
    """
    Calculate redistribution function using elliptic integrals.

    This is the result of a direct integration of the Henyey-
    Greenstein phase function.

    It is not terribly useful because we cannot use the
    delta-M method to more accurate model highly anisotropic
    phase functions.
    """

#     if (fabs(slab.g) >= 1)
# 	    throw("Get_Phi was called with a bad g_calc value")

    n = method.quad_pts

    # Isotropic phase function is constant
    g = slab.g**n
    if g == 0:
        h = np.ones([n, n])
        return h, h

    hp = np.zeros([n, n])
    hm = np.zeros([n, n])
    for i in range(n):
        for j in range(i+1):
            ni = method.nu[i]
            nj = method.nu[j]
            gamma = 2 * g * np.sqrt(1-ni**2) * np.sqrt(1-nj**2)

            alpha = 1 + g*g - 2 * g * ni * nj
            const = 2/np.pi * (1-g*g)/(alpha-gamma)/np.sqrt(alpha+gamma)
            arg = np.sqrt(2*gamma/(alpha+gamma))
            hp[i, j] = const * scipy.special.ellipe(arg)

            alpha = 1 + g*g + 2 * g * ni * nj
            const = 2/np.pi * (1-g*g)/(alpha-gamma)/np.sqrt(alpha+gamma)
            arg = np.sqrt(2*gamma/(alpha+gamma))
            hm[i, j] = const * scipy.special.ellipe(arg)

# 	fill symmetric entries
    for i in range(n):
        for j in range(i+1, n):
            hp[i, j] = hp[j, i]
            hm[i, j] = hm[j, i]

    return hp, hm
