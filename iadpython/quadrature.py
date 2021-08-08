# pylint: disable=invalid-name

"""
Module for obtaining quadrature points and weights.

Three types of gaussian quadrature are supported: normal Gaussian,
Radau quadrature, and Lobatto quadrature.  The first method does not
include either endpoint of integration, Radau quadrature includes one
endpoint of the integration range, and Lobatto quadrature includes both
endpoints.::

    import iad.quadrature

    n=8
    x, w = iad.quadrature.gauss(n)
    print(" i        x         weight")
    for i,x in enumerate(xi):
        print("%2d   %+.12f %+.12f" % (i, x[i], w[i]))

    x, w = iad.quadrature.radau(n)
    print(" i        x         weight")
    for i,x in enumerate(xi):
        print("%2d   %+.12f %+.12f" % (i, x[i], w[i]))

    x, w = iad.quadrature.lobatto(n)
    print(" i        x         weight")
    for i,x in enumerate(xi):
        print("%2d   %+.12f %+.12f" % (i, x[i], w[i]))

"""

import functools
import scipy.special
import scipy.optimize
import numpy as np

__all__ = ('gauss',
           'radau',
           'lobatto',
           )


def _gauss_func(n, x):
    """Zeroes of this function are the Gaussian quadrature points."""
    return scipy.special.legendre(n)(x)


def _radau_func(n, x):
    """Zeros of this function are the Radau quadrature points."""
    return (scipy.special.eval_legendre(n - 1, x) + scipy.special.eval_legendre(n, x)) / (1 + x)


def _lobatto_func(n, x):
    """Zeros of this function are the Lobatto quadrature points."""
    return scipy.special.legendre(n - 1).deriv(1)(x)


def gauss(n, a=-1, b=1):
    """
    Return abscissas and weights for Gaussian quadrature.

    The definite integral ranges from a to b.  The default
    interval is -1 to 1.  The quadrature approximation is
    just the sum of w_i f(x_i).  Neither a nor b is included
    in the list of quadrature abscissas.

    The result should be exact when integrating any polynomial
    of degree 2n-1 or less.

    If -a=b, then abscissas will be symmetric about the origin

    Args:
        n: number of quadrature points
        a: lower limit of integral
        b: upper limit of integral
    Returns:
        x: array of abscissas of length n
        w: array of weights of length n
    """
    x, w, _ = scipy.special.roots_legendre(n, mu=True)

    # scale for desired interval
    x *= 0.5 * (a - b)
    x += 0.5 * (a + b)
    w *= 0.5 * (b - a)

    return np.flip(x), np.flip(w)


def radau(n, a=-1, b=1):
    """
    Return abscissas and weights for Radau quadrature.

    The definite integral ranges from a to b.  The default
    interval is -1 to 1.  The quadrature approximation is
    just the sum of w_i f(x_i).  The upper endpoint b is include
    in the list of quadrature abscissas.

    The result should be exact when integrating any polynomial
    of degree 2n-2 or less.

    Args:
        n: number of quadrature points
        a: lower limit of integral
        b: upper limit of integral
    Returns:
        x: array of abscissas of length n
        w: array of weights of length n
    """
    x = np.zeros(n)
    w = np.zeros(n)
    x[0] = -1

    # the roots of P_{n} bracket the roots of P_{n}':
    brackets = scipy.special.roots_legendre(n)[0]

    f = functools.partial(_radau_func, n)
    for i in range(n - 1):
        x[i + 1] = scipy.optimize.brentq(f, brackets[i], brackets[i + 1])

    pp = scipy.special.legendre(n - 1).deriv(1)
    w[0] = 2 / n**2
    w[1:] = 1 / pp(x[1:])**2 / (1 - x[1:])

    # scale for desired interval
    x *= 0.5 * (a - b)
    x += 0.5 * (b + a)
    w *= 0.5 * (b - a)

    return np.flip(x), np.flip(w)


def lobatto(n, a=-1, b=1):
    """
    Return abscissas and weights for Lobatto quadrature.

    The definite integral ranges from a to b.  The default
    interval is -1 to 1.  The quadrature approximation is
    just the sum of w_i f(x_i).  Both endpoints a and b are include
    in the list of quadrature abscissas.

    The result should be exact when integrating any polynomial
    of degree 2n-3 or less.

    If -a=b, then abscissas will be symmetric about the origin

    Args:
        n: number of quadrature points
        a: lower limit of integral
        b: upper limit of integral
    Returns:
        x: array of abscissas of length n
        w: array of weights of length n
    """
    x = np.zeros(n)
    w = np.full(n, 2 / n / (n - 1))
    x[0] = -1
    x[-1] = 1

    # The roots of P_{n-1} bracket the roots of P_{n-1}':
    brackets = scipy.special.roots_legendre(n - 1)[0]

    f = functools.partial(_lobatto_func, n)
    for i in range(n - 2):
        x[i + 1] = scipy.optimize.brentq(f, brackets[i], brackets[i + 1])

    pp = scipy.special.legendre(n - 1)(x)
    w[1:-1] = w[1:-1] / pp[1:-1]**2

    # scale for desired interval
    x *= 0.5 * (a - b)
    x += 0.5 * (a + b)
    w *= 0.5 * (b - a)

    return np.flip(x), np.flip(w)
