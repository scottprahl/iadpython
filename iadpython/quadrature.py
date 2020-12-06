# pylint: disable=invalid-name

"""
Module for obtaining quadrature points and w.

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
           'gauss_func',
           'radau_func',
           'lobatto_func'
           )

def gauss_func(n, x):
    """Zeros of this function are the abscissas for Gaussian quadrature."""
    return scipy.special.legendre(n)(x)

def radau_func(n, x):
    """Zeros of this function are the abscissas for Radau quadrature."""
    return (scipy.special.eval_legendre(n-1, x) + scipy.special.eval_legendre(n, x))/(1+x)

def lobatto_func(n, x):
    """Zeros of this function are the abscissas for Lobatto quadrature."""
    return scipy.special.legendre(n-1).deriv(1)(x)

def gauss(n, a=-1, b=1):
    """
    Return abscissas and weights for Gaussian quadrature.
    integral from a to b of f(x) dx = sum w_i f(x_i)
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
    x *= -0.5 * (b - a)
    x += 0.5 * (b + a)
    w *= 0.5 * (b - a)

    return(np.flip(x), np.flip(w))

def radau(n, a=-1, b=1):
    """
    Return abscissas and weights for Radau quadrature.
    integral from a to b of f(x) dx = sum w_i f(x_i)
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
    w[0] = 2/n**2

    # the roots of P_{n} bracket the roots of P_{n}':
    brackets = scipy.special.roots_legendre(n)[0]

    f = functools.partial(radau_func, n)
    for i in range(n-1):
        x[i+1] = scipy.optimize.brentq(f, brackets[i], brackets[i+1])

    pp = scipy.special.legendre(n-1).deriv(1)
    w[1:] = 1/pp(x[1:])**2/(1-x[1:])

    # scale for desired interval
    x *= -0.5 * (b - a)
    x += 0.5 * (b + a)
    w *= 0.5 * (b - a)

    return(np.flip(x), np.flip(w))

def lobatto(n, a=-1, b=1):
    """
    Return abscissas and weights for Lobatto quadrature.
    integral from a to b of f(x) dx = sum w_i f(x_i)
    Args:
        n: number of quadrature points
        a: lower limit of integral
        b: upper limit of integral
    Returns:
        x: array of abscissas of length n
        w: array of weights of length n
    """
    x = np.zeros(n)
    x[0] = -1
    x[-1] = 1
    w = np.full(n, 2/n/(n-1))

    # The roots of P_{n-1} bracket the roots of P_{n-1}':
    brackets = scipy.special.roots_legendre(n-1)[0]

    f = functools.partial(lobatto_func, n)
    for i in range(n-2):
        x[i+1] = scipy.optimize.brentq(f, brackets[i], brackets[i+1])

    pp = scipy.special.legendre(n-1)(x)
    w[1:-1] = w[1:-1]/pp[1:-1]**2

    # scale for desired interval
    x *= -0.5 * (b - a)
    x += 0.5 * (b + a)
    w *= 0.5 * (b - a)

    return(np.flip(x), np.flip(w))
