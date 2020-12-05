# pylint: disable=invalid-name

"""
Module for obtaining quadrature points and weights.

    import iad.quadrature

    n=8
    x, w = iad.quadrature.gauss_quad(n)
    print(" i        x         weight")
    for i,x in enumerate(xi):
        print("%2d   %+.12f %+.12f" % (i, x[i], w[i]))

    x, w = iad.quadrature.radau_quad(n)
    print(" i        x         weight")
    for i,x in enumerate(xi):
        print("%2d   %+.12f %+.12f" % (i, x[i], w[i]))

    x, w = iad.quadrature.lobatto_quad(n)
    print(" i        x         weight")
    for i,x in enumerate(xi):
        print("%2d   %+.12f %+.12f" % (i, x[i], w[i]))

"""

import functools
import scipy.special
import scipy.optimize
import numpy as np

__all__ = ('gauss_quad',
           'radau_quad',
           'lobatto_quad',
           'gauss_func',
           'radau_func',
           'lobatto_func'
           )

def gauss_func(n, x):
    """Zeros of this function are nodes for Gaussian quadrature."""
    return scipy.special.legendre(n)(x)

def radau_func(n, x):
    """Zeros of this function are nodes for Radau quadrature."""
    return (scipy.special.eval_legendre(n-1, x) + scipy.special.eval_legendre(n, x))/(1+x)

def lobatto_func(n, x):
    """Zeros of this function are nodes for Lobatto quadrature."""
    return scipy.special.legendre(n-1).deriv(1)(x)

def gauss_quad(n):
    """Return nodes and weights for Gaussian quadrature."""
    nodes, weights, _ = scipy.special.roots_legendre(n, mu=True)
    return(np.flip(nodes), np.flip(weights))

def radau_quad(n):
    """Return nodes and weights for Radau quadrature."""
    nodes = np.zeros(n)
    weights = np.zeros(n)
    nodes[0] = -1
    weights[0] = 2/n**2

    # Use the fact that the roots of P_{n} bracket the roots of P_{n-1}':
    brackets = scipy.special.roots_legendre(n)[0]

    f = functools.partial(radau_func, n)
    for i in range(n-1):
        nodes[i+1] = scipy.optimize.brentq(f, brackets[i], brackets[i+1])

    pp = scipy.special.legendre(n-1).deriv(1)
    weights[1:] = 1/pp(nodes[1:])**2/(1-nodes[1:])

    return(np.flip(nodes), np.flip(weights))

def lobatto_quad(n):
    """Return nodes and weights for Lobatto quadrature."""
    nodes = np.zeros(n)
    nodes[0] = -1
    nodes[-1] = 1
    weights = np.full(n, 2/n/(n-1))

    # Use the fact that the roots of P_{n-1} bracket the roots of P_{n-1}':
    brackets = scipy.special.roots_legendre(n-1)[0]

    f = functools.partial(lobatto_func, n)
    for i in range(n-2):
        nodes[i+1] = scipy.optimize.brentq(f, brackets[i], brackets[i+1])

    pp = scipy.special.legendre(n-1)(nodes)
    weights[1:-1] = weights[1:-1]/pp[1:-1]**2

    return(np.flip(nodes), np.flip(weights))
