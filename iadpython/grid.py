# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=consider-using-f-string

"""
Class for doing inverse adding-doubling calculations for a sample.

import iadpython

exp = iadpython.Experiment(0.5, 0.1, default_g=0.5)
grid = iadpython.Grid()
grid.calc(exp)
print(grid)
"""

import numpy as np


class Grid():
    """Container class for details of an experiment."""

    def __init__(self, search=None, default=None, N=21):
        """Object initialization."""
        self.search = search
        self.default = default
        self.N = N
        self.a = np.zeros((N, N))
        self.b = np.zeros((N, N))
        self.g = np.zeros((N, N))
        self.ur1 = np.zeros((N, N))
        self.ut1 = np.zeros((N, N))

    def __str__(self):
        """Return basic details as a string for printing."""
        s = "---------------- Grid ---------------\n"
        if self.search is None:
            s += "search  = None\n"
        else:
            s += "search = %s\n" % self.search

        if self.default is None:
            s += "default = None\n"
        else:
            s += "default = %.5f\n" % self.default

        s += matrix_as_string(self.a, "a")
        s += matrix_as_string(self.b, "b")
        s += matrix_as_string(self.g, "g")
        s += matrix_as_string(self.ur1, "ur1")
        s += matrix_as_string(self.ut1, "ut1")

        return s

    def calc(self, exp):
        """Precalculate a grid."""
        a = np.linspace(0, 1, self.N)
        b = np.linspace(0, 10, self.N)
        g = np.linspace(-0.95, 0.95, self.N)

        self.search = exp.search

        if self.search == 'find_ab':
            self.default = exp.default_g
            self.g = np.full((self.N, self.N), self.default)
            self.a, self.b = np.meshgrid(a, b)

        if self.search == 'find_ag':
            self.default = exp.default_b
            self.b = np.full((self.N, self.N), self.default)
            self.a, self.g = np.meshgrid(a, g)

        if self.search == 'find_bg':
            self.default = exp.default_a
            self.a = np.full((self.N, self.N), self.default)
            self.b, self.g = np.meshgrid(b, g)

        for i in range(self.N):
            for j in range(self.N):
                exp.sample.a = self.a[i, j]
                exp.sample.b = self.b[i, j]
                exp.sample.g = self.g[i, j]
                self.ur1[i, j], self.ut1[i, j], _, _ = exp.sample.rt()

    def min_abg(self, mr, mt):
        """Find closest a, b, g closest to mr and mt."""
        if self.ur1 is None:
            raise Exception("Grid.calc(exp) must be called before Grid.min_abg")

        A = np.abs(mr - self.ur1) + np.abs(mt - self.ut1)
        ii_flat = A.argmin()
        i, j = ii_flat // A.shape[1], ii_flat % A.shape[1]
        return self.a[i, j], self.b[i, j], self.g[i, j]


def matrix_as_string(x, label=''):
    """Return matrix as a string."""
    if x is None:
        return ""

    n, m = x.shape

    ndashes = (80 - len(label) - 2) // 2
    s = "\n"
    s += '-' * ndashes + ' ' + label + ' ' + '-' * ndashes
    s += "\n[\n"
    for i in range(n):
        s += '['
        for j in range(m):
            s += "%6.3f, " % x[i, j]
        s += "], \n"
    s += "]\n"
    return s