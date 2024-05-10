"""Module for reading rxt files.

This reads an output txt file and saves the parameters into an
Experiment object.

Example::

    >>> import numpy
    >>> import matplotlib.pyplot as plt
    >>> import iadpython
    >>>
    >>> filename = 'ink.txt'
    >>> exp, data = iadpython.read_txt(filename)
    >>> plt.plot(data.lam, data.mr)
    >>> plt.plot(data.lam, data.rr)
    >>> plt.xlabel("Wavelength")
    >>> plt.ylabel("measured reflectance")
    >>> plt.title(filename)
    >>> plt.show()
"""

import os
import re
import numpy as np
import iadpython

__all__ = ('read_txt', 'IADResult')


class IADResult():
    """Container class results in an iad output file."""
    def __init__(self):
        """Initialization."""
        self.lam = np.array([0], dtype=float)
        self.mr = np.array([0], dtype=float)
        self.cr = np.array([0], dtype=float)
        self.mt = np.array([0], dtype=float)
        self.ct = np.array([0], dtype=float)
        self.mua = np.array([0], dtype=float)
        self.musp = np.array([0], dtype=float)
        self.g = np.array([0], dtype=float)
        self.success = np.array([0], dtype=bool)
        self.mus = np.array([0], dtype=float)


def verify_magic(fp, magic):
    """Verify that the file's initial bytes match the string 'magic'."""
    fp.seek(0)
    chunk = fp.read(len(magic))
    fp.seek(0)
    return chunk == magic


def get_number_from_line(fp):
    """Read the number on the next line after an = sign."""
    s = fp.readline()
#    print("starting=%s" % s)
    if s[0] != '#':
        print("line in file should start with `#`")
        return 0

    s = re.sub(r'.*= *', '', s)
    s = re.sub(r' .*', '', s)

#    print("finish x=%10.5f" % float(s))
    return float(s)


def read_sphere(fp):
    """Read the information for a sphere."""
    line = fp.readline()
    d_sphere = get_number_from_line(fp)
    d_sample = get_number_from_line(fp)
    sphere = iadpython.Sphere(d_sphere, d_sample)
    sphere.baffle = 'a baffle' in line
    sphere.third.d = get_number_from_line(fp)
    sphere.detector.d = get_number_from_line(fp)
    sphere.detector.uru = get_number_from_line(fp) / 100
    sphere.r_wall = get_number_from_line(fp) / 100
    sphere.r_std = get_number_from_line(fp) / 100
    return sphere


def read_misc(fp, _exp):
    """Read info after sphere data but before data."""
    for _ in range(14):
        fp.readline()


def read_txt(filename):
    """Read an IAD output file in .rxt format."""
    if not os.path.exists(filename):
        raise ValueError('input file "%s" must end in ".txt"' % filename)

    # verify that file is an output file
    with open(filename, encoding="utf-8") as fp:
        if not verify_magic(fp, '# Inverse Adding-Doubling'):
            raise ValueError('"%s" does not start with "# Inverse Adding-Doubling"' % filename)

        # create experiment object
        exp = iadpython.Experiment()

        # now read the header
        fp.readline()
        fp.readline()
        exp.d_beam = get_number_from_line(fp)
        exp.sample.d = get_number_from_line(fp)
        exp.sample.d_above = get_number_from_line(fp)
        exp.sample.d_below = get_number_from_line(fp)
        exp.sample.n = get_number_from_line(fp)
        exp.sample.n_above = get_number_from_line(fp)
        exp.sample.n_below = get_number_from_line(fp)
        fp.readline()
        exp.fraction_of_rc_in_mr = get_number_from_line(fp) / 100
        exp.fraction_of_tc_in_mt = get_number_from_line(fp) / 100
        fp.readline()
        exp.r_sphere = read_sphere(fp)
        fp.readline()
        exp.t_sphere = read_sphere(fp)
        read_misc(fp, exp)

        data = IADResult()
        position = fp.tell()

        result = np.loadtxt(fp, usecols=range(8), delimiter='\t')
        lam, mr, cr, mt, ct, mua, musp, g = result.T
        data.lam = np.atleast_1d(lam)
        data.mr = np.atleast_1d(mr)
        data.cr = np.atleast_1d(cr)
        data.mt = np.atleast_1d(mt)
        data.ct = np.atleast_1d(ct)
        data.mua = np.atleast_1d(mua)
        data.musp = np.atleast_1d(musp)
        data.g = np.atleast_1d(g)
        data.mus = data.musp / (1 - g)
        exp.m_r = data.mr
        exp.m_t = data.mt
        exp.lambda0 = data.lam

        fp.seek(position)
        converters = {8: lambda s: s.lstrip(b'#').strip()}
        status = np.loadtxt(fp, usecols=[8], dtype=str, converters=converters)
        status = np.atleast_1d(status)
        data.success = status == '*'

    return exp, data
