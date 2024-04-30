# pylint: disable=invalid-name
# pylint: disable=too-many-locals
# pylint: disable=consider-using-f-string
# pylint: disable=too-many-branches

"""Module for reading rxt files.

This reads an input rxt file and saves the parameters into an
Experiment object.

Example::

    >>> import numpy
    >>> import matplotlib.pyplot as plt
    >>> import iadpython
    >>>
    >>> filename = 'ink.rxt'
    >>> exp = iadpython.read_rxt(filename)
    >>> if exp.lambda0 is None:
    >>>     plt.plot(exp.m_r)
    >>> else:
    >>>     plt.plot(exp.lambda0, exp.m_r)
    >>> plt.ylabel("measured reflectance")
    >>> plt.title(filename)
    >>> plt.show()
"""

import os
import re
import numpy as np
import iadpython

__all__ = ('read_rxt', 'read_and_remove_notation')


def read_and_remove_notation(filename):
    """Read file and remove all whitespace and comments."""
    s = ''

    if not os.path.exists(filename):
        raise ValueError('input file "%s" must end in ".rxt"' % filename)

    with open(filename, encoding="utf-8") as f:
        for line in f:
            line = re.sub(r'\s*#.*', '', line)
            line = re.sub(r', ', ' ', line)
            s += line

    if len(re.findall('IAD1', s)) == 0:
        raise ValueError("Not an .rxt file. (Does not start with IAD1)")

    s = re.sub(r'IAD1', '', s)
    s = re.sub(r'\s+', ' ', s)
    s = s.rstrip()
    s = s.lstrip()
    return s


def fill_in_data(exp, data, column_letters_str):
    if column_letters_str == '':
        columns = int(exp.num_measures)
        if data[0] > 1:
            columns += 1
    else:
        columns = len(column_letters_str)

    print(column_letters_str)
    data_in_columns = data.reshape(-1, columns)
    exp.lambda0 = None
    if column_letters_str == '':
        col = 0
        if data[0] > 1:
            exp.lambda0 = data_in_columns[:, 0]
            col = 1
        exp.m_r = data_in_columns[:, col]
        if exp.num_measures >= 2:
            exp.m_t = data_in_columns[:, col + 1]
        if exp.num_measures >= 3:
            exp.m_u = data_in_columns[:, col + 2]
        if exp.num_measures >= 4:
            exp.r_sphere.r_wall = data_in_columns[:, col + 3]
        if exp.num_measures >= 5:
            exp.t_sphere.r_wall = data_in_columns[:, col + 4]
        if exp.num_measures >= 6:
            exp.r_sphere.r_std = data_in_columns[:, col + 5]
        if exp.num_measures >= 7:
            exp.t_sphere.r_std = data_in_columns[:, col + 6]
        if exp.num_measures > 7:
            raise ValueError('unimplemented')
        return

    for col, letter in enumerate(column_letters_str):
        if letter == 'a':
            exp.default_a = data_in_columns[:, col]
        elif letter == 'b':
            exp.default_b = data_in_columns[:, col]
        elif letter == 'B':
            exp.d_beam = data_in_columns[:, col]
        elif letter == 'c':
            exp.fraction_of_rc_in_mr = data_in_columns[:, col]
        elif letter == 'C':
            exp.fraction_of_tc_in_mt = data_in_columns[:, col]
        elif letter == 'd':
            exp.sample.d = data_in_columns[:, col]
        elif letter == 'D':
            exp.sample.d_above = data_in_columns[:, col]
            exp.sample.d_below = data_in_columns[:, col]
        elif letter == 'E':
            exp.sample.b_above = data_in_columns[:, col]
            exp.sample.b_below = data_in_columns[:, col]
        elif letter == 'e':
            exp.error = data_in_columns[:, col]
        elif letter == 'g':
            exp.default_g = data_in_columns[:, col]
        elif letter == 't':
            exp.m_t = data_in_columns[:, col]
        elif letter == 'L':
            exp.lambda0 = data_in_columns[:, col]
        elif letter == 'n':
            exp.n = data_in_columns[:, col]
        elif letter == 'N':
            exp.n_above = data_in_columns[:, col]
            exp.n_below = data_in_columns[:, col]
        elif letter == 'r':
            exp.m_r = data_in_columns[:, col]
        elif letter == 'R':
            exp.r_sphere.r_std = data_in_columns[:, col]
        elif letter == 'S':
            exp.num_spheres = data_in_columns[:, col]
        elif letter == 'T':
            exp.t_sphere.r_rstd = data_in_columns[:, col]
        elif letter == 'u':
            exp.m_u = data_in_columns[:, col]
        elif letter == 'w':
            exp.r_sphere.r_wall = data_in_columns[:, col]
        elif letter == 'W':
            exp.t_sphere.r_wall = data_in_columns[:, col]
        else:
            raise ValueError('unimplemented column type "%s"' % letter)


def read_rxt(filename):
    """Read an IAD input file in .rxt format.

    Args:
        filename: .rxt filename

    Returns:
        Experiment object
    """
    s = read_and_remove_notation(filename)
    x = s.split(' ')

    # Remove single-letter entries and save them
    column_letters = [item for item in x if len(item) == 1 and item.isalpha()]
    x = [item for item in x if not (len(item) == 1 and item.isalpha())]
    column_letters_str = ''.join(column_letters)

    x = np.array([float(item) for item in x])

    sample = iadpython.Sample(a=None, b=None, g=None)
    sample.n = x[0]
    sample.n_above = x[1]
    sample.d = x[2]
    sample.d_above = x[3]

    # try and save people from themselves
    if sample.d_above == 0:
        sample.n_above = 1
    if sample.n_above == 1:
        sample.d_above = 0
    if sample.n_above == 0:
        sample.n_above = 1
        sample.d_above = 0
    sample.d_below = sample.d_above
    sample.n_below = sample.n_above
    exp = iadpython.Experiment(sample=sample)

    exp.d_beam = x[4]
    exp.rstd_r = x[5]
    exp.num_spheres = x[6]
    exp.method = 'substitution'

    if exp.num_spheres > 0:
        exp.r_sphere = iadpython.Sphere(x[7], x[8], x[9], x[10], 0, x[11])

    if exp.num_spheres > 0:
        exp.t_sphere = iadpython.Sphere(x[12], x[13], x[14], x[15], 0, x[16])

    exp.num_measures = 0
    if column_letters_str == '':
        exp.num_measures = x[17]
        data = x[18:]
    else:
        if 'r' in column_letters_str:
            exp.num_measures += 1
        if 't' in column_letters_str:
            exp.num_measures += 1
        if 'u' in column_letters_str:
            exp.num_measures += 1
        data = x[17:]

    fill_in_data(exp, data, column_letters_str)
    return exp
