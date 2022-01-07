# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=consider-using-f-string

"""
Module for reading rxt and txt files.

Two types of starting methods are possible.

    import iadpython

    filename = 'ink.rxt'
    exp = iadpython.read_iad_input(filename);

    filename = 'ink.txt'
    exp = iadpython.read_iad_output(filename)
"""

import re
import numpy as np
import iadpython

__all__ = ('read_rxt',
           'read_iad_output',
           )

def read_and_remove_notation(filename):
    """Read file and remove all whitespace and comments."""
    s = ''
    with open(filename) as f:
        for line in f:
            line = re.sub(r'#.*', '', line)
            line = re.sub(r'\s', ' ', line)
            line = re.sub(r',', ' ', line)
            s += line
   
    if len(re.findall('IAD1',s)) == 0:
        raise Exception("Not an .rxt file. (Does not start with IAD1)")
    
    s = re.sub(r'IAD1', '', s)
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'^\s+', '', s)
    s = re.sub(r'\s+$', '', s)
    return s
    
def read_rxt(filename):
    """
    Read a .rxt file.

    Args:
        filename: .rxt filename

    Returns:
        Experiment object
    """
    s = read_and_remove_notation(filename)
    x = np.array([float(value) for value in s.split(' ')])

    sample = iadpython.Sample()
    sample.a = None
    sample.b = None
    sample.g = None
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

	# m->num_measures = (*params >= 3) ? 3 : *params;
    exp.num_measures = x[17]	

    exp.lambda0 = np.zeros(0)
    if exp.num_measures >= 1:
        exp.m_r = np.zeros(0)

    if exp.num_measures >= 2:
        exp.m_t = np.zeros(0)

    if exp.num_measures >= 3:
        exp.m_u = np.zeros(0)

    count_per_line = 1
    for i in range(18,len(x)):

        if x[i] > 1:
            exp.lambda0 = np.append(exp.lambda0, x[i])
            continue
        if count_per_line == 1:
            exp.m_r = np.append(exp.m_r, x[i])
        elif count_per_line == 2:
            exp.m_t = np.append(exp.m_t, x[i])
        elif count_per_line == 3:
            exp.m_u = np.append(exp.m_u, x[i])
        if count_per_line >= exp.num_measures:
            count_per_line = 1
        else:
            count_per_line += 1

    if len(exp.lambda0) == 0:
        exp.lambda0 = None
    return exp


def read_iad_output(filename):
    """
    Read a .txt file produced by iad.

    Args:
        filename: .txt filename

    Returns:
        Experiment Object
    """
    exp = iadpython.Experiment()
    return exp
