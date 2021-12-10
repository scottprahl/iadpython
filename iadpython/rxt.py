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

import numpy as np
import iadpython

__all__ = ('read_iad_input',
           'read_iad_output',
           )


def read_iad_input(filename):
    """
    Read a .rxt file.

    Args:
        filename: .rxt filename

    Returns:
        Experiment object
    """
    exp = iadpython.Experiment()
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
