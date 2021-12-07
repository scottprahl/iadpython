"""
Module for reading rxt and txt files.

Two types of starting methods are possible.

    import iadpython

    filename = 'ink.rxt'
    slab, method, data = iadpython.read_iad_input(filename);

    filename = 'ink.txt'
    slab, method, data = iadpython.read_iad_output(filename)
"""
__all__ = ('read_iad_input',
           'read_iad_output',
           )


def read_iad_input(filename):
    """
    Read a .rxt file.

    Args:
        filename: .rxt filename

    Returns:
        slab, method, data
    """
    return None, None, None


def read_iad_output(filename):
    """
    Read a .txt file produced by iad.

    Args:
        filename: .txt filename

    Returns:
        slab, method, data
    """
    return None, None, None
