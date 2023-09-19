"""Module for importing reflection spectra from NIST database.

Example::

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import iadpython as iad

    >>> # Retrieve and plot subject 5

    >>> subject_number = 5
    >>> lambda0, R = iad.subject_average_reflectance(subject_number)
    >>> plt.plot(lambda0, R)
    >>> plt.xlabel("Wavelength (nm)")
    >>> plt.ylabel("Total Reflectance")
    >>> plt.title("Subject #%d" % subject_number)
    >>> plt.show()

    >>> # Retrieve and plot all subjects

    >>> lambda0, R = iad.all_average_reflectances()
    >>> for i in range(100):
    >>>     plt.plot(lambda0, R[:, i])
    >>> plt.xlabel("Wavelength (nm)")
    >>> plt.ylabel("Total Reflectance")
    >>> plt.title("All Subjects")
    >>> plt.show()

Reference:
    <https://nvlpubs.nist.gov/nistpubs/jres/122/jres.122.026.pdf>
    nist_db = 'https://doi.org/10.18434/M38597'
"""

import os
import numpy as np

__all__ = ('subject_reflectances',
           'subject_average_reflectance',
           'all_average_reflectances',
           )

def get_subject_data(cols):
    """
    Load and return data from a CSV file based on specified columns.

    This function reads data from a CSV file located in the 'data' directory relative to the script's location.
    It allows you to specify which columns of data to extract from the file.

    Args:
        cols: A tuple of column indices to extract from the CSV file.

    Returns:
        An array containing the requested data.

    Example:
        To extract columns 1 and 2 from the CSV file, you can call the function like this:
        >>> cols_to_extract = (1, 2)
        >>> data = get_subject_data(cols_to_extract)

    Note:
        - The data file is expected to have headers with at least 8 rows of metadata before the data.
        - The delimiter for the CSV file is assumed to be ',' (comma).
        - The encoding of the file is assumed to be 'latin1'.
    """
    script_dir = os.path.dirname(__file__)  # Path to directory of this file
    data_file_path = os.path.join(script_dir, 'data', 'M38597.csv')
    data = np.loadtxt(data_file_path, skiprows=8, usecols=cols, delimiter=',', encoding='latin1')
    return data


def subject_reflectances(subject_number):
    """Extract all reflection data for one subject."""
    if subject_number <= 0 or subject_number > 100:
        raise ValueError("subject_number must be 1 to 100")

    col = (subject_number - 1) * 4 + 1

    cols = (0, col, col + 1, col + 2, col + 3)

    data = get_subject_data(cols)

    lambda0 = data[:, 0]
    r_1 = data[:, 1]
    r_2 = data[:, 2]
    r_3 = data[:, 3]
    r_ave = data[:, 4]

    return lambda0, r_1, r_2, r_3, r_ave


def subject_average_reflectance(subject_number):
    """Extract average reflection for one subject."""
    if subject_number <= 0 or subject_number > 100:
        raise ValueError("subject_number must be 1 to 100")

    col = (subject_number - 1) * 4 + 4

    cols = (0, col)

    data = get_subject_data(cols)

    lambda0 = data[:, 0]
    r_ave = data[:, 1]

    return lambda0, r_ave


def all_average_reflectances():
    """Extract average reflectance for all subjects."""
    cols = [4 * i for i in range(101)]

    data = get_subject_data(cols)

    lambda0 = data[:, 0]
    r_ave = data[:, 1:]

    return lambda0, r_ave
