"""
Module for importing reflection spectra from NIST database.

https://nvlpubs.nist.gov/nistpubs/jres/122/jres.122.026.pdf

Two types of starting methods are possible.

    Example:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> import iadpython as iad

        # Retrieve and plot subject 5

        >>> subject_number = 5
        >>> lambda0, R = iad.subject_average_reflectance(subject_number)
        >>> plt.plot(lambda0, R)
        >>> plt.xlabel("Wavelength (nm)")
        >>> plt.ylabel("Total Reflectance")
        >>> plt.title("Subject #%d" % subject_number)
        >>> plt.show()

        # Retrieve and plot all subjects

        >>> lambda0, R = iad.all_average_reflectances()
        >>> for i in range(100):
        >>>     plt.plot(lambda0, R[:, i])
        >>> plt.xlabel("Wavelength (nm)")
        >>> plt.ylabel("Total Reflectance")
        >>> plt.title("All Subjects")
        >>> plt.show()
"""

import pkg_resources
import numpy as np

# nist_db = 'https://doi.org/10.18434/M38597'

__all__ = ('subject_reflectances',
           'subject_average_reflectance',
           'all_average_reflectances',
           )


def subject_reflectances(subject_number):
    """Extract all reflection data for one subject."""
    if subject_number <= 0 or subject_number > 100:
        raise Exception("subject_number must be 1 to 100")

    col = (subject_number - 1) * 4 + 1

    cols = (0, col, col + 1, col + 2, col + 3)

    nist_db = pkg_resources.resource_filename(__name__, 'data/M38597.csv')
    data = np.loadtxt(nist_db, skiprows=8, usecols=cols, delimiter=',', encoding='latin1')

    lambda0 = data[:, 0]
    r_1 = data[:, 1]
    r_2 = data[:, 2]
    r_3 = data[:, 3]
    r_ave = data[:, 4]

    return lambda0, r_1, r_2, r_3, r_ave


def subject_average_reflectance(subject_number):
    """Extract average reflection for one subject."""
    if subject_number <= 0 or subject_number > 100:
        raise Exception("subject_number must be 1 to 100")

    col = (subject_number - 1) * 4 + 4

    cols = (0, col)

    nist_db = pkg_resources.resource_filename(__name__, 'data/M38597.csv')
    data = np.loadtxt(nist_db, skiprows=8, usecols=cols, delimiter=',', encoding='latin1')

    lambda0 = data[:, 0]
    r_ave = data[:, 1]

    return lambda0, r_ave


def all_average_reflectances():
    """Extract average reflectance for all subjects."""
    cols = [4 * i for i in range(101)]

    nist_db = pkg_resources.resource_filename(__name__, 'data/M38597.csv')
    data = np.loadtxt(nist_db, skiprows=8, usecols=cols, delimiter=',', encoding='latin1')

    lambda0 = data[:, 0]
    r_ave = data[:, 1:]

    return lambda0, r_ave
