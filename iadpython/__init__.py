"""
Forward and inverse adding-doubling radiative transport calculations.

Extensive documentation is at <https://iadpython.readthedocs.io>

`iadpython` is a pure Python module to do radiative transport calculations
in layered slabs.  Core forward and inverse calculations are available in
pure Python. Integrating-sphere support is being ported incrementally and
now includes single-sphere and double-sphere forward measurement formulas.

An example::

    >>> import iadpython as iad

    >>> mu_s = 10  # scattering coefficient [1/mm]
    >>> mu_a = 0.1 # absorption coefficient [1/mm]
    >>> g = 0.9    # scattering anisotropy
    >>> d = 1      # thickness mm

    >>> a = mu_s/(mu_a+mu_s)
    >>> b = mu_s/(mu_a+mu_s) * d

    >>> # air / glass / sample / glass / air
    >>> s = iadpython.Sample(a=a, b=b, g=g, n=1.4, n_above=1.5, n_below=1.5)
    >>> ur1, ut1, uru, utu = s.rt()

    >>> print('Collimated light incident perpendicular to sample')
    >>> print('  total reflection = %.5f' % ur1)
    >>> print('  total transmission = %.5f' % ut1)

    >>> print('Diffuse light incident on sample')
    >>> print('  total reflection = %.5f' % uru)
    >>> print('  total transmission = %.5f' % utu)
"""

__author__ = "Scott Prahl"
__email__ = "scott.prahl@oit.edu"
__copyright__ = "2018-2026, Scott Prahl"
__license__ = "MIT"
__url__ = "https://github.com/scottprahl/iadpython"
__version__ = "0.7.0"

from .constants import *
from .fresnel import *
from .start import *
from .ad import *
from .quadrature import *
from .combine import *
from .redistribution import *
from .sphere import *
from .nist import *
from .iad import *
from .grid import *
from .agrid import *
from .rxt import *
from .txt import *
from .port import *
from .double import *
