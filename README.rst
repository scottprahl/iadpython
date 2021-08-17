iadpython
=========

.. image:: https://img.shields.io/pypi/v/iadpython.svg
   :target: https://pypi.org/project/iadpython/

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/scottprahl/iadpython/blob/master

.. image:: https://img.shields.io/badge/readthedocs-latest-blue.svg
   :target: https://iadpython.readthedocs.io

.. image:: https://img.shields.io/badge/github-code-green.svg
   :target: https://github.com/scottprahl/iadpython

.. image:: https://img.shields.io/badge/BSD-license-yellow.svg
   :target: https://github.com/scottprahl/iadpython/blob/master/LICENSE.txt

.. image:: https://github.com/scottprahl/iadpython/actions/workflows/test.yml/badge.svg
   :target: https://github.com/scottprahl/iadpython/actions/workflows/test.yml

__________

iadpython will be a pure Python module to do forward and inverse multiple light
scattering (radiative transport) in layered materials.  Calculations are done using 
van de Hulst's adding-doubling technique.

The original adding-doubling algorithm was developed by van de Hulst to model light
propagation through layered media.  I extended it to handle Fresnel 
reflection at boundaries as well as interactions with integrating spheres. 
Finally, the code was further extended to handle lost light using
Monte Carlo techniques.

Version v0.4.0 started the migration to a pure-python implementation.  This 
version includes a tested forward calculation of light transport through
a layered 1D structure.  

The long-term goal is rewrite the integrating sphere, inverse algorithm, and
lost light calculations in pure python so that one can do 
inverse calculations (i.e., reflection and transmission measurements to 
intrinsic absorption and scattering properties). 

Both inverse and forward calculations are currently possible using the `iadc` framework.
This is a python interface to the inverse 
adding-doubling package written in C by Scott Prahl 
<https://github.com/scottprahl/iad>.  This works now
but is a nuisance to install an maintain because of the dependence on the 
C library.

See <https://iadpython.readthedocs.io> for full documentation.

Usage
-----

The following will do a forward calculation::

    import iadpython as iad

    mu_s = 10  # scattering coefficient [1/mm]
    mu_a = 0.1 # absorption coefficient [1/mm]
    g = 0.9    # scattering anisotropy
    d = 1      # thickness mm

    a = mu_s/(mu_a+mu_s)
    b = mu_s/(mu_a+mu_s) * d

    # air / glass / sample / glass / air
    s = iadpython.Sample(a=a, b=b, g=g, n=1.4, n_above=1.5, n_below=1.5)
    ur1, ut1, uru, utu = s.rt()

    print('Collimated light incident perpendicular to sample')
    print('  total reflection = %.5f' % ur1)
    print('  total transmission = %.5f' % ut1)
 
    print('Diffuse light incident on sample')
    print('  total reflection = %.5f' % uru)
    print('  total transmission = %.5f' % utu)


Installation
------------

If you want the pure python version then just do

    pip install iadpython
    
If you want to use the `iadc` module that allows both forward and inverse
calculation, then you will need to first install the `iad` library and build
the library.

    git clone https://github.com/scottprahl/iad.git
    cd iad
    # edit Makefile as neede
    make install-lib

Then install this python module using `pip`

    pip install --user iadpython

Test by changing the iadpython directory and try doing

    ad -a 0.5


Dependencies
------------
For installation: setuptools

Required Python modules: numpy, matplotlib, ctypes, scipy


License
-------

iadpython is licensed under the terms of the MIT license.