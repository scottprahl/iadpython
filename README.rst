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

`iadpython` will be a pure Python module to do forward and inverse multiple light
scattering (radiative transport) in layered materials.  Calculations are done using 
van de Hulst's adding-doubling technique.

The original adding-doubling algorithm was developed by van de Hulst to model light
propagation through layered media.  I extended it to handle Fresnel 
reflection at boundaries as well as interactions with integrating spheres. 

Finally, the code was further extended to handle lost light using
Monte Carlo techniques for inverse calculations.

Version v0.4.0 started the migration to a pure-python implementation.  This 
version includes the ability to do forward calculations of light transport through
layered 1D structures.  

The long-term goal is rewrite the integrating sphere, inverse algorithm, and
lost light calculations in pure python so that one can do 
inverse calculations (i.e., reflection and transmission measurements to 
intrinsic absorption and scattering properties). 

Both inverse and forward calculations are currently possible using the `iadc` framework.
This is a python interface to the inverse 
adding-doubling package written in C by Scott Prahl 
<https://github.com/scottprahl/iad>.  This works now
but is a nuisance to install and maintain because of its dependence on compiling
and installing a C library.

See <https://iadpython.readthedocs.io> for full documentation of `iadpython`.

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

To install::

    pip3 install iadpython

If you just want to do forward calculations then you're done.

If you want to do inverse calculations, then you'll need to build and
install the `libiad` library::

    git clone https://github.com/scottprahl/iad.git
    cd iad
    # edit Makefile as needed
    make install-lib


Dependencies
------------

Required Python modules: numpy, matplotlib, ctypes, scipy


License
-------

`iadpython` is licensed under the terms of the MIT license.