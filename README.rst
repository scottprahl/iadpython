.. |pypi| image:: https://img.shields.io/pypi/v/iadpython?color=68CA66
   :target: https://pypi.org/project/iadpython/
   :alt: pypi

.. |github| image:: https://img.shields.io/github/v/tag/scottprahl/iadpython?label=github&color=68CA66
   :target: https://github.com/scottprahl/iadpython
   :alt: github

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/iadpython?label=conda&color=68CA66
   :target: https://github.com/conda-forge/iadpython-feedstock
   :alt: conda

.. |doi| image:: https://zenodo.org/badge/102148844.svg
   :target: https://zenodo.org/badge/latestdoi/102148844
   :alt: doi

.. |license| image:: https://img.shields.io/github/license/scottprahl/iadpython?color=68CA66
   :target: https://github.com/scottprahl/iadpython/blob/main/LICENSE.txt
   :alt: License

.. |test| image:: https://github.com/scottprahl/iadpython/actions/workflows/test.yaml/badge.svg
   :target: https://github.com/scottprahl/iadpython/actions/workflows/test.yaml
   :alt: Testing

.. |docs| image:: https://readthedocs.org/projects/iadpython/badge
   :target: https://iadpython.readthedocs.io
   :alt: Docs

.. |downloads| image:: https://img.shields.io/pypi/dm/iadpython?color=68CA66
   :target: https://pypi.org/project/iadpython/
   :alt: Downloads

iadpython
=========

by Scott Prahl

|pypi| |github| |conda| |doi|

|license| |test| |docs| |downloads|

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
    b = (mu_a+mu_s) * d

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

Use ``pip``::

    pip install iadpython

or ``conda``::

    conda install -c conda-forge iadpython

or use immediately by clicking the Google Colaboratory button below

.. image:: https://colab.research.google.com/assets/colab-badge.svg
  :target: https://colab.research.google.com/github/scottprahl/iadpython/blob/main
  :alt: Colab

Inverse Calculations
---------------------

As of version 0.5.3, the forward and inverse calculations work fine as long as you do not need to
include integrating sphere effects or lost-light calculations.

If you want to do these, then you're probably best served by downloading and compiling
the C-code (on unix or macos) or the `.exe` file for Windows.  <https://omlc.org/software/iad/>


Dependencies
------------

Required Python modules: numpy, matplotlib, ctypes, scipy


License
-------

`iadpython` is licensed under the terms of the MIT license.
