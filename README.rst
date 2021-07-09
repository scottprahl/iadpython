iadpython
=========

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/scottprahl/iadpython/blob/master

.. image:: https://img.shields.io/badge/readthedocs-latest-blue.svg
   :target: https://iadpython.readthedocs.io

.. image:: https://img.shields.io/badge/github-code-green.svg
   :target: https://github.com/scottprahl/iadpython

.. image:: https://img.shields.io/badge/BSD-license-yellow.svg
   :target: https://github.com/scottprahl/iadpython/blob/master/LICENSE.txt

__________

iadpython will ultimately be a pure Python module to calculate multiple light scattering 
(radiative transport) in layered materials.  Calculations are done using 
van de Hulst's adding-doubling technique.

Currently, the pure-python code only does the forward calculation.  The long-term goal is
to do the inverse calculation (reflection and transmission measurements to 
intrinsic absorption and scattering properties). 

To do inverse calculations, a python interface to the inverse 
adding-doubling package written in C by Scott Prahl.  This works nows
but is a nuisance to install an maintain because of the dependence on the 
C library.

The original adding-doubling was developed by van de Hulst to model light
propagation through layered media.  It was extended to handle Fresnel 
reflection at boundaries as well as interactions with integrating spheres. 
Finally, the code was further extended to handle lost light by including 
Monte Carlo techniques.


Usage
-----

    import iadpython as iad
    
    albedo = 0.8
    anisotropy = 0.9
    optical_thickness = 2.0
    
    UR1, UT1 = iad.
    
For examples and use cases, see the `docs` folder on github or view
iadpython.readthedocs.com

Installation
------------

First install the `iad` library

    git clone https://github.com/scottprahl/iad.git

    cd iad
    make install-lib

Then install this python module using `pip`

    pip install --user iadpython

Test by changing the iadpython directory and doing

    prompt> ad -a 0.5

Then, add the iadpython directory to your PYTHONPATH or somehow


Dependencies
------------
For installation: setuptools

Required Python modules: numpy, matplotlib, ctypes, scipy


License
-------

iadpython is licensed under the terms of the MIT license.