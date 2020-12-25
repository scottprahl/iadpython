iadpython
=========

iadpython is a Python module to calculate multiple light scattering (radiative
transport) of layered materials.  This is a python interface to the inverse 
adding-doubling package written
in C by Scott Prahl.  This allows users to extract the intrinisic optical 
properties of materials from measurements of total reflected and total 
transmitted light.

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

    pip install iadpython

Alternatively you can install from github

    git clone https://github.com/scottprahl/iadpython.git

Test by changing the iadpython directory and doing

    make test

Then, add the iadpython directory to your PYTHONPATH or somehow


Dependencies
------------
For installation: setuptools

Required Python modules: numpy, matplotlib, ctypes, scipy


License
-------

iadpython is licensed under the terms of the MIT license.