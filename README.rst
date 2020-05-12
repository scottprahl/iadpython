iadpython
========

iadpython is a Python module to calculate multiple light scattering (radiative
transport) of layered materials.  This is python wrapper for the inverse adding-doubling
(IAD) library.

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

    nosetests iadpython/test_iadpython.py

Then, add the iadpython directory to your PYTHONPATH or somehow


Dependencies
------------
For installation: setuptools

Required Python modules: numpy, matplotlib, cytpes


License
-------

iadpython is licensed under the terms of the MIT license.