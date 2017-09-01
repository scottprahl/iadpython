# iadpython

iadpython is a Python module to calculate multiple light scattering (radiative
transport) of layered materials.  This is a thin wrapper over the inverse adding-doubling
(IAD) library.

## Usage
For examples and use cases, see test folder

## Installation

### First install the library

    git clone https://github.com/scottprahl/iad.git

    cd iad
    make install-lib

### Then install this python module

One way is to use pip

    pip install iadpython

Alternatively you can install from github

    git clone https://github.com/scottprahl/iadpython.git

Test by changing the iadpython directory and doing

    nosetests iadpython/test_iadpython.py

Then, add the iadpython directory to your PYTHONPATH or somehow


### Dependencies

For installation: setuptools

Required Python modules: numpy, matplotlib, cytpes


### License

iadpython is licensed under the terms of the MIT license.