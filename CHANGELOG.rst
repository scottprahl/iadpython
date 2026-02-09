Changelog for `iadpython` package
=================================

0.7.0
-----
* align build, packaging, CI, and docs infrastructure with miepython patterns
* move JupyterLite configuration into package-managed location
* integrate adaptive AGrid warm starts for inverse-search acceleration
* align `iadcommand` behavior more closely with legacy `iad` CLI
* improve wavelength handling to match legacy `iad` behavior
* add and expand round-trip coverage (including one-sphere workflows)
* increase minimum supported Python version to 3.10
* improve linting quality and add/expand pylint checks
* port two-sphere integrating-sphere gain and normalization equations from CWEB/C into `DoubleSphere`
* wire two-sphere forward measurement path into `Experiment.measured_rt()` for `num_spheres == 2`
* fix `DoubleSphere.do_one_photon()` return contract to return two values by default
* expand two-sphere tests to cover CWEB-equation parity, normalization anchors, and experiment wiring

0.6.0
------
* fully support single integrating spheres
* add command-line support
* add Monte Carlo sphere simulations
* add Port class for Sphere class
* change entrance port to empty port
* improve sphere-single.ipynb
* add sphere-random.ipynb
* improve single sphere testing
* add port testing
* add notebook on measurements
* rearrange documentation


0.5.3
------
* add github actions
* add citation with zenodo DOI
* add copyright to docs
* add conda support
* improve badges
* add github auto testing
* lint files
* start fixing math in docstrings
* remove tox

v0.5.2
------
* fix search, one-sphere round-trips now
* fix packaging
* fix html generation
* linting

v0.5.1
------
* inverse calculation works for 0 spheres
* much more testing
* basic multiple layers support
* single location for version number
* revert to sphinx_rtd_theme

v0.4.0
------
* forward adding-doubling calculation is pure python now
* create pure python packages
* include wheel file
* package as python3 only
* use sphinx_book_theme for docs

v0.3.0
------
* improve packaging
* improve documentation with Sphinx
* use tox
* add better diagnostics for finding libiad library

v0.2.5
------
* add doc files to distribution
* promise only python 3 and non-zip
* improve long description

v0.2.4
* fix bugs found by C Regan
* (unreleased??)

v0.2.3
------
* fix __init__.py so "import iadpython" works

v0.2.2
------
* necessary because setup.py version did not get updated

v0.2.1
------
* remove ctypes as a requirement

v0.2.0
------
* initial checkin
* Convert to markdown format
* update to correct contents
* add ctypes as dependancy, fix test path
* update my notes on how to cut a release
