SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = docs/_build

check:
	-pyroma -d .
	-check-manifest
	make pylint
	make pydoc

pylint:
	-pylint iadpython/quadrature.py
	-pylint iadpython/redistribution.py
	-pylint iadpython/start.py
	-pylint iadpython/test_quadrature.py
	-pylint iadpython/test_redistribution.py
	-pylint iadpython/test_start.py
#	-pylint iadpython/iadpython.py
#	-pylint iadpython/test_iadpython.py
	
pydoc:
	-pydocstyle iadpython/quadrature.py
	-pydocstyle iadpython/redistribution.py
	-pydocstyle iadpython/start.py
	-pydocstyle iadpython/test_quadrature.py
	-pydocstyle iadpython/test_redistribution.py
	-pydocstyle iadpython/test_start.py
#	-pydocstyle iadpython/iadpython.py
#	-pydocstyle iadpython/test_iadpython.py

test:
	nosetests iadpython/test_start.py
	nosetests iadpython/test_redistribution.py
	nosetests iadpython/test_quadrature.py
#	nosetests iadpython/test_iadpython.py
	
clean:
	rm -rf dist
	rm -rf iadpython.egg-info
	rm -rf iadpython/__pycache__
	rm -rf .tox

realclean:
	make clean

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	
.PHONY: clean realclean test check pylint pep257 html