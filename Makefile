SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = docs/_build

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	open docs/_build/index.html

lintcheck:
	-pylint iadpython/ad.py
	-pylint iadpython/combine.py
	-pylint iadpython/constants.py
	-pylint iadpython/fresnel.py
	-pylint iadpython/grid.py
	-pylint iadpython/iad.py
	-pylint iadpython/nist.py
	-pylint iadpython/quadrature.py
	-pylint iadpython/redistribution.py
	-pylint iadpython/rxt.py
	-pylint iadpython/sphere.py
	-pylint iadpython/start.py
	-pylint tests/test_boundary.py
	-pylint tests/test_combo.py
	-pylint tests/test_fresnel.py
	-pylint tests/test_grid.py
	-pylint tests/test_iad.py
	-pylint tests/test_layer.py
	-pylint tests/test_layers.py
	-pylint tests/test_quadrature.py
	-pylint tests/test_redistribution.py
	-pylint tests/test_start.py
	-pylint tests/test_ur1_uru.py
	-pylint tests_iadc/test_iadc.py
	-pylint tests_iadc/test_performance.py

xpylint:
	-pylint iadpython/iadc.py
	-pylint tests/test_iadc.py
	
doccheck:
	-pydocstyle iadpython/ad.py
	-pydocstyle --ignore D301 iadpython/combine.py
	-pydocstyle iadpython/constants.py
	-pydocstyle --ignore D301 iadpython/iad.py
	-pydocstyle --ignore D301 iadpython/fresnel.py
	-pydocstyle iadpython/grid.py
	-pydocstyle iadpython/nist.py
	-pydocstyle iadpython/quadrature.py
	-pydocstyle iadpython/redistribution.py
	-pydocstyle iadpython/rxt.py
	-pydocstyle --ignore D301 iadpython/sphere.py
	-pydocstyle iadpython/start.py
	-pydocstyle tests/test_boundary.py
	-pydocstyle tests/test_combo.py
	-pydocstyle tests/test_fresnel.py
	-pydocstyle tests/test_grid.py
	-pydocstyle tests/test_iad.py
	-pydocstyle tests/test_layer.py
	-pydocstyle tests/test_layers.py
	-pydocstyle tests/test_quadrature.py
	-pydocstyle tests/test_redistribution.py
	-pydocstyle tests/test_start.py
	-pydocstyle tests/test_ur1_uru.py
	-pydocstyle tests/test_nist.py
	-pydocstyle tests_iadc/test_iadc.py
	-pydocstyle tests_iadc/test_performance.py

xpydoc:
	-pydocstyle iadpython/iadc.py
	-pydocstyle tests/test_iadc.py

notecheck:
	make clean
	pytest --notebooks tests/test_all_notebooks.py
	rm -rf __pycache__

rcheck:
	make doccheck
	make lintcheck
	make notecheck
	make test
	flake8 .
	pyroma -d .
	check-manifest

test:
	pytest tests

xtest:
	pytest tests_iadc

clean:
	rm -rf .pytest_cache
	rm -rf .tox
	rm -rf __pycache__
	rm -rf dist
	rm -rf iadpython.egg-info
	rm -rf iadpython/__pycache__
	rm -rf iadpython/__init__.pyc
	rm -rf iadpython/.ipynb_checkpoints
	rm -rf docs/_build 
	rm -rf docs/api 
	rm -rf docs/doi.org/
	rm -rf docs/.ipynb_checkpoints
	rm -rf tests/__pycache__
	rm -rf tests/tests_iadc/__pycache__

realclean:
	make clean

.PHONY: clean realclean test check pylint pydoc html