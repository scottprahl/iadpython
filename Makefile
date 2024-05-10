SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = docs/_build

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	open docs/_build/index.html

lint:
	-pylint iadpython/ad.py
	-pylint iadpython/combine.py
	-pylint iadpython/constants.py
	-pylint iadpython/fresnel.py
	-pylint iadpython/grid.py
	-pylint iadpython/iad.py
	-pylint iadpython/nist.py
	-pylint --ignored-modules=scipy.special iadpython/quadrature.py
	-pylint --ignored-modules=scipy.special iadpython/redistribution.py
	-pylint iadpython/rxt.py
	-pylint iadpython/sphere.py
	-pylint iadpython/port.py
	-pylint iadpython/start.py
	-pylint iadpython/txt.py

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

lintc:
	-pylint iadpython/iadc.py
	-pylint tests_iadc/test_iadc.py

notetest:
	make clean
	pytest --notebooks tests/test_all_notebooks.py
	rm -rf __pycache__

test:
	pytest --verbose tests/test_boundary.py
	pytest --verbose tests/test_combo.py
	pytest --verbose tests/test_fresnel.py
	pytest --verbose tests/test_grid.py
	pytest --verbose tests/test_layer.py
	pytest --verbose tests/test_layers.py
	pytest --verbose tests/test_nist.py
	pytest --verbose tests/test_one_sphere.py
	pytest --verbose tests/test_quadrature.py
	pytest --verbose tests/test_redistribution.py
	pytest --verbose tests/test_rxt.py
	pytest --verbose tests/test_txt.py
	pytest --verbose tests/test_sphere.py
	pytest --verbose tests/test_port.py
	pytest --verbose tests/test_start.py
	pytest --verbose tests/test_ur1_uru.py
	pytest --verbose tests/test_iad.py

testc:
	pytest --verbose tests_iadc/test_iadc.py
	pytest --verbose tests_iadc/test_performance.py

rcheck:
	-ruff check
	-flake8 .
	make lint
	make lintc
	make test
	make testc
	pyroma -d .
	check-manifest
	make notetest

clean:
	rm -rf .pytest_cache
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

.PHONY: clean realclean test lintpylint pydoc html
