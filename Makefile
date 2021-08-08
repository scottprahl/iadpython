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
	-pylint iadpython/fresnel.py
	-pylint iadpython/quadrature.py
	-pylint iadpython/redistribution.py
	-pylint iadpython/sphere.py
	-pylint iadpython/start.py
	-pylint iadpython/test_boundary.py
	-pylint iadpython/test_combo.py
	-pylint iadpython/test_fresnel.py
	-pylint iadpython/test_quadrature.py
	-pylint iadpython/test_redistribution.py
	-pylint iadpython/test_start.py
	-pylint iadpython/test_ur1_uru.py
	-pylint iadpython/test_iadc.py
	-pylint iadpython/perf_test.py

xpylint:
	-pylint iadpython/iadc.py
	
doccheck:
	-pydocstyle iadpython/ad.py
	-pydocstyle iadpython/combine.py
	-pydocstyle iadpython/fresnel.py
	-pydocstyle iadpython/quadrature.py
	-pydocstyle iadpython/redistribution.py
	-pydocstyle iadpython/sphere.py
	-pydocstyle iadpython/start.py
	-pydocstyle iadpython/test_boundary.py
	-pydocstyle iadpython/test_combo.py
	-pydocstyle iadpython/test_fresnel.py
	-pydocstyle iadpython/test_quadrature.py
	-pydocstyle iadpython/test_redistribution.py
	-pydocstyle iadpython/test_start.py
	-pydocstyle iadpython/test_ur1_uru.py
	-pydocstyle iadpython/test_iadc.py
	-pydocstyle iadpython/perf_test.py

xpydoc:
	-pydocstyle iadpython/iadc.py

notecheck:
	make clean
	pytest --verbose -n 4 tests/test_all_notebooks.py
	rm -rf __pycache__

rcheck:
	make doccheck
	make lintcheck
	make notecheck
	make test
	flake8
	pyroma -d .
	check-manifest

test:
	pytest tests

xtest:
	pytest --runbinary tests

clean:
	rm -rf dist
	rm -rf iadpython.egg-info
	rm -rf iadpython/__pycache__
	rm -rf .tox
	rm -rf docs/_build 
	rm -rf docs/api 
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf test/__pycache__

realclean:
	make clean

.PHONY: clean realclean test check pylint pydoc html