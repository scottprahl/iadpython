SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = docs/_build

check:
	-pyroma -d .
	-check-manifest
	make pylint
	make pydoc

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

pylint:
	-pylint iadpython/ad.py
	-pylint iadpython/combine.py
	-pylint iadpython/fresnel.py
	-pylint iadpython/quadrature.py
	-pylint iadpython/redistribution.py
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
	
pydoc:
	-pydocstyle iadpython/ad.py
	-pydocstyle iadpython/combine.py
	-pydocstyle iadpython/fresnel.py
	-pydocstyle iadpython/quadrature.py
	-pydocstyle iadpython/redistribution.py
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

test:
	python -m unittest iadpython/test_fresnel.py
	python -m unittest iadpython/test_quadrature.py
	python -m unittest iadpython/test_redistribution.py
	python -m unittest iadpython/test_start.py
	python -m unittest iadpython/test_layer.py
	python -m unittest iadpython/test_boundary.py
	python -m unittest iadpython/test_combo.py
	python -m unittest iadpython/test_ur1_uru.py
	python -m unittest iadpython/test_iadc.py

xtest:
	python -m unittest iadpython/test_iadpython.py
	
perf:
	python -m unittest iadpython/perf_test.py

clean:
	rm -rf dist
	rm -rf iadpython.egg-info
	rm -rf iadpython/__pycache__
	rm -rf .tox
	rm -rf docs/_build 
	rm -rf docs/api 

realclean:
	make clean

.PHONY: clean realclean test check pylint pydoc html