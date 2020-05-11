check:
	-pyroma -d .
	-check-manifest
	make pylint
	make pep257

pylint:
	-pylint iadpython/iadpython.py
	
pep257:
	-pep257 iadpython/iadpython.py

test:
	nosetests iadpython/test_iadpython.py
	
clean:
	rm -rf dist
	rm -rf iadpython.egg-info
	rm -rf iadpython/__pycache__

realclean:
	make clean
	
.PHONY: clean realclean test check pylint pep257