PYTHON ?= python
PIP ?= pip
PYTEST ?= pytest
CTAGS ?= ctags

# installation instructions are not clear to me:
#   setup.py install -> the package cannot be imported from outside the repository even though he is 
#   visible in the conda list as <develop> in the same way as when 
#   setup.py develop -> everything is ok
#install: clean-ctags
#	$(PYTHON) setup.py install

uninstall: clean
	$(PIP) uninstall coolpyplots

install:
	$(PIP) install .

develop: ctags
	$(PYTHON) setup.py develop

test:
	$(PYTEST) --cov-config=.coveragerc --cov-report term-missing --cov coolpyplots coolpyplots 

ctags:
	$(CTAGS) --python-kinds=-i --exclude=*/tests/* -R coolpyplots

clean:
	rm -f tags
	$(PYTHON) setup.py clean
	rm -rf coolpyplots.egg-info
	rm -rf dist
	rm -rf build
