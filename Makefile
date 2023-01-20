# simple makefile to simplify repetitive build env management tasks under posix
# taken from Makefile for mne-python and adapted for jumeg

PYTHON ?= python
PYTESTS ?= py.test
CTAGS ?= ctags
CODESPELL_SKIPS ?= "doc/_build,doc/auto_*,*.fif,*.eve,*.gz,*.tgz,*.zip,*.mat,*.stc,*.label,*.w,*.bz2,*.annot,*.sulc,*.log,*.local-copy,*.orig_avg,*.inflated_avg,*.gii,*.pyc,*.doctree,*.pickle,*.inv,*.png,*.edf,*.touch,*.thickness,*.nofix,*.volume,*.defect_borders,*.mgh,lh.*,rh.*,COR-*,FreeSurferColorLUT.txt,*.examples,.xdebug_mris_calc,bad.segments,BadChannels,*.hist,empty_file,*.orig,*.js,*.map,*.ipynb,searchindex.dat,plot_*.rst,*.rst.txt,c_EULA.rst*,*.html,gdf_encodes.txt,*.svg,references.bib,*.css,*.edf,*.bdf,*.vhdr"
CODESPELL_DIRS ?= jumeg/ doc/ examples/
all: clean inplace test test-doc

clean-pyc:
	find . -name "*.pyc" | xargs rm -f

clean-so:
	find . -name "*.so" | xargs rm -f
	find . -name "*.pyd" | xargs rm -f

clean-build:
	rm -rf build dist

clean-ctags:
	rm -f tags

clean-cache:
	find . -name "__pycache__" | xargs rm -rf

clean: clean-build clean-pyc clean-so clean-ctags clean-cache

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

wheel:
	$(PYTHON) setup.py sdist bdist_wheel

wheel_quiet:
	$(PYTHON) setup.py -q sdist bdist_wheel

pytest: test

test: in
	rm -f .coverage
	$(PYTESTS) -m 'not ultraslowtest' jumeg

test-verbose: in
	rm -f .coverage
	$(PYTESTS) -m 'not ultraslowtest' jumeg --verbose

test-fast: in
	rm -f .coverage
	$(PYTESTS) -m 'not slowtest' jumeg

test-full: in
	rm -f .coverage
	$(PYTESTS) jumeg

flake:
	@if command -v flake8 > /dev/null; then \
		echo "Running flake8"; \
		flake8 --count; \
	else \
		echo "flake8 not found, please install it!"; \
		exit 1; \
	fi;
	@echo "flake8 passed"

codespell:  # running manually
	@codespell --builtin clear,rare,informal,names,usage -w -i 3 -q 3 -S $(CODESPELL_SKIPS) --ignore-words=ignore_words.txt --uri-ignore-words-list=bu $(CODESPELL_DIRS)

codespell-error:  # running on travis
	@codespell --builtin clear,rare,informal,names,usage -i 0 -q 7 -S $(CODESPELL_SKIPS) --ignore-words=ignore_words.txt --uri-ignore-words-list=bu $(CODESPELL_DIRS)

pydocstyle:
	@echo "Running pydocstyle"
	@pydocstyle jumeg

pep:
	@$(MAKE) -k flake pydocstyle docstring codespell-error check-manifest nesting check-readme

build-doc:
	cd docs; make clean
	# cd docs; DISPLAY=:1.0 xvfb-run -n 1 -s "-screen 0 1280x1024x24 -noreset -ac +extension GLX +render" make html
	cd docs; make html

docstyle: pydocstyle
