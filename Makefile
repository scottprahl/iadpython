PACKAGE         := iadpython
GITHUB_USER     := scottprahl

# -------- venv config --------
PY_VERSION      ?= 3.12
VENV            ?= .venv
PY              := /opt/homebrew/opt/python@$(PY_VERSION)/bin/python$(PY_VERSION)
PYTHON          := $(VENV)/bin/python
PYPROJECT       := pyproject.toml

DOCS_DIR        := docs
HTML_DIR        := $(DOCS_DIR)/_build/html

ROOT            := $(abspath .)
OUT_ROOT        := $(ROOT)/_site
OUT_DIR         := $(OUT_ROOT)/$(PACKAGE)
STAGE_DIR       := $(ROOT)/.lite_src
DOIT_DB         := $(ROOT)/.jupyterlite.doit.db
LITE_CONFIG     := $(ROOT)/$(PACKAGE)/jupyter_lite_config.json

# --- GitHub Pages deploy config ---
PAGES_BRANCH    := gh-pages
WORKTREE        := .gh-pages
REMOTE          := origin

# --- server config (override on CLI if needed) ---
HOST            := 127.0.0.1
PORT            := 8000

PYTEST          := $(VENV)/bin/pytest
PYLINT          := $(VENV)/bin/pylint
SPHINX          := $(VENV)/bin/sphinx-build
RUFF            := $(VENV)/bin/ruff
BLACK           := $(VENV)/bin/black
CHECKMANIFEST   := $(VENV)/bin/check-manifest
PYROMA          := $(PYTHON) -m pyroma
RSTCHECK        := $(PYTHON) -m rstcheck
YAMLLINT        := $(PYTHON) -m yamllint

PYTEST_OPTS     := -q
SPHINX_OPTS     := -T -E -b html -d $(DOCS_DIR)/_build/doctrees -D language=en

.PHONY: help
help:
	@echo "Build Targets:"
	@echo "  dist           - Build sdist+wheel locally"
	@echo "  html           - Build Sphinx HTML documentation"
	@echo "  lab            - Start jupyterlab"
	@echo "  venv           - Create/provision the virtual environment ($(VENV))"
	@echo ""
	@echo "Test Targets:"
	@echo "  test           - Run pytest"
	@echo "  note-test      - Test all notebooks for errors"
	@echo ""
	@echo "Packaging Targets:"
	@echo "  lint           - Run pylint"
	@echo "  rcheck         - Distribution release checks"
	@echo "  manifest-check - Validate MANIFEST"
	@echo "  pylint-check   - Same as lint above"
	@echo "  pyroma-check   - Validate overall packaging"
	@echo "  black-check    - Check formatting with black"
	@echo "  rst-check      - Validate all RST files"
	@echo "  ruff-check     - Lint all .py and .ipynb files"
	@echo "  yaml-check     - Validate YAML files"
	@echo ""
	@echo "JupyterLite Targets:"
	@echo "  lite           - Build JupyterLite site into $(OUT_DIR)"
	@echo "  lite-serve     - Serve $(OUT_DIR) at http://$(HOST):$(PORT)"
	@echo "  lite-deploy    - Upload to github"
	@echo ""
	@echo "Clean Targets:"
	@echo "  clean          - Remove build caches and docs output"
	@echo "  lite-clean     - Remove JupyterLite outputs"
	@echo "  realclean      - clean + remove $(VENV)"

# venv bootstrap
$(VENV)/.ready: Makefile $(PYPROJECT)
	@echo "==> Ensuring venv at $(VENV) using $(PY)"
	@if [ ! -x "$(PY)" ]; then \
		echo "❌ Homebrew Python $(PY_VERSION) not found at $(PY)"; \
		echo "   Try: brew install python@$(PY_VERSION)"; \
		exit 1; \
	fi
	@if [ ! -d "$(VENV)" ]; then \
		"$(PY)" -m venv "$(VENV)"; \
	fi
	@$(PYTHON) -m pip -q install --upgrade pip wheel
	@echo "==> Installing iadpython + dev extras"
	@$(PYTHON) -m pip install -e ".[dev,docs,lite]"
	
	@touch "$(VENV)/.ready"
	@echo "✅ venv ready"

.PHONY: venv
venv: $(VENV)/.ready
	@:

.PHONY: dist
dist: venv
	$(PYTHON) -m build
	
.PHONY: test
test: venv
	$(PYTEST) $(PYTEST_OPTS) --ignore=tests/test_double.py tests

.PHONY: note-test
note-test: venv
	$(PYTEST) --verbose tests/test_all_notebooks.py
	@echo "✅ Notebook check complete"

.PHONY: html
html: venv
	@mkdir -p "$(HTML_DIR)"
	$(SPHINX) $(SPHINX_OPTS) "$(DOCS_DIR)" "$(HTML_DIR)"
	@command -v open >/dev/null 2>&1 && open "$(HTML_DIR)/index.html" || true

.PHONY: lint
lint: pylint-check

.PHONY: pylint-check
pylint-check: venv
	-@$(PYLINT) .github/scripts/update_citation.py
	-@$(PYLINT) iadpython/ad.py
	-@$(PYLINT) iadpython/agrid.py
	-@$(PYLINT) iadpython/cache.py
	-@$(PYLINT) iadpython/combine.py
	-@$(PYLINT) iadpython/constants.py
	-@$(PYLINT) iadpython/double.py
	-@$(PYLINT) iadpython/fresnel.py
	-@$(PYLINT) iadpython/grid.py
	-@$(PYLINT) iadpython/iad.py
	-@$(PYLINT) iadpython/iadcommand.py
	-@$(PYLINT) iadpython/nist.py
	-@$(PYLINT) iadpython/port.py
	-@$(PYLINT) --ignored-modules=scipy.special iadpython/quadrature.py
	-@$(PYLINT) --ignored-modules=scipy.special iadpython/redistribution.py
	-@$(PYLINT) iadpython/rxt.py
	-@$(PYLINT) iadpython/sphere.py
	-@$(PYLINT) iadpython/start.py
	-@$(PYLINT) iadpython/txt.py

	-@$(PYLINT) tests/test_boundary.py
	-@$(PYLINT) tests/test_combo.py
	-@$(PYLINT) tests/test_fresnel.py
	-@$(PYLINT) tests/test_grid.py
	-@$(PYLINT) tests/test_iad.py
	-@$(PYLINT) tests/test_layer.py
	-@$(PYLINT) tests/test_layers.py
	-@$(PYLINT) tests/test_nist.py
	-@$(PYLINT) tests/test_one_sphere.py
	-@$(PYLINT) tests/test_port.py
	-@$(PYLINT) tests/test_quadrature.py
	-@$(PYLINT) tests/test_redistribution.py
	-@$(PYLINT) tests/test_roundtrip_0_spheres.py
	-@$(PYLINT) tests/test_roundtrip_1_sphere.py
	-@$(PYLINT) tests/test_rxt.py
	-@$(PYLINT) tests/test_sphere.py
	-@$(PYLINT) tests/test_start.py
	-@$(PYLINT) tests/test_txt.py
	-@$(PYLINT) tests/test_ur1_uru.py
	-@$(PYLINT) tests_iadc/test_iadc.py
	-@$(PYLINT) tests_iadc/test_performance.py
	-@$(PYLINT) tests/test_all_notebooks.py

.PHONY: yaml-check
yaml-check: venv
	-@$(YAMLLINT) .github/workflows/citation.yaml
	-@$(YAMLLINT) .github/workflows/pypi.yaml
	-@$(YAMLLINT) .github/workflows/test.yaml
	-@$(YAMLLINT) .readthedocs.yaml

.PHONY: rst-check
rst-check: venv
	-@$(RSTCHECK) README.rst
	-@$(RSTCHECK) CHANGELOG.rst
	-@$(RSTCHECK) $(DOCS_DIR)/index.rst
	-@$(RSTCHECK) $(DOCS_DIR)/changelog.rst
	-@$(RSTCHECK) --ignore-directives automodapi $(DOCS_DIR)/$(PACKAGE).rst

.PHONY: ruff-check
ruff-check: venv
	$(RUFF) check

.PHONY: black-check
black-check: venv
	$(BLACK) --check .

.PHONY: manifest-check
manifest-check: venv
	$(CHECKMANIFEST)

.PHONY: pyroma-check
pyroma-check: venv
	$(PYROMA) -d .

.PHONY: rcheck
rcheck:
	@echo "Running all release checks..."
	@$(MAKE) realclean
	@$(MAKE) ruff-check
	@$(MAKE) black-check
	@$(MAKE) pylint-check
	@$(MAKE) rst-check
	@$(MAKE) manifest-check
	@$(MAKE) pyroma-check
	@$(MAKE) html
	@$(MAKE) lite
	@$(MAKE) dist
	@$(MAKE) test
	@$(MAKE) note-test
	@echo "✅ Release checks complete"
	
.PHONY: lite
lite: venv $(LITE_CONFIG)
	@echo "==> Building package wheel for PyOdide"
	@$(PYTHON) -m build

	@echo "==> Checking for .gh-pages worktree"
	@if [ -d "$(WORKTREE)" ]; then \
		echo "    Found .gh-pages worktree, removing..."; \
		git worktree remove "$(WORKTREE)" --force 2>/dev/null || true; \
		git worktree prune; \
		rm -rf "$(WORKTREE)"; \
		echo "    ✓ Removed"; \
	else \
		echo "    No .gh-pages worktree found"; \
	fi

	@echo "==> Cleaning previous builds"
	@/bin/rm -rf "$(OUT_ROOT)"
	@/bin/rm -rf "$(DOIT_DB)"
	@/bin/rm -rf ".doit.db"
	@/bin/rm -rf ".jupyterlite.doit.db.db"
	@echo "    ✓ Cleaned"

	@echo "==> Staging notebooks from docs -> $(STAGE_DIR)"
	@/bin/rm -rf "$(STAGE_DIR)"; mkdir -p "$(STAGE_DIR)"
	@if ls docs/*.ipynb 1> /dev/null 2>&1; then \
		/bin/cp docs/*.ipynb "$(STAGE_DIR)"; \
		echo "==> Clearing outputs from staged notebooks"; \
		"$(PYTHON)" -m jupyter nbconvert --clear-output --inplace "$(STAGE_DIR)"/*.ipynb; \
	else \
		echo "⚠️  No notebooks found in docs/"; \
	fi

	@echo "==> Building JupyterLite"
	@"$(PYTHON)" -m jupyter lite build \
		--config="$(LITE_CONFIG)" \
		--contents="$(STAGE_DIR)" \
		--output-dir="$(OUT_DIR)"

	@echo "==> Adding .nojekyll for GitHub Pages"
	@touch "$(OUT_DIR)/.nojekyll"
	
	@echo "✅ Build complete -> $(OUT_DIR)"

.PHONY: lite-serve
lite-serve: venv
	@test -d "$(OUT_DIR)" || { echo "❌ run 'make lite' first"; exit 1; }
	@echo "Serving at"
	@echo "   http://$(HOST):$(PORT)/$(PACKAGE)/?disableCache=1"
	@echo ""
	"$(PYTHON)" -m http.server -d "$(OUT_ROOT)" --bind $(HOST) $(PORT)

.PHONY: lite-deploy
lite-deploy: 
	@echo "==> Sanity check"
	@test -d "$(OUT_DIR)" || { echo "❌ Run 'make lite' first"; exit 1; }

	@echo "==> Ensure $(PAGES_BRANCH) branch exists"
	@if ! git show-ref --verify --quiet refs/heads/$(PAGES_BRANCH); then \
	  CURRENT=$$(git branch --show-current); \
	  git switch --orphan $(PAGES_BRANCH); \
	  git commit --allow-empty -m "Initialize $(PAGES_BRANCH)"; \
	  git switch $$CURRENT; \
	fi

	@echo "==> Setup deployment worktree"
	@git worktree remove "$(WORKTREE)" --force 2>/dev/null || true
	@git worktree prune || true
	@rm -rf "$(WORKTREE)"
	@git worktree add "$(WORKTREE)" "$(PAGES_BRANCH)"
	@git -C "$(WORKTREE)" pull "$(REMOTE)" "$(PAGES_BRANCH)" 2>/dev/null || true

	@echo "==> Deploy $(OUT_DIR) -> $(WORKTREE)"
	@rsync -a --delete --exclude ".git*" "$(OUT_DIR)/" "$(WORKTREE)/"
	@touch "$(WORKTREE)/.nojekyll"
	@date -u +"%Y-%m-%d %H:%M:%S UTC" > "$(WORKTREE)/.pages-ping"

	@echo "==> Commit & push"
	@cd "$(WORKTREE)" && \
	  git add -A && \
	  if git diff --quiet --cached; then \
	    echo "✅ No changes to deploy"; \
	  else \
	    git commit -m "Deploy $$(date -u +'%Y-%m-%d %H:%M:%S UTC')" && \
	    git push "$(REMOTE)" "$(PAGES_BRANCH)" && \
	    echo "✅ Deployed to https://$(GITHUB_USER).github.io/$(PACKAGE)/"; \
	  fi

.PHONY: lab
lab: venv
	@echo "==> Launching JupyterLab using venv ($(PYTHON))"
	"$(PYTHON)" -m jupyter lab --ServerApp.root_dir="$(CURDIR)"

.PHONY: clean
clean:
	@echo "==> Cleaning build artifacts"	
	@find . -name '__pycache__' -type d -exec rm -rf {} +
	@find . -name '.DS_Store' -type f -delete
	@find . -name '.ipynb_checkpoints' -type d -prune -exec rm -rf {} +
	@find . -name '.pytest_cache' -type d -prune -exec rm -rf {} +
	rm -rf .ruff_cache
	rm -rf $(PACKAGE).egg-info
	rm -rf docs/api
	rm -rf docs/_build
	rm -rf tests/charts
	rm -rf dist

.PHONY: lite-clean
lite-clean:
	@echo "==> Cleaning JupyterLite build artifacts"
	@/bin/rm -rf "$(STAGE_DIR)"
	@/bin/rm -rf "$(OUT_ROOT)"
	@/bin/rm -rf ".lite_root"
	@/bin/rm -rf "$(DOIT_DB)"
	@/bin/rm -rf "_output"

.PHONY: realclean
realclean: lite-clean clean
	@echo "==> Deep cleaning: removing venv and deployment worktree"
	@git worktree remove "$(WORKTREE)" --force 2>/dev/null || true
	@/bin/rm -rf "$(WORKTREE)"
	@/bin/rm -rf "$(VENV)"
