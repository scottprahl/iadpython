"""
Sphinx configuration for iadpython documentation.

Uses:
- sphinx.ext.napoleon for Google-style docstrings
- nbsphinx for rendering pre-executed Jupyter notebooks
"""

from importlib.metadata import PackageNotFoundError, version as pkg_version

project = "iadpython"
root_doc = "index"

try:
    release = pkg_version(project)
except PackageNotFoundError:
    release = "0+unknown"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_automodapi.automodapi",
    "nbsphinx",
]
napoleon_use_param = False
napoleon_use_rtype = False
numpydoc_show_class_members = False

exclude_patterns = [
    "_build",
    ".ipynb_checkpoints",
    "iad-with-spheres.ipynb",
]

nbsphinx_execute = "never"
nbsphinx_allow_errors = True

html_theme = "sphinx_rtd_theme"
html_scaled_image_link = False
html_sourcelink_suffix = ""
