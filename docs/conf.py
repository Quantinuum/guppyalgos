# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import pathlib
import sys

sys.path.insert(0, pathlib.Path(__file__).parents[1].resolve().as_posix())

# Guppylang replaces its decorator with a lightweight dummy during Sphinx
# builds. Guppylang 1.0.1's dummy does not yet implement ``type_alias`` or
# keyword arguments for ``comptime``, which prevents autodoc from importing
# modules that define generic aliases or daggerable comptime helpers.
from guppylang import guppy

if not hasattr(guppy, "type_alias"):
    setattr(guppy, "type_alias", lambda *_args, **_kwargs: tuple)

    def dummy_comptime(*args: object, **_kwargs: object) -> object:
        """Accept both Guppy comptime decorator forms during docs builds."""
        if args:
            return args[0]
        return lambda function: function

    setattr(guppy, "comptime", dummy_comptime)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# Alias needed here to prevent naming clash with something else sphinx is doing.
from importlib.metadata import version as check_version

# Checking the version used in the Python environment means that pyproject.toml
# serves as the single source of truth for the version of guppyalgos.
project = "guppyalgos"
project_copyright = "2026, Quantinuum"
author = "Quantinuum"
release = check_version("guppyalgos")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


html_theme = "quantinuum_sphinx"
html_theme_options = {
    "sidebar_hide_name": False,
}

templates_path = ["_templates"]

master_doc = "index"

extensions = [
    "myst_nb",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx_math_dollar",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinxcontrib.tikz",
]

# Parse ``$$ ... $$`` display equations in MyST Markdown. Inline ``$...$``
# equations were already handled by sphinx-math-dollar.
myst_enable_extensions = ["dollarmath"]

# Resolve notebook links to sections, including third-level headings.
myst_heading_anchors = 3

# Render TikZ and Quantikz diagrams as SVGs in local and CI builds.
tikz_proc_suite = "pdf2svg"
tikz_tikzlibraries = "quantikz2"


mathjax3_config = {
  "loader": {"load": ["[tex]/braket"]},
  "tex": {
    "packages": {"[+]": ["braket"]},
    "inlineMath": [['\\(', '\\)']],
    "displayMath": [["\\[", "\\]"]],
  }
}

# __all__ dictates which classes and functions are shown in the docs
autosummary_ignore_module_all = False  # Respect __all__ if specified


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "hugr": ("https://quantinuum.github.io/hugr/", None),
    "zixy": ("https://quantinuum.github.io/zixy/", None),
    "guppylang": ("https://docs.quantinuum.com/guppy/", None),
}

exclude_patterns = ["build", "_build", "jupyter_execute", ".jupyter_cache"]

# Guppy uses compile-time natural-number variables in annotations. These are
# valid Guppy types but cannot be resolved as Python forward references by
# sphinx-autodoc-typehints.
suppress_warnings = ["sphinx_autodoc_typehints.forward_reference"]


autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,
}
napoleon_use_ivar = True

nb_execution_mode = "cache"
nb_execution_timeout = 120
nb_execution_raise_on_error = True

nb_execution_excludepatterns = [
    "examples/*",
    "examples/*/*",
    "examples/*/*/*",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
