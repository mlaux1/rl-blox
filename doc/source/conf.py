import sys

sys.path.extend(["..", "../.."])

import rl_blox

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "RL-BLOX"
copyright = "2025, Melvin Laux, Alexander Fabisch"
author = "Melvin Laux, Alexander Fabisch"
release = rl_blox.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
]

templates_path = ['_templates']
exclude_patterns = []

# theme
html_logo = "_static/rl_blox_logo_v1.png"
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "logo": {
        "alt_text": f"{project} {release}",
    },
    "collapse_navigation": True,
}
html_static_path = ["_static"]
html_show_sourcelink = False
html_show_sphinx = False
html_show_copyright = True

# autodoc
autodoc_default_options = {"member-order": "bysource"}
autosummary_generate = True  # generate files at doc/source/_apidoc
class_members_toctree = False
numpydoc_show_class_members = False

# intersphinx
intersphinx_mapping = {
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}
intersphinx_timeout = 10
