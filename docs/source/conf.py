# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

# sys.path.insert(0, os.path.abspath("../../transformerx/"))
from datetime import datetime
from pygments.styles import get_style_by_name

PYTHONPATH = "../../transformerx/"
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "TransformerX"
copyright = "2023, TensorOps"
author = "TensorOps"
release = "v1.0.0-rc"

# style = get_style_by_name("friendly")
# style.background_color = "#f3f2f1"
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_markdown_builder",
]

templates_path = ["_templates"]

napoleon_use_rtype = False

napoleon_include_init_with_doc = True
napoleon_google_docstring = True
napoleon_use_param = True
napoleon_use_ivar = True

# pygments_style = "friendly"

language = "english"


exclude_patterns = []


html_theme = "sphinx_rtd_theme"
html_title = "TransformerX Documentation"
html_show_sourcelink = False
html_baseurl = "https://github.com/tensorops/transformerx"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "alabaster"
html_static_path = ["_static"]

html_theme_options = {
    "enable_search_shortcuts": True,
    "globaltoc_collapse": True,
    "prev_next_buttons_location": "both",
    # "style_nav_header_background": "#F5A603",
    "navigation_depth": 2,
    "collapse_navigation": True,
    "sticky_navigation": False,
    "logo_only": False,
    "display_version": True,
    "style_external_links": True,
    "titles_only": True,
}

napoleon_use_param = False
