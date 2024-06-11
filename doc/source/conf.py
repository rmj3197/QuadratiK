# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, basedir)

project = "QuadratiK"
copyright = (
    "2023, Giovanni Saraceno, Marianthi Markatou, Raktim Mukhopadhyay, Mojgan Golzy"
)
author = "Giovanni Saraceno, Marianthi Markatou, Raktim Mukhopadhyay, Mojgan Golzy"
release = "1.1.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.githubpages",
    "nbsphinx",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
strip_signature_backslash = True

# -- Options for Latex output -------------------------------------------------
latex_elements = {
    "extraclassoptions": "openany,oneside",
    "preamble": r"""
    \usepackage{fancyhdr}
    \makeatletter
    \fancypagestyle{normal}{
        \fancyhf{}
        \fancyfoot[R]{\py@HeaderFamily\thepage}
        \renewcommand{\headrulewidth}{0pt}
        \renewcommand{\footrulewidth}{0.4pt}
    }
    \makeatother
    """,
}
latex_documents = [
    (
        "index",
        "QuadratiK.tex",
        "QuadratiK",
        author.replace("Raktim", "\\and Raktim"),
        "manual",
    )
]
