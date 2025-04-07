"""
WDBX Documentation Configuration.

This file contains the configuration for Sphinx to generate WDBX documentation.
"""

import os
import sys
from datetime import datetime

# Add project root to path for proper module imports
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'WDBX'
author = 'WDBX Team'
copyright = '2023, WDBX Team'
version = '0.1.0'
release = version

# General configuration
extensions = [
    'sphinx.ext.autodoc',       # Generate API documentation from docstrings
    'sphinx.ext.viewcode',      # Add links to highlighted source code
    'sphinx.ext.napoleon',      # Support for NumPy and Google style docstrings
    'sphinx.ext.intersphinx',   # Link to other project's documentation
    'sphinx.ext.coverage',      # Check documentation coverage
    'sphinx.ext.todo',          # Support for TODO items
    'sphinx.ext.mathjax',       # Render math via MathJax
    'myst_parser',              # MyST parser for Markdown support
    'nbsphinx',                 # Jupyter notebook support
]

# Configure autodoc
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
}

# Configure napoleon (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True

# Configure intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'aiohttp': ('https://docs.aiohttp.org/en/stable/', None),
}

# Add any paths that contain templates
templates_path = ['_templates']

# Exclude patterns
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# The name of the Pygments (syntax highlighting) style to use
pygments_style = 'sphinx'

# HTML output configuration
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_logo = '_static/wdbx_logo.png'
html_favicon = '_static/favicon.ico'
html_title = "WDBX"

# PyData Sphinx Theme options
html_theme_options = {
    "logo": {
        "image_light": "_static/logo-light.png",
        "image_dark": "_static/logo-dark.png",
    },
    "show_toc_level": 2,
    "navbar_align": "left",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "use_edit_page_button": True,
    "github_url": "https://github.com/wdbx/wdbx-python",
    "icon_links": [
        {
            "name": "Discord",
            "url": "https://discord.gg/wdbx",
            "icon": "fab fa-discord",
        },
    ],
    "switcher": {
        "json_url": "https://wdbx.io/versions.json",
        "version_match": release,
    },
    "navigation_with_keys": True,
    "announcement": "This documentation is under active development!",
}

# GitHub repository information
html_context = {
    "github_user": "wdbx",
    "github_repo": "wdbx-python",
    "github_version": "main",
    "doc_path": "docs",
}

# LaTeX output configuration
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '11pt',
    'figure_align': 'htbp',
}

latex_documents = [
    (
        'index',
        'wdbx.tex',
        'WDBX Documentation',
        author,
        'manual',
    ),
]

# Manual page output
man_pages = [
    (
        'index',
        'wdbx',
        'WDBX Documentation',
        [author],
        1,
    ),
]

# Texinfo output configuration
texinfo_documents = [
    (
        'index',
        'wdbx',
        'WDBX Documentation',
        author,
        'WDBX',
        'Vector database and processing system.',
        'Databases',
    ),
]

# Epub output configuration
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright

# PDF configuration
pdf_documents = [
    ('index', 'wdbx', 'WDBX Documentation', author),
]

# Create .nojekyll file for GitHub Pages
html_extra_path = ['.nojekyll']

# Configure todo extension
todo_include_todos = True

# Enable linkcheck builder
linkcheck_ignore = [
    r'http://localhost',
]

# Markdown support
source_suffix = ['.rst', '.md']

# Add any extra paths that contain custom files
# (such as robots.txt or .htaccess)
html_extra_path = []

# Don't show module names in front of class names
add_module_names = False

# Enable numfig for figure numbering
numfig = True
numfig_format = {
    'figure': 'Figure %s',
    'table': 'Table %s',
    'code-block': 'Listing %s',
    'section': 'Section %s',
}

# Custom sidebar templates - not needed for PyData theme
# html_sidebars = {}

# Add custom files to include
html_css_files = [
    'custom.css',
]

html_js_files = [
    'custom.js',
]

# Make sure templates are found
templates_path = ['_templates']

# MyST parser configuration
myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'dollarmath',
    'fieldlist',
    'html_admonition',
    'html_image',
    'linkify',
    'replacements',
    'smartquotes',
    'substitution',
    'tasklist',
] 