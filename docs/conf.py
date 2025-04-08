"""
WDBX Documentation Configuration.

This file contains the configuration for Sphinx to generate WDBX documentation.
It defines all settings needed for building comprehensive documentation in multiple formats.
"""

import os
import sys
from datetime import datetime

from pygments.lexers.data import IniLexer, TomlLexer, YamlLexer
from pygments.lexers.markup import MarkdownLexer, RstLexer, TexLexer
from pygments.lexers.python import Python3Lexer, PythonLexer
from pygments.lexers.shell import BashLexer, PowerShellLexer
from pygments.lexers.sql import SqlLexer
from pygments.lexers.text import DiffLexer
from pygments.lexers.web import CssLexer, HtmlLexer, JsLexer, JsonLexer
from sphinx.highlighting import lexers

# Add project root to Python path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Project information
project = "WDBX"
copyright = f"{datetime.now().year}, WDBX Team"
author = "WDBX Team"
release = "1.3.0"
version = "1.0.0"

# General configuration
extensions = [
    # Core Sphinx extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.ifconfig",
    "sphinx.ext.extlinks",
    # Third-party extensions
    "myst_parser",
    "sphinx_sitemap",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_togglebutton",
    "sphinx_search.extension",
    "sphinx_autodoc_typehints",
    "sphinx_rtd_theme",
    "sphinxcontrib.mermaid",
    "sphinxcontrib.plantuml",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.programoutput",
    "sphinxcontrib.images",
    "sphinxcontrib.youtube",
    "sphinxcontrib.redirects",
]

# Add any paths that contain templates here
templates_path = ["_templates"]

# List of patterns to exclude from source
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "env/**",
    "venv/**",
    ".venv/**",
    "**/.git/**",
    "**/__pycache__/**",
    "**/.pytest_cache/**",
    "**/node_modules/**",
    "**/dist/**",
    "**/build/**",
    "**/coverage/**",
]

# The suffix of source filenames
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
    ".txt": "markdown",
    ".py": "python",
    ".ipynb": "nbsphinx",
}

# The master toctree document
master_doc = "index"

# Configure code block lexers
lexers["python"] = PythonLexer(startinline=True)
lexers["python3"] = Python3Lexer(startinline=True)
lexers["json"] = JsonLexer(startinline=True)
lexers["yaml"] = YamlLexer(startinline=True)
lexers["toml"] = TomlLexer(startinline=True)
lexers["ini"] = IniLexer(startinline=True)
lexers["markdown"] = MarkdownLexer(startinline=True)
lexers["rst"] = RstLexer(startinline=True)
lexers["bash"] = BashLexer(startinline=True)
lexers["powershell"] = PowerShellLexer(startinline=True)
lexers["shell"] = BashLexer(startinline=True)
lexers["sql"] = SqlLexer(startinline=True)
lexers["html"] = HtmlLexer(startinline=True)
lexers["css"] = CssLexer(startinline=True)
lexers["javascript"] = JsLexer(startinline=True)
lexers["diff"] = DiffLexer(startinline=True)
lexers["tex"] = TexLexer(startinline=True)

# The name of the Pygments (syntax highlighting) style to use
pygments_style = "sphinx"
highlight_language = "python3"

# Configure autodoc
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
    "inherited-members": False,
    "private-members": False,
    "ignore-module-all": False,
    "imported-members": False,
    "show-inheritance-diagram": True,
    "inherited-members-show": False,
}

# Add mock imports for missing modules
autodoc_mock_imports = [
    "wdbx.utils.logging",
    "wdbx.client",
    "wdbx.utils",
    "numpy",
    "pandas",
    "torch",
    "matplotlib",
    "scipy",
    "sklearn",
    "sqlalchemy",
    "aiohttp",
    "fastapi",
    "pydantic",
]

# Napoleon settings
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
napoleon_attr_annotations = True
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "ndarray": "numpy.ndarray",
    "DataFrame": "pandas.DataFrame",
    "Series": "pandas.Series",
    "Tensor": "torch.Tensor",
    "Model": "torch.nn.Module",
    "Dataset": "torch.utils.data.Dataset",
    "DataLoader": "torch.utils.data.DataLoader",
    "AsyncClient": "aiohttp.ClientSession",
    "FastAPI": "fastapi.FastAPI",
    "BaseModel": "pydantic.BaseModel",
    "SQLAlchemy": "sqlalchemy.orm.Session",
}

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "aiohttp": ("https://docs.aiohttp.org/en/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "sqlalchemy": ("https://docs.sqlalchemy.org/en/latest/", None),
    "pytest": ("https://docs.pytest.org/en/latest/", None),
    "fastapi": ("https://fastapi.tiangolo.com/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "jinja2": ("https://jinja.palletsprojects.com/en/latest/", None),
    "click": ("https://click.palletsprojects.com/en/latest/", None),
    "requests": ("https://requests.readthedocs.io/en/latest/", None),
    "flask": ("https://flask.palletsprojects.com/en/latest/", None),
    "django": ("https://docs.djangoproject.com/en/stable/", None),
    "celery": ("https://docs.celeryq.dev/en/stable/", None),
    "redis": ("https://redis.readthedocs.io/en/latest/", None),
    "postgresql": ("https://www.postgresql.org/docs/current/", None),
    "mysql": ("https://dev.mysql.com/doc/refman/8.0/en/", None),
    "mongodb": ("https://www.mongodb.com/docs/manual/", None),
    "elasticsearch": ("https://www.elastic.co/guide/en/elasticsearch/reference/current/", None),
    "kafka": ("https://kafka.apache.org/documentation/", None),
    "docker": ("https://docs.docker.com/engine/reference/", None),
    "kubernetes": ("https://kubernetes.io/docs/reference/", None),
    "aws": ("https://boto3.amazonaws.com/v1/documentation/api/latest/", None),
    "gcp": ("https://cloud.google.com/python/docs/reference/", None),
    "azure": ("https://docs.microsoft.com/en-us/python/api/overview/azure/", None),
}

# HTML output configuration
html_theme = "furo"
html_static_path = ["_static"]
html_logo = "_static/wdbx_logo.png" if os.path.exists("_static/wdbx_logo.png") else None
html_favicon = "_static/favicon.ico" if os.path.exists("_static/favicon.ico") else None
html_title = f"{project} Documentation"
html_short_title = project
html_baseurl = os.environ.get("SPHINX_HTML_BASE_URL", "http://127.0.0.1:8000/")
html_use_smartypants = True
html_last_updated_fmt = "%b %d, %Y"
html_copy_source = True
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True
html_use_index = True
html_split_index = False
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
    ]
}

# Theme options
html_theme_options = {
    # Navigation
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
    "display_version": True,
    "prev_next_buttons_location": "both",
    "style_external_links": True,
    "logo_only": True,
    # Announcement
    "announcement": "🎉 Welcome to WDBX Documentation! Press ? for keyboard shortcuts.",
    # UI/UX
    "top_of_page_button": "auto",
    "show_nav_level": 2,
    "navigation_with_keys": True,
    "show_toc_level": 3,
    # Footer
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/wdbx/wdbx-python",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
    # Light theme
    "light_css_variables": {
        "font-stack": "'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
        "font-stack--monospace": "'JetBrains Mono', 'SFMono-Regular', Consolas, Monaco, 'Liberation Mono', monospace",
        "color-brand-primary": "#3B82F6",
        "color-brand-content": "#2563EB",
        "color-background-hover": "#F1F5F9",
        "color-background-secondary": "#F8FAFC",
        "color-background-border": "#E2E8F0",
        "color-link": "#2563EB",
        "color-link-hover": "#1E40AF",
        "color-link-underline": "rgba(37, 99, 235, 0.2)",
        "color-link-underline--hover": "rgba(37, 99, 235, 0.5)",
        "color-announcement-background": "#FEF3C7",
        "color-announcement-text": "#92400E",
    },
    # Dark theme
    "dark_css_variables": {
        "font-stack": "'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
        "font-stack--monospace": "'JetBrains Mono', 'SFMono-Regular', Consolas, Monaco, 'Liberation Mono', monospace",
        "color-brand-primary": "#60A5FA",
        "color-brand-content": "#93C5FD",
        "color-background-hover": "#1E293B",
        "color-background-secondary": "#0F172A",
        "color-background-border": "#1E293B",
        "color-link": "#60A5FA",
        "color-link-hover": "#93C5FD",
        "color-link-underline": "rgba(96, 165, 250, 0.2)",
        "color-link-underline--hover": "rgba(96, 165, 250, 0.5)",
        "color-announcement-background": "#78350F",
        "color-announcement-text": "#FEF3C7",
    },
}

# Add custom static files
html_css_files = [
    "custom.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",
    "https://cdn.jsdelivr.net/npm/@docsearch/css@3",
]

html_js_files = [
    "modern.js",
    "custom.js",
    "https://cdn.jsdelivr.net/npm/@docsearch/js@3",
]

# GitHub repository information
html_context = {
    "github_user": "wdbx",
    "github_repo": "wdbx-python",
    "github_version": "main",
    "doc_path": "docs",
    "display_github": True,
    "conf_py_path": "/docs/",
    "source_suffix": [".rst", ".md"],
}

# LaTeX output configuration
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "11pt",
    "figure_align": "htbp",
    "preamble": r"""
        \usepackage{xcolor}
        \usepackage{fancyhdr}
        \pagestyle{fancy}
        \fancyhead[L]{\textit{WDBX Documentation}}
        \fancyhead[R]{\thepage}
        \fancyfoot[C]{}
        \usepackage{listings}
        \usepackage{tcolorbox}
        \usepackage{minted}
        \usepackage{hyperref}
        \hypersetup{colorlinks=true, linkcolor=blue, filecolor=magenta, urlcolor=cyan}
    """,
    "sphinxsetup": "verbatimwithframe=false,VerbatimColor={rgb}{0.97,0.97,0.97}",
    "extraclassoptions": "openany,oneside",
    "maketitle": r"\maketitle\newpage\tableofcontents\newpage",
}

latex_documents = [
    (
        "index",
        "wdbx.tex",
        "WDBX Documentation",
        author,
        "manual",
    ),
]

# Manual page output
man_pages = [
    (
        "index",
        "wdbx",
        "WDBX Documentation",
        [author],
        1,
    ),
]

# Texinfo output configuration
texinfo_documents = [
    (
        "index",
        "wdbx",
        "WDBX Documentation",
        author,
        "WDBX",
        "Vector database and processing system.",
        "Databases",
    ),
]

# Epub output configuration
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright
epub_show_urls = "footnote"
epub_use_index = True
epub_tocdup = False
epub_tocdepth = 3
epub_fix_images = False
epub_max_image_width = 0

# PDF configuration
pdf_documents = [
    ("index", "wdbx", "WDBX Documentation", author),
]

# Create .nojekyll file for GitHub Pages
html_extra_path = []
try:
    with open(os.path.join(html_static_path[0], ".nojekyll"), "w") as f:
        pass
    print("Created .nojekyll file for GitHub Pages")
except (OSError, IndexError) as e:
    print(f"Could not create .nojekyll file: {e}")

# Configure todo extension
todo_include_todos = True
todo_emit_warnings = True
todo_link_only = False

# Enable linkcheck builder
linkcheck_ignore = [
    r"^https?://localhost.*",
    r"^https?://127\.0\.0\.1.*",
    r"^https?://twitter\.com.*",
    r"^https?://t\.co.*",
    r".*github\.com/.*/(pull|issues)/.*",
    "http://example.com",
    r"^https?://docs\.python\.org.*",
]

# Dictionary mapping regular expressions to replacement URLs
linkcheck_allowed_redirects = {
    "https://github.com/([^/]+)/([^/]+)/blob/master/(.*)": "https://github.com/\\1/\\2/\\3",
    "https://github.com/([^/]+)/([^/]+)/tree/master/(.*)": "https://github.com/\\1/\\2/\\3",
}

linkcheck_timeout = 15
linkcheck_workers = 10
linkcheck_anchors = True
linkcheck_anchors_ignore = ["^!"]
linkcheck_retries = 3

# Markdown support
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
    ".py": "markdown",
    ".ipynb": "nbsphinx",
}

# Don't show module names in front of class names
add_module_names = False

# Enable numfig for figure numbering
numfig = True
numfig_format = {
    "figure": "Figure %s",
    "table": "Table %s",
    "code-block": "Listing %s",
    "section": "Section %s",
}
numfig_secnum_depth = 2

# MyST parser configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 4
myst_update_mathjax = False
myst_highlight_code_blocks = True
myst_all_links_external = False
myst_url_schemes = ("http", "https", "mailto", "ftp")
myst_footnote_transition = True
myst_dmath_double_inline = True
myst_enable_checkboxes = True

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = True
autosummary_ignore_module_all = False

# Configure sphinx-copybutton
copybutton_prompt_text = ">>> |\\.\\.\\. |\\$ |In \\[\\d*\\]: | {2,5}\\.\\.\\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_remove_prompts = True
copybutton_line_continuation_character = "\\"
copybutton_here_doc_delimiter = "EOT"
copybutton_image_svg = """
<svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-copy" width="44" height="44" viewBox="0 0 24 24" stroke-width="1.5" stroke="#000000" fill="none" stroke-linecap="round" stroke-linejoin="round">
  <path stroke="none" d="M0 0h24v24H0z" fill="none"/>
  <rect x="8" y="8" width="12" height="12" rx="2" />
  <path d="M16 8v-2a2 2 0 0 0 -2 -2h-8a2 2 0 0 0 -2 2v8a2 2 0 0 0 2 2h2" />
</svg>
"""
copybutton_selector = "div:not(.no-copybutton) > div.highlight > pre"

# Configure sphinx-search
search_language = "en"
search_scorer = "proximity"
search_sort = "proximity"
search_keep_score = 15
search_adapter = "readthedocs"
search_ignore_files = [
    "search.html",
    "genindex.html",
    "py-modindex.html",
    "404.html",
    "robots.txt",
]

# Auto-generate API documentation
autodoc_member_order = "groupwise"
autoclass_content = "both"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_typehints_description_target = "documented"
autodoc_inherit_docstrings = True
autodoc_preserve_defaults = True
autodoc_warningiserror = False
autodoc_mock_imports = []

# Configure sphinx-gallery for examples
sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": "/example_",
    "ignore_pattern": r"__init__\.py",
    "plot_gallery": "True",
    "thumbnail_size": (400, 300),
    "min_reported_time": 1,
    "download_all_examples": False,
    "line_numbers": True,
    "remove_config_comments": True,
    "default_thumb_file": "_static/default_thumb.png",
    "capture_repr": ("_repr_html_", "__repr__"),
    "image_scrapers": ("matplotlib", "plotly"),
    "reference_url": {
        "wdbx": None,
    },
    "show_memory": True,
    "junit": "_build/junit-results.xml",
    "inspect_global_variables": True,
    "matplotlib_animations": True,
    "image_srcset": ["2x"],
    "show_signature": True,
}

# Configure nbsphinx for Jupyter notebooks
nbsphinx_execute = "auto"
nbsphinx_allow_errors = False
nbsphinx_timeout = 60
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]
nbsphinx_kernel_name = "python3"
nbsphinx_prompt_width = "0.5em"
nbsphinx_responsive_width = "100%"
nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=None) %}

.. note::

   This page was generated from a Jupyter notebook.
   You can download the notebook `here <https://github.com/wdbx/wdbx-python/blob/main/{{ docname }}>`_.
"""
nbsphinx_epilog = r"""
----

Generated by nbsphinx from a Jupyter notebook.
"""

# Configure doctest
doctest_global_setup = """
import numpy as np
import pandas as pd
import torch
import os
import sys
import json
import yaml
import datetime
import tempfile
import pathlib
"""
doctest_test_doctest_blocks = "default"
doctest_global_cleanup = ""
doctest_default_flags = 0

# Configure mermaid
mermaid_output_format = "svg"
mermaid_params = ["--theme", "default"]
mermaid_cmd = "mmdc"
mermaid_pdfcrop = ""

# Configure PlantUML
plantuml = "java -jar /usr/share/plantuml/plantuml.jar"
plantuml_output_format = "svg"

# Configure sphinx-design
sd_fontawesome_latex = True

# Configure sitemap
sitemap_url_scheme = "{link}"
sitemap_filename = "sitemap.xml"


# Configure linkcode
def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to a Python object.
    """
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    # Handle special cases like builtins
    if modname == "builtins":
        return None

    obj = sys.modules.get(modname)
    if obj is None:
        return None

    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except (AttributeError, TypeError):
            return None

    try:
        import inspect

        fn = inspect.getsourcefile(obj)
    except (TypeError, AttributeError):
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        return None

    if fn is None:
        return None

    fn = os.path.relpath(fn, start=os.path.dirname(os.path.abspath(__file__)) + "/..")
    if fn.startswith(".."):
        return None

    return (
        f"https://github.com/wdbx/wdbx-python/blob/main/{fn}#L{lineno}-L{lineno + len(source) - 1}"
    )


def generate_api_docs():
    """
    Automatically generate API documentation for all modules in the project.
    """
    # Create API directory if it doesn't exist
    api_dir = os.path.join(os.path.dirname(__file__), "api")
    os.makedirs(api_dir, exist_ok=True)

    # Path to the project's Python modules
    modules_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Create index file for API documentation
    with open(os.path.join(api_dir, "index.rst"), "w") as f:
        f.write("API Reference\n")
        f.write("============\n\n")
        f.write(
            "This section provides detailed API documentation for all modules in the WDBX project.\n\n"
        )
        f.write(".. toctree::\n")
        f.write("   :maxdepth: 2\n\n")

    # Find all Python modules in the project
    module_files = []
    for root, dirs, files in os.walk(modules_dir):
        # Skip hidden directories, tests, and virtual environments
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".")
            and d not in ["venv", ".venv", "env", "__pycache__", "tests", "test", ".git", ".github"]
        ]

        for file in files:
            if file.endswith(".py") and not file.startswith("_"):
                module_files.append(os.path.join(root, file))

    # Sort modules by path for consistent ordering
    module_files.sort()

    # Track top-level modules for index
    top_level_modules = set()

    # Generate API documentation for each module
    for module_file in module_files:
        rel_path = os.path.relpath(module_file, modules_dir)
        module_path = os.path.splitext(rel_path)[0].replace(os.path.sep, ".")

        # Skip test files and setup files
        if (
            "test" in module_path
            or "tests" in module_path
            or "setup.py" in module_path
            or "conf.py" in module_path
        ):
            continue

        # Get top-level module name
        top_level = module_path.split(".")[0]
        top_level_modules.add(top_level)

        # Create directory structure if needed
        module_dir = os.path.dirname(os.path.join(api_dir, module_path))
        os.makedirs(module_dir, exist_ok=True)

        # Create RST file for the module
        output_file = os.path.join(api_dir, f"{module_path}.rst")

        with open(output_file, "w") as f:
            # Module title
            f.write(f"{module_path}\n")
            f.write("=" * len(module_path) + "\n\n")

            # Module documentation
            f.write(f".. automodule:: {module_path}\n")
            f.write("   :members:\n")
            f.write("   :undoc-members:\n")
            f.write("   :show-inheritance:\n")
            f.write("   :special-members: __init__\n\n")

            # Add class documentation
            f.write("Classes\n")
            f.write("-------\n\n")

            # We'll use a placeholder that will be filled if classes are found
            f.write(f".. currentmodule:: {module_path}\n\n")
            f.write(".. autosummary::\n")
            f.write("   :toctree: _autosummary\n")
            f.write("   :template: custom-class-template.rst\n\n")
            f.write("   # Classes will be auto-populated here\n\n")

            # Add function documentation
            f.write("Functions\n")
            f.write("---------\n\n")
            f.write(".. autosummary::\n")
            f.write("   :toctree: _autosummary\n")
            f.write("   :template: custom-function-template.rst\n\n")
            f.write("   # Functions will be auto-populated here\n")

    # Update the API index with top-level modules
    with open(os.path.join(api_dir, "index.rst"), "a") as f:
        for module in sorted(top_level_modules):
            f.write(f"   {module}\n")

    # Create module index files for each top-level module
    for module in top_level_modules:
        with open(os.path.join(api_dir, f"{module}.rst"), "w") as f:
            f.write(f"{module}\n")
            f.write("=" * len(module) + "\n\n")
            f.write(f".. automodule:: {module}\n\n")
            f.write(".. toctree::\n")
            f.write("   :maxdepth: 2\n\n")

            # Find all submodules
            for module_file in module_files:
                rel_path = os.path.relpath(module_file, modules_dir)
                module_path = os.path.splitext(rel_path)[0].replace(os.path.sep, ".")

                if module_path.startswith(f"{module}.") and not (
                    "test" in module_path or "tests" in module_path
                ):
                    f.write(f"   {module_path}\n")


def compile_all_docs():
    """
    Find and compile all .py, .md, and .rst files in the project into documentation.
    This creates a comprehensive documentation structure organized by file type and directory.
    """
    # Path to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Create documentation directories
    docs_dir = os.path.join(os.path.dirname(__file__), "compiled_docs")
    os.makedirs(docs_dir, exist_ok=True)

    # Create subdirectories for different file types
    py_docs_dir = os.path.join(docs_dir, "python_files")
    md_docs_dir = os.path.join(docs_dir, "markdown_files")
    rst_docs_dir = os.path.join(docs_dir, "rst_files")

    os.makedirs(py_docs_dir, exist_ok=True)
    os.makedirs(md_docs_dir, exist_ok=True)
    os.makedirs(rst_docs_dir, exist_ok=True)

    # Create main index file
    with open(os.path.join(docs_dir, "index.rst"), "w") as f:
        f.write("Project Documentation\n")
        f.write("====================\n\n")
        f.write(
            "This documentation includes all Python, Markdown, and reStructuredText files found in the project.\n\n"
        )
        f.write(".. toctree::\n")
        f.write("   :maxdepth: 1\n")
        f.write("   :caption: Documentation Types:\n\n")
        f.write("   python_files/index\n")
        f.write("   markdown_files/index\n")
        f.write("   rst_files/index\n")

    # Create index files for each file type
    for dir_name, title in [
        (py_docs_dir, "Python Files"),
        (md_docs_dir, "Markdown Files"),
        (rst_docs_dir, "reStructuredText Files"),
    ]:
        with open(os.path.join(dir_name, "index.rst"), "w") as f:
            f.write(f"{title}\n")
            f.write("=" * len(title) + "\n\n")
            f.write(f"This section contains documentation generated from {title.lower()}.\n\n")
            f.write(".. toctree::\n")
            f.write("   :maxdepth: 2\n")
            f.write("   :caption: Contents:\n\n")

    # Find all documentation files
    py_files = []
    md_files = []
    rst_files = []

    for root, dirs, files in os.walk(project_root):
        # Skip hidden directories, tests, and virtual environments
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".")
            and d
            not in [
                "venv",
                ".venv",
                "env",
                "__pycache__",
                "tests",
                "test",
                ".git",
                ".github",
                "_build",
                "build",
                "dist",
            ]
        ]

        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, project_root)

            # Skip certain files and directories
            if any(
                exclude in rel_path
                for exclude in [
                    "__pycache__",
                    ".git",
                    ".pytest_cache",
                    ".ipynb_checkpoints",
                    "setup.py",
                    "conf.py",
                    ".pyc",
                    ".pyo",
                    ".pyd",
                ]
            ):
                continue

            if file.endswith(".py"):
                py_files.append((file_path, rel_path))
            elif file.endswith(".md"):
                md_files.append((file_path, rel_path))
            elif file.endswith(".rst"):
                rst_files.append((file_path, rel_path))

    # Process Python files
    process_files(py_files, py_docs_dir, "python")

    # Process Markdown files
    process_files(md_files, md_docs_dir, "markdown")

    # Process RST files
    process_files(rst_files, rst_docs_dir, "rst")

    # Create directory structure documentation
    create_directory_structure_docs(project_root, docs_dir)


def process_files(files, output_dir, file_type):
    """
    Process a list of files and generate documentation for them.

    Args:
        files: List of (file_path, rel_path) tuples
        output_dir: Directory to write documentation to
        file_type: Type of files being processed ('python', 'markdown', or 'rst')
    """
    # Sort files by relative path
    files.sort(key=lambda x: x[1])

    # Group files by directory
    dir_files = {}
    for file_path, rel_path in files:
        dir_name = os.path.dirname(rel_path)
        if dir_name == "":
            dir_name = "root"

        if dir_name not in dir_files:
            dir_files[dir_name] = []

        dir_files[dir_name].append((file_path, rel_path))

    # Create directory index files
    for dir_name, dir_file_list in dir_files.items():
        safe_dir_name = dir_name.replace("/", "_").replace("\\", "_").replace(".", "_")
        dir_output_dir = os.path.join(output_dir, safe_dir_name)
        os.makedirs(dir_output_dir, exist_ok=True)

        # Create directory index file
        with open(os.path.join(dir_output_dir, "index.rst"), "w") as f:
            title = f"Files in {dir_name}"
            f.write(f"{title}\n")
            f.write("=" * len(title) + "\n\n")
            f.write(
                f"This section contains {file_type} files from the ``{dir_name}`` directory.\n\n"
            )
            f.write(".. toctree::\n")
            f.write("   :maxdepth: 1\n\n")

            # Process each file in the directory
            for file_path, rel_path in dir_file_list:
                file_name = os.path.basename(rel_path)
                file_base, file_ext = os.path.splitext(file_name)

                # Create a safe filename for the output
                safe_name = file_base.replace(".", "_")
                output_file = os.path.join(dir_output_dir, f"{safe_name}.rst")

                # Add to directory index
                f.write(f"   {safe_name}\n")

                # Create documentation file
                with open(output_file, "w") as file_f:
                    # File title
                    file_title = f"{file_name}"
                    file_f.write(f"{file_title}\n")
                    file_f.write("=" * len(file_title) + "\n\n")
                    file_f.write(f"Path: ``{rel_path}``\n\n")

                    # Include the file content based on its type
                    if file_ext == ".rst":
                        # For RST files, include them directly
                        file_f.write(f".. include:: ../../{rel_path}\n")
                    elif file_ext == ".md":
                        # For Markdown files, use the myst parser
                        file_f.write(f".. mdinclude:: ../../{rel_path}\n")
                    elif file_ext == ".py":
                        # For Python files, extract docstrings and include as code
                        file_f.write("Source Code:\n\n")
                        file_f.write(".. code-block:: python\n\n")

                        # Read the Python file and include it with proper indentation
                        with open(file_path, encoding="utf-8") as py_file:
                            py_content = py_file.read()
                            # Indent each line for code block
                            py_content_indented = "\n".join(
                                ["    " + line for line in py_content.split("\n")]
                            )
                            file_f.write(py_content_indented + "\n\n")

                        # Try to extract module docstring
                        try:
                            import ast

                            module = ast.parse(py_content)
                            docstring = ast.get_docstring(module)
                            if docstring:
                                file_f.write("\nModule Documentation:\n")
                                file_f.write("-" * 20 + "\n\n")
                                file_f.write(docstring + "\n\n")
                        except Exception as e:
                            file_f.write(f"\nError extracting docstring: {str(e)}\n\n")
                    else:
                        # For unknown file types, just note the file type
                        file_f.write(f"\nFile type {file_ext} not specifically handled.\n\n")


def create_directory_structure_docs(project_root, docs_dir):
    """
    Create documentation for the project directory structure.

    Args:
        project_root: Path to the project root directory
        docs_dir: Path to the documentation directory
    """
    output_dir = os.path.join(docs_dir, "generated", "structure")
    os.makedirs(output_dir, exist_ok=True)

    # Create index file
    with open(os.path.join(output_dir, "index.rst"), "w") as f:
        title = "Project Directory Structure"
        f.write(f"{title}\n")
        f.write("=" * len(title) + "\n\n")
        f.write("This section provides an overview of the project directory structure.\n\n")

        # Create a tree-like structure of the project
        f.write(".. code-block:: text\n\n")

        # Use os.walk to get the directory structure
        for root, dirs, files in os.walk(project_root):
            # Skip hidden directories and files
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".") and d != "docs" and d != "build" and d != "dist"
            ]

            rel_path = os.path.relpath(root, project_root)
            if rel_path == ".":
                # Root directory
                f.write("    Project Root\n")
                level = 1
            else:
                # Subdirectory
                level = rel_path.count(os.sep) + 1
                indent = "    " + "│   " * (level - 1) + "├── "
                f.write(f"{indent}{os.path.basename(root)}/\n")

            # Add files
            for file in sorted(files):
                if not file.startswith("."):
                    indent = "    " + "│   " * level + "├── "
                    f.write(f"{indent}{file}\n")


def main():
    """
    Main entry point for documentation compilation.
    """
    # Generate API documentation
    generate_api_docs()

    # Compile all documentation
    compile_all_docs()


if __name__ == "__main__":
    main()
