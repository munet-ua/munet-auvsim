# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Path setup --------------------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
import munetauvsim

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'muNet-AUVsim'
author = 'JP Crawford'
release = '0.1.0-beta'

project_urls = {
    'GitHub Repository': 'https://github.com/munet-ua/munet-auvsim',
    'Issue Tracker': 'https://github.com/munet-ua/munet-auvsim/issues',
    'Documentation': 'https://munet-ua.github.io/munet-auvsim/',
}


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',           # Auto-extract docstrings
    'sphinx.ext.napoleon',          # Support NumPy/Google docstrings
    'sphinx.ext.viewcode',          # Add source code links
    'sphinx.ext.mathjax',           # Render LaTeX math
    'sphinx.ext.intersphinx',       # Link to other projects' docs
    'sphinx.ext.autosummary',       # Generate summary tables
    'sphinx_copybutton',            # Adds copy button to code blocks
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_show_copyright = False

# RTD theme options
html_theme_options = {
    'navigation_depth': 4,          # Show 4 levels in sidebar
    'collapse_navigation': False,   # Keep navigation expanded
    'sticky_navigation': True,      # Sidebar scrolls with page
    'includehidden': True,          # Show hidden toctree items
    'titles_only': False,           # Show all headers, not just titles
}


# -- Extension configuration -------------------------------------------------

# Napoleon settings (NumPy and Google docstring support)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_custom_sections = [
    'Attributes',
    'Methods',
    'Alternative Constructors',
    'Properties',
]

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-member': True,
    'show-inheritance': True,
    'member-order': 'bysource',
    'ignore-module-all': True,
}
autodoc_mock_imports = [
    'numpy', 
    'matplotlib', 
    'scipy', 
    'pickle', 
    'struct',
]

# Type hint configuration for sphinx_autodoc_typehints
autodoc_typehints = 'description'

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = True

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True
copybutton_remove_prompts = True

# Intersphinx mapping (links to external docs)
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}
