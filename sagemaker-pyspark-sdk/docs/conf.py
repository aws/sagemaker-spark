#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -- General configuration ------------------------------------------------

import os
import sys
import subprocess

from unittest.mock import MagicMock
sys.path.insert(0, os.path.abspath('../src/'))

if "READTHEDOCS" in os.environ:
    # pyspark can't be installed on the readthedocs build system
    # due to out of memory errors. And we can't mock it either due to
    # inheritance constraints.
    p = subprocess.Popen("sh get_pyspark.sh".split())
    p.communicate()

    class Mock(MagicMock):

        @classmethod
        def __getattr__(cls, name):
            return MagicMock()

    MOCK_MODULES = ['py4j', 'py4j.protocol', 'py4j.java_gateway', 'py4j.java_collections']
    sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.doctest',
              'sphinx.ext.intersphinx', 'sphinx.ext.todo',
              'sphinx.ext.coverage', 'sphinx.ext.autosummary',
              'sphinx.ext.napoleon']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'sagemaker_pyspark'
copyright = '2017, Amazon Web Services'
author = 'Amazon Web Services'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = '1.0'
# The full version, including alpha/beta/rc tags.
release = '1.0'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

add_module_names = False

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

# Read The docs calls their theme 'default' however on the external world
# it is called sphinx_rtd_theme. We should set the name accordingly.
if 'READTHEDOCS' in os.environ:
    html_theme = 'default'
else:
    import sphinx_rtd_theme
    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_theme_options = {
    'collapse_navigation': False,
    'display_version': False,
    'navigation_depth': 3,
}

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# This is required for the alabaster theme
# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
html_sidebars = {
    '**': [
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
    ]
}


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'SageMakerPySparkSDKdoc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'sagemaker_pyspark.tex', 'sagemaker_pyspark Documentation',
     'Amazon Web Services', 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'sagemakerpysparksdk', 'sagemaker_pyspark Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'sagemaker_pyspark', 'sagemaker_pyspark Documentation',
     author, 'sagemaker_pyspark', 'One line description of project.',
     'Miscellaneous'),
]
