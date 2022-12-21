.. edgaro documentation master file, created by
   sphinx-quickstart on Mon Nov 28 20:17:27 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to edgaro's documentation!
==================================

.. toctree::
   :hidden:
   :maxdepth: 2

   self

.. include:: ../README.md
   :parser: myst_parser.sphinx_

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User manual:

   source/manual

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Class documentation:

   source/edgaro.data
   source/edgaro.balancing
   source/edgaro.model
   source/edgaro.explain

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Package changelog:

   source/changelog.rst
