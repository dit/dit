.. api/index.rst

.. _api reference:

*************
API Reference
*************

This reference is generated from the public namespace of each :mod:`dit`
subpackage, so it lists every function and class you can import from it. The
narrative pages elsewhere in the documentation explain what the measures mean
and show worked examples; this section is the place to look up exact
signatures.

Subpackages re-export the contents of their nested modules. For example,
:mod:`dit.multivariate` includes the common informations and secret key
agreement bounds, :mod:`dit.pid` includes every ``PID_*`` measure, and
:mod:`dit.rate_distortion` includes the Gray–Wyner tools.

Core
====

.. autosummary::
   :toctree: generated/

   dit
   dit.example_dists
   dit.symbolic
   dit.math
   dit.utils

Measures
========

.. autosummary::
   :toctree: generated/

   dit.shannon
   dit.multivariate
   dit.divergences
   dit.other

Decompositions and Profiles
===========================

.. autosummary::
   :toctree: generated/

   dit.pid
   dit.profiles
   dit.visualization

Channels, Coding, and Rate-Distortion
=====================================

.. autosummary::
   :toctree: generated/

   dit.example_channels
   dit.channelorder
   dit.coding
   dit.rate_distortion

Estimation and Optimization
===========================

.. autosummary::
   :toctree: generated/

   dit.inference
   dit.algorithms
