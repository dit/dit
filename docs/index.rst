.. dit documentation master file, created by
   sphinx-quickstart on Thu Oct 31 02:07:43 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. py:module:: dit

***************************************
:mod:`dit`: discrete information theory
***************************************

:mod:`dit` is a Python package for discrete information theory.

Introduction
------------

Information theory is a powerful extension to probability and statistics, quantifying dependencies
among arbitrary random variables in a way tha tis consistent and comparable across systems and
scales. Information theory was originally developed to quantify how quickly and reliably information
could be transmitted across an arbitrary channel. The demands of modern, data-driven science have
been coopting and extending these quantities and methods into unknown, multivariate settings where
the interpretation and best practices are not known. For example, there are at least four reasonable
multivariate generalizations of the mutual information, none of which inherit all the
interpretations of the standard bivariate case. Which is best to use is context-dependent. ``dit``
implements a vast range of multivariate information measures in an effort to allow information
practitioners to study how these various measures behave and interact in a variety of contexts. We
hope that having all these measures and techniques implemented in one place will allow the
development of robust techniques for the automated quantification of dependencies within a system
and concrete interpretation of what those dependencies mean.

For a quick tour, see the :ref:`Quickstart <quickstart>`. Otherwise, work
your way through the various sections. Note that all code snippets in this
documentation assume that the following lines of code have already been run:

.. ipython:: python

   In [1]: from __future__ import division # true division for Python 2.7

   In [2]: import dit

   In [3]: import numpy as np

Contents:

.. toctree::
   :maxdepth: 2

   generalinfo
   notation
   distributions/distributions
   operations
   hypothesis
   optimization
   measures/measures
   profiles
   rate_distortion
   measures/pid
   stumbling
   zreferences

.. todolist::


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
