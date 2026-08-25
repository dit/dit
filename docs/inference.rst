.. inference.rst
.. py:module:: dit.inference

*********
Inference
*********

:mod:`dit.inference` estimates distributions and information quantities from
samples. Plug-in counts live alongside Miller–Madow-style and NSB-style
entropy estimators, plus k-nearest-neighbor / Kraskov–Stögbauer–Grassberger
estimators for differential entropy and total correlation.

The kNN / KSG implementations optionally use ``scikit-learn`` (installed
with the ``dit[optional]`` extra) when it is available.

From samples
============

:func:`distribution_from_data` builds a joint over words of length ``L``
from a sequence of symbols. :func:`dist_from_timeseries` treats each
column of a multivariate series as a variable and appends a
``history_length`` past together with the present.

.. ipython::

   In [1]: from dit.inference import distribution_from_data, entropy_0, entropy_1

   In [2]: data = [0, 0, 0, 1, 1, 1]

   In [3]: d = distribution_from_data(data, L=1, base='linear')

   @doctest
   In [4]: d.outcomes
   Out[4]: (0, 1)

   @doctest float
   In [5]: entropy_0(data)
   Out[5]: 1.0

Estimators
==========

* :func:`entropy_0` — plug-in entropy of length-``length`` blocks
* :func:`entropy_1` — Miller–Madow-style digamma correction
* :func:`entropy_2` — higher-order (NSB-style) bias correction
* :func:`differential_entropy_knn` — Kozachenko–Leonenko kNN
* :func:`total_correlation_ksg` — Kraskov–Stögbauer–Grassberger

API
===

.. autofunction:: distribution_from_data

.. autofunction:: dist_from_timeseries

.. autofunction:: entropy_0

.. autofunction:: entropy_1

.. autofunction:: entropy_2

.. autofunction:: differential_entropy_knn

.. autofunction:: total_correlation_ksg
