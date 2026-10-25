.. lautum_information.rst
.. py:currentmodule:: dit.other

******************
Lautum Information
******************

The lautum information :cite:`palomar2008lautum` is, in a sense, the mutual information in reverse (*lautum* is *mutual* backwards):

.. math::

   \L{X_{0:n}} = \DKL{X_0 \cdot X_1 \cdot \ldots \cdot X_n || X_{0:n}}


Unlike the mutual information, the lautum information is infinite whenever the
product of the marginals puts mass on an outcome the joint distribution
forbids. For two correlated bits with full support:

.. ipython::

   In [1]: from dit.other import lautum_information

   In [2]: d = dit.Distribution(['00', '01', '10', '11'], [0.4, 0.1, 0.1, 0.4])

   @doctest float
   In [3]: lautum_information(d)
   Out[3]: 0.3219280948873622

   @doctest float
   In [4]: dit.shannon.mutual_information(d, [0], [1])
   Out[4]: 0.2780719051126379

API
===

.. autofunction:: lautum_information
   :no-index:
