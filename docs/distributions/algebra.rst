.. algebra.rst

.. py:currentmodule:: dit

Algebra
=======

Named-variable distributions multiply and divide according to the chain rule.
The free/given metadata on each operand determines the result.

Throughout, start from named ``xor``:

.. ipython::

   In [1]: from dit import Distribution

   In [2]: d = Distribution(['000', '011', '101', '110'], [1/4]*4)

   In [3]: d.set_rv_names('XYZ')

Marginal and condition
----------------------

:meth:`~Distribution.marginal` keeps the listed free variables;
:meth:`~Distribution.marginalize` drops them. Both accept positional names
or a single list:

.. ipython::

   @doctest
   In [4]: d.marginal('X', 'Y')
   Out[4]: <Distribution p(X,Y)>

   @doctest
   In [5]: d.marginalize('X', 'Y')
   Out[5]: <Distribution p(Z)>

Native :meth:`~Distribution.condition_on` takes positional variable names and
returns a **single** conditional distribution:

.. ipython::

   In [6]: p_z_xy = d.condition_on('X', 'Y')

   @doctest
   In [7]: p_z_xy
   Out[7]: <Distribution p(Z|X,Y)>

   @doctest
   In [8]: p_z_xy.is_conditional()
   Out[8]: True

Multiplication (chain rule)
---------------------------

Multiplying a marginal by a compatible conditional reconstructs the joint:

.. math::

   p(X,Y) \cdot p(Z \mid X,Y) = p(X,Y,Z)

.. ipython::

   In [9]: p_xy = d.marginal('X', 'Y')

   In [10]: rebuilt = p_xy * p_z_xy

   @doctest
   In [11]: rebuilt
   Out[11]: <Distribution p(X,Y,Z)>

   @doctest
   In [12]: rebuilt.is_approx_equal(d)
   Out[12]: True

Partial application leaves unmatched given variables given:

.. math::

   p(X) \cdot p(Z \mid X,Y) = p(X,Z \mid Y)

.. ipython::

   @doctest
   In [13]: d.marginal('X') * p_z_xy
   Out[13]: <Distribution p(X,Z|Y)>

The free variables of the two operands must be disjoint.
:meth:`Distribution.from_factors` is the named constructor for the same
product.

Division (conditioning)
-----------------------

Dividing a joint by a marginal conditions on the marginal's free variables:

.. math::

   p(X,Y) / p(X) = p(Y \mid X)

.. ipython::

   @doctest
   In [14]: p_xy / d.marginal('X')
   Out[14]: <Distribution p(Y|X)>

Numeric-outcome ``*`` is different
----------------------------------

On unnamed one-dimensional numeric distributions, ``+``, ``*``, ``%``, and
friends transform **outcomes** (two dice, scaling a die, …). See
:doc:`npscalardist`. Named-variable ``*`` is always the chain rule.

Compatibility ``condition_on``
------------------------------

Passing a list, or the ``crvs`` / ``rvs`` keywords, uses the older
dit-compatible return format ``(marginal, list_of_slices)`` — one
distribution per outcome of the conditioning variables. Prefer the native
form above; the tuple form is still used by some algorithms.

.. ipython::

   In [15]: marg, cdists = d.condition_on(['X', 'Y'], rvs=['Z'])

   @doctest
   In [16]: marg
   Out[16]: <Distribution p(X,Y)>

   @doctest
   In [17]: len(cdists)
   Out[17]: 4

API
===

.. automethod:: Distribution.marginal

.. automethod:: Distribution.marginalize

.. automethod:: Distribution.condition_on

.. automethod:: Distribution.coalesce
