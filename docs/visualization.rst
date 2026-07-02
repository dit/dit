.. visualization.rst
.. py:module:: dit.visualization

*************
Visualization
*************

Beyond the tabular and profile-based summaries of a distribution, ``dit`` can
render the structure of a multivariate distribution graphically.

.. py:module:: dit.visualization.upset

Information UpSet Plots
=======================

Venn (or Euler) diagrams are the traditional way to depict the information
diagram of a joint distribution, but they become unreadable beyond three or four
variables. UpSet plots :cite:`Lex2014` are a scalable alternative for
visualizing intersections among many sets. The :class:`InformationUpsetPlot`
casts the atoms of a distribution's Shannon information diagram (its
:class:`~dit.profiles.information_partitions.ShannonPartition`) as an UpSet plot,
so an arbitrary number of variables can be visualized at once.

Each column of the plot is an atom of the information diagram, e.g.
:math:`\I[X_0 : X_1 \mid X_2]`. The dot matrix encodes membership: a filled dot
means the variable participates in the atom, while an empty dot means it is
conditioned upon. The bar chart above each column gives the atom's value; unlike
ordinary set cardinalities, information atoms may be **negative** (for example,
the co-information of ``xor`` is :math:`-1` bit), so bars are colored by sign.
The bars beside the matrix give each variable's marginal entropy
:math:`\H[X_i]`.

We reuse the four examples from :cite:`Allen2014`:

.. ipython::

   In [1]: import dit

   In [2]: from dit.visualization import InformationUpsetPlot

   In [3]: ex1 = dit.Distribution(['000', '001', '010', '011', '100', '101', '110', '111'], [1/8]*8)

   In [4]: ex2 = dit.Distribution(['000', '111'], [1/2]*2)

   In [5]: ex3 = dit.Distribution(['000', '001', '110', '111'], [1/4]*4)

   In [6]: ex4 = dit.Distribution(['000', '011', '101', '110'], [1/4]*4)

The independent distribution ``ex1`` has only three nonzero atoms, one per
variable:

.. ipython::

   @savefig upset_ex1.png width=8in
   In [7]: InformationUpsetPlot(ex1).draw();

The giant bit ``ex2`` concentrates all of its information in the single
three-way atom :math:`\I[X_0 : X_1 : X_2]`:

.. ipython::

   @savefig upset_ex2.png width=8in
   In [8]: InformationUpsetPlot(ex2).draw();

And ``ex4`` (the ``xor`` distribution) exhibits a negative co-information atom:

.. ipython::

   @savefig upset_ex4.png width=8in
   In [9]: InformationUpsetPlot(ex4).draw();

The plot can be tuned: ``sort_by`` (``"value"``, ``"magnitude"``, or
``"degree"``) controls the column order, ``min_degree`` hides low-order atoms,
and ``show_values`` toggles the bar annotations. Passing ``partition`` lets you
visualize a different information diagram, such as the strictly-positive
X-diagram (:class:`~dit.profiles.information_partitions.ExtropyPartition`).

.. autoclass:: InformationUpsetPlot
   :members:
