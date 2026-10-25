.. genetic_code.rst
.. py:currentmodule:: dit.example_dists

The Genetic Code
================

:func:`genetic_code` returns the standard nuclear genetic code (NCBI translation
table 1) as a joint :class:`~dit.Distribution` over a codon and the amino acid it
codes for, with the 64 codons drawn uniformly.

The map is deterministic but many-to-one, which makes it a compact example of a
lossy channel built from real biology rather than construction:
:math:`H[\mathrm{Codon}] = 6` bits exactly, while
:math:`H[\mathrm{AminoAcid}] \approx 4.2181` bits, so translation erases
:math:`H[\mathrm{Codon} \mid \mathrm{AminoAcid}] \approx 1.7819` bits -- the
average size of a synonymous codon family. Because the map is deterministic, the
mutual information equals the amino-acid entropy. See :cite:`Crick1968` and
:cite:`Yockey2005` for the biological and information-theoretic background.

.. ipython::

   In [1]: from dit.example_dists import genetic_code

   In [2]: d = genetic_code()

   In [3]: dit.shannon.entropy(d.marginal('Codon'))

   In [4]: dit.shannon.entropy(d.marginal('AminoAcid'))

   In [5]: dit.shannon.conditional_entropy(d, ['Codon'], ['AminoAcid'])

Passing ``rna=True`` swaps thymine for uracil. Passing ``bases=True`` expands the
codon into three nucleotide random variables ``Base1``, ``Base2``, ``Base3``,
which exposes the "wobble" asymmetry: the third codon position carries far less
information about the amino acid than the first two, so third-position mutations
are frequently silent.

.. ipython::

   In [6]: from dit.multivariate import coinformation as I

   In [7]: b = genetic_code(bases=True)

   In [8]: [I(b, [[f'Base{i}'], ['AminoAcid']]) for i in (1, 2, 3)]

APIs
====

.. autofunction:: genetic_code
   :no-index:
