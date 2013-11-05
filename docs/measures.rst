.. measures.rst

********************
Information Measures
********************

``dit`` supports many information measures, ranging from as standard as the
Shannon entropy to as exotic as Gács-Körner common information (with even more
esoteric measure coming soon!). We organize these quantities into the following
groups.

We first have the Shannon-like measures. These quantities are based on sums and
differences of entropies, conditional entropies, or mutual informations of
random variables:

.. toctree::
   :maxdepth: 2

   measures/shannon
   measures/entropy
   measures/coinformation
   measures/total_correlation
   measures/binding_information
   measures/residual_entropy

The next group of measures are Shannon-esque measures. These are measure that,
while not quite based directly on the canonical Shannon measures like above,
they are directly comparable and can be expressed on information-diagrams:

.. toctree::
   :maxdepth: 2

   measures/interaction_information
   measures/gk_common_information

This next group of measures can not be represented on information diagrams, and
can not really be directly compared to the measures above:

.. toctree::
   :maxdepth: 2

   measures/perplexity

There are also measures of "distance" or divergence:

.. toctree::
   :maxdepth: 2

   measures/jensen_shannon_divergence
