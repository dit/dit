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

   shannon
   multivariate/multivariate

This next group of measures can not be represented on information diagrams, and
can not really be directly compared to the measures above:

.. toctree::
   :maxdepth: 2

   other/other

There are also measures of "distance" or divergence between two (and im some
cases, more) distribution:

.. toctree::
   :maxdepth: 2

   divergences/divergences
