.. wyner_common_information.rst
.. py:module:: dit.multivariate.wyner_common_information

************************
Wyner Common Information
************************

The Wyner common information :cite:`wyner1975common,liu2010common` measures the minimum amount of information necessary needed to reconstruct a joint distribution from each marginal.

.. math::

   \C[X_{0:n}|Y_{0:m}] = \min_{\ind X_{0:n} \mid Y_{0:m}, V} \I[X_{0:n} : V | Y_{0:m}]

Binary Symmetric Erasure Channel
================================

Ther Wyner common information of the binary symmetric erausre channel is known to be:

.. math::

   \C[X:Y] =
   \begin{cases}
      1     &p < \frac{1}{2} \\
      \H(p) &p \ge \frac{1}{2}
   \end{cases}.

We can verify this:

.. ipython::

   In [1]: from dit.multivariate import wyner_common_information as C

   In [2]: ps = np.linspace(1e-6, 1-1e-6, 11)

   In [3]: sbec = lambda p: dit.Distribution(['00', '0e', '1e', '11'], [(1-p)/2, p/2, p/2, (1-p)/2])

   In [4]: wci_true = [1 if p < 1/2 else dit.shannon.entropy(p) for p in ps]

   In [5]: wci_opt = [C(sbec(p)) for p in ps]

   In [6]: plt.plot(ps, wci_true, ls='-', alpha=0.5, c='b');

   In [7]: plt.plot(ps, wci_opt, ls='--', lw=2, c='b');

   In [8]: plt.xlabel(r'Probability of erasure $p$');

   In [9]: plt.ylabel(r'Wyner common information $C[X:Y]$');

   @savefig sbec.png width=500 align=center
   In [10]: plt.show()

API
===

.. autofunction:: wyner_common_information
