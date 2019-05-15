.. copy_mutual_information.rst
.. py:module:: dit.divergences.copy_mutual_information

***********************
Copy Mutual Information
***********************

The copy mutual information :cite:`kolchinsky2019decomposing` is a measure capturing the portion of the :ref:`mutual_information` between :math:`X` and :math:`Y` which is due to :math:`X=Y`:

.. math::

   \op{I^{copy}}[X \to Y] = \sum_{x \in \mathcal{X}} p(X = x) \begin{cases} d_{KL}\left(p(Y=x|X=x)||p(Y=x)\right) & \textrm{if} p(Y=x|X=x) > p(Y=x) \\
                                                                            0                                     & \textrm{otherwise}
                                                              \end{cases}

Consider the binary symmetric channel. With probabilities :math:`\leq \frac{1}{2}`, the input (:math:`X`) is largely copied to the output (:math:`Y`); while when the probabilities :math:`\geq \frac{1}{2}`, the output is largely opposite the input. We therefore expect the mutual information to be "copy-like" for :math:`0 \leq p \leq \frac{1}{2}`, while the mutual information should be not "copy-like" for :math:`\frac{1}{2} \leq p \leq 1`:

.. ipython::

   In [1]: from dit.divergences import copy_mutual_information as Icopy

   In [2]: from dit.shannon import mutual_information as I

   In [3]: bsc = lambda p: dit.Distribution(['00', '01', '10', '11'], [(1-p)/2, p/2, p/2, (1-p)/2])

   In [4]: ps = np.linspace(0, 1, 101)

   In [5]: ds = [bsc(p) for p in ps]

   In [6]: mis = [I(d, [0], [1]) for d in ds]

   In [7]: cmis = [Icopy(d, [0], [1]) for d in ds]

   In [8]: plt.plot(ps, cmis, ls='-', lw=2, label='$I_{copy}$');

   In [9]: plt.plot(ps, [mi - cmi for mi, cmi in zip(mis, cmis), ls='-', lw=2, label='$I_{tran}$');

   In [10]: plt.xlabel(r'Probability of error $p$');

   In [11]: plt.ylabel(r'Information');

   In [12]: plt.legend(loc='best');

   @savefig copy_mutual_information.png width=1000 align=center
   In [13]: plt.show()

API
---

.. autofunction:: copy_mutual_information
