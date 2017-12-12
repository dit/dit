.. secret_keys.rst
.. py:module:: dit.multivariate.secret_key_agreement

********************
Secret Key Agreement
********************

One of the only methods of encrypting a message from Alice to Bomb such that no third party (Eve) can possibly decrypt it is a one-time pad. This technique requires that Alice and Bob have a secret sequence of bits, :math:`S`, which Alice then encrypts by computing the exclusive-or of it with the plaintext, :math:`P`, to produce the cyphertext, :math:`C`: :math:`C = S \oplus P`. Bob can then decrypt by xoring again: :math:`P = S \oplus C`.

In order to pull this off, Alice and Bob need to construct :math:`S` out of some sort joint randomness, :math:`p(x, y, z)`, and public communication, :math:`V`, which is assumed to have perfect fidelity. The maximum rate at which :math:`S` can be constructed in the *secret key agreement rate*.

Background
----------

Given :math:`N` IID copies of a joint distribution governed by :math:`p(x, y, z)`, let :math:`X^N` denote the random variables observed by Alice, :math:`Y^N` denote the random variables observed by Bob, and :math:`Z^N` denote the random variables observed by Even. Furthermore, let :math:`S[X : Y || Z]` be the maximum rate :math:`R` such that, for :math:`N > 0`, :math:`\epsilon > 0`, some public communication :math:`V`, and functions :math:`f` and :math:`g`:

.. math::

    S_X = f(X^N, V) \\
    S_Y = g(Y^N, V) \\
    \Prob{S_X \neq S_Y \neq S} \leq \epsilon \\
    \frac{1}{N} \I{S : V Z^N} \leq \epsilon \\
    \frac{1}{N} \H{S} \geq R - \epsilon

Intuitively, this means there exists some procedure such that, for every :math:`N` observations, Alice and Bob can publicly converse and then construct :math:`S` bits which agree almost surely, and are almost surely independent of everything Eve has access to. :math:`S` is then known as a *secret key*.

Lower Bounds
------------

Two lower bounds are known for :math:`S[X : Y || Z]`, one tighter than the other. The first is known in `dit` as the :py:func:`lower_intrinsic_mutual_information`, and is given by:

.. math::

    \I{X : Y \uparrow Z} = \max\{ \I{X : Y} - \I{X : Z}, \I{X : Y} - \I{Y : Z}, 0 \}

A tighter bound is given by the :py:func:`necessary_intrinsic_mutual_information`:

.. math::

    \I{X : Y \uparrow \uparrow Z} = \max\{ & \max_{V - U - X - YZ} \I{U : Y | V} - \I{U : Z | V} \\
                                           & \max_{V - U - Y - XZ} \I{U : X | V} - \I{U : Z | V} \}


Upper Bounds
------------

The secret key agreement rate is trivially upper bounded by:

.. math::

    \min\{ \I{X : Y}, \I{X : Y | Z} \}

A less trivial bound is given by the :py:func:`intrinsic_total_correlation`:

.. math::

    \I{X : Y \downarrow Z} = \min_{p(\overline{z} | z)} \I{X : Y | \overline{Z}}

This bound can be improved, producing the :py:func:`reduced_intrinsic_total_correlation`:

.. math::

    \I{X : Y \downarrow\downarrow Z} = \min_{U} \I{X : Y \downarrow ZU} + \H{U}

which can be further reduced into the :py:func:`minimal_intrinsic_total_correlation`:

.. math::

    \I{X : Y \downarrow\downarrow\downarrow Z} = \min_{U} \I{X : Y | U} + \I{XY : U | Z}


Taken together, we see the following structure:

.. math::

    \min\{ \I{X : Y}, \I{X : Y | Z} \} \\
    \quad \geq \I{X : Y \downarrow Z} \\
    \quad\quad \geq \I{X : Y \downarrow \downarrow Z} \\
    \quad\quad\quad \geq \I{X : Y \downarrow \downarrow \downarrow Z} \\
    \quad\quad\quad\quad \geq S[X : Y || Z] \\
    \quad\quad\quad\quad\quad \geq \I{X : Y \uparrow \uparrow Z} \\
    \quad\quad\quad\quad\quad\quad \geq \I{X : Y \uparrow Z} \\
    \quad\quad\quad\quad\quad\quad\quad \geq 0.0
