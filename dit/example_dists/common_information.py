r"""
Witnesses for the common-information ordering.

The common informations of a set of random variables are totally ordered
:cite:`james2017multivariate`,

.. math::

   0 \le \operatorname{K} \le \operatorname{J} \le \operatorname{B}
     \le \operatorname{C} \le \operatorname{G} \le \operatorname{F}
     \le \operatorname{M} \le \operatorname{H},

running from the Gacs-Korner common information :math:`\operatorname{K}`, through
the CAEKL mutual information :math:`\operatorname{J}`, the dual total correlation
:math:`\operatorname{B}`, the Wyner :cite:`wyner1975common` common information
:math:`\operatorname{C}`, the exact :cite:`kumar2014exact` common information
:math:`\operatorname{G}`, and the functional :math:`\operatorname{F}` and minimal
sufficient statistic :math:`\operatorname{M}` common informations, up to the joint
entropy :math:`\operatorname{H}`. A *witness* is a single distribution on which
every one of these inequalities is strict, certifying that no two of the measures
coincide as functionals.

:func:`symmetric_common_information_witness` builds an exchangeable witness for any
number of variables as the independent product of three layers, so that each
variable is

.. math::

   X_i = (G,\; Z \oplus N_i,\; P_i):

a shared "giant bit" :math:`G` (see :func:`~dit.example_dists.giant_bit`, which
lifts :math:`0 < \operatorname{K}`), a latent fair bit :math:`Z` observed through
an independent binary symmetric channel with :math:`N_i \sim
\operatorname{Bernoulli}(q)` (the workhorse :func:`noisy_copies`, which fans out
the middle of the chain), and a private fair bit :math:`P_i` (which opens
:math:`\operatorname{M} < \operatorname{H}`). Each variable has a common alphabet
of eight symbols, so the witness is exchangeable with :math:`2^{2n+1}` outcomes.
"""

from itertools import permutations, product

from ..distribution import Distribution

__all__ = (
    "erasure_channel",
    "noisy_copies",
    "reverse_erasure_channel",
    "symmetric_common_information_witness",
    "task_assignment",
)

# BSC crossover q, per n, each verified to strictly separate the whole chain. n=2
# reproduces Wyner's doubly symmetric binary source at crossover 0.30, since
# 2 q (1 - q) = 0.30 gives q = (1 - sqrt(0.4)) / 2. At fixed q the exact-minus-Wyner
# gap shrinks as n grows (more copies pin down the latent Z), but these values keep
# every rung open through n = 5; larger n may require larger q.
_TUNED_Q = {2: (1 - 0.4**0.5) / 2, 3: 0.10, 4: 0.15, 5: 0.15}


def noisy_copies(q, n=3):
    r"""
    ``n`` noisy copies of a latent fair bit: the ``n``-variable doubly symmetric
    binary source.

    A hidden fair bit :math:`Z \sim \operatorname{U}\{0, 1\}` is observed through
    ``n`` independent binary symmetric channels, :math:`X_i = Z \oplus N_i` with
    :math:`N_i \sim \operatorname{Bernoulli}(q)`, and :math:`Z` is marginalized
    out. The result is exchangeable and depends only on the Hamming weight ``k``
    of an outcome,

    .. math::

       p_k = \tfrac{1}{2}\left( q^k (1 - q)^{n - k} + q^{n - k} (1 - q)^k \right).

    It has no shared deterministic structure, so its Gacs-Korner common
    information is zero, and no private randomness, so its minimal sufficient
    statistic common information equals its entropy; between those extremes it
    strictly separates the CAEKL, dual total correlation, Wyner
    :cite:`wyner1975common`, exact :cite:`kumar2014exact`, and functional common
    informations. For ``n = 2`` it is Wyner's doubly symmetric binary source, with
    crossover :math:`2 q (1 - q)`.

    Parameters
    ----------
    q : float
        The binary symmetric channel crossover :math:`P(X_i \ne Z)`, in
        :math:`[0, 1/2]`.
    n : int
        The number of noisy copies. Defaults to 3.

    Returns
    -------
    d : Distribution
        The ``n``-variable doubly symmetric binary source, over ``n`` binary
        variables with :math:`2^n` outcomes.
    """
    pmf = {}
    for z in (0, 1):
        for bits in product((0, 1), repeat=n):
            weight = 0.5
            for b in bits:
                weight *= q if b != z else (1 - q)
            outcome = tuple(str(b) for b in bits)
            pmf[outcome] = pmf.get(outcome, 0.0) + weight
    return Distribution(list(pmf), list(pmf.values()))


def symmetric_common_information_witness(n=3, q=None):
    r"""
    An exchangeable witness separating the entire common-information ordering.

    Returns the independent product of a shared giant bit, ``n`` noisy copies of a
    latent fair bit (:func:`noisy_copies`), and ``n`` private fair bits, so that
    each variable is :math:`X_i = (G, Z \oplus N_i, P_i)` over a common eight-symbol
    alphabet. Every rung of the ordering

    .. math::

       0 < \operatorname{K} < \operatorname{J} < \operatorname{B}
         < \operatorname{C} < \operatorname{G} < \operatorname{F}
         < \operatorname{M} < \operatorname{H}

    is strict :cite:`james2017multivariate`. For ``n = 2`` the CAEKL mutual
    information and dual total correlation coincide with the mutual information, so
    the chain has one fewer distinct rung.

    Parameters
    ----------
    n : int
        The number of variables; at least 2. Defaults to 3.
    q : float, optional
        The binary symmetric channel crossover for the noisy workhorse. When
        ``None`` (default) a value tuned per ``n`` (for ``n`` up to 5) is used,
        falling back to ``0.15`` otherwise.

    Returns
    -------
    d : Distribution
        The exchangeable witness, over ``n`` variables with an eight-symbol
        alphabet and :math:`2^{2n+1}` outcomes.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    if q is None:
        q = _TUNED_Q.get(n, 0.15)

    pmf = {}
    for giant in (0, 1):
        for latent in (0, 1):
            for noise in product((0, 1), repeat=n):
                weight = 0.25  # p(giant) p(latent)
                for e in noise:
                    weight *= q if e else (1 - q)
                for private in product((0, 1), repeat=n):
                    outcome = tuple(str(4 * giant + 2 * (latent ^ e) + p) for e, p in zip(noise, private, strict=True))
                    pmf[outcome] = pmf.get(outcome, 0.0) + weight * 0.5**n
    return Distribution(list(pmf), list(pmf.values()))


def erasure_channel(p):
    r"""
    A fair bit passed through a binary erasure channel.

    The input :math:`X` is a fair bit and the output :math:`Y` equals :math:`X`
    with probability :math:`1 - p`, and is the erasure symbol ``e`` otherwise. Its
    Wyner common information is :cite:`cuff2013distributed`

    .. math::

       \operatorname{C}[X : Y] = \begin{cases} 1 & p < 1/2, \\ h(p) & p \ge 1/2, \end{cases}

    where :math:`h` is the binary entropy, while :math:`\I{X : Y} = 1 - p` and the
    necessary conditional entropy is :math:`\H{Y \dagger X} = h(p)`.

    Parameters
    ----------
    p : float
        The erasure probability, in :math:`[0, 1]`.

    Returns
    -------
    d : Distribution
        The joint distribution of input and output.
    """
    return Distribution(["00", "0e", "1e", "11"], [(1 - p) / 2, p / 2, p / 2, (1 - p) / 2])


def reverse_erasure_channel(p):
    r"""
    The binary erasure channel run backwards.

    The input :math:`X` is the erasure symbol ``e`` with probability :math:`p`
    and otherwise a fair bit; the output :math:`Y` copies a bit and replaces ``e``
    with a fair coin flip :cite:`cuff2013distributed`. It is
    :func:`erasure_channel` with input and output exchanged.

    Parameters
    ----------
    p : float
        The erasure probability, in :math:`[0, 1]`.

    Returns
    -------
    d : Distribution
        The joint distribution of input and output.
    """
    return Distribution(["00", "e0", "e1", "11"], [(1 - p) / 2, p / 2, p / 2, (1 - p) / 2])


def task_assignment(k, n=2):
    r"""
    ``n`` distinct tasks dealt uniformly at random from ``k``.

    The distribution is uniform over ``n``-tuples of distinct values from
    ``range(k)`` :cite:`cuff2010coordination`. For ``n = 2`` this is the
    "scatter" channel of :cite:`cuff2013distributed`, uniform over :math:`x \ne y`,
    with

    .. math::

       \I{X : Y} = \log_2 \frac{k}{k - 1}, \qquad
       \H{Y \dagger X} = \log_2 (k - 1), \qquad
       \operatorname{C}[X : Y] = 2 - \log_2 \frac{\lceil k \rceil_2}{\lceil k \rceil_2 - 1},

    where :math:`\lceil k \rceil_2` is ``k`` rounded up to an even number. For
    ``n = 3`` and ``k`` divisible by three, the Wyner common information is

    .. math::

       \operatorname{C}[X : Y : Z] = 3 \log_2 3 - \log_2 \frac{k}{k - 1} - \log_2 \frac{k}{k - 2}.

    Parameters
    ----------
    k : int
        The number of tasks.
    n : int
        The number of variables, each assigned a different task. Defaults to 2.

    Returns
    -------
    d : Distribution
        The uniform distribution over the :math:`k! / (k - n)!` assignments.
    """
    if not 1 <= n <= k:
        raise ValueError("n must be between 1 and k")
    outcomes = [tuple(str(t) for t in o) for o in permutations(range(k), n)]
    return Distribution(outcomes, [1 / len(outcomes)] * len(outcomes))
