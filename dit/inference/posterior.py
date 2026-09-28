"""
Bayesian posterior over the entropy rate of a finite-order Markov chain.
"""

from dataclasses import dataclass

import numpy as np

from ._symbols import as_generator, standardize_trials, word_codes

__all__ = (
    "EntropyRatePosterior",
    "entropy_rate_posterior",
)

#: Largest number of histories (alphabet size ** order) handled.
_MAX_HISTORIES = 2**20


@dataclass(frozen=True)
class EntropyRatePosterior:
    """
    Posterior samples of the entropy rate, from :func:`entropy_rate_posterior`.

    Attributes
    ----------
    samples : np.ndarray
        Entropy rates (bits per symbol) of chains drawn from the posterior.
    order : int
        The Markov order of the model.
    """

    samples: np.ndarray
    order: int

    @property
    def mean(self):
        """The posterior mean."""
        return float(np.mean(self.samples))

    def interval(self, confidence=0.95):
        """
        The equal-tailed credible interval with the given coverage.
        """
        tail = (1 - confidence) / 2
        low, high = np.quantile(self.samples, [tail, 1 - tail])
        return float(low), float(high)


def _stationary(P, M, tol=1e-12, max_iter=100_000):
    """
    Stationary distribution of the order-k word chain with rows ``P[u, b]``,
    where history ``u`` moves to ``(u * M + b) mod M**k`` on symbol ``b``.
    """
    n = P.shape[0]
    successors = (np.arange(n)[:, None] * M + np.arange(M)[None, :]) % n
    pi = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        new = np.bincount(successors.ravel(), weights=(pi[:, None] * P).ravel(), minlength=n)
        if np.abs(new - pi).sum() < tol:
            return new
        pi = new
    return pi


def entropy_rate_posterior(data, order, prior=0.5, n_samples=1000, prng=None):
    """
    Posterior samples of the entropy rate under an order-`order` Markov model.

    Each history's next-symbol distribution gets an independent symmetric
    Dirichlet(`prior`) prior, so its posterior is Dirichlet(counts + `prior`).
    For each posterior draw of the transition probabilities, the entropy rate
    :math:`h_\\mu = \\sum_u \\pi(u) H[X_k \\mid X_{0:k} = u]` is computed from the
    draw's own stationary distribution :math:`\\pi` :cite:`Strelioff2007`.

    Parameters
    ----------
    data : iterable or Trials
        The observed sequence, or independent :class:`~dit.inference.Trials`.
        Rows of a 2D array are joint symbols.
    order : int
        The Markov order :math:`k`; choose it with
        :func:`~dit.inference.select_markov_order`.
    prior : float
        The Dirichlet concentration per symbol. ``0.5`` is the Jeffreys prior,
        ``1`` the uniform prior.
    n_samples : int
        The number of posterior draws.
    prng : None, int, Generator, RandomState
        Source of randomness.

    Returns
    -------
    posterior : EntropyRatePosterior

    Notes
    -----
    Every history, observed or not, is included, and a positive `prior` makes the
    sampled chain irreducible, so each draw has a unique stationary distribution.
    Histories never observed contribute prior-dominated rows weighted by their
    (small) stationary probability. The posterior reflects uncertainty *given*
    the order; it does not account for choosing the order from the same data.
    """
    if prior <= 0:
        raise ValueError("`prior` must be positive.")
    if order < 0:
        raise ValueError("`order` must be non-negative.")
    trials, alphabet = standardize_trials(data)
    M = len(alphabet)
    n_histories = M**order
    if n_histories > _MAX_HISTORIES:
        raise ValueError(f"{M}**{order} histories is too many; lower the order.")
    rng = as_generator(prng)

    words = word_codes(trials, order + 1, M)
    counts = np.bincount(words, minlength=n_histories * M).reshape(n_histories, M).astype(float)
    alpha = counts + prior

    samples = np.empty(n_samples)
    for i in range(n_samples):
        P = rng.gamma(alpha)
        P /= P.sum(axis=1, keepdims=True)
        pi = _stationary(P, M)
        with np.errstate(divide="ignore", invalid="ignore"):
            H = -np.sum(np.where(P > 0, P * np.log2(P), 0.0), axis=1)
        samples[i] = float(pi @ H)
    return EntropyRatePosterior(samples, int(order))
