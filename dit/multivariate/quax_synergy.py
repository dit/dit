"""
Synergistic information via Synergistic Random Variables (SRVs).

Quax, Har-Shemesh & Sloot (2017), "Quantifying Synergistic Information
Using Intermediate Stochastic Variables", Entropy 19(2):85.
https://doi.org/10.3390/e19020085

An SRV S of sources X = {X_i} satisfies I(S:X) > 0 and I(S:X_i) = 0
for all i.  The synergistic information that a target Y stores about
sources X is defined as I(Y:S) where S maximises I(S:X) under the
zero-MI constraints.

This measure is *not* a PID synergy: synergistic and individual
information can coexist in Y simultaneously.
"""

from ..algorithms import BaseAuxVarOptimizer
from ..helpers import normalize_rvs
from ..math import prod
from ..shannon import conditional_entropy
from ..shannon import entropy as shannon_entropy
from ..utils import flatten, unitful

__all__ = (
    "max_synergistic_entropy",
    "quax_synergy",
)


class SRVOptimizer(BaseAuxVarOptimizer):
    """
    Find a Synergistic Random Variable (SRV) S of sources X that
    maximises I(S : X) subject to I(S : X_i) = 0 for all i.

    After optimisation, compute I(Y : S) as the synergistic information
    that target Y stores about X.

    Parameters
    ----------
    dist : Distribution
        The joint distribution over sources and target.
    sources : list of lists
        Each inner list gives the indices of one source variable group.
    target : list
        The indices of the target variable group.
    crvs : list, None
        Variables to condition on.
    bound : int, None
        Cardinality bound on S.  If None, a theoretical bound is used.
    """

    _PENALTY_WEIGHT = 500

    def __init__(self, dist, sources, target, crvs=None, bound=None):
        self._n_sources = len(sources)

        rvs = list(sources) + [list(target)]
        super().__init__(dist, rvs=rvs, crvs=crvs)

        self._source_indices = set(range(self._n_sources))
        self._target_index = {self._n_sources}

        theoretical_bound = self.compute_bound()
        bound = min(bound, theoretical_bound) if bound else theoretical_bound

        self._construct_auxvars([(self._source_indices | self._crvs, bound)])

        self._mi_closures = self._build_mi_closures()

        for i in sorted(self._source_indices):
            mi_i = self._mi_closures["per_source"][i]
            self.constraints.append(
                {
                    "type": "eq",
                    "fun": lambda x, mi_fn=mi_i: self._squared_mi(x, mi_fn),
                }
            )

        self._default_hops = 5

        self._additional_options = {
            "options": {
                "maxiter": 1000,
                "ftol": 1e-6,
                "eps": 1.4901161193847656e-9,
            },
        }

    def _build_mi_closures(self):
        """Pre-compute MI closure functions used in objective and constraints."""
        return {
            "joint": self._mutual_information(self._arvs, self._source_indices),
            "target": self._mutual_information(self._target_index, self._arvs),
            "per_source": {i: self._mutual_information(self._arvs, {i}) for i in sorted(self._source_indices)},
        }

    def compute_bound(self):
        """
        Upper bound on the cardinality of S.

        From the Caratheodory--Fenchel theorem: |S| <= prod(|X_i|) + 1.

        Returns
        -------
        bound : int
        """
        return prod(self._shape[i] for i in self._source_indices) + 1

    def _squared_mi(self, x, mi_fn):
        """Equality constraint residual: I(S:X_i)^2 == 0."""
        pmf = self.construct_joint(x)
        return mi_fn(pmf) ** 2

    def _objective(self):
        """
        Minimise -I(S : X_joint) with a quadratic penalty for violating
        the per-source zero-MI constraints.

        Returns
        -------
        obj : callable
        """
        mi_joint = self._mi_closures["joint"]
        per_source = self._mi_closures["per_source"]
        w = self._PENALTY_WEIGHT

        def objective(self, x):
            pmf = self.construct_joint(x)
            neg_mi = -mi_joint(pmf)
            penalty = sum(per_source[i](pmf) ** 2 for i in per_source)
            return neg_mi + w * penalty

        return objective

    def synergistic_information(self, x):
        """
        Compute I(Y : S) from an optimisation vector.

        Parameters
        ----------
        x : np.ndarray
            The optimisation vector (typically ``self._optima``).

        Returns
        -------
        isyn : float
        """
        pmf = self.construct_joint(x)
        return self._mi_closures["target"](pmf)


@unitful
def quax_synergy(dist, sources, target, crvs=None, niter=None, maxiter=1000, polish=1e-6, bound=None):
    """
    Compute the synergistic information I_syn(sources -> target) as
    defined by Quax, Har-Shemesh & Sloot (2017).

    Finds a Synergistic Random Variable (SRV) S that maximises I(S : X)
    subject to I(S : X_i) = 0 for each source X_i, then returns I(Y : S).

    .. math::

        I_{\\mathrm{syn}}(X \\to Y)
            = \\max_{S:\\; I(S:X)>0,\\; \\forall i\\, I(S:X_i)=0}
              I(Y : S)

    Parameters
    ----------
    dist : Distribution
        The joint distribution over sources and target.
    sources : list of lists
        Each inner list gives the indices (or names) of one source variable
        group X_i.
    target : list
        The indices (or names) of the target variable Y.
    crvs : list, None
        Variables to condition on.
    niter : int, None
        Number of basin-hopping restarts.
    maxiter : int
        Maximum iterations per local optimisation.
    polish : float, False
        If a float, perform a polishing pass zeroing probabilities below
        this threshold.  If False, skip polishing.
    bound : int, None
        Cardinality bound on S.  If None, a theoretical bound is used.

    Returns
    -------
    isyn : float
        The synergistic information, in bits (before unit conversion).
    """
    if len(sources) < 2:
        return 0.0

    opt = SRVOptimizer(dist, sources, target, crvs=crvs, bound=bound)
    opt.optimize(niter=niter, maxiter=maxiter, polish=polish)
    val = opt.synergistic_information(opt._optima)
    return max(val, 0.0)


@unitful
def max_synergistic_entropy(dist, rvs=None, crvs=None):
    """
    Compute the analytical upper bound on the mutual information that
    any SRV can have about a set of variables.

    .. math::

        H(X_1, \\ldots, X_n) - \\max_i H(X_i)

    This is the maximum possible synergistic entropy of the sources,
    per Equation 17 of Quax et al. (2017).

    Parameters
    ----------
    dist : Distribution
        The distribution from which the bound is calculated.
    rvs : list, None
        The random variable groups.  If None, each variable is its own
        group.
    crvs : list, None
        Variables to condition on.

    Returns
    -------
    bound : float
        The upper bound on synergistic entropy.
    """
    rvs, crvs = normalize_rvs(dist, rvs, crvs)

    all_idx = list(set(flatten(flatten(rvs))))

    if crvs:
        h_joint = conditional_entropy(dist, all_idx, crvs)
        h_max = max(conditional_entropy(dist, list(flatten(rv)), crvs) for rv in rvs)
    else:
        h_joint = shannon_entropy(dist, all_idx)
        h_max = max(shannon_entropy(dist, list(flatten(rv))) for rv in rvs)

    return max(h_joint - h_max, 0.0)
