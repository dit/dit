r"""
The communication rate needed for distributed channel synthesis.

An encoder observes an i.i.d. source :math:`X` and sends a message at rate
:math:`R` to a decoder which must produce :math:`Y` such that the pair
:math:`(X^n, Y^n)` is statistically indistinguishable from i.i.d. draws of
:math:`p(x, y)`. Both nodes additionally share common randomness at rate
:math:`R_0`, independent of :math:`X`. Cuff :cite:`cuff2013distributed` showed
the achievable rate pairs are exactly those for which some :math:`U` with
:math:`X - U - Y` satisfies

.. math::

   R \ge \I{X : U}, \qquad R + R_0 \ge \I{X, Y : U}.
"""

import numpy as np

from ...exceptions import OptimizationException, ditException
from ...helpers import normalize_rvs
from ...utils import unitful
from .._backend import _make_backend_subclass
from ..coinformation import coinformation
from ..dual_total_correlation import dual_total_correlation
from .base_markov_optimizer import MarkovVarOptimizer

__all__ = ("channel_synthesis_rate",)


class ChannelSynthesisMixin:
    """
    Compute the minimal communication rate for synthesizing the channel from the
    first variable in `rvs` to the remaining ones, given a common-randomness rate.

    The rate :math:`\\max\\{I[X:U], I[XY:U] - R_0\\}` is usually minimized on the
    kink where the two terms are equal, which gradient-based solvers handle
    poorly. Each instance therefore minimizes one smooth `branch` of it,
    ``"input"`` (:math:`I[X:U]` where it is the larger term) or ``"joint"``
    (:math:`I[XY:U] - R_0` where that is the larger term); the rate is the
    smaller of the two branch minima.
    """

    name = "channel synthesis"
    description = "min max{I[X:U], I[XY:U] - R_0} such that U renders X, Y_i independent"

    def __init__(self, dist, rvs=None, crvs=None, bound=None, common_randomness=0.0, public=False, branch="input"):
        """
        Initialize the optimizer.

        Parameters
        ----------
        dist : Distribution
            The distribution of the channel input and outputs.
        rvs : list
            A list of lists. The first inner list is the channel input, the
            remaining are the outputs, one per decoder.
        crvs : list, None
            A single list of indexes specifying the random variables to
            condition on. If None, then no variables are conditioned on.
        bound : int, None
            Place an artificial bound on the size of U.
        common_randomness : float
            The rate of common randomness, :math:`R_0`, in bits.
        public : bool
            Whether the message must look independent of the synthesized
            input and outputs.
        branch : str
            Which term of the rate to minimize, ``"input"`` or ``"joint"``.
            Ignored when `public` is True.
        """
        if branch not in ("input", "joint"):
            raise ValueError("branch must be 'input' or 'joint'")
        self._common_randomness = common_randomness
        self._public = public
        self._branch = "input" if public else branch
        mutual_information = coinformation(dist, [rvs[0], [i for rv in rvs[1:] for i in rv]], crvs)
        if self._branch == "input":
            self._objective_bound = mutual_information
        else:
            self._objective_bound = dual_total_correlation(dist, rvs, crvs) - common_randomness
        super().__init__(dist, rvs=rvs, crvs=crvs, bound=bound)

        self._input_information = self._conditional_mutual_information(self._input, self._W, self._crvs)
        self._joint_information = self._conditional_mutual_information(self._rvs, self._W, self._crvs)
        self.constraints += [
            {
                "type": "ineq",
                "fun": self.constraint_branch,
            },
        ]

    def compute_bound(self):
        """
        An upper bound on the cardinality of the auxiliary random variable, from
        the Caratheodory-Fenchel theorem.

        Returns
        -------
        bound : int
            The upper bound on the alphabet size of the auxiliary variable.
        """
        return np.prod(self._shape) + 1

    @property
    def _input(self):
        return {min(self._rvs)}

    @property
    def _outputs(self):
        return self._rvs - self._input

    def _terms(self, x):
        """The two rate terms, :math:`I[X:U]` and :math:`I[XY:U] - R_0`."""
        pmf = self.construct_joint(x)
        return self._input_information(pmf), self._joint_information(pmf) - self._common_randomness

    def constraint_branch(self, x):
        """
        Keep the optimization on its branch. For a public channel this is
        :math:`R_0 - I[X, Y : U] \\ge 0`; otherwise the minimized term must be
        the larger of the two.

        Parameters
        ----------
        x : np.ndarray
            An optimization vector.

        Returns
        -------
        slack : float
            Non-negative when the constraint is satisfied.
        """
        input_term, joint_term = self._terms(x)
        if self._public:
            return -joint_term
        if self._branch == "input":
            return input_term - joint_term
        return joint_term - input_term

    def rate(self, x):
        """
        The communication rate implied by the auxiliary variable.

        Parameters
        ----------
        x : np.ndarray
            An optimization vector.

        Returns
        -------
        rate : float
            :math:`I[X:U]` for a public channel, else
            :math:`\\max\\{I[X:U], I[XY:U] - R_0\\}`.
        """
        input_term, joint_term = (
            float(t.detach().cpu().item()) if hasattr(t, "detach") else float(t) for t in self._terms(x)
        )
        return input_term if self._public else max(input_term, joint_term)

    def _objective(self):
        """
        The term of the rate minimized on this branch.

        Returns
        -------
        obj : func
            The objective function.
        """
        index = 0 if self._branch == "input" else 1

        def objective(self, x):
            """
            Compute the branch's rate term.

            Parameters
            ----------
            x : np.ndarray
                An optimization vector.

            Returns
            -------
            obj : float
                The value of the objective.
            """
            return self._terms(x)[index]

        return objective

    def local_randomness(self):
        """
        The rate of private randomness the decoders need at the optimum,
        :math:`H[Y_1, \\ldots, Y_m \\mid U]` :cite:`cuff2013distributed`.

        Returns
        -------
        r_l : float
            The local-randomness rate.
        """
        pmf = self.construct_joint(self._optima)
        return float(self._entropy(self._outputs, self._W | self._crvs)(pmf))


class ChannelSynthesis(ChannelSynthesisMixin, MarkovVarOptimizer):
    """
    Compute the minimal communication rate for synthesizing a channel.
    """


def _as_groups(outputs):
    """Interpret `outputs` as a list of output groups, one per decoder."""
    if all(isinstance(o, (list, tuple)) for o in outputs):
        return [list(o) for o in outputs]
    return [list(outputs)]


@unitful
def channel_synthesis_rate(
    dist,
    inputs,
    outputs,
    common_randomness=0.0,
    public=False,
    crvs=None,
    niter=None,
    maxiter=1000,
    polish=1e-6,
    bound=None,
    backend="numpy",
):
    r"""
    The minimal communication rate for distributed channel synthesis
    :cite:`cuff2013distributed`.

    An encoder sees i.i.d. `inputs` :math:`X` and must, together with decoders
    sharing common randomness at rate :math:`R_0`, make the decoders produce
    `outputs` :math:`Y_1, \ldots, Y_m` that are indistinguishable from i.i.d.
    draws of the joint distribution. The minimal rate is

    .. math::

       R(R_0) = \min_{U} \max\left\{ \I{X : U},\ \I{X, Y_1, \ldots, Y_m : U} - R_0 \right\},

    where :math:`U` renders :math:`X, Y_1, \ldots, Y_m` mutually conditionally
    independent. With one output, :math:`R(0)` is the Wyner common information
    and :math:`R(R_0) = \I{X : Y}` once :math:`R_0 \ge \H{Y \dagger X}`, the
    necessary conditional entropy :cite:`cuff2010coordination`. Several outputs
    form a broadcast channel whose decoders all receive the same message.

    If the message is sent over a public channel and must look independent of the
    synthesized variables, the rate pair must instead satisfy
    :math:`R \ge \I{X : U}` and :math:`R_0 \ge \I{X, Y_1, \ldots, Y_m : U}`, so
    that :math:`R_0` must be at least the Wyner common information.

    Parameters
    ----------
    dist : Distribution
        The joint distribution of the channel input and outputs.
    inputs : list
        The indexes of the channel input.
    outputs : list
        The indexes of the channel output, or a list of such lists, one per
        decoder of a broadcast channel.
    common_randomness : float
        The rate of common randomness, :math:`R_0`.
    public : bool
        Whether the message is public. Defaults to False.
    crvs : list, None
        A single list of indexes specifying the random variables to condition
        on. If None, then no variables are conditioned on.
    niter : int > 0
        Number of basin hoppings to perform during the optimization.
    maxiter : int > 0
        The number of iterations of the optimization subroutine to perform.
    polish : False, float
        Whether to polish the result or not. If a float, this will perform a
        second optimization seeded with the result of the first, but with
        smaller tolerances and probabilities below polish set to 0. If False,
        don't polish.
    bound : int
        Bound the size of the auxiliary variable.
    backend : str
        The optimization backend to use. One of ``'numpy'`` (default),
        ``'jax'``, or ``'torch'``.

    Returns
    -------
    rate : float
        The minimal communication rate.

    Raises
    ------
    ditException
        Raised if `public` is True and no auxiliary variable meets the
        common-randomness budget, i.e. it is below the Wyner common information.
    """
    rvs = [list(inputs), *_as_groups(outputs)]
    rvs, crvs = normalize_rvs(dist, rvs, crvs)

    dtc = dual_total_correlation(dist, rvs, crvs)
    if public and common_randomness < dtc - 1e-9:
        msg = "The common randomness rate is below the Wyner common information, so a public channel cannot synthesize this distribution."
        raise ditException(msg)

    cls = _make_backend_subclass(ChannelSynthesis, backend)
    rates = []
    for branch in ["input"] if public else ["input", "joint"]:
        cs = cls(dist, rvs, crvs, bound, common_randomness=common_randomness, public=public, branch=branch)
        try:
            cs.optimize(niter=niter, maxiter=maxiter, polish=polish)
        except OptimizationException:
            continue
        if cs.constraint_branch(cs._optima) >= -1e-4:
            rates.append(cs.rate(cs._optima))

    if not rates:
        msg = "No auxiliary variable found within the common randomness budget; it may be below the Wyner common information."
        raise ditException(msg)

    mutual_information = coinformation(dist, [rvs[0], [i for rv in rvs[1:] for i in rv]], crvs)
    lower = mutual_information if public else max(mutual_information, dtc - common_randomness)
    return max(min(rates), lower, 0.0)
