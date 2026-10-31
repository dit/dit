"""
The redundancy measure of Griffith & Ho, their I_alpha:

    I_GH(X_1, ..., X_n -> Y) = max_Q I(Q; Y)  such that  Q - X_i - Y for every i,

where Q may depend on the sources and the target jointly.

An outcome q of Q is given by u(z) = s(q | z) on the outcomes z of the sources and the
target, and Q - X_i - Y says s(q | x_i, y) = s(q | x_i), a set of linear equalities in u.
So the vectors of the outcomes lie in a polyhedral cone and add up to the all-ones
vector, and I(Q; Y) is a sum over the outcomes of a convex, positively homogeneous
function of them. As for the degradation redundancy of `ideg`, the maximum is therefore
the value of a linear program over the extreme rays of the cone.

References
----------
.. [1] V. Griffith and T. Ho, "Quantifying Redundant Information in Predicting a Target
       Random Variable", Entropy 17, 4644, 2015.
.. [2] A. Kolchinsky, "A Novel Approach to the Partial Information Decomposition",
       Entropy 24, 403, 2022.
"""

from collections import defaultdict
from fractions import Fraction

import numpy as np

from ...algorithms.optimization import BaseAuxVarOptimizer, BaseConvexOptimizer, OptimizationException
from ...math import prod
from ...shannon import entropy, mutual_information
from ..pid import BasePID
from .ideg import _double_description, _integer_row, _ray_program, _verified_cdd_rays

__all__ = ("PID_GH",)


def _independence_cone(pjoint):
    """
    The equalities under which an outcome of Q is independent of the target given each
    source.

    Parameters
    ----------
    pjoint : Distribution
        The distribution of the sources and the target, the last variable.

    Returns
    -------
    exact : list of list of int
        The rows a of the equalities a @ u = 0, on the outcomes z of positive probability,
        scaled to integers.
    rows : ndarray
        The same rows in floating point, each scaled to have largest entry 1.
    columns : ndarray, shape (|Y|, |Z|)
        The map p(z | y) [z_y = y] from u to the column of K_Q, up to scale.
    p_y : ndarray
        The probabilities of the target values.
    """
    outcomes, probs = zip(*pjoint.zipped(), strict=True)
    probs = [Fraction(float(p)) for p in probs]
    targets = list(dict.fromkeys(z[-1] for z in outcomes))
    p_y = defaultdict(Fraction)
    for z, p in zip(outcomes, probs, strict=True):
        p_y[z[-1]] += p

    exact = []
    for i in range(len(outcomes[0]) - 1):
        p_x, p_xy = defaultdict(Fraction), defaultdict(Fraction)
        for z, p in zip(outcomes, probs, strict=True):
            p_x[z[i]] += p
            p_xy[z[i], z[-1]] += p
        for (x, y), q in p_xy.items():
            # p(q, x, y) p(x) = p(q, x) p(x, y)
            row = [
                p * ((z[-1] == y) * p_x[x] - q) if z[i] == x else Fraction(0)
                for z, p in zip(outcomes, probs, strict=True)
            ]
            if any(row):
                exact.append(_integer_row(row))

    rows = np.array([[float(a) for a in row] for row in exact]).reshape(-1, len(outcomes))
    if rows.size:
        rows /= np.abs(rows).max(axis=1, keepdims=True)
    columns = np.array(
        [[float(p / p_y[y]) if z[-1] == y else 0.0 for z, p in zip(outcomes, probs, strict=True)] for y in targets]
    )
    return exact, rows, columns, np.array([float(p_y[y]) for y in targets])


def _gh_lp(pjoint):
    """
    Compute I_GH exactly, by a linear program over the extreme rays of the cone of
    outcomes of Q which are independent of the target given each source.

    Parameters
    ----------
    pjoint : Distribution
        The distribution of the sources and the target, the last variable.

    Returns
    -------
    igh : float
        The value of I_GH, in bits.
    """
    exact, rows, columns, p_y = _independence_cone(pjoint)
    if len(p_y) < 2:
        return 0.0
    dim = columns.shape[1]
    rays = _verified_cdd_rays(rows, dim)
    if rays is None:
        rays = _double_description(exact, dim)
    rays = np.clip(rays, 0.0, None)
    res = _ray_program(rays, rays @ columns.T, p_y)
    if res.status != 0:
        # The rays from cddlib can pass verification and still miss some extreme rays.
        rays = np.clip(_double_description(exact, dim), 0.0, None)
        res = _ray_program(rays, rays @ columns.T, p_y)
    if res.status != 0:
        raise RuntimeError(f"The linear program for I_GH failed: {res.message}")
    return max(0.0, -res.fun)


class GHOptimizer(BaseConvexOptimizer, BaseAuxVarOptimizer):
    """
    Optimizer for the Griffith & Ho redundancy.
    """

    construct_initial = BaseAuxVarOptimizer.construct_copy_initial

    def __init__(self, dist, sources, target, bound=None):
        """
        Initialize the optimizer.

        Parameters
        ----------
        dist : Distribution
            The distribution to compute i_gh of.
        sources : list, None
            A list of lists. Each inner list specifies the indexes of the random
            variables used to calculate the intrinsic mutual information. If None,
            then it is calculated over all random variables, which is equivalent
            to passing `rvs=dist.rvs`.
        target : list
            A single list of indexes specifying the random variables to
            treat as the target.
        bound : int, None
            Specifies a bound on the size of the auxiliary random variable. If None,
            then the theoretical bound is used.
        """
        super().__init__(dist, sources, target)

        q_size = prod(self._shape) + 1
        bound = min([bound, q_size]) if bound is not None else q_size

        self._construct_auxvars([(self._rvs | self._crvs, bound)])

        entropies = [entropy(dist, source + target) for source in sources]
        mutual_informations = [mutual_information(dist, source, target) for source in sources]
        trivial = all(abs(h - i) < 1e-6 for h, i in zip(entropies, mutual_informations, strict=True))
        if not trivial:
            self.constraints += [
                {
                    "type": "eq",
                    "fun": self.constraint_markov_chains(),
                    "jac": self.constraint_markov_chains_jac(),
                },
            ]

        self._additional_options = {
            "options": {
                "maxiter": 300,
                "ftol": 1e-6,
                "eps": 1.4901161193847656e-9,
            }
        }

    def constraint_markov_chains(self):
        """
        Constrain I[Q : Y | Xi] = 0.

        Parameters
        ----------
        x : np.ndarray
            An optimization vector.
        """
        cmis = [self._conditional_mutual_information(self._crvs, self._arvs, {rv}) for rv in self._rvs]

        def constraint(x):
            pmf = self.construct_joint(x)
            return sum(cmi(pmf) for cmi in cmis)

        return constraint

    def constraint_markov_chains_jac(self):
        """
        Analytic Jacobian of :meth:`constraint_markov_chains` w.r.t. ``x``.

        Composes the sum of conditional-mutual-information pmf-gradients with the
        channel-parametrization VJP, mirroring :meth:`_jacobian`. Lets SLSQP
        avoid finite-differencing the equality constraint.
        """
        cmi_grads = [self._conditional_mutual_information_grad(self._crvs, self._arvs, {rv}) for rv in self._rvs]

        def jac(x):
            pmf = self.construct_joint(x)
            g = sum(cmi_grad(pmf) for cmi_grad in cmi_grads)
            return self._construct_joint_vjp(x, np.asarray(g, dtype=float))

        return jac

    def _objective_gradient(self):
        """
        Analytic gradient of the ``-I[Q : Y]`` objective w.r.t. the joint.

        Matches the objective ``mi = self._mutual_information(self._arvs,
        self._crvs)`` in :meth:`_objective`; wired into SciPy via the base
        :meth:`_jacobian`.
        """
        grad = self._mutual_information_grad(self._arvs, self._crvs)
        return lambda pmf: -grad(pmf)

    def _objective(self):
        """
        The mutual information between the target and the auxiliary variable.

        Returns
        -------
        obj : func
            The objective function.
        """
        mi = self._mutual_information(self._arvs, self._crvs)

        def objective(self, x):
            """
            Compute I[Q : Y]

            Parameters
            ----------
            x : np.ndarray
                An optimization vector.

            Returns
            -------
            obj : float
                The value of the objective.
            """
            pmf = self.construct_joint(x)
            return -mi(pmf)

        return objective


class PID_GH(BasePID):
    """
    The I_GH measure defined by Griffith & Ho.

    Redundancy is the maximum I(Q; Y) over variables Q which are independent of the target
    given each source. By default it is computed exactly, by a linear program over the
    extreme rays of the cone of such outcomes of Q. Passing ``method='optimizer'`` instead
    runs a local optimization of a bounded auxiliary variable, which only gives a lower
    bound.

    References
    ----------
    V. Griffith and T. Ho, "Quantifying Redundant Information in Predicting a Target Random
    Variable", Entropy 17, 4644, 2015.
    """

    _name = "I_GH"

    @staticmethod
    def _measure(d, sources, target, method="exact"):
        """
        Compute I_GH(sources : target) =
            \\max I[Q : target] such that I[Q : Y | Xi] = 0

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_gh for.
        sources : iterable of iterables
            The source variables.
        target : iterable
            The target variable.
        method : {'exact', 'optimizer'}
            'exact' solves a linear program over the extreme rays of the cone of outcomes
            of Q which are independent of the target given each source. 'optimizer' runs
            a local optimization, which only gives a lower bound.

        Returns
        -------
        igh : float
            The value of I_GH.
        """
        if method not in ("exact", "optimizer"):
            raise ValueError(f"method must be 'exact' or 'optimizer', not {method!r}")
        if len(sources) == 1:
            return mutual_information(d, sources[0], target)
        if method == "exact":
            return _gh_lp(d.coalesce(list(sources) + [target]))
        md = d.coalesce(sources)
        upper_bound = prod(len(a) for a in md.alphabet) + 1
        for bound in range(upper_bound, md.outcome_length(), -1):
            try:
                gho = GHOptimizer(d, sources, target, bound=bound)
                gho.optimize()
                break
            except OptimizationException:
                continue
        else:
            msg = "I_GH: optimization failed for every auxiliary bound."
            raise OptimizationException(msg)
        # The objective is -I[Q : Y].
        return -gho.objective(gho._optima)
