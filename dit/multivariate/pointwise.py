"""
Pointwise (local) forms of the Shannon-like information measures.

Each of these measures is a signed sum of conditional entropies. Replacing every
conditional entropy H[A | C] with the local entropy h(a | c) = -log2 p(a | c) of a
single outcome gives the measure's pointwise form, whose expectation over the
distribution is the measure itself.
"""

import numpy as np

from ..exceptions import InvalidOutcome, ditException
from ..helpers import normalize_rvs
from ..utils import build_table, flatten, powerset

__all__ = (
    "PointwiseMeasure",
    "pointwise_coinformation",
    "pointwise_dual_total_correlation",
    "pointwise_entropy",
    "pointwise_interaction_information",
    "pointwise_mutual_information",
    "pointwise_o_information",
    "pointwise_residual_entropy",
    "pointwise_s_information",
    "pointwise_total_correlation",
)


def _native(value):
    """Convert a numpy scalar to the matching Python value."""
    return value.item() if hasattr(value, "item") else value


class PointwiseMeasure:
    """
    The pointwise (local) values of an information measure.

    A pointwise measure assigns a value to each outcome of the variables the
    measure depends on, and averaging those values over their distribution
    gives the measure itself. Values are in the base of the distribution (bits
    for linear distributions) and are nan off its support.

    Attributes
    ----------
    dist : Distribution
        The joint distribution of the variables the measure depends on.
    data : xarray.DataArray
        The values, with one dimension per variable of `dist`.
    name : str
        A label for the measure, such as ``'i[X:Y|Z]'``.
    """

    def __init__(self, dist, data, name):
        """
        Parameters
        ----------
        dist : Distribution
            The distribution the values were computed from.
        data : xarray.DataArray
            The values, in the base of `dist`, with one dimension per variable
            they depend on (a subset of the variables of `dist`).
        name : str
            A label for the measure.
        """
        dims = [dim for dim in dist.dims if dim in data.dims]
        self._source = dist
        self.dist = dist.marginal(dims)
        support = self.dist._linear_data() > 0
        # Adding zero turns negative zeros, e.g. -log2(1), into zeros.
        self.data = data.where(support).transpose(*dims) + 0.0
        self.name = name
        self._index = {dim: {_native(v): i for i, v in enumerate(self.data.coords[dim].values)} for dim in dims}

    def __repr__(self):
        return f"<PointwiseMeasure {self.name}>"

    def __str__(self):
        return self.to_string()

    def _assignment(self, outcome):
        """Map `outcome` to a ``{variable: value}`` dict."""
        if isinstance(outcome, dict):
            return outcome
        if isinstance(outcome, np.ndarray):
            outcome = outcome.tolist()
        if isinstance(outcome, list):
            outcome = tuple(outcome)
        dims = self.data.dims
        if len(dims) == 1 and not isinstance(outcome, tuple) and outcome in self._index[dims[0]]:
            return {dims[0]: outcome}
        if isinstance(outcome, (tuple, str)):
            for names in (dims, self._source.dims):
                if len(outcome) == len(names):
                    return dict(zip(names, outcome, strict=True))
        raise InvalidOutcome(outcome)

    def __getitem__(self, outcome):
        """
        The value at an outcome.

        Parameters
        ----------
        outcome : tuple, str, dict, or symbol
            An outcome of `dist`, an outcome of the distribution the measure
            was computed from (the variables the measure does not depend on
            are ignored), or a dict mapping each variable of `dist` to a value.

        Returns
        -------
        value : float
            The pointwise value, nan if `outcome` is outside the support.

        Raises
        ------
        InvalidOutcome
            Raised if `outcome` is not in the sample space.
        """
        try:
            values = self._assignment(outcome)
            index = tuple(self._index[dim][values[dim]] for dim in self.data.dims)
        except (KeyError, TypeError) as err:
            raise InvalidOutcome(outcome) from err
        return float(self.data.values[index])

    def to_dict(self):
        """
        The value at each outcome in the support of `dist`.

        Returns
        -------
        values : dict
            ``{outcome: value}``.
        """
        return {outcome: self[outcome] for outcome, _ in self.dist.zipped()}

    def expectation(self, over=None):
        """
        Average the values over `dist`.

        Parameters
        ----------
        over : list, None
            The variables to average over, conditional on the remaining
            variables of `dist`. Variables the measure does not depend on are
            ignored. If None, average over every variable.

        Returns
        -------
        value : float or PointwiseMeasure
            The average if no variables remain, otherwise the partially
            averaged values as a pointwise measure on the remaining variables.

        Raises
        ------
        ditException
            Raised if `over` contains variables of neither `dist` nor the
            distribution the measure was computed from.
        """
        p = self.dist._linear_data()
        weighted = p * self.data
        if over is None:
            return float(weighted.sum())
        source = self._source.dims
        if not isinstance(over, (list, tuple, set, frozenset, str)):
            over = [over]
        try:
            over = {source[rv] if isinstance(rv, (int, np.integer)) else rv for rv in flatten(over)}
        except IndexError as err:
            raise ditException(f"Random variable index out of range: {over}") from err
        unknown = over - set(source)
        if unknown:
            raise ditException(f"Cannot average over {sorted(unknown)}: not variables of the distribution.")
        summed = [dim for dim in self.data.dims if dim in over]
        keep = [dim for dim in self.data.dims if dim not in over]
        if not keep:
            return float(weighted.sum())
        marginal = p.sum(summed)
        values = weighted.sum(summed) / marginal.where(marginal > 0)
        return PointwiseMeasure(self._source, values, f"E[{self.name} | {','.join(keep)}]")

    def to_string(self, digits=4):
        """
        A table of the values at each outcome in the support of `dist`.

        Parameters
        ----------
        digits : int
            The number of decimal places to show.

        Returns
        -------
        table : str
            One row per outcome, giving its value for each variable, its
            probability, and the pointwise value.
        """
        dims = list(self.dist.dims)
        table = build_table(dims + ["p", self.name])
        for column in ("p", self.name):
            table.float_format[column] = f"{digits + 2}.{digits}"
        for outcome, p in self.dist.zipped():
            symbols = outcome if isinstance(outcome, tuple) else (outcome,)
            table.add_row([str(s) for s in symbols] + [p, self[outcome]])
        return table.get_string()


def _prepare(dist, rvs, crvs):
    """
    Resolve `rvs` and `crvs` to variable names, and build the local entropies.

    Returns
    -------
    groups : list of list
        The variable names of each group in `rvs`.
    crvs : list
        The names of the variables to condition on.
    h : func
        ``h(rvs, crvs)`` gives the local entropy -log2 p(rvs | crvs), in bits,
        as a DataArray over the named variables.
    """
    if dist.is_symbolic():
        raise NotImplementedError("Pointwise measures do not support symbolic distributions.")
    rvs, crvs = normalize_rvs(dist, rvs, crvs)
    groups = [dist._resolve_rv_names(list(rv)) for rv in rvs]
    crvs = dist._resolve_rv_names(crvs)

    logs = {}

    def log_p(names):
        key = frozenset(names)
        if key not in logs:
            p = dist.marginal(sorted(key))._linear_data()
            logs[key] = np.log2(p.where(p > 0))
        return logs[key]

    def h(rvs, crvs=()):
        rvs, crvs = set(rvs), set(crvs)
        local = -log_p(rvs | crvs)
        if crvs:
            local = local + log_p(crvs)
        return local

    return groups, crvs, h


def _label(symbol, groups, crvs):
    """A label such as ``i[X:Y|Z]``."""
    label = ":".join(",".join(map(str, group)) for group in groups)
    if crvs:
        label += "|" + ",".join(map(str, crvs))
    return f"{symbol}[{label}]"


def _pointwise(dist, values, name):
    """Convert `values` from bits to the base of `dist` and wrap them."""
    if dist.is_log():
        values = values / np.log2(dist.get_base(numerical=True))
    return PointwiseMeasure(dist, values, name)


def _coinformation(h, groups, crvs):
    return sum((-1) ** (len(sub) + 1) * h(set().union(*sub), crvs) for sub in powerset(groups) if sub)


def _total_correlation(h, groups, crvs):
    return sum(h(group, crvs) for group in groups) - h(set().union(*groups), crvs)


def _residual_entropy(h, groups, crvs):
    union = set().union(*groups)
    return sum(h(group, (union - set(group)) | set(crvs)) for group in groups)


def _dual_total_correlation(h, groups, crvs):
    return h(set().union(*groups), crvs) - _residual_entropy(h, groups, crvs)


def pointwise_entropy(dist, rvs=None, crvs=None):
    """
    Compute the pointwise (local) conditional entropy, h(x|z) = -log2 p(x|z).

    Parameters
    ----------
    dist : Distribution
        The distribution from which the pointwise entropy is calculated.
    rvs : list, None
        The indexes of the random variables whose entropy is calculated. If
        None, then the entropy is calculated over all random variables.
    crvs : list, None
        The indexes of the random variables to condition on. If None, then no
        variables are conditioned on.

    Returns
    -------
    h : PointwiseMeasure
        The pointwise entropy, whose expectation is the entropy.

    Examples
    --------
    >>> d = dit.Distribution(['A', 'B', 'C'], [1/2, 1/4, 1/4])
    >>> dit.multivariate.pointwise_entropy(d)['B']
    2.0
    """
    if rvs is not None:
        rvs = [list(flatten(rvs))]
    groups, crvs, h = _prepare(dist, rvs, crvs)
    rvs = list(flatten(groups))
    return _pointwise(dist, h(rvs, crvs), _label("h", [rvs], crvs))


def pointwise_coinformation(dist, rvs=None, crvs=None):
    """
    Compute the pointwise (local) coinformation.

    For two groups of random variables this is the pointwise mutual
    information, i(x:y|z) = log2 p(x, y|z) / (p(x|z) p(y|z)).

    Parameters
    ----------
    dist : Distribution
        The distribution from which the pointwise coinformation is calculated.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables of one group. If None, then each random variable is its own
        group.
    crvs : list, None
        The indexes of the random variables to condition on. If None, then no
        variables are conditioned on.

    Returns
    -------
    i : PointwiseMeasure
        The pointwise coinformation, whose expectation is the coinformation.

    Examples
    --------
    >>> d = dit.Distribution(['00', '01', '10', '11'], [0.4, 0.1, 0.1, 0.4])
    >>> pmi = dit.multivariate.pointwise_mutual_information(d)
    >>> pmi['00'], pmi['01']
    (0.6780719051126378, -1.3219280948873622)
    >>> pmi.expectation()
    0.27807190511263785
    """
    groups, crvs, h = _prepare(dist, rvs, crvs)
    return _pointwise(dist, _coinformation(h, groups, crvs), _label("i", groups, crvs))


pointwise_mutual_information = pointwise_coinformation


def pointwise_total_correlation(dist, rvs=None, crvs=None):
    """
    Compute the pointwise (local) total correlation,
    t(x_0, ..., x_n | z) = sum_i h(x_i | z) - h(x_0, ..., x_n | z).

    Parameters
    ----------
    dist : Distribution
        The distribution from which the pointwise total correlation is
        calculated.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables of one group. If None, then each random variable is its own
        group.
    crvs : list, None
        The indexes of the random variables to condition on. If None, then no
        variables are conditioned on.

    Returns
    -------
    t : PointwiseMeasure
        The pointwise total correlation, whose expectation is the total
        correlation.
    """
    groups, crvs, h = _prepare(dist, rvs, crvs)
    return _pointwise(dist, _total_correlation(h, groups, crvs), _label("t", groups, crvs))


def pointwise_dual_total_correlation(dist, rvs=None, crvs=None):
    """
    Compute the pointwise (local) dual total correlation,
    b(x_0, ..., x_n | z) = h(x_0, ..., x_n | z) - sum_i h(x_i | x_{-i}, z),
    where x_{-i} is every group but the i-th.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the pointwise dual total correlation is
        calculated.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables of one group. If None, then each random variable is its own
        group.
    crvs : list, None
        The indexes of the random variables to condition on. If None, then no
        variables are conditioned on.

    Returns
    -------
    b : PointwiseMeasure
        The pointwise dual total correlation, whose expectation is the dual
        total correlation.
    """
    groups, crvs, h = _prepare(dist, rvs, crvs)
    return _pointwise(dist, _dual_total_correlation(h, groups, crvs), _label("b", groups, crvs))


def pointwise_residual_entropy(dist, rvs=None, crvs=None):
    """
    Compute the pointwise (local) residual entropy,
    r(x_0, ..., x_n | z) = sum_i h(x_i | x_{-i}, z), where x_{-i} is every
    group but the i-th.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the pointwise residual entropy is
        calculated.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables of one group. If None, then each random variable is its own
        group.
    crvs : list, None
        The indexes of the random variables to condition on. If None, then no
        variables are conditioned on.

    Returns
    -------
    r : PointwiseMeasure
        The pointwise residual entropy, whose expectation is the residual
        entropy.
    """
    groups, crvs, h = _prepare(dist, rvs, crvs)
    return _pointwise(dist, _residual_entropy(h, groups, crvs), _label("r", groups, crvs))


def pointwise_interaction_information(dist, rvs=None, crvs=None):
    """
    Compute the pointwise (local) interaction information, (-1)^n times the
    pointwise coinformation of n groups.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the pointwise interaction information is
        calculated.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables of one group. If None, then each random variable is its own
        group.
    crvs : list, None
        The indexes of the random variables to condition on. If None, then no
        variables are conditioned on.

    Returns
    -------
    ii : PointwiseMeasure
        The pointwise interaction information, whose expectation is the
        interaction information.
    """
    groups, crvs, h = _prepare(dist, rvs, crvs)
    values = (-1) ** len(groups) * _coinformation(h, groups, crvs)
    return _pointwise(dist, values, _label("ii", groups, crvs))


def pointwise_o_information(dist, rvs=None, crvs=None):
    """
    Compute the pointwise (local) O-information :cite:`scagliarini2022quantifying`,
    the pointwise total correlation minus the pointwise dual total correlation.
    Positive values mark redundancy-dominated outcomes, and negative values
    synergy-dominated ones.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the pointwise O-information is calculated.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables of one group. If None, then each random variable is its own
        group.
    crvs : list, None
        The indexes of the random variables to condition on. If None, then no
        variables are conditioned on.

    Returns
    -------
    o : PointwiseMeasure
        The pointwise O-information, whose expectation is the O-information.
    """
    groups, crvs, h = _prepare(dist, rvs, crvs)
    values = _total_correlation(h, groups, crvs) - _dual_total_correlation(h, groups, crvs)
    return _pointwise(dist, values, _label("o", groups, crvs))


def pointwise_s_information(dist, rvs=None, crvs=None):
    """
    Compute the pointwise (local) S-information, the pointwise total
    correlation plus the pointwise dual total correlation.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the pointwise S-information is calculated.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables of one group. If None, then each random variable is its own
        group.
    crvs : list, None
        The indexes of the random variables to condition on. If None, then no
        variables are conditioned on.

    Returns
    -------
    s : PointwiseMeasure
        The pointwise S-information, whose expectation is the S-information.
    """
    groups, crvs, h = _prepare(dist, rvs, crvs)
    values = _total_correlation(h, groups, crvs) + _dual_total_correlation(h, groups, crvs)
    return _pointwise(dist, values, _label("s", groups, crvs))
