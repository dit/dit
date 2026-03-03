"""
Functions for pruning or expanding the sample space of a distribution.

This can be important when calculating meet and join random variables. It
is also important for the calculations of various PID quantities.
"""

from dit.exceptions import InvalidOutcome

__all__ = (
    "expanded_samplespace",
    "pruned_samplespace",
)


def pruned_samplespace(d, sample_space=None):
    """
    Returns a new distribution with pruned sample space.

    The pruning is such that zero probability outcomes are removed.

    Parameters
    ----------
    d : distribution
        The distribution used to create the pruned distribution.

    sample_space : set
        A list of outcomes with zero probability that should be kept in the
        sample space. If `None`, then all outcomes with zero probability
        will be removed.

    Returns
    -------
    pd : distribution
        The distribution with a pruned sample space.

    """
    if sample_space is None:
        sample_space = []

    keep = set(sample_space)
    outcomes = []
    pmf = []
    for o, p in d.zipped(mode="atoms"):
        if not d.ops.is_null_exact(p) or o in keep:
            outcomes.append(o)
            pmf.append(p)

    # For numerical 1-D distributions, outcomes come back as bare values;
    # wrap them in 1-tuples for the constructor.
    if outcomes and not isinstance(outcomes[0], tuple):
        outcomes = [(o,) for o in outcomes]

    pd = d.__class__(outcomes, pmf, base=d.get_base())
    return pd


def expanded_samplespace(d, alphabets=None, union=True):
    """
    Returns a new distribution with an expanded sample space.

    Expand the sample space so that it is the Cartesian product of the
    alphabets for each random variable.

    Parameters
    ----------
    d : distribution
        The distribution used to create the pruned distribution.

    alphabets : list
        A list of alphabets, with length equal to the outcome length in `d`.
        Each alphabet specifies the alphabet to be used for a single index
        random variable. The sample space of the new distribution will be the
        Cartesian product of these alphabets.

    union : bool
        If True, then the alphabet for each random variable is unioned.
        The unioned alphabet is then used for each random variable.

    Returns
    -------
    ed : distribution
        The distribution with an expanded sample space.

    """
    import itertools

    joint = d.is_joint()

    if alphabets is None:
        alphabets = list(map(sorted, d.alphabet))
    elif joint and len(alphabets) != d.outcome_length():
        L = len(alphabets)
        raise Exception(f"You need to provide {L} alphabets")

    if joint and union:
        alphabet = set.union(*map(set, alphabets))
        alphabet = sorted(alphabet)
        alphabets = [alphabet] * len(alphabets)

    # Validate that all existing outcomes can be represented
    for o in d.outcomes:
        o_tuple = o if isinstance(o, tuple) else (o,)
        if joint:
            for i, v in enumerate(o_tuple):
                if v not in alphabets[i]:
                    raise InvalidOutcome(v, "not in expanded alphabet")
        else:
            if o_tuple[0] not in alphabets:
                raise InvalidOutcome(o, "not in expanded alphabet")

    # Build the new distribution by constructing the full sample space
    new_ss = list(itertools.product(*alphabets)) if joint else [(v,) for v in alphabets]

    # Map old outcomes to probabilities
    old_probs = dict(d.zipped())
    outcomes = []
    pmf = []
    for o in new_ss:
        lookup = o[0] if d._unwrap_scalar else o
        p = old_probs.get(lookup, 0.0)
        outcomes.append(o)
        pmf.append(p)

    ed = d.__class__(outcomes, pmf, base=d.get_base(), trim=False)
    return ed
