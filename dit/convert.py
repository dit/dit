"""
Helper functions to convert between Distribution and ScalarDistribution.

"""

import dit

__all__ = [
    'DtoSD',
    'SDtoD'
]

def DtoSD(dist, extract):
    """
    Convert a Distribution to a ScalarDistribution.

    Parameters
    ----------
    dist : Distribution
        The Distribution to convert to a ScalarDistribution.
    extract : bool
        If `True` and the outcome length is 1, then we extract the sole
        element from each outcome and use that value as the scalar outcome.

    """
    if extract and dist.outcome_length() == 1:
        outcomes = tuple(outcome[0] for outcome in dist.outcomes)
        sample_space = dist.alphabet[0]
    else:
        outcomes = dist.outcomes
        sample_space = None

    # If people really want it, we can use _make_distribution.
    # But we have to decide if we want to set the alphabet to the
    # entire sample or just the sample space represented in outcomes.
    d = dit.ScalarDistribution(outcomes, dist.pmf,
                               sample_space=sample_space,
                               base=dist.get_base(),
                               prng=dist.prng,
                               sort=False,
                               sparse=dist.is_sparse(),
                               validate=False)

    return d

def SDtoD(dist):
    """
    Convert a ScalarDistribution to a Distribution.

    Parameters
    ----------
    dist : ScalarDistribution
        The ScalarDistribution to convert to a Distribution.

    """
    from dit.exceptions import ditException, InvalidDistribution
    import dit.validate as v

    if len(dist.pmf) == 0:
        msg = "Cannot convert from empty ScalarDistribution."
        raise InvalidDistribution(msg)

    # Check if every element of the sample space is a sequence of the same
    # length. If so, this is an easy conversion.  If not, then we make
    # every outcome a 1-tuple and then construct the joint distribution.
    try:
        # Each outcome is of the same class.
        v.validate_outcome_class(dist.outcomes)
        # Each outcome has the same length.
        v.validate_outcome_length(dist.outcomes)
        # Each outcome is a 'sequence'.
        v.validate_sequence(dist.outcomes[0])
    except ditException:
        # Nested translation.
        outcomes = [(o,) for o in dist.outcomes]
    else:
        outcomes = dist.outcomes

    d = dit.Distribution(outcomes, dist.pmf, base=dist.get_base())

    return d
