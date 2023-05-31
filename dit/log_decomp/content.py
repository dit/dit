"""
Tools for examining the set-theoretical contents of variables in logarithmic decomposition.
For more information see the preprint:
https://arxiv.org/abs/2305.07554
"""
# Import some things.
import more_itertools
from .measures import interior_loss

# Specify all functions defined in this module.
__all__=[
    'content'
]

def content(dist, rvs):
    """
    Compute the content of a collection of random variables in a distribution.

    Parameters
    ----------
    dist : dit.Distribution
        The distribution to be analysed.
    rvs : list
        A list of random variables over which the content is taken.

    Returns
    -------
    variable_content : set
        The set of atomic contents associated to the union of the random variables 'rvs'.

    Notes
    -----
    For more information, see the logarithmic decomposition preprint:
    https://arxiv.org/abs/2305.07554
    """
    # Get the list of all outcomes.
    outcomes = dist.outcomes
    # Create a tuple from the named rvs.
    rv_tuple = tuple(dist._rvs[name] for name in rvs)
    # Using this tuple, create a list of outcomes selecting these variables.
    outcomes_on_rvs = tuple(tuple(''.join(outcome[j] for j in rv_tuple)) for outcome in outcomes)
    # Initialise an empty content set.
    variable_content = set([])
    # Now test to see which indices are different.
    for subset in more_itertools.powerset(outcomes_on_rvs):
        # The atom is detected if the length of the unique subset is not 1.
        if len(set(subset)) > 1:
            # For each of the combinations of original outcomes,
            for outcome_tuple in more_itertools.powerset(outcomes):
                # Check if it looks like the marginalised set
                if tuple(tuple(''.join(outcome[j] for j in rv_tuple))
                          for outcome in outcome_tuple) == subset:
                    # Then add it to the content.
                    variable_content.add(outcome_tuple)
    return variable_content
