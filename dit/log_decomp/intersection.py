"""
Tools for computing shared information between variables.
For more information see the preprint:
https://arxiv.org/abs/2305.07554
"""
# Specify all functions defined in this module.
__all__=[
    'shared'
]

# Do some imports.
from ..npdist import Distribution
from .content import content, measure
from .combinatorics import get_n_atoms, upper_set

def shared(dist, list_of_rv_groups, order = 2, log_base = 2):
    """
    Returns the shared information derived from the logarithmic decomposition
    shared between the groups of random variables in 'list_of_rv_groups'.

    Parameters
    ----------
    dist : dit.Distribution
        The distribution to be examined.
    list_of_rv_groups : list
        A list of lists, containing random variable groupings for computing shared information.
    order : int, string
        Default value is 2. If set to n, redundancy is computed using n-atom upper sets.
        If given as "even", then will be generated for all even n-atoms. Likewise for "odd".
    log_base : int, float
        The base of the logarithm used for computing information.

    Returns
    -------
    shared_information : float
        The information shared between the groups of variables.

    Examples
    --------
    To see what information X and Y redundantly give about Z, do [["X"], ["Y"], ["Z"]]
    For information they give together about Z, do [["X", "Y"], ["Z"]].
    """
    # Check the inputs are the correct types.
    if not isinstance(dist, Distribution):
        raise TypeError("'dist' must be a dit distribution.")
    if not isinstance(list_of_rv_groups, list):
        raise TypeError("'list_of_rv_groups' must be a list of lists of random variables.")
    if not isinstance(order, (int, str)):
        raise TypeError("'order' must be either an integer, 'even' or 'odd'.")
    if not isinstance(log_base, (int, float)):
        raise TypeError("'log_base' must be an int or a float.")
    if isinstance(order, str):
        if order not in ["even", "odd"]:
            raise ValueError("'order' must be either an integer, 'even' or 'odd'.")
    # Compute the relevant content.
    relevant_contents = {frozenset(content(dist, group)) for group in list_of_rv_groups}
    # Initialise a large set.
    intersection_set = relevant_contents.pop()
    # Loop over an intersection.
    for content_set in relevant_contents:
        intersection_set = intersection_set.intersection(content_set)
    # Take the intersection_set and find the relevant atoms.
    filtered_atoms = get_n_atoms(set(intersection_set), order)
    # Compute the upper set of these atoms.
    set_to_measure = upper_set(dist, filtered_atoms)
    # Measure the set.
    shared_information = measure(dist, set_to_measure, log_base)
    # Return this.
    return shared_information
