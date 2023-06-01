"""
Tools for examining the set-theoretical contents of variables in logarithmic decomposition.
For more information see the preprint:
https://arxiv.org/abs/2305.07554
"""
# Import some things.
import more_itertools
from .measures import interior_loss
from ..npdist import Distribution

# Specify all functions defined in this module.
__all__=[
    'content',
    'get_measures',
    'measure'
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
    # Check that the variables have been names.
    if dist._rvs is None:
        raise ValueError("Variables in dist do not have names. Please assign names and try again.")
    # Check some instance types.
    if not isinstance(dist, Distribution):
        raise TypeError("'dist' must be a dit distribution.")
    elif not (rvs is None or isinstance(rvs, list)):
        raise TypeError("'rvs' must be a list of random variables or empty.")
    # If no rvs are specified, set the list of rvs to be *all* of the rvs.
    if rvs is None:
        rvs = list(dist._rvs.keys())
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

def get_measures(dist, atoms, log_base = 2):
    """
    Create a dictionary containing the associated entropy for a collection of given atoms.

    Parameters
    ----------
    dist : dit.Distribution
        The distribution to be analysed.
    atoms : set, list, tuple
        A set or list of atoms to measure (which are given as tuples of strings), or
        a single tuple representing an atom to be measured.

    Returns
    -------
    atom_measures : dict
        A dictionary returning the entropy associated to each atom given.

    Notes
    -----
    For more information, see the logarithmic decomposition preprint:
    https://arxiv.org/abs/2305.07554
    """
    # Check some instance types.
    if not isinstance(dist, Distribution):
        raise TypeError("'dist' must be a dit distribution.")
    elif not isinstance(atoms, (set, list, tuple)):
        raise TypeError("'atoms' must be a set or list of atoms, or a tuple for a single atom.")
    # Initialise an empty dictionary.
    atom_measures = {}
    # Check the type. If a list or set, then add several to the dictionary.
    if isinstance(atoms, (set, list)):
        # For each atom,
        for atom in atoms:
            # Compute the measure and save it.
            atom_measures[atom] = interior_loss(dist, atom, log_base)
    # If it's just a tuple,
    elif isinstance(atoms, tuple):
        # Then just record the one entry.
        atom_measures[atoms] = interior_loss(dist, atoms, log_base)
    else:
        raise TypeError("""Could not parse atoms. Expected set, list or tuple,
          received type """ + str(type(atoms)) + ".")

    return atom_measures

def measure(dist, atoms, log_base = 2):
    """
    Gives the total measure of a collection of atoms 'atoms' in a distribution 'dist'.

    Parameters
    ----------
    dist : dit.Distribution
        The distribution being analysed.
    atoms : set, list
        A collection of atoms to measure.
    log_base : int, float
        The base of the logarithm in the entropy calculation.

    Returns
    -------
    atomic_entropy : float
        The total entropy associated to all of the atoms in 'atoms'.

    Notes
    -----
    More detail on the interior entropy loss can be found on the arxiv preprint:
    https://arxiv.org/abs/2305.07554
    """
    # Get the measures of all of the atoms.
    measure_dictionary = get_measures(dist, atoms, log_base)
    # Sum up the values.
    atomic_entropy = sum(measure_dictionary.values())

    return atomic_entropy
