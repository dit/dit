"""
Methods for calculating the measures relating to the logarithmic decomposition (LD).
"""

# Import numpy and itertools.
import numpy as np
import more_itertools
from ..npdist import Distribution

# Specify all functions defined in this module.
__all__=[
    'total_loss',
    'interior_loss'
]

def total_loss(dist, events, log_base = 2):
    """
    Compute the entropy lost when merging each of the events in 'events'
    inside of the distribution 'dist'.

    Parameters
    ----------
    dist : dit.Distribution
        The distribution to be analysed.
    events : list
        A list of events over which the entropy loss will be calculated when merged.

    Returns
    -------
    entropy_loss : float
        The entropy lost when merging each of the events in the distribution.
    """
    # Check that the inputs are correct.
    if not isinstance(events, list):
        raise TypeError("'events' must be a list.")
    elif not isinstance(dist, Distribution):
        raise TypeError("'dist' must be a dit distribution.")
    elif not isinstance(log_base, (int, float)):
        raise TypeError("'log_base' must be a float.")
    # Get the total probability of all of the events.
    new_probability = dist.event_probability(events)
    # Calculate the new entropy of this event.
    new_entropy = new_probability * np.emath.logn(log_base, 1.0/new_probability)
    # Calculate the old entropy of these events.
    old_entropy = sum([ prob * np.emath.logn(log_base, 1.0/prob) for prob in \
                       [dist.event_probability([event]) for event in events] ])
    # Calculate the entropy lost.
    entropy_loss = old_entropy - new_entropy
    # If the events was empty, then set the loss to zero.
    if events == []:
        entropy_loss = 0
    # Return the entropy loss.
    return entropy_loss

def interior_loss(dist, events, log_base = 2):
    """
    Compute the interior entropy loss (Down and Mediano 2023), L^o, associated
    to the logarithmic decomposition atom given by 'events' inside of the distribution 'dist'.

    Parameters
    ----------
    dist : dit.Distribution
        The distribution to be analysed.
    events : list, tuple
        A list of events specifying the logarithmic decomposition atom to be computed.
    log_base : int, float
        The base of the logarithm used for computing the measure.

    Returns
    -------
    interior_loss_measure : float
        The interior entropy loss associated with the logarithmic decomposition atom
        specified by 'events'.

    Notes
    -----
    More detail on the interior entropy loss can be found on the arxiv preprint:
    https://arxiv.org/abs/2305.07554

    If 1-outcome atom is required, add a trailing comma in the tuple.
    """
    # Check the inputs are the correct types.
    if not isinstance(events, (list, tuple)):
        raise TypeError("'events' must be a list or tuple. Got type " + str(type(events)) + ".")
    elif not isinstance(dist, Distribution):
        raise TypeError("'dist' must be a dit distribution.")
    elif not isinstance(log_base, (int, float)):
        raise TypeError("'log_base' must be a float.")
    # Compute the interior loss.
    interior_loss_measure = (-1)**len(events) * sum(
                            [(-1)**len(x) * total_loss(dist, list(x), log_base)
                            for x in more_itertools.powerset(events)])
    # Return the result.
    return interior_loss_measure
