"""
Methods for calculating the measures relating to the logarithmic decomposition (LD).
"""

# Import numpy.
import numpy as np


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
    dist : dit.distribution
        The distribution to be analysed.
    events : list
        A list of events over which the entropy loss will be calculated when merged.

    Returns
    -------
    entropy : float
        The entropy lost when merging each of the events in the distribution.
    """
    # Get the total probability of all of the events.
    new_probability = dist.event_probability(events)
    # Calculate the new entropy of this event.
    new_entropy = new_probability * np.emath.logn(log_base, 1.0/new_probability)
    # Calculate the old entropy of these events.
    old_entropy = sum([ prob * np.emath.logn(log_base, 1.0/prob) for prob in \
                       [dist.event_probability([event]) for event in events] ])
    # Calculate the entropy lost.
    entropy_loss = old_entropy - new_entropy
    return entropy_loss