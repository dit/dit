"""
Shared helpers for building example channels.
"""

from math import isclose

from ..exceptions import ditException

__all__ = ()


def conditional_from_matrix(P, inputs, outputs, input_name="X", output_name="Y"):
    """
    Build a conditional ``Distribution`` ``p(Y | X)`` from a transition matrix.

    Parameters
    ----------
    P : sequence of sequence of float
        The transition matrix, with ``P[i][j] = p(Y = outputs[j] | X = inputs[i])``.
        Each row must sum to one.
    inputs : sequence
        The input alphabet (one entry per row of ``P``).
    outputs : sequence
        The output alphabet (one entry per column of ``P``).
    input_name : str
        The name of the input random variable.
    output_name : str
        The name of the output random variable.

    Returns
    -------
    channel : Distribution
        The conditional distribution ``p(Y | X)``.
    """
    from ..distribution import Distribution

    if len(P) != len(inputs):
        raise ditException("P must have one row per input symbol.")
    n_in = len(inputs)
    joint = {}
    for i, x in enumerate(inputs):
        row = P[i]
        if len(row) != len(outputs):
            raise ditException("P must have one column per output symbol.")
        if not isclose(sum(row), 1.0, abs_tol=1e-9):
            raise ditException(f"Row {i} of the transition matrix does not sum to one.")
        for y, prob in zip(outputs, row, strict=True):
            if prob > 0:
                joint[(x, y)] = prob / n_in

    dist = Distribution(joint, rv_names=[input_name, output_name])
    return dist.condition_on(input_name)
