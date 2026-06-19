"""
Tests for the example-channel construction helper and parameter validation.
"""

import pytest

from dit.example_channels import binary_asymmetric_channel, z_channel
from dit.example_channels._util import conditional_from_matrix
from dit.exceptions import ditException


def test_conditional_from_matrix_roundtrips():
    """A valid transition matrix produces a conditional distribution."""
    channel = conditional_from_matrix([[0.9, 0.1], [0.2, 0.8]], [0, 1], [0, 1])
    assert channel.is_conditional()


def test_wrong_number_of_rows_raises():
    """A transition matrix needs one row per input symbol."""
    with pytest.raises(ditException):
        conditional_from_matrix([[1.0, 0.0]], [0, 1], [0, 1])


def test_wrong_number_of_columns_raises():
    """A transition matrix needs one column per output symbol."""
    with pytest.raises(ditException):
        conditional_from_matrix([[1.0], [1.0]], [0, 1], [0, 1])


def test_row_not_normalized_raises():
    """Each row of the transition matrix must sum to one."""
    with pytest.raises(ditException):
        conditional_from_matrix([[0.5, 0.4], [0.2, 0.8]], [0, 1], [0, 1])


def test_z_channel_invalid_parameter_raises():
    """The Z-channel rejects an out-of-range probability."""
    with pytest.raises(ditException):
        z_channel(1.5)


def test_binary_asymmetric_invalid_parameter_raises():
    """The binary asymmetric channel rejects out-of-range probabilities."""
    with pytest.raises(ditException):
        binary_asymmetric_channel(0.5, 1.5)


def test_binary_erasure_invalid_parameter_raises():
    """The binary erasure channel rejects an out-of-range probability."""
    from dit.example_channels import binary_erasure_channel

    with pytest.raises(ditException):
        binary_erasure_channel(1.5)


def test_qary_invalid_parameters_raise():
    """The q-ary channels reject bad alphabet sizes and probabilities."""
    from dit.example_channels import q_ary_erasure_channel, q_ary_symmetric_channel

    with pytest.raises(ditException):
        q_ary_symmetric_channel(4, 1.5)  # p out of range
    with pytest.raises(ditException):
        q_ary_erasure_channel(1, 0.1)  # q too small
