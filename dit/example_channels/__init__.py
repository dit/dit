"""
A catalog of canonical discrete memoryless channels.

Each constructor returns a conditional :class:`~dit.Distribution` ``p(Y | X)`` --
the representation consumed by :func:`dit.algorithms.channel_capacity` and the
channel-coding evaluation layer in :mod:`dit.coding`. Alphabets are integer-valued;
an erasure is the integer just past the input alphabet (binary erasure ``2``,
q-ary erasure ``q``).
"""

from .binary import (
    binary_asymmetric_channel,
    binary_erasure_channel,
    binary_symmetric_channel,
    binary_symmetric_erasure_channel,
    z_channel,
)
from .qary import (
    noisy_typewriter,
    q_ary_erasure_channel,
    q_ary_symmetric_channel,
)
from .trivial import identity_channel, useless_channel

__all__ = (
    "binary_asymmetric_channel",
    "binary_erasure_channel",
    "binary_symmetric_channel",
    "binary_symmetric_erasure_channel",
    "identity_channel",
    "noisy_typewriter",
    "q_ary_erasure_channel",
    "q_ary_symmetric_channel",
    "useless_channel",
    "z_channel",
)
