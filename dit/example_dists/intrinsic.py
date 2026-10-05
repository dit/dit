"""
Distributions useful for illustrating the behavior of the various intrinsic
measures.
"""

from ..distribution import Distribution

__all__ = (
    "intrinsic_1",
    "intrinsic_2",
    "intrinsic_3",
    "bound_information",
)


# from the intrinsic information paper
intrinsic_1 = Distribution(["000", "011", "101", "110", "222", "333"], [1 / 8] * 4 + [1 / 4] * 2)
intrinsic_1.secret_rate = 0.0


# from the reduced intrinsic information paper
intrinsic_2 = Distribution(["000", "011", "101", "110", "220", "331"], [1 / 8] * 4 + [1 / 4] * 2)
intrinsic_2.secret_rate = 1.0


# from the minimal intrinsic information paper, with alpha_1 = 1/3 and alpha_2 = 1/2
intrinsic_3 = Distribution(
    ["000", "001", "012", "013", "102", "103", "110", "111", "220", "221", "332", "333"],
    [1 / 24, 1 / 12, 1 / 24, 1 / 12, 1 / 24, 1 / 12, 1 / 24, 1 / 12, 1 / 8, 1 / 8, 1 / 8, 1 / 8],
)
intrinsic_3.secret_rate = 0.97927916037609197


# from the bipartite bound information paper; Eve's symbol 2 is the erasure
bound_information = Distribution(
    ["000", "002", "010", "011", "012", "100", "101", "102", "111", "112"],
    [5 / 36, 5 / 36, 2 / 36, 2 / 36, 4 / 36, 2 / 36, 2 / 36, 4 / 36, 5 / 36, 5 / 36],
)
bound_information.secret_rate = 0.0
