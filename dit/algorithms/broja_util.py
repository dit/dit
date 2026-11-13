"""Shared helpers for BROJA distribution optimizers."""


def optimized_pmf(opt):
    """
    The renormalized joint pmf, as an ndarray, of a solved distribution optimizer.

    Small entries are kept, since they can be genuine: an 8e-7 entry carries about
    p log(1/p) = 2e-5 bits, more than the tolerance of the BROJA guard.
    """
    # construct_vector returns the optimizer's own buffer
    pmf = opt.construct_vector(opt._optima).copy()
    pmf[pmf < 0] = 0  # round-off
    pmf /= pmf.sum()
    return pmf.reshape(opt._shape)
