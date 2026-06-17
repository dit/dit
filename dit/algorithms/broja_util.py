"""Shared helpers for BROJA distribution optimizers."""


def optimized_pmf(opt, cutoff=1e-6):
    """
    Cutoff + renormalized joint pmf ndarray of a solved distribution optimizer.
    """
    pmf = opt.construct_vector(opt._optima.copy())
    pmf[pmf < cutoff] = 0
    pmf /= pmf.sum()
    return pmf.reshape(opt._shape)
