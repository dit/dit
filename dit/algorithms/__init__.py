"""
Various algorithms related to manipulating or measuring properties of
distributions.
"""

from .channelcapacity import channel_capacity, channel_capacity_joint
from .convex_maximization import *
from .ipf import *
from .lattice import insert_join, insert_meet
from .maxentropy import *
from .maxentropyfw import *
from .minimal_sufficient_statistic import *
from .optimization import *
from .distribution_optimizers import *
from .mixture_of_products import fit_mixture_of_products, mixture_of_products_dists
from .marginal_lifts import fit_marginal_lift_mixture, lift_marginal, marginal_lift_dists
from .mprojection import (
    m_projection,
    m_projection_eps_limit,
    m_projection_from_subsets,
    mflat_design_matrix,
    mflat_mprojection_dists,
    mflat_subsets_from_dependency,
    symmetric_smooth,
)
from .prune_expand import expanded_samplespace, pruned_samplespace
from .stats import (
    cdf,
    central_moment,
    correlation,
    covariance,
    expectation,
    iqr,
    kurtosis,
    maximum,
    mean,
    median,
    minimum,
    mode,
    percentile,
    quantile,
    range_,
    skewness,
    standard_deviation,
    standard_moment,
    variance,
)

# Don't expose anything yet.
# from . import pid_broja
