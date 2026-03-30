"""
Channel comparison: orderings and deficiencies.

This module implements the Blackwell (output-degraded), input-degraded,
less noisy, more capable, and Shannon inclusion preorders on channels,
together with Le Cam deficiencies and KL deficiencies that quantify
deviations from these orderings.
"""

from .deficiency import (
    le_cam_deficiency,
    le_cam_distance,
    output_kl_deficiency,
    weighted_input_kl_deficiency,
    weighted_le_cam_deficiency,
    weighted_output_kl_deficiency,
    weighted_output_kl_deficiency_joint,
)
from .orderings import (
    blackwell_order_joint,
    is_blackwell_sufficient,
    is_input_degraded,
    is_less_noisy,
    is_more_capable,
    is_output_degraded,
    is_shannon_included,
)
