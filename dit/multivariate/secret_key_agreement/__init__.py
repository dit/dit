"""
Various bounds on secret key agreement rates.
"""


from .intrinsic_mutual_informations import (intrinsic_total_correlation,
                                            intrinsic_dual_total_correlation,
                                            intrinsic_caekl_mutual_information,
                                            )
from .minimal_intrinsic_mutual_informations import (minimal_intrinsic_total_correlation,
                                                    minimal_intrinsic_dual_total_correlation,
                                                    minimal_intrinsic_CAEKL_mutual_information,
                                                    )
from .skar_lower_bounds import (necessary_intrinsic_mutual_information,
                                secrecy_capacity,
                                )
from .reduced_intrinsic_mutual_informations import (reduced_intrinsic_total_correlation,
                                                    reduced_intrinsic_dual_total_correlation,
                                                    reduced_intrinsic_CAEKL_mutual_information,
                                                    )
from .trivial_bounds import (lower_intrinsic_mutual_information,
                             upper_intrinsic_total_correlation,
                             upper_intrinsic_dual_total_correlation,
                             upper_intrinsic_caekl_mutual_information,
                             )

intrinsic_mutual_information = intrinsic_total_correlation
reduced_intrinsic_mutual_information = reduced_intrinsic_total_correlation
minimal_intrinsic_mutual_information = minimal_intrinsic_total_correlation
upper_intrinsic_mutual_information = upper_intrinsic_total_correlation
