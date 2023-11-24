"""
The logarithmic decomposition intersection measure, I_logdec.
For more information, see the preprint:
https://arxiv.org/abs/2305.07554
"""

from ..pid import BasePID
from ...log_decomp.intersection import shared

__all__ = [
    'PID_LogDec'
]

class PID_LogDec(BasePID):
    """
    A partial information decomposition built on the Logarithmic
    Decomposition framework (Down and Mediano 2023).
    """

    _name = "I_LogDec"

    @staticmethod
    def _measure(d, sources, target, order = 2):
        """
        I_LogDec measures the entropy associated to 2-atom upper sets by default.
        
        Parameters
        ----------
        d : Distribution
            The distribution for which I_LogDec is computed.
        sources : iterable of iterables
            The source variables.
        target : iterable
            The target variable.
        order : int, string
            The order of generators. Default is 2. If set to 'even' then all even
            atoms are taken as generators. If 'odd', then all odd atoms are
            generators.

        Returns
        -------
        ilogdec : float
            The value of I_LogDec for the given variables.
        """
        # Format appropriately for log_decomp.shared.
        source_list = [tuple([list(x) for x in sources])]
        return shared(d, [*source_list[0], list(target)], order)
