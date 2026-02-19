"""
Logging configuration for dit.

Uses loguru as the logging backend. The "dit" logger is disabled by default;
users opt in with::

    from loguru import logger
    logger.enable("dit")
"""

from loguru import logger

__all__ = (
    'logger',
    'basic_logger',
)


def basic_logger(name, level):
    """
    Return a loguru-bound logger for backward compatibility.

    Parameters
    ----------
    name : str
        The name to bind to log messages.
    level : int, bool, None
        If ``None`` or ``False``, logging is effectively silenced.
        If ``True`` or a positive integer, logging is enabled.

    Returns
    -------
    logger : loguru.Logger
        A loguru logger instance bound to *name*.
    """
    bound = logger.bind(name=name)

    if level is False or level is None:
        bound.disable("dit")
    elif level is True:
        bound.enable("dit")
    else:
        bound.enable("dit")

    return bound
