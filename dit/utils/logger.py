import logging

def basic_logger(name, level):
    """
    Returns a basic logger, with no special string formatting.

    Parameters
    ----------
    name : str
        The name of the logger.
    level : int
        An integer, such as `logging.INFO`, indicating the desired severity
        level. As in the `logging` module, messages which are less severe
        than the verbosity level will be ignored---though, the level value of
        0 is treated specially. If `None`, then we use `logging.NOTSET` and
        the level is determined by a parent logger. This typically corresponds
        to `logging.WARNING`, which is equal to 30.

    Returns
    -------
    logger : logger
        The logger.

    """
    # To have various levels at the INFO category, use numbers greater
    # than logging.INFO == 20 but less than logging.WARNING == 30. For example,
    # logging at level LESSINFO = 25, will only cause messages at or above
    # level 25 to be seen by the user.

    if level is None:
        level = logging.NOTSET
    if level is True:
        level = logging.WARNING
    if level is False:
        level = 1000000

    logger = logging.getLogger(name)
    if len(logger.handlers) == 0:
        sh = logging.StreamHandler()
        formatter = logging.Formatter("%(message)s")
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    logger.setLevel(level)
    return logger
