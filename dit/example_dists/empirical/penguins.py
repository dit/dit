"""
The Palmer Penguins dataset.

A modern ecological alternative to the classic Iris dataset, recording three
penguin species observed across three islands of the Palmer Archipelago,
Antarctica. It is a compact illustration of how geography and biology become
completely entangled: each species occupies only a subset of the islands, so
many of the eighteen possible species-island-sex combinations (such as a Gentoo
penguin on Torgersen island) never occur -- their joint probability is exactly
zero. Only ten of the eighteen combinations are realized, while sex is
essentially independent of both species and island.
"""

import csv
import io
import urllib.request

from ...distribution import Distribution

__all__ = ("penguins",)


# The canonical seaborn ``penguins.csv``: one row per observed penguin with
# ``species``, ``island``, ``sex`` and four continuous morphometric columns.
_PENGUINS_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"

_SEX = {"MALE": "M", "FEMALE": "F"}


def penguins():
    """
    The empirical joint distribution of the Palmer Penguins dataset.

    The three random variables are, in order:

    * ``Species`` -- ``Adelie``, ``Chinstrap``, or ``Gentoo``.
    * ``Island``  -- ``Biscoe``, ``Dream``, or ``Torgersen``.
    * ``Sex``     -- ``M`` or ``F``.

    The distribution is estimated from the 333 penguins with a recorded sex (11
    of the 344 records have no recorded sex and are dropped). Only the three
    categorical columns are used; the four continuous morphometric measurements
    are ignored. The source data is fetched from ``mwaskom/seaborn-data`` at call
    time.

    Returns
    -------
    d : Distribution
        The empirical joint distribution over ``(Species, Island, Sex)``.

    Raises
    ------
    RuntimeError
        If the source data cannot be fetched.
    """
    try:
        with urllib.request.urlopen(_PENGUINS_URL) as response:
            text = response.read().decode("utf-8")
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Could not fetch the Palmer Penguins dataset from {_PENGUINS_URL}: {e}") from e

    counts = {}
    total = 0
    for row in csv.DictReader(io.StringIO(text)):
        sex = row["sex"].strip()
        if sex not in _SEX:
            # No recorded sex (blank or ``NA``); cannot classify.
            continue
        outcome = (row["species"], row["island"], _SEX[sex])
        counts[outcome] = counts.get(outcome, 0) + 1
        total += 1

    outcomes = sorted(counts)
    pmf = [counts[o] / total for o in outcomes]
    d = Distribution(outcomes, pmf)
    d.set_rv_names(("Species", "Island", "Sex"))
    return d
