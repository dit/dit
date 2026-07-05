"""
The Titanic passenger dataset.

A classic empirical dataset relating passenger socioeconomic attributes to
survival of the 1912 RMS Titanic disaster. It is a compact illustration of how
several categorical variables jointly determine an outcome: the marginal
probability of survival is low, yet conditioning on class, sex, and age exposes
the historical "women and children first" evacuation protocol, driving some
joint states near certain survival and others near certain death.
"""

import csv
import io
import urllib.request

from ...distribution import Distribution

__all__ = ("titanic",)


# The canonical Kaggle ``train.csv`` (891 passengers), hosted as a plain CSV with
# raw ``Age``/``Pclass``/``Sex``/``Survived`` columns and no library-specific
# preprocessing.
_TITANIC_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

# Passengers younger than this (in years) are binned as children.
_CHILD_AGE = 16

_CLASS = {"1": "1st", "2": "2nd", "3": "3rd"}
_SEX = {"male": "M", "female": "F"}
_SURVIVED = {"0": "No", "1": "Yes"}


def titanic():
    """
    The empirical joint distribution of the Titanic passenger dataset.

    The four random variables are, in order:

    * ``Class``    -- passenger class: ``1st``, ``2nd``, ``3rd``.
    * ``Sex``      -- ``M`` or ``F``.
    * ``Age``      -- ``Child`` if age < 16, else ``Adult``.
    * ``Survived`` -- ``Yes`` or ``No``.

    The distribution is estimated from the 714 passengers with a recorded age
    (177 of the 891 records have no age and are dropped). The source data is
    fetched from ``datasciencedojo/datasets`` at call time.

    Returns
    -------
    d : Distribution
        The empirical joint distribution over ``(Class, Sex, Age, Survived)``.

    Raises
    ------
    RuntimeError
        If the source data cannot be fetched.
    """
    try:
        with urllib.request.urlopen(_TITANIC_URL) as response:
            text = response.read().decode("utf-8")
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Could not fetch the Titanic dataset from {_TITANIC_URL}: {e}") from e

    counts = {}
    total = 0
    for row in csv.DictReader(io.StringIO(text)):
        age = row["Age"].strip()
        if not age:
            # No recorded age; cannot bin.
            continue
        outcome = (
            _CLASS[row["Pclass"]],
            _SEX[row["Sex"]],
            "Child" if float(age) < _CHILD_AGE else "Adult",
            _SURVIVED[row["Survived"]],
        )
        counts[outcome] = counts.get(outcome, 0) + 1
        total += 1

    outcomes = sorted(counts)
    pmf = [counts[o] / total for o in outcomes]
    d = Distribution(outcomes, pmf)
    d.set_rv_names(("Class", "Sex", "Age", "Survived"))
    return d
