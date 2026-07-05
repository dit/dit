"""
The UCI Car Evaluation dataset.

An empirical joint relating a handful of discrete consumer-preference features to
an overall purchase recommendation. Unlike the softer social datasets, this one
is a compact illustration of deterministic logical rules embedded in a joint
distribution: the recommendation is a hierarchical expert-system function of the
inputs, so conditioning exposes hard constraints. Most starkly, whenever the
safety rating is ``Low`` the recommendation is never ``Good`` or ``VeryGood`` --
that combination has exactly zero probability, no matter how favorable the price
and maintenance costs are.
"""

import csv
import io
import urllib.request

from ...distribution import Distribution

__all__ = ("car",)


# The canonical UCI ``car.data`` file: a headerless CSV with 1728 rows spanning
# the complete grid of the six input features, plus the evaluation class.
_CAR_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"

_PRICE = {"vhigh": "VeryHigh", "high": "High", "med": "Medium", "low": "Low"}
_SAFETY = {"low": "Low", "med": "Medium", "high": "High"}
_DECISION = {
    "unacc": "Unacceptable",
    "acc": "Acceptable",
    "good": "Good",
    "vgood": "VeryGood",
}


def car():
    """
    The empirical joint distribution of the UCI Car Evaluation dataset.

    The four random variables are, in order:

    * ``Buying``      -- buying price: ``VeryHigh``, ``High``, ``Medium``,
      ``Low``.
    * ``Maintenance`` -- maintenance cost: ``VeryHigh``, ``High``, ``Medium``,
      ``Low``.
    * ``Safety``      -- estimated safety: ``Low``, ``Medium``, ``High``.
    * ``Decision``    -- overall evaluation: ``Unacceptable``, ``Acceptable``,
      ``Good``, ``VeryGood``.

    The distribution is estimated from all 1728 rows of ``car.data``; only these
    four of the seven columns are used (the two omitted inputs, number of doors
    and passenger capacity, are marginalized out). Because the evaluation is a
    deterministic hierarchical function of the inputs, only 82 of the
    ``4 x 4 x 3 x 4 = 192`` combinations occur -- in particular, ``Low`` safety
    never yields a ``Good`` or ``VeryGood`` decision. The source data is fetched
    from the UCI Machine Learning Repository at call time.

    Returns
    -------
    d : Distribution
        The empirical joint distribution over
        ``(Buying, Maintenance, Safety, Decision)``.

    Raises
    ------
    RuntimeError
        If the source data cannot be fetched.
    """
    try:
        with urllib.request.urlopen(_CAR_URL) as response:
            text = response.read().decode("utf-8")
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Could not fetch the Car Evaluation dataset from {_CAR_URL}: {e}") from e

    counts = {}
    total = 0
    # Headerless rows: buying, maint, doors, persons, lug_boot, safety, class.
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 7:
            continue
        outcome = (
            _PRICE[row[0]],
            _PRICE[row[1]],
            _SAFETY[row[5]],
            _DECISION[row[6]],
        )
        counts[outcome] = counts.get(outcome, 0) + 1
        total += 1

    outcomes = sorted(counts)
    pmf = [counts[o] / total for o in outcomes]
    d = Distribution(outcomes, pmf)
    d.set_rv_names(("Buying", "Maintenance", "Safety", "Decision"))
    return d
