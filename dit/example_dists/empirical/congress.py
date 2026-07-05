"""
The 1984 U.S. Congressional Voting Records dataset.

An empirical joint over the party affiliation of each member of the U.S. House of
Representatives and their votes on the sixteen key roll calls identified by the
Congressional Quarterly Almanac (CQA) for the 98th Congress. It is a compact
illustration of how a single categorical label (``Party``) is almost perfectly
predicted by a bundle of correlated binary features: party membership carries
just under one bit of entropy, and the sixteen votes jointly resolve essentially
all of it, while individual votes range from near-perfect party proxies (the
physician-fee-freeze vote) down to votes that split both caucuses evenly (the
water-project vote).

Following the CQA coding, each of the nine raw vote dispositions is collapsed to
three states: ``yea`` (voted / paired / announced for), ``nay`` (voted / paired /
announced against), and ``?`` (voted present, present to avoid a conflict of
interest, or did not vote). As the source documentation stresses, ``?`` is *not*
missing data -- it is a genuine third disposition -- so it is retained as its own
state rather than dropped.
"""

import csv
import io
import urllib.request

from ...distribution import Distribution

__all__ = ("congress",)


# The canonical UCI ``house-votes-84.data`` file: 435 rows of
# ``party,vote_1,...,vote_16`` with votes coded ``y``/``n``/``?``.
_CONGRESS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data"

_PARTY = {"democrat": "D", "republican": "R"}
_VOTE = {"y": "yea", "n": "nay", "?": "?"}

# The sixteen key votes, in file order, named after the issue voted on.
_ISSUES = (
    "HandicappedInfants",
    "WaterProject",
    "BudgetResolution",
    "PhysicianFeeFreeze",
    "ElSalvadorAid",
    "ReligiousGroupsInSchools",
    "AntiSatelliteBan",
    "AidToContras",
    "MXMissile",
    "Immigration",
    "Synfuels",
    "EducationSpending",
    "SuperfundRightToSue",
    "Crime",
    "DutyFreeExports",
    "SouthAfricaExportAct",
)

_RV_NAMES = ("Party",) + _ISSUES


def congress():
    """
    The empirical joint distribution of the 1984 congressional voting records.

    The seventeen random variables are, in order:

    * ``Party`` -- party affiliation: ``D`` (democrat) or ``R`` (republican).
    * one variable per key vote (sixteen in all), each taking ``yea``, ``nay``,
      or ``?`` (present / abstained / did not vote). In file order these are
      ``HandicappedInfants``, ``WaterProject``, ``BudgetResolution``,
      ``PhysicianFeeFreeze``, ``ElSalvadorAid``, ``ReligiousGroupsInSchools``,
      ``AntiSatelliteBan``, ``AidToContras``, ``MXMissile``, ``Immigration``,
      ``Synfuels``, ``EducationSpending``, ``SuperfundRightToSue``, ``Crime``,
      ``DutyFreeExports``, and ``SouthAfricaExportAct``.

    The distribution is estimated from all 435 House members (267 democrats, 168
    republicans). The source data is fetched from the UCI Machine Learning
    Repository at call time.

    Returns
    -------
    d : Distribution
        The empirical joint distribution over ``(Party, <16 votes>)``.

    Raises
    ------
    RuntimeError
        If the source data cannot be fetched.
    """
    try:
        with urllib.request.urlopen(_CONGRESS_URL) as response:
            text = response.read().decode("utf-8")
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Could not fetch the congressional voting dataset from {_CONGRESS_URL}: {e}") from e

    counts = {}
    total = 0
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        outcome = (_PARTY[row[0]],) + tuple(_VOTE[v] for v in row[1:])
        counts[outcome] = counts.get(outcome, 0) + 1
        total += 1

    outcomes = sorted(counts)
    pmf = [counts[o] / total for o in outcomes]
    d = Distribution(outcomes, pmf)
    d.set_rv_names(_RV_NAMES)
    return d
