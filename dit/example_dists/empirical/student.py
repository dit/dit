"""
The UCI Student Performance dataset.

An empirical joint relating the social and behavioral habits of Portuguese
secondary-school students to their academic success. It is a compact
illustration of the opposite lesson to the Titanic distribution: here the
lifestyle attributes are only weakly informative about the outcome. Home
internet access, being in a romantic relationship, and weekend alcohol
consumption jointly carry only a few hundredths of a bit about whether a
student passes, so the marginal pass rate (~85%) is barely sharpened by
conditioning -- a realistic reminder that real-world social confounders are
often much softer predictors than a designed example would suggest.
"""

import csv
import io
import urllib.request
import zipfile

from ...distribution import Distribution

__all__ = ("student",)


# The canonical UCI ``student.zip`` archive, containing ``student-mat.csv`` and
# ``student-por.csv`` (semicolon-delimited, quoted string fields). The
# Portuguese-language course file (649 students) is used, as it realizes all
# forty combinations.
_STUDENT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
_STUDENT_FILE = "student-por.csv"

# Final grades (``G3``, on a 0-20 scale) at or above this threshold pass.
_PASS_GRADE = 10

_YESNO = {"yes": "Yes", "no": "No"}
_WALC = {"1": "VeryLow", "2": "Low", "3": "Medium", "4": "High", "5": "VeryHigh"}


def student():
    """
    The empirical joint distribution of the UCI Student Performance dataset.

    The four random variables are, in order:

    * ``Internet``       -- home internet access: ``Yes`` or ``No``.
    * ``Romantic``       -- in a romantic relationship: ``Yes`` or ``No``.
    * ``WeekendAlcohol`` -- weekend alcohol consumption: ``VeryLow``, ``Low``,
      ``Medium``, ``High``, or ``VeryHigh``.
    * ``Grade``          -- ``Pass`` if the final grade ``G3`` >= 10 (on the
      0-20 scale), else ``Fail``.

    The distribution is estimated from all 649 students in the Portuguese
    course file (``student-por.csv``); all forty
    ``2 x 2 x 5 x 2`` combinations occur. Only these four of the thirty-three
    columns are used. The source archive is fetched from the UCI Machine
    Learning Repository at call time.

    Returns
    -------
    d : Distribution
        The empirical joint distribution over
        ``(Internet, Romantic, WeekendAlcohol, Grade)``.

    Raises
    ------
    RuntimeError
        If the source data cannot be fetched.
    """
    try:
        with urllib.request.urlopen(_STUDENT_URL) as response:
            archive = response.read()
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Could not fetch the Student Performance dataset from {_STUDENT_URL}: {e}") from e

    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        text = zf.read(_STUDENT_FILE).decode("utf-8")

    counts = {}
    total = 0
    for row in csv.DictReader(io.StringIO(text), delimiter=";"):
        outcome = (
            _YESNO[row["internet"]],
            _YESNO[row["romantic"]],
            _WALC[row["Walc"]],
            "Pass" if int(row["G3"]) >= _PASS_GRADE else "Fail",
        )
        counts[outcome] = counts.get(outcome, 0) + 1
        total += 1

    outcomes = sorted(counts)
    pmf = [counts[o] / total for o in outcomes]
    d = Distribution(outcomes, pmf)
    d.set_rv_names(("Internet", "Romantic", "WeekendAlcohol", "Grade"))
    return d
