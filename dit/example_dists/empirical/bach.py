"""
The chorales of Johann Sebastian Bach.

Following Rosas et al. (2019), each of Bach's four-part chorales is modeled as a
joint distribution over its four voices -- Soprano, Alto, Tenor, and Bass -- with
each voice taking one of the twelve pitch classes (or a rest). Every chorale is
transposed to C so that pieces in different keys share one alphabet, and the
joint is estimated by the empirical frequency of the four notes sounding
together. It is a compact illustration of statistical *synergy*: the four voices
are only weakly dependent pairwise, yet strongly coordinated as a whole, so the
O-information is negative -- the hallmark of a synergy-dominated system, which the
authors read as an intrinsic feature of Bach's counterpoint.

The scores ship with the optional :mod:`music21` dependency, so this constructor
needs no network access (unlike the other empirical distributions).
"""

from ...distribution import Distribution
from ._music import _require_music21, _sample_score

__all__ = ("bach",)

# The Bach chorales are four-part harmonizations for these voices, low to high.
_VOICES = ("Soprano", "Alto", "Tenor", "Bass")


def bach(limit=None):
    """
    The empirical joint distribution of the notes in Bach's chorales.

    The four random variables are the four voices, in order: ``Soprano``,
    ``Alto``, ``Tenor``, and ``Bass``. Each takes one of the twelve pitch classes
    (``C``, ``C#``, ..., ``B``) or ``R`` (rest).

    The distribution is estimated over the four-part Major-mode chorales in the
    :mod:`music21` core corpus (351 of the 371 Riemenschneider chorales), each
    transposed to C major and sampled on the grid of its smallest note value. The
    O-information of the result is negative: the chorales are synergy-dominated.

    Parameters
    ----------
    limit : int, None
        If given, only the first ``limit`` usable chorales are included. This is
        primarily useful for keeping tests fast; the default (``None``) uses all
        of them.

    Returns
    -------
    d : Distribution
        The empirical joint distribution over ``(Soprano, Alto, Tenor, Bass)``.

    Raises
    ------
    ImportError
        If :mod:`music21` is not installed.
    """
    music21 = _require_music21()

    counts = {}
    total = 0
    used = 0
    for score in music21.corpus.chorales.Iterator():
        if limit is not None and used >= limit:
            break
        chords = _sample_score(score, len(_VOICES))
        if chords is None:
            continue
        used += 1
        for chord in chords:
            counts[chord] = counts.get(chord, 0) + 1
            total += 1

    outcomes = sorted(counts)
    pmf = [counts[o] / total for o in outcomes]
    d = Distribution(outcomes, pmf)
    d.set_rv_names(_VOICES)
    return d
