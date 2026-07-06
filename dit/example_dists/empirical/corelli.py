"""
The trio sonatas of Arcangelo Corelli.

Following Rosas et al. (2019), Corelli's four-part sonatas are modeled as a joint
distribution over their four instrumental lines -- two violins, the violone (bass
strings), and the organ continuo -- with each line taking one of the twelve pitch
classes (or a rest). Every movement is transposed to C and the joint is estimated
by the empirical frequency of the four notes sounding together. It is a compact
illustration of the opposite lesson to the Bach chorales: here the O-information
is *positive*, so the ensemble is redundancy-dominated. Much of that redundancy
is carried by the low voices, which frequently double the same basso continuo
line.

The scores are the Op. 1, Op. 3, and Op. 4 sonatas from the CCARH-derived
Humdrum ``**kern`` collection, fetched at call time. Parsing requires the
optional :mod:`music21` dependency. Op. 5 (two parts) and Op. 6 (seven-part
concerti) are excluded so that the joint is over a consistent four voices.
"""

import urllib.request

from ...distribution import Distribution
from ._music import _require_music21, _sample_score

__all__ = ("corelli",)

# The GitHub mirror of the CCARH Corelli **kern scores. The trees API lists every
# file; the raw host serves the individual movements.
_TREE_URL = "https://api.github.com/repos/harshshredding/corelli/git/trees/master?recursive=1"
_RAW_BASE = "https://raw.githubusercontent.com/harshshredding/corelli/master/"

# The four-part sonatas: Op. 1, Op. 3, and Op. 4. Their movement files begin with
# these prefixes (Op. 1 is zero-padded, the others are not).
_FOUR_PART_PREFIXES = ("op01", "op3", "op4")

# The four lines of a Corelli trio sonata, in score order.
_VOICES = ("Violin1", "Violin2", "Violone", "Organo")


def _kern_paths():
    """
    List the four-part Corelli movement paths in the kern mirror.

    Returns
    -------
    paths : list of str
        Repository-relative paths of the Op. 1, 3, and 4 ``.krn`` files.
    """
    import json

    with urllib.request.urlopen(_TREE_URL) as response:
        tree = json.load(response)["tree"]

    paths = []
    for entry in tree:
        path = entry["path"]
        if not path.endswith(".krn"):
            continue
        name = path.rsplit("/", 1)[-1]
        if name.startswith(_FOUR_PART_PREFIXES):
            paths.append(path)
    return paths


def corelli(limit=None):
    """
    The empirical joint distribution of the notes in Corelli's trio sonatas.

    The four random variables are the four instrumental lines, in order:
    ``Violin1``, ``Violin2``, ``Violone``, and ``Organo``. Each takes one of the
    twelve pitch classes (``C``, ``C#``, ..., ``B``) or ``R`` (rest).

    The distribution is estimated over the four-part Major-mode movements of the
    Op. 1, Op. 3, and Op. 4 sonatas, each transposed to C major and sampled on
    the grid of its smallest note value. The source scores are fetched from the
    CCARH-derived Humdrum ``**kern`` mirror at call time. The O-information of the
    result is positive: the ensemble is redundancy-dominated.

    Parameters
    ----------
    limit : int, None
        If given, only the first ``limit`` usable movements are included. This is
        primarily useful for keeping tests fast; the default (``None``) uses all
        of them.

    Returns
    -------
    d : Distribution
        The empirical joint distribution over
        ``(Violin1, Violin2, Violone, Organo)``.

    Raises
    ------
    ImportError
        If :mod:`music21` is not installed.
    RuntimeError
        If the source data cannot be fetched.
    """
    music21 = _require_music21()

    try:
        paths = _kern_paths()
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Could not list the Corelli scores from {_TREE_URL}: {e}") from e

    counts = {}
    total = 0
    used = 0
    for path in paths:
        if limit is not None and used >= limit:
            break
        url = _RAW_BASE + path
        try:
            with urllib.request.urlopen(url) as response:
                text = response.read().decode("utf-8", errors="replace")
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Could not fetch the Corelli score from {url}: {e}") from e

        score = music21.converter.parse(text, format="humdrum")
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
