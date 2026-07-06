"""
Shared helpers for the music-derived empirical distributions.

Both :func:`dit.example_dists.empirical.bach` and
:func:`dit.example_dists.empirical.corelli` model a polyphonic score as a joint
distribution over its voices, following Rosas et al. (2019). Each voice becomes a
random variable whose alphabet is the twelve pitch classes plus a rest, and the
joint is estimated by the empirical frequency of the simultaneously sounding
notes (the "chords"). Scores are transposed to C so that pieces in different keys
share one alphabet.

The score parsing is delegated to :mod:`music21`, which is an optional
dependency; it is imported lazily so that importing ``dit`` never requires it.
"""

__all__ = ()

# The twelve pitch classes, spelled with sharps, plus a rest symbol.
_PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_REST = "R"


def _require_music21():
    """
    Import and return :mod:`music21`, or raise a helpful error if it is absent.

    Returns
    -------
    music21 : module
        The imported ``music21`` package.

    Raises
    ------
    ImportError
        If ``music21`` is not installed.
    """
    try:
        import music21
    except ImportError as e:  # pragma: no cover
        raise ImportError("The music distributions require music21; install it with `pip install dit[music]`.") from e
    return music21


def _voice_sequence(part, grid, end):
    """
    Sample a single voice onto a fixed time grid as a list of pitch classes.

    Each note is expanded across its entire duration so that a sustained note
    fills every grid slot it spans; only genuinely silent slots become rests.

    Parameters
    ----------
    part : music21.stream.Part
        The (already transposed) voice to sample.
    grid : float
        The grid spacing, in quarter-note lengths.
    end : float
        The offset (in quarter lengths) at which sampling stops.

    Returns
    -------
    seq : list of str
        One pitch-class name (or ``"R"``) per grid slot.
    """
    slots = {}
    for element in part.flatten().notesAndRests:
        duration = float(element.quarterLength)
        if duration <= 0:
            continue
        if element.isNote:
            value = _PITCH_CLASSES[element.pitch.pitchClass]
        elif element.isChord:
            value = _PITCH_CLASSES[element.pitches[0].pitchClass]
        else:
            value = _REST
        offset = float(element.offset)
        slot = offset
        while slot < offset + duration - 1e-9:
            slots[round(slot, 4)] = value
            slot += grid

    n_slots = int(round(end / grid))
    return [slots.get(round(i * grid, 4), _REST) for i in range(n_slots)]


def _sample_score(score, n_voices):
    """
    Turn a Major-mode score into a list of pitch-class chords over its voices.

    The score is skipped (``None`` is returned) unless it is in a Major key and
    has exactly ``n_voices`` parts. Otherwise it is transposed to C major and
    each part is sampled onto the grid of the smallest note value in the piece.

    Parameters
    ----------
    score : music21.stream.Score
        The parsed score.
    n_voices : int
        The required number of parts; scores with a different count are skipped.

    Returns
    -------
    chords : list of tuple, or None
        One tuple of ``n_voices`` pitch-class names per grid slot, or ``None`` if
        the score is not a Major-mode piece with ``n_voices`` parts.
    """
    music21 = _require_music21()

    key = score.analyze("key")
    if key.mode != "major":
        return None

    parts = score.parts
    if len(parts) != n_voices:
        return None

    to_c = music21.interval.Interval(key.tonic, music21.pitch.Pitch("C"))
    parts = [part.transpose(to_c) for part in parts]

    end = max(part.flatten().highestTime for part in parts)
    durations = [float(n.quarterLength) for part in parts for n in part.flatten().notesAndRests if n.quarterLength > 0]
    if not durations:
        return None
    grid = min(durations)

    sequences = [_voice_sequence(part, grid, end) for part in parts]
    return list(zip(*sequences, strict=True))
