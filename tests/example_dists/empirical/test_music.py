"""
Tests for dit.example_dists.empirical._music.

music21 is an optional dependency, so ``_voice_sequence`` is exercised with small
duck-typed stand-ins for music21 parts, notes, chords and rests.
"""

import sys
from fractions import Fraction
from types import ModuleType, SimpleNamespace

import pytest

from dit.example_dists.empirical._music import _PITCH_CLASSES, _require_music21, _voice_sequence


def _note(pitch_class, offset, length):
    """
    A music21-like Note of ``pitch_class`` sounding from ``offset`` for ``length`` quarters.
    """
    pitch = SimpleNamespace(pitchClass=pitch_class)
    return SimpleNamespace(isNote=True, isChord=False, pitch=pitch, offset=offset, quarterLength=length)


def _chord(pitch_classes, offset, length):
    """
    A music21-like Chord whose pitches are listed in the given order.
    """
    pitches = [SimpleNamespace(pitchClass=pc) for pc in pitch_classes]
    return SimpleNamespace(isNote=False, isChord=True, pitches=pitches, offset=offset, quarterLength=length)


def _rest(offset, length):
    """
    A music21-like Rest.
    """
    return SimpleNamespace(isNote=False, isChord=False, offset=offset, quarterLength=length)


def _part(*elements):
    """
    A music21-like Part whose flattened stream holds ``elements``.
    """
    flat = SimpleNamespace(notesAndRests=list(elements))
    return SimpleNamespace(flatten=lambda: flat)


def test_sustained_note_fills_every_slot():
    """
    A held note sounds on every grid slot it spans, not only at its onset.

    Regression: recording the in-between slots of held notes as rests flipped the
    sign of the Bach chorales' O-information.
    """
    part = _part(_note(0, 0.0, 2.0), _note(7, 2.0, 1.0))
    assert _voice_sequence(part, 0.5, 3.0) == ["C", "C", "C", "C", "G", "G"]
    assert _voice_sequence(_part(_note(9, 0, 4)), 0.25, 4.0) == ["A"] * 16


def test_silent_slots_and_rests_become_rests():
    """
    Explicit rests, gaps between notes and the tail after the last note are "R".
    """
    part = _part(_note(4, 0, 1), _rest(1, 1), _note(11, 3, 1))
    assert _voice_sequence(part, 1.0, 5.0) == ["E", "R", "R", "B", "R"]
    assert _voice_sequence(_part(), 1.0, 3.0) == ["R", "R", "R"]


def test_zero_duration_elements_are_ignored():
    """
    Zero-length elements (e.g. grace notes) neither interrupt a held note nor fill
    a silent slot.
    """
    part = _part(_note(0, 0, 2), _note(2, 1, 0), _chord([5, 9], 2, 0.0))
    assert _voice_sequence(part, 1.0, 3.0) == ["C", "C", "R"]


def test_chord_uses_its_first_pitch():
    """
    A chord is represented by its first listed pitch, not its lowest or highest.
    """
    part = _part(_chord([7, 0, 4], 0, 1), _chord([2, 11], 1, 1))
    assert _voice_sequence(part, 1.0, 2.0) == ["G", "D"]


def test_pitch_classes_use_sharp_spellings():
    """
    Pitch classes 0-11 map onto the sharp spellings C, C#, ..., B.
    """
    expected = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    part = _part(*(_note(pc, pc, 1) for pc in range(12)))
    assert _voice_sequence(part, 1.0, 12.0) == expected
    assert list(_PITCH_CLASSES) == expected


def test_fractional_offsets_and_lengths():
    """
    music21's Fraction offsets and lengths (e.g. triplets) land on the matching
    grid, and a long held note still fills a non-binary grid exactly.
    """
    third = Fraction(1, 3)
    part = _part(_note(0, 0, third), _note(2, third, third), _note(4, 2 * third, third), _note(5, 1, 1))
    assert _voice_sequence(part, float(third), 2.0) == ["C", "D", "E", "F", "F", "F"]
    assert _voice_sequence(_part(_note(9, 0, 64)), float(third), 64.0) == ["A"] * 192


def test_sampling_stops_at_end():
    """
    There is one entry per grid slot before ``end``; later material is dropped and
    silence after the voice ends is padded with rests.
    """
    part = _part(_note(0, 0, 4))
    assert _voice_sequence(part, 1.0, 2.0) == ["C", "C"]
    assert _voice_sequence(part, 0.5, 6.0) == ["C"] * 8 + ["R"] * 4


def test_require_music21_returns_module(monkeypatch):
    """
    When music21 is importable, the helper returns that module.
    """
    fake_music21 = ModuleType("music21")
    monkeypatch.setitem(sys.modules, "music21", fake_music21)
    assert _require_music21() is fake_music21


def test_require_music21_missing(monkeypatch):
    """
    Without music21, a helpful ImportError points at the ``music`` extra.
    """
    monkeypatch.setitem(sys.modules, "music21", None)
    with pytest.raises(ImportError, match=r"require music21; install it with `pip install dit\[music\]`"):
        _require_music21()
