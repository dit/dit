"""
Tests for dit.example_dists.genetic_code
"""

from collections import Counter

import pytest

from dit.example_dists import genetic_code
from dit.multivariate import coinformation as I
from dit.multivariate import entropy as H

_H_AMINO = 4.218139062229566


@pytest.fixture(scope="module", params=[False, True], ids=["dna", "rna"])
def d(request):
    """
    The codon-form genetic-code distribution (DNA and RNA).
    """
    return genetic_code(rna=request.param)


@pytest.fixture(scope="module", params=[False, True], ids=["dna", "rna"])
def d_bases(request):
    """
    The per-base genetic-code distribution (DNA and RNA).
    """
    return genetic_code(rna=request.param, bases=True)


def test_normalized(d):
    """
    The pmf sums to one.
    """
    assert sum(d.pmf) == pytest.approx(1.0)


def test_rv_names(d):
    """
    The two random variables are named.
    """
    assert d.get_rv_names() == ("Codon", "AminoAcid")


def test_support(d):
    """
    All 64 codons appear exactly once.
    """
    assert len(d.outcomes) == 64
    codons = {outcome[0] for outcome in d.outcomes}
    assert len(codons) == 64


def test_amino_acid_alphabet(d):
    """
    Twenty amino acids plus Stop, with the known synonymous-family sizes.
    """
    counts = Counter(aa for _, aa in d.outcomes)
    assert len(counts) == 21
    assert counts["Stop"] == 3
    assert counts["Met"] == 1
    assert counts["Trp"] == 1
    assert counts["Leu"] == 6
    assert counts["Ser"] == 6
    assert counts["Arg"] == 6


def test_known_codons(d):
    """
    Spot-check a few canonical codon assignments in the active alphabet.
    """
    mapping = dict(d.outcomes)
    thymine = "U" if any("U" in codon for codon in mapping) else "T"
    assert mapping[f"A{thymine}G"] == "Met"
    assert mapping[f"{thymine}GG"] == "Trp"
    assert mapping[f"{thymine}AA"] == "Stop"
    assert mapping[f"{thymine}AG"] == "Stop"
    assert mapping[f"{thymine}GA"] == "Stop"
    assert mapping[f"{thymine}{thymine}{thymine}"] == "Phe"


def test_alphabet(d):
    """
    DNA uses T (no U); RNA uses U (no T).
    """
    letters = {base for codon, _ in d.outcomes for base in codon}
    if "U" in letters:
        assert "T" not in letters
        assert letters == {"U", "C", "A", "G"}
    else:
        assert letters == {"T", "C", "A", "G"}


def test_deterministic_translation(d):
    """
    AminoAcid is a deterministic function of Codon.
    """
    assert H(d, ["AminoAcid"], ["Codon"]) == pytest.approx(0.0, abs=1e-12)


def test_entropies(d):
    """
    Uniform codon prior: 6 bits of codon entropy, ~4.22 bits of amino-acid
    entropy, equal to the mutual information.
    """
    assert H(d, ["Codon"]) == pytest.approx(6.0)
    assert H(d, ["AminoAcid"]) == pytest.approx(_H_AMINO, abs=1e-12)
    assert H(d) == pytest.approx(6.0)
    assert I(d, [["Codon"], ["AminoAcid"]]) == pytest.approx(_H_AMINO, abs=1e-12)


def test_dna_rna_correspondence():
    """
    RNA is DNA with T replaced by U; amino-acid labels are unchanged.
    """
    dna = dict(genetic_code().outcomes)
    rna = dict(genetic_code(rna=True).outcomes)
    assert {codon.replace("T", "U"): aa for codon, aa in dna.items()} == rna


def test_bases_normalized(d_bases):
    """
    The per-base pmf sums to one.
    """
    assert sum(d_bases.pmf) == pytest.approx(1.0)


def test_bases_rv_names(d_bases):
    """
    The four random variables are named.
    """
    assert d_bases.get_rv_names() == ("Base1", "Base2", "Base3", "AminoAcid")


def test_bases_support(d_bases):
    """
    All 64 base triplets appear exactly once.
    """
    assert len(d_bases.outcomes) == 64
    triplets = {(b1, b2, b3) for b1, b2, b3, _ in d_bases.outcomes}
    assert len(triplets) == 64


def test_bases_known_codons(d_bases):
    """
    Spot-check a few canonical assignments in the per-base form.
    """
    mapping = {(b1, b2, b3): aa for b1, b2, b3, aa in d_bases.outcomes}
    thymine = "U" if any(b1 == "U" for b1, _, _, _ in d_bases.outcomes) else "T"
    assert mapping[("A", thymine, "G")] == "Met"
    assert mapping[(thymine, "G", "G")] == "Trp"
    assert mapping[(thymine, "A", "A")] == "Stop"


def test_bases_deterministic_translation(d_bases):
    """
    AminoAcid is a deterministic function of the three bases.
    """
    assert H(d_bases, ["AminoAcid"], ["Base1", "Base2", "Base3"]) == pytest.approx(0.0, abs=1e-12)


def test_bases_entropies(d_bases):
    """
    Uniform bases: 2 bits each, 6 bits jointly, ~4.22 bits of amino-acid entropy.
    """
    assert H(d_bases, ["Base1"]) == pytest.approx(2.0)
    assert H(d_bases, ["Base2"]) == pytest.approx(2.0)
    assert H(d_bases, ["Base3"]) == pytest.approx(2.0)
    assert H(d_bases, ["Base1", "Base2", "Base3"]) == pytest.approx(6.0)
    assert H(d_bases, ["AminoAcid"]) == pytest.approx(_H_AMINO, abs=1e-12)
    assert H(d_bases) == pytest.approx(6.0)
    assert I(d_bases, [["Base1", "Base2", "Base3"], ["AminoAcid"]]) == pytest.approx(_H_AMINO, abs=1e-12)


def test_bases_wobble(d_bases):
    """
    The third ("wobble") position carries far less amino-acid information than
    the first two.
    """
    i1 = I(d_bases, [["Base1"], ["AminoAcid"]])
    i2 = I(d_bases, [["Base2"], ["AminoAcid"]])
    i3 = I(d_bases, [["Base3"], ["AminoAcid"]])
    assert i1 == pytest.approx(1.7417292966721742, abs=1e-12)
    assert i2 == pytest.approx(1.8708646483360871, abs=1e-12)
    assert i3 == pytest.approx(0.43688906222956625, abs=1e-12)
    assert i3 < i1 < i2


def test_bases_matches_codon_form():
    """
    Expanding DNA/RNA codon outcomes into bases recovers the bases=True joint.
    """
    for rna in (False, True):
        codon = genetic_code(rna=rna)
        bases = genetic_code(rna=rna, bases=True)
        expanded = {(*c, aa): p for (c, aa), p in zip(codon.outcomes, codon.pmf, strict=True)}
        assert dict(zip(bases.outcomes, bases.pmf, strict=True)) == expanded
