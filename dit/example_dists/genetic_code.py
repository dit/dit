"""
The standard genetic code as a joint distribution over codons and amino acids.

The genetic code is a deterministic map from the 64 DNA nucleotide triplets
(codons) onto the 20 amino acids plus a stop signal. Under a uniform codon
prior this joint is a compact illustration of a *lossy* but highly structured
channel: :math:`H(\\mathrm{Codon}) = 6` bits while
:math:`H(\\mathrm{AminoAcid}) \\approx 4.22` bits, so roughly 1.78 bits of codon
information is erased by translation. The erasure is far from uniform --
synonymous codon families have sizes 1, 2, 3, 4, or 6 -- which makes the
distribution a natural example for mutual information, channel capacity under
codon mutation noise, and redundancy measures.

The mapping used here is NCBI translation table 1 (the standard nuclear code),
with DNA bases ``T/C/A/G`` by default (pass ``rna=True`` for the RNA alphabet
``U/C/A/G``) and three-letter amino-acid names (``Stop`` for the three stop
codons). Pass ``bases=True`` to expand each codon into three nucleotide random
variables ``Base1``, ``Base2``, ``Base3`` instead of a single ``Codon``. See,
e.g., Crick, F.H.C. *The origin of the genetic code* (J. Mol. Biol., 1968) and
Yockey, H.P. *Information Theory, Evolution, and the Origin of Life*
(Cambridge, 2005) for the biological and information-theoretic background.
"""

from ..distribution import Distribution

__all__ = ("genetic_code",)


# Standard nuclear genetic code (NCBI translation table 1), DNA alphabet.
_GENETIC_CODE = {
    "TTT": "Phe",
    "TTC": "Phe",
    "TTA": "Leu",
    "TTG": "Leu",
    "TCT": "Ser",
    "TCC": "Ser",
    "TCA": "Ser",
    "TCG": "Ser",
    "TAT": "Tyr",
    "TAC": "Tyr",
    "TAA": "Stop",
    "TAG": "Stop",
    "TGT": "Cys",
    "TGC": "Cys",
    "TGA": "Stop",
    "TGG": "Trp",
    "CTT": "Leu",
    "CTC": "Leu",
    "CTA": "Leu",
    "CTG": "Leu",
    "CCT": "Pro",
    "CCC": "Pro",
    "CCA": "Pro",
    "CCG": "Pro",
    "CAT": "His",
    "CAC": "His",
    "CAA": "Gln",
    "CAG": "Gln",
    "CGT": "Arg",
    "CGC": "Arg",
    "CGA": "Arg",
    "CGG": "Arg",
    "ATT": "Ile",
    "ATC": "Ile",
    "ATA": "Ile",
    "ATG": "Met",
    "ACT": "Thr",
    "ACC": "Thr",
    "ACA": "Thr",
    "ACG": "Thr",
    "AAT": "Asn",
    "AAC": "Asn",
    "AAA": "Lys",
    "AAG": "Lys",
    "AGT": "Ser",
    "AGC": "Ser",
    "AGA": "Arg",
    "AGG": "Arg",
    "GTT": "Val",
    "GTC": "Val",
    "GTA": "Val",
    "GTG": "Val",
    "GCT": "Ala",
    "GCC": "Ala",
    "GCA": "Ala",
    "GCG": "Ala",
    "GAT": "Asp",
    "GAC": "Asp",
    "GAA": "Glu",
    "GAG": "Glu",
    "GGT": "Gly",
    "GGC": "Gly",
    "GGA": "Gly",
    "GGG": "Gly",
}


def genetic_code(rna=False, bases=False):
    """
    The joint distribution of codons and the amino acids they code for.

    Codons are drawn uniformly from the 64 nucleotide triplets, and each codon
    is mapped to its amino acid (or ``Stop``) by the standard nuclear genetic
    code. The map is deterministic, so conditioning on the full codon (or on all
    three bases) leaves no residual amino-acid uncertainty, and the mutual
    information equals the amino-acid entropy
    (:math:`H(\\mathrm{AminoAcid}) \\approx 4.22` bits).

    By default the two random variables are, in order:

    * ``Codon``     -- one of the 64 triplets (e.g. ``ATG`` or ``AUG``).
    * ``AminoAcid`` -- one of 20 amino acids (three-letter names) or ``Stop``.

    With ``bases=True`` the codon is expanded into its three nucleotide
    positions, yielding:

    * ``Base1``, ``Base2``, ``Base3`` -- the first, second, and third bases.
    * ``AminoAcid``                   -- as above.

    The per-base form makes the classic "wobble" asymmetry visible: the third
    position carries far less information about the amino acid than the first
    two.

    Parameters
    ----------
    rna : bool
        If ``False`` (default), codons use the DNA alphabet ``{T, C, A, G}``.
        If ``True``, thymine is replaced by uracil and codons use the RNA
        alphabet ``{U, C, A, G}``.
    bases : bool
        If ``False`` (default), return the joint over ``(Codon, AminoAcid)``.
        If ``True``, expand each codon into three base random variables and
        return the joint over ``(Base1, Base2, Base3, AminoAcid)``.

    Returns
    -------
    d : Distribution
        The joint distribution over ``(Codon, AminoAcid)``, or over
        ``(Base1, Base2, Base3, AminoAcid)`` when ``bases=True``.
    """
    if rna:
        mapping = [(codon.replace("T", "U"), aa) for codon, aa in _GENETIC_CODE.items()]
    else:
        mapping = list(_GENETIC_CODE.items())

    if bases:
        outcomes = [(c[0], c[1], c[2], aa) for c, aa in mapping]
        names = ("Base1", "Base2", "Base3", "AminoAcid")
    else:
        outcomes = mapping
        names = ("Codon", "AminoAcid")

    d = Distribution(outcomes, [1 / 64] * 64)
    d.set_rv_names(names)
    return d
