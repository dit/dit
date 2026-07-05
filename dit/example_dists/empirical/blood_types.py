"""
Human blood group distributions structured by continent.

A compact empirical-style illustration of a *common-cause* dependency: a single
latent geographic variable (``Region``) couples together several red-cell
antigen systems that are otherwise genetically independent. Each blood group
system sits on a different chromosome, so within any one population the systems
segregate independently; the only reason knowing a person's Duffy type tells you
anything about their Kidd type is that both are informative about *where the
person's ancestry is from*.

Formally the joint is built as a naive-Bayes / common-cause model,

    P(Region, ABO, Rh, Kell, Duffy, Kidd, MNS)
        = P(Region)
          * P(ABO    | Region)
          * P(Rh     | Region)
          * P(Kell   | Region)
          * P(Duffy  | Region)
          * P(Kidd   | Region)
          * P(MNS    | Region),

so every antigen system is conditionally independent of the others given
``Region``. Marginalizing ``Region`` out recovers the "global" mixture over
blood types; conditioning on ``Region`` recovers a single continent.

This makes the distribution a natural stress test for multivariate measures: the
entire dependence among the six antigen systems is redundancy induced by the
shared cause ``Region`` (large total correlation, essentially zero conditional
total correlation given ``Region``), while any single system carries positive
mutual information about geography -- the Duffy Fy(a-b-) null phenotype most
dramatically, at roughly 80% in sub-Saharan Africa versus about 1% elsewhere.

Data provenance
---------------
The per-continent antigen frequencies are representative teaching values adapted
from standard immunohematology compilations -- principally

* Daniels, G. *Human Blood Groups*, 3rd ed. (Wiley-Blackwell, 2013), and
* Reid, M.E., Lomas-Francis, C., Olsson, M.L. *The Blood Group Antigen
  FactsBook*, 3rd ed. (Academic Press, 2012),

cross-checked against published ABO/Rh distributions by country. They are coarse
continental priors, not census-grade estimates; continent-level summaries are
especially unstable for the highly admixed Americas and Oceania. The continental
population weights used to form the global marginal are approximate 2020s figures
(in billions: Asia 4.7, Africa 1.4, Americas 1.03, Europe 0.75, Oceania 0.045).

The MNS system is genetically linked (the MN and Ss antigens are encoded by the
adjacent *GYPA*/*GYPB* genes), so it is modeled as a single variable with nine
phenotype states rather than as two independent variables. The rare null
phenotypes -- Jk(a-b-) for Kidd and S-s- for MNS -- are dropped and the
remaining phenotype frequencies renormalized.
"""

import itertools

from ...distribution import Distribution

__all__ = ("blood_types",)


_REGIONS = ("Africa", "Americas", "Asia", "Europe", "Oceania")

# Approximate 2020s population by continent, in billions. Used only to weight the
# regions when forming the global (Region-marginal) distribution.
_POPULATION = {
    "Africa": 1.40,
    "Americas": 1.03,
    "Asia": 4.70,
    "Europe": 0.75,
    "Oceania": 0.045,
}

# --- ABO -------------------------------------------------------------------
_ABO = ("O", "A", "B", "AB")
_ABO_FREQ = {
    "Africa": (0.52, 0.24, 0.20, 0.04),
    "Americas": (0.55, 0.30, 0.11, 0.04),
    "Asia": (0.35, 0.28, 0.28, 0.09),
    "Europe": (0.42, 0.42, 0.12, 0.04),
    "Oceania": (0.45, 0.40, 0.12, 0.03),
}

# --- Rh (D antigen) --------------------------------------------------------
_RH = ("+", "-")
_RH_FREQ = {
    "Africa": (0.95, 0.05),
    "Americas": (0.93, 0.07),
    "Asia": (0.99, 0.01),
    "Europe": (0.85, 0.15),
    "Oceania": (0.93, 0.07),
}

# --- Kell (K antigen) ------------------------------------------------------
_KELL = ("K+", "K-")
_KELL_FREQ = {
    "Africa": (0.02, 0.98),
    "Americas": (0.06, 0.94),
    "Asia": (0.01, 0.99),
    "Europe": (0.09, 0.91),
    "Oceania": (0.03, 0.97),
}

# --- Duffy (Fya/Fyb) -------------------------------------------------------
_DUFFY = ("Fy(a+b-)", "Fy(a-b+)", "Fy(a+b+)", "Fy(a-b-)")
_DUFFY_FREQ = {
    "Africa": (0.10, 0.05, 0.05, 0.80),
    "Americas": (0.22, 0.30, 0.38, 0.10),
    "Asia": (0.80, 0.12, 0.07, 0.01),
    "Europe": (0.20, 0.34, 0.45, 0.01),
    "Oceania": (0.40, 0.25, 0.30, 0.05),
}

# --- Kidd (Jka/Jkb), the Jk(a-b-) null dropped and renormalized ------------
_KIDD = ("Jk(a+b-)", "Jk(a-b+)", "Jk(a+b+)")
_KIDD_FREQ = {
    "Africa": (0.50, 0.10, 0.40),
    "Americas": (0.32, 0.18, 0.50),
    "Asia": (0.28, 0.24, 0.48),
    "Europe": (0.27, 0.23, 0.50),
    "Oceania": (0.35, 0.20, 0.45),
}

# --- MNS: MN phenotype x Ss phenotype (S-s- null dropped, renormalized) ----
# Modeled as one linked system: the nine states are the product of the MN and
# Ss phenotypes within each region.
_MN = ("M+N-", "M+N+", "M-N+")
_MN_FREQ = {
    "Africa": (0.28, 0.50, 0.22),
    "Americas": (0.30, 0.50, 0.20),
    "Asia": (0.33, 0.49, 0.18),
    "Europe": (0.30, 0.50, 0.20),
    "Oceania": (0.30, 0.48, 0.22),
}
_SS = ("S+s-", "S+s+", "S-s+")
_SS_FREQ = {
    "Africa": (0.05, 0.25, 0.70),
    "Americas": (0.09, 0.36, 0.55),
    "Asia": (0.02, 0.08, 0.90),
    "Europe": (0.11, 0.44, 0.45),
    "Oceania": (0.05, 0.20, 0.75),
}

_MNS = tuple(mn + ss for mn in _MN for ss in _SS)


def _mns_freq(region):
    """The nine MNS phenotype probabilities for ``region``, as MN x Ss."""
    mn = _MN_FREQ[region]
    ss = _SS_FREQ[region]
    return {mn_label + ss_label: mn[i] * ss[j]
            for i, mn_label in enumerate(_MN)
            for j, ss_label in enumerate(_SS)}


def blood_types():
    """
    The common-cause joint distribution of human blood group systems by continent.

    The seven random variables are, in order:

    * ``Region`` -- continent: ``Africa``, ``Americas``, ``Asia``, ``Europe``,
      ``Oceania`` (weighted by population).
    * ``ABO``    -- ``O``, ``A``, ``B``, ``AB``.
    * ``Rh``     -- ``+`` or ``-`` (the D antigen).
    * ``Kell``   -- ``K+`` or ``K-``.
    * ``Duffy``  -- ``Fy(a+b-)``, ``Fy(a-b+)``, ``Fy(a+b+)``, ``Fy(a-b-)``.
    * ``Kidd``   -- ``Jk(a+b-)``, ``Jk(a-b+)``, ``Jk(a+b+)``.
    * ``MNS``    -- one of nine ``M#N#S#s#`` phenotypes.

    Each antigen system is conditionally independent of the others given
    ``Region``. The global (continent-agnostic) blood type distribution is the
    ``Region`` marginal; a single continent is recovered by conditioning on
    ``Region``.

    Returns
    -------
    d : Distribution
        The joint distribution over
        ``(Region, ABO, Rh, Kell, Duffy, Kidd, MNS)``.
    """
    total_pop = sum(_POPULATION.values())

    outcomes = []
    pmf = []
    for region in _REGIONS:
        p_region = _POPULATION[region] / total_pop
        mns_freq = _mns_freq(region)
        for abo, rh, kell, duffy, kidd, mns in itertools.product(
            _ABO, _RH, _KELL, _DUFFY, _KIDD, _MNS
        ):
            p = (
                p_region
                * _ABO_FREQ[region][_ABO.index(abo)]
                * _RH_FREQ[region][_RH.index(rh)]
                * _KELL_FREQ[region][_KELL.index(kell)]
                * _DUFFY_FREQ[region][_DUFFY.index(duffy)]
                * _KIDD_FREQ[region][_KIDD.index(kidd)]
                * mns_freq[mns]
            )
            outcomes.append((region, abo, rh, kell, duffy, kidd, mns))
            pmf.append(p)

    d = Distribution(outcomes, pmf)
    d.set_rv_names(("Region", "ABO", "Rh", "Kell", "Duffy", "Kidd", "MNS"))
    return d
