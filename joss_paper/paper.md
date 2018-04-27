---
title: "``dit``: a Python package for discrete information theory"
tags:
  - information theory
authors:
 - name: Ryan G. James
   orcid: 0000-0003-2149-9085
   affiliation: 1
 - name: Christopher J. Ellison
 - name: James P. Crutchfield
   affiliation: 1
affiliations:
 - name: Complexity Sciences Center, Department of Physics, University of California at Davis
   index: 1
date: 27 April 2018
bibliography: paper.bib
---

# Summary

``dit``[@dit] is a Python package for the study of **d**iscrete **i**nformation **t**heory. Information theory is a mathematical framework for the study of quantifying, compressing, and communicating random variables [@Cover2006][@MacKay2003][@Yeung2008]. More recently, information theory has been utilized within the physical and social sciences to quantify how different components of a system interact. ``dit`` is primarily concerned with this aspect of the theory.

``dit`` implements the vast majority of information measure defined in the literature, including entropies (Shannon[@Cover2006], Renyi, Tsallis), multivariate mutual informations (co-information[@Bell2003][@mcgill1954multivariate], total correlation[@watanabe1960information], dual total correlation[@te1980multiple][@Han1975linear][@Abdallah2012], CAEKL mutual information[@chan2015multivariate]), common informations (Gács-Körner[@Gacs1973][@tyagi2011function], Wyner[@wyner1975common][@liu2010common], exact[@kumar2014exact], functional, minimal sufficient statistic), and channel capacity[@Cover2006]. It includes methods of studying joint distributions including information diagrams, connected informations[@Schneidman2003][@Amari2001], marginal utility of information[@Allen2014], and the complexity profile[@Baryam2004]. It also includes several more specialized modules including bounds on the secret key agreement rate[@maurer1997intrinsic], partial information decomposition[@williams2010nonnegative], rate-distortion theory[@Cover2006] & information bottleneck[@tishby2000information], and others. Please see the [``dit`` homepage](https://github.com/dit/dit) for a complete and up-to-date list.

Where possible, the implementations in ``dit`` support multivariate, conditional forms even if not defined that way in the literature. For example, ``dit`` implements the multivariate, conditional exact common information even though it was only defined for two variables.

# References