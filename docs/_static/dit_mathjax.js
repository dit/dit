/* -*- Mode: Javascript; indent-tabs-mode:nil; js-indent-level: 2 -*- */
/* vim: set ts=2 et sw=2 tw=80: */

/*************************************************************
 *
 *  MathJax/config/local/local.js
 *
 *  Include changes and configuration local to your installation
 *  in this file.  For example, common macros can be defined here
 *  (see below).  To use this file, add "local/local.js" to the
 *  config array in MathJax.js or your MathJax.Hub.Config() call.
 *
 *  ---------------------------------------------------------------------
 *
 *  Copyright (c) 2009-2013 The MathJax Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


MathJax.Hub.Config({
    "HTML-CSS": { availableFonts: ["TeX"] },
    TeX: {
        Macros: {
            op: ["\\operatorname{#1}", 1],
            H: "\\op{H}",
            I: "\\op{I}",
            T: "\\op{T}",
            B: "\\op{B}",
            J: "\\op{J}",
            R: "\\op{R}",
            II: "\\op{II}",
            TSE: "\\op{TSE}",
            K: "\\op{K}",
            C: "\\op{C}",
            G: "\\op{G}",
            F: "\\op{F}",
            M: "\\op{M}",
            P: "\\op{P}",
            X: "\\op{X}",
            CRE: "\\op{\\mathcal{E}}",
            GCRE: "\\op{\\mathcal{E^\\prime}}",
            RE: "\\opREI}",
            TE: "\\op{TE}",
            xH: "\\op{xH}",
            DKL: "\\op{D_{KL}}",
            JSD: "\\op{D_{JS}}",
            meet: "\\curlywedge",
            join: "\\curlyvee",
            iless: "\\preceq",
            imore: "\\succeq",
            ieq: "\\cong",
            mss: "\\searrow",
            meetop: "\\DeclareMathOperator*{\\meetop}{\\scalerel*{\\meet}{\\textstyle\\sum}}",
            joinop: "\\DeclareMathOperator*{\\joinop}{\\scalerel*{\\join}{\\textstyle\\sum}}",
            ind: "\\mathrel{\\text{\\scalebox{1.07}{$\\perp\\mkern-10mu\\perp$}}}"
        }
    }
});

MathJax.Ajax.loadComplete("https://raw.githubusercontent.com/dit/dit/master/docs/_static/dit_mathjax.js");
