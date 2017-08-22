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
            op: ["\\operatorname{#1}\\left[#2\\right]", 2],
            H: ["\\op{H}{#1}", 1],
            I: ["\\op{I}{#1}", 1],
            T: ["\\op{T}{#1}", 1],
            B: ["\\op{B}{#1}", 1],
            J: ["\\op{J}{#1}", 1],
            R: ["\\op{R}{#1}", 1],
            II: ["\\op{II}{#1}", 1],
            TSE: ["\\op{TSE}{#1}", 1],
            K: ["\\op{K}{#1}", 1],
            C: ["\\op{C}{#1}", 1],
            G: ["\\op{G}{#1}", 1],
            F: ["\\op{F}{#1}", 1],
            M: ["\\op{M}{#1}", 1],
            P: ["\\op{P}{#1}", 1],
            X: ["\\op{X}{#1}", 1],
            CRE: ["\\op{\\mathcal{E}}{#1}", 1],
            GCRE: ["\\op{\\mathcal{E^\\prime}}{#1}", 1],
            RE: ["\\op{H_{#1}}{#2}", 2],
            TE: ["\\op{S_{#1}}{#2}", 2],
            xH: ["\\op{xH}{#1}", 1],
            DKL: ["\\op{D_{KL}}{#1}", 1],
            JSD: ["\\op{D_{JS}}{#1}", 1],
            Icap: ["\\op{I_{\\cap}}{#1}", 1],
            Imin: ["\\op{I_{min}}{#1}", 1],
            Immi: ["\\op{I_{MMI}}{#1}", 1],
            Iwedge: ["\\op{I_{\\wedge}}{#1}", 1],
            Iproj: ["\\op{I_{proj}}{#1}", 1],
            Ibroja: ["\\op{I_{BROJA}}{#1}", 1],
            Iccs: ["\\op{I_{ccs}}{#1}", 1],
            Ida: ["\\op{I_{\\downarrow}}{#1}", 1],
            Idda: ["\\op{I_{\\Downarrow}}{#1}", 1],
            Idep: ["\\op{I_{dep}}{#1}", 1],
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

MathJax.Ajax.loadComplete("http://dit.readthedocs.io/en/latest/_static/dit_mathjax.js");
