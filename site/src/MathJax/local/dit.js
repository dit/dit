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


MathJax.Hub.Register.StartupHook("TeX Jax Ready",function () {
  var TEX = MathJax.InputJax.TeX;

  TEX.Macro("op", "\\operatorname{#1}", 1)
  TEX.Macro("H", "\\op{H}");
  TEX.Macro("I", "\\op{I}");
  TEX.Macro("T", "\\op{T}");
  TEX.Macro("B", "\\op{B}");
  TEX.Macro("R", "\\op{R}");
  TEX.Macro("K", "\\op{K}");
  TEX.Macro("II", "\\op{II}");
  TEX.Macro("DKL", "\\op{D_{KL}}");
  TEX.Macro("xH", "\\op{xH}");
  TEX.Macro("JSD", "\\op{JSD}");

  // some binary operators
  TEX.Macro("meet", "\\curlywedge");
  TEX.Macro("join", "\\curlyvee");
  TEX.Macro("iless", "\\preceq");
  TEX.Macro("imore", "\\succeq");

  // don't use stix, it's pretty ugly
  MathJax.Hub.Config({
    "HTML-CSS": { availableFonts: ["TeX"] }
  });
  // place macros here.  E.g.:
  //   TEX.Macro("R","{\\bf R}");
  //   TEX.Macro("op","\\mathop{\\rm #1}",1); // a macro with 1 parameter

});

MathJax.Ajax.loadComplete("http://dit.io/MathJax/local/dit.js");
