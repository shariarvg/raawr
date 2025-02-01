# RAAWR: Regenerative Adversarial Attacks for Watermark Removal

Given a watermarked text generation $G$ from a language model $A$, can a bidirectional
language model $B$ iteratively update the words in $G$ until the watermark is no longer detectable?

The motivation here is to see how well watermarks can withstand watermark removal, thereby challenging
whether the standard algorithms for watermarking are sufficiently robust. I also am interested in 
AI detection, which is a difficult problem to solve, and I think that the robustness of AI detection
algorithms (on watermarked and un-watermarked LM generations) can be studied-well in the context of
watermark removal. 

TLDR, removing watermarks is really easy, but removing watermarks while preserving semantic similarity and writing quality is extremely hard, butttttttt there exists a natural, obvious benchmark (GPT evaluation of generations before and after watermark removal).

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview
-- Implementation of generation with hard watermarking as described by [Kirchenbauer et. al](https://proceedings.mlr.press/v202/kirchenbauer23a.html) for open-source models with logit biases
-- An implementation of watermarking via non-open-source models, where I literally just generate a random blacklist and then generate a token at a time, re-sampling that token until it's not in the blacklist anymore.
-- Implementation of detection of hard watermarking 
-- A few implementations of watermark removal
---

Note that this project is a work in-progress. If you have any feedback or are working on this problem as well, please reach out to me (firstname.lastnamewithoutthehyphen at duke dot edu)

