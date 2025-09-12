# Lip Reading Model

Dual Python and Rust implementation of an acoustic lip-reading model. Model processes video to predict spoken text.

# Model Architecture

Python implementation is based from Nicholas Renotte's YouTube tutorial.

# References

- Alex Graves. 2006. _Connectionist Temporal Classification_. DOI: https://doi.org/10.1145/1143844.1143891
- Y. M. Assael et al. 2016. _LipNet_. arXiv: https://arxiv.org/abs/1611.01599

# TODO

- Implement CTC decoder first with Greedy, then with Beam search
- Create a new file called eval.rs or inference.rs
- Implement Levenshtein distance solver inside this file
- Implement Burkhard-Keller tree for fuzzy sequence comparisons
- Implement CER (Character Error Rate) as an eval metric for decoded model outputs
- CER = edit_dist(substitutions + insertions + deletions) / (target chars)
- Include a language model for inference assistance (n-gram or neural LM) for language context ambiguity resolution
- change loss/decode function signature params from inputs to logits for following convention