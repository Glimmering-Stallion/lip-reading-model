//! Connectionist Temporal Classification (CTC) decoder implementation.
//! 
//! This module implements methods to transform raw model logits into
//! final discrete token sequences. It supports two decoding search strategies:
//! - Greedy Search: A fast, best-path decoder that selects the most probable
//! token at each timestep and collapses the resulting path.
//! - Prefix Beam Search: A more complex, strategy that builds upon multiple
//! best prefix sequence candidate paths(hypotheses) and selects the highest
//! scoring final sequence path.
//! 
//! For Prefix Beam Search, there is support for shallow fusion with an external
//! `LanguageModel` to rescore hypothesis paths during the search process.

// if using prefix beam search as the decoder, expect O(DWK * log(WK)) complexity
// where:
// D = depth (# timesteps)
// W = beam width (# of kept hypotheses at each timestep)
// K = branch factor (vocab size pruned to top K token candidates at each timestep)



// custom imports
use crate::{
    ctc::lm::{
        LanguageModel,
        LanguageModelConfig,
    },
    utils::log_sum_exp_2_scalar,
};

// imports
use burn::{
    config::Config,
    tensor::{
        activation::log_softmax,
        backend::Backend,
        Tensor,
    },
};
use rustc_hash::FxHashMap;



/// Single prefix hypothesis on the beam during CTC prefix beam search.
pub(crate) struct Prefix {
    pub(crate) sequence: Vec<usize>,       // sequence of token IDs in vocab
    pub(crate) log_prob_blank: f32,        // log-prob of prefix ending in blank
    pub(crate) log_prob_non_blank: f32,    // log-prob of prefix ending in non-blank
    pub(crate) log_prob_lm: f32,           // log-prob from language model (for LM fusion)
    pub(crate) combined_log_prob: f32,     // combined accumulated log-prob score of this prefix
}



/// Active beam: the ordered list of prefix hypotheses for one decode step.
pub(crate) struct Beam(Vec<Prefix>);



impl Prefix {
    pub(crate) fn new(
        sequence: Vec<usize>,
        log_prob_blank: f32,
        log_prob_non_blank: f32,
        log_prob_lm: f32,
        alpha: f32,
        beta: f32,
    ) -> Self {
        let vsrm_score = log_sum_exp_2_scalar(log_prob_blank, log_prob_non_blank);
        let lm_score = log_prob_lm;
        let length_reward = sequence.len() as f32;

        // combined accumulated log-prob score of this prefix
        let combined_log_prob = vsrm_score + (alpha * lm_score) + (beta * length_reward); // shallow LM fusion

        Self {
            sequence,
            log_prob_blank,
            log_prob_non_blank,
            log_prob_lm,
            combined_log_prob,
        }
    }

    pub(crate) fn last_char(&self) -> Option<usize> { self.sequence.last().copied() }
}



impl Beam {
    /// Create initial beam with single empty-prefix hypothesis before first timestep (t = -1).
    pub(crate) fn new() -> Self {
        Self(vec![
            Prefix::new(
                Vec::new(),
                0.0,                  // log(1)
                f32::NEG_INFINITY,    // log(0)
                0.0,
                0.0,
                0.0,
            ),
        ])
    }

    /// Create new beam from given prefix hypotheses for next timestep.
    pub(crate) fn from_hypotheses(hypotheses: Vec<Prefix>) -> Self { Self(hypotheses) }

    /// Returns slice of prefix hypotheses in beam for current timestep.
    pub(crate) fn hypotheses(&self) -> &[Prefix] { &self.0 }

    /// Returns vector of prefix hypotheses in beam for current timestep.
    pub(crate) fn into_hypotheses(self) -> Vec<Prefix> { self.0 }
}



#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum CtcDecodeType {
    GreedySearch,
    BeamSearch,
}



#[derive(Config, Debug)]
pub struct CtcDecoderConfig {
    #[config(default = "0")]
    pub blank_id: usize,                  // ID of blank token in vocab

    #[config(default = "CtcDecodeType::GreedySearch")]
    pub search_type: CtcDecodeType,       // search type to use within CTC decoder (greedy/beam)

    // rest are beam search params (ignored for greedy)

    #[config(default = "5")]
    pub beam_width: usize,                // beam width for beam search

    #[config(default = "None")]
    pub lm: Option<LanguageModelConfig>,  // optional language model to supplement beam search
    
    #[config(default = 0.0)]              // with LM, default should be between [0.2, 3.0]
    pub lm_alpha: f32,                    // weight of language model score when combining with visual model score
    
    #[config(default = 0.0)]              // with LM, default should be between [1.5, 5.0]
    pub lm_beta: f32,                     // length normalization factor for beam search (to avoid short sequence bias)
}



impl CtcDecoderConfig {
    pub fn init(&self) -> CtcDecoder {
        CtcDecoder {
            blank_id: self.blank_id,
            search_type: self.search_type,
            beam_width: self.beam_width,
            lm: self.lm.as_ref().map(|lm| lm.init()),
            lm_alpha: self.lm_alpha,
            lm_beta: self.lm_beta,
        }
    }
}



#[derive(Debug)]
pub struct CtcDecoder {
    pub blank_id: usize,
    pub search_type: CtcDecodeType,
    pub beam_width: usize,
    pub lm: Option<Box<dyn LanguageModel + Send + Sync>>,
    pub lm_alpha: f32,
    pub lm_beta: f32,
}



/// One hypothesis on the beam after a CTC prefix beam step (for inspection / visualization).
#[derive(Clone, Debug)]
pub struct BeamHypothesisSnapshot {
    pub sequence: Vec<usize>,
    pub log_prob_blank: f32,
    pub log_prob_non_blank: f32,
    pub combined_log_prob: f32,
}



impl Clone for CtcDecoder {
    fn clone(&self) -> Self {
        Self {
            blank_id: self.blank_id,
            search_type: self.search_type,
            beam_width: self.beam_width,
            lm: self.lm.as_ref().map(|lm| lm.clone_box()), 
            lm_alpha: self.lm_alpha,
            lm_beta: self.lm_beta,
        }
    }
}



impl CtcDecoder {
    /// Applies CTC decode for batch of samples.
    /// 
    /// Inputs assumed to be padded to max timesteps length in batch.
    ///
    /// ### Params:
    /// - `inputs`: [N, T, Vocab] logits from model.
    ///
    /// ### Returns:
    /// Sequences of predicted token IDs (collapsed paths) for each sample in batch [N, L].
    pub fn forward<B: Backend>(
        &self,
        inputs: Tensor<B, 3>,
    ) -> Vec<Vec<i64>> {
        let [n, t, v] = inputs.dims();

        assert!(n > 0, "no samples in batch");
        assert!(t > 0, "no timesteps in inputs");
        assert!(v > 1, "vocab size must be > 1 for CTC decode");
        assert!(self.blank_id < v, "blank ID ({}) is out of vocabulary size bounds ({})", self.blank_id, v);

        match &self.search_type {
            CtcDecodeType::GreedySearch => self.greedy_search_decode(inputs),
            CtcDecodeType::BeamSearch => self.beam_search_decode(inputs),
        }
    }
}



impl CtcDecoder {
    /// Greedy search decode for batch of samples.
    /// 
    /// Inputs assumed to be padded to max timesteps length in batch.
    ///
    /// ### Params:
    /// - `inputs`: [N, T, Vocab] logits from model.
    ///
    /// ### Returns:
    /// Sequences of predicted token IDs (collapsed paths) for each sample in batch [N, L].
    fn greedy_search_decode<B: Backend>(
        &self,
        inputs: Tensor<B, 3>
    ) -> Vec<Vec<i64>> {
        let [n, t, _v] = inputs.dims();

        // grab most probable token ID from vocab distribution (dim 2), per timestep (dim 1), per sample (dim 0)
        let argmax_ids = inputs          // [N, T, Vocab]
            .argmax(2)                   // [N, T, 1]
            .reshape([n, t])           // [N, T]
            .to_data()
            .convert::<i64>()
            .to_vec()
            .unwrap();

        // collapse singular most probable path per sample by removing dupes and blanks
        argmax_ids
            .chunks(t)
            .map(|seq| Self::collapse_path(seq, self.blank_id as i64))
            .collect::<Vec<Vec<i64>>>()
    }

    /// Helper function for greedy search to collapse a path by:
    /// - removing blanks and false duplicates (repeated chars between blanks),
    /// - keeping true duplicates (repeated chars separated by blanks).
    /// 
    /// This helps to align time-based predictions with text-based targets.
    ///
    /// ### Params:
    /// - `path`: Sequence of token int IDs deemed most probable by model (with blanks and duplicates) [T].
    /// - `blank_id`: ID of blank token in vocab.
    ///
    /// ### Returns:
    /// Collapsed sequence of token int IDs (without blanks or duplicates) [L].
    fn collapse_path<T: PartialEq + Copy>(path: &[T], blank_id: T) -> Vec<T> {
        let mut collapsed_path = Vec::with_capacity(path.len());
        let mut prev = None;

        // ignore dupes and blanks
        for &token in path {
            if Some(token) != prev && token != blank_id { collapsed_path.push(token); }
            prev = Some(token);
        }

        collapsed_path
    }
}



impl CtcDecoder {
    /// Beam search decode for batch of samples.
    /// 
    /// Inputs assumed to be padded to max timesteps length in batch.
    ///
    /// ### Params:
    /// - `inputs`: [N, T, Vocab] logits from model.
    ///
    /// ### Returns:
    /// Sequences of predicted token IDs (collapsed paths) for each sample in batch [N, L].
    fn beam_search_decode<B: Backend>(
        &self,
        inputs: Tensor<B, 3>
    ) -> Vec<Vec<i64>> {
        let log_probs = log_softmax(inputs, 2); // [N, T, V]
        let [n, t, vocab_size] = log_probs.dims();
        let mut top_seq_ids = Vec::with_capacity(n);

        assert!((1..=15).contains(&self.beam_width), "beam width ({}) must be in [1, 15]", self.beam_width);
        if self.lm.is_some() {
            assert!((0.2..=3.0).contains(&self.lm_alpha), "LM alpha value ({}) must be in [0.2, 3.0]", self.lm_alpha);
            assert!((1.5..=5.0).contains(&self.lm_beta), "LM beta value ({}) must be in [1.5, 5.0]", self.lm_beta);
        };

        // loop over samples in batch
        for sample in 0..n {
            let sample_log_probs = log_probs.clone().slice([sample..(sample + 1), 0..t, 0..vocab_size]).squeeze::<2>(); // [T, V]
            top_seq_ids.push(self.decode_sample(sample_log_probs));
        }

        top_seq_ids
    }

    /// Beam search decode for single sample.
    /// 
    /// Decoding obtained via: logits --> log softmax --> prefix beam search.
    /// Delegates to [`Beam::new`], [`select_top_k_pairs`], [`prefix_extend_step`],
    /// [`beam_prune_step`], and [`select_best_prefix`].
    /// 
    /// Works by:
    /// - initializing beam with empty prefix hypothesis before first timestep,
    /// - iterating over timesteps and token-extending/pruning beam prefixes according to CTC prefix rules,
    /// - selecting best scoring prefix at end as final decoded sequence.
    ///
    /// ### Params:
    /// - `log_probs`: Log-probabilities for each vocab token per-timestep given by model [T, Vocab].
    ///
    /// ### Returns:
    /// Sequence of predicted token IDs (collapsed path) [L].
    #[inline]
    fn decode_sample<B: Backend>(
        &self,
        log_probs: Tensor<B, 2>,
    ) -> Vec<i64> {
        let [t, v] = log_probs.dims();
        let w = self.beam_width;
        let k = w.min(v - 1); // branch factor (vocab size pruned to top K token candidates at each timestep)

        // t = -1 (base case)
        let mut beam = Beam::new();

        // vector buffer to store per-timestep (ID, log-prob) pairs
        let mut top_k_pairs: Vec<(usize, f32)> = vec![(0, 0.0); v];

        // pull log-probs GPU tensor to CPU
        let mut log_probs = log_probs
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();

        // 0 ≤ t ≤ T - 1 (recurrence case)
        for t_idx in 0..t {
            // grab log-probs of each token given by model at current timestep
            let t_chunk = t_idx * v;
            let log_probs_t = &mut log_probs[t_chunk..(t_chunk + v)];
            
            // unconditionally obtain blank log-prob at current timestep for blank extension (then mask it out)
            // separately obtain top-K char log-probs for non-blank extension
            let log_prob_blank = log_probs_t[self.blank_id];
            log_probs_t[self.blank_id] = f32::NEG_INFINITY;
            self.select_top_k_pairs(log_probs_t, k, &mut top_k_pairs);

            let extended_prefix = self.prefix_extend_step(
                beam.into_hypotheses(),
                &top_k_pairs[..k],
                log_prob_blank,
                self.lm.as_deref(),
            );

            beam = Beam::from_hypotheses(self.beam_prune_step(extended_prefix, w, self.lm_alpha, self.lm_beta));
        }

        self.select_best_prefix(beam.into_hypotheses())
    }

    /// Fills a given buffer with `(id, log_prob)` pairs and partial-sorts to place the top `k` first.
    /// 
    /// This is used to efficiently select the top `k` candidate tokens from the vocabulary at each timestep for beam search extension.
    /// 
    /// ### Params:
    /// - `log_probs_t`: Log-probabilities of each token in vocab at current timestep (length = vocab size).
    /// - `k`: Number of top candidate tokens to select.
    /// - `buf`: Mutable buffer to store `(id, log_prob)` pairs for each token in vocab (length = vocab size).
    pub(crate) fn select_top_k_pairs(
        &self,
        log_probs_t: &[f32],
        k: usize,
        buf: &mut [(usize, f32)],
    ) {
        // fill (id, log-prob) pairs buffer with current timestep's pair values
        for (char_id, &char_log_prob) in log_probs_t.iter().enumerate()
        { buf[char_id] = (char_id, char_log_prob); }

        // get top K highest log-prob from those (id, log-prob) pairs
        buf.select_nth_unstable_by(k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap() // sort by log-prob (descending)
        });
    }

    /// Extends all beam prefixes by one timestep using CTC prefix beam rules.
    ///
    /// For each prefix, computes blank extension and non-blank extensions (cases A/B/C)
    /// with optional language model rescoring.
    /// 
    /// ### Params:
    /// - `prefixes`: Current beam prefixes to extend.
    /// - `top_k_pairs`: Top K (ID, log-prob) pairs for current timestep to consider for extension.
    /// - `log_prob_blank`: Log-prob of blank token at current timestep (for blank extension).
    /// - `lm`: Optional language model for rescoring extensions with LM log-prob adjustments (for LM fusion).
    /// 
    /// ### Returns:
    /// Accumulated `(blank_lp, non_blank_lp, lm_lp)` per unique sequence key.
    pub(crate) fn prefix_extend_step(
        &self,
        prefixes: Vec<Prefix>,
        top_k_pairs: &[(usize, f32)],
        log_prob_blank: f32,
        lm: Option<&(dyn LanguageModel + Send + Sync)>,
    ) -> FxHashMap<Vec<usize>, (f32, f32, f32)> {
        let sentinel = f32::NEG_INFINITY;
        let max_capacity = prefixes.len() * (top_k_pairs.len() + 1);
        let mut next_prefixes: FxHashMap<Vec<usize>, (f32, f32, f32)> =
            FxHashMap::with_capacity_and_hasher(max_capacity, Default::default());

        for prefix in prefixes.into_iter() {
            // build key for current prefix sequence hypothesis
            let sequence = prefix.sequence.clone();

            // compute log-prob of extending prefix with blank at current timestep
            // this is the succeeding prob-mass for both blank/non-blank source states
            let ext_log_prob = log_sum_exp_2_scalar(
                prefix.log_prob_blank + log_prob_blank,
                prefix.log_prob_non_blank + log_prob_blank,
            );

            // update total accumulated blank log-prob for current prefix path
            let entry = next_prefixes
                .entry(sequence.clone())
                .or_insert((sentinel, sentinel, prefix.log_prob_lm));
            entry.0 = log_sum_exp_2_scalar(entry.0, ext_log_prob);

            // extend with non-blank characters (char is the candidate token ID in vocab)
            for &(char_id, char_log_prob) in top_k_pairs {
                // case A (remain):     same char, previous path state ended with non-blank
                // case B (repeat):     same char, previous path state ended with blank
                // case C (append):     diff char, previous path state ended with either blank or non-blank
                match prefix.last_char() {
                    // if char equals last char in prefix path
                    Some(last_char_id) if last_char_id == char_id => {

                        // --------------- (A) ---------------
                        // - non-blank source state
                        // - cont. with same prefix sequence
                        // - don't append duplicate char

                        // build key for current prefix sequence hypothesis
                        let sequence_a = sequence.clone();

                        // compute log-prob of extending prefix with same char for current timestep
                        // this is the succeeding prob-mass for the non-blank source state only
                        let ext_log_prob_a = prefix.log_prob_non_blank + char_log_prob;

                        // update total accumulated non-blank log-prob for current prefix path
                        let entry_a = next_prefixes
                            .entry(sequence_a)
                            .or_insert((sentinel, sentinel, prefix.log_prob_lm));
                        entry_a.1 = log_sum_exp_2_scalar(entry_a.1, ext_log_prob_a);

                        // --------------- (B) ---------------
                        // - blank source state
                        // - start new prefix sequence
                        // - do append duplicate char

                        // build key for current prefix sequence hypothesis
                        let mut sequence_b = sequence.clone();
                        sequence_b.push(char_id); // append duplicate char

                        // compute log-prob of extending prefix with same char for current timestep
                        // this is the succeeding prob-mass for the blank source state only
                        let ext_log_prob_b = prefix.log_prob_blank + char_log_prob;

                        // optional language model score adjustment (applied for extending with duplicate char)
                        let mut new_lm_log_prob = prefix.log_prob_lm;
                        if let Some(lm) = lm
                        { new_lm_log_prob += lm.next_log_prob(&prefix.sequence, char_id); }

                        // update total accumulated non-blank log-prob for current prefix path
                        let entry_b = next_prefixes
                            .entry(sequence_b)
                            .or_insert((sentinel, sentinel, new_lm_log_prob));
                        entry_b.1 = log_sum_exp_2_scalar(entry_b.1, ext_log_prob_b);
                    }

                    // if char differs from last char in prefix path
                    _ => {

                        // --------------- (C) ---------------
                        // - new prefix sequence
                        // - append char from either state

                        // build key for current prefix sequence hypothesis
                        let mut sequence_c = sequence.clone();
                        sequence_c.push(char_id);

                        // compute log-prob of extending prefix with different char for current timestep
                        // this is the succeeding prob-mass for both blank/non-blank source states
                        let ext_log_prob_c = log_sum_exp_2_scalar(
                            prefix.log_prob_blank,
                            prefix.log_prob_non_blank,
                        ) + char_log_prob;

                        // optional language model score adjustment (applied for extending with different char)
                        let mut new_lm_log_prob = prefix.log_prob_lm;
                        if let Some(lm) = lm
                        { new_lm_log_prob += lm.next_log_prob(&prefix.sequence, char_id); }

                        // update total accumulated non-blank log-prob for current prefix path
                        let entry_c = next_prefixes
                            .entry(sequence_c)
                            .or_insert((sentinel, sentinel, new_lm_log_prob));
                        entry_c.1 = log_sum_exp_2_scalar(entry_c.1, ext_log_prob_c);
                    }
                }
            }
        }

        next_prefixes
    }

    /// Converts extended prefix map to sorted `Vec<Prefix>`, pruning to top `beam_width`.
    /// 
    /// If number of extended prefixes exceeds `beam_width`, partitions to top `beam_width` by combined log-prob score and drops rest.
    /// 
    /// ### Params:
    /// - `extended_prefix`: Map of extended prefix sequences to their accumulated log-prob scores (blank, non-blank, LM).
    /// - `beam_width`: Max number of prefix hypotheses to keep for next timestep.
    /// - `lm_alpha`: Weight of language model score when combining with visual model score (for LM fusion).
    /// - `lm_beta`: Length normalization factor for beam search (to avoid short sequence bias, for LM fusion).
    /// 
    /// ### Returns:
    /// Sorted vector of top `beam_width` prefix hypotheses for next timestep, with combined log-prob scores.
    pub(crate) fn beam_prune_step(
        &self,
        extended_prefix: FxHashMap<Vec<usize>, (f32, f32, f32)>,
        beam_width: usize,
        lm_alpha: f32,
        lm_beta: f32,
    ) -> Vec<Prefix> {
        // convert HashMap to Vec<Prefix> for sorting
        let mut candidates: Vec<Prefix> = extended_prefix
            .into_iter()
            .map(|(sequence, (log_prob_blank, log_prob_non_blank, log_prob_lm))| {
                Prefix::new(
                    sequence,
                    log_prob_blank,
                    log_prob_non_blank,
                    log_prob_lm,
                    lm_alpha,
                    lm_beta,
                )
            }).collect();

        // if candidates extended past beam_width
        // partition to top w by log-prob score
        // then drop remaining candidates
        if candidates.len() > beam_width {
            // select top beam_width prefix candidates
            candidates.select_nth_unstable_by(beam_width - 1, |a, b| {
                b.combined_log_prob.partial_cmp(&a.combined_log_prob).unwrap() // sort by score (descending)
            });
            candidates.truncate(beam_width); // remove non-top beam_width prefixes
        }

        candidates
    }

    /// Selects highest-scoring prefix and returns its sequence as `Vec<i64>`.
    /// 
    /// Assumes input prefixes are already pruned to top `beam_width` and sorted by score.
    /// 
    /// ### Params:
    /// - `prefixes`: Candidate prefix sequences with accumulated log-prob scores to select from.
    /// 
    /// ### Returns:
    /// Sequence of predicted token IDs (collapsed path) for best prefix.
    pub(crate) fn select_best_prefix(&self, prefixes: Vec<Prefix>) -> Vec<i64> {
        // return highest scoring prefix sequence
        let best_prefix = prefixes.into_iter()
            .max_by(|a, b|
                a.combined_log_prob.partial_cmp(&b.combined_log_prob).unwrap()
            )
            .unwrap();

        // convert usize IDs to i64 IDs
        best_prefix.sequence.into_iter().map(|u| u as i64).collect()
    }
}



// tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use crate::ctc::lm::LanguageModelConfig;
    use std::{
        array,
        path::Path,
    };
    use burn::{
        backend::ndarray::NdArray,
        tensor::{
            Distribution,
            Tensor,
        },
    };

    type B = NdArray<f32>;


    // helper to create one-hot logits array of vocab size for testing
    fn one_hot_logits<const V: usize>(hot: usize, hi: f32, lo: f32) -> [f32; V] {
        let mut row = [lo; V];
        row[hot] = hi;
        row
    }

    // helper to create multi-hot logits array of vocab size for testing
    fn multi_hot_logits<const V: usize>(hots: &[usize], hi: &[f32], lo: f32) -> [f32; V] {
        let mut row = [lo; V];
        for i in 0..hots.len() {
            let char_id = hots[i];
            let score = hi[i];
            row[char_id] = score;
        }
        row
    }

    #[test]
    fn test_collapse_path() {
        let vocab = VOCAB;
        let blank_id = BLANK_ID;
        let token_map = TokenMap::new(vocab);

        let path = vec![7, 4, 4, blank_id, 11, blank_id, 11, 11, 14, 14, 14]; // "h e e _ l _ l l o o o"
        let collapsed_path = CtcDecoder::collapse_path(&path, blank_id);

        println!("\nOriginal path IDs: {:?}", path);
        println!("Collapsed path IDs: {:?}\n", collapsed_path);
        println!("Original path chars: {:?}", token_map.ids_to_chars(&path.clone()));
        println!("Collapsed path chars: {:?}\n", token_map.ids_to_chars(&collapsed_path.clone()));

        assert_eq!(collapsed_path, vec![7, 4, 11, 11, 14]);
    }

    #[test]
    fn greedy_search_decode_random() {
        let vocab = VOCAB;
        let vocab_size = VOCAB_SIZE;
        let blank_id = BLANK_ID;
        let token_map = TokenMap::new(vocab);
        let device = Default::default();
        let dist = Distribution::Uniform(0.0, 1.0);

        // dummy logits for 2 samples, 5 timesteps, V vocab tokens
        let logits = Tensor::<B, 3>::random([2, 5, vocab_size], dist, &device);

        let decoder = CtcDecoderConfig::new()
            .with_search_type(CtcDecodeType::GreedySearch)
            .with_blank_id(blank_id)
            .init();

        let decoded_id_sequences = decoder.forward(logits);
        let decoded_char_sequences = decoded_id_sequences
            .iter()
            .map(|seq| {
                let ids: Vec<usize> = seq.iter().map(|&id| id as usize).collect();
                token_map.ids_to_chars(&ids).unwrap()
            })
            .collect::<Vec<Vec<char>>>();
        
        println!("\nDecoded ID sequences: {:?}", decoded_id_sequences);
        println!("Decoded char sequences: {:?}\n", decoded_char_sequences);

        assert_eq!(decoded_id_sequences.len(), 2);
        assert!(!decoded_id_sequences[0].contains(&(blank_id as i64))); // no blanks expected
        assert!(!decoded_id_sequences[1].contains(&(blank_id as i64))); // no blanks expected
    }

    #[test]
    fn greedy_search_decode_fixed() {
        const N: usize = 2;
        const T: usize = 11;
        const V: usize = VOCAB_SIZE;

        const HI: f32 = 8.0; // big logit for target token
        const LO: f32 = 0.0; // baseline for rest

        let blank_id = BLANK_ID;
        let token_map = TokenMap::new(VOCAB);
        let device = Default::default();

        let ids_1 = [7, 4, 4, blank_id, 11, blank_id, 11, 11, 14, 14, 14]; // "h e e _ l _ l l o o o"
        let ids_2 = [22, 22, 14, blank_id, 17, 17, blank_id, 11, 11, 3, 3]; // "w w o _ r r _ l l d d"

        let chars_1 = token_map.ids_to_chars(&ids_1).unwrap();
        let chars_2 = token_map.ids_to_chars(&ids_2).unwrap();

        // dummy logits for 2 samples, 11 timesteps, V vocab tokens (manually biased high blanks)
        let logits: [[[f32; V]; T]; N] = [
            array::from_fn(|t| one_hot_logits::<V>(ids_1[t], HI, LO)),
            array::from_fn(|t| one_hot_logits::<V>(ids_2[t], HI, LO)),
        ];
        let logits = Tensor::<B, 3>::from_data(logits, &device);

        let decoder = CtcDecoderConfig::new()
            .with_search_type(CtcDecodeType::GreedySearch)
            .with_blank_id(blank_id)
            .init();

        let decoded_id_sequences = decoder.forward(logits);
        let decoded_char_sequences = decoded_id_sequences
            .iter()
            .map(|seq| {
                let ids: Vec<usize> = seq.iter().map(|&id| id as usize).collect();
                token_map.ids_to_chars(&ids).unwrap()
            })
            .collect::<Vec<Vec<char>>>();

        println!("\nOriginal sample ID sequences: {:?}, {:?}", ids_1, ids_2);
        println!("Original sample char sequences: {:?}, {:?}\n", chars_1, chars_2);
        
        println!("Decoded ID sequences: {:?}", decoded_id_sequences);
        println!("Decoded char sequences: {:?}\n", decoded_char_sequences);

        assert_eq!(decoded_id_sequences.len(), 2);
        assert_eq!(decoded_id_sequences[0], vec![7, 4, 11, 11, 14]); // expected collapsed path for sample 1
        assert_eq!(decoded_id_sequences[1], vec![22, 14, 17, 11, 3]); // expected collapsed path for sample 2
    }

    #[test]
    fn beam_search_decode_random() {
        let vocab = VOCAB;
        let vocab_size = VOCAB_SIZE;
        let blank_id = BLANK_ID;
        let token_map = TokenMap::new(vocab);
        let device = Default::default();
        let dist = Distribution::Uniform(0.0, 1.0);

        // dummy logits for 2 samples, 5 timesteps, V vocab tokens
        let logits = Tensor::<B, 3>::random([2, 5, vocab_size], dist, &device);

        let decoder = CtcDecoderConfig::new()
            .with_search_type(CtcDecodeType::BeamSearch)
            .with_beam_width(10)
            .with_blank_id(blank_id)
            .init();

        let decoded_id_sequences = decoder.forward(logits);
        let decoded_char_sequences = decoded_id_sequences
            .iter()
            .map(|seq| {
                let ids: Vec<usize> = seq.iter().map(|&id| id as usize).collect();
                token_map.ids_to_chars(&ids).unwrap()
            })
            .collect::<Vec<Vec<char>>>();
        
        println!("\nDecoded id sequences: {:?}", decoded_id_sequences);
        println!("Decoded char sequences: {:?}\n", decoded_char_sequences);

        assert_eq!(decoded_id_sequences.len(), 2);
        assert!(!decoded_id_sequences[0].contains(&(blank_id as i64))); // no blanks expected
        assert!(!decoded_id_sequences[1].contains(&(blank_id as i64))); // no blanks expected
    }

    #[test]
    fn beam_search_decode_fixed() {
        const N: usize = 2;
        const T: usize = 11;
        const V: usize = VOCAB_SIZE;

        const HI: f32 = 8.0; // big logit for target token
        const LO: f32 = 0.0; // baseline for rest

        let blank_id = BLANK_ID;
        let token_map = TokenMap::new(VOCAB);
        let device = Default::default();

        let ids_1 = [7, 4, 4, blank_id, 11, blank_id, 11, 11, 14, 14, 14];   // "h e e _ l _ l l o o o"
        let ids_2 = [22, 22, 14, blank_id, 17, 17, blank_id, 11, 11, 3, 3];  // "w w o _ r r _ l l d d"

        let chars_1 = token_map.ids_to_chars(&ids_1).unwrap();
        let chars_2 = token_map.ids_to_chars(&ids_2).unwrap();

        // dummy logits for 2 samples, 11 timesteps, V vocab tokens (manually biased high blanks)
        let logits: [[[f32; V]; T]; N] = [
            array::from_fn(|t| one_hot_logits::<V>(ids_1[t], HI, LO)),
            array::from_fn(|t| one_hot_logits::<V>(ids_2[t], HI, LO)),
        ];
        let logits = Tensor::<B, 3>::from_data(logits, &device);

        let decoder = CtcDecoderConfig::new()
            .with_search_type(CtcDecodeType::BeamSearch)
            .with_beam_width(1)
            .with_blank_id(blank_id)
            .init();

        let decoded_id_sequences = decoder.forward(logits);
        let decoded_char_sequences = decoded_id_sequences
            .iter()
            .map(|seq| {
                let ids: Vec<usize> = seq.iter().map(|&id| id as usize).collect();
                token_map.ids_to_chars(&ids).unwrap()
            })
            .collect::<Vec<Vec<char>>>();

        println!("\nOriginal sample ID sequences: {:?}, {:?}", ids_1, ids_2);
        println!("Original sample char sequences: {:?}, {:?}\n", chars_1, chars_2);
        
        println!("Decoded ID sequences: {:?}", decoded_id_sequences);
        println!("Decoded char sequences: {:?}\n", decoded_char_sequences);

        assert_eq!(decoded_id_sequences.len(), 2);
        assert_eq!(decoded_id_sequences[0], vec![7, 4, 11, 11, 14]);  // expected collapsed path for sample 1
        assert_eq!(decoded_id_sequences[1], vec![22, 14, 17, 11, 3]); // expected collapsed path for sample 2
    }

    #[test]
    fn beam_eq_greedy_when_beam_width_1() {
        const N: usize = 2;
        const T: usize = 11;
        const V: usize = VOCAB_SIZE;

        const HI: f32 = 8.0; // big logit for target token
        const LO: f32 = 0.0; // baseline for rest

        let blank_id = BLANK_ID;
        let token_map = TokenMap::new(VOCAB);
        let device = Default::default();

        let ids_1 = [12, 0, 0, blank_id, blank_id, 19, 2, 2, 7, 7, 7];     // "m a a _ _ t c c h h h"
        let ids_2 = [18, 20, 2, blank_id, 2, 4, 4, 18, blank_id, 18, 18];  // "s u c _ c e e s _ s s"

        let chars_1 = token_map.ids_to_chars(&ids_1).unwrap();
        let chars_2 = token_map.ids_to_chars(&ids_2).unwrap();

        // dummy logits for 2 samples, 11 timesteps, V vocab tokens (manually biased high blanks)
        let logits: [[[f32; V]; T]; N] = [
            array::from_fn(|t| one_hot_logits::<V>(ids_1[t], HI, LO)),
            array::from_fn(|t| one_hot_logits::<V>(ids_2[t], HI, LO)),
        ];
        let logits = Tensor::<B, 3>::from_data(logits, &device);

        let beam_decoder = CtcDecoderConfig::new()
            .with_search_type(CtcDecodeType::BeamSearch)
            .with_beam_width(1)
            .with_blank_id(blank_id)
            .init();

        let greedy_decoder = CtcDecoderConfig::new()
            .with_search_type(CtcDecodeType::GreedySearch)
            .with_blank_id(blank_id)
            .init();

        let beam_decoded_id_sequences = beam_decoder.forward(logits.clone());
        let greedy_decoded_id_sequences = greedy_decoder.forward(logits);

        let beam_decoded_char_sequences = beam_decoded_id_sequences
            .iter()
            .map(|seq| {
                let ids: Vec<usize> = seq.iter().map(|&id| id as usize).collect();
                token_map.ids_to_chars(&ids).unwrap()
            })
            .collect::<Vec<Vec<char>>>();
        let greedy_decoded_char_sequences = greedy_decoded_id_sequences
            .iter()
            .map(|seq| {
                let ids: Vec<usize> = seq.iter().map(|&id| id as usize).collect();
                token_map.ids_to_chars(&ids).unwrap()
            })
            .collect::<Vec<Vec<char>>>();

        println!("\nOriginal sample ID sequences: {:?}, {:?}", ids_1, ids_2);
        println!("Original sample char sequences: {:?}, {:?}\n", chars_1, chars_2);

        println!("Beam decoded ID sequences: {:?}", beam_decoded_id_sequences);
        println!("Beam decoded char sequences: {:?}\n", beam_decoded_char_sequences);

        println!("Greedy decoded ID sequences: {:?}", greedy_decoded_id_sequences);
        println!("Greedy decoded char sequences: {:?}\n", greedy_decoded_char_sequences);

        // with beam width 1, no LM, with logits high-contrast and unambiguous, beam output should match greedy
        assert_eq!(beam_decoded_id_sequences, greedy_decoded_id_sequences);
    }

    #[test]
    fn decode_preserves_duplicates_when_blanks_present() {
        const N: usize = 5;
        const T: usize = 14;
        const V: usize = VOCAB_SIZE;

        const HI: f32 = 8.0; // big logit for target token
        const LO: f32 = 0.0; // baseline for rest

        let blank_id = BLANK_ID;
        let token_map = TokenMap::new(VOCAB);
        let device = Default::default();

        let ids_1 = [6, 14, 27, 14, 3, 26, 6, 14, 14, 3, 27, 27, 27, 27];    // "g o _ o d   g o o d _ _ _"
        let ids_2 = [11, 0, 19, 27, 19, 4, 17, 26, 11, 0, 19, 19, 4, 17];    // "l a t _ t e r   l a t t e r"
        let ids_3 = [3, 8, 13, 27, 13, 4, 17, 26, 3, 8, 13, 13, 4, 17];      // "d i n _ n e r   d i n n e r"
        let ids_4 = [18, 20, 15, 27, 15, 4, 17, 26, 18, 20, 15, 15, 4, 17];  // "s u p _ p e r   s u p p e r"
        let ids_5 = [7, 14, 15, 27, 15, 4, 3, 26, 7, 14, 15, 15, 4, 3];      // "h o p _ p e d   h o p p e d"

        let chars_1 = token_map.ids_to_chars(&ids_1).unwrap();
        let chars_2 = token_map.ids_to_chars(&ids_2).unwrap();
        let chars_3 = token_map.ids_to_chars(&ids_3).unwrap();
        let chars_4 = token_map.ids_to_chars(&ids_4).unwrap();
        let chars_5 = token_map.ids_to_chars(&ids_5).unwrap();

        let expected_outputs = [
            vec![6, 14, 14, 3, 26, 6, 14, 3],                     // "g o o d   g o d"
            vec![11, 0, 19, 19, 4, 17, 26, 11, 0, 19, 4, 17],     // "l a t t e r   l a t e r"
            vec![3, 8, 13, 13, 4, 17, 26, 3, 8, 13, 4, 17],       // "d i n n e r   d i n e r"
            vec![18, 20, 15, 15, 4, 17, 26, 18, 20, 15, 4, 17],   // "s u p p e r   s u p e r"
            vec![7, 14, 15, 15, 4, 3, 26, 7, 14, 15, 4, 3],       // "h o p p e d   h o p e d"
        ];

        // dummy logits for 5 samples, 14 timesteps, V vocab tokens (manually biased high blanks)
        let logits: [[[f32; V]; T]; N] = [
            array::from_fn(|t| one_hot_logits::<V>(ids_1[t], HI, LO)),
            array::from_fn(|t| one_hot_logits::<V>(ids_2[t], HI, LO)),
            array::from_fn(|t| one_hot_logits::<V>(ids_3[t], HI, LO)),
            array::from_fn(|t| one_hot_logits::<V>(ids_4[t], HI, LO)),
            array::from_fn(|t| one_hot_logits::<V>(ids_5[t], HI, LO)),

        ];
        let logits = Tensor::<B, 3>::from_data(logits, &device);

        let beam_decoder = CtcDecoderConfig::new()
            .with_search_type(CtcDecodeType::BeamSearch)
            .with_beam_width(10)
            .with_blank_id(blank_id)
            .init();

        let greedy_decoder = CtcDecoderConfig::new()
            .with_search_type(CtcDecodeType::GreedySearch)
            .with_blank_id(blank_id)
            .init();

        let beam_decoded_id_sequences = beam_decoder.forward(logits.clone());
        let greedy_decoded_id_sequences = greedy_decoder.forward(logits);

        let beam_decoded_char_sequences = beam_decoded_id_sequences
            .iter()
            .map(|seq| {
                let ids: Vec<usize> = seq.iter().map(|&id| id as usize).collect();
                token_map.ids_to_chars(&ids).unwrap()
            })
            .collect::<Vec<Vec<char>>>();
        let greedy_decoded_char_sequences = greedy_decoded_id_sequences
            .iter()
            .map(|seq| {
                let ids: Vec<usize> = seq.iter().map(|&id| id as usize).collect();
                token_map.ids_to_chars(&ids).unwrap()
            })
            .collect::<Vec<Vec<char>>>();

        println!("\nOriginal sample id sequences:\n{:?}\n{:?}\n{:?}\n{:?}\n{:?}\n", ids_1, ids_2, ids_3, ids_4, ids_5);
        println!("Original sample char sequences:\n{:?}\n{:?}\n{:?}\n{:?}\n{:?}\n", chars_1, chars_2, chars_3, chars_4, chars_5);
        
        println!("Beam decoded id sequences:");
        for seq in &beam_decoded_id_sequences { println!("  {:?}", seq); }
        println!();
        println!("Beam decoded char sequences:");
        for seq in &beam_decoded_char_sequences { println!("  {}", seq.iter().collect::<String>()); }
        println!();
        
        println!("Greedy decoded id sequences:");
        for seq in &greedy_decoded_id_sequences { println!("  {:?}", seq); }
        println!();
        println!("Greedy decoded char sequences:");
        for seq in &greedy_decoded_char_sequences { println!("  {}", seq.iter().collect::<String>()); }
        println!();

        // sequences with duplicates separated by blanks should preserve duplicates in output
        // similarly, sequences with duplicates not separated by blanks should collapse in output
        for n in 0..N {
            assert_eq!(beam_decoded_id_sequences[n], expected_outputs[n]);
            assert_eq!(greedy_decoded_id_sequences[n], expected_outputs[n]);
        }
        assert_eq!(beam_decoded_id_sequences, greedy_decoded_id_sequences);
    }
    
    #[test]
    fn beam_outperforms_greedy_on_global_optimum() {
        const N: usize = 1;
        const T: usize = 5;
        const V: usize = 5;

        let vocab = "abcd_";
        let blank_id = vocab.len() - 1;
        let token_map = TokenMap::new(vocab);
        let device = Default::default();

        // at t = 2, greedy picks "b", but globally "aa" is more probable
        let logits: [[[f64; V]; T]; N] = [[
            [ 10.0, -10.0, -10.0, -10.0, -10.0], // t = 0:  "a"
            [-10.0, -10.0, -10.0, -10.0,  10.0], // t = 1:  "_"
            [ -0.5,  -0.4,  -2.0,  -5.0,  -0.5], // t = 2:  "b" is local max here, but 'a' + '_' combined is stronger globally
            [ 10.0, -10.0, -10.0, -10.0, -10.0], // t = 3:  "a"
            [-10.0, -10.0, -10.0, -10.0,  10.0], // t = 4:  "_"
        ]];
        let logits = Tensor::<B, 3>::from_data(logits, &device);

        let beam_decoder = CtcDecoderConfig::new()
            .with_search_type(CtcDecodeType::BeamSearch)
            .with_beam_width(3)
            .with_blank_id(blank_id)
            .init();

        let greedy_decoder = CtcDecoderConfig::new()
            .with_search_type(CtcDecodeType::GreedySearch)
            .with_blank_id(blank_id)
            .init();

        let beam_decoded_id_sequences = beam_decoder.forward(logits.clone());
        let greedy_decoded_id_sequences = greedy_decoder.forward(logits);

        let beam_decoded_char_sequences = beam_decoded_id_sequences
            .iter()
            .map(|seq| {
                let ids: Vec<usize> = seq.iter().map(|&id| id as usize).collect();
                token_map.ids_to_chars(&ids).unwrap()
            })
            .collect::<Vec<Vec<char>>>();
        let greedy_decoded_char_sequences = greedy_decoded_id_sequences
            .iter()
            .map(|seq| {
                let ids: Vec<usize> = seq.iter().map(|&id| id as usize).collect();
                token_map.ids_to_chars(&ids).unwrap()
            })
            .collect::<Vec<Vec<char>>>();

        println!("\nBeam decoded ID sequences: {:?}", beam_decoded_id_sequences);
        println!("Beam decoded char sequences: {:?}\n", beam_decoded_char_sequences);

        println!("Greedy decoded ID sequences: {:?}", greedy_decoded_id_sequences);
        println!("Greedy decoded char sequences: {:?}\n", greedy_decoded_char_sequences);

        // beam sums log-probs across alignments
        // greedy picks per-timestep optimums
        assert_eq!(beam_decoded_id_sequences[0], vec![0, 0]); // "aa"
        assert_eq!(greedy_decoded_id_sequences[0], vec![0, 1, 0]); // "aba"
    }

    #[test]
    fn beam_lm_integration() {
        const N: usize = 1;
        const T: usize = 4;
        const V: usize = VOCAB_SIZE;

        const HI: f32 = 8.0; // big logit for target token
        const LO: f32 = 0.0; // baseline for rest

        let token_map = TokenMap::new(VOCAB);
        let device = Default::default();

        let (a_id, e_id, h_id, t_id, blank_id) = (0, 4, 7, 19, BLANK_ID); // "a", "e", "h", "t", "_"

        // dummy logits for 1 sample, 4 timesteps, V vocab tokens (manually biased high blanks)
        let logits: [[[f32; V]; T]; N] = [[
            one_hot_logits(t_id, HI, LO),                            // t = 0: "t"
            one_hot_logits(h_id, HI, LO),                            // t = 1: "h"
            multi_hot_logits(&[e_id, a_id], &[HI, HI], LO),     // t = 2: "e"/"a" tie
            one_hot_logits(blank_id, HI, LO),                        // t = 3: "_"
        ]];
        let logits = Tensor::<B, 3>::from_data(logits, &device);

        let ngram_lm_path = Path::new(&Context::new().rust_root)
            .join("models")
            .join("ngram_lm.bin");

        let ngram_lm = NgramConfig::new()
            .with_n(3)
            .with_vocab_size(VOCAB_SIZE)
            .with_path(ngram_lm_path.to_str().map(|s| s.to_string()));

        let lm = LanguageModelConfig::Ngram(ngram_lm); // LM wrapper

        let beam_decoder = CtcDecoderConfig::new()
            .with_search_type(CtcDecodeType::BeamSearch)
            .with_beam_width(10)
            .with_blank_id(blank_id)
            .with_lm(Some(lm))
            .with_lm_alpha(2.0)
            .with_lm_beta(1.5)
            .init();

        let beam_decoded_id_sequences = beam_decoder.forward(logits.clone());
        let beam_decoded_char_sequences = beam_decoded_id_sequences
            .iter()
            .map(|seq| {
                let ids: Vec<usize> = seq.iter().map(|&id| id as usize).collect();
                token_map.ids_to_chars(&ids).unwrap()
            })
            .collect::<Vec<Vec<char>>>();

        println!("\nBeam decoded id sequences: {:?}", beam_decoded_id_sequences);
        println!("Beam decoded char sequences: {:?}\n", beam_decoded_char_sequences);

        assert_eq!(beam_decoded_id_sequences[0], vec![19, 7, 4]); // expected "the"
    }

    #[test]
    fn lm_overrules_vsrm_logits() {
        const N: usize = 1;
        const T: usize = 6;
        const V: usize = VOCAB_SIZE;

        const HI: f32 = 8.0; // big logit for target token
        const LO: f32 = 0.0; // baseline for rest

        let token_map = TokenMap::new(VOCAB);
        let device = Default::default();

        let (b_id, e_id, m_id, o_id, r_id, y_id, blank_id) = (1, 4, 12, 14, 17, 24, BLANK_ID);

        let logits: [[[f32; V]; T]; N] = [[
            one_hot_logits(m_id, HI, LO),                            // t = 0: "m"
            one_hot_logits(e_id, HI, LO),                            // t = 1: "e"
            one_hot_logits(m_id, HI, LO),                            // t = 2: "m"
            multi_hot_logits(&[o_id, b_id], &[6.0, 7.0], LO),   // t = 3: "o"/"b" tie
            one_hot_logits(r_id, HI, LO),                            // t = 4: "r"
            one_hot_logits(y_id, HI, LO),                            // t = 5: "y"
        ]];
        let logits = Tensor::<B, 3>::from_data(logits, &device);

        let ngram_lm_path = Path::new(&Context::new().rust_root)
            .join("models")
            .join("ngram_lm.bin");

        let ngram_lm = NgramConfig::new()
            .with_n(3)
            .with_vocab_size(VOCAB_SIZE)
            .with_path(ngram_lm_path.to_str().map(|s| s.to_string()));

        let lm = LanguageModelConfig::Ngram(ngram_lm); // LM wrapper

        // native beam search decoder without LM
        let native_beam_decoder = CtcDecoderConfig::new()
            .with_search_type(CtcDecodeType::BeamSearch)
            .with_beam_width(10)
            .with_blank_id(blank_id)
            .init();

        // hybrid beam search decoder with LM
        let hybrid_beam_decoder = CtcDecoderConfig::new()
            .with_search_type(CtcDecodeType::BeamSearch)
            .with_beam_width(10)
            .with_blank_id(blank_id)
            .with_lm(Some(lm))
            .with_lm_alpha(2.0)
            .with_lm_beta(2.0)
            .init();

        let native_beam_decoded_id_sequences = native_beam_decoder.forward(logits.clone());
        let hybrid_beam_decoded_id_sequences = hybrid_beam_decoder.forward(logits.clone());

        let native_beam_decoded_char_sequences = native_beam_decoded_id_sequences
            .iter()
            .map(|seq| {
                let ids: Vec<usize> = seq.iter().map(|&id| id as usize).collect();
                token_map.ids_to_chars(&ids).unwrap()
            })
            .collect::<Vec<Vec<char>>>();
        let hybrid_beam_decoded_char_sequences = hybrid_beam_decoded_id_sequences
            .iter()
            .map(|seq| {
                let ids: Vec<usize> = seq.iter().map(|&id| id as usize).collect();
                token_map.ids_to_chars(&ids).unwrap()
            })
            .collect::<Vec<Vec<char>>>();

        // with LM, "memory" should be preferred over "membry"
        println!("\nNative Beam decoded ID sequences: {:?}", native_beam_decoded_id_sequences);
        println!("Native Beam decoded char sequences: {:?}\n", native_beam_decoded_char_sequences);

        println!("Hybrid decoded ID sequences: {:?}", hybrid_beam_decoded_id_sequences);
        println!("Hybrid decoded char sequences: {:?}\n", hybrid_beam_decoded_char_sequences);

        assert_eq!(hybrid_beam_decoded_id_sequences[0], vec![12, 4, 12, 14, 17, 24]); // expected "memory"
    }
}
