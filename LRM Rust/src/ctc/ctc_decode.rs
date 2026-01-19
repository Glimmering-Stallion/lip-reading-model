// Connectionist Temporal Classification (CTC) decoder implementation

// if using prefix beam search as the decoder, expect O(DWB * log(WB)) complexity
// where D = depth (timesteps), W = beam width (# of kept hypotheses), B = branch factor (vocab size)



// custom imports
use crate::ctc::lm::{LanguageModel, LanguageModelConfig};
use crate::utils::{log_sum_exp_2_scalar, log_sum_exp_3_scalar};

// imports
use burn::{
    config::Config,
    tensor::{activation::log_softmax, backend::Backend, Tensor},
};
use std::{
    collections::HashMap,
};



struct BeamPrefix {
    sequence: Vec<usize>,       // sequence of symbol ids in vocab
    log_prob_blank: f32,        // log-prob of prefix ending in blank
    log_prob_non_blank: f32,    // log-prob of prefix ending in non-blank
    log_prob_lm: f32,           // log-prob from language model (for LM fusion)
}



impl BeamPrefix {
    fn score(&self, alpha: f32, beta: f32) -> f32 {
        let vsrm_score = log_sum_exp_2_scalar(self.log_prob_blank, self.log_prob_non_blank);
        let lm_score = self.log_prob_lm;
        let length_reward = self.sequence.len() as f32;

        // total score
        vsrm_score + (alpha * lm_score) + (beta * length_reward)  // shallow LM fusion
    }
    fn last_char(&self) -> Option<usize> { self.sequence.last().copied() }
}



#[derive(Config, Debug)]
pub enum CtcDecodeType {
    GreedySearch,
    BeamSearch,
}



#[derive(Config, Debug)]
pub struct CtcDecoderConfig {
    #[config(default = "0")]
    pub blank_id: usize, // id of blank token in vocab

    #[config(default = "CtcDecodeType::GreedySearch")]
    pub search_type: CtcDecodeType, // search type to use within CTC decoder (greedy/beam)

    // rest are beam search params (ignored for greedy)

    #[config(default = "5")]
    pub beam_width: usize, // beam width for beam search

    #[config(default = "None")]
    pub lm: Option<LanguageModelConfig>, // optional language model to supplement beam search
    
    #[config(default = 0.5)] // typically between [0.0, 2.0]
    pub lm_alpha: f32, // weight of language model score when combining with acoustic model score
    
    #[config(default = 1.5)] // typically between [0.0, 10.0]
    pub lm_beta: f32, // length normalization factor for beam search (to avoid short sequence bias)
}



impl CtcDecoderConfig {
    pub fn init(&self) -> CtcDecoder {
        CtcDecoder {
            blank_id: self.blank_id,
            search_type: self.search_type.clone(),
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



impl CtcDecoder {
    /// apply CTC decode for batch of samples
    /// inputs assumed to be padded to max length in batch
    /// params:
    /// - inputs: [N, T, Vocab] logits from model
    /// returns: sequences of predicted symbol ids (collapsed paths) for each sample in batch [N, L]
    pub fn forward<B: Backend>(
        &self,
        inputs: Tensor<B, 3>,
    ) -> Vec<Vec<i64>> {
        match &self.search_type {
            CtcDecodeType::GreedySearch => self.greedy_search_decode(inputs),
            CtcDecodeType::BeamSearch => self.beam_search_decode(inputs),
        }
    }

    /// greedy search decode for batch of samples
    /// inputs assumed to be padded to max length in batch
    /// params:
    /// - inputs: [N, T, Vocab] logits from model
    /// returns: sequences of predicted symbol ids (collapsed paths) for each sample in batch [N, L]
    fn greedy_search_decode<B: Backend>(
        &self,
        inputs: Tensor<B, 3>
    ) -> Vec<Vec<i64>> {
        // grab most probable symbol id from vocab distribution (dim 2), per frame (dim 1), per sample (dim 0)
        let argmax_ids = inputs.clone().argmax(2); // [N, T]
        let argmax_ids = argmax_ids.to_data().convert::<i64>().to_vec().unwrap();
        let (n, t) = (inputs.clone().dims()[0], inputs.dims()[1]);

        (0..n)
            .map(|sample| {
                let sequence = &argmax_ids[(sample * t)..((sample + 1) * t)];
                collapse_path(sequence, self.blank_id as i64)
            })
            .collect()
    }

    /// beam search decode for batch of samples
    /// inputs assumed to be padded to max length in batch
    /// params:
    /// - inputs: [N, T, Vocab] logits from model
    /// returns: sequences of predicted symbol ids (collapsed paths) for each sample in batch [N, L]
    fn beam_search_decode<B: Backend>(
        &self,
        inputs: Tensor<B, 3>
    ) -> Vec<Vec<i64>> {
        let log_probs = log_softmax(inputs, 2); // [N, T, V]
        let (n, t, vocab_size) = (log_probs.clone().dims()[0], log_probs.clone().dims()[1], log_probs.clone().dims()[2]);
        let mut top_seq_ids = Vec::with_capacity(n);

        // loop over samples in batch
        for sample in 0..n {
            let sample_log_probs = log_probs.clone().slice([sample..(sample + 1), 0..t, 0..vocab_size]).squeeze(0);
            top_seq_ids.push(self.per_sample_decode(sample_log_probs));
        }

        top_seq_ids
    }

    /// beam search decode for single sample
    /// decoding obtained via: logits --> log softmax --> prefix beam search
    /// params:
    /// - log_probs: log-probabilities for each vocab symbol per-timestep given by model [T, Vocab]
    /// returns: sequence of predicted symbol ids (collapsed path) [L]
    fn per_sample_decode<B: Backend>(
        &self,
        log_probs: Tensor<B, 2>,
    ) -> Vec<i64> {
        let blank = self.blank_id;
        // let sentinel_value = -1e30;
        let sentinel_value = f32::NEG_INFINITY;
        let (timesteps, vocab_size) = (log_probs.dims()[0], log_probs.dims()[1]);

        // t = -1 (base case)
        // initialize beam with empty prefix (starts with size 1 and grows to beam_width)
        let mut prefixes = vec![
            BeamPrefix {
                sequence: Vec::new(),
                log_prob_blank: 0.0, // log(1)
                log_prob_non_blank: sentinel_value, // log(0)
                log_prob_lm: 0.0,
            }
        ];

        // 0 ≤ t ≤ T - 1 (recurrence case)
        for t in 0..timesteps {
            // reset buffer
            // maps sequence of symbol ids to (log_prob_blank, log_prob_non_blank, log_prob_lm)
            let mut next_prefixes: HashMap<Vec<usize>, (f32, f32, f32)> = HashMap::new();

            // grab chunk of log-probs of each symbol at current timestep (given by model)
            let log_probs_t: Vec<f32> = log_probs.clone().slice([t..(t + 1), 0..vocab_size])
                .squeeze::<1>(0)
                .to_data().convert::<f32>().to_vec().unwrap();

            for prefix in prefixes.into_iter() {
                // build key for next prefix sequence hypothesis
                let sequence_key = prefix.sequence.clone();

                // compute succeeding log-prob of extending prefix with blank at current timestep
                // can extend from either path ending with blank or non-blank
                let ext_log_prob = log_sum_exp_2_scalar(
                    prefix.log_prob_blank + log_probs_t[blank],
                    prefix.log_prob_non_blank + log_probs_t[blank]
                );

                // update log-prob for current prefix path (ending with blanks)
                // accumulate total log-prob of prefix ending with blank
                let entry = next_prefixes
                    .entry(sequence_key.clone())
                    .or_insert((sentinel_value, sentinel_value, prefix.log_prob_lm));
                entry.0 = log_sum_exp_2_scalar(entry.0, ext_log_prob);

                // extend with non-blank characters (v is the candidate symbol id in vocab)
                for v in 0..vocab_size {
                    if v == blank { continue; }

                    // case A (skip):       same char, previous path ended with non-blank
                    // case B (stretch):    same char, previous path ended with blank
                    // case C (append):     diff char, previous path ended with either blank or non-blank
                    match prefix.last_char() {
                        // if v equals last char in prefix path
                        // can only extend from path ending with blank log-prob
                        Some(last_char) if last_char == v => {

                            // --------------- (A) ---------------
                            // - non-blank source state
                            // - cont. with same prefix sequence
                            // - don't append duplicate char

                            // build key for next prefix sequence hypothesis
                            let sequence_key_a = prefix.sequence.clone();

                            // compute succeeding log-prob of extending prefix with same symbol at current timestep
                            let ext_log_prob_a = prefix.log_prob_non_blank + log_probs_t[v];

                            // update log-prob for current prefix path (ending with non-blanks)
                            // accumulate total log-prob of prefix ending with non-blank
                            let entry_a = next_prefixes
                                .entry(sequence_key_a.clone())
                                .or_insert((sentinel_value, sentinel_value, prefix.log_prob_lm));
                            entry_a.1 = log_sum_exp_2_scalar(entry_a.1, ext_log_prob_a);

                            // --------------- (B) ---------------
                            // - blank source state
                            // - start new prefix sequence
                            // - do append duplicate char

                            // build key for next prefix sequence hypothesis
                            let mut sequence_key_b = prefix.sequence.clone();
                            sequence_key_b.push(v); // append duplicate char

                            // compute succeeding log-prob of extending prefix with same symbol at current timestep
                            let mut ext_log_prob_b = prefix.log_prob_blank + log_probs_t[v];

                            // optional language model score adjustment (applied for extending with duplicate char)
                            let mut new_lm_log_prob = prefix.log_prob_lm;
                            if let Some(lm) = self.lm.as_ref() { new_lm_log_prob += lm.next_log_prob(&prefix.sequence, v); }

                            // update log-prob for current prefix path (ending with non-blanks)
                            // accumulate total log-prob of prefix ending with non-blank
                            let entry_b = next_prefixes
                                .entry(sequence_key_b.clone())
                                .or_insert((sentinel_value, sentinel_value, new_lm_log_prob));
                            entry_b.1 = log_sum_exp_2_scalar(entry_b.1, ext_log_prob_b);
                        }

                        // if v differs from last char in prefix path
                        // can extend from either path ending with blank or non-blank log-prob
                        _ => {

                            // --------------- (C) ---------------
                            // - new prefix sequence
                            // - append char from either state

                            // build key for next prefix sequence hypothesis
                            let mut sequence_key_c = prefix.sequence.clone();
                            sequence_key_c.push(v);

                            // compute succeeding log-prob of extending prefix with different symbol at current timestep
                            let mut ext_log_prob_c = log_sum_exp_2_scalar(
                                prefix.log_prob_blank,
                                prefix.log_prob_non_blank,
                            ) + log_probs_t[v];

                            // optional language model score adjustment (applied for extending with different char)
                            let mut new_lm_log_prob = prefix.log_prob_lm;
                            if let Some(lm) = self.lm.as_ref() { new_lm_log_prob += lm.next_log_prob(&prefix.sequence, v); }

                            // update log-prob for current prefix path (ending with non-blanks)
                            // accumulate total log-prob of prefix ending with non-blank
                            let entry_c = next_prefixes
                                .entry(sequence_key_c.clone())
                                .or_insert((sentinel_value, sentinel_value, new_lm_log_prob));
                            entry_c.1 = log_sum_exp_2_scalar(entry_c.1, ext_log_prob_c);
                        }
                    }
                }
            }

            // convert HashMap to Vec<BeamPrefix> for sorting
            let mut next_prefixes_vec: Vec<BeamPrefix> = next_prefixes
                .into_iter()
                .map(|(sequence, (log_prob_blank, log_prob_non_blank, log_prob_lm))| BeamPrefix {
                    sequence,
                    log_prob_blank,
                    log_prob_non_blank,
                    log_prob_lm,
                }).collect();

            // sort by score (descending)
            next_prefixes_vec.sort_by(|a, b|
                b.score(self.lm_alpha, self.lm_beta)
                    .partial_cmp(&a.score(self.lm_alpha, self.lm_beta))
                    .unwrap()
            );

            // keep top beam_width prefixes
            prefixes = next_prefixes_vec.into_iter().take(self.beam_width).collect();
        }

        // return highest scoring prefix sequence
        let best_prefix = prefixes.into_iter()
            .max_by(|a, b|
                a.score(self.lm_alpha, self.lm_beta)
                    .partial_cmp(&b.score(self.lm_alpha, self.lm_beta))
                    .unwrap()
            )
            .unwrap();

        // convert usize ids to i64 ids
        best_prefix.sequence.into_iter().map(|u| u as i64).collect()
    }
}



/// helper function for greedy search to collapse a path by removing consecutive duplicates between blanks
/// blank tokens are then removed subsequently
/// this helps to align time-based predictions with text-based targets
/// params:
/// - path: sequence of symbol int IDs deemed most probable by model (with blanks and duplicates) [T]
/// - blank_id: id of blank token in vocab
/// returns: collapsed sequence of symbol int IDs (without blanks or duplicates) [L]
pub fn collapse_path<T: PartialEq + Copy>(path: &[T], blank_id: T) -> Vec<T> {
    let mut collapsed_path = Vec::new();
    let mut prev = None;

    // dupe removal
    for &token in path {
        if Some(token) != prev { collapsed_path.push(token); }
        prev = Some(token);
    }

    // blank removal
    collapsed_path.into_iter().filter(|&token| token != blank_id).collect()
}



// tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use std::{
        array,
        env,
        path::Path,
    };
    use burn::{
        backend::ndarray::NdArray,
        tensor::{backend::Backend, Distribution, Tensor},
    };

    type B = NdArray<f32>;

    use crate::prelude::*;
    use crate::ctc::lm::LanguageModelConfig;

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

        let path = vec![7, 4, 4, blank_id, 11, blank_id, 11, 11, 14, 14, 28];
        let collapsed_path = collapse_path(&path, blank_id);

        println!("\nOriginal path ids: {:?}", path);
        println!("Collapsed path ids: {:?}", collapsed_path);
        println!("Original path chars: {:?}", token_map.ids_to_chars(path.clone()));
        println!("Collapsed path chars: {:?}\n", token_map.ids_to_chars(collapsed_path.clone()));
        assert_eq!(collapsed_path, vec![7, 4, 11, 11, 14, 28]);
    }

    #[test]
    fn test_greedy_search_decode_random() {
        let vocab = VOCAB;
        let vocab_size = VOCAB_SIZE;
        let blank_id = BLANK_ID;
        let token_map = TokenMap::new(vocab);
        let device = Default::default();
        let dist = Distribution::Uniform(0.0, 1.0);

        // dummy logits for 2 samples, 5 timesteps, 41 vocab symbols
        let logits = Tensor::<B, 3>::random([2, 5, vocab_size], dist, &device);

        let decoder = CtcDecoderConfig::new()
            .with_search_type(CtcDecodeType::GreedySearch)
            .with_blank_id(blank_id)
            .init();

        let decoded_id_sequences = decoder.forward(logits);
        let decoded_char_sequences = decoded_id_sequences
            .iter()
            .map(|seq| {
                token_map
                    .ids_to_chars(seq.iter().map(|&id| id as usize).collect())
                    .unwrap()
            })
            .collect::<Vec<Vec<char>>>();
        
        println!("Decoded id sequences: {:?}", decoded_id_sequences);
        println!("Decoded char sequences: {:?}", decoded_char_sequences);

        assert_eq!(decoded_id_sequences.len(), 2);
        assert!(!decoded_id_sequences[0].contains(&(blank_id as i64))); // no blanks expected
        assert!(!decoded_id_sequences[1].contains(&(blank_id as i64))); // no blanks expected
    }

    #[test]
    fn test_greedy_search_decode_fixed() {
        const V: usize = VOCAB_SIZE;
        const N: usize = 2;
        const T: usize = 11;

        const HI: f32 = 8.0; // big logit for target symbol
        const LO: f32 = 0.0; // baseline for rest

        let blank_id = BLANK_ID;
        let token_map = TokenMap::new(VOCAB);
        let device = Default::default();

        let ids_1 = [7, 4, 4, 40, 11, 40, 11, 11, 14, 14, 28]; // "h e e _ l _ l l o o !"
        let ids_2 = [22, 22, 14, 40, 17, 17, 40, 11, 11, 3, 28]; // "w w o _ r r _ l l d !"

        let chars_1 = token_map.ids_to_chars(ids_1.to_vec()).unwrap();
        let chars_2 = token_map.ids_to_chars(ids_2.to_vec()).unwrap();

        // dummy logits for 2 samples, 11 timesteps, 41 vocab symbols (manually biased high blanks)
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
                token_map
                    .ids_to_chars(seq.iter().map(|&id| id as usize).collect())
                    .unwrap()
            })
            .collect::<Vec<Vec<char>>>();

        println!("\nOriginal sample id sequences: {:?}, {:?}", ids_1, ids_2);
        println!("Original sample char sequences: {:?}, {:?}\n", chars_1, chars_2);
        
        println!("Decoded id sequences: {:?}", decoded_id_sequences);
        println!("Decoded char sequences: {:?}", decoded_char_sequences);

        assert_eq!(decoded_id_sequences.len(), 2);
        assert_eq!(decoded_id_sequences[0], vec![7, 4, 11, 11, 14, 28]); // expected collapsed path for sample 1
        assert_eq!(decoded_id_sequences[1], vec![22, 14, 17, 11, 3, 28]); // expected collapsed path for sample 2
    }

    #[test]
    fn test_beam_search_decode_random() {
        let vocab = VOCAB;
        let vocab_size = VOCAB_SIZE;
        let blank_id = BLANK_ID;
        let token_map = TokenMap::new(vocab);
        let device = Default::default();
        let dist = Distribution::Uniform(0.0, 1.0);

        // dummy logits for 2 samples, 5 timesteps, 41 vocab symbols
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
                token_map
                    .ids_to_chars(seq.iter().map(|&id| id as usize).collect())
                    .unwrap()
            })
            .collect::<Vec<Vec<char>>>();
        
        println!("Decoded id sequences: {:?}", decoded_id_sequences);
        println!("Decoded char sequences: {:?}", decoded_char_sequences);

        assert_eq!(decoded_id_sequences.len(), 2);
        assert!(!decoded_id_sequences[0].contains(&(blank_id as i64))); // no blanks expected
        assert!(!decoded_id_sequences[1].contains(&(blank_id as i64))); // no blanks expected
    }

    #[test]
    fn test_beam_search_decode_fixed() {
        const V: usize = VOCAB_SIZE;
        const N: usize = 2;
        const T: usize = 11;

        const HI: f32 = 8.0; // big logit for target symbol
        const LO: f32 = 0.0; // baseline for rest

        let blank_id = BLANK_ID;
        let token_map = TokenMap::new(VOCAB);
        let device = Default::default();

        let ids_1 = [7, 4, 4, 40, 11, 40, 11, 11, 14, 14, 28];    // "h e e _ l _ l l o o !"
        let ids_2 = [22, 22, 14, 40, 17, 17, 40, 11, 11, 3, 28];  // "w w o _ r r _ l l d !"

        let chars_1 = token_map.ids_to_chars(ids_1.to_vec()).unwrap();
        let chars_2 = token_map.ids_to_chars(ids_2.to_vec()).unwrap();

        // dummy logits for 2 samples, 11 timesteps, 41 vocab symbols (manually biased high blanks)
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
                token_map
                    .ids_to_chars(seq.iter().map(|&id| id as usize).collect())
                    .unwrap()
            })
            .collect::<Vec<Vec<char>>>();

        println!("\nOriginal sample id sequences: {:?}, {:?}", ids_1, ids_2);
        println!("Original sample char sequences: {:?}, {:?}\n", chars_1, chars_2);
        
        println!("Decoded id sequences: {:?}", decoded_id_sequences);
        println!("Decoded char sequences: {:?}", decoded_char_sequences);

        assert_eq!(decoded_id_sequences.len(), 2);
        assert_eq!(decoded_id_sequences[0], vec![7, 4, 11, 11, 14, 28]);  // expected collapsed path for sample 1
        assert_eq!(decoded_id_sequences[1], vec![22, 14, 17, 11, 3, 28]); // expected collapsed path for sample 2
    }

    #[test]
    fn test_beam_eq_greedy_when_beam_width_1() {
        const V: usize = VOCAB_SIZE;
        const N: usize = 2;
        const T: usize = 11;

        const HI: f32 = 8.0; // big logit for target symbol
        const LO: f32 = 0.0; // baseline for rest

        let blank_id = BLANK_ID;
        let token_map = TokenMap::new(VOCAB);
        let device = Default::default();

        let ids_1 = [12, 0, 0, 40, 40, 19, 2, 2, 7, 7, 7];     // "m a a _ _ t c c h h h "
        let ids_2 = [18, 20, 2, 40, 2, 4, 4, 18, 40, 18, 18];  // "s u c _ c e e s _ s s"

        let chars_1 = token_map.ids_to_chars(ids_1.to_vec()).unwrap();
        let chars_2 = token_map.ids_to_chars(ids_2.to_vec()).unwrap();

        // dummy logits for 2 samples, 11 timesteps, 41 vocab symbols (manually biased high blanks)
        let logits: [[[f32; V]; T]; N] = [
            array::from_fn(|t| one_hot_logits::<V>(ids_1[t], HI, LO)),
            array::from_fn(|t| one_hot_logits::<V>(ids_2[t], HI, LO)),
        ];
        let logits = Tensor::<B, 3>::from_data(logits, &device);

        let beam_decoder = CtcDecoderConfig::new()
            .with_search_type(CtcDecodeType::BeamSearch)
            .with_beam_width(1)
            .with_blank_id(blank_id)
            .with_lm_alpha(0.0)
            .with_lm_beta(0.0)
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
                token_map
                    .ids_to_chars(seq.iter().map(|&id| id as usize).collect())
                    .unwrap()
            })
            .collect::<Vec<Vec<char>>>();
        let greedy_decoded_char_sequences = greedy_decoded_id_sequences
            .iter()
            .map(|seq| {
                token_map
                    .ids_to_chars(seq.iter().map(|&id| id as usize).collect())
                    .unwrap()
            })
            .collect::<Vec<Vec<char>>>();

        println!("\nOriginal sample id sequences: {:?}, {:?}", ids_1, ids_2);
        println!("Original sample char sequences: {:?}, {:?}\n", chars_1, chars_2);

        println!("\nBeam decoded id sequences: {:?}", beam_decoded_id_sequences);
        println!("Beam decoded char sequences: {:?}\n", beam_decoded_char_sequences);

        println!("\nGreedy decoded id sequences: {:?}", greedy_decoded_id_sequences);
        println!("Greedy decoded char sequences: {:?}\n", greedy_decoded_char_sequences);

        // with beam width 1, no LM, with logits high-contrast and unambiguous, beam output should match greedy
        assert_eq!(beam_decoded_id_sequences, greedy_decoded_id_sequences);
    }

    #[test]
    fn test_decode_preserves_duplicates_when_blanks_present() {
        const V: usize = VOCAB_SIZE;
        const N: usize = 5;
        const T: usize = 14;

        const HI: f32 = 8.0; // big logit for target symbol
        const LO: f32 = 0.0; // baseline for rest

        let blank_id = BLANK_ID;
        let token_map = TokenMap::new(VOCAB);
        let device = Default::default();

        let ids_1 = [6, 14, 40, 14, 3, 39, 6, 14, 14, 3, 40, 40, 40, 40];       // "g o _ o d   g o o d - - -"
        let ids_2 = [11, 0, 19, 40, 19, 4, 17, 39, 11, 0, 19, 19, 4, 17];       // "l a t _ t e r   l a t t e r"
        let ids_3 = [3, 8, 13, 40, 13, 4, 17, 39, 3, 8, 13, 13, 4, 17];         // "d i n _ n e r   d i n n e r"
        let ids_4 = [18, 20, 15, 40, 15, 4, 17, 39, 18, 20, 15, 15, 4, 17];     // "s u p _ p e r   s u p p e r"
        let ids_5 = [7, 14, 15, 40, 15, 4, 3, 39, 7, 14, 15, 15, 4, 3];         // "h o p _ p e d   h o p p e d"

        let chars_1 = token_map.ids_to_chars(ids_1.to_vec()).unwrap();
        let chars_2 = token_map.ids_to_chars(ids_2.to_vec()).unwrap();
        let chars_3 = token_map.ids_to_chars(ids_3.to_vec()).unwrap();
        let chars_4 = token_map.ids_to_chars(ids_4.to_vec()).unwrap();
        let chars_5 = token_map.ids_to_chars(ids_5.to_vec()).unwrap();

        let expected_outputs = [
            vec![6, 14, 14, 3, 39, 6, 14, 3],                     // "g o o d   g o d"
            vec![11, 0, 19, 19, 4, 17, 39, 11, 0, 19, 4, 17],     // "l a t t e r   l a t e r"
            vec![3, 8, 13, 13, 4, 17, 39, 3, 8, 13, 4, 17],       // "d i n n e r   d i n e r"
            vec![18, 20, 15, 15, 4, 17, 39, 18, 20, 15, 4, 17],   // "s u p p e r   s u p e r"
            vec![7, 14, 15, 15, 4, 3, 39, 7, 14, 15, 4, 3],       // "h o p p e d   h o p e d"
        ];

        // dummy logits for 5 samples, 14 timesteps, 41 vocab symbols (manually biased high blanks)
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
            .with_lm_alpha(0.0)
            .with_lm_beta(0.0)
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
                token_map
                    .ids_to_chars(seq.iter().map(|&id| id as usize).collect())
                    .unwrap()
            })
            .collect::<Vec<Vec<char>>>();
        let greedy_decoded_char_sequences = greedy_decoded_id_sequences
            .iter()
            .map(|seq| {
                token_map
                    .ids_to_chars(seq.iter().map(|&id| id as usize).collect())
                    .unwrap()
            })
            .collect::<Vec<Vec<char>>>();

        println!("\nOriginal sample id sequences:\n{:?}\n{:?}\n{:?}\n{:?}\n{:?}\n", ids_1, ids_2, ids_3, ids_4, ids_5);
        println!("\nOriginal sample char sequences:\n{:?}\n{:?}\n{:?}\n{:?}\n{:?}\n", chars_1, chars_2, chars_3, chars_4, chars_5);
        
        println!("\nBeam decoded id sequences:");
        for seq in &beam_decoded_id_sequences { println!("  {:?}", seq); }
        println!();
        println!("Beam decoded char sequences:");
        for seq in &beam_decoded_char_sequences { println!("  {}", seq.iter().collect::<String>()); }
        println!();
        
        println!("\nGreedy decoded id sequences:");
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
    fn test_beam_outperforms_greedy_on_global_optimum() {
        const V: usize = 3;
        const N: usize = 1;
        const T: usize = 5;

        let vocab = "ab_";
        let blank_id = 2;
        let token_map = TokenMap::new(vocab);
        let device = Default::default();

        // at t = 2, greedy picks 'b', but globally 'aa' is more probable
        let logits: [[[f64; V]; T]; N] = [[
            [-0.02, -3.00, -3.00], // t = 0
            [-1.02, -3.00, -0.01], // t = 1
            [-0.22, -0.21, -0.30], // t = 2
            [-0.02, -3.00, -0.05], // t = 3
            [-0.02, -3.00, -0.03], // t = 4
        ]];
        let logits = Tensor::<B, 3>::from_data(logits, &device);

        let beam_decoder = CtcDecoderConfig::new()
            .with_search_type(CtcDecodeType::BeamSearch)
            .with_beam_width(3)
            .with_blank_id(blank_id)
            .with_lm_alpha(0.0)
            .with_lm_beta(0.0)
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
                token_map
                    .ids_to_chars(seq.iter().map(|&id| id as usize).collect())
                    .unwrap()
            })
            .collect::<Vec<Vec<char>>>();
        let greedy_decoded_char_sequences = greedy_decoded_id_sequences
            .iter()
            .map(|seq| {
                token_map
                    .ids_to_chars(seq.iter().map(|&id| id as usize).collect())
                    .unwrap()
            })
            .collect::<Vec<Vec<char>>>();

        println!("\nBeam decoded id sequences: {:?}", beam_decoded_id_sequences);
        println!("Beam decoded char sequences: {:?}\n", beam_decoded_char_sequences);

        println!("\nGreedy decoded id sequences: {:?}", greedy_decoded_id_sequences);
        println!("Greedy decoded char sequences: {:?}\n", greedy_decoded_char_sequences);

        // beam sums log-probs across alignments
        // greedy picks per-frame optimums
        assert_eq!(beam_decoded_id_sequences[0], vec![0, 0]); // "aa"
        assert_eq!(greedy_decoded_id_sequences[0], vec![0, 1, 0]); // "aba"
    }

    #[test]
    fn test_beam_lm_integration() {
        const V: usize = VOCAB_SIZE;
        const N: usize = 1;
        const T: usize = 4;

        const HI: f32 = 8.0; // big logit for target symbol
        const LO: f32 = 0.0; // baseline for rest

        let token_map = TokenMap::new(VOCAB);
        let device = Default::default();

        let (a_id, e_id, h_id, t_id, blank_id) = (0, 4, 7, 19, BLANK_ID); // "a", "e", "h", "t", "_"

        // dummy logits for 1 sample, 4 timesteps, 41 vocab symbols (manually biased high blanks)
        let logits: [[[f32; V]; T]; N] = [[
            one_hot_logits(t_id, HI, LO),                            // t = 0: "t"
            one_hot_logits(h_id, HI, LO),                            // t = 1: "h"
            multi_hot_logits(&[e_id, a_id], &[HI, HI], LO),     // t = 2: "e"/"a" tie
            one_hot_logits(blank_id, HI, LO),                        // t = 3: "_"
        ]];
        let logits = Tensor::<B, 3>::from_data(logits, &device);

        let rust_root = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
        let ngram_lm_path = Path::new(&rust_root)
            .join("models")
            .join("ngram_lm.bin");

        let ngram_lm = NgramLMConfig::new()
            .with_n(3)
            .with_vocab_size(VOCAB_SIZE)
            .with_path(ngram_lm_path.to_str().map(|s| s.to_string()));

        let lm = LanguageModelConfig::NgramLM(ngram_lm); // lm wrapper

        let beam_decoder = CtcDecoderConfig::new()
            .with_search_type(CtcDecodeType::BeamSearch)
            .with_beam_width(10)
            .with_blank_id(blank_id)
            .with_lm(Some(lm))
            .with_lm_alpha(2.0)
            .init();

        let beam_decoded_id_sequences = beam_decoder.forward(logits.clone());
        let beam_decoded_char_sequences = beam_decoded_id_sequences
            .iter()
            .map(|seq| {
                token_map
                    .ids_to_chars(seq.iter().map(|&id| id as usize).collect())
                    .unwrap()
            })
            .collect::<Vec<Vec<char>>>();

        println!("\nBeam decoded id sequences: {:?}", beam_decoded_id_sequences);
        println!("Beam decoded char sequences: {:?}\n", beam_decoded_char_sequences);

        assert_eq!(beam_decoded_id_sequences[0], vec![19, 7, 4]); // expected "the"
    }

    #[test]
    fn test_lm_overrules_vsrm_logits() {
        const V: usize = VOCAB_SIZE;
        const N: usize = 1;
        const T: usize = 6;

        const HI: f32 = 8.0; // big logit for target symbol
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

        let rust_root = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
        let ngram_lm_path = Path::new(&rust_root)
            .join("models")
            .join("ngram_lm.bin");

        let ngram_lm = NgramLMConfig::new()
            .with_n(3)
            .with_vocab_size(VOCAB_SIZE)
            .with_path(ngram_lm_path.to_str().map(|s| s.to_string()));

        let lm = LanguageModelConfig::NgramLM(ngram_lm); // lm wrapper

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
                token_map
                    .ids_to_chars(seq.iter().map(|&id| id as usize).collect())
                    .unwrap()
            })
            .collect::<Vec<Vec<char>>>();
        let hybrid_beam_decoded_char_sequences = hybrid_beam_decoded_id_sequences
            .iter()
            .map(|seq| {
                token_map
                    .ids_to_chars(seq.iter().map(|&id| id as usize).collect())
                    .unwrap()
            })
            .collect::<Vec<Vec<char>>>();

        // with LM, "memory" should be preferred over "membry"
        println!("\nNative Beam decoded id sequences: {:?}", native_beam_decoded_id_sequences);
        println!("Native Beam decoded char sequences: {:?}", native_beam_decoded_char_sequences);

        println!("\nHybrid decoded id sequences: {:?}", hybrid_beam_decoded_id_sequences);
        println!("Hybrid decoded char sequences: {:?}\n", hybrid_beam_decoded_char_sequences);

        assert_eq!(hybrid_beam_decoded_id_sequences[0], vec![12, 4, 12, 14, 17, 24]); // expected "memory"
    }
}