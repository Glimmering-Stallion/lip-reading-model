//! Custom CTC-compatible Burn-adapted CER/WER evaluation metrics for VSRM.
//! 
//! Provides implementations for Character Error Rate (CER) and Word Error Rate (WER)
//! adapted to the Burn training dashboard. These metrics use Levenshtein distance to
//! compare decoded model predictions against ground-truth targets aross normalized
//! sequences.



// imports
use burn::{
    prelude::Int,
    tensor::{
        Tensor,
        backend::Backend,
    },
    train::{
        ItemLazy,
        metric::{
            Adaptor,
            Numeric,
            NumericEntry,
            LossInput,
            Metric,
            MetricMetadata,
            SerializedEntry,
        },
    },
};
use std::{
    cmp::min,
    sync::Arc,
    marker::PhantomData,
};

use crate::{ctc::ctc_decode::CtcDecoder, vocab::TokenMap};



/// raw result of a training/validation step
/// training/validation step will return this
pub struct VsrmStepOutput<B: Backend> {
    pub loss: Tensor<B, 1>,                 // [N]
    pub outputs: Tensor<B, 3>,              // [N, T, V] (logits)
    pub targets: Tensor<B, 2, Int>,         // [N, T]
    pub output_lengths: Tensor<B, 1, Int>,  // [N]
    pub target_lengths: Tensor<B, 1, Int>,  // [N]
}



/// standardized input for CTC-compatible metrics
/// custom CER/WER metrics will intake this
pub struct VsrmMetricInput<B: Backend> {
    pub loss: Tensor<B, 1>,                 // [N]
    pub inputs: Tensor<B, 3>,               // [N, T, V] (logits)
    pub targets: Tensor<B, 2, Int>,         // [N, T]
    pub input_lengths: Tensor<B, 1, Int>,   // [N]
    pub target_lengths: Tensor<B, 1, Int>,  // [N]
}



/// tracks Character Error Rate (CER) over an epoch
/// compares raw token ID sequences
#[derive(Clone)]
pub struct CtcCharErrorRate<B: Backend> {
    decoder: CtcDecoder,
    total_error: usize,     // total char mistakes made across entire val set
    total_chars: usize,     // total chars in entire val set
    _phantom: PhantomData<B>,
}



/// tracks Word Error Rate (WER) over an epoch
/// compares white-space-separated word sequences
#[derive(Clone)]
pub struct CtcWordErrorRate<B: Backend> {
    decoder: CtcDecoder,
    token_map: TokenMap,
    total_error: usize,     // total word mistakes made across entire val set
    total_words: usize,     // total words in entire val set
    _phantom: PhantomData<B>,
}



/// marks step output as ready for async processing by Burn's Metric system
impl<B: Backend> ItemLazy for VsrmStepOutput<B> {
    type ItemSync = Self;

    fn sync(self) -> Self::ItemSync {
        self
    }
}



/// translator to feed standard Burn Loss graph
impl<B: Backend> Adaptor<LossInput<B>> for VsrmStepOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}



/// translator to feed our custom CER/WEr metrics
impl<B: Backend> Adaptor<VsrmMetricInput<B>> for VsrmStepOutput<B> {
    fn adapt(&self) -> VsrmMetricInput<B> {
        VsrmMetricInput {
            loss: self.loss.clone(),
            inputs: self.outputs.clone(),
            targets: self.targets.clone(),
            input_lengths: self.output_lengths.clone(),
            target_lengths: self.target_lengths.clone(),
        }
    }
}



impl<B: Backend> CtcCharErrorRate<B> {
    pub fn new(decoder: CtcDecoder) -> Self {
        Self {
            decoder,
            total_error: 0,
            total_chars: 0,
            _phantom: PhantomData,
        }
    }
}



impl<B: Backend> CtcWordErrorRate<B> {
    pub fn new(decoder: CtcDecoder, token_map: TokenMap) -> Self {
        Self {
            decoder,
            token_map,
            total_error: 0,
            total_words: 0,
            _phantom: PhantomData,
        }
    }
}



/// required to plot CER as a numerical val in the training dashboard
impl<B: Backend> Numeric for CtcCharErrorRate<B> {
    fn value(&self) -> NumericEntry {
        if self.total_chars == 0 { NumericEntry::Value(0.0) }
        else { NumericEntry::Value(self.total_error as f64 / self.total_chars as f64) }
    }

    fn running_value(&self) -> NumericEntry {
        NumericEntry::Aggregated {
            aggregated_value:
                if self.total_chars == 0 { 0.0 }
                else { self.total_error as f64 / self.total_chars as f64 },
            count: self.total_chars,
        }
    }
}



/// required to plot WER as a numerical val in the training dashboard
impl<B: Backend> Numeric for CtcWordErrorRate<B> {
    fn value(&self) -> NumericEntry {
        if self.total_words == 0 { NumericEntry::Value(0.0) }
        else { NumericEntry::Value(self.total_error as f64 / self.total_words as f64) }
    }

    fn running_value(&self) -> NumericEntry {
        NumericEntry::Aggregated {
            aggregated_value:
                if self.total_words == 0 { 0.0 }
                else { self.total_error as f64 / self.total_words as f64 },
            count: self.total_words,
        }
    }
}



/// for real-time plotting to visualize CER progression in the training dashboard
impl<B: Backend> Metric for CtcCharErrorRate<B> {
    type Input = VsrmMetricInput<B>;

    fn name(&self) -> Arc<String> {
        let mode = self.decoder.search_type;
        Arc::new(format!("CTC Decoder Char Error Rate ({:?})", mode))
    }

    /// processes a batch of logits to compute running CER
    /// params:
    /// - input_item: container for model loss, logits, ground truth target IDs as well as logit/target lengths
    /// - _metadata: progress tracking from Burn learner
    /// returns: a SerializedEntry container of formatted CER percentage and raw float val for plotting
    fn update(&mut self, input_item: &VsrmMetricInput<B>, _metadata: &MetricMetadata) -> SerializedEntry {
        // decode logits to predicted token sequences (tensor to vec)
        let predictions = self.decoder.forward(input_item.inputs.clone())
            .into_iter()
            .map(|seq| seq.into_iter().map(|id| id as usize).collect())
            .collect::<Vec<Vec<usize>>>();

        // extract ground truth targets (tensor to vec)
        let targets = unpad_to_vec(
            input_item.targets.clone(),
            input_item.target_lengths.clone(),
        );

        debug_assert!(predictions.len() == targets.len(), "Predictions/targets batch size mismatch");

        // init per batch counts
        let mut batch_error = 0;
        let mut batch_chars = 0;

        // compare each prediction's IDs with each target's IDs in a batch
        for i in 0..predictions.len() {
            let prediction_ids = &predictions[i];
            let target_ids = &targets[i];

            if prediction_ids.is_empty() { log::warn!("Decoder produced empty prediction sequence"); }
            if target_ids.is_empty() { log::warn!("Encountered empty target sequence"); }

            let edit_distance = levenshtein(prediction_ids, target_ids);

            // accumulate batch totals
            batch_error += edit_distance;
            batch_chars += target_ids.len();
        }

        // update running totals
        self.total_error += batch_error;
        self.total_chars += batch_chars.max(1);
        let curr_cer = (self.total_error as f64 / self.total_chars as f64) * 100.0;

        // return for Burn dashboard
        SerializedEntry::new(
            format!("{:.2}%", curr_cer),
            curr_cer.to_string(),
        )
    }

    fn clear(&mut self) {
        self.total_error = 0;
        self.total_chars = 0;
    }
}



/// for real-time plotting to visualize WER progression in the training dashboard
impl<B: Backend> Metric for CtcWordErrorRate<B> {
    type Input = VsrmMetricInput<B>;

    fn name(&self) -> Arc<String> {
        let mode = self.decoder.search_type;
        Arc::new(format!("CTC Decoder Word Error Rate ({:?})", mode))
    }

    /// processes a batch of logits to compute running WER
    /// decodes sequence IDs, maps to chars with TokenMap, and splits by whitespace
    /// params:
    /// - input_item: container for model loss, logits, ground truth target IDs as well as logit/target lengths
    /// - _metadata: progress tracking from Burn learner
    /// returns: a SerializedEntry container of formatted WER percentage and raw float val for plotting
    fn update(&mut self, input_item: &VsrmMetricInput<B>, _metadata: &MetricMetadata) -> SerializedEntry {
        // decode logits to predicted token sequences (tensor to vec)
        let predictions = self.decoder.forward(input_item.inputs.clone())
            .into_iter()
            .map(|seq| seq.into_iter().map(|id| id as usize).collect())
            .collect::<Vec<Vec<usize>>>();

        // extract ground truth targets (tensor to vec)
        let targets = unpad_to_vec(
            input_item.targets.clone(),
            input_item.target_lengths.clone(),
        );

        debug_assert!(predictions.len() == targets.len(), "Predictions/targets batch size mismatch");

        // init per batch counts
        let mut batch_error = 0;
        let mut batch_words = 0;

        // compare each prediction's string(s) with each target's string(s) in a batch
        for i in 0..predictions.len() {
            // obtain ID sequence and convert to char sequence (for both prediction and target)
            let prediction_chars: Vec<char> = self.token_map
                .ids_to_chars(&predictions[i])
                .expect("Failed to convert prediction IDs to chars");
            let target_chars = self.token_map
                .ids_to_chars(&targets[i])
                .expect("Failed to convert target IDs to chars");

            if prediction_chars.is_empty() { log::warn!("Decoder produced empty prediction sequence"); }
            if target_chars.is_empty() { log::warn!("Encountered empty target sequence"); }

            // convert char sequence to a container of words (for both prediction and target)
            let prediction_words: Vec<String> = String::from_iter(prediction_chars)
                .split_whitespace()
                .map(|s| s.to_string())
                .collect();
            let target_words: Vec<String> = String::from_iter(target_chars)
                .split_whitespace()
                .map(|s| s.to_string())
                .collect();

            let edit_distance = levenshtein(&prediction_words, &target_words);

            // accumulate batch totals
            batch_error += edit_distance;
            batch_words += target_words.len();
        }

        // update running totals
        self.total_error += batch_error;
        self.total_words += batch_words.max(1);
        let curr_wer = (self.total_error as f64 / self.total_words as f64) * 100.0;

        // return for Burn dashboard
        SerializedEntry::new(
            format!("{:.2}%", curr_wer),
            curr_wer.to_string(),
        )
    }

    fn clear(&mut self) {
        self.total_error = 0;
        self.total_words = 0;
    }
}



/// edit distance solver for finding min number of edits needed to change one sequence into another
/// params:
/// - seq1: predicted sequence of items (IDs/words)
/// - seq2: ground truth sequence of items
/// returns: total count of insertions, deletions, and substitutions
fn levenshtein<T: PartialEq>(seq1: &[T], seq2: &[T]) -> usize {
    if seq1 == seq2 { return 0 }
    if seq1.is_empty() { return seq2.len() }
    if seq2.is_empty() { return seq1.len() }

    let mut col: Vec<usize> = (0..=seq2.len()).collect();

    for (i, el1) in seq1.iter().enumerate() {
        let mut last_diag = col[0];
        col[0] = i + 1;
        for (j, el2) in seq2.iter().enumerate() {
            let old_col_j = col[j + 1];
            col[j + 1] = if el1 == el2 {
                last_diag
            } else {
                1 + min(min(col[j], col[j + 1]), last_diag)
            };
            last_diag = old_col_j;
        }
    }
    col[seq2.len()]
}



/// transform padded batch tensor into a ragged vector of sequences
/// conversion performed by: tensor --> data --> flat slice --> chunk by length
/// params:
/// - data_tensor: padded tensor containing batch sequences [N, Max_L]
/// - lengths_tensor: tensor containing the actual length of each sequence [N]
/// returns: a ragged vector of sequences with padding removed [[L1], [L2], ... [LN]]
fn unpad_to_vec<B: Backend>(
    data_tensor: Tensor<B, 2, Int>,
    lengths_tensor: Tensor<B, 1, Int>,
) -> Vec<Vec<usize>> {
    let data = data_tensor.into_data().convert::<i64>();
    let data_slice = data.as_slice::<i64>().expect("Conversion failed");

    let lengths = lengths_tensor.into_data().convert::<i64>();
    let lengths_slice = lengths.as_slice::<i64>().expect("Conversion failed");

    let max_length = data.shape[1];

    (0..lengths_slice.len())
        .map(|i| {
            let start_chunk = i * max_length;
            let end_chunk = start_chunk + lengths_slice[i] as usize;
            data_slice[start_chunk..end_chunk].iter().map(|&x| x as usize).collect()
        })
        .collect()
}



#[cfg(test)]
mod tests {
    use crate::{
        ctc::ctc_decode::{
            CtcDecodeType,
            CtcDecoderConfig,
        },
        vocab::{BLANK_ID, VOCAB, VOCAB_SIZE},
        training::trainer::B,
    };

    use super::*;

    use burn::{
        backend::NdArray,
        tensor::Tensor,
        data::dataloader::Progress,
    };

    // helper to create one-hot logits array of vocab size for testing
    fn one_hot_logits<const V: usize>(hot: usize, hi: f32, lo: f32) -> [f32; V] {
        let mut row = [lo; V];
        row[hot] = hi;
        row
    }

    #[test]
    fn test_levenshtein_distance() {

        // example sequence "cat" (2, 0, 19) vs. "rat" (17, 0, 19):
        // when i = 0, j = 0:    el1 = c,    el2 = r,    col = [0, 1, 2, 3],    last_diag = 0,    col[0] = 1,    old_col_j = 1
        // when i = 0, j = 1:    el1 = c,    el2 = a,    col = [0, 1, 2, 3],    last_diag = 1,    col[0] = 1,    old_col_j = 2
        // when i = 0, j = 2:    el1 = c,    el2 = t,    col = [0, 1, 2, 3],    last_diag = 2,    col[0] = 1,    old_col_j = 3

        // when i = 1, j = 0:    el1 = a,    el2 = r,    col = [2, 1, 2, 2],    last_diag = 0,    col[0] = 1,    old_col_j = 1
        // when i = 1, j = 1:    el1 = a,    el2 = a,    col = [2, 1, 1, 2],    last_diag = 1,    col[0] = 1,    old_col_j = 2
        // when i = 1, j = 2:    el1 = a,    el2 = t,    col = [2, 1, 1, 2],    last_diag = 2,    col[0] = 1,    old_col_j = 2

        // when i = 2, j = 0:    el1 = t,    el2 = r,    col = [3, 2, 1, 2],    last_diag = 2,    col[0] = 1,    old_col_j = 2
        // when i = 2, j = 1:    el1 = t,    el2 = a,    col = [3, 2, 2, 2],    last_diag = 2,    col[0] = 1,    old_col_j = 1
        // when i = 2, j = 2:    el1 = t,    el2 = t,    col = [3, 2, 2, 1],    last_diag = 1,    col[0] = 1,    old_col_j = 2

        // final edit distance: col[seq2.len()] = 1

        // should result in edit distances of 1 for all of these
        assert_eq!(levenshtein(&[2, 0, 19], &[17, 0, 19]), 1);  // cat vs rat (IDs)
        assert_eq!(levenshtein(&[1, 2, 3], &[1, 2]), 1);        // deletion
        assert_eq!(levenshtein(&[1, 2], &[1, 2, 3]), 1);        // insertion
        assert_eq!(levenshtein(&[], &[1, 2]), 2);               // empty vs sequence
    }

    #[test]
    fn test_unpad_to_vec() {
        // N = 3, max_l = 5
        // Row 0: [1, 2, 3, 0, 0]  length = 3
        // Row 1: [4, 5, 0, 0, 0]  length = 2
        // Row 2: [6, 7, 8, 9, 0]  length = 4

        let data = Tensor::<B, 2, Int>::from_data(
            [
                [1, 2, 3, 0, 0],
                [4, 5, 0, 0, 0],
                [6, 7, 8, 9, 0],
            ],
            &Default::default(),
        );

        let lengths = Tensor::<B, 1, Int>::from_data(
            [3, 2, 4],
            &Default::default(),
        );

        let result = unpad_to_vec(data, lengths);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], vec![1, 2, 3]);
        assert_eq!(result[1], vec![4, 5]);
        assert_eq!(result[2], vec![6, 7, 8, 9]);
    }

    #[test]
    fn test_cer_metric() {
        const N: usize = 1;
        const T: usize = 33;
        const L: usize = 11;
        const V: usize = VOCAB_SIZE;

        const HI: f32 = 8.0; // big logit for target symbol
        const LO: f32 = 0.0; // baseline for rest

        let token_map = TokenMap::new(VOCAB);
        let device = Default::default();

        let (d_id, e_id, h_id, l_id, o_id, r_id, u_id, w_id, space_id, blank_id) = (3, 4, 7, 11, 14, 17, 20, 22, BLANK_ID - 1, BLANK_ID); // "a", "e", "h", "t", "_"

        // mock data of logits from frames
        let logits_data: [[[f32; V]; T]; N] = [[
            one_hot_logits(h_id, HI, LO),                        // t = 0:   "h"
            one_hot_logits(h_id, HI, LO),                        // t = 1:   "h"
            one_hot_logits(h_id, HI, LO),                        // t = 2:   "h"
            one_hot_logits(e_id, HI, LO),                        // t = 3:   "e"
            one_hot_logits(e_id, HI, LO),                        // t = 4:   "e"
            one_hot_logits(l_id, HI, LO),                        // t = 5:   "l"
            one_hot_logits(blank_id, HI, LO),                    // t = 6:   "_"
            one_hot_logits(l_id, HI, LO),                        // t = 7:   "l"
            one_hot_logits(l_id, HI, LO),                        // t = 8:   "l"
            one_hot_logits(l_id, HI, LO),                        // t = 9:   "l"
            one_hot_logits(o_id, HI, LO),                        // t = 10:  "o"
            one_hot_logits(o_id, HI, LO),                        // t = 11:  "o"
            one_hot_logits(o_id, HI, LO),                        // t = 12:  "o"
            one_hot_logits(space_id, HI, LO),                    // t = 13:  " "
            one_hot_logits(space_id, HI, LO),                    // t = 14:  " "
            one_hot_logits(space_id, HI, LO),                    // t = 15:  " "
            one_hot_logits(space_id, HI, LO),                    // t = 16:  " "
            one_hot_logits(w_id, HI, LO),                        // t = 17:  "w"
            one_hot_logits(w_id, HI, LO),                        // t = 18:  "w"
            one_hot_logits(w_id, HI, LO),                        // t = 19:  "w"
            one_hot_logits(u_id, HI, LO),                        // t = 20:  "u"
            one_hot_logits(u_id, HI, LO),                        // t = 21:  "u"
            one_hot_logits(u_id, HI, LO),                        // t = 22:  "u"
            one_hot_logits(r_id, HI, LO),                        // t = 23:  "r"
            one_hot_logits(r_id, HI, LO),                        // t = 24:  "r"
            one_hot_logits(r_id, HI, LO),                        // t = 25:  "r"
            one_hot_logits(r_id, HI, LO),                        // t = 26:  "r"
            one_hot_logits(r_id, HI, LO),                        // t = 27:  "r"
            one_hot_logits(l_id, HI, LO),                        // t = 28:  "l"
            one_hot_logits(l_id, HI, LO),                        // t = 29:  "l"
            one_hot_logits(l_id, HI, LO),                        // t = 30:  "l"
            one_hot_logits(d_id, HI, LO),                        // t = 31:  "d"
            one_hot_logits(d_id, HI, LO),                        // t = 32:  "d"
        ]];

        let targets_data: [[i64; L]; N] = [[
            h_id        as i64,
            e_id        as i64,
            l_id        as i64,
            l_id        as i64,
            o_id        as i64,
            space_id    as i64,
            w_id        as i64,
            o_id        as i64,
            r_id        as i64,
            l_id        as i64,
            d_id        as i64,
        ]];

        // init metric input tensors from dummy data
        let loss = Tensor::<B, 1>::from_data([0.0], &device);
        let inputs = Tensor::<B, 3>::from_data(logits_data, &device);
        let targets = Tensor::<B, 2, Int>::from_data(targets_data, &device);
        let input_lengths = Tensor::<B, 1, Int>::from_data([T], &device);
        let target_lengths = Tensor::<B, 1, Int>::from_data([L], &device);

        // init VSRM logit outputs as metric input item
        let metric_input = VsrmMetricInput{
            loss,
            inputs: inputs.clone(),
            targets: targets.clone(),
            input_lengths,
            target_lengths,
        };

        // init dummy metadata
        let metadata = MetricMetadata {
            progress: Progress {
                items_processed: 1,
                items_total: 1,
            },
            epoch: 1,
            epoch_total: 1,
            iteration: 1,
            lr: None,
        };

        // init CTC decoder
        let decoder = CtcDecoderConfig::new()
            .with_search_type(CtcDecodeType::GreedySearch)
            .with_blank_id(BLANK_ID)
            .init();

        // init CER metric
        let mut cer_metric = CtcCharErrorRate::<B>::new(decoder);

        // obtain CER metric
        let metric_entry = cer_metric.update(&metric_input, &metadata);

        // --------------------------------- DEBUGGING BLOCK ---------------------------------

        let prediction: Vec<usize> = cer_metric.decoder.forward(inputs.clone())[0]
            .iter()
            .map(|&x| x as usize)
            .collect();
        let target: Vec<usize> = targets_data[0].iter().map(|&x| x as usize).collect();

        let prediction_chars = token_map.ids_to_chars(&prediction)
            .expect("ID to char mapping failed")
            .iter()
            .collect::<String>();
        let target_chars = token_map.ids_to_chars(&target)
            .expect("ID to char mapping failed")
            .iter()
            .collect::<String>();

        println!("Prediction: '{}'", prediction_chars);
        println!("Target: '{}'", target_chars);
        println!("CER: {}", metric_entry.formatted);

        // -----------------------------------------------------------------------------------

        // 1 substitution / 11 total chars ≈ 9.09%
        assert_eq!(metric_entry.formatted, "9.09%");
    }

    #[test]
    fn test_wer_metric() {
        const N: usize = 1;
        const T: usize = 33;
        const L: usize = 11;
        const V: usize = VOCAB_SIZE;

        const HI: f32 = 8.0; // big logit for target symbol
        const LO: f32 = 0.0; // baseline for rest

        let token_map = TokenMap::new(VOCAB);
        let device = Default::default();

        let (d_id, e_id, h_id, l_id, o_id, r_id, u_id, w_id, space_id, blank_id) = (3, 4, 7, 11, 14, 17, 20, 22, BLANK_ID - 1, BLANK_ID); // "a", "e", "h", "t", "_"

        // mock data of logits from frames
        let logits_data: [[[f32; V]; T]; N] = [[
            one_hot_logits(h_id, HI, LO),                        // t = 0:   "h"
            one_hot_logits(h_id, HI, LO),                        // t = 1:   "h"
            one_hot_logits(h_id, HI, LO),                        // t = 2:   "h"
            one_hot_logits(e_id, HI, LO),                        // t = 3:   "e"
            one_hot_logits(e_id, HI, LO),                        // t = 4:   "e"
            one_hot_logits(l_id, HI, LO),                        // t = 5:   "l"
            one_hot_logits(blank_id, HI, LO),                    // t = 6:   "_"
            one_hot_logits(l_id, HI, LO),                        // t = 7:   "l"
            one_hot_logits(l_id, HI, LO),                        // t = 8:   "l"
            one_hot_logits(l_id, HI, LO),                        // t = 9:   "l"
            one_hot_logits(o_id, HI, LO),                        // t = 10:  "o"
            one_hot_logits(o_id, HI, LO),                        // t = 11:  "o"
            one_hot_logits(o_id, HI, LO),                        // t = 12:  "o"
            one_hot_logits(space_id, HI, LO),                    // t = 13:  " "
            one_hot_logits(space_id, HI, LO),                    // t = 14:  " "
            one_hot_logits(space_id, HI, LO),                    // t = 15:  " "
            one_hot_logits(space_id, HI, LO),                    // t = 16:  " "
            one_hot_logits(w_id, HI, LO),                        // t = 17:  "w"
            one_hot_logits(w_id, HI, LO),                        // t = 18:  "w"
            one_hot_logits(w_id, HI, LO),                        // t = 19:  "w"
            one_hot_logits(u_id, HI, LO),                        // t = 20:  "u"
            one_hot_logits(u_id, HI, LO),                        // t = 21:  "u"
            one_hot_logits(u_id, HI, LO),                        // t = 22:  "u"
            one_hot_logits(r_id, HI, LO),                        // t = 23:  "r"
            one_hot_logits(r_id, HI, LO),                        // t = 24:  "r"
            one_hot_logits(r_id, HI, LO),                        // t = 25:  "r"
            one_hot_logits(r_id, HI, LO),                        // t = 26:  "r"
            one_hot_logits(r_id, HI, LO),                        // t = 27:  "r"
            one_hot_logits(l_id, HI, LO),                        // t = 28:  "l"
            one_hot_logits(l_id, HI, LO),                        // t = 29:  "l"
            one_hot_logits(l_id, HI, LO),                        // t = 30:  "l"
            one_hot_logits(d_id, HI, LO),                        // t = 31:  "d"
            one_hot_logits(d_id, HI, LO),                        // t = 32:  "d"
        ]];

        let targets_data: [[i64; L]; N] = [[
            h_id        as i64,
            e_id        as i64,
            l_id        as i64,
            l_id        as i64,
            o_id        as i64,
            space_id    as i64,
            w_id        as i64,
            o_id        as i64,
            r_id        as i64,
            l_id        as i64,
            d_id        as i64,
        ]];

        // init metric input tensors from dummy data
        let loss = Tensor::<B, 1>::from_data([0.0], &device);
        let inputs = Tensor::<B, 3>::from_data(logits_data, &device);
        let targets = Tensor::<B, 2, Int>::from_data(targets_data, &device);
        let input_lengths = Tensor::<B, 1, Int>::from_data([T], &device);
        let target_lengths = Tensor::<B, 1, Int>::from_data([L], &device);

        // init VSRM logit outputs as metric input item
        let metric_input = VsrmMetricInput{
            loss,
            inputs: inputs.clone(),
            targets: targets.clone(),
            input_lengths,
            target_lengths,
        };

        // init dummy metadata
        let metadata = MetricMetadata {
            progress: Progress {
                items_processed: 1,
                items_total: 1,
            },
            epoch: 1,
            epoch_total: 1,
            iteration: 1,
            lr: None,
        };

        // init CTC decoder
        let decoder = CtcDecoderConfig::new()
            .with_search_type(CtcDecodeType::GreedySearch)
            .with_blank_id(BLANK_ID)
            .init();

        // init WER metric
        let mut wer_metric = CtcWordErrorRate::<B>::new(decoder, token_map.clone());

        // obtain WER metric
        let metric_entry = wer_metric.update(&metric_input, &metadata);

        // --------------------------------- DEBUGGING BLOCK ---------------------------------

        let prediction: Vec<usize> = wer_metric.decoder.forward(inputs.clone())[0]
            .iter()
            .map(|&x| x as usize)
            .collect();
        let target: Vec<usize> = targets_data[0].iter().map(|&x| x as usize).collect();

        let prediction_chars = token_map.ids_to_chars(&prediction)
            .expect("ID to char mapping failed")
            .iter()
            .collect::<String>();
        let target_chars = token_map.ids_to_chars(&target)
            .expect("ID to char mapping failed")
            .iter()
            .collect::<String>();

        println!("Prediction: '{}'", prediction_chars);
        println!("Target: '{}'", target_chars);
        println!("WER: {}", metric_entry.formatted);

        // -----------------------------------------------------------------------------------

        // 1 error / 2 total chars ≈ 50.00%
        assert_eq!(metric_entry.formatted, "50.00%");
    }
}