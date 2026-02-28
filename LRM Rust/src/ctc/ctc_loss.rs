//! Connectionist Temporal Classification (CTC) loss implementation.
//! 
//! This module implements the forward algorithm for CTC loss, which helps to train
//! sequence-to-sequence (Seq2Seq) models where the alignment between input frames
//! and target sequences is unknown.



// custom imports
use crate::utils::{
    log_sum_exp_2_tensor,
    log_sum_exp_3_tensor,
};
// imports
use burn::{
    config::Config,
    module::Ignored,
    nn::loss::Reduction,
    tensor::{
        activation::log_softmax,
        backend::Backend,
        Tensor,
        {Int, Bool},
    },
};



#[derive(Debug, Config)]
pub struct CtcLossConfig {
    #[config(default = "0")]
    pub blank_id: usize, // ID of blank token in vocab

    #[config(default = "Reduction::Mean")]
    pub reduction: Reduction, // Burn's reduction method (mean, sum, auto)
}



impl CtcLossConfig {
    pub fn init(&self) -> CtcLoss {
        CtcLoss {
            blank_id: self.blank_id,
            reduction: Ignored(self.reduction.clone()),
        }
    }
}



#[derive(Clone, Debug)]
pub struct CtcLoss {
    pub blank_id: usize,
    pub reduction: Ignored<Reduction>,
}



impl CtcLoss {
    /// compute CTC loss for batch of samples
    /// since inputs/targets are padded to max length found in batch,
    /// needs separate input/target lengths info for true lengths
    /// params:
    /// - inputs: [N, T_max, Vocab] (time-padded logits from model)
    /// - targets: [N, L_max] (length-padded target sequences)
    /// - input_lengths: [N] (non-padded lengths of inputs)
    /// - target_lengths: [N] (non-padded lengths of targets)
    /// returns: scalar loss (if reduction is mean or sum)
    pub fn forward<B: Backend>(
        &self,
        inputs: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
        input_lengths: Tensor<B, 1, Int>,
        target_lengths: Tensor<B, 1, Int>,
    ) -> Tensor<B, 1> {
        let tensor = self.forward_no_reduction(inputs, targets, input_lengths, target_lengths);
        match &self.reduction.0 {
            Reduction::Mean => tensor.mean(),
            Reduction::Sum => tensor.sum(),
            other => panic!("{other:?} reduction is not supported"),
        }
    }

    /// like `forward`, but without reduction
    /// since inputs/targets are padded to max length found in batch, needs separate input/target lengths info for true lengths
    /// works by:
    /// - computing log-probs from logits via log softmax
    /// - modifying target sequences by interleaving blanks between symbols
    /// - iteratively computing accummulated log-probabilities of all possible paths through time-sequence grid that align with the modified target sequence
    /// params:
    /// - inputs: [N, T_max, Vocab] (time-padded logits from model)
    /// - targets: [N, L_max] (length-padded target sequences)
    /// - input_lengths: [N] (non-padded lengths of inputs)
    /// - target_lengths: [N] (non-padded lengths of targets)
    /// returns: [N] (loss per sample in batch)
    pub fn forward_no_reduction<B: Backend>(
        &self,
        inputs: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
        input_lengths: Tensor<B, 1, Int>,
        target_lengths: Tensor<B, 1, Int>,
    ) -> Tensor<B, 1> {
        let device = inputs.device();
        let [n, t, vocab_size] = inputs.dims();
        // let sentinel_value = f32::NEG_INFINITY; // possibly causes NaNs
        let sentinel_value = -1e30; // numerically stable
        let log_probs = log_softmax(inputs.clone(), 2); // turn logits into log-probs

        assert_eq!(inputs.dims()[0], targets.dims()[0], "Inputs/targets batch size mismatch");
        assert_eq!(input_lengths.dims()[0], inputs.shape()[0], "Inputs/lengths batch size mismatch");
        assert_eq!(target_lengths.dims()[0], targets.shape()[0], "Targets/lengths batch size mismatch");

        // original and blank-interleaved lengths of target sequence (padded length)
        let orig_pad_targ_length: usize = targets.dims()[1];             // L_max
        let intr_pad_targ_length = 2 * orig_pad_targ_length + 1;  // 2L_max + 1

        // get batch-wise interleaved target lengths for masking (true lengths)
        let intr_targ_lengths = target_lengths.clone() * 2 + 1; // [N]

        // perform batch-wise interleaving of blank IDs into targets
        let targets = self.interleave_targets_with_blanks(targets, &device); // [N, (2L + 1)]

        // init DP buffer for storing accumulated log-probs for paths through time-sequence grid
        let mut curr_fwd: Tensor<B, 2> = Tensor::full([n, intr_pad_targ_length], sentinel_value, &device); // [N, (2L_max + 1)]

        // precompute skip validity masks for 1-pos and 2-pos jumps in DP grid
        // (these are same across timesteps, so can compute once here and reuse in DP loop)
        let (skip_1_mask, skip_2_mask) = self.compute_skip_validity_masks(targets.clone(), &device); // both [N, (2L + 1)]

        // build batch-wise length mask to mask out padded positions in sequence
        let length_mask = Tensor::<B, 1, Int>::arange(0..(intr_pad_targ_length as i64), &device) // [(2L_max + 1)]
            .expand([n, intr_pad_targ_length])  // [N, (2L_max + 1)]
            .lower(intr_targ_lengths.clone().reshape([n, 1]));   // [N, (2L_max + 1)] but holds true for only for lengths within per sample (2L + 1)

        // t = 0: DP base case (initialization)
        // init DP with log-probs of first two symbols in blank-interleaved target sequence
        let initial_log_probs = log_probs.clone().slice([0..n, 0..1, 0..vocab_size]); // [N, 1, Vocab]
        for i in 0..2 {
            // grab batch-wise symbol IDs of i-th position in blank-interleaved target sequence
            let id_0_i = targets.clone()
                .slice([0..n, i..(i + 1)])  // [N, 1]
                .reshape([n, 1, 1]);                          // [N, 1, 1]

            // gather batch-wise log-probs of those symbol IDs at t = 0
            let log_prob_0_i = initial_log_probs.clone()
                .gather(2, id_0_i)  // [N, 1, 1]
                .reshape([n, 1]);                      // [N, 1]

            // write gathered log-probs to DP buffer
            curr_fwd = curr_fwd.slice_assign([0..n, i..(i + 1)], log_prob_0_i);
        }

        // t ≥ 1: DP recurrence
        // per timestep, compute batch-wise log-probs of all possible paths to each position in blank-interleaved target sequence
        for t_idx in 1..t {
            // build a batch-wise time mask to mask out padded timesteps beyond true input lengths in batch
            // (since inputs are time-padded to max length in batch, but each sample has different true length)
            let time_mask = input_lengths.clone()
                .greater_elem(t_idx as i64)
                .unsqueeze_dim::<2>(1)
                .expand([n, intr_pad_targ_length]);

            // get batch-wise log-probs of each symbol ID in blank-interleaved target sequence at current timestep
            // then mask out log-probs at positions beyond true input timestep lengths
            let log_probs_t = log_probs.clone()                                          // [N, T, Vocab]
                .slice([0..n, t_idx..(t_idx + 1), 0..vocab_size])                              // [N, 1, Vocab]
                .expand([n, intr_pad_targ_length, vocab_size])                                  // [N, (2L + 1), Vocab]
                .gather(2, targets.clone().reshape([n, intr_pad_targ_length, 1]))  // [N, (2L + 1), 1]
                .reshape([n, intr_pad_targ_length])                                             // [N, (2L + 1)]
                .mask_where(
                    time_mask.clone().bool_not(),
                    Tensor::full([n, intr_pad_targ_length], sentinel_value, &device)
                );

            // possible actions at current timestep (based on what could have happened at previous frame)
            // - stay:  log-prob of staying at same symbol in interleaved target sequence   (from same position)
            // - adv_1: log-prob of advancing by one symbol                                 (from one position earlier)
            // - adv_2: log-prob of skipping blank to advance by two symbols                (from two positions earlier)
            let stay = curr_fwd.clone();
            let adv_1 = curr_fwd.clone()
                .roll_dim(-1, 1) // shift right by 1 on dim 1
                .mask_fill(skip_1_mask.clone().bool_not(), sentinel_value);
            let adv_2 = curr_fwd.clone()
                .roll_dim(-2, 1) // shift right by 2 on dim 1
                .mask_fill(skip_2_mask.clone().bool_not(), sentinel_value);

            // compute accumulated log-prob for current path (in time-sequence grid) from all possible actions
            // then mask out log-probs at positions beyond true blank-interleaved target lengths
            let next_fwd = (log_sum_exp_3_tensor(stay, adv_1, adv_2) + log_probs_t.clone()) // [N, (2L + 1)]
                .mask_fill(length_mask.clone().bool_not(), sentinel_value);

            curr_fwd = curr_fwd.mask_where(time_mask, next_fwd); // 
        }

        // t = T: DP termination
        // end at last blank or symbol before it in blank-interleaved target sequence
        let end_1 = curr_fwd.clone().gather(1, (intr_targ_lengths.clone() - 1).reshape([n, 1])); // [N, 1]
        let end_2 = curr_fwd.clone().gather(1, (intr_targ_lengths.clone() - 2).reshape([n, 1])); // [N, 1]

        let total_log_prob = log_sum_exp_2_tensor(end_1, end_2); // [N, 1]
        -total_log_prob.squeeze_dim(1) // [N]
    }


    /// interleave blank IDs into target sequences in batch for CTC loss computation
    /// params:
    /// - targets: [N, L_max] (length-padded target sequences)
    /// - device: backend device to create tensors on
    /// returns: [N, (2L + 1)] (target sequences with blanks interleaved, where L is original target length without blanks)
    fn interleave_targets_with_blanks<B: Backend>(
        &self,
        targets: Tensor<B, 2, Int>, // [N, L_max]
        device: &B::Device,
    ) -> Tensor<B, 2, Int> {
        let blank = self.blank_id as i64;
        let n = targets.dims()[0]; // batch size

        // original and blank-interleaved lengths of target sequence
        let len_targ_orig: usize = targets.dims()[1];      // L
        let len_targ_intr = 2 * len_targ_orig + 1;  // 2L + 1

        // make [N, L + 1] tensor of blank IDs (actually want shape [N, L], but stack requires same shape)
        // take [N, L] tensor of target IDs + single blank ID concatenated at end for a [N, L + 1] tensor
        let blanks = Tensor::<B, 2, Int>::full([n, len_targ_orig + 1], blank, device);
        let labels = Tensor::<B, 2, Int>::cat(vec![targets, Tensor::<B, 2, Int>::full([n, 1], blank, device)], 1);

        // perform:
        // - stack      [N, 2, L + 1]
        // - transpose  [N, 2L + 2]
        // - flatten    [N, 2L + 1]
        // to obtain: [blank, y1, blank, y2, ..., blank, yT, blank] (IDs of blank-interleaved targets of length 2L + 1)
        let interleaved = Tensor::stack::<3>(vec![blanks, labels], 1) // [N, 2, L + 1]
                .swap_dims(1, 2)
                .reshape([n, (len_targ_orig + 1) * 2])
                .slice([0..n, 0..len_targ_intr]);
        assert_eq!(interleaved.dims()[1], len_targ_intr, "Interleaved targets/length mismatch");

        interleaved
    }

    /// compute batch-wise skip validity masks for 1-pos and 2-pos jumps in time-sequence DP grid
    /// params:
    /// - interleaved_targets: [N, 2L + 1] (length-padded target sequences with blanks interleaved)
    /// - device: backend device to create tensors on
    /// returns: tuple of masks, where each is a [N, 2L + 1] bool tensor indicating whether it's valid to skip by 1 or 2 positions in DP grid at a position in blank-interleaved target sequence
    fn compute_skip_validity_masks<B: Backend>(
        &self,
        interleaved_targets: Tensor<B, 2, Int>,
        device: &B::Device,
    ) -> (Tensor<B, 2, Bool>, Tensor<B, 2, Bool>) {
        let blank = self.blank_id as i64;
        let [n, l] = interleaved_targets.dims();

        // blank-interleaved symbol positions and IDs [N, 2L + 1]
        let intr_targ_ids = interleaved_targets.clone();
        let intr_targ_pos = Tensor::<B, 1, Int>::arange(0..(l as i64), device)
            .reshape([1, l])
            .expand([n, l]);

        // goal: (pos is odd) && (pos > 1) && (ids[pos] != ids[pos − 2]) && (ids[pos] != blank)
        let prev_2 = intr_targ_ids.clone().roll_dim(-2, 1); // shift right by 2 on dim 1
        let odd_mask = intr_targ_pos.clone().remainder_scalar(2).not_equal_elem(0);
        let pos_gt_1_mask = intr_targ_pos.clone().greater_elem(1);
        let neq_prev_2_mask = intr_targ_ids.clone().not_equal(prev_2.clone());
        let not_blank_mask = intr_targ_ids.clone().not_equal_elem(blank);

        // can-skip-by-1 & can-skip-by-2 validity masks
        let can_skip_1_mask = intr_targ_pos.clone().greater_elem(0);
        let can_skip_2_mask = odd_mask
            .clone()
            .bool_and(pos_gt_1_mask.clone())
            .bool_and(neq_prev_2_mask.clone())
            .bool_and(not_blank_mask.clone());

        (can_skip_1_mask, can_skip_2_mask)
    }
}



// testing
#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        backend::ndarray::NdArray,
        nn::loss::Reduction,
        prelude::Int,
        tensor::Tensor,
    };

    type B = NdArray<f32>;

    /// helper to create length tensors for batch
    fn tensorize_lengths(
        batch_size: usize,
        input_length: usize,
        target_length: usize,
        device: &<B as burn::tensor::backend::Backend>::Device,
    ) -> (Tensor<B, 1, Int>, Tensor<B, 1, Int>) {
        let data_inputs: Vec<i64> = vec![input_length as i64; batch_size];
        let data_targets: Vec<i64> = vec![target_length as i64; batch_size];

        let input_length = Tensor::<B, 1, Int>::from_ints(data_inputs.as_slice(), device);
        let target_len = Tensor::<B, 1, Int>::from_ints(data_targets.as_slice(), device);
        (input_length, target_len)
    }

    #[test]
    fn debug_roll_dim_shift_directionality() {
        let device = Default::default();

        // dummy 1D-like tensor: [[0, 1, 2, 3, 4]]
        let data = Tensor::<B, 2>::from_floats([[0.0, 1.0, 2.0, 3.0, 4.0]], &device);
        println!("\nOriginal: {:?}\n", data.to_data().convert::<f32>().into_vec::<f32>().unwrap());

        // test positive roll
        // for most frameworks, this should move right (4 moves to index 0):
        // [0, 1, 2, 3, 4] --> [4, 0, 1, 2, 3]
        println!("Attempting Roll +1");
        let pos_roll = data.clone().roll_dim(1, 1); 
        println!("Roll +1: {:?}\n", pos_roll.to_data().convert::<f32>().into_vec::<f32>().unwrap());

        // test negative roll
        // for most frameworks, this should move left (0 moves to index 4):
        // [0, 1, 2, 3, 4] --> [1, 2, 3, 4, 0]
        println!("Attempting Roll -1");
        let neg_roll = data.clone().roll_dim(-1, 1);
        println!("Roll -1: {:?}\n", neg_roll.to_data().convert::<f32>().into_vec::<f32>().unwrap());

        // takeaway after testing (as of Burn 0.20.1):
        // - positive shift val rolls left
        // - negative shift val rolls right
    }

    #[test]
    fn ctc_base_case_single_char_no_repeat() {
        let device = Default::default();
        let blank_id = 0usize;
        let vocab = 3usize; // {blank = 0, 'A' = 1, 'B' = 2}
        let t = 2usize;
        let threshold = 5e-2;
        let reduction = Reduction::Mean;

        // target = "A"
        let targets = Tensor::<B, 2, Int>::from_ints([[1i64]], &device);
        let (in_len, tgt_len) = tensorize_lengths(1, t, 1, &device);

        let loss = CtcLossConfig::new()
            .with_blank_id(blank_id)
            .with_reduction(reduction)
            .init();

        // logits favor sequence: [blank at t = 0, 'A' at t = 1]
        // shape [N = 1, T = 2, V = 3]
        let mut logits = Tensor::<B, 3>::zeros([1, t, vocab], &device);

        // t = 0 strongly blank
        logits = logits.slice_assign(
            [0..1, 0..1, 0..1],
            Tensor::<B, 3>::from_floats([[[5.0]]], &device),
        );
        // t = 1 strongly 'A' (ID = 1)
        logits = logits.slice_assign(
            [0..1, 1..2, 1..2],
            Tensor::<B, 3>::from_floats([[[5.0]]], &device),
        );

        let loss = loss.forward(logits, targets, in_len, tgt_len).into_scalar();

        // with such peaked logits, loss should be near zero
        println!("\nLoss = {:?} | Threshold = {:?}\n", loss, threshold);
        assert!(loss < threshold, "loss too large");
    }

    #[test]
    fn ctc_disallows_skip_two_on_repeat_labels() {
        let device = Default::default();
        let blank_id = 0usize;
        let vocab = 3usize; // {blank = 0, 'A' = 1, 'B' = 2}
        let t_1 = 3usize;
        let t_2 = 4usize;
        let reduction = Reduction::Mean;

        let loss = CtcLossConfig::new()
            .with_blank_id(blank_id)
            .with_reduction(reduction.clone())
            .init();

        // target = "AA" (repeat), which should forbid the 2-pos jump when same consecutive symbol
        let targets = Tensor::<B, 2, Int>::from_ints([[1i64, 1i64]], &device);

        // case 1 (hostile alignment, t = 3): can't skip,
        let (in_len_1, tgt_len_1) = tensorize_lengths(1, t_1, 2, &device);
        let mut logits_1 = Tensor::<B, 3>::zeros([1, t_1, vocab], &device);
        logits_1 = logits_1.slice_assign(
            [0..1, 0..1, 0..1],
            Tensor::<B, 3>::from_floats([[[4.0]]], &device),
        ); // high blank at t = 0
        logits_1 = logits_1.slice_assign(
            [0..1, 1..2, 1..2],
            Tensor::<B, 3>::from_floats([[[4.0]]], &device),
        ); // high 'A' at t = 1
        logits_1 = logits_1.slice_assign(
            [0..1, 2..3, 1..2],
            Tensor::<B, 3>::from_floats([[[4.0]]], &device),
        ); // high 'A' at t = 2
        let loss_hostile = loss
            .forward(logits_1, targets.clone(), in_len_1, tgt_len_1)
            .into_scalar();

        // case 2 (friendly alignment, t = 4): can't skip, doesn't skip, and uses intermediate blank
        let (in_len_2, tgt_len_2) = tensorize_lengths(1, t_2, 2, &device);
        let mut logits_2 = Tensor::<B, 3>::zeros([1, t_2, vocab], &device);
        logits_2 = logits_2.slice_assign(
            [0..1, 0..1, 0..1],
            Tensor::<B, 3>::from_floats([[[4.0]]], &device),
        ); // high blank at t = 0
        logits_2 = logits_2.slice_assign(
            [0..1, 1..2, 1..2],
            Tensor::<B, 3>::from_floats([[[4.0]]], &device),
        ); // high 'A' at t = 1
        logits_2 = logits_2.slice_assign(
            [0..1, 2..3, 0..1],
            Tensor::<B, 3>::from_floats([[[4.0]]], &device),
        ); // high blank at t = 2
        logits_2 = logits_2.slice_assign(
            [0..1, 3..4, 1..2],
            Tensor::<B, 3>::from_floats([[[4.0]]], &device),
        ); // high 'A' at t = 3
        let loss_friendly = loss
            .forward(logits_2, targets.clone(), in_len_2, tgt_len_2)
            .into_scalar();

        // should be: case 1 loss > case 2 loss
        println!(
            "\nLoss 1 = {:?} | Loss 2 = {:?}\n",
            loss_hostile, loss_friendly
        );
        assert!(loss_hostile.is_finite() && loss_friendly.is_finite());
        assert!(
            loss_hostile > loss_friendly,
            "hostile loss case (t = 3) should be higher than friendly loss case (t = 4)"
        );
    }

    #[test]
    fn ctc_allows_skip_two_when_adjacent_labels_differ() {
        let device = Default::default();
        let blank_id = 0usize;
        let vocab = 3usize; // {blank = 0, 'A' = 1, 'B' = 2}
        let t = 3usize;
        let threshold = 5e-2;
        let reduction = Reduction::Mean;

        let loss = CtcLossConfig::new()
            .with_blank_id(blank_id)
            .with_reduction(reduction.clone())
            .init();

        // target = "AB" (different), allows jump over blank when confident
        let targets = Tensor::<B, 2, Int>::from_ints([[1i64, 2i64]], &device);
        let (in_len, tgt_len) = tensorize_lengths(1, t, 2, &device);

        let mut logits = Tensor::<B, 3>::zeros([1, t, vocab], &device);
        // t = 0 blank, t = 1 'A', t = 2 'B'
        logits = logits.slice_assign(
            [0..1, 0..1, 0..1],
            Tensor::<B, 3>::from_floats([[[5.0]]], &device),
        );
        logits = logits.slice_assign(
            [0..1, 1..2, 1..2],
            Tensor::<B, 3>::from_floats([[[5.0]]], &device),
        );
        logits = logits.slice_assign(
            [0..1, 2..3, 2..3],
            Tensor::<B, 3>::from_floats([[[5.0]]], &device),
        );

        let loss = loss.forward(logits, targets, in_len, tgt_len).into_scalar();

        println!("\nLoss = {:?} | Threshold = {:?}\n", loss, threshold);
        assert!(loss < threshold, "loss too large");
    }

    #[test]
    fn ctc_padding_respected_by_lengths() {
        let device = Default::default();
        let blank_id = 0usize;
        let vocab = 3usize; // {blank = 0, 'A' = 1, 'B' = 2}
        let t = 4usize;
        let threshold = 5e-2;
        let reduction = Reduction::Mean;

        let loss = CtcLossConfig::new()
            .with_blank_id(blank_id)
            .with_reduction(reduction.clone())
            .init();

        // targets (padded to 2)
        let targets = Tensor::<B, 2, Int>::from_ints([[1i64, 0i64], [2i64, 0i64]], &device);
        // true lengths: input_len = 2 for both; target_len = 1 for both
        let in_len = Tensor::<B, 1, Int>::from_ints([2, 2], &device);
        let tgt_len = Tensor::<B, 1, Int>::from_ints([1, 1], &device);

        // batch of 2 with different true lengths
        let mut logits = Tensor::<B, 3>::zeros([2, t, vocab], &device);
        // sample 0: strong blank, then 'A'
        logits = logits.slice_assign(
            [0..1, 0..1, 0..1],
            Tensor::<B, 3>::from_floats([[[5.0]]], &device),
        );
        logits = logits.slice_assign(
            [0..1, 1..2, 1..2],
            Tensor::<B, 3>::from_floats([[[5.0]]], &device),
        );
        // sample 1: strong blank, then 'B'
        logits = logits.slice_assign(
            [1..2, 0..1, 0..1],
            Tensor::<B, 3>::from_floats([[[5.0]]], &device),
        );
        logits = logits.slice_assign(
            [1..2, 1..2, 2..3],
            Tensor::<B, 3>::from_floats([[[5.0]]], &device),
        );

        let losses = loss.forward_no_reduction(logits, targets, in_len, tgt_len);
        // both should be small and similar
        let loss_0 = losses.clone().slice([0..1]).into_scalar();
        let loss_1 = losses.clone().slice([1..2]).into_scalar();

        println!(
            "\nLoss 0 = {:?} | Loss 1 = {:?} | Threshold = {:?}\n",
            loss_0, loss_1, threshold
        );
        assert!(
            loss_0 < threshold && loss_1 < threshold,
            "padding not respected"
        );
    }

    #[test]
    fn ctc_monotonicity_more_confidence_lower_loss() {
        let device = Default::default();
        let blank_id = 0usize;
        let vocab = 3usize; // {blank = 0, 'A' = 1, 'B' = 2}
        let t = 2usize;
        let reduction = Reduction::Mean;

        let loss = CtcLossConfig::new()
            .with_blank_id(blank_id)
            .with_reduction(reduction.clone())
            .init();

        let targets = Tensor::<B, 2, Int>::from_ints([[1i64]], &device);
        let (in_len, tgt_len) = (
            Tensor::<B, 1, Int>::from_ints([t as i64], &device),
            Tensor::<B, 1, Int>::from_ints([1i64], &device),
        );

        // case A: modest confidence
        let mut logits_a = Tensor::<B, 3>::zeros([1, t, vocab], &device);
        logits_a = logits_a.slice_assign(
            [0..1, 0..1, 0..1],
            Tensor::<B, 3>::from_floats([[[1.0]]], &device),
        );
        logits_a = logits_a.slice_assign(
            [0..1, 1..2, 1..2],
            Tensor::<B, 3>::from_floats([[[1.0]]], &device),
        );

        // case B: higher confidence (increase same correct logits)
        let mut logits_b = logits_a.clone();
        logits_b = logits_b.slice_assign(
            [0..1, 0..1, 0..1],
            Tensor::<B, 3>::from_floats([[[3.0]]], &device),
        );
        logits_b = logits_b.slice_assign(
            [0..1, 1..2, 1..2],
            Tensor::<B, 3>::from_floats([[[3.0]]], &device),
        );

        let loss_a = loss
            .forward(logits_a, targets.clone(), in_len.clone(), tgt_len.clone())
            .into_scalar();

        let loss_b = loss
            .forward(logits_b, targets, in_len, tgt_len)
            .into_scalar();

        println!("\nLoss a = {:?} | Loss b = {:?}\n", loss_a, loss_b);
        assert!(loss_b < loss_a, "increasing confidence should lower loss");
    }

    #[test]
    fn ctc_reduction_avg_matches_external_computed_avg() {
        let device = Default::default();
        let blank_id = 0usize;
        let vocab = 3usize; // {blank = 0, 'A' = 1, 'B' = 2}
        let t = 4usize;
        let threshold = 1e-6;
        let reduction = Reduction::Mean;

        let loss = CtcLossConfig::new()
            .with_blank_id(blank_id)
            .with_reduction(reduction.clone())
            .init();

        let targets = Tensor::<B, 2, Int>::from_ints([[1i64], [2i64]], &device);
        let (in_len, tgt_len) = tensorize_lengths(2, t, 1, &device);

        let mut logits = Tensor::<B, 3>::zeros([2, t, vocab], &device);
        logits = logits.slice_assign(
            [0..1, 0..1, 0..1],
            Tensor::<B, 3>::from_floats([[[4.0]]], &device),
        );
        logits = logits.slice_assign(
            [0..1, 1..2, 1..2],
            Tensor::<B, 3>::from_floats([[[4.0]]], &device),
        );
        logits = logits.slice_assign(
            [1..2, 0..1, 0..1],
            Tensor::<B, 3>::from_floats([[[4.0]]], &device),
        );
        logits = logits.slice_assign(
            [1..2, 1..2, 2..3],
            Tensor::<B, 3>::from_floats([[[4.0]]], &device),
        );

        let avg_internal_loss = loss
            .forward(
                logits.clone(),
                targets.clone(),
                in_len.clone(),
                tgt_len.clone(),
            )
            .into_scalar();

        let losses = loss.forward_no_reduction(
            logits.clone(),
            targets.clone(),
            in_len.clone(),
            tgt_len.clone(),
        );
        let loss_0 = losses.clone().slice([0..1]).into_scalar();
        let loss_1 = losses.clone().slice([1..2]).into_scalar();
        let avg_external_loss = (loss_0 + loss_1) / 2.0;

        println!(
            "\nInternal avg loss = {:?} | External avg loss = {:?}\n",
            avg_internal_loss, avg_external_loss
        );
        assert!(
            (avg_internal_loss - avg_external_loss) < threshold,
            "internally computed avg loss should be identical to externally computed avg loss",
        );
    }

    #[test]
    fn ctc_reduction_sum_matches_external_computed_sum() {
        let device = Default::default();
        let blank_id = 0usize;
        let vocab = 3usize; // {blank = 0, 'A' = 1, 'B' = 2}
        let t = 4usize;
        let threshold = 1e-6;
        let reduction = Reduction::Sum;

        let loss = CtcLossConfig::new()
            .with_blank_id(blank_id)
            .with_reduction(reduction.clone())
            .init();

        let targets = Tensor::<B, 2, Int>::from_ints([[1i64], [2i64]], &device);
        let (in_len, tgt_len) = tensorize_lengths(2, t, 1, &device);

        let mut logits = Tensor::<B, 3>::zeros([2, t, vocab], &device);
        logits = logits.slice_assign(
            [0..1, 0..1, 0..1],
            Tensor::<B, 3>::from_floats([[[4.0]]], &device),
        );
        logits = logits.slice_assign(
            [0..1, 1..2, 1..2],
            Tensor::<B, 3>::from_floats([[[4.0]]], &device),
        );
        logits = logits.slice_assign(
            [1..2, 0..1, 0..1],
            Tensor::<B, 3>::from_floats([[[4.0]]], &device),
        );
        logits = logits.slice_assign(
            [1..2, 1..2, 2..3],
            Tensor::<B, 3>::from_floats([[[4.0]]], &device),
        );

        let sum_internal_loss = loss
            .forward(
                logits.clone(),
                targets.clone(),
                in_len.clone(),
                tgt_len.clone(),
            )
            .into_scalar();

        let losses = loss.forward_no_reduction(
            logits.clone(),
            targets.clone(),
            in_len.clone(),
            tgt_len.clone(),
        );
        let loss_0 = losses.clone().slice([0..1]).into_scalar();
        let loss_1 = losses.clone().slice([1..2]).into_scalar();
        let sum_external_loss = loss_0 + loss_1;

        println!(
            "\nInternal sum loss = {:?} | External sum loss = {:?}\n",
            sum_internal_loss, sum_external_loss
        );
        assert!(
            (sum_internal_loss - sum_external_loss) < threshold,
            "internally computed sum loss should be identical to externally computed sum loss",
        );
    }
}
