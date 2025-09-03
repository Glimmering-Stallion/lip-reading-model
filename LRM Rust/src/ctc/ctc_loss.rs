// Connectionist Temporal Classification (CTC) implementation

// CTC is used for sequence-to-sequence tasks where input and output alignments is unknown

// if using beam search as the decoder, expect O(DWB * log(WB)) complexity
// where D = depth (timesteps), W = beam width (# of kept hypotheses), B = branch factor (vocab size)



// custom imports
use crate::utils::{log_sum_exp_2, log_sum_exp_3};
// imports
use burn::{
    config::Config,
    module::Ignored,
    nn::loss::Reduction,
    prelude::Int,
    tensor::{activation::log_softmax, backend::Backend, Shape, Tensor},
};



#[derive(Debug)]
pub struct CtcLoss {
    pub blank_id: usize,      // index of blank token in vocab
    pub reduction: Ignored<Reduction>, // reduction method (mean, sum, none)
}



impl CtcLoss {
    pub fn forward<B: Backend>(
        &self,
        inputs: Tensor<B, 3>, // [N, T, Vocab] (raw, unnormalized predictions)
        targets: Tensor<B, 2, Int>, // [N, T_max] (target sequences)
        input_lengths: Tensor<B, 1, Int>, // [N]
        target_lengths: Tensor<B, 1, Int>, // [N]
    ) -> Tensor<B, 1> {
        let tensor = self.forward_no_reduction(inputs, targets, input_lengths, target_lengths);
        match &self.reduction.0 {
            Reduction::Mean => tensor.mean(),
            Reduction::Sum => tensor.sum(),
            other => panic!("{other:?} reduction is not supported"),
        }
    }

    /// like `forward`, but without reduction (returns one loss per batch item)
    pub fn forward_no_reduction<B: Backend>(
        &self,
        inputs: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
        input_lengths: Tensor<B, 1, Int>,
        target_lengths: Tensor<B, 1, Int>,
    ) -> Tensor<B, 1> {
        let device = inputs.device();
        let [n, _, vocab_size] = inputs.dims();
        let log_probs = log_softmax(inputs, 2);
        let mut losses: Vec<Tensor<B, 1>> = Vec::with_capacity(n);

        let input_lengths_to_host = input_lengths
            .to_data()
            .convert::<i64>()
            .as_slice::<i64>()
            .unwrap()
            .to_vec();
        let target_lengths_to_host = target_lengths
            .to_data()
            .convert::<i64>()
            .as_slice::<i64>()
            .unwrap()
            .to_vec();

        for sample in 0..n {
            let sample_input_length = input_lengths_to_host[sample] as usize;
            let sample_target_length = target_lengths_to_host[sample] as usize;

            let sample_log_probs = log_probs
                .clone()
                .slice([sample..(sample + 1), 0..sample_input_length, 0..vocab_size])
                .reshape([sample_input_length, vocab_size]);
            let sample_target = targets
                .clone()
                .slice([sample..(sample + 1), 0..sample_target_length])
                .reshape([sample_target_length]);

            let sample_loss =
                self.per_sample_loss(sample_log_probs, sample_target, vocab_size, &device);
            losses.push(sample_loss)
        }

        Tensor::cat(losses, 0)
    }

    /// compute neg log-likelihood for single sample
    /// loss computed via: logits --> log softmax --> forward DP --> loss
    /// params:
    /// - log_probs: log-probabilities for each timestep and vocab symbol given by model [T, V]
    /// - targets: ground-truth target sequence of symbol int IDs [L]
    /// - vocab_size: size of vocabulary V
    /// - device: specifier for where tensors ops should be computed
    #[inline]
    fn per_sample_loss<B: Backend>(
        &self,
        log_probs: Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
        vocab_size: usize,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let blank = self.blank_id as i64;
        let timesteps = log_probs.dims()[0];
        // let sentinel_value = f32::NEG_INFINITY;
        let sentinel_value = -1e30;
        let neg_inf = Tensor::<B, 1>::full([1], sentinel_value, device);

        // original and blank-extended lengths of target sequence
        let l: usize = targets.dims()[0];
        let l_ext = 2 * l + 1;

        // make [T + 1] blanks and take [T] labels, padded to [T + 1]
        let blanks = Tensor::<B, 1, Int>::full([l + 1], blank, device);
        let labels = Tensor::<B, 1, Int>::cat(
            vec![
                targets.clone(),
                Tensor::<B, 1, Int>::full([1], blank, device),
            ],
            0,
        );

        // perform: stack --> transpose --> flatten
        // to obtain: [blank, y1, blank, y2, ..., blank, yT, blank] (IDs of blank-extended targets)
        let ext_ids = {
            let blank_label_pairs = Tensor::stack::<2>(vec![blanks, labels], 1);
            blank_label_pairs.reshape([2 * (l + 1)]).slice([0..l_ext])
        };

        // can-skip-by-1 & can-skip-by-2 validity masks
        // goal: (pos is odd) && (pos > 1) && (ext_ids[pos] != ext_ids[pos − 2]) && (ext_ids[pos] != blank)
        let ext_pos = Tensor::<B, 1, Int>::arange(0..(l_ext as i64), device); // blank-extended symbol positions
        let odd_mask = ext_pos.clone().remainder_scalar(2).not_equal_elem(0);
        let gt_1_mask = ext_pos.clone().greater_elem(1);
        let prev_2_ids = ext_ids.clone().roll_dim(-2, 0); // shift right by 2 on dim 0
        let neq_prev_2_ids_mask = ext_ids.clone().not_equal(prev_2_ids.clone());
        let not_blank_mask = ext_ids.clone().not_equal_elem(blank);
        let can_skip_1_mask = ext_pos.clone().greater_elem(0);
        let can_skip_2_mask = odd_mask.clone()
            .bool_and(gt_1_mask.clone())
            .bool_and(neq_prev_2_ids_mask.clone())
            .bool_and(not_blank_mask.clone());

        // curr/next transfer buffers of forward log-probs
        let mut curr_fwd = Tensor::<B, 1>::full(Shape::new([l_ext]), sentinel_value, device);
        let mut next_fwd = Tensor::<B, 1>::full(Shape::new([l_ext]), sentinel_value, device);

        // t = 0 (base case)
        // valid start positions: begin at first blank or symbol after it
        let log_probs_0_v = log_probs.clone().slice([0..1, 0..vocab_size]);
        if l_ext >= 1 {
            let id_0_0 = ext_ids.clone().slice([0..1]).unsqueeze_dim(0);
            let log_prob_0_0 = log_probs_0_v.clone().gather(1, id_0_0).reshape([1]);
            curr_fwd = curr_fwd.slice_assign([0..1], log_prob_0_0);
        }
        if l_ext >= 2 {
            let id_0_1 = ext_ids.clone().slice([1..2]).unsqueeze_dim(0);
            let log_prob_0_1 = log_probs_0_v.clone().gather(1, id_0_1).reshape([1]);
            curr_fwd = curr_fwd.slice_assign([1..2], log_prob_0_1);
        }

        // t >= 1 (recurrence case)
        for t in 1..timesteps {

            // reset buffer
            next_fwd = Tensor::<B, 1>::full_like(&next_fwd, sentinel_value);

            // grab chunk of log-probs of each symbol at current timestep
            let log_prob_row = log_probs.clone().slice([t..(t + 1), 0..vocab_size]);

            // probs at previous positions
            let prev_1_probs = curr_fwd.clone().roll_dim(-1, 0); // shift by 1
            let prev_2_probs = curr_fwd.clone().roll_dim(-2, 0); // shift by 2

            for pos in 0..l_ext {
                // possible actions at current timestep (based on what could have happened at previous frame)
                // - stay: log-prob of staying at same symbol in extended target sequence    (from same position)
                // - adv_1: log-prob of advancing by one symbol                              (from one position earlier)
                // - adv_2: log-prob of skipping blank to advance by two symbols             (from two positions earlier)
                let stay = curr_fwd.clone().slice([pos..(pos + 1)]);
                let adv_1 = neg_inf.clone().mask_where(
                    can_skip_1_mask.clone().slice([pos..(pos + 1)]),
                    prev_1_probs.clone().slice([pos..(pos + 1)]),
                );
                let adv_2 = neg_inf.clone().mask_where(
                    can_skip_2_mask.clone().slice([pos..(pos + 1)]),
                    prev_2_probs.clone().slice([pos..(pos + 1)]),
                );

                // compute accumulated log-prob for current path (in time-vocab grid) from all possible actions
                let path_log_prob = log_sum_exp_3(stay, adv_1, adv_2);

                // grab symbol log-prob at current position in extended target sequence given by model
                let sym_id = ext_ids.clone().slice([pos..(pos + 1)]).unsqueeze_dim(0);
                let sym_log_prob = log_prob_row.clone().gather(1, sym_id).reshape([1]);

                // prep updated collection of forward log-probs
                // combine log-prob of current path with log-prob of current symbol (from model output)
                next_fwd = next_fwd.slice_assign([pos..(pos + 1)], path_log_prob.add(sym_log_prob));
            }

            core::mem::swap(&mut curr_fwd, &mut next_fwd);
        }

        // end at last blank or symbol before it
        let end_1 = curr_fwd.clone().slice([(l_ext - 1)..l_ext]);
        let end_2 = if l_ext >= 2 {
            curr_fwd.clone().slice([(l_ext - 2)..(l_ext - 1)])
        } else {
            neg_inf.clone()
        };
        let total_log_prob = log_sum_exp_2(end_1.clone(), end_2.clone());

        -total_log_prob // loss
    }
}



#[derive(Debug, Config)]
pub struct CtcLossConfig {
    #[config(default = "0")]
    pub blank_id: usize, // id of blank token in vocab

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



// testing
#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        backend::ndarray::NdArray,
        prelude::Int,
        nn::loss::Reduction,
        tensor::Tensor,
    };
    type B = NdArray<f32>;

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
        // t = 1 strongly 'A' (id = 1)
        logits = logits.slice_assign(
            [0..1, 1..2, 1..2],
            Tensor::<B, 3>::from_floats([[[5.0]]], &device),
        );
        
        let loss = loss.forward(
            logits,
            targets,
            in_len,
            tgt_len
        ).into_scalar();

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
        logits_1 = logits_1.slice_assign([0..1, 0..1, 0..1], Tensor::<B, 3>::from_floats([[[4.0]]], &device)); // high blank at t = 0
        logits_1 = logits_1.slice_assign([0..1, 1..2, 1..2], Tensor::<B, 3>::from_floats([[[4.0]]], &device)); // high 'A' at t = 1
        logits_1 = logits_1.slice_assign([0..1, 2..3, 1..2], Tensor::<B, 3>::from_floats([[[4.0]]], &device)); // high 'A' at t = 2
        let loss_hostile = loss.forward(
            logits_1,
            targets.clone(),
            in_len_1,
            tgt_len_1
        ).into_scalar();

        // case 2 (friendly alignment, t = 4): can't skip, doesn't skip, and uses intermediate blank
        let (in_len_2, tgt_len_2) = tensorize_lengths(1, t_2, 2, &device);
        let mut logits_2 = Tensor::<B, 3>::zeros([1, t_2, vocab], &device);
        logits_2 = logits_2.slice_assign([0..1, 0..1, 0..1], Tensor::<B, 3>::from_floats([[[4.0]]], &device)); // high blank at t = 0
        logits_2 = logits_2.slice_assign([0..1, 1..2, 1..2], Tensor::<B, 3>::from_floats([[[4.0]]], &device)); // high 'A' at t = 1
        logits_2 = logits_2.slice_assign([0..1, 2..3, 0..1], Tensor::<B, 3>::from_floats([[[4.0]]], &device)); // high blank at t = 2
        logits_2 = logits_2.slice_assign([0..1, 3..4, 1..2], Tensor::<B, 3>::from_floats([[[4.0]]], &device)); // high 'A' at t = 3
        let loss_friendly = loss.forward(
            logits_2,
            targets.clone(),
            in_len_2,
            tgt_len_2
        )
        .into_scalar();

        // should be: case 1 loss > case 2 loss
        println!("\nLoss 1 = {:?} | Loss 2 = {:?}\n", loss_hostile, loss_friendly);
        assert!(loss_hostile.is_finite() && loss_friendly.is_finite());
        assert!(loss_hostile > loss_friendly, "hostile loss case (t = 3) should be higher than friendly loss case (t = 4)");
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

        let loss = loss.forward(
            logits,
            targets,
            in_len,
            tgt_len
        )
        .into_scalar();

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

        println!("\nLoss 0 = {:?} | Loss 1 = {:?} | Threshold = {:?}\n", loss_0, loss_1, threshold);
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

        let loss_a = loss.forward(
            logits_a,
            targets.clone(),
            in_len.clone(),
            tgt_len.clone()
        )
        .into_scalar();

        let loss_b = loss.forward(
            logits_b,
            targets,
            in_len,
            tgt_len
        )
        .into_scalar();

        println!("\nLoss a = {:?} | Loss b = {:?}\n", loss_a, loss_b);
        assert!(
            loss_b < loss_a,
            "increasing confidence should lower loss"
        );
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

        let avg_internal_loss = loss.forward(
            logits.clone(),
            targets.clone(),
            in_len.clone(),
            tgt_len.clone()
        )
        .into_scalar();

        let losses = loss.forward_no_reduction(
            logits.clone(),
            targets.clone(),
            in_len.clone(),
            tgt_len.clone()
        );
        let loss_0 = losses.clone().slice([0..1]).into_scalar();
        let loss_1 = losses.clone().slice([1..2]).into_scalar();
        let avg_external_loss = (loss_0 + loss_1) / 2.0;

        println!("\nInternal avg loss = {:?} | External avg loss = {:?}\n", avg_internal_loss, avg_external_loss);
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

        let sum_internal_loss = loss.forward(
            logits.clone(),
            targets.clone(),
            in_len.clone(),
            tgt_len.clone()
        )
        .into_scalar();

        let losses = loss.forward_no_reduction(
            logits.clone(),
            targets.clone(),
            in_len.clone(),
            tgt_len.clone()
        );
        let loss_0 = losses.clone().slice([0..1]).into_scalar();
        let loss_1 = losses.clone().slice([1..2]).into_scalar();
        let sum_external_loss = loss_0 + loss_1;

        println!("\nInternal sum loss = {:?} | External sum loss = {:?}\n", sum_internal_loss, sum_external_loss);
        assert!(
            (sum_internal_loss - sum_external_loss) < threshold,
            "internally computed sum loss should be identical to externally computed sum loss",
        );
    }
}
