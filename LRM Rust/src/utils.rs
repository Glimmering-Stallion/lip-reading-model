//! General-purpose utilities for the lip-reading model.
//!
//! This module provides statistical helpers (mean, std_dev), numerically stable
//! log-sum-exp operations for scalars and tensors, Levenshtein edit distance for
//! sequence comparison, and the `io_err` builder for constructing app-level
//! errors from messages and `io::ErrorKind`.

// imports
use crate::prelude::ESS;
use burn::tensor::{Tensor, backend::Backend};
use num_traits::Float;
use std::cmp::min;
use std::io::{Error, ErrorKind};

// #[inline]
// pub fn mean(data: &Vec<f32>) -> f32 {
//     let count = data.len() as f32;
//     let sum: f32 = data.iter().sum();
//     sum / count
// }

// #[inline]
// pub fn std_dev(data: &Vec<f32>) -> f32 {
//     let count = data.len();
//     let mean = mean(data);
//     let variance: f32 = data.iter().map(|x| x - mean).map(|x| x * x).sum::<f32>() / count as f32;
//     variance.sqrt()
// }



#[inline]
pub fn mean<T: Float>(data: &[T]) -> T {
    let count = T::from(data.len()).unwrap();
    let sum = data.iter().copied().fold(T::zero(), |acc, x| acc + x);
    sum / count
}

#[inline]
pub fn std_dev<T: Float>(data: &[T]) -> T {
    let count = T::from(data.len()).unwrap();
    let mean = mean(data);
    let variance = data
        .iter()
        .copied()
        .fold(T::zero(), |acc, x| acc + ((x - mean) * (x - mean)))
        / count;
    variance.sqrt()
}



#[inline]
pub fn log_sum_exp_2_scalar(a: f32, b: f32) -> f32 {
    let max = a.max(b);
    let sum = (a - max).exp() + (b - max).exp();
    let lse = max + sum.ln();

    // handle pairwise (-inf, -inf) cases (to avoid NaNs from -inf - -inf)
    if lse.is_nan() { f32::NEG_INFINITY } else { lse }
}

#[inline]
pub fn log_sum_exp_3_scalar(a: f32, b: f32, c: f32) -> f32 {
    log_sum_exp_2_scalar(log_sum_exp_2_scalar(a, b), c)
}



#[inline]
pub fn log_sum_exp_2_tensor<B: Backend, const D: usize>(
    a: Tensor<B, D>,
    b: Tensor<B, D>,
) -> Tensor<B, D> {
    let max = a.clone().max_pair(b.clone()); // element-wise maxxing
    let sum = (a.sub(max.clone())).exp().add((b.sub(max.clone())).exp());
    let lse = max.add(sum.log());

    // handle pairwise (-inf, -inf) cases (to avoid NaNs from -inf - -inf)
    let nan_mask = lse.clone().is_nan();
    lse.mask_fill(nan_mask, f32::NEG_INFINITY)
}

#[inline]
pub fn log_sum_exp_3_tensor<B: Backend, const D: usize>(
    a: Tensor<B, D>,
    b: Tensor<B, D>,
    c: Tensor<B, D>,
) -> Tensor<B, D> {
    let max = a.clone().max_pair(b.clone()).max_pair(c.clone());
    let sum =
            ((a.sub(max.clone())).exp())
        .add((b.sub(max.clone())).exp())
        .add((c.sub(max.clone())).exp());
    let lse = max.add(sum.log());

    // handle pairwise (-inf, -inf) cases (to avoid NaNs from -inf - -inf)
    let nan_mask = lse.clone().is_nan();
    lse.mask_fill(nan_mask, f32::NEG_INFINITY)
}



/// Computes Levenshtein (edit) distance between two sequences.
///
/// ### Params:
/// - `seq1`: Predicted sequence of items (IDs/words).
/// - `seq2`: Ground truth sequence of items.
///
/// ### Returns:
/// Total count of insertions, deletions, and substitutions.
#[inline]
pub fn levenshtein<T: PartialEq>(seq1: &[T], seq2: &[T]) -> usize {
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



/// Builds an `ESS` from a message and `io::ErrorKind`.
///
/// Wraps the message as an `io::Error`, then converts to the shared app error type.
///
/// `ESS` (`Box<dyn Error + Send + Sync>`) is used project-wide so errors can propagate
/// across threads (e.g. parallel data loading, training workers).
///
/// Use any `io::ErrorKind` (e.g. `InvalidInput` for CLI validation, `Other` for I/O).
///
/// ### Params:
/// - `msg`: Error message (any type implementing `Into<String>`).
/// - `kind`: `io::ErrorKind` for the error.
///
/// ### Returns:
/// An `ESS` suitable for `Result` propagation.
#[inline]
pub fn io_err(msg: impl Into<String>, kind: ErrorKind) -> ESS {
    Error::new(kind, msg.into()).into()
}
