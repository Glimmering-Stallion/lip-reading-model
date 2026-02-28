// Modules

#![recursion_limit = "512"]
pub mod pipeline;
pub mod training;
// pub mod inference;
pub mod vsrm;
pub mod ctc;
pub mod context;
pub mod utils;
pub mod vocab;



pub mod prelude {
    pub use crate::{
        context::Context,
        utils::{
            log_sum_exp_2_scalar,
            log_sum_exp_2_tensor,
            log_sum_exp_3_scalar,
            log_sum_exp_3_tensor,
            mean,
            std_dev,
        },
        vocab::{
            TokenMap,
            BLANK_ID,
            VOCAB,
            VOCAB_SIZE,
        },
        pipeline::{
            io::{
                extract_slr_corpus,
                stream_corpus_lines,
            },
            adapters::grid::{GridDataset},
            batcher::{
                Batch,
                VsrmItem,
                VsrmBatcher,
            },
        },
        training::{
            learner::{
                VsrmLearnerConfig,
                train,
            },
            trainer,
            metrics::{
                VsrmStepOutput,
                VsrmMetricInput,
                CtcCharErrorRate,
                CtcWordErrorRate,
            },
        },
        vsrm::{
            VsrModel,
            VsrModelConfig,
        },
        ctc::lm::{
            Ngram,
            NgramConfig,
        },
    };
}