// Modules

#![recursion_limit = "512"]
pub mod pipeline;
pub mod training;
// pub mod inference;
pub mod vsrm;
pub mod ctc;
pub mod utils;
pub mod vocab;



pub mod prelude {
    pub use crate::{
        pipeline::{
            io::{
                extract_grid_corpus,
                extract_slr_corpus,
                load_grid_corpus,
                stream_corpus_lines,
                stream_jsonl_gz,
                stream_txt_lines,
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
    };
}