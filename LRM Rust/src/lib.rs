#![recursion_limit = "2048"]

pub mod cli;
pub mod pipeline;
pub mod training;
pub mod inference;
pub mod vsrm;
pub mod ctc;
pub mod context;
pub mod utils;
pub mod vocab;



pub mod prelude {
    pub type ESS = Box<dyn std::error::Error + Send + Sync>;

    pub use crate::{
        context::Context,
        utils::{
            io_err,
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
            SPACE_ID,
            VOCAB,
            VOCAB_SIZE,
        },
        pipeline::{
            io::{
                load_json,
                save_json,
                extract_slr_corpus,
                stream_corpus_lines,
            },
            tracker::{
                LipTrackerBackend,
                TrackerConfig,
                HaarTrackerConfig,
            },
            adapters::grid::{GridDataset},
            batcher::{
                Batch,
                VsrmItem,
                VsrmBatcher,
            },
        },
        vsrm::{
            VsrModel,
            VsrModelConfig,
        },
        ctc::{
            ctc_decode::CtcDecodeType,
            lm::{
                Ngram,
                NgramConfig,
                build_or_load_ngram_lm,
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
        inference::{
            predictor::{
                VsrmPredictorConfig,
                InferenceSession,
                SlidingWindow,
                infer,
            },
            loader::{
                open_camera,
                load_frame,
            },
            overlay::OverlayRenderer,
        },
    };
}