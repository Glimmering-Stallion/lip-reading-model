// Modules

pub mod io;
pub mod batcher;
pub mod preprocessors;
pub mod train;
pub mod learner;
pub mod model;
pub mod ctc;
pub mod utils;
pub mod vocab;



pub mod prelude {
    pub use crate::{
        io::{
            extract_grid_corpus, extract_slr_corpus, load_grid_corpus, stream_corpus_lines,
            stream_jsonl_gz, stream_txt_lines,
        },
        preprocessors::grid::{GridDataset},
        batcher::{Batch, VsrmItem, VsrmBatcher},
        train::{train_epoch, train_loop},
        model::VsrModel,
        ctc::lm::{Ngram, NgramConfig},
        utils::{
            log_sum_exp_2_scalar, log_sum_exp_2_tensor, log_sum_exp_3_scalar, log_sum_exp_3_tensor,
            mean, std_dev,
        },
        vocab::{TokenMap, BLANK_ID, VOCAB, VOCAB_SIZE},
    };
}



#[derive(Debug)]
pub enum DatasetSplit {
    Train,
    Val,
    Test,
}