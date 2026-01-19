// Modules

pub mod ctc;
pub mod model;
pub mod train;
pub mod utils;
pub mod vocab;
pub mod data_handler;



pub mod prelude {
    pub use crate::vocab::{TokenMap, VOCAB, VOCAB_SIZE, BLANK_ID};
    pub use crate::model::LRModel;
    pub use crate::ctc::lm::{NgramLM, NgramLMConfig};
    pub use crate::utils::{
        mean,
        std_dev,
        log_sum_exp_2_tensor,
        log_sum_exp_3_tensor,
        log_sum_exp_2_scalar,
        log_sum_exp_3_scalar,
    };
    pub use crate::data_handler::{load_data, stream_txt_lines, stream_jsonl_gz, stream_corpus_lines, extract_slr_dataset, extract_grid_data};
}