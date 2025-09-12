// Language Model (LM) for CTC decoding with beam search



// imports
use burn::{
    config::Config,
    module::Module,
    tensor::backend::Backend,
};



#[derive(Config, Debug)]
pub struct LanguageModelConfig {
    pub n: usize,
    // other LM params
}



impl LanguageModelConfig {
    pub fn init(&self) -> LanguageModel {
        // initialize LM
        todo!()
    }
}



#[derive(Debug)]
pub struct LanguageModel;



impl LanguageModel {
    pub fn score(&self, sequence: &[usize]) -> f64 {
        // return log-prob
        todo!()
    }

    /// return a log-prob bonus for extending `prefix` with `token`.
    pub fn next_log_prob(&self, _prefix: &[usize], _token: usize) -> f32 { 0.0 }
}