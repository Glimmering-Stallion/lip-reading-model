// Language Model (LM) for CTC decoding with beam search

// implement n-gram LM with backoff (e.g. Kneser-Ney smoothing)
// allow a configurable n



// imports
use burn::{
    config::Config,
};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    fmt::Debug,
    error::Error,
};

use bincode::{self, config};
use serde::{
    Deserialize,
    Serialize,
};



#[derive(Config, Debug)]
pub enum LanguageModelConfig {
    NgramLM(NgramLMConfig),
    // maybe add later: NeuralLM(NeuralLMConfig),
}



impl LanguageModelConfig {
    pub fn init(&self) -> Box<dyn LanguageModel + Send + Sync> {
        match self {
            Self::NgramLM(cfg) => Box::new(cfg.init()),
            // Self::NeuralLM(NeuralLMConfig) => Box::new(cfg.init()),
        }
    }
}



pub trait LanguageModel: Debug {
    fn score(&self, sequence: &[usize]) -> f32;
    fn next_log_prob(&self, prefix: &[usize], next: usize) -> f32;
}



#[derive(Config, Debug)]
pub struct NgramLMConfig {
    #[config(default = 3)]
    n: usize, // n-gram size

    #[config(default = 0)]
    vocab_size: usize,
}



impl NgramLMConfig {
    pub fn init(&self) -> NgramLM {
        NgramLM {
            n: self.n,
            vocab_size: self.vocab_size,
            n_gram_counts: HashMap::new(),
            prefix_counts: HashMap::new(),
            unique_followers: HashMap::new(),
        }
    }
}



#[derive(Debug, Serialize, Deserialize)]
pub struct NgramLM {
    n: usize,   // n-gram size
    vocab_size: usize, // total vocab size

    // <sequence: count> maps
    n_gram_counts: HashMap<Vec<usize>, usize>, // frequency counts of n-grams sequences
    prefix_counts: HashMap<Vec<usize>, usize>, // total counts of (n-1)-gram prefixes
    unique_followers: HashMap<Vec<usize>, usize>, // num distinct followers after prefixes
}



impl LanguageModel for NgramLM {
    fn score(&self, sequence: &[usize]) -> f32 {
        if sequence.is_empty() { return 0.0; } // log-prob of empty sequence is 0

        // compute log-prob of entire sequence with chain rule
        let mut total_log_prob = 0.0;
        for i in 0..sequence.len() {
            let start = i.saturating_sub(self.n - 1);
            let prefix = &sequence[start..i]; // n - 1 history
            let next = sequence[i]; // next char
            total_log_prob += self.next_log_prob(prefix, next);
        }

        total_log_prob
    }

    /// return a log-prob bonus for extending a given prefix with a next char
    fn next_log_prob(&self, prefix: &[usize], next: usize) -> f32 {
        // get n-gram (prefix + next)
        let mut n_gram = prefix.to_vec();
        n_gram.push(next);

        // get counts
        let c = *self.n_gram_counts.get(&n_gram).unwrap_or(&0) as f32;
        let t = *self.unique_followers.get(prefix).unwrap_or(&0) as f32;
        let n = *self.prefix_counts.get(prefix).unwrap_or(&0) as f32;
        
        // apply Witten-Bell smoothing
        let lambda = if n > 0.0 { t / (t + n) } else { 1.0 };
        let prob_mle = if n > 0.0 { c as f32 / n } else { 0.0 };
        let prob_bo = self.prob_backoff(prefix, next);
        ((1.0 - lambda) * prob_mle + lambda * prob_bo).max(1e-12).ln() // avoid log(0) situations
    }
}



impl NgramLM {
    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let mut f = File::create(path)?;
        bincode::serde::encode_into_std_write(self, &mut f, config::standard())?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn Error>> {
        let mut f = File::open(path)?;
        let (lm, _len): (Self, usize) = bincode::serde::decode_from_std_read(&mut f, config::standard())?;
        Ok(lm)
    }

    pub fn train(&mut self, data: Box<dyn Iterator<Item = Vec<usize>>>) {
        // map of prefix keys and unique follower sets
        let mut uf_container: HashMap<Vec<usize>, HashSet<usize>> = HashMap::new();

        // count all n-grams in training data
        for sequence in data {
            for i in 0..sequence.len() {
                for j in 1..=self.n {
                    if i + j <= sequence.len() {
                        let n_gram = &sequence[i..(i + j)];
                        *self.n_gram_counts.entry(n_gram.to_vec()).or_insert(0) += 1;

                        // count prefixes and update unique follower counts
                        if n_gram.len() > 1 {
                            let (prefix, next) = (n_gram[..(n_gram.len() - 1)].to_vec(), *n_gram.last().unwrap());
                            *self.prefix_counts.entry(prefix.clone()).or_insert(0) += 1;
                            uf_container.entry(prefix).or_default().insert(next);
                        }
                    }
                }
            }
        }

        // finalize unique follower counts
        self.unique_followers = uf_container.into_iter()
            .map(|(k, s)| (k, s.len()))
            .collect();
    }

    fn prob_backoff(&self, prefix: &[usize], next: usize) -> f32 {
        if prefix.is_empty() {
            // unigram case
            1.0 / (self.vocab_size - 1) as f32 // uniform over vocab excluding blank
        } else {
            // recursive backoff
            let shorter_prefix = if prefix.len() > 1 { &prefix[1..] } else { &[] };
            self.next_log_prob(shorter_prefix, next).exp()
        }
    }
}



// tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ngram_lm_score_with_known_counts() {
        let mut lm = NgramLM {
            n: 3,
            vocab_size: 5, // e.g. a, b, c,  , _
            n_gram_counts: HashMap::new(),
            prefix_counts: HashMap::new(),
            unique_followers: HashMap::new(),
        };

        // example counts for testing
        lm.n_gram_counts.insert(vec![0], 10); // 'a'
        lm.n_gram_counts.insert(vec![1], 5);  // 'b'
        lm.n_gram_counts.insert(vec![0, 0], 6); // 'aa'
        lm.n_gram_counts.insert(vec![0, 1], 4); // 'ab'
        lm.n_gram_counts.insert(vec![1, 0], 2); // 'ba'
        lm.n_gram_counts.insert(vec![0, 0, 0], 3); // 'aaa'
        lm.n_gram_counts.insert(vec![0, 0, 1], 3); // 'aab'
        lm.n_gram_counts.insert(vec![0, 1, 0], 2); // 'aba'

        lm.prefix_counts.insert(vec![], 15); // total unigrams
        lm.prefix_counts.insert(vec![0], 10); // total bigrams starting with 'a'
        lm.prefix_counts.insert(vec![1], 5);  // total bigrams starting with 'b'
        lm.prefix_counts.insert(vec![0, 0], 6); // total trigrams starting with 'aa'
        lm.prefix_counts.insert(vec![0, 1], 4); // total trigrams starting with 'ab'

        lm.unique_followers.insert(vec![], 2); // distinct unigrams
        lm.unique_followers.insert(vec![0], 2); // distinct bigrams after 'a'
        lm.unique_followers.insert(vec![1], 1); // distinct bigrams after 'b'
        lm.unique_followers.insert(vec![0, 0], 2); // distinct trigrams after 'aa'
        lm.unique_followers.insert(vec![0, 1], 1); // distinct trigrams after 'ab'

        // test scoring
        let seq = vec![0, 0, 1]; // "aab"
        let log_prob = lm.score(&seq);
        println!("Log-prob of sequence {:?}: {:.4}", seq, log_prob);
        assert!(log_prob.is_finite());
    }

    #[test]
    fn test_ngram_lm_load_and_score_saved_model() {
        todo!()
    }
}