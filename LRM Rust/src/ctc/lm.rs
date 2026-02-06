// Language Model (LM) for CTC decoding with beam search

// implement N-gram LM with backoff (e.g. Kneser-Ney smoothing)
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
    Ngram(NgramConfig),
    // NeuralLM(NeuralLMConfig), // maybe add later
}



impl LanguageModelConfig {
    pub fn init(&self) -> Box<dyn LanguageModel + Send + Sync> {
        match self {
            Self::Ngram(cfg) => Box::new(cfg.init()),
            // Self::NeuralLM(NeuralLMConfig) => Box::new(cfg.init()), // placeholder for future neural LM
        }
    }
}



pub trait LanguageModel: Debug + Send + Sync {
    fn score(&self, sequence: &[usize]) -> f32;
    fn perplexity(&self, data: Box<dyn Iterator<Item = Vec<usize>>>) -> f32;
    fn next_log_prob(&self, prefix: &[usize], next: usize) -> f32;
    fn clone_box(&self) -> Box<dyn LanguageModel + Send + Sync>;
}



#[derive(Config, Debug)]
pub struct NgramConfig {
    #[config(default = 3)]
    n: usize, // N-gram size

    #[config(default = 0)]
    vocab_size: usize,

    // none by default
    path: Option<String>,
}



impl NgramConfig {
    pub fn init(&self) -> Ngram {
        assert!((1..=5).contains(&self.n), "N-gram size ({}) must be in [1, 5]", self.n);

        if let Some(path) = &self.path {
            Ngram::load(path).unwrap()
        } else {
            Ngram {
                n: self.n,
                vocab_size: self.vocab_size,
                n_gram_counts: HashMap::new(),
                prefix_counts: HashMap::new(),
                unique_followers: HashMap::new(),
            }
        }
    }
}



#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ngram {
    n: usize,           // N-gram size
    vocab_size: usize,  // total vocab size

    // <sequence: count> maps
    n_gram_counts: HashMap<Vec<usize>, usize>,    // frequency counts of N-gram sequences                   (e.g. count of "the")
    prefix_counts: HashMap<Vec<usize>, usize>,    // frequency counts of (n-1)-gram prefixes                (e.g. count of "th")
    unique_followers: HashMap<Vec<usize>, usize>, // frequency counts of distinct followers after prefixes  (e.g. count of distinct followers after "th")
}



impl LanguageModel for Ngram {
    /// score a complete sequence using probability chain rule
    /// sums log-probabilities of each character given its history
    /// reminder: in log space for numerical stability
    /// params:
    /// - sequence: list of token IDs to evaluate
    /// returns: total log-probability of the sequence
    fn score(&self, sequence: &[usize]) -> f32 {
        if sequence.is_empty() { return 0.0; } // log-prob of empty sequence is 0

        let mut total_log_prob = 0.0;

        // compute log-prob of entire sequence with probability chain rule
        for i in 0..sequence.len() {
            let start = i.saturating_sub(self.n - 1);      // i - (n - 1)
            let prefix = &sequence[start..i];           // (n - 1) history
            let next = sequence[i];                        // next char
            total_log_prob += self.next_log_prob(prefix, next);   // total sum of log-probs
        }

        total_log_prob
    }

    /// calculate perplexity of LM on a dataset
    /// by finding average neg log-prob and exponentiates it
    /// metric for how well the probability distribution predicts a sample
    /// params:
    /// - data: iterator over validation sequences
    /// returns: perplexity score (lower means more confidence in unseen data)
    /// note: confidence ≠ accuracy
    fn perplexity(&self, data: Box<dyn Iterator<Item = Vec<usize>>>) -> f32 {
        let mut total_log_prob = 0.0;
        let mut total_tokens = 0;
        
        // accumulate per-sequence log-probs and tokens
        for sequence in data {
            total_log_prob += self.score(&sequence);
            total_tokens += sequence.len();
        }

        // debugging
        // println!("Total tokens in eval set: {}", total_tokens);

        if total_tokens == 0 { return f32::INFINITY; } // perplexity undefined for empty dataset

        // perplexity score
        let avg_log_prob = total_log_prob / total_tokens as f32;
        (-avg_log_prob).exp()
    }

    /// compute smoothed log-probability of a token given a prefix
    /// uses Witten-Bell smoothing to interpolate between MLE and backoff probs
    /// note: mutually recursive with "prob_backoff" to walk down N-gram orders
    /// params:
    /// - prefix: sequence of preceding token IDs
    /// - next: candidate next token ID
    /// returns: log-probability of 'next' following 'prefix'
    fn next_log_prob(&self, prefix: &[usize], next: usize) -> f32 {
        // get N-gram (prefix + next)
        let mut n_gram = prefix.to_vec();
        n_gram.push(next);

        // get counts and assign to floats with simpler var names
        let c = *self.n_gram_counts.get(&n_gram).unwrap_or(&0) as f32;
        let n = *self.prefix_counts.get(prefix).unwrap_or(&0) as f32; // not to be confused with N-gram size here
        let t = *self.unique_followers.get(prefix).unwrap_or(&0) as f32;
        
        // apply Witten-Bell smoothing (to handle unseen N-grams for robustness)
        let lambda = if n > 0.0 { t / (t + n) } else { 1.0 };       // backoff weight (based on num unique followers)
        let prob_mle = if n > 0.0 { c / n } else { 0.0 };           // MLE prob from current N-gram
        let prob_bo = self.prob_backoff(prefix, next);              // backoff prob from (n-1)-gram

        // final prob for current window: convert to log space (with max(1e-12) here to avoid log(0) situations)
        ((1.0 - lambda) * prob_mle + lambda * prob_bo).max(1e-12).ln()
    }

    fn clone_box(&self) -> Box<dyn LanguageModel + Send + Sync> {
        Box::new(self.clone())
    }
}



impl Ngram {
    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let mut f = File::create(path)?;
        bincode::serde::encode_into_std_write(self, &mut f, config::standard())?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn Error>> {
        let mut f = File::open(path)?;
        let lm: Self = bincode::serde::decode_from_std_read(&mut f, config::standard())?;
        Ok(lm)
    }

    /// train N-gram model on some corpus of text
    /// accumulates counts for all N-grams, prefixes, and unique followers
    /// params:
    /// - data: iterator over training sequences
    /// returns: none (updates internal state)
    pub fn train(&mut self, data: Box<dyn Iterator<Item = Vec<usize>>>) {
        // map of prefix keys and unique follower sets
        let mut uf_container: HashMap<Vec<usize>, HashSet<usize>> = HashMap::new();

        // count all N-grams in training data
        for sequence in data {       // loop through each training sequence (text lines)
            for i in 0..sequence.len() {  // loop through each position in sequence (chars)
                for j in 1..=self.n {     // loop through each N-gram size (1 to n)
                    if i + j <= sequence.len() { // out of bounds check
                        let n_gram = &sequence[i..(i + j)];
                        *self.n_gram_counts.entry(n_gram.to_vec()).or_insert(0) += 1;

                        // count prefixes and update unique follower counts
                        if n_gram.len() > 1 {
                            let (prefix, next) = (n_gram[..(n_gram.len() - 1)].to_vec(), *n_gram.last().unwrap());
                            *self.prefix_counts.entry(prefix.clone()).or_insert(0) += 1;
                            uf_container.entry(prefix).or_default().insert(next); // for given prefix, add "next" to its unique follower set
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

    /// calculate backoff probability for a token
    /// used when a higher-order N-gram has insufficient data or to smooth MLE
    /// provides "fallback" probability by calling "next_log_prob" with a reduced prefix
    /// falls back from N-gram -> (n-1)-gram -> ... -> uniform unigram distribution
    /// params:
    /// - prefix: the current context being reduced
    /// - next: the target token to score
    /// returns: probability (f32) in linear space (0.0 to 1.0)
    fn prob_backoff(&self, prefix: &[usize], next: usize) -> f32 {
        if prefix.is_empty() {
            // unigram base case
            1.0 / (self.vocab_size - 1) as f32 // uniform over vocab excluding blank
        } else {
            // recursive backoff
            let shorter_prefix = if prefix.len() > 1 { &prefix[1..] } else { &[] }; // drop first char
            self.next_log_prob(shorter_prefix, next).exp()
        }
    }
}



// tests
#[cfg(test)]
mod tests {
    use std::{env, path::Path};

    use crate::vocab::{
        TokenMap,
        VOCAB,
        VOCAB_SIZE,
    };
    use super::*;

    #[test]
    fn test_ngram_lm_score_with_known_counts() {
         // a tri-gram with vocab: a, b, c,  , _
        let mut ngram_lm = NgramConfig::new()
            .with_n(3)
            .with_vocab_size(5)
            .init();

        // example counts for testing:
        // N-gram counts arbitraryly chosen
        // prefix and unique follower counts consistent with chosen N-gram counts
        ngram_lm.n_gram_counts.insert(vec![0], 10);       // 'a'
        ngram_lm.n_gram_counts.insert(vec![1], 5);        // 'b'
        ngram_lm.n_gram_counts.insert(vec![0, 0], 6);     // 'aa'
        ngram_lm.n_gram_counts.insert(vec![0, 1], 4);     // 'ab'
        ngram_lm.n_gram_counts.insert(vec![1, 0], 2);     // 'ba'
        ngram_lm.n_gram_counts.insert(vec![0, 0, 0], 3);  // 'aaa'
        ngram_lm.n_gram_counts.insert(vec![0, 0, 1], 3);  // 'aab'
        ngram_lm.n_gram_counts.insert(vec![0, 1, 0], 2);  // 'aba'

        ngram_lm.prefix_counts.insert(vec![], 15);        // total unigrams
        ngram_lm.prefix_counts.insert(vec![0], 10);       // total bigrams starting with 'a'
        ngram_lm.prefix_counts.insert(vec![1], 5);        // total bigrams starting with 'b'
        ngram_lm.prefix_counts.insert(vec![0, 0], 6);     // total trigrams starting with 'aa'
        ngram_lm.prefix_counts.insert(vec![0, 1], 4);     // total trigrams starting with 'ab'

        ngram_lm.unique_followers.insert(vec![], 2);      // distinct unigrams
        ngram_lm.unique_followers.insert(vec![0], 2);     // distinct bigrams after 'a'
        ngram_lm.unique_followers.insert(vec![1], 1);     // distinct bigrams after 'b'
        ngram_lm.unique_followers.insert(vec![0, 0], 2);  // distinct trigrams after 'aa'
        ngram_lm.unique_followers.insert(vec![0, 1], 1);  // distinct trigrams after 'ab'

        // example sequence (aab – 001), v = 5 (a, b, c,  , _):
        // when i = 0 (a – 0):      c = 10,     n = 15,     t = 2,     λ = 2/17,    mle = 2/3,     bo = 1/4,        P(0)       = (1 - 2/17)(2/3) + (2/17)(1/4)  = 21/34   ≈ 0.6176
        // when i = 1 (aa – 00):    c = 6,      n = 10,     t = 2,     λ = 1/6,     mle = 3/5,     bo = P(0),       P(0 | 0)   = (1 - 1/6)(3/5) + (1/6)(21/34)  = 41/68   ≈ 0.6029
        // when i = 2 (aab – 001):  c = 3,      n = 6,      t = 2,     λ = 1/4,     mle = 1/2,     bo = P(1 | 0),   P(1 | 00)  = (1 - 1/4)(1/2) + (1/4)(79/204) = 385/816 ≈ 0.4718
        // where for P(1 | 0):      c = 4,      n = 10,     t = 2,     λ = 1/6,     mle = 2/5,     bo = P(1),       P(1 | 0)   = (1 - 1/6)(2/5) + (1/6)(11/34)  = 79/204  ≈ 0.3872
        // where for P(1):          c = 5,      n = 15,     t = 2,     λ = 2/17,    mle = 1/3,     bo = 1/4,        P(1)       = (1 - 2/17)(1/3) + (2/17)(1/4)  = 11/34   ≈ 0.3235
        // final log-prob: ln(0.6176 * 0.6029 * 0.4718) ≈ ln(0.1757) ≈ -1.7389

        // test scoring
        let seq = vec![0, 0, 1]; // "aab"
        let log_prob = ngram_lm.score(&seq);
        println!("Log-prob of sequence {:?}: {:.4}", seq, log_prob); // should print approx. -1.7389
        assert!(log_prob.is_finite());
    }

    #[test]
    fn test_ngram_lm_load_and_score_saved_model() {
        let rust_root = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
        let ngram_lm_path = Path::new(&rust_root)
            .join("models")
            .join("ngram_lm.bin");

        let ngram_lm = NgramConfig::new()
            .with_n(3)
            .with_vocab_size(VOCAB_SIZE)
            .with_path(ngram_lm_path.to_str().map(|s| s.to_string()))
            .init();

        let token_map = TokenMap::new(VOCAB);

        // test strings (that are converted to char vecs)
        let string_1 = "".chars().collect::<Vec<char>>();
        let string_2 = "the".chars().collect::<Vec<char>>();
        let string_3 = "xyz".chars().collect::<Vec<char>>();
        let string_4 = "happ".chars().collect::<Vec<char>>();
        let string_5 = "happy".chars().collect::<Vec<char>>();
        let string_6 = "happz".chars().collect::<Vec<char>>();

        // convert to numerical ID vec sequences
        let test_seq_1 = token_map.chars_to_ids(&string_1).unwrap(); // empty sequence
        let test_seq_2 = token_map.chars_to_ids(&string_2).unwrap(); // common word
        let test_seq_3 = token_map.chars_to_ids(&string_3).unwrap(); // gibberish sequence
        let test_seq_4 = token_map.chars_to_ids(&string_4).unwrap(); // base prefix sequence
        let test_seq_5 = token_map.chars_to_ids(&string_5).unwrap(); // high prob sequence
        let test_seq_6 = token_map.chars_to_ids(&string_6).unwrap(); // low prob sequence

        // pretrained N-gram scores
        let score_1 = ngram_lm.score(&test_seq_1);
        let score_2 = ngram_lm.score(&test_seq_2);
        let score_3 = ngram_lm.score(&test_seq_3);
        let score_4 = ngram_lm.score(&test_seq_4);
        let score_5 = ngram_lm.score(&test_seq_5);
        let score_6 = ngram_lm.score(&test_seq_6);

        // sanity checks
        println!("N-gram score for \'{:?}\': {}", string_1, score_1);
        println!("N-gram score for \'{:?}\': {}", string_2, score_2);
        println!("N-gram score for \'{:?}\': {}", string_3, score_3);
        println!("N-gram baseline score for \'{:?}\': {}", string_4, score_4);
        println!("N-gram score for \'{:?}\' following \'{:?}\': {}", string_5.last().unwrap(), string_4, (score_5 - score_4));
        println!("N-gram score for \'{:?}\' following \'{:?}\': {}", string_6.last().unwrap(), string_4, (score_6 - score_4));

        assert_eq!(score_1, 0.0);   // empty sequence should have log-prob of zero
        assert!(score_2 > score_3); // common word should have higher score than gibberish
        assert!((score_5 - score_4) > (score_6 - score_4)); // 'y' should have higher prob than 'z' after 'happ'
    }
}