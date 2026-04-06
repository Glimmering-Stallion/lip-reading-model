//! Vocabulary definitions and character-to-index mappings.
//!
//! This module handles the bidirectional mapping between natural language characters
//! and numerical IDs used by the model. It includes the character set for the
//! Visual Speech Recognition (VSR) task and a `TokenMap` utility for
//! sequence conversion.

// imports
use std::collections::HashMap;



// pub const VOCAB: &str = "abcdefghijklmnopqrstuvwxyz'?!0123456789 _"; // complex char set
pub const VOCAB: &str = "abcdefghijklmnopqrstuvwxyz _"; // simple char set
pub const VOCAB_SIZE: usize = VOCAB.len();
pub const BLANK_ID: usize = VOCAB_SIZE - 1;
pub const SPACE_ID: usize = VOCAB_SIZE - 2;

#[derive(Clone)]
pub struct TokenMap {
    id_of: HashMap<char, usize>,
    char_of: Vec<char>,
}

impl TokenMap {
    /// Builds a bidirectional char <--> ID map from the given vocabulary string.
    pub fn new(vocab: &str) -> Self {
        let vocab: Vec<char> = vocab.chars().collect();

        // character to numerical index map and vice versa
        let mut id_of = HashMap::new();
        for (idx, ch) in vocab.iter().enumerate() {
            id_of.insert(*ch, idx);
        }

        Self {
            id_of,
            char_of: vocab,
        }
    }

    /// Returns the ID for a character, or `None` if not in vocabulary.
    pub fn id_of(&self, ch: char) -> Option<usize> { self.id_of.get(&ch).copied() }
    /// Returns the character for an ID, or `None` if out of range.
    pub fn char_of(&self, id: usize) -> Option<char> { self.char_of.get(id).copied() }
    /// Converts a sequence of chars to IDs; returns `None` if any char is out of vocabulary.
    pub fn chars_to_ids(&self, seq: &[char]) -> Option<Vec<usize>> { seq.iter().map(|&ch| self.id_of(ch)).collect() }
    /// Converts a sequence of IDs to chars; returns `None` if any ID is out of range.
    pub fn ids_to_chars(&self, seq: &[usize]) -> Option<Vec<char>> { seq.iter().map(|&id| self.char_of(id)).collect() }
}
