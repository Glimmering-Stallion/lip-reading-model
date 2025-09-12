// Define vocabulary and bidirectional token-id mapping



// imports
use std::{
    collections::HashMap,
};



pub const VOCAB: &str = "abcdefghijklmnopqrstuvwxyz'?!0123456789 _";
pub const BLANK_ID: usize = VOCAB.len() - 1;



pub struct TokenMap {
    id_of: HashMap<char, usize>,
    char_of: Vec<char>,
}

impl TokenMap {
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

    // helpers for conversions
    // id_of()/char_of():               converts between single symbols
    // chars_to_ids()/ids_to_chars():   converts between sequences of symbols
    pub fn id_of(&self, ch: char) -> Option<usize> { self.id_of.get(&ch).copied() }
    pub fn char_of(&self, id: usize) -> Option<char> { self.char_of.get(id).copied() }
    pub fn chars_to_ids(&self, seq: Vec<char>) -> Option<Vec<usize>> { seq.iter().map(|&ch| self.id_of(ch)).collect() }
    pub fn ids_to_chars(&self, seq: Vec<usize>) -> Option<Vec<char>> { seq.iter().map(|&id| self.char_of(id)).collect() }
}