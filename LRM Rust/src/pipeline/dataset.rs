//! Dataset source abstractions and partitioning logic.
//! 
//! This module provides the `DatasetSplit` utility to create deterministic training,
//! validation, and testing sets from a base dataset, as well as the `DatasetSource`
//! enum to manage various data origins (e.g., GRID, LRW, etc.)



use burn::data::dataset::Dataset;
use rand::{
    seq::SliceRandom,
    SeedableRng,
    rngs::StdRng,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;



#[derive(Clone)]
pub struct DatasetSplit<I> {
    dataset: Arc<dyn Dataset<I>>,
    indices: Vec<usize>,
}



#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DatasetStats {
    pub mean: f32,
    pub std_dev: f32,
}



#[derive(Debug, Clone, Copy)]
pub enum DatasetSource {
    Grid,
    // Lrw, // maybe add later
}



impl<I> DatasetSplit<I> {
    /// creates a new split from a base dataset, for example:
    /// train (evaluation as remainder): 0.8 (80%),
    /// validation (test as remainder): 0.1 (10%)
    /// params:
    /// - dataset: base dataset to split
    /// - train_pct: percentage of total data to allocate for training, rest for evaluation (e.g. 0.8 for 80%)
    /// - valid_pct: percentage of total data to allocate for validation, rest for testing (e.g. 0.1 for 10%)
    /// - seed: random seed for deterministic shuffling before splitting
    /// returns: a tuple of (train_split, eval_split, test_split) DatasetSplit instances
    pub fn split(
        dataset: Arc<dyn Dataset<I>>,
        train_pct: f32,
        valid_pct: f32,
        seed: u64,
    ) -> (Self, Self, Self) {
        let total = dataset.len();
        let mut indices: Vec<usize> = (0..total).collect();
        
        // deterministic shuffle
        let mut rng = StdRng::seed_from_u64(seed);
        indices.shuffle(&mut rng);

        // find split index points from percentages
        let train_end = (total as f32 * train_pct).round() as usize;
        let eval_end = train_end + (total as f32 * valid_pct).round() as usize;

        // create DatasetSplit instances based on split indices
        let (train, valid, test) = (
            Self { dataset: dataset.clone(), indices: indices[0..train_end].to_vec() },
            Self { dataset: dataset.clone(), indices: indices[train_end..eval_end].to_vec() },
            Self { dataset: dataset.clone(), indices: indices[eval_end..].to_vec() },
        );

        (train, valid, test)
    }
}



impl<I: Send + Sync> Dataset<I> for DatasetSplit<I> {
    fn get(&self, index: usize) -> Option<I> {
        let original_index = self.indices.get(index)?;
        self.dataset.get(*original_index)
    }

    fn len(&self) -> usize {
        self.indices.len()
    }
}



impl DatasetStats {
    pub fn new(mean: f32, std_dev: f32) -> Self {
        Self { mean, std_dev }
    }
}



impl DatasetSource {
    pub fn tag(&self) -> &'static str {
        match self {
            DatasetSource::Grid => "grid",
            // DatasetSource::Lrw => "lrw", // maybe add later
        }
    }
}