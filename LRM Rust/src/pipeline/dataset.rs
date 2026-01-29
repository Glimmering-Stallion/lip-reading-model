// Dataset source abstraction/management for VSRM data ingestion



#[derive(Debug)]
pub enum DatasetSplit {
    Train,
    Val,
    Test,
}



#[derive(Debug, Clone, Copy)]
pub enum DatasetSource {
    Grid,
    // Lrw, // maybe add later
}



impl DatasetSource {
    pub fn tag(&self) -> &'static str {
        match self {
            DatasetSource::Grid => "grid",
            // DatasetSource::Lrw => "lrw", // maybe add later
        }
    }
}