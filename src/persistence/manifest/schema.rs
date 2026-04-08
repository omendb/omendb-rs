//! Manifest schema placeholders for the rewrite.

use crate::catalog::CollectionSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileManifest {
    pub path: PathBuf,
    pub bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrozenDenseSection {
    pub index: Option<FileManifest>,
    pub values: Option<FileManifest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseSection {
    pub postings: Option<FileManifest>,
    pub payload: Option<FileManifest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextSection {
    pub index_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivedManifest {
    pub dense: Option<FrozenDenseSection>,
    pub sparse: Option<SparseSection>,
    pub text: Option<TextSection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationManifest {
    pub generation: u64,
    pub files: Vec<FileManifest>,
    pub derived: DerivedManifest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionManifest {
    pub schema: CollectionSchema,
    pub generations: Vec<GenerationManifest>,
}
