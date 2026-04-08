//! Text index contracts.

#[derive(Debug, Clone, PartialEq)]
pub struct TextSearchHit {
    pub id: String,
    pub score: f32,
}
