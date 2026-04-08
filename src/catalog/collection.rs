//! Lightweight catalog-facing collection wrapper.

use crate::catalog::schema::CollectionSchema;

#[derive(Debug, Clone)]
pub struct CollectionDefinition {
    pub schema: CollectionSchema,
}

impl CollectionDefinition {
    #[must_use]
    pub fn new(schema: CollectionSchema) -> Self {
        Self { schema }
    }
}
