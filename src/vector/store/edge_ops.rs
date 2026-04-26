//! Edge operations on VectorStore.
//!
//! Public API for the typed directed edge graph embedded in VectorStore.

use super::VectorStore;
use super::edge_store::{Edge, EdgeDirection, Subgraph, TraversalHit};
use anyhow::Result;
use serde_json::Value as JsonValue;

impl VectorStore {
    /// Enable graph support on this store.
    ///
    /// This is the explicit runtime opt-in for edge operations on stores that
    /// were created without `schema.graph`.
    pub fn enable_graph(&mut self) -> Result<()> {
        {
            let mut graph_schema = self.graph_schema.write();
            match graph_schema.as_mut() {
                Some(schema) => {
                    schema.enabled = true;
                }
                None => {
                    *graph_schema = Some(super::default_graph_schema());
                }
            }
        }

        if self.edge_store.read().is_none() {
            *self.edge_store.write() = Some(super::edge_store::EdgeStore::new());
        }

        if let Some(ref storage) = self.storage {
            let schema = self.schema();
            let mut storage = storage.write();
            storage.set_schema(schema)?;
            storage.sync()?;
        }

        Ok(())
    }

    /// Ensure edge graph storage is enabled before mutating graph state.
    fn ensure_edges_enabled(&mut self) -> Result<()> {
        {
            let graph_schema = self.graph_schema.read();
            match graph_schema.as_ref() {
                Some(schema) if schema.enabled => {}
                _ => {
                    anyhow::bail!(
                        "edge operations require graph.enabled=true in the collection schema or an explicit enable_graph() call"
                    );
                }
            }
        }

        if self.edge_store.read().is_none() {
            *self.edge_store.write() = Some(super::edge_store::EdgeStore::new());
        }

        Ok(())
    }

    /// Whether edge graph storage has been initialized.
    #[must_use]
    pub fn has_edges(&self) -> bool {
        self.edge_store.read().is_some()
    }

    /// Total number of edges stored.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edge_store
            .read()
            .as_ref()
            .map_or(0, super::edge_store::EdgeStore::edge_count)
    }

    /// Add a typed directed edge between two document IDs.
    ///
    /// Replaces an existing edge of the same type between the same nodes.
    /// Requires graph support to be enabled explicitly.
    ///
    /// # Durability
    ///
    /// Written to WAL immediately. Call [`flush()`](Self::flush) to persist to manifest.
    pub fn add_edge(
        &mut self,
        from_id: &str,
        to_id: &str,
        edge_type: &str,
        weight: f32,
        metadata: Option<JsonValue>,
    ) -> Result<()> {
        self.ensure_edges_enabled()?;

        let needs_checkpoint = if let Some(ref storage) = self.storage {
            let mut storage = storage.write();
            let meta_bytes = metadata.as_ref().map(serde_json::to_vec).transpose()?;
            storage.log_insert_edge(from_id, to_id, edge_type, weight, meta_bytes.as_deref())?;
            storage.sync()?;
            storage.wal_len() >= self.wal_auto_checkpoint_entries()
        } else {
            false
        };

        self.edge_store
            .write()
            .as_mut()
            .expect("ensure_edges_enabled() was just called")
            .add_edge(Edge {
                from_id: from_id.to_string(),
                to_id: to_id.to_string(),
                edge_type: edge_type.to_string(),
                weight,
                metadata,
            });

        if needs_checkpoint {
            self.checkpoint_wal()?;
        }

        Ok(())
    }

    /// Remove the edge of the given type between two nodes.
    ///
    /// Returns `true` if an edge was found and removed.
    ///
    /// # Durability
    ///
    /// Written to WAL immediately. Call [`flush()`](Self::flush) to persist to manifest.
    pub fn remove_edge(&mut self, from_id: &str, to_id: &str, edge_type: &str) -> Result<bool> {
        if self.edge_store.read().is_none() {
            return Ok(false);
        }

        let removed = self
            .edge_store
            .write()
            .as_mut()
            .expect("checked above")
            .remove_edge(from_id, to_id, edge_type);

        let needs_checkpoint = if removed && let Some(ref storage) = self.storage {
            let mut storage = storage.write();
            storage.log_delete_edge(from_id, to_id, edge_type)?;
            storage.sync()?;
            storage.wal_len() >= self.wal_auto_checkpoint_entries()
        } else {
            false
        };

        if needs_checkpoint {
            self.checkpoint_wal()?;
        }

        Ok(removed)
    }

    /// Get edges for a node in the given direction, optionally filtered by type.
    #[must_use]
    pub fn get_edges(
        &self,
        id: &str,
        direction: EdgeDirection,
        edge_type: Option<&str>,
    ) -> Vec<Edge> {
        self.edge_store
            .read()
            .as_ref()
            .map_or_else(Vec::new, |e| e.get_edges(id, direction, edge_type))
    }

    /// BFS traversal from a starting node.
    ///
    /// Returns all IDs reachable within `max_depth` hops, not including the start node.
    #[must_use]
    pub fn traverse(
        &self,
        start_id: &str,
        direction: EdgeDirection,
        max_depth: usize,
        edge_type_filter: Option<&str>,
    ) -> Vec<String> {
        self.edge_store.read().as_ref().map_or_else(Vec::new, |e| {
            e.traverse(start_id, direction, max_depth, edge_type_filter)
        })
    }

    /// Direct lookup of a single edge.
    #[must_use]
    pub fn get_edge(&self, from_id: &str, to_id: &str, edge_type: &str) -> Option<Edge> {
        self.edge_store
            .read()
            .as_ref()
            .and_then(|e| e.get_edge(from_id, to_id, edge_type))
    }

    /// Get neighbor IDs for a node.
    #[must_use]
    pub fn neighbors(
        &self,
        id: &str,
        direction: EdgeDirection,
        edge_type: Option<&str>,
    ) -> Vec<String> {
        self.edge_store
            .read()
            .as_ref()
            .map_or_else(Vec::new, |e| e.neighbors(id, direction, edge_type))
    }

    /// Count edges for a node without allocating.
    #[must_use]
    pub fn node_degree(
        &self,
        id: &str,
        direction: EdgeDirection,
        edge_type: Option<&str>,
    ) -> usize {
        self.edge_store
            .read()
            .as_ref()
            .map_or(0, |e| e.node_degree(id, direction, edge_type))
    }

    /// Check if a path exists between two nodes.
    #[must_use]
    pub fn has_path(
        &self,
        from_id: &str,
        to_id: &str,
        direction: EdgeDirection,
        max_depth: usize,
        edge_type: Option<&str>,
    ) -> bool {
        self.edge_store
            .read()
            .as_ref()
            .map_or(from_id == to_id, |e| {
                e.has_path(from_id, to_id, direction, max_depth, edge_type)
            })
    }

    /// Find shortest path between two nodes.
    #[must_use]
    pub fn shortest_path(
        &self,
        from_id: &str,
        to_id: &str,
        direction: EdgeDirection,
        max_depth: usize,
        edge_type: Option<&str>,
    ) -> Option<Vec<String>> {
        if from_id == to_id {
            return Some(vec![from_id.to_string()]);
        }
        self.edge_store
            .read()
            .as_ref()
            .and_then(|e| e.shortest_path(from_id, to_id, direction, max_depth, edge_type))
    }

    /// BFS traversal returning the discovery edge for each node.
    #[must_use]
    pub fn traverse_edges(
        &self,
        start_id: &str,
        direction: EdgeDirection,
        max_depth: usize,
        edge_type_filter: Option<&str>,
    ) -> Vec<TraversalHit> {
        self.edge_store.read().as_ref().map_or_else(Vec::new, |e| {
            e.traverse_edges(start_id, direction, max_depth, edge_type_filter)
        })
    }

    /// Extract the ego-graph around a node.
    #[must_use]
    pub fn subgraph(
        &self,
        id: &str,
        max_depth: usize,
        direction: EdgeDirection,
        edge_type: Option<&str>,
    ) -> Subgraph {
        self.edge_store.read().as_ref().map_or_else(
            || Subgraph {
                node_ids: vec![id.to_string()],
                edges: Vec::new(),
            },
            |e| e.subgraph(id, max_depth, direction, edge_type),
        )
    }

    /// Batch add edges with a single WAL sync.
    pub fn add_edges(&mut self, edges: Vec<Edge>) -> Result<usize> {
        if edges.is_empty() {
            return Ok(0);
        }
        self.ensure_edges_enabled()?;

        let needs_checkpoint = if let Some(ref storage) = self.storage {
            let mut storage = storage.write();
            for edge in &edges {
                let meta_bytes = edge.metadata.as_ref().map(serde_json::to_vec).transpose()?;
                storage.log_insert_edge(
                    &edge.from_id,
                    &edge.to_id,
                    &edge.edge_type,
                    edge.weight,
                    meta_bytes.as_deref(),
                )?;
            }
            storage.sync()?;
            storage.wal_len() >= self.wal_auto_checkpoint_entries()
        } else {
            false
        };

        let added = self
            .edge_store
            .write()
            .as_mut()
            .expect("ensure_edges_enabled() was just called")
            .add_edges(edges);

        if needs_checkpoint {
            self.checkpoint_wal()?;
        }

        Ok(added)
    }

    /// Collect all unique edge types.
    #[must_use]
    pub fn edge_types(&self) -> Vec<String> {
        self.edge_store
            .read()
            .as_ref()
            .map_or_else(Vec::new, super::edge_store::EdgeStore::edge_types)
    }

    /// Collect all node IDs that have at least one edge.
    #[must_use]
    pub fn node_ids(&self) -> Vec<String> {
        self.edge_store
            .read()
            .as_ref()
            .map_or_else(Vec::new, super::edge_store::EdgeStore::node_ids)
    }

    /// Expand search results by following edges.
    ///
    /// For each result ID, traverses outgoing edges (depth 1) and returns the
    /// union of result IDs and their neighbors.
    #[must_use]
    pub fn expand(
        &self,
        ids: &[String],
        direction: EdgeDirection,
        edge_type_filter: Option<&str>,
    ) -> Vec<String> {
        let edge_store = self.edge_store.read();
        let Some(store) = edge_store.as_ref() else {
            return ids.to_vec();
        };

        let mut expanded: rustc_hash::FxHashSet<String> = ids.iter().cloned().collect();

        for id in ids {
            for neighbor in store.traverse(id, direction, 1, edge_type_filter) {
                expanded.insert(neighbor);
            }
        }

        expanded.into_iter().collect()
    }
}
