//! EdgeStore: typed directed edge graph over document IDs.
//!
//! Adjacency list storage for entity relationships used by the memory service.
//! Edges are directed and typed, stored in dual FxHashMaps for O(1) forward/backward lookup.

use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::VecDeque;

/// A typed directed edge between two document IDs.
#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    pub from_id: String,
    pub to_id: String,
    pub edge_type: String,
    pub weight: f32,
    pub metadata: Option<JsonValue>,
}

/// Direction for edge queries and traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeDirection {
    /// Follow edges outward (from_id → to_id)
    Outgoing,
    /// Follow edges inward (to_id → from_id)
    Incoming,
    /// Both outgoing and incoming
    Both,
}

/// Internal adjacency list record.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EdgeRecord {
    peer_id: String,
    edge_type: String,
    weight: f32,
    metadata: Option<JsonValue>,
}

/// Serialization wire format: flat edge list reconstructed into adjacency lists on load.
///
/// Metadata is serialized as JSON bytes because postcard cannot serialize serde_json::Value
/// (it doesn't support dynamic maps). JSON bytes round-trip losslessly.
#[derive(Serialize, Deserialize)]
struct EdgeStoreWire {
    /// (from_id, to_id, edge_type, weight, metadata_json_bytes)
    edges: Vec<(String, String, String, f32, Option<Vec<u8>>)>,
}

/// Typed directed edge graph over document IDs.
///
/// Dual adjacency list: outgoing + incoming per node.
/// At <1M edges, Vec scan per node is fast enough — no secondary indexes.
#[derive(Debug, Default)]
pub struct EdgeStore {
    /// Outgoing edges per node: from_id → records
    outgoing: FxHashMap<String, Vec<EdgeRecord>>,
    /// Incoming edges per node: to_id → records
    incoming: FxHashMap<String, Vec<EdgeRecord>>,
    /// Total edge count
    edge_count: usize,
}

impl EdgeStore {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Total number of edges stored.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edge_count
    }

    /// Whether this store has any edges.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.edge_count == 0
    }

    /// Add an edge. Replaces an existing edge of the same type between the same nodes.
    pub fn add_edge(&mut self, edge: Edge) {
        let out_list = self.outgoing.entry(edge.from_id.clone()).or_default();

        let already_exists = if let Some(pos) = out_list
            .iter()
            .position(|r| r.peer_id == edge.to_id && r.edge_type == edge.edge_type)
        {
            out_list[pos] = EdgeRecord {
                peer_id: edge.to_id.clone(),
                edge_type: edge.edge_type.clone(),
                weight: edge.weight,
                metadata: edge.metadata.clone(),
            };
            true
        } else {
            out_list.push(EdgeRecord {
                peer_id: edge.to_id.clone(),
                edge_type: edge.edge_type.clone(),
                weight: edge.weight,
                metadata: edge.metadata.clone(),
            });
            false
        };

        let in_list = self.incoming.entry(edge.to_id.clone()).or_default();
        if already_exists {
            if let Some(pos) = in_list
                .iter()
                .position(|r| r.peer_id == edge.from_id && r.edge_type == edge.edge_type)
            {
                in_list[pos] = EdgeRecord {
                    peer_id: edge.from_id.clone(),
                    edge_type: edge.edge_type.clone(),
                    weight: edge.weight,
                    metadata: edge.metadata.clone(),
                };
            }
        } else {
            in_list.push(EdgeRecord {
                peer_id: edge.from_id.clone(),
                edge_type: edge.edge_type.clone(),
                weight: edge.weight,
                metadata: edge.metadata.clone(),
            });
            self.edge_count += 1;
        }
    }

    /// Get all edges for a node in the given direction.
    #[must_use]
    pub fn get_edges(&self, id: &str, direction: EdgeDirection) -> Vec<Edge> {
        match direction {
            EdgeDirection::Outgoing => self.get_outgoing(id),
            EdgeDirection::Incoming => self.get_incoming(id),
            EdgeDirection::Both => {
                let mut edges = self.get_outgoing(id);
                edges.extend(self.get_incoming(id));
                edges
            }
        }
    }

    fn get_outgoing(&self, id: &str) -> Vec<Edge> {
        self.outgoing
            .get(id)
            .map(|records| {
                records
                    .iter()
                    .map(|r| Edge {
                        from_id: id.to_string(),
                        to_id: r.peer_id.clone(),
                        edge_type: r.edge_type.clone(),
                        weight: r.weight,
                        metadata: r.metadata.clone(),
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn get_incoming(&self, id: &str) -> Vec<Edge> {
        self.incoming
            .get(id)
            .map(|records| {
                records
                    .iter()
                    .map(|r| Edge {
                        from_id: r.peer_id.clone(),
                        to_id: id.to_string(),
                        edge_type: r.edge_type.clone(),
                        weight: r.weight,
                        metadata: r.metadata.clone(),
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Remove the edge of the given type between two nodes.
    ///
    /// Returns `true` if an edge was found and removed.
    pub fn remove_edge(&mut self, from_id: &str, to_id: &str, edge_type: &str) -> bool {
        let removed = if let Some(list) = self.outgoing.get_mut(from_id) {
            if let Some(pos) = list
                .iter()
                .position(|r| r.peer_id == to_id && r.edge_type == edge_type)
            {
                list.swap_remove(pos);
                true
            } else {
                false
            }
        } else {
            false
        };

        if removed {
            if let Some(list) = self.incoming.get_mut(to_id) {
                if let Some(pos) = list
                    .iter()
                    .position(|r| r.peer_id == from_id && r.edge_type == edge_type)
                {
                    list.swap_remove(pos);
                }
            }
            self.edge_count = self.edge_count.saturating_sub(1);
        }

        removed
    }

    /// Remove all edges involving a node (cascade delete).
    ///
    /// Called when a document is deleted.
    pub fn remove_all_for(&mut self, id: &str) {
        if let Some(records) = self.outgoing.remove(id) {
            for r in &records {
                if let Some(list) = self.incoming.get_mut(&r.peer_id) {
                    list.retain(|e| !(e.peer_id == id && e.edge_type == r.edge_type));
                }
                self.edge_count = self.edge_count.saturating_sub(1);
            }
        }

        if let Some(records) = self.incoming.remove(id) {
            for r in &records {
                if let Some(list) = self.outgoing.get_mut(&r.peer_id) {
                    list.retain(|e| !(e.peer_id == id && e.edge_type == r.edge_type));
                }
                self.edge_count = self.edge_count.saturating_sub(1);
            }
        }
    }

    /// BFS traversal from a starting node.
    ///
    /// Returns all IDs reachable within `max_depth` hops, not including the start node.
    /// Cycle-safe via visited set.
    #[must_use]
    pub fn traverse(
        &self,
        start_id: &str,
        direction: EdgeDirection,
        max_depth: usize,
        edge_type_filter: Option<&str>,
    ) -> Vec<String> {
        if max_depth == 0 {
            return Vec::new();
        }

        let mut visited = FxHashSet::default();
        visited.insert(start_id.to_string());

        let mut queue: VecDeque<(String, usize)> = VecDeque::new();
        queue.push_back((start_id.to_string(), 0));

        let mut result = Vec::new();

        while let Some((node, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            let neighbors = self.neighbors_at(&node, direction, edge_type_filter);

            for neighbor in neighbors {
                if visited.insert(neighbor.clone()) {
                    result.push(neighbor.clone());
                    queue.push_back((neighbor, depth + 1));
                }
            }
        }

        result
    }

    fn neighbors_at(
        &self,
        node: &str,
        direction: EdgeDirection,
        edge_type_filter: Option<&str>,
    ) -> Vec<String> {
        let matches_filter = |edge_type: &str| edge_type_filter.map_or(true, |t| edge_type == t);

        match direction {
            EdgeDirection::Outgoing => self
                .outgoing
                .get(node)
                .map(|records| {
                    records
                        .iter()
                        .filter(|r| matches_filter(&r.edge_type))
                        .map(|r| r.peer_id.clone())
                        .collect()
                })
                .unwrap_or_default(),
            EdgeDirection::Incoming => self
                .incoming
                .get(node)
                .map(|records| {
                    records
                        .iter()
                        .filter(|r| matches_filter(&r.edge_type))
                        .map(|r| r.peer_id.clone())
                        .collect()
                })
                .unwrap_or_default(),
            EdgeDirection::Both => {
                let mut neighbors = Vec::new();
                if let Some(records) = self.outgoing.get(node) {
                    neighbors.extend(
                        records
                            .iter()
                            .filter(|r| matches_filter(&r.edge_type))
                            .map(|r| r.peer_id.clone()),
                    );
                }
                if let Some(records) = self.incoming.get(node) {
                    neighbors.extend(
                        records
                            .iter()
                            .filter(|r| matches_filter(&r.edge_type))
                            .map(|r| r.peer_id.clone()),
                    );
                }
                neighbors
            }
        }
    }

    /// Return all edges as a flat iterator (used for WAL merge on open).
    pub fn all_edges(&self) -> impl Iterator<Item = Edge> + '_ {
        self.outgoing.iter().flat_map(|(from_id, records)| {
            records.iter().map(move |r| Edge {
                from_id: from_id.clone(),
                to_id: r.peer_id.clone(),
                edge_type: r.edge_type.clone(),
                weight: r.weight,
                metadata: r.metadata.clone(),
            })
        })
    }

    /// GC: remove edges referencing IDs not in the live set.
    ///
    /// Called by `compact()` as a safety net after slot reassignment.
    /// Returns the number of edges removed.
    pub fn gc_orphaned(&mut self, live_ids: &FxHashSet<String>) -> usize {
        let to_remove: Vec<(String, String, String)> = self
            .outgoing
            .iter()
            .flat_map(|(from_id, records)| {
                records.iter().filter_map(|r| {
                    if !live_ids.contains(from_id) || !live_ids.contains(&r.peer_id) {
                        Some((from_id.clone(), r.peer_id.clone(), r.edge_type.clone()))
                    } else {
                        None
                    }
                })
            })
            .collect();

        let removed = to_remove.len();
        for (from_id, to_id, edge_type) in to_remove {
            self.remove_edge(&from_id, &to_id, &edge_type);
        }
        removed
    }

    /// Serialize to bytes for manifest storage (postcard format).
    ///
    /// Metadata is serialized as JSON bytes since postcard cannot handle serde_json::Value.
    pub fn to_bytes(&self) -> Result<Vec<u8>, postcard::Error> {
        let edges: Vec<(String, String, String, f32, Option<Vec<u8>>)> = self
            .outgoing
            .iter()
            .flat_map(|(from_id, records)| {
                records.iter().map(move |r| {
                    let meta_bytes = r.metadata.as_ref().and_then(|m| serde_json::to_vec(m).ok());
                    (
                        from_id.clone(),
                        r.peer_id.clone(),
                        r.edge_type.clone(),
                        r.weight,
                        meta_bytes,
                    )
                })
            })
            .collect();
        postcard::to_allocvec(&EdgeStoreWire { edges })
    }

    /// Deserialize from bytes (postcard format).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, postcard::Error> {
        let wire: EdgeStoreWire = postcard::from_bytes(bytes)?;
        let mut store = Self::new();
        for (from_id, to_id, edge_type, weight, meta_bytes) in wire.edges {
            let metadata = meta_bytes
                .as_deref()
                .and_then(|b| serde_json::from_slice(b).ok());
            store.add_edge(Edge {
                from_id,
                to_id,
                edge_type,
                weight,
                metadata,
            });
        }
        Ok(store)
    }
}
