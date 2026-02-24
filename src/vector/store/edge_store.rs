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

/// A node discovered during edge-aware traversal.
#[derive(Debug, Clone)]
pub struct TraversalHit {
    pub id: String,
    pub depth: usize,
    pub edge: Edge,
}

/// Ego-graph extraction result.
#[derive(Debug, Clone)]
pub struct Subgraph {
    pub node_ids: Vec<String>,
    pub edges: Vec<Edge>,
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
#[derive(Debug, Clone)]
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
    #[allow(clippy::type_complexity)]
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
            } else {
                debug_assert!(
                    false,
                    "EdgeStore invariant: incoming entry missing for existing outgoing edge {}->{}",
                    edge.from_id, edge.to_id
                );
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

    /// Get edges for a node in the given direction, optionally filtered by type.
    #[must_use]
    pub fn get_edges(
        &self,
        id: &str,
        direction: EdgeDirection,
        edge_type: Option<&str>,
    ) -> Vec<Edge> {
        match direction {
            EdgeDirection::Outgoing => self.get_outgoing(id, edge_type),
            EdgeDirection::Incoming => self.get_incoming(id, edge_type),
            EdgeDirection::Both => {
                let mut edges = self.get_outgoing(id, edge_type);
                // Incoming edges for self-loops (from_id == to_id == id) are
                // already covered by get_outgoing, so skip them to avoid dupes.
                let incoming = self.get_incoming(id, edge_type);
                edges.extend(incoming.into_iter().filter(|e| e.from_id != e.to_id));
                edges
            }
        }
    }

    fn get_outgoing(&self, id: &str, edge_type: Option<&str>) -> Vec<Edge> {
        let Some(records) = self.outgoing.get(id) else {
            return Vec::new();
        };
        let mut result = Vec::with_capacity(records.len());
        for r in records {
            if edge_type.is_none_or(|t| r.edge_type == t) {
                result.push(Edge {
                    from_id: id.to_string(),
                    to_id: r.peer_id.clone(),
                    edge_type: r.edge_type.clone(),
                    weight: r.weight,
                    metadata: r.metadata.clone(),
                });
            }
        }
        result
    }

    fn get_incoming(&self, id: &str, edge_type: Option<&str>) -> Vec<Edge> {
        let Some(records) = self.incoming.get(id) else {
            return Vec::new();
        };
        let mut result = Vec::with_capacity(records.len());
        for r in records {
            if edge_type.is_none_or(|t| r.edge_type == t) {
                result.push(Edge {
                    from_id: r.peer_id.clone(),
                    to_id: id.to_string(),
                    edge_type: r.edge_type.clone(),
                    weight: r.weight,
                    metadata: r.metadata.clone(),
                });
            }
        }
        result
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

            let neighbors = self.neighbors(&node, direction, edge_type_filter);

            for neighbor in neighbors {
                if visited.insert(neighbor.clone()) {
                    result.push(neighbor.clone());
                    queue.push_back((neighbor, depth + 1));
                }
            }
        }

        result
    }

    /// Get neighbor IDs for a node in the given direction, optionally filtered by type.
    #[must_use]
    pub fn neighbors(
        &self,
        node: &str,
        direction: EdgeDirection,
        edge_type_filter: Option<&str>,
    ) -> Vec<String> {
        let matches_filter = |edge_type: &str| edge_type_filter.is_none_or(|t| edge_type == t);

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

    /// Direct lookup of a single edge by endpoints and type.
    #[must_use]
    pub fn get_edge(&self, from_id: &str, to_id: &str, edge_type: &str) -> Option<Edge> {
        self.outgoing.get(from_id).and_then(|records| {
            records
                .iter()
                .find(|r| r.peer_id == to_id && r.edge_type == edge_type)
                .map(|r| Edge {
                    from_id: from_id.to_string(),
                    to_id: r.peer_id.clone(),
                    edge_type: r.edge_type.clone(),
                    weight: r.weight,
                    metadata: r.metadata.clone(),
                })
        })
    }

    /// Count edges for a node without allocating.
    ///
    /// For `Both`: sums outgoing + incoming counts, subtracting self-loop overlap
    /// (consistent with `get_edges(Both)` dedup).
    #[must_use]
    pub fn node_degree(
        &self,
        id: &str,
        direction: EdgeDirection,
        edge_type: Option<&str>,
    ) -> usize {
        let count_records = |records: &[EdgeRecord]| -> usize {
            if let Some(t) = edge_type {
                records.iter().filter(|r| r.edge_type == t).count()
            } else {
                records.len()
            }
        };

        match direction {
            EdgeDirection::Outgoing => self.outgoing.get(id).map_or(0, |r| count_records(r)),
            EdgeDirection::Incoming => self.incoming.get(id).map_or(0, |r| count_records(r)),
            EdgeDirection::Both => {
                let out = self.outgoing.get(id).map_or(0, |r| count_records(r));
                let inc = self.incoming.get(id).map_or(0, |r| count_records(r));
                // Subtract self-loop incoming records (already counted in outgoing),
                // matching get_edges(Both) which filters incoming where from_id == to_id.
                let self_loop_overlap = self.incoming.get(id).map_or(0, |records| {
                    records
                        .iter()
                        .filter(|r| r.peer_id == id && edge_type.is_none_or(|t| r.edge_type == t))
                        .count()
                });
                out + inc - self_loop_overlap
            }
        }
    }

    /// Early-exit BFS reachability check.
    #[must_use]
    pub fn has_path(
        &self,
        from_id: &str,
        to_id: &str,
        direction: EdgeDirection,
        max_depth: usize,
        edge_type: Option<&str>,
    ) -> bool {
        if from_id == to_id {
            return true;
        }
        if max_depth == 0 {
            return false;
        }

        let mut visited = FxHashSet::default();
        visited.insert(from_id.to_string());

        let mut queue: VecDeque<(String, usize)> = VecDeque::new();
        queue.push_back((from_id.to_string(), 0));

        while let Some((node, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }
            for neighbor in self.neighbors(&node, direction, edge_type) {
                if neighbor == to_id {
                    return true;
                }
                if visited.insert(neighbor.clone()) {
                    queue.push_back((neighbor, depth + 1));
                }
            }
        }
        false
    }

    /// BFS shortest path. Returns path including start and end, or None if unreachable.
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
        if max_depth == 0 {
            return None;
        }

        let mut parent: FxHashMap<String, String> = FxHashMap::default();
        parent.insert(from_id.to_string(), String::new());

        let mut queue: VecDeque<(String, usize)> = VecDeque::new();
        queue.push_back((from_id.to_string(), 0));

        while let Some((node, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }
            for neighbor in self.neighbors(&node, direction, edge_type) {
                if !parent.contains_key(&neighbor) {
                    parent.insert(neighbor.clone(), node.clone());
                    if neighbor == to_id {
                        let mut path = vec![to_id.to_string()];
                        let mut current = to_id.to_string();
                        while let Some(p) = parent.get(&current) {
                            if p.is_empty() {
                                break;
                            }
                            path.push(p.clone());
                            current = p.clone();
                        }
                        path.reverse();
                        return Some(path);
                    }
                    queue.push_back((neighbor, depth + 1));
                }
            }
        }
        None
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
            let edges = self.get_edges(&node, direction, edge_type_filter);
            for edge in edges {
                let peer = match direction {
                    EdgeDirection::Incoming => edge.from_id.clone(),
                    EdgeDirection::Outgoing => edge.to_id.clone(),
                    EdgeDirection::Both => {
                        if edge.to_id == node {
                            edge.from_id.clone()
                        } else {
                            edge.to_id.clone()
                        }
                    }
                };
                if visited.insert(peer.clone()) {
                    queue.push_back((peer.clone(), depth + 1));
                    result.push(TraversalHit {
                        id: peer,
                        depth: depth + 1,
                        edge,
                    });
                }
            }
        }

        result
    }

    /// Extract the ego-graph: all nodes reachable within max_depth and all edges between them.
    #[must_use]
    pub fn subgraph(
        &self,
        id: &str,
        max_depth: usize,
        direction: EdgeDirection,
        edge_type: Option<&str>,
    ) -> Subgraph {
        let reachable = self.traverse(id, direction, max_depth, edge_type);
        let mut node_set: FxHashSet<String> = reachable.into_iter().collect();
        node_set.insert(id.to_string());

        let mut edges = Vec::new();
        let mut seen_edges: FxHashSet<(String, String, String)> = FxHashSet::default();

        for node in &node_set {
            if let Some(records) = self.outgoing.get(node.as_str()) {
                for r in records {
                    if node_set.contains(&r.peer_id) && edge_type.is_none_or(|t| r.edge_type == t) {
                        let key = (node.clone(), r.peer_id.clone(), r.edge_type.clone());
                        if seen_edges.insert(key) {
                            edges.push(Edge {
                                from_id: node.clone(),
                                to_id: r.peer_id.clone(),
                                edge_type: r.edge_type.clone(),
                                weight: r.weight,
                                metadata: r.metadata.clone(),
                            });
                        }
                    }
                }
            }
            // For Both direction, also collect incoming edges where peer is in set
            if direction == EdgeDirection::Both || direction == EdgeDirection::Incoming {
                if let Some(records) = self.incoming.get(node.as_str()) {
                    for r in records {
                        if node_set.contains(&r.peer_id)
                            && edge_type.is_none_or(|t| r.edge_type == t)
                        {
                            let key = (r.peer_id.clone(), node.clone(), r.edge_type.clone());
                            if seen_edges.insert(key) {
                                edges.push(Edge {
                                    from_id: r.peer_id.clone(),
                                    to_id: node.clone(),
                                    edge_type: r.edge_type.clone(),
                                    weight: r.weight,
                                    metadata: r.metadata.clone(),
                                });
                            }
                        }
                    }
                }
            }
        }

        Subgraph {
            node_ids: node_set.into_iter().collect(),
            edges,
        }
    }

    /// Batch add edges. Returns the number of new edges added (not updates).
    pub fn add_edges(&mut self, edges: Vec<Edge>) -> usize {
        let before = self.edge_count;
        for edge in edges {
            self.add_edge(edge);
        }
        self.edge_count - before
    }

    /// Collect all unique edge types in the store.
    #[must_use]
    pub fn edge_types(&self) -> Vec<String> {
        let mut types = FxHashSet::default();
        for records in self.outgoing.values() {
            for r in records {
                types.insert(r.edge_type.clone());
            }
        }
        types.into_iter().collect()
    }

    /// Collect all node IDs that have at least one edge.
    #[must_use]
    pub fn node_ids(&self) -> Vec<String> {
        let mut ids: FxHashSet<String> = self.outgoing.keys().cloned().collect();
        ids.extend(self.incoming.keys().cloned());
        ids.into_iter().collect()
    }

    /// Returns (from_id, to_id, edge_type) for all edges touching a node.
    ///
    /// Used by VectorStore to emit WAL DeleteEdge entries during cascade delete.
    /// Self-loops (from_id == to_id == id) appear only once (via outgoing).
    pub fn edges_involving(&self, id: &str) -> Vec<(String, String, String)> {
        let mut result = Vec::new();
        if let Some(records) = self.outgoing.get(id) {
            for r in records {
                result.push((id.to_string(), r.peer_id.clone(), r.edge_type.clone()));
            }
        }
        if let Some(records) = self.incoming.get(id) {
            for r in records {
                // Skip self-loops — already emitted via outgoing above.
                if r.peer_id != id {
                    result.push((r.peer_id.clone(), id.to_string(), r.edge_type.clone()));
                }
            }
        }
        result
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
    #[allow(clippy::type_complexity)]
    pub fn to_bytes(&self) -> anyhow::Result<Vec<u8>> {
        let mut edges: Vec<(String, String, String, f32, Option<Vec<u8>>)> =
            Vec::with_capacity(self.edge_count);
        for (from_id, records) in &self.outgoing {
            for r in records {
                let meta_bytes = r
                    .metadata
                    .as_ref()
                    .map(serde_json::to_vec)
                    .transpose()
                    .map_err(|e| anyhow::anyhow!("Failed to serialize edge metadata: {e}"))?;
                edges.push((
                    from_id.clone(),
                    r.peer_id.clone(),
                    r.edge_type.clone(),
                    r.weight,
                    meta_bytes,
                ));
            }
        }
        Ok(postcard::to_allocvec(&EdgeStoreWire { edges })?)
    }

    /// Deserialize from bytes (postcard format).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, postcard::Error> {
        let wire: EdgeStoreWire = postcard::from_bytes(bytes)?;
        let mut store = Self::new();
        for (from_id, to_id, edge_type, weight, meta_bytes) in wire.edges {
            let metadata = meta_bytes.as_deref().and_then(|b| {
                serde_json::from_slice(b).unwrap_or_else(|e| {
                    tracing::warn!(
                        from_id = %from_id, to_id = %to_id, edge_type = %edge_type,
                        "Dropping corrupt edge metadata during load: {e}"
                    );
                    None
                })
            });
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
