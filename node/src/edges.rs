//! Node.js bindings for EdgeStore operations.

use napi::bindgen_prelude::*;
use napi_derive::napi;
use omendb_lib::vector::store::edge_store::EdgeDirection;
use serde_json::Value as JsonValue;

use crate::conversions::convert_error;
use crate::database::VectorDatabase;

#[napi(object)]
pub struct EdgeResult {
    pub from_id: String,
    pub to_id: String,
    pub edge_type: String,
    pub weight: f64,
    #[napi(ts_type = "Record<string, unknown> | null")]
    pub metadata: Option<JsonValue>,
}

fn parse_direction(direction: &str) -> Result<EdgeDirection> {
    match direction {
        "outgoing" => Ok(EdgeDirection::Outgoing),
        "incoming" => Ok(EdgeDirection::Incoming),
        "both" => Ok(EdgeDirection::Both),
        other => Err(Error::new(
            Status::InvalidArg,
            format!("Invalid direction '{other}'. Expected 'outgoing', 'incoming', or 'both'."),
        )),
    }
}

#[napi]
impl VectorDatabase {
    /// Add a typed directed edge between two document IDs.
    ///
    /// @param fromId - Source document ID
    /// @param toId - Target document ID
    /// @param edgeType - Edge type label (e.g. "related", "parent")
    /// @param weight - Edge weight (default: 1.0)
    /// @param metadata - Arbitrary JSON-compatible metadata
    #[napi(js_name = "addEdge")]
    pub fn add_edge(
        &self,
        from_id: String,
        to_id: String,
        edge_type: String,
        weight: Option<f64>,
        metadata: Option<JsonValue>,
    ) -> Result<()> {
        let weight = weight.unwrap_or(1.0) as f32;
        self.inner
            .write()
            .store
            .add_edge(&from_id, &to_id, &edge_type, weight, metadata)
            .map_err(convert_error)
    }

    /// Remove the edge of the given type between two nodes.
    ///
    /// @returns true if an edge was found and removed
    #[napi(js_name = "removeEdge")]
    pub fn remove_edge(&self, from_id: String, to_id: String, edge_type: String) -> Result<bool> {
        self.inner
            .write()
            .store
            .remove_edge(&from_id, &to_id, &edge_type)
            .map_err(convert_error)
    }

    /// Get all edges for a node in the given direction.
    ///
    /// @param id - Node ID
    /// @param direction - "outgoing", "incoming", or "both" (default: "both")
    /// @returns Array of edges with fromId, toId, edgeType, weight, metadata
    #[napi(js_name = "getEdges")]
    pub fn get_edges(
        &self,
        id: String,
        direction: Option<String>,
        edge_type: Option<String>,
    ) -> Result<Vec<EdgeResult>> {
        let dir = parse_direction(direction.as_deref().unwrap_or("both"))?;
        let edges = self
            .inner
            .read()
            .store
            .get_edges(&id, dir, edge_type.as_deref());
        Ok(edges
            .into_iter()
            .map(|e| EdgeResult {
                from_id: e.from_id,
                to_id: e.to_id,
                edge_type: e.edge_type,
                weight: e.weight as f64,
                metadata: e.metadata,
            })
            .collect())
    }

    /// BFS traversal from a starting node.
    ///
    /// @param startId - Starting node ID
    /// @param direction - "outgoing", "incoming", or "both" (default: "outgoing")
    /// @param maxDepth - Maximum traversal depth (default: 1)
    /// @param edgeType - Filter by edge type
    /// @returns Reachable node IDs (not including startId)
    #[napi(js_name = "traverse")]
    pub fn traverse(
        &self,
        start_id: String,
        direction: Option<String>,
        max_depth: Option<u32>,
        edge_type: Option<String>,
    ) -> Result<Vec<String>> {
        let dir = parse_direction(direction.as_deref().unwrap_or("outgoing"))?;
        let depth = max_depth.unwrap_or(1) as usize;
        Ok(self
            .inner
            .read()
            .store
            .traverse(&start_id, dir, depth, edge_type.as_deref()))
    }

    /// Expand a list of IDs by following their edges (depth=1).
    ///
    /// @param ids - Starting node IDs
    /// @param direction - "outgoing", "incoming", or "both" (default: "outgoing")
    /// @param edgeType - Filter by edge type
    /// @returns Expanded ID set (includes original IDs + neighbors)
    #[napi(js_name = "expand")]
    pub fn expand(
        &self,
        ids: Vec<String>,
        direction: Option<String>,
        edge_type: Option<String>,
    ) -> Result<Vec<String>> {
        let dir = parse_direction(direction.as_deref().unwrap_or("outgoing"))?;
        Ok(self
            .inner
            .read()
            .store
            .expand(&ids, dir, edge_type.as_deref()))
    }

    /// Number of edges in the graph.
    #[napi(getter, js_name = "edgeCount")]
    pub fn edge_count(&self) -> u32 {
        self.inner
            .read()
            .store
            .edge_count()
            .try_into()
            .unwrap_or(u32::MAX)
    }
}
