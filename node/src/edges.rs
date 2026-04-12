//! Node.js bindings for EdgeStore operations.

use napi::bindgen_prelude::*;
use napi_derive::napi;
use omendb_lib::vector::store::edge_store::{Edge, EdgeDirection};
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

#[napi(object)]
pub struct TraversalHitResult {
    pub id: String,
    pub depth: u32,
    pub edge: EdgeResult,
}

#[napi(object)]
pub struct SubgraphResult {
    pub node_ids: Vec<String>,
    pub edges: Vec<EdgeResult>,
}

fn edge_to_result(e: omendb_lib::Edge) -> EdgeResult {
    EdgeResult {
        from_id: e.from_id,
        to_id: e.to_id,
        edge_type: e.edge_type,
        weight: e.weight as f64,
        metadata: e.metadata,
    }
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
    /// Enable graph support explicitly for edge operations.
    #[napi(js_name = "enableGraph")]
    pub fn enable_graph(&self) -> Result<()> {
        self.inner
            .write()
            .store_mut()?
            .enable_graph()
            .map_err(convert_error)
    }

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
            .store_mut()?
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
            .store_mut()?
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
            .store()?
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
            .store()?
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
            .store()?
            .expand(&ids, dir, edge_type.as_deref()))
    }

    /// Number of edges in the graph.
    #[napi(getter, js_name = "edgeCount")]
    pub fn edge_count(&self) -> Result<u32> {
        Ok(self
            .inner
            .read()
            .store()?
            .edge_count()
            .try_into()
            .unwrap_or(u32::MAX))
    }

    /// Look up a single edge by endpoints and type.
    #[napi(js_name = "getEdge")]
    pub fn get_edge(
        &self,
        from_id: String,
        to_id: String,
        edge_type: String,
    ) -> Result<Option<EdgeResult>> {
        Ok(self.inner
            .read()
            .store()?
            .get_edge(&from_id, &to_id, &edge_type)
            .map(edge_to_result))
    }

    /// Get neighbor IDs for a node.
    #[napi(js_name = "neighbors")]
    pub fn neighbors(
        &self,
        id: String,
        direction: Option<String>,
        edge_type: Option<String>,
    ) -> Result<Vec<String>> {
        let dir = parse_direction(direction.as_deref().unwrap_or("outgoing"))?;
        Ok(self
            .inner
            .read()
            .store()?
            .neighbors(&id, dir, edge_type.as_deref()))
    }

    /// Count edges for a node.
    #[napi(js_name = "nodeDegree")]
    pub fn node_degree(
        &self,
        id: String,
        direction: Option<String>,
        edge_type: Option<String>,
    ) -> Result<u32> {
        let dir = parse_direction(direction.as_deref().unwrap_or("both"))?;
        Ok(self
            .inner
            .read()
            .store()?
            .node_degree(&id, dir, edge_type.as_deref()) as u32)
    }

    /// Check if a path exists between two nodes.
    #[napi(js_name = "hasPath")]
    pub fn has_path(
        &self,
        from_id: String,
        to_id: String,
        direction: Option<String>,
        max_depth: Option<u32>,
        edge_type: Option<String>,
    ) -> Result<bool> {
        let dir = parse_direction(direction.as_deref().unwrap_or("outgoing"))?;
        let depth = max_depth.unwrap_or(10) as usize;
        Ok(self
            .inner
            .read()
            .store()?
            .has_path(&from_id, &to_id, dir, depth, edge_type.as_deref()))
    }

    /// Find shortest path between two nodes.
    #[napi(js_name = "shortestPath")]
    pub fn shortest_path(
        &self,
        from_id: String,
        to_id: String,
        direction: Option<String>,
        max_depth: Option<u32>,
        edge_type: Option<String>,
    ) -> Result<Option<Vec<String>>> {
        let dir = parse_direction(direction.as_deref().unwrap_or("outgoing"))?;
        let depth = max_depth.unwrap_or(10) as usize;
        Ok(self.inner.read().store()?.shortest_path(
            &from_id,
            &to_id,
            dir,
            depth,
            edge_type.as_deref(),
        ))
    }

    /// BFS traversal returning discovery edges.
    #[napi(js_name = "traverseEdges")]
    pub fn traverse_edges(
        &self,
        start_id: String,
        direction: Option<String>,
        max_depth: Option<u32>,
        edge_type: Option<String>,
    ) -> Result<Vec<TraversalHitResult>> {
        let dir = parse_direction(direction.as_deref().unwrap_or("outgoing"))?;
        let depth = max_depth.unwrap_or(1) as usize;
        let hits =
            self.inner
                .read()
                .store()?
                .traverse_edges(&start_id, dir, depth, edge_type.as_deref());
        Ok(hits
            .into_iter()
            .map(|h| TraversalHitResult {
                id: h.id,
                depth: h.depth as u32,
                edge: edge_to_result(h.edge),
            })
            .collect())
    }

    /// Extract ego-graph around a node.
    #[napi(js_name = "subgraph")]
    pub fn subgraph(
        &self,
        id: String,
        max_depth: Option<u32>,
        direction: Option<String>,
        edge_type: Option<String>,
    ) -> Result<SubgraphResult> {
        let dir = parse_direction(direction.as_deref().unwrap_or("outgoing"))?;
        let depth = max_depth.unwrap_or(1) as usize;
        let sg = self
            .inner
            .read()
            .store()?
            .subgraph(&id, depth, dir, edge_type.as_deref());
        Ok(SubgraphResult {
            node_ids: sg.node_ids,
            edges: sg.edges.into_iter().map(edge_to_result).collect(),
        })
    }

    /// Batch add edges with a single WAL sync.
    #[napi(js_name = "addEdges")]
    pub fn add_edges(&self, edges: Vec<EdgeInput>) -> Result<u32> {
        let edge_vec: Vec<Edge> = edges
            .into_iter()
            .map(|e| Edge {
                from_id: e.from_id,
                to_id: e.to_id,
                edge_type: e.edge_type,
                weight: e.weight.unwrap_or(1.0) as f32,
                metadata: e.metadata,
            })
            .collect();
        self.inner
            .write()
            .store_mut()?
            .add_edges(edge_vec)
            .map(|n| n as u32)
            .map_err(convert_error)
    }

    /// Get all unique edge types.
    #[napi(js_name = "edgeTypes")]
    pub fn edge_types(&self) -> Result<Vec<String>> {
        Ok(self.inner.read().store()?.edge_types())
    }

    /// Get all node IDs with edges.
    #[napi(js_name = "nodeIds")]
    pub fn node_ids(&self) -> Result<Vec<String>> {
        Ok(self.inner.read().store()?.node_ids())
    }
}

#[napi(object)]
pub struct EdgeInput {
    pub from_id: String,
    pub to_id: String,
    pub edge_type: String,
    pub weight: Option<f64>,
    #[napi(ts_type = "Record<string, unknown> | null")]
    pub metadata: Option<JsonValue>,
}
