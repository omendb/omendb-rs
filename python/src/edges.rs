//! Python bindings for EdgeStore operations.

use crate::conversions::{convert_error, json_to_pyobject, pyobject_to_json};
use crate::database::VectorDatabase;
use omendb_lib::vector::store::edge_store::{Edge, EdgeDirection};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};

fn parse_direction(direction: &str) -> PyResult<EdgeDirection> {
    match direction {
        "outgoing" => Ok(EdgeDirection::Outgoing),
        "incoming" => Ok(EdgeDirection::Incoming),
        "both" => Ok(EdgeDirection::Both),
        other => Err(PyValueError::new_err(format!(
            "Invalid direction '{other}'. Expected 'outgoing', 'incoming', or 'both'."
        ))),
    }
}

fn edge_to_dict(py: Python<'_>, edge: &Edge) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("from_id", &edge.from_id)?;
    dict.set_item("to_id", &edge.to_id)?;
    dict.set_item("edge_type", &edge.edge_type)?;
    dict.set_item("weight", edge.weight)?;
    let meta = match &edge.metadata {
        Some(m) => json_to_pyobject(py, m)?,
        None => py.None(),
    };
    dict.set_item("metadata", meta)?;
    Ok(dict.into())
}

#[pymethods]
impl VectorDatabase {
    /// Add a typed directed edge between two document IDs.
    ///
    /// Args:
    ///     from_id (str): Source document ID
    ///     to_id (str): Target document ID
    ///     edge_type (str): Edge type label (e.g. "related", "parent")
    ///     weight (float): Edge weight (default: 1.0)
    ///     metadata (dict, optional): Arbitrary JSON-compatible metadata
    ///
    /// Raises:
    ///     RuntimeError: On storage error
    #[pyo3(signature = (from_id, to_id, edge_type, weight=1.0, metadata=None))]
    pub fn add_edge(
        &self,
        from_id: &str,
        to_id: &str,
        edge_type: &str,
        weight: f32,
        metadata: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let meta_json = metadata.map(|m| pyobject_to_json(&m)).transpose()?;
        let mut inner = self.inner.write();
        inner
            .store_mut()
            .add_edge(from_id, to_id, edge_type, weight, meta_json)
            .map_err(convert_error)
    }

    /// Remove the edge of the given type between two nodes.
    ///
    /// Returns:
    ///     bool: True if an edge was found and removed
    #[pyo3(signature = (from_id, to_id, edge_type))]
    pub fn remove_edge(&self, from_id: &str, to_id: &str, edge_type: &str) -> PyResult<bool> {
        let mut inner = self.inner.write();
        inner
            .store_mut()
            .remove_edge(from_id, to_id, edge_type)
            .map_err(convert_error)
    }

    /// Get all edges for a node in the given direction.
    ///
    /// Args:
    ///     id (str): Node ID
    ///     direction (str): "outgoing", "incoming", or "both" (default: "both")
    ///
    /// Returns:
    ///     list[dict]: Edges with keys: from_id, to_id, edge_type, weight, metadata
    #[pyo3(signature = (id, direction="both", edge_type=None))]
    pub fn get_edges<'py>(
        &self,
        py: Python<'py>,
        id: &str,
        direction: &str,
        edge_type: Option<&str>,
    ) -> PyResult<Bound<'py, PyList>> {
        let dir = parse_direction(direction)?;
        let edges = self.inner.read().store().get_edges(id, dir, edge_type);
        let list = PyList::empty(py);
        for edge in &edges {
            list.append(edge_to_dict(py, edge)?)?;
        }
        Ok(list)
    }

    /// BFS traversal from a starting node.
    ///
    /// Args:
    ///     start_id (str): Starting node ID
    ///     direction (str): "outgoing", "incoming", or "both" (default: "outgoing")
    ///     max_depth (int): Maximum traversal depth (default: 1)
    ///     edge_type (str, optional): Filter by edge type
    ///
    /// Returns:
    ///     list[str]: Reachable node IDs (not including start_id)
    #[pyo3(signature = (start_id, direction="outgoing", max_depth=1, edge_type=None))]
    pub fn traverse<'py>(
        &self,
        py: Python<'py>,
        start_id: &str,
        direction: &str,
        max_depth: usize,
        edge_type: Option<&str>,
    ) -> PyResult<Bound<'py, PyList>> {
        let dir = parse_direction(direction)?;
        let ids = self.inner.read().store().traverse(start_id, dir, max_depth, edge_type);
        let list = PyList::empty(py);
        for id in &ids {
            list.append(PyString::new(py, id))?;
        }
        Ok(list)
    }

    /// Expand a list of IDs by following their edges (depth=1).
    ///
    /// For each ID, finds all neighbors and returns the union.
    ///
    /// Args:
    ///     ids (list[str]): Starting node IDs
    ///     direction (str): "outgoing", "incoming", or "both" (default: "outgoing")
    ///     edge_type (str, optional): Filter by edge type
    ///
    /// Returns:
    ///     list[str]: Expanded ID set (includes original IDs + neighbors)
    #[pyo3(signature = (ids, direction="outgoing", edge_type=None))]
    pub fn expand<'py>(
        &self,
        py: Python<'py>,
        ids: Vec<String>,
        direction: &str,
        edge_type: Option<&str>,
    ) -> PyResult<Bound<'py, PyList>> {
        let dir = parse_direction(direction)?;
        let expanded = self.inner.read().store().expand(&ids, dir, edge_type);
        let list = PyList::empty(py);
        for id in &expanded {
            list.append(PyString::new(py, id))?;
        }
        Ok(list)
    }

    /// Number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.inner.read().store().edge_count()
    }

    /// Look up a single edge by endpoints and type.
    ///
    /// Returns:
    ///     dict | None: Edge dict or None if not found
    #[pyo3(signature = (from_id, to_id, edge_type))]
    pub fn get_edge(
        &self,
        py: Python<'_>,
        from_id: &str,
        to_id: &str,
        edge_type: &str,
    ) -> PyResult<Option<Py<PyDict>>> {
        self.inner
            .read()
            .store()
            .get_edge(from_id, to_id, edge_type)
            .map(|e| edge_to_dict(py, &e))
            .transpose()
    }

    /// Get neighbor IDs for a node.
    ///
    /// Returns:
    ///     list[str]: Neighbor node IDs
    #[pyo3(signature = (id, direction="outgoing", edge_type=None))]
    pub fn neighbors<'py>(
        &self,
        py: Python<'py>,
        id: &str,
        direction: &str,
        edge_type: Option<&str>,
    ) -> PyResult<Bound<'py, PyList>> {
        let dir = parse_direction(direction)?;
        let ids = self.inner.read().store().neighbors(id, dir, edge_type);
        let list = PyList::empty(py);
        for id in &ids {
            list.append(PyString::new(py, id))?;
        }
        Ok(list)
    }

    /// Count edges for a node without allocating.
    ///
    /// Returns:
    ///     int: Number of edges
    #[pyo3(signature = (id, direction="both", edge_type=None))]
    pub fn node_degree(
        &self,
        id: &str,
        direction: &str,
        edge_type: Option<&str>,
    ) -> PyResult<usize> {
        let dir = parse_direction(direction)?;
        Ok(self.inner.read().store().node_degree(id, dir, edge_type))
    }

    /// Check if a path exists between two nodes.
    ///
    /// Returns:
    ///     bool: True if reachable within max_depth
    #[pyo3(signature = (from_id, to_id, direction="outgoing", max_depth=10, edge_type=None))]
    pub fn has_path(
        &self,
        from_id: &str,
        to_id: &str,
        direction: &str,
        max_depth: usize,
        edge_type: Option<&str>,
    ) -> PyResult<bool> {
        let dir = parse_direction(direction)?;
        Ok(self
            .inner
            .read()
            .store()
            .has_path(from_id, to_id, dir, max_depth, edge_type))
    }

    /// Find shortest path between two nodes.
    ///
    /// Returns:
    ///     list[str] | None: Path including start and end, or None
    #[pyo3(signature = (from_id, to_id, direction="outgoing", max_depth=10, edge_type=None))]
    pub fn shortest_path<'py>(
        &self,
        py: Python<'py>,
        from_id: &str,
        to_id: &str,
        direction: &str,
        max_depth: usize,
        edge_type: Option<&str>,
    ) -> PyResult<Option<Bound<'py, PyList>>> {
        let dir = parse_direction(direction)?;
        match self
            .inner
            .read()
            .store()
            .shortest_path(from_id, to_id, dir, max_depth, edge_type)
        {
            Some(path) => {
                let list = PyList::empty(py);
                for id in &path {
                    list.append(PyString::new(py, id))?;
                }
                Ok(Some(list))
            }
            None => Ok(None),
        }
    }

    /// BFS traversal returning discovery edges.
    ///
    /// Returns:
    ///     list[dict]: Each dict has 'id', 'depth', 'edge' (nested dict)
    #[pyo3(signature = (start_id, direction="outgoing", max_depth=1, edge_type=None))]
    pub fn traverse_edges<'py>(
        &self,
        py: Python<'py>,
        start_id: &str,
        direction: &str,
        max_depth: usize,
        edge_type: Option<&str>,
    ) -> PyResult<Bound<'py, PyList>> {
        let dir = parse_direction(direction)?;
        let hits = self
            .inner
            .read()
            .store()
            .traverse_edges(start_id, dir, max_depth, edge_type);
        let list = PyList::empty(py);
        for hit in &hits {
            let dict = PyDict::new(py);
            dict.set_item("id", &hit.id)?;
            dict.set_item("depth", hit.depth)?;
            dict.set_item("edge", edge_to_dict(py, &hit.edge)?)?;
            list.append(dict)?;
        }
        Ok(list)
    }

    /// Extract ego-graph around a node.
    ///
    /// Returns:
    ///     dict: {"node_ids": list[str], "edges": list[dict]}
    #[pyo3(signature = (id, max_depth=1, direction="outgoing", edge_type=None))]
    pub fn subgraph(
        &self,
        py: Python<'_>,
        id: &str,
        max_depth: usize,
        direction: &str,
        edge_type: Option<&str>,
    ) -> PyResult<Py<PyDict>> {
        let dir = parse_direction(direction)?;
        let sg = self
            .inner
            .read()
            .store()
            .subgraph(id, max_depth, dir, edge_type);
        let dict = PyDict::new(py);
        let node_list = PyList::empty(py);
        for nid in &sg.node_ids {
            node_list.append(PyString::new(py, nid))?;
        }
        dict.set_item("node_ids", node_list)?;
        let edge_list = PyList::empty(py);
        for edge in &sg.edges {
            edge_list.append(edge_to_dict(py, edge)?)?;
        }
        dict.set_item("edges", edge_list)?;
        Ok(dict.into())
    }

    /// Batch add edges with a single WAL sync.
    ///
    /// Returns:
    ///     int: Number of new edges added
    #[pyo3(signature = (edges,))]
    pub fn add_edges(&self, edges: Vec<Bound<'_, PyDict>>) -> PyResult<usize> {
        let mut edge_vec = Vec::with_capacity(edges.len());
        for dict in &edges {
            let from_id: String = dict
                .get_item("from_id")?
                .ok_or_else(|| PyValueError::new_err("Edge dict missing 'from_id'"))?
                .extract()?;
            let to_id: String = dict
                .get_item("to_id")?
                .ok_or_else(|| PyValueError::new_err("Edge dict missing 'to_id'"))?
                .extract()?;
            let edge_type: String = dict
                .get_item("edge_type")?
                .ok_or_else(|| PyValueError::new_err("Edge dict missing 'edge_type'"))?
                .extract()?;
            let weight: f32 = dict
                .get_item("weight")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(1.0);
            let metadata = dict
                .get_item("metadata")?
                .map(|m| pyobject_to_json(&m))
                .transpose()?;
            edge_vec.push(Edge {
                from_id,
                to_id,
                edge_type,
                weight,
                metadata,
            });
        }
        let mut inner = self.inner.write();
        inner.store_mut().add_edges(edge_vec).map_err(convert_error)
    }

    /// Get all unique edge types.
    ///
    /// Returns:
    ///     list[str]: Unique edge type strings
    pub fn edge_types<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let types = self.inner.read().store().edge_types();
        let list = PyList::empty(py);
        for t in &types {
            list.append(PyString::new(py, t)).expect("append edge type");
        }
        list
    }

    /// Get all node IDs with edges.
    ///
    /// Returns:
    ///     list[str]: Node IDs
    pub fn node_ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let ids = self.inner.read().store().node_ids();
        let list = PyList::empty(py);
        for id in &ids {
            list.append(PyString::new(py, id)).expect("append node id");
        }
        list
    }
}
