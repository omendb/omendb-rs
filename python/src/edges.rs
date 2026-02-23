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
        self.inner
            .write()
            .store
            .add_edge(from_id, to_id, edge_type, weight, meta_json)
            .map_err(convert_error)
    }

    /// Remove the edge of the given type between two nodes.
    ///
    /// Returns:
    ///     bool: True if an edge was found and removed
    #[pyo3(signature = (from_id, to_id, edge_type))]
    pub fn remove_edge(&self, from_id: &str, to_id: &str, edge_type: &str) -> PyResult<bool> {
        self.inner
            .write()
            .store
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
    #[pyo3(signature = (id, direction="both"))]
    pub fn get_edges<'py>(
        &self,
        py: Python<'py>,
        id: &str,
        direction: &str,
    ) -> PyResult<Bound<'py, PyList>> {
        let dir = parse_direction(direction)?;
        let edges = self.inner.read().store.get_edges(id, dir);
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
        let ids = self
            .inner
            .read()
            .store
            .traverse(start_id, dir, max_depth, edge_type);
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
    pub fn expand_via_edges<'py>(
        &self,
        py: Python<'py>,
        ids: Vec<String>,
        direction: &str,
        edge_type: Option<&str>,
    ) -> PyResult<Bound<'py, PyList>> {
        let dir = parse_direction(direction)?;
        let expanded = self
            .inner
            .read()
            .store
            .expand_via_edges(&ids, dir, edge_type);
        let list = PyList::empty(py);
        for id in &expanded {
            list.append(PyString::new(py, id))?;
        }
        Ok(list)
    }

    /// Number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.inner.read().store.edge_count()
    }
}
