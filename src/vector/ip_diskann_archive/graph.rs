/*
 * Bi-directional graph structure for IP-DiskANN
 *
 * Key difference from standard DiskANN:
 * - Tracks both out-neighbors (forward edges) AND in-neighbors (reverse edges)
 * - Enables efficient in-place deletion (find who points to deleted node)
 * - Required for IP-DiskANN's in-place update protocol
 *
 * Reference: IP-DiskANN paper (arXiv 2502.13826, Section 3)
 */

use super::types::NodeId;
use std::collections::HashMap;

/// Bi-directional graph node
///
/// Standard Vamana/DiskANN only tracks out_neighbors.
/// IP-DiskANN requires in_neighbors for efficient in-place deletes.
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Forward edges (who I point to)
    pub out_neighbors: Vec<NodeId>,

    /// Reverse edges (who points to me) - IP-DiskANN addition
    pub in_neighbors: Vec<NodeId>,
}

impl GraphNode {
    pub fn new() -> Self {
        Self {
            out_neighbors: Vec::new(),
            in_neighbors: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            out_neighbors: Vec::with_capacity(capacity),
            in_neighbors: Vec::with_capacity(capacity),
        }
    }

    /// Add an outgoing edge (I point to target)
    pub fn add_out_neighbor(&mut self, target: NodeId) {
        if !self.out_neighbors.contains(&target) {
            self.out_neighbors.push(target);
        }
    }

    /// Add an incoming edge (source points to me)
    pub fn add_in_neighbor(&mut self, source: NodeId) {
        if !self.in_neighbors.contains(&source) {
            self.in_neighbors.push(source);
        }
    }

    /// Remove an outgoing edge
    pub fn remove_out_neighbor(&mut self, target: NodeId) {
        self.out_neighbors.retain(|&id| id != target);
    }

    /// Remove an incoming edge
    pub fn remove_in_neighbor(&mut self, source: NodeId) {
        self.in_neighbors.retain(|&id| id != source);
    }

    /// Get degree (out-degree)
    pub fn degree(&self) -> usize {
        self.out_neighbors.len()
    }

    /// Get in-degree
    pub fn in_degree(&self) -> usize {
        self.in_neighbors.len()
    }
}

impl Default for GraphNode {
    fn default() -> Self {
        Self::new()
    }
}

/// Bi-directional graph for IP-DiskANN
///
/// Maintains both forward and reverse edges for efficient in-place updates.
#[derive(Debug)]
pub struct BiDirectionalGraph {
    nodes: HashMap<NodeId, GraphNode>,
    entry_node: Option<NodeId>,
}

impl BiDirectionalGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            entry_node: None,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: HashMap::with_capacity(capacity),
            entry_node: None,
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, id: NodeId) {
        self.nodes.entry(id).or_insert_with(GraphNode::new);

        // Set first node as entry point
        if self.entry_node.is_none() {
            self.entry_node = Some(id);
        }
    }

    /// Add a directed edge (from -> to)
    ///
    /// Updates both out-neighbors (from) and in-neighbors (to).
    /// This is the key for IP-DiskANN's bi-directional tracking.
    pub fn add_edge(&mut self, from: NodeId, to: NodeId) {
        // Ensure both nodes exist
        self.add_node(from);
        self.add_node(to);

        // Forward edge: from points to 'to'
        if let Some(from_node) = self.nodes.get_mut(&from) {
            from_node.add_out_neighbor(to);
        }

        // Reverse edge: 'to' is pointed to by from
        if let Some(to_node) = self.nodes.get_mut(&to) {
            to_node.add_in_neighbor(from);
        }
    }

    /// Remove a directed edge (from -> to)
    ///
    /// Updates both out-neighbors and in-neighbors.
    pub fn remove_edge(&mut self, from: NodeId, to: NodeId) {
        // Remove forward edge
        if let Some(from_node) = self.nodes.get_mut(&from) {
            from_node.remove_out_neighbor(to);
        }

        // Remove reverse edge
        if let Some(to_node) = self.nodes.get_mut(&to) {
            to_node.remove_in_neighbor(from);
        }
    }

    /// Remove a node and all its edges
    ///
    /// IP-DiskANN advantage: Can efficiently find all in-neighbors
    /// and remove edges pointing to this node.
    pub fn remove_node(&mut self, id: NodeId) -> Option<GraphNode> {
        // Clone neighbor lists to avoid borrow checker issues
        let (out_neighbors, in_neighbors) = if let Some(node) = self.nodes.get(&id) {
            (node.out_neighbors.clone(), node.in_neighbors.clone())
        } else {
            return None;
        };

        // Remove all outgoing edges (I -> others)
        for neighbor in out_neighbors {
            if let Some(neighbor_node) = self.nodes.get_mut(&neighbor) {
                neighbor_node.remove_in_neighbor(id);
            }
        }

        // Remove all incoming edges (others -> I)
        // This is efficient because we track in-neighbors!
        for neighbor in in_neighbors {
            if let Some(neighbor_node) = self.nodes.get_mut(&neighbor) {
                neighbor_node.remove_out_neighbor(id);
            }
        }

        self.nodes.remove(&id)
    }

    /// Get node (immutable)
    pub fn get_node(&self, id: NodeId) -> Option<&GraphNode> {
        self.nodes.get(&id)
    }

    /// Get node (mutable)
    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut GraphNode> {
        self.nodes.get_mut(&id)
    }

    /// Get entry node for search
    pub fn entry_node(&self) -> Option<NodeId> {
        self.entry_node
    }

    /// Set entry node
    pub fn set_entry_node(&mut self, id: NodeId) {
        self.entry_node = Some(id);
    }

    /// Number of nodes
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

impl Default for BiDirectionalGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bi_directional_edge() {
        let mut graph = BiDirectionalGraph::new();

        graph.add_edge(1, 2);

        // Forward edge: 1 -> 2
        let node1 = graph.get_node(1).unwrap();
        assert_eq!(node1.out_neighbors, vec![2]);
        assert_eq!(node1.in_neighbors, Vec::<NodeId>::new());

        // Reverse edge: 2 has in-neighbor 1
        let node2 = graph.get_node(2).unwrap();
        assert_eq!(node2.in_neighbors, vec![1]);
        assert_eq!(node2.out_neighbors, Vec::<NodeId>::new());
    }

    #[test]
    fn test_remove_node() {
        let mut graph = BiDirectionalGraph::new();

        // Create graph: 1 -> 2 <- 3
        graph.add_edge(1, 2);
        graph.add_edge(3, 2);

        // Remove node 2
        let removed = graph.remove_node(2);
        assert!(removed.is_some());

        // Node 1 should have no out-neighbors
        assert_eq!(graph.get_node(1).unwrap().out_neighbors.len(), 0);

        // Node 3 should have no out-neighbors
        assert_eq!(graph.get_node(3).unwrap().out_neighbors.len(), 0);
    }

    #[test]
    fn test_entry_node() {
        let mut graph = BiDirectionalGraph::new();

        assert_eq!(graph.entry_node(), None);

        graph.add_node(42);
        assert_eq!(graph.entry_node(), Some(42));
    }
}
