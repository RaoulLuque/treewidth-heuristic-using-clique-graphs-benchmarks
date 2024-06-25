use itertools::Itertools;
use petgraph::{graph::NodeIndex, Graph, Undirected};
use std::collections::HashSet;

/// Greedy heuristic for computing an upper bound on the treewidth as described in
/// [this](https://doi.org/10.1016/j.ic.2009.03.008) paper by Bodlaender and Koster
/// (this variant is Greedy Degree + Fill In).
pub fn greedy_degree_fill_in_heuristic<N: Clone, E: Clone + Default>(
    graph: &Graph<N, E, Undirected>,
) -> usize {
    greedy_degree_fill_in(graph).1
}

/// See [greedy_degree_fill_in_heuristic]
fn greedy_degree_fill_in<N: Clone, E: Clone + Default>(
    graph: &Graph<N, E, Undirected>,
) -> (Vec<NodeIndex>, usize) {
    let mut elimination_ordering: Vec<_> = Vec::with_capacity(graph.node_count());
    let mut treewidth_upper_bound: usize = usize::MAX;
    let mut number_of_iterations = 0;
    let mut graph_copy = graph.clone();
    let mut remaining_vertices: HashSet<NodeIndex> = graph.node_indices().collect();

    while !remaining_vertices.is_empty() {
        // Minimize the sum of the number of edges that would have to be filled up choosing a vertex as the next
        // vertex in the elimination ordering and the number of neighbors of the vertex across all remaining vertices
        let current_vertex = remaining_vertices
            .iter()
            .min_by_key(|v| {
                let mut number_of_edges_that_would_have_to_be_added = 0;
                for tuple in graph_copy.neighbors(**v).combinations(2) {
                    let (x, y) = (
                        tuple
                            .get(0)
                            .expect("Tuple should contain exactly two elements"),
                        tuple
                            .get(1)
                            .expect("Tuple should contain exactly two elements"),
                    );
                    if !graph_copy.contains_edge(*x, *y) {
                        number_of_edges_that_would_have_to_be_added += 1;
                    }
                }
                number_of_edges_that_would_have_to_be_added
                    + graph_copy.neighbors(**v).collect_vec().len()
            })
            .expect("remaining vertices shouldn't be empty by loop invariant")
            .clone();

        let mut edges_to_be_added: HashSet<_> = HashSet::new();
        for tuple in graph_copy.neighbors(current_vertex).combinations(2) {
            let (x, y) = (
                tuple
                    .get(0)
                    .expect("Tuple should contain exactly two elements"),
                tuple
                    .get(1)
                    .expect("Tuple should contain exactly two elements"),
            );
            if !graph_copy.contains_edge(*x, *y) {
                edges_to_be_added.insert((*x, *y));
            }
        }
        // Treewidth is max number of neighbors that appear after a vertex v in the final
        // elimination ordering (maximizing over the v in the elimination ordering)
        treewidth_upper_bound = treewidth_upper_bound.max(edges_to_be_added.len());

        for (x, y) in edges_to_be_added {
            graph_copy.add_edge(x, y, E::default());
        }

        elimination_ordering.insert(number_of_iterations, current_vertex);
        number_of_iterations += 1;
        remaining_vertices.remove(&current_vertex);
    }

    (elimination_ordering, treewidth_upper_bound)
}
