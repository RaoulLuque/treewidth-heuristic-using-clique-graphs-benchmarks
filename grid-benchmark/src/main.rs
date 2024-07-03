use std::collections::HashSet;

use log::{debug, info};
use petgraph::{
    graph::{EdgeIndex, NodeIndex},
    Graph, Undirected,
};
use treewidth_heuristic_using_clique_graphs::{
    construct_clique_graph::construct_clique_graph_with_bags,
    find_maximal_cliques::find_maximal_cliques,
};

fn main() {
    env_logger::init();

    for n in 1..10 {
        let grid_graph = generate_grid_graph(n);

        let cliques_in_clique_graph =
            find_maximal_cliques::<Vec<_>, _, std::hash::RandomState>(&grid_graph);
        let (clique_graph_of_grid_graph, clique_graph_map) = construct_clique_graph_with_bags(
            cliques_in_clique_graph,
            treewidth_heuristic_using_clique_graphs::negative_intersection::<std::hash::RandomState>,
        );

        let mut best_treewidth_upper_bound = usize::MAX;

        // Loop over all possible spanning trees
        let starting_vertex = grid_graph
            .node_indices()
            .next()
            .expect("Graph shouldn't be empty");
        let mut stack = Vec::new();

        let mut current_spanning_tree_vertices = Vec::new();
        current_spanning_tree_vertices.push(starting_vertex);

        let current_spanning_tree_edges = Vec::new();

        let mut current_spanning_tree: Graph<
            std::collections::HashSet<petgraph::prelude::NodeIndex>,
            i32,
            Undirected,
        > = clique_graph_of_grid_graph.clone();
        for i in (0..current_spanning_tree.edge_count()).rev() {
            current_spanning_tree.remove_edge(EdgeIndex::new(i));
        }

        let mut promising_edges = Vec::new();
        for neighbour in grid_graph.neighbors(starting_vertex) {
            promising_edges.push((starting_vertex, neighbour));
        }

        let mut dict_with_seen_edges: HashSet<_> = HashSet::new();

        stack.push((current_spanning_tree_vertices, current_spanning_tree_edges));

        while !stack.is_empty() {
            debug!(
                "Current state of stack: {:?}",
                stack
                    .clone()
                    .into_iter()
                    .map(|(a, b)| (a, b))
                    .unzip::<_, _, Vec<_>, Vec<_>>()
                    .0
            );
            let (current_spanning_tree_vertices, current_spanning_tree_edges) = stack
                .pop()
                .expect("Stack shouldn't be empty by loop invariant");

            debug!("Current state of dict: {:?}", dict_with_seen_edges);

            let mut edges: Vec<_> = current_spanning_tree_edges.clone();
            edges.sort();
            if !dict_with_seen_edges.insert(edges) {
                continue;
            }

            let promising_edges =
                find_promising_edges(&clique_graph_of_grid_graph, &current_spanning_tree_vertices);

            if promising_edges.is_empty() {
                let mut current_spanning_tree = current_spanning_tree.clone();
                for (v, w) in current_spanning_tree_edges {
                    current_spanning_tree.add_edge(v, w, 0);
                }

                let upper_bound = compute_upper_bound_for_given_spanning_tree(
                    &mut current_spanning_tree,
                    &clique_graph_map,
                );

                if upper_bound < best_treewidth_upper_bound {
                    best_treewidth_upper_bound = upper_bound;
                    info!(
                        "Best upper bound reduced to: {}",
                        best_treewidth_upper_bound
                    );
                    if best_treewidth_upper_bound == n {
                        break;
                    }
                }
            } else {
                for (current_vertex, new_vertex) in promising_edges {
                    let mut new_spanning_tree_vertices = current_spanning_tree_vertices.clone();
                    let mut new_spanning_tree_edges = current_spanning_tree_edges.clone();
                    new_spanning_tree_vertices.push(new_vertex);
                    new_spanning_tree_edges.push((current_vertex, new_vertex));

                    stack.push((new_spanning_tree_vertices, new_spanning_tree_edges));
                }
            }
        }

        println!(
            "n is: {} and the best upper bound found is: {}",
            n, best_treewidth_upper_bound
        );

        println!(
            "Total number of spanning trees: {}",
            dict_with_seen_edges
                .iter()
                .filter(|v| v.len() == 11)
                .collect::<Vec<_>>()
                .len()
        )
    }
}

fn generate_grid_graph(n: usize) -> Graph<i32, i32, Undirected> {
    let mut graph: Graph<i32, i32, petgraph::prelude::Undirected> =
        petgraph::Graph::new_undirected();

    let mut grid_vec = Vec::new();
    for _ in 0..n {
        grid_vec.push(Vec::new());
    }

    for i in 0..n {
        for _ in 0..n {
            grid_vec[i].push(graph.add_node(0));
        }
    }

    for i in 0..n {
        for j in 0..n {
            if i != n - 1 {
                graph.add_edge(grid_vec[i][j], grid_vec[i + 1][j], 0);
            }
            if j != n - 1 {
                graph.add_edge(grid_vec[i][j], grid_vec[i][j + 1], 0);
            }
        }
    }

    graph
}

fn find_promising_edges(
    graph: &Graph<std::collections::HashSet<petgraph::prelude::NodeIndex>, i32, Undirected>,
    current_spanning_tree_vertices: &Vec<NodeIndex>,
) -> Vec<(NodeIndex, NodeIndex)> {
    let mut promising_edges = Vec::new();
    for vertex in current_spanning_tree_vertices {
        for neighbor in graph.neighbors(*vertex) {
            if !current_spanning_tree_vertices.contains(&neighbor) {
                promising_edges.push((*vertex, neighbor));
            }
        }
    }

    promising_edges
}

fn compute_upper_bound_for_given_spanning_tree(
    current_spanning_tree: &mut Graph<
        std::collections::HashSet<petgraph::prelude::NodeIndex>,
        i32,
        Undirected,
    >,
    clique_graph_map: &std::collections::HashMap<
        NodeIndex,
        std::collections::HashSet<NodeIndex, std::hash::RandomState>,
        std::hash::RandomState,
    >,
) -> usize {
    treewidth_heuristic_using_clique_graphs::fill_bags_along_paths::fill_bags_along_paths_using_structure(current_spanning_tree, clique_graph_map);

    treewidth_heuristic_using_clique_graphs::find_width_of_tree_decomposition::find_width_of_tree_decomposition(current_spanning_tree)
}

fn is_tree<S: Clone, E: Clone>(graph: &mut Graph<S, E, Undirected>) -> bool {
    graph.node_count() == graph.edge_count() + 1 && treewidth_heuristic_using_clique_graphs::find_connected_components::find_connected_components::<Vec<_>, _, _, std::hash::RandomState>(&graph).collect::<Vec<_>>().len() == 1
}
