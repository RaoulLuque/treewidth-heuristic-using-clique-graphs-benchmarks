use chrono::offset::Local;
use csv::Writer;
use dimacs_petgraph_parser::read_graph;
use petgraph::Graph;
use std::fs::{self, File};
use std::io::Write;
use std::time::SystemTime;

use benchmark_suites::*;
use treewidth_heuristic_clique_graph::compute_treewidth_upper_bound_not_connected;

const NUMBER_OF_REPETITIONS_PER_GRAPH: usize = 5;

// Debug version
#[cfg(debug_assertions)]
type Hasher = std::hash::BuildHasherDefault<rustc_hash::FxHasher>;

// Non-debug version
#[cfg(not(debug_assertions))]
type Hasher = std::hash::RandomState;

fn main() {
    env_logger::init();
    let date_and_time = Local::now().to_string();

    for (heuristic_variants, benchmark_name) in TEST_SUITE {
        // Creating writers
        let mut per_run_runtime_writer = Writer::from_writer(
            File::create(format!(
                "dimacs-benchmarks/benchmark_results/{}_dimacs_per_run_runtime_{}",
                benchmark_name, date_and_time,
            ))
            .expect("Dimacs log file should be creatable"),
        );

        let mut average_runtime_writer = Writer::from_writer(
            File::create(format!(
                "dimacs-benchmarks/benchmark_results/{}_dimacs_average_runtime_{}",
                benchmark_name, date_and_time,
            ))
            .expect("Dimacs log file should be creatable"),
        );

        let mut per_run_bound_writer = Writer::from_writer(
            File::create(format!(
                "dimacs-benchmarks/benchmark_results/{}_dimacs_per_run_runtime_{}",
                benchmark_name, date_and_time,
            ))
            .expect("Dimacs log file should be creatable"),
        );

        let mut average_bound_writer = Writer::from_writer(
            File::create(format!(
                "dimacs-benchmarks/benchmark_results/{}_dimacs_average_runtime_{}",
                benchmark_name, date_and_time,
            ))
            .expect("Dimacs log file should be creatable"),
        );



        // Sorting files in dimacs directory
        let dimacs_graphs_paths: fs::ReadDir =
            fs::read_dir("dimacs-benchmarks/dimacs_graphs/color/").unwrap();
        let mut dimacs_graph_paths_vec = Vec::new();
        for graph_path_res in dimacs_graphs_paths {
            if let Ok(graph_path) = graph_path_res {
                if let Some(extension) = graph_path.path().extension() {
                    if extension == "col" {
                        dimacs_graph_paths_vec.push(graph_path);
                    }
                }
            }
        }
        dimacs_graph_paths_vec.sort_by(|a, b| b.file_name().cmp(&a.file_name()));

        let mut log = "".to_string();
        log.push_str(&format!(
            "| {0: <20} | {1: <12} |",
            "Graph name", "Upper bound"
        ));

        let header_vec: Vec<_> = Vec::new();
        header_vec.push("Graph name");
        for heuristic in HEURISTICS_BEING_TESTED {
            let heuristic_string = heuristic.to_string();
            log.push_str(&format!(" {0: <15} |", heuristic_string))
        }
        log.push_str("\n");

        for graph_path in dimacs_graph_paths_vec {
            let graph_file_name = graph_path.file_name();
            let graph_file =
                File::open(graph_path.path()).expect("Graph file should exist and be readable");

            let (graph, _, _, upper_bound): (
                Graph<i32, i32, petgraph::prelude::Undirected>,
                _,
                _,
                _,
            ) = read_graph(graph_file).expect("Graph should be in correct format");

            println!("Starting calculation on graph: {:?}", graph_file_name);
            let mut calculation_vec = Vec::new();
            for heuristic in HEURISTICS_BEING_TESTED {
                // Time the calculation
                let start = SystemTime::now();
                let mut treewidth: usize = usize::MAX;

                let edge_weight_heuristic = heuristic_to_edge_weight_heuristic(&heuristic);
                let computation_type = heuristic_to_computation_type(&heuristic);
                let clique_bound = heuristic_to_clique_bound(&heuristic);

                for i in 0..NUMBER_OF_REPETITIONS_PER_GRAPH {
                    println!("Iteration: {} for heuristic: {:?}", i, heuristic);
                    let computed_treewidth = match edge_weight_heuristic {
                        EdgeWeightTypes::ReturnI32(edge_weight_heuristic) => {
                            compute_treewidth_upper_bound_not_connected::<_, _, Hasher, _>(
                                &graph,
                                edge_weight_heuristic,
                                computation_type,
                                false,
                                clique_bound,
                            )
                        }
                        EdgeWeightTypes::ReturnI32Tuple(edge_weight_heuristic) => {
                            compute_treewidth_upper_bound_not_connected::<_, _, Hasher, _>(
                                &graph,
                                edge_weight_heuristic,
                                computation_type,
                                false,
                                clique_bound,
                            )
                        }
                    };

                    if computed_treewidth < treewidth {
                        treewidth = computed_treewidth;
                    }
                }

                calculation_vec.push((
                    treewidth,
                    start
                        .elapsed()
                        .expect("Time should be trackable")
                        .as_millis()
                        / NUMBER_OF_REPETITIONS_PER_GRAPH,
                ))
            }

            log.push_str("\n");

            benchmark_log_file
                .write_all(log.as_bytes())
                .expect("Writing to Dimacs log file should be possible");
        }
    }
}
