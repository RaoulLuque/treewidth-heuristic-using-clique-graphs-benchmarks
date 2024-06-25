use chrono::offset::Local;
use csv::{Writer, WriterBuilder};
use dimacs_petgraph_parser::read_graph;
use petgraph::Graph;
use std::fs::{self, File};
use std::time::SystemTime;

use benchmark_suites::*;
use greedy_degree_fill_in_heuristic::greedy_degree_fill_in_heuristic;
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
    let date_and_time = Local::now()
        .to_utc()
        .to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
        .to_string();

    for (heuristic_variants, benchmark_name) in TEST_SUITE {
        println!("Starting new part of test_suite: {}", benchmark_name);
        let heuristics_variants_being_tested = heuristic_variants();

        // Creating writers
        let mut per_run_runtime_writer = WriterBuilder::new().flexible(false).from_writer(
            File::create(format!(
                "dimacs-benchmarks/benchmark_results/{}_dimacs_per_run_runtime_{}.csv",
                benchmark_name, date_and_time,
            ))
            .expect("Dimacs log file should be creatable"),
        );

        let mut average_runtime_writer = WriterBuilder::new().flexible(false).from_writer(
            File::create(format!(
                "dimacs-benchmarks/benchmark_results/{}_dimacs_average_runtime_{}.csv",
                benchmark_name, date_and_time,
            ))
            .expect("Dimacs log file should be creatable"),
        );

        let mut per_run_bound_writer = WriterBuilder::new().flexible(false).from_writer(
            File::create(format!(
                "dimacs-benchmarks/benchmark_results/{}_dimacs_per_run_bound_{}.csv",
                benchmark_name, date_and_time,
            ))
            .expect("Dimacs log file should be creatable"),
        );

        let mut average_bound_writer = WriterBuilder::new().flexible(false).from_writer(
            File::create(format!(
                "dimacs-benchmarks/benchmark_results/{}_dimacs_average_bound_{}.csv",
                benchmark_name, date_and_time,
            ))
            .expect("Dimacs log file should be creatable"),
        );

        let mut header_vec: Vec<String> = Vec::new();
        header_vec.push("Graph name".to_string());
        header_vec.push("Upper Bound".to_string());
        for heuristic in heuristics_variants_being_tested.iter() {
            for _ in 0..NUMBER_OF_REPETITIONS_PER_GRAPH {
                header_vec.push(heuristic.to_string());
            }
        }

        // Write header to csvs
        write_to_csv(
            &mut header_vec.clone(),
            &mut header_vec,
            &mut average_bound_writer,
            &mut per_run_bound_writer,
            &mut average_runtime_writer,
            &mut per_run_runtime_writer,
            NUMBER_OF_REPETITIONS_PER_GRAPH,
            true,
        )
        .expect("Writing to csv should be possible");

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

            println!(
                "{} Starting calculation on graph: {:?}",
                Local::now().to_utc().time().format("%H:%M:%S"),
                graph_file_name
            );
            let mut per_run_bound_data = Vec::new();
            let mut per_run_runtime_data = Vec::new();
            let graph_file_name_string = graph_file_name
                .into_string()
                .expect("Graph file name should be a valid utf8 String");

            // Convert optional upper bound to string
            let upper_bound_string = match upper_bound {
                Some(i) => i.to_string(),
                None => "None".to_string(),
            };

            per_run_bound_data.push(graph_file_name_string.to_owned());
            per_run_bound_data.push(upper_bound_string.to_owned());
            per_run_runtime_data.push(graph_file_name_string);
            per_run_runtime_data.push(upper_bound_string);

            for heuristic in heuristics_variants_being_tested.iter() {
                let spanning_tree_computation_and_edge_weight =
                    heuristic_to_spanning_tree_computation_type_and_edge_weight_heuristic(
                        &heuristic,
                    );
                let clique_bound = heuristic_to_clique_bound(&heuristic);

                for _ in 0..NUMBER_OF_REPETITIONS_PER_GRAPH {
                    // DEBUG
                    // println!("Iteration: {} for heuristic: {:?}", i, heuristic);
                    // Time the calculation
                    let start = SystemTime::now();

                    let computed_treewidth_upper_bound =
                        match spanning_tree_computation_and_edge_weight {
                            Some((
                                computation_type,
                                EdgeWeightTypes::ReturnI32(edge_weight_heuristic),
                            )) => compute_treewidth_upper_bound_not_connected::<_, _, Hasher, _>(
                                &graph,
                                edge_weight_heuristic,
                                computation_type,
                                false,
                                clique_bound,
                            ),
                            Some((
                                computation_type,
                                EdgeWeightTypes::ReturnI32Tuple(edge_weight_heuristic),
                            )) => compute_treewidth_upper_bound_not_connected::<_, _, Hasher, _>(
                                &graph,
                                edge_weight_heuristic,
                                computation_type,
                                false,
                                clique_bound,
                            ),
                            None => greedy_degree_fill_in_heuristic(&graph),
                        };

                    per_run_bound_data.push(computed_treewidth_upper_bound.to_string());
                    per_run_runtime_data.push(
                        start
                            .elapsed()
                            .expect("Time should be trackable")
                            .as_millis()
                            .to_string(),
                    );
                }
            }

            write_to_csv(
                &mut per_run_bound_data,
                &mut per_run_runtime_data,
                &mut average_bound_writer,
                &mut per_run_bound_writer,
                &mut average_runtime_writer,
                &mut per_run_runtime_writer,
                NUMBER_OF_REPETITIONS_PER_GRAPH,
                false,
            )
            .expect("Writing to csv should be possible");
        }
    }
}
