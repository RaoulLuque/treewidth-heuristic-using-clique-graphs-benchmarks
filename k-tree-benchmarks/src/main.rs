use chrono::offset::Local;
use csv::WriterBuilder;
use log::{debug, info};
use petgraph::dot::{Config, Dot};
use petgraph::graph::NodeIndex;
use petgraph::Graph;
use std::collections::HashSet;
use std::fmt::Debug;
use std::fs::{self, File};
use std::io::Write;
use std::time::SystemTime;
use std::{env, thread};

use benchmark_suites::*;
use greedy_degree_fill_in_heuristic::greedy_degree_fill_in_heuristic;
use treewidth_heuristic_using_clique_graphs::compute_treewidth_upper_bound_not_connected;

// DEBUG
// const NUMBER_OF_REPETITIONS_PER_GRAPH: usize = 1;
// const NUMBER_OF_TREES_PER_BENCHMARK_VARIANT: usize = 1;

const NUMBER_OF_REPETITIONS_PER_GRAPH: usize = 2;
const NUMBER_OF_TREES_PER_BENCHMARK_VARIANT: usize = 2;

// Debug version
#[cfg(debug_assertions)]
type Hasher = std::hash::BuildHasherDefault<rustc_hash::FxHasher>;

// Non-debug version
#[cfg(not(debug_assertions))]
type Hasher = std::hash::RandomState;

/// First coordinate is the n, second k, third p
pub const PARTIAL_K_TREE_CONFIGURATIONS: [(usize, usize, usize); 9] = [
    (100, 10, 30),
    (100, 10, 40),
    (100, 10, 50),
    // // (100, 20, 30),
    // // (100, 20, 40),
    // // (100, 20, 50),
    (200, 10, 30),
    (200, 10, 40),
    (200, 10, 50),
    // // (200, 20, 30),
    // // (200, 20, 40),
    // // (200, 20, 50),
    (500, 10, 30),
    (500, 10, 40),
    (500, 10, 50),
    // // (500, 20, 30),
    // // (500, 20, 40),
    // // (500, 20, 50),
    // (1000, 10, 30),
    // (1000, 10, 40),
    // (1000, 10, 50),
    // // (1000, 20, 30),
    // // (1000, 20, 40),
    // // (1000, 20, 50),
];

// pub const PARTIAL_K_TREE_CONFIGURATIONS: [(usize, usize, usize); 30] = [
//     (10, 5, 5),
//     (10, 5, 10),
//     (10, 5, 15),
//     (10, 5, 20),
//     (10, 5, 25),
//     (10, 5, 30),
//     (10, 5, 35),
//     (10, 5, 40),
//     (10, 5, 45),
//     (10, 5, 50),
//     (10, 5, 55),
//     (10, 5, 60),
//     (10, 5, 65),
//     (10, 5, 70),
//     (10, 5, 75),
//     (20, 5, 5),
//     (20, 5, 10),
//     (20, 5, 15),
//     (20, 5, 20),
//     (20, 5, 25),
//     (20, 5, 30),
//     (20, 5, 35),
//     (20, 5, 40),
//     (20, 5, 45),
//     (20, 5, 50),
//     (20, 5, 55),
//     (20, 5, 60),
//     (20, 5, 65),
//     (20, 5, 70),
//     (20, 5, 75),
// ];

fn main() {
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        if let Some(command_line_argument) = args.get(1) {
            if command_line_argument == "multithread" {
                multithread_benchmark()
            }
        }
    } else {
        single_thread_benchmark()
    }
}

fn single_thread_benchmark() {
    let date_and_time = current_time();

    for (heuristic_variants, benchmark_name) in TEST_SUITE {
        info!("Starting new part of test_suite: {}", benchmark_name);
        let heuristics_variants_being_tested = heuristic_variants();

        // Creating writers
        let mut per_run_runtime_writer = WriterBuilder::new().flexible(false).from_writer(
            File::create(format!(
                "k-tree-benchmarks/benchmark_results/{}_k-tree_per_run_runtime_{}.csv",
                benchmark_name, date_and_time,
            ))
            .expect("k-tree log file should be creatable"),
        );

        let mut average_runtime_writer = WriterBuilder::new().flexible(false).from_writer(
            File::create(format!(
                "k-tree-benchmarks/benchmark_results/{}_k-tree_average_runtime_{}.csv",
                benchmark_name, date_and_time,
            ))
            .expect("k-tree log file should be creatable"),
        );

        let mut per_run_bound_writer = WriterBuilder::new().flexible(false).from_writer(
            File::create(format!(
                "k-tree-benchmarks/benchmark_results/{}_k-tree_per_run_bound_{}.csv",
                benchmark_name, date_and_time,
            ))
            .expect("k-tree log file should be creatable"),
        );

        let mut average_bound_writer = WriterBuilder::new().flexible(false).from_writer(
            File::create(format!(
                "k-tree-benchmarks/benchmark_results/{}_k-tree_average_bound_{}.csv",
                benchmark_name, date_and_time,
            ))
            .expect("k-tree log file should be creatable"),
        );

        let mut header_vec: Vec<String> = Vec::new();
        header_vec.push("Graph name".to_string());
        header_vec.push("Upper Bound".to_string());
        for heuristic in heuristics_variants_being_tested.iter() {
            for i in 0..NUMBER_OF_TREES_PER_BENCHMARK_VARIANT {
                for _ in 0..NUMBER_OF_REPETITIONS_PER_GRAPH {
                    header_vec.push(format!("{}_tree_nr_{}", heuristic.to_string(), i));
                }
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

        for (n, k, p) in PARTIAL_K_TREE_CONFIGURATIONS {
            debug!(
                "{} Starting calculation on graph: {:?}",
                Local::now().to_utc().time().format("%H:%M:%S"),
                (n, k, p)
            );

            // Vec with vecs for each heuristic variant storing all results in successive order to merge together afterwards
            let mut per_run_bound_data_multidimensional = Vec::new();
            let mut per_run_runtime_data_multidimensional = Vec::new();

            for _ in heuristics_variants_being_tested.iter() {
                per_run_bound_data_multidimensional.push(Vec::new());
                per_run_runtime_data_multidimensional.push(Vec::new());
            }

            for i in 0..NUMBER_OF_TREES_PER_BENCHMARK_VARIANT {
                let graph: Graph<i32, i32, petgraph::prelude::Undirected> =
                treewidth_heuristic_using_clique_graphs::generate_partial_k_tree_with_guaranteed_treewidth(
                    k,
                    n,
                    p,
                    &mut rand::thread_rng(),
                )
                .expect("n should be greater than k");

                for (heuristic_number, heuristic) in
                    heuristics_variants_being_tested.iter().enumerate()
                {
                    let computation_method =
                        heuristic_to_spanning_tree_computation_type_and_edge_weight_heuristic(
                            heuristic,
                        );
                    let clique_bound = heuristic_to_clique_bound(heuristic);

                    for j in 0..NUMBER_OF_REPETITIONS_PER_GRAPH {
                        debug!(
                            "{} Starting calculation for tree number: {}, heuristic {:?} and {}-th graph",
                            Local::now().to_utc().time().format("%H:%M:%S"),
                            i, heuristic, j
                        );

                        // Time the calculation
                        let start = SystemTime::now();

                        let computed_treewidth = match computation_method {
                            Some((computation_type, EdgeWeightTypes::ReturnI32(a))) => {
                                compute_treewidth_upper_bound_not_connected::<_, _, Hasher, _>(
                                    &graph,
                                    a,
                                    computation_type,
                                    false,
                                    clique_bound,
                                )
                            }
                            Some((computation_type, EdgeWeightTypes::ReturnI32Tuple(a))) => {
                                compute_treewidth_upper_bound_not_connected::<_, _, Hasher, _>(
                                    &graph,
                                    a,
                                    computation_type,
                                    false,
                                    clique_bound,
                                )
                            }
                            None => greedy_degree_fill_in_heuristic(&graph),
                        };
                        per_run_bound_data_multidimensional
                            .get_mut(heuristic_number)
                            .expect("Index should be in bound by loop invariant")
                            .push(computed_treewidth.to_string());
                        per_run_runtime_data_multidimensional
                            .get_mut(heuristic_number)
                            .expect("Index should be in bound by loop invariant")
                            .push(
                                start
                                    .elapsed()
                                    .expect("Time should be trackable")
                                    .as_millis()
                                    .to_string(),
                            );
                    }
                }
            }

            let mut per_run_bound_data = Vec::new();
            let mut per_run_runtime_data = Vec::new();

            per_run_bound_data.push(format!("({};{};{})", n, k, p));
            per_run_bound_data.push(k.to_string());
            per_run_runtime_data.push(format!("({};{};{})", n, k, p));
            per_run_runtime_data.push(k.to_string());

            for bound_data_for_one_heuristic in per_run_bound_data_multidimensional.iter_mut() {
                per_run_bound_data.append(bound_data_for_one_heuristic);
            }

            for runtime_data_for_one_heuristic in per_run_runtime_data_multidimensional.iter_mut() {
                per_run_runtime_data.append(runtime_data_for_one_heuristic);
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

fn multithread_benchmark() {
    println!("Multithreading");

    let date_and_time = current_time();

    for (heuristic_variants, benchmark_name) in TEST_SUITE {
        info!("Starting new part of test_suite: {}", benchmark_name);
        let heuristics_variants_being_tested = heuristic_variants();

        // Creating writers
        let mut per_run_runtime_writer = WriterBuilder::new().flexible(false).from_writer(
            File::create(format!(
                "k-tree-benchmarks/benchmark_results/{}_k-tree_per_run_runtime_{}_multithreaded.csv",
                benchmark_name, date_and_time,
            ))
            .expect("k-tree log file should be creatable"),
        );

        let mut average_runtime_writer = WriterBuilder::new().flexible(false).from_writer(
            File::create(format!(
                "k-tree-benchmarks/benchmark_results/{}_k-tree_average_runtime_{}_multithreaded.csv",
                benchmark_name, date_and_time,
            ))
            .expect("k-tree log file should be creatable"),
        );

        let mut per_run_bound_writer = WriterBuilder::new().flexible(false).from_writer(
            File::create(format!(
                "k-tree-benchmarks/benchmark_results/{}_k-tree_per_run_bound_{}_multithreaded.csv",
                benchmark_name, date_and_time,
            ))
            .expect("k-tree log file should be creatable"),
        );

        let mut average_bound_writer = WriterBuilder::new().flexible(false).from_writer(
            File::create(format!(
                "k-tree-benchmarks/benchmark_results/{}_k-tree_average_bound_{}_multithreaded.csv",
                benchmark_name, date_and_time,
            ))
            .expect("k-tree log file should be creatable"),
        );

        let mut header_vec: Vec<String> = Vec::new();
        header_vec.push("Graph name".to_string());
        header_vec.push("Upper Bound".to_string());
        for heuristic in heuristics_variants_being_tested.iter() {
            for i in 0..NUMBER_OF_TREES_PER_BENCHMARK_VARIANT {
                for _ in 0..NUMBER_OF_REPETITIONS_PER_GRAPH {
                    header_vec.push(format!("{}_tree_nr_{}", heuristic.to_string(), i));
                }
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
        let mut thread_vec = Vec::new();

        for thread_index in 0..PARTIAL_K_TREE_CONFIGURATIONS.len() {
            thread_vec.push(thread::spawn(move || {
                debug!("Thread {} starting", thread_index);
                let (n,k,p) = PARTIAL_K_TREE_CONFIGURATIONS[thread_index];

                debug!(
                    "{} Starting calculation on graph: {:?}",
                    Local::now().to_utc().time().format("%H:%M:%S"),
                    (n, k, p)
                );
                let heuristics_variants_being_tested = heuristic_variants();

                let mut per_run_bound_data_multidimensional = Vec::new();
                let mut per_run_runtime_data_multidimensional = Vec::new();

                for _ in 0..heuristics_variants_being_tested.len() {
                    per_run_bound_data_multidimensional.push(Vec::new());
                    per_run_runtime_data_multidimensional.push(Vec::new());
                }

                // let mut thread_vec_number_of_trees = Vec::new();

                for i_th_tree in 0..NUMBER_OF_TREES_PER_BENCHMARK_VARIANT {

                    // thread_vec_number_of_trees.push(thread::spawn(move || {
                    let graph: Graph<i32, i32, petgraph::prelude::Undirected> =
                    treewidth_heuristic_using_clique_graphs::generate_partial_k_tree_with_guaranteed_treewidth(
                        k,
                        n,
                        p,
                        &mut rand::thread_rng(),
                    )
                    .expect("n should be greater than k");



                    let mut heuristic_variant_thread_vec = Vec::new();
                    let heuristics_variants_being_tested = heuristic_variants();

                    for (heuristic_number, heuristic) in
                        heuristics_variants_being_tested.iter().enumerate()
                    {
                        
                        let clique_bound = heuristic_to_clique_bound(heuristic);
                        let graph = graph.clone();
                        
                        let heuristic: HeuristicTypes = heuristic.to_owned();

                        heuristic_variant_thread_vec.push(thread::spawn(move || {

                        let mut graph_repetition_thread_vec = Vec::new();
                        for j in 0..NUMBER_OF_REPETITIONS_PER_GRAPH {
                            debug!(
                                "Thread {} (n, k, p) = {:?}: {} Starting calculation for tree number: {}, heuristic {:?} and {}-th graph",
                                thread_index, (n,k,p),
                                Local::now().to_utc().time().format("%H:%M:%S"),
                                i_th_tree, heuristic, j
                            );
                            let graph = graph.clone();

                            graph_repetition_thread_vec.push(thread::spawn(move || {
                                let computation_method =
                                heuristic_to_spanning_tree_computation_type_and_edge_weight_heuristic(
                                    &heuristic,
                                );

                            // Time the calculation
                            let start = SystemTime::now();

                            let computed_treewidth = match computation_method {
                                Some((computation_type, EdgeWeightTypes::ReturnI32(a))) => {
                                    compute_treewidth_upper_bound_not_connected::<_, _, Hasher, _>(
                                        &graph,
                                        a,
                                        computation_type,
                                        false,
                                        clique_bound,
                                    )
                                }
                                Some((computation_type, EdgeWeightTypes::ReturnI32Tuple(a))) => {
                                    compute_treewidth_upper_bound_not_connected::<_, _, Hasher, _>(
                                        &graph,
                                        a,
                                        computation_type,
                                        false,
                                        clique_bound,
                                    )
                                }
                                None => greedy_degree_fill_in_heuristic(&graph),
                            };
                            (computed_treewidth, start
                                .elapsed()
                                .expect("Time should be trackable")
                                .as_millis()
                                .to_string())
                            }));
                            // per_run_bound_data_multidimensional
                            //     .get_mut(heuristic_number)
                            //     .expect("Index should be in bound by loop invariant")
                            //     .push(computed_treewidth.to_string());
                            // per_run_runtime_data_multidimensional
                            //     .get_mut(heuristic_number)
                            //     .expect("Index should be in bound by loop invariant")
                            //     .push(
                            //         start
                                // .elapsed()
                                // .expect("Time should be trackable")
                                // .as_millis()
                                // .to_string()),
                            //     );
                        }
                        let mut results_from_graph_repetitions = Vec::new();
                        for thread_handle in graph_repetition_thread_vec {
                            results_from_graph_repetitions.push(thread_handle.join());
                        }
                        let results_from_graph_repetitions: Vec<_> = results_from_graph_repetitions
                        .iter_mut()
                        .map(|thread_results| {
                            thread_results
                                .as_mut()
                                .expect("Threads should return results")
                        })
                        .collect();

                        let (graph_repetition_bound_result, graph_repetition_runtime_result): (Vec<_>, Vec<_>) = results_from_graph_repetitions.into_iter().map(|(a, b)| (a.to_string(), b.to_owned())).unzip();

                        // These are the results of one heuristic on one graph (NUMBER_OF_REPETITIONS_PER_GRAPH number of runs)
                        (heuristic_number, graph_repetition_bound_result, graph_repetition_runtime_result)
                        }));
                    }
                    let mut results_from_each_heuristic_variant = Vec::new();
                    for thread_handle in heuristic_variant_thread_vec {
                        results_from_each_heuristic_variant.push(thread_handle.join());
                    }
                    let results_from_each_heuristic_variant: Vec<_> = results_from_each_heuristic_variant
                        .iter_mut()
                        .map(|thread_results| {
                            thread_results
                                .as_mut()
                                .expect("Threads should return results")
                        })
                        .collect();
                    for (heuristic_number, bound_results, runtime_results) in results_from_each_heuristic_variant.into_iter() {
                        per_run_bound_data_multidimensional[*heuristic_number].append(bound_results);
                        per_run_runtime_data_multidimensional[*heuristic_number].append(runtime_results);
                    }
                }
                // let mut results_from_each_tree = Vec::new();
                // for thread_handle in thread_vec_number_of_trees {
                //     results_from_each_tree.push(thread_handle.join());
                // }
                // let results_from_each_tree: Vec<_> = results_from_each_tree
                //         .iter_mut()
                //         .map(|thread_results| {
                //             thread_results
                //                 .as_mut()
                //                 .expect("Threads should return results")
                //         })
                //         .collect();

                // // Vec with vecs for each heuristic variant storing all results in successive order to merge together afterwards
                // let mut per_run_bound_data_multidimensional = Vec::new();
                // let mut per_run_runtime_data_multidimensional = Vec::new();

                // for (index, (per_run_bound_data_tree, per_run_runtime_data_tree)) in results_from_each_tree.into_iter().enumerate() {
                //     if index == 0 {
                //         per_run_bound_data_multidimensional = per_run_bound_data_tree.to_vec();
                //         per_run_runtime_data_multidimensional = per_run_runtime_data_tree.to_vec();
                //     } else {
                //         for heuristic_index in 0..per_run_bound_data_tree.len() {
                //             per_run_bound_data_multidimensional[heuristic_index].append(&mut per_run_bound_data_tree[heuristic_index]);
                //             per_run_runtime_data_multidimensional[heuristic_index].append(&mut per_run_runtime_data_tree[heuristic_index]);
                //         }   
                //     }
                // }


                info!("Thread {} (n, k, p) {:?}: Finished", thread_index, (n,k,p));
                ((n,k,p), per_run_bound_data_multidimensional, per_run_runtime_data_multidimensional)
            }));
        }
        let mut results_from_each_thread = Vec::new();
        for thread_handle in thread_vec {
            results_from_each_thread.push(thread_handle.join());
        }

        let results_from_each_thread: Vec<_> = results_from_each_thread
            .iter_mut()
            .map(|thread_results| {
                thread_results
                    .as_mut()
                    .expect("Threads should return results")
            })
            .collect();

        for (
            (n, k, p),
            per_run_bound_data_multidimensional,
            per_run_runtime_data_multidimensional,
        ) in results_from_each_thread
        {
            let mut per_run_bound_data = Vec::new();
            let mut per_run_runtime_data = Vec::new();

            per_run_bound_data.push(format!("({};{};{})", n, k, p));
            per_run_bound_data.push(k.to_string());
            per_run_runtime_data.push(format!("({};{};{})", n, k, p));
            per_run_runtime_data.push(k.to_string());

            for bound_data_for_one_heuristic in per_run_bound_data_multidimensional.iter_mut() {
                per_run_bound_data.append(bound_data_for_one_heuristic);
            }

            for runtime_data_for_one_heuristic in per_run_runtime_data_multidimensional.iter_mut() {
                per_run_runtime_data.append(runtime_data_for_one_heuristic);
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

// Converting dot files to pdf in bulk:
// FullPath -type f -name "*.dot" | xargs dot -Tpdf -O
#[allow(dead_code)]
fn create_dot_files<O: Debug, S>(
    graph: &Graph<i32, i32, petgraph::prelude::Undirected>,
    clique_graph: &Graph<HashSet<NodeIndex, S>, O, petgraph::prelude::Undirected>,
    clique_graph_tree_after_filling_up: &Graph<
        HashSet<NodeIndex, S>,
        O,
        petgraph::prelude::Undirected,
    >,
    clique_graph_tree_before_filling_up: &Option<
        Graph<HashSet<NodeIndex, S>, O, petgraph::prelude::Undirected>,
    >,
    i: usize,
    name: &str,
) {
    fs::create_dir_all("k_tree_benchmarks/benchmark_results/visualizations")
        .expect("Could not create directory for visualizations");

    let start_graph_dot_file = Dot::with_config(graph, &[Config::EdgeNoLabel]);
    let result_graph_dot_file =
        Dot::with_config(clique_graph_tree_after_filling_up, &[Config::EdgeNoLabel]);
    let clique_graph_dot_file = Dot::with_config(&clique_graph, &[Config::EdgeNoLabel]);

    if let Some(clique_graph_tree_before_filling_up) = clique_graph_tree_before_filling_up {
        let clique_graph_tree_before_filling_up_dot_file =
            Dot::with_config(clique_graph_tree_before_filling_up, &[Config::EdgeNoLabel]);
        let clique_graph_node_indices = Dot::with_config(
            clique_graph_tree_before_filling_up,
            &[Config::EdgeNoLabel, Config::NodeIndexLabel],
        );

        let mut w = fs::File::create(format!(
            "k_tree_benchmarks/benchmark_results/visualizations/{}_result_graph_before_filling_{}.dot",
            i, name
        ))
        .expect("Result graph without filling up file could not be created");
        write!(&mut w, "{:?}", clique_graph_tree_before_filling_up_dot_file)
            .expect("Unable to write dotfile for result graph without filling up to files");

        let mut w = fs::File::create(format!(
            "k_tree_benchmarks/benchmark_results/visualizations/{}_result_graph_node_indices_{}.dot",
            i, name
        ))
        .expect("Clique graph node indices file could not be created");
        write!(&mut w, "{:?}", clique_graph_node_indices)
            .expect("Unable to write dotfile for Clique graph node indices  to files");
    }

    let mut w = fs::File::create(format!(
        "k_tree_benchmarks/benchmark_results/visualizations/{}_starting_graph_{}.dot",
        i, name
    ))
    .expect("Start graph file could not be created");
    write!(&mut w, "{:?}", start_graph_dot_file)
        .expect("Unable to write dotfile for start graph to files");

    let mut w = fs::File::create(format!(
        "k_tree_benchmarks/benchmark_results/visualizations/{}_clique_graph_{}.dot",
        i, name
    ))
    .expect("Start graph file could not be created");
    write!(&mut w, "{:?}", clique_graph_dot_file)
        .expect("Unable to write dotfile for start graph to files");

    let mut w = fs::File::create(format!(
        "k_tree_benchmarks/benchmark_results/visualizations/{}_result_graph_{}.dot",
        i, name
    ))
    .expect("Result graph file could not be created");
    write!(&mut w, "{:?}", result_graph_dot_file)
        .expect("Unable to write dotfile for result graph to files");
}
