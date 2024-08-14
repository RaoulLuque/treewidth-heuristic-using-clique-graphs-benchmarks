use chrono::offset::Local;
use csv::WriterBuilder;
use log::{debug, info};
use petgraph::Graph;
use std::fs::File;
use std::time::SystemTime;
use std::{env, thread};

use benchmark_suites::*;
use greedy_degree_fill_in_heuristic::greedy_degree_fill_in_heuristic;
use treewidth_heuristic_using_clique_graphs::compute_treewidth_upper_bound_not_connected;

// DEBUG
// const NUMBER_OF_REPETITIONS_PER_GRAPH: usize = 1;
// const NUMBER_OF_TREES_PER_BENCHMARK_VARIANT: usize = 1;

const NUMBER_OF_REPETITIONS_PER_GRAPH: usize = 300;
const NUMBER_OF_TREES_PER_BENCHMARK_VARIANT: usize = 1;

// Debug version
#[cfg(debug_assertions)]
type Hasher = std::hash::BuildHasherDefault<rustc_hash::FxHasher>;

// Non-debug version
#[cfg(not(debug_assertions))]
type Hasher = std::hash::RandomState;

/// First coordinate is the n, second k, third p
pub const PARTIAL_K_TREE_CONFIGURATIONS: [(usize, usize, usize); 1] = [
    // (50, 5, 30),
    // (50, 5, 40),
    // (50, 5, 50),
    // (100, 5, 30),
    // (100, 5, 40),
    // (100, 5, 50),
    // (200, 5, 30),
    // (200, 5, 40),
    // (200, 5, 50),
    // (500, 5, 30),
    (500, 5, 40),
    // (500, 5, 50),
    // (50, 10, 30),
    // (50, 10, 40),
    // (50, 10, 50),
    // (100, 10, 30),
    // (100, 10, 40),
    // (100, 10, 50),
    // (200, 10, 30),
    // (200, 10, 40),
    // (200, 10, 50),
    // (500, 10, 30),
    // (500, 10, 40),
    // (500, 10, 50),
];

fn main() {
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    let mut test_suite_vec: Vec<(fn() -> Vec<HeuristicVariant>, &str)> = Vec::new();

    for command_line_argument in args.clone() {
        if let Ok(int) = command_line_argument.parse::<usize>() {
            test_suite_vec.push(TEST_SUITE[int]);
        }
    }

    if test_suite_vec.len() == 0 {
        test_suite_vec.extend(TEST_SUITE.iter());
    }

    if args.len() > 1 {
        if let Some(command_line_argument) = args.get(1) {
            if command_line_argument == "multithread" {
                multithread_benchmark(test_suite_vec)
            } else {
                single_thread_benchmark(test_suite_vec)
            }
        } else {
            single_thread_benchmark(test_suite_vec) 
        }
    } else {
        single_thread_benchmark(test_suite_vec)
    }
}

fn single_thread_benchmark(test_suite_vec: Vec<(fn() -> Vec<HeuristicVariant>, &str)>) {
    let date_and_time = current_time();

    for (heuristic_variants, benchmark_name) in test_suite_vec {
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
                            current_time(),
                            i, heuristic, j
                        );

                        // Time the calculation
                        let start = SystemTime::now();

                        let computed_treewidth = match computation_method {
                            Some((computation_type, EdgeWeightTypes::ReturnI32(a))) => {
                                compute_treewidth_upper_bound_not_connected::<_, _, _, Hasher>(
                                    &graph,
                                    a,
                                    computation_type,
                                    false,
                                    clique_bound,
                                )
                            }
                            Some((computation_type, EdgeWeightTypes::ReturnI32Tuple(a))) => {
                                compute_treewidth_upper_bound_not_connected::<_, _, _, Hasher>(
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

fn multithread_benchmark(test_suite_vec: Vec<(fn() -> Vec<HeuristicVariant>, &str)>) {
    println!("Multithreading");

    let date_and_time = current_time();

    for (heuristic_variants, benchmark_name) in test_suite_vec {
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
                    current_time(),
                    (n, k, p)
                );
                let heuristics_variants_being_tested = heuristic_variants();

                let mut per_run_bound_data_multidimensional = Vec::new();
                let mut per_run_runtime_data_multidimensional = Vec::new();

                for _ in 0..heuristics_variants_being_tested.len() {
                    per_run_bound_data_multidimensional.push(Vec::new());
                    per_run_runtime_data_multidimensional.push(Vec::new());
                }

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

                    if k == 10 && n >= 200 && (p == 30 || p == 40) && i_th_tree % 5 == 0 ||
                     k == 5 && n >= 200 && i_th_tree % 5 == 0 {
                        info!("{} (n, k, p) = {:?}: Starting calculation for tree number: {}",
                        current_time(), (n,k,p), i_th_tree);
                    }

                    let mut heuristic_variant_thread_vec = Vec::new();
                    let heuristics_variants_being_tested = heuristic_variants();

                    for (heuristic_number, heuristic) in
                        heuristics_variants_being_tested.iter().enumerate()
                    {
                        
                        let clique_bound = heuristic_to_clique_bound(heuristic);
                        let graph = graph.clone();
                        
                        let heuristic: HeuristicVariant = heuristic.to_owned();

                        heuristic_variant_thread_vec.push(thread::spawn(move || {

                        let mut graph_repetition_thread_vec = Vec::new();
                        for j in 0..NUMBER_OF_REPETITIONS_PER_GRAPH {
                            debug!(
                                "Thread {} (n, k, p) = {:?}: {} Starting calculation for tree number: {}, heuristic {:?} and {}-th graph",
                                thread_index, (n,k,p),
                                current_time(),
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
                                    compute_treewidth_upper_bound_not_connected::<_, _, _, Hasher>(
                                        &graph,
                                        a,
                                        computation_type,
                                        false,
                                        clique_bound,
                                    )
                                }
                                Some((computation_type, EdgeWeightTypes::ReturnI32Tuple(a))) => {
                                    compute_treewidth_upper_bound_not_connected::<_, _, _, Hasher>(
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