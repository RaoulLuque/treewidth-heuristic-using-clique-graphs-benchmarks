#[derive(Debug, Copy, Clone)]

/// Naming: 12345 I 678910 (I BC)
/// 12345 Abbreviation of spanning tree construction
/// 5678910 Abbreviation of edge weights used
/// BC Bounded Cliques (optional)
/// I separation letter
pub enum HeuristicTypes {
    // For comparison of edge weights:
    MSTreILeDif,
    MSTreINegIn,
    MSTreIUnion,
    MSTreIDisjU,
    MSTreIConst,
    MSTreIRandd,

    // For comparison of combined edge weights:
    MSTreINiTLd,
    MSTreILdTNi,

    // For comparison of spanning tree construction:
    FilWhINiTLd,
    FWhUEINiTLd, // Update edges in clique graph according to filling up whilst building spanning tree
    FWBagINonee,

    // For comparison with GreedyX
    GreedyDegreeFillIn,

    // For comparison with bounded clique
    FilWhINiTLdIBC(usize),

    // Old / Not used
    FilWhINegIn,
    FilWhILeDif,
    FiWhTINiTLd,
    FilWhILdTNi,
    MSTreINiTLdIBC(usize),
    FiWhTINiTLdIBC(usize),
}

use chrono::TimeZone;
use csv::Writer;
use log::debug;
use petgraph::graph::NodeIndex;
use HeuristicTypes::*;

pub const TEST_SUITE: [(fn() -> Vec<HeuristicTypes>, &str); 6] = [
    // DEBUG
    // (test_test_suite, "z_test_test_suite"),
    // Actual Tests
    (comparison_of_edge_weights, "comparison_of_edge_weights"),
    (
        comparison_of_combined_edge_weights,
        "comparison_of_combined_edge_weights",
    ),
    (
        comparison_of_spanning_tree_construction,
        "comparison_of_spanning_tree_construction",
    ),
    (
        comparison_with_greedy_degree_fill_in,
        "comparison_with_greedy_degree_fill_in",
    ),
    (
        comparison_runtime_mst_and_fill_while,
        "comparison_running_time_mst_and_fill_while",
    ),
    (
        comparison_of_exotic_spanning_tree_construction,
        "comparison_of_exotic_spanning_tree_construction",
    ),
];

pub fn test_if_fill_while_works() -> Vec<HeuristicTypes> {
    vec![FilWhINiTLd, MSTreINegIn, MSTreINiTLd]
}

pub fn test_test_suite() -> Vec<HeuristicTypes> {
    vec![MSTreINegIn, MSTreIConst]
}

pub fn comparison_of_edge_weights() -> Vec<HeuristicTypes> {
    vec![
        MSTreILeDif,
        MSTreINegIn,
        MSTreIUnion,
        MSTreIDisjU,
        MSTreIConst,
        MSTreIRandd,
    ]
}

pub fn comparison_of_combined_edge_weights() -> Vec<HeuristicTypes> {
    vec![MSTreINegIn, MSTreINiTLd, MSTreILdTNi]
}

pub fn comparison_of_spanning_tree_construction() -> Vec<HeuristicTypes> {
    vec![
        MSTreINegIn,
        MSTreINiTLd,
        MSTreILdTNi,
        FilWhINegIn,
        FilWhINiTLd,
        FilWhILdTNi,
    ]
}

pub fn comparison_of_exotic_spanning_tree_construction() -> Vec<HeuristicTypes> {
    vec![FilWhINiTLd, FWhUEINiTLd, FWBagINonee]
}

pub fn comparison_with_greedy_degree_fill_in() -> Vec<HeuristicTypes> {
    vec![FilWhINiTLd, GreedyDegreeFillIn]
}

pub fn comparison_runtime_mst_and_fill_while() -> Vec<HeuristicTypes> {
    vec![FilWhINiTLd, MSTreINiTLd]
}

impl std::fmt::Display for HeuristicTypes {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let display_string = match self {
            MSTreIUnion => "MSTreIUnion".to_string(),
            MSTreIDisjU => "MSTreIDisjU".to_string(),
            MSTreINegIn => "MSTreINegIn".to_string(),
            FilWhINegIn => "FilWhINegIn".to_string(),
            MSTreILeDif => "MSTreILeDif".to_string(),
            FilWhILeDif => "FilWhILeDif".to_string(),
            MSTreINiTLd => "MSTreINiTLd".to_string(),
            FilWhINiTLd => "FilWhINiTLd".to_string(),
            FWhUEINiTLd => "FWhUEINiTLd".to_string(),
            MSTreILdTNi => "MSTreILdTNi".to_string(),
            FilWhILdTNi => "FilWhILdTNi".to_string(),
            FiWhTINiTLd => "FiWhTINiTLd".to_string(),
            FWBagINonee => "FWBagINonee".to_string(),
            MSTreINiTLdIBC(clique_bound) => format!("MSTreINiTLdIBC {}", clique_bound),
            FilWhINiTLdIBC(clique_bound) => format!("FilWhINiTLdIBC {}", clique_bound),
            FiWhTINiTLdIBC(clique_bound) => format!("FiWhTINiTLdIBC {}", clique_bound),
            GreedyDegreeFillIn => format!("GreedyDegreeFillIn"),
            MSTreIConst => "MSTreIConst".to_string(),
            MSTreIRandd => "MSTreIRandd".to_string(),
        };
        write!(f, "{}", display_string)
    }
}

pub enum EdgeWeightTypes<S> {
    ReturnI32(fn(&HashSet<NodeIndex, S>, &HashSet<NodeIndex, S>) -> i32),
    ReturnI32Tuple(fn(&HashSet<NodeIndex, S>, &HashSet<NodeIndex, S>) -> (i32, i32)),
}

use std::{collections::HashSet, error::Error, fs::File, hash::BuildHasher, io::Write};

pub fn heuristic_to_spanning_tree_computation_type_and_edge_weight_heuristic<
    S: BuildHasher + Default,
>(
    heuristic: &HeuristicTypes,
) -> Option<(
    treewidth_heuristic_using_clique_graphs::SpanningTreeConstructionMethod,
    EdgeWeightTypes<S>,
)> {
    use treewidth_heuristic_using_clique_graphs::SpanningTreeConstructionMethod::*;
    use treewidth_heuristic_using_clique_graphs::*;
    use EdgeWeightTypes::*;
    match heuristic {
        MSTreIUnion => Some((MSTAndUseTreeStructure, ReturnI32(union))),
        MSTreIDisjU => Some((MSTAndUseTreeStructure, ReturnI32(disjoint_union))),
        MSTreINegIn => Some((MSTAndUseTreeStructure, ReturnI32(negative_intersection))),
        FilWhINegIn => Some((FillWhilstMST, ReturnI32(negative_intersection))),
        MSTreILeDif => Some((MSTAndUseTreeStructure, ReturnI32(least_difference))),
        FilWhILeDif => Some((FillWhilstMST, ReturnI32(least_difference))),
        MSTreILdTNi => Some((
            MSTAndUseTreeStructure,
            EdgeWeightTypes::ReturnI32Tuple(least_difference_then_negative_intersection),
        )),
        FilWhILdTNi => Some((
            FillWhilstMST,
            EdgeWeightTypes::ReturnI32Tuple(least_difference_then_negative_intersection),
        )),
        MSTreINiTLd => Some((
            MSTAndUseTreeStructure,
            EdgeWeightTypes::ReturnI32Tuple(negative_intersection_then_least_difference),
        )),
        FilWhINiTLd => Some((
            FillWhilstMST,
            EdgeWeightTypes::ReturnI32Tuple(negative_intersection_then_least_difference),
        )),
        FWhUEINiTLd => Some((
            FillWhilstMSTEdgeUpdate,
            EdgeWeightTypes::ReturnI32Tuple(negative_intersection_then_least_difference),
        )),
        FiWhTINiTLd => Some((
            FillWhilstMSTTree,
            EdgeWeightTypes::ReturnI32Tuple(negative_intersection_then_least_difference),
        )),
        FWBagINonee => Some((
            FillWhilstMSTBagSize,
            EdgeWeightTypes::ReturnI32Tuple(negative_intersection_then_least_difference),
        )),
        MSTreINiTLdIBC(_) => Some((
            MSTAndUseTreeStructure,
            EdgeWeightTypes::ReturnI32Tuple(negative_intersection_then_least_difference),
        )),
        FilWhINiTLdIBC(_) => Some((
            FillWhilstMST,
            EdgeWeightTypes::ReturnI32Tuple(negative_intersection_then_least_difference),
        )),
        FiWhTINiTLdIBC(_) => Some((
            FillWhilstMSTTree,
            EdgeWeightTypes::ReturnI32Tuple(negative_intersection_then_least_difference),
        )),
        GreedyDegreeFillIn => None,
        MSTreIConst => Some((MSTAndUseTreeStructure, ReturnI32(constant))),
        MSTreIRandd => Some((MSTAndUseTreeStructure, ReturnI32(random))),
    }
}

pub fn heuristic_to_clique_bound(heuristic: &HeuristicTypes) -> Option<usize> {
    match heuristic {
        MSTreIUnion => None,
        MSTreIDisjU => None,
        MSTreINegIn => None,
        FilWhINegIn => None,
        MSTreILeDif => None,
        FilWhILeDif => None,
        MSTreILdTNi => None,
        FilWhILdTNi => None,
        MSTreINiTLd => None,
        FilWhINiTLd => None,
        FWhUEINiTLd => None,
        FiWhTINiTLd => None,
        FWBagINonee => None,
        MSTreINiTLdIBC(clique_bound) => Some(*clique_bound),
        FilWhINiTLdIBC(clique_bound) => Some(*clique_bound),
        FiWhTINiTLdIBC(clique_bound) => Some(*clique_bound),
        GreedyDegreeFillIn => None,
        MSTreIConst => None,
        MSTreIRandd => None,
    }
}

/// Per Run Bound&Runtime Data Header should be of the form
/// Graph HeuristicVersion1 HeuristicVersion1 ... HeuristicVersion1 (number_of_runs_per_graph often) HeuristicVersion2 ...
///
/// Per Run Bound&Runtime Data should be of the form
/// Graph name HeuristicVersion1ComputedBoundFirstRun HeuristicVersion1ComputedBoundSecondRun ... HeuristicVersion1ComputedBoundNumber_of_runs_per_graph-thRun HeuristicVersion2 ....
pub fn write_to_csv(
    per_run_bound_data: &mut Vec<String>,
    per_run_runtime_data: &mut Vec<String>,
    average_bound_writer: &mut Writer<File>,
    per_run_bound_writer: &mut Writer<File>,
    average_runtime_writer: &mut Writer<File>,
    per_run_runtime_writer: &mut Writer<File>,
    number_of_runs_per_graph: usize,
    header: bool,
) -> Result<(), Box<dyn Error>> {
    if header {
        let mut average_bound_data: Vec<_> = Vec::new();
        let mut average_runtime_data: Vec<_> = Vec::new();
        let mut offset_counter = 1;

        debug!(
            "Per Run Bound Data Header length: {:?}",
            per_run_bound_data.len()
        );

        for i in 0..per_run_bound_data.len() {
            if i == 0 || i == 1 {
                average_bound_data.push(
                    per_run_bound_data
                        .get(i)
                        .expect("Index should be in bound by loop invariant")
                        .to_owned(),
                );
                average_runtime_data.push(
                    per_run_runtime_data
                        .get(i)
                        .expect("Index should be in bound by loop invariant")
                        .to_owned(),
                );
            } else {
                if offset_counter == number_of_runs_per_graph {
                    average_bound_data.push(
                        per_run_bound_data
                            .get(i)
                            .expect("Index should be in bound by loop invariant")
                            .to_owned(),
                    );
                    average_runtime_data.push(
                        per_run_runtime_data
                            .get(i)
                            .expect("Index should be in bound by loop invariant")
                            .to_owned(),
                    );
                    offset_counter = 1;
                } else {
                    offset_counter += 1;
                }
            }
        }
        debug!("Writing to writers");
        debug!("Trying to write: {:?}", average_bound_data);
        per_run_bound_writer.write_record(per_run_bound_data)?;
        per_run_runtime_writer.write_record(per_run_runtime_data)?;

        average_bound_writer.write_record(average_bound_data)?;
        average_runtime_writer.write_record(average_runtime_data)?;
    } else {
        let mut average_bound_data: Vec<f64> = Vec::new();
        let mut average_bound_header: Vec<String> = Vec::new();
        let mut average_runtime_data: Vec<f64> = Vec::new();
        let mut average_runtime_header: Vec<String> = Vec::new();
        let mut offset_counter = 1;
        let mut average_runtime: f64 = 0.0;
        let mut average_bound: f64 = f64::MAX;
        if number_of_runs_per_graph != 1 {
            for i in 0..per_run_bound_data.len() {
                if i == 0 || i == 1 {
                    average_bound_header.push(
                        per_run_bound_data
                            .get(i)
                            .expect("Index should be in bound by loop invariant")
                            .to_owned(),
                    );
                    average_runtime_header.push(
                        per_run_runtime_data
                            .get(i)
                            .expect("Index should be in bound by loop invariant")
                            .to_owned(),
                    );
                } else {
                    if offset_counter == number_of_runs_per_graph {
                        average_bound = average_bound.min(
                            per_run_bound_data
                                .get(i)
                                .expect("Index should be in bound by loop invariant")
                                .parse::<f64>()
                                .expect("Entries of data vectors should be valid f64"),
                        );
                        average_runtime += per_run_runtime_data
                            .get(i)
                            .expect("Index should be in bound by loop invariant")
                            .parse::<f64>()
                            .expect("Entries of data vectors should be valid f64");

                        average_runtime /= number_of_runs_per_graph as f64;

                        average_bound_data.push(average_bound);
                        average_runtime_data.push(average_runtime);

                        // Reset average bound and runtime and counter
                        average_bound = f64::MAX;
                        average_runtime = 0.0;
                        offset_counter = 1;
                    } else {
                        average_bound = average_bound.min(
                            per_run_bound_data
                                .get(i)
                                .expect("Index should be in bound by loop invariant")
                                .parse::<f64>()
                                .expect("Entries of data vectors should be valid f64"),
                        );
                        average_runtime += per_run_runtime_data
                            .get(i)
                            .expect("Index should be in bound by loop invariant")
                            .parse::<f64>()
                            .expect("Entries of data vectors should be valid f64");
                        offset_counter += 1;
                        // if i == per_run_bound_data.len() - 1 {
                        //     average_bound_data.push(average_bound);
                        //     average_runtime_data.push(average_runtime);
                        // }
                    }
                }
            }
        } else {
            for i in 0..per_run_bound_data.len() {
                if i == 0 || i == 1 {
                    average_bound_header.push(
                        per_run_bound_data
                            .get(i)
                            .expect("Index should be in bound by loop invariant")
                            .to_owned(),
                    );
                    average_runtime_header.push(
                        per_run_runtime_data
                            .get(i)
                            .expect("Index should be in bound by loop invariant")
                            .to_owned(),
                    );
                } else {
                    average_bound_data.push(
                        per_run_bound_data
                            .get(i)
                            .expect("Index should be in bound by loop invariant")
                            .to_owned()
                            .parse::<f64>()
                            .expect("Entry should be valid f64"),
                    );
                    average_runtime_data.push(
                        per_run_runtime_data
                            .get(i)
                            .expect("Index should be in bound by loop invariant")
                            .to_owned()
                            .parse::<f64>()
                            .expect("Entry should be valid f64"),
                    );
                }
            }
        }

        let average_bound_data: Vec<String> = average_bound_header
            .into_iter()
            .chain(average_bound_data.iter().map(|u| u.to_string()))
            .collect();
        let average_runtime_data: Vec<_> = average_runtime_header
            .into_iter()
            .chain(average_runtime_data.iter().map(|u| u.to_string()))
            .collect();

        per_run_bound_writer.write_record(per_run_bound_data)?;
        per_run_runtime_writer.write_record(per_run_runtime_data)?;

        average_bound_writer.write_record(average_bound_data)?;
        average_runtime_writer.write_record(average_runtime_data)?;
    }
    per_run_bound_writer.flush()?;
    per_run_runtime_writer.flush()?;
    average_bound_writer.flush()?;
    average_runtime_writer.flush()?;

    Ok(())
}

pub fn current_time() -> String {
    let tz_offset = chrono::FixedOffset::east_opt(2 * 3600).unwrap();
    chrono::Local::from_offset(&tz_offset)
        .from_utc_datetime(&chrono::Local::now().to_utc().naive_utc())
        .to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
        .to_string()
}

// Converting dot files to pdf in bulk:
// FullPath -type f -name "*.dot" | xargs dot -Tpdf -O
#[allow(dead_code)]
pub fn create_dot_files<O: std::fmt::Debug, S>(
    clique_graph_tree_after_filling_up: &petgraph::Graph<
        HashSet<NodeIndex, S>,
        O,
        petgraph::prelude::Undirected,
    >,
    name: &str,
) {
    std::fs::create_dir_all("visualizations")
        .expect("Could not create directory for visualizations");

    let result_graph_dot_file = petgraph::dot::Dot::with_config(
        clique_graph_tree_after_filling_up,
        &[petgraph::dot::Config::EdgeNoLabel],
    );

    let mut w = std::fs::File::create(format!("visualizations/result_graph_{}.dot", name))
        .expect("Result graph file could not be created");
    write!(&mut w, "{:?}", result_graph_dot_file)
        .expect("Unable to write dotfile for result graph to files");
}
