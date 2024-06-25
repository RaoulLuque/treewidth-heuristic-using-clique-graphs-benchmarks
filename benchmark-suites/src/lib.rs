#[derive(Debug, Copy, Clone)]
pub enum HeuristicTypes {
    // For comparison of edge weights:
    MstTreeLd,
    MstTreeNi,
    MstTreeUn,
    MstTreeDU,

    // For comparison of combined edge weights:
    MstTreeNiTLd,
    MstTreeLdTNi,

    // For comparison of spanning tree construction:
    FillWhileNiTLd,
    FillWhileUpNiTLd,
    FillWhileBag,

    // For comparison with bounded clique
    FillWhileNiTLdBC(usize),

    // Old / Not used
    FillWhileNi,
    FillWhileLd,
    FillWhileTreeNiTLd,
    FillWhileLdTNi,
    MstTreeNiTLdBC(usize),
    FillWhileTreeNiTLdBC(usize),
}

use csv::Writer;
use petgraph::graph::NodeIndex;
use HeuristicTypes::*;

pub const TEST_SUITE: [(fn() -> Vec<HeuristicTypes>, &str); 3] = [
    (comparison_of_edge_weights, "comparison_of_edge_weights"),
    (
        comparison_of_combined_edge_weights,
        "comparison_of_combined_edge_weights",
    ),
    (
        comparison_of_spanning_tree_construction,
        "comparison_of_spanning_tree_construction",
    ),
];

pub fn comparison_of_edge_weights() -> Vec<HeuristicTypes> {
    vec![MstTreeLd, MstTreeNi, MstTreeUn, MstTreeDU]
}

pub fn comparison_of_combined_edge_weights() -> Vec<HeuristicTypes> {
    vec![MstTreeNi, MstTreeNiTLd, MstTreeLdTNi]
}

pub fn comparison_of_spanning_tree_construction() -> Vec<HeuristicTypes> {
    vec![MstTreeNiTLd, FillWhileNiTLd, FillWhileUpNiTLd, FillWhileBag]
}

impl std::fmt::Display for HeuristicTypes {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let display_string = match self {
            MstTreeUn => "MTrUn".to_string(),
            MstTreeDU => "MTrDU".to_string(),
            MstTreeNi => "MTrNi".to_string(),
            FillWhileNi => "FiWhNi".to_string(),
            MstTreeLd => "MTrLd".to_string(),
            FillWhileLd => "FiWhLd".to_string(),
            MstTreeNiTLd => "MTrNiTLd".to_string(),
            FillWhileNiTLd => "FiWhNiTLd".to_string(),
            FillWhileUpNiTLd => "FWUNiTLd".to_string(),
            MstTreeLdTNi => "MTrLdTNi".to_string(),
            FillWhileLdTNi => "FiWhLdTNi".to_string(),
            FillWhileTreeNiTLd => "FWTNiTLd".to_string(),
            FillWhileBag => "FWB".to_string(),
            MstTreeNiTLdBC(clique_bound) => format!("MTrNiTLdBC {}", clique_bound),
            FillWhileNiTLdBC(clique_bound) => format!("FiWhLdTNiBC {}", clique_bound),
            FillWhileTreeNiTLdBC(clique_bound) => format!("FWTNiTLd {}", clique_bound),
        };
        write!(f, "{}", display_string)
    }
}

pub enum EdgeWeightTypes<S> {
    ReturnI32(fn(&HashSet<NodeIndex, S>, &HashSet<NodeIndex, S>) -> i32),
    ReturnI32Tuple(fn(&HashSet<NodeIndex, S>, &HashSet<NodeIndex, S>) -> (i32, i32)),
}

use std::{collections::HashSet, error::Error, fs::File, hash::BuildHasher, io};

pub fn heuristic_to_edge_weight_heuristic<S: BuildHasher + Default>(
    heuristic: &HeuristicTypes,
) -> EdgeWeightTypes<S> {
    use treewidth_heuristic_clique_graph::*;
    use EdgeWeightTypes::*;
    match heuristic {
        MstTreeUn => ReturnI32(union),
        MstTreeDU => ReturnI32(disjoint_union),
        MstTreeNi => ReturnI32(negative_intersection),
        FillWhileNi => ReturnI32(negative_intersection),
        MstTreeLd => ReturnI32(least_difference),
        FillWhileLd => ReturnI32(least_difference),
        MstTreeLdTNi => {
            EdgeWeightTypes::ReturnI32Tuple(least_difference_then_negative_intersection)
        }
        FillWhileLdTNi => {
            EdgeWeightTypes::ReturnI32Tuple(least_difference_then_negative_intersection)
        }
        MstTreeNiTLd => {
            EdgeWeightTypes::ReturnI32Tuple(negative_intersection_then_least_difference)
        }
        FillWhileNiTLd => {
            EdgeWeightTypes::ReturnI32Tuple(negative_intersection_then_least_difference)
        }
        FillWhileUpNiTLd => {
            EdgeWeightTypes::ReturnI32Tuple(negative_intersection_then_least_difference)
        }
        FillWhileTreeNiTLd => {
            EdgeWeightTypes::ReturnI32Tuple(negative_intersection_then_least_difference)
        }
        FillWhileBag => {
            EdgeWeightTypes::ReturnI32Tuple(negative_intersection_then_least_difference)
        }
        MstTreeNiTLdBC(_) => {
            EdgeWeightTypes::ReturnI32Tuple(negative_intersection_then_least_difference)
        }
        FillWhileNiTLdBC(_) => {
            EdgeWeightTypes::ReturnI32Tuple(negative_intersection_then_least_difference)
        }
        FillWhileTreeNiTLdBC(_) => {
            EdgeWeightTypes::ReturnI32Tuple(negative_intersection_then_least_difference)
        }
    }
}

pub fn heuristic_to_computation_type(
    heuristic: &HeuristicTypes,
) -> treewidth_heuristic_clique_graph::TreewidthComputationMethod {
    use treewidth_heuristic_clique_graph::TreewidthComputationMethod::*;
    match heuristic {
        MstTreeUn => MSTAndUseTreeStructure,
        MstTreeDU => MSTAndUseTreeStructure,
        MstTreeNi => MSTAndUseTreeStructure,
        FillWhileNi => FillWhilstMST,
        MstTreeLd => MSTAndUseTreeStructure,
        FillWhileLd => FillWhilstMST,
        MstTreeLdTNi => MSTAndUseTreeStructure,
        FillWhileLdTNi => FillWhilstMST,
        MstTreeNiTLd => MSTAndUseTreeStructure,
        FillWhileNiTLd => FillWhilstMST,
        FillWhileUpNiTLd => FillWhilstMSTEdgeUpdate,
        FillWhileTreeNiTLd => FillWhilstMSTTree,
        FillWhileBag => FillWhilstMSTBagSize,
        MstTreeNiTLdBC(_) => MSTAndUseTreeStructure,
        FillWhileNiTLdBC(_) => FillWhilstMST,
        FillWhileTreeNiTLdBC(_) => FillWhilstMSTTree,
    }
}

pub fn heuristic_to_clique_bound(heuristic: &HeuristicTypes) -> Option<usize> {
    match heuristic {
        MstTreeUn => None,
        MstTreeDU => None,
        MstTreeNi => None,
        FillWhileNi => None,
        MstTreeLd => None,
        FillWhileLd => None,
        MstTreeLdTNi => None,
        FillWhileLdTNi => None,
        MstTreeNiTLd => None,
        FillWhileNiTLd => None,
        FillWhileUpNiTLd => None,
        FillWhileTreeNiTLd => None,
        FillWhileBag => None,
        MstTreeNiTLdBC(clique_bound) => Some(*clique_bound),
        FillWhileNiTLdBC(clique_bound) => Some(*clique_bound),
        FillWhileTreeNiTLdBC(clique_bound) => Some(*clique_bound),
    }
}

/// Per Run Bound&Runtime Data should be of the form
/// Graphname HeuristicVersion1ComputedBoundFirstRun HeuristicVersion1ComputedBoundSecondRun ... HeuristicVersion1ComputedBoundNumber_of_runs_per_graph-thRun HeuristicVersion2 ....
pub fn write_to_csv(
    per_run_bound_data: Vec<usize>,
    per_run_runtime_data: Vec<usize>,
    average_bound_writer: &mut Writer<File>,
    per_run_bound_writer: &mut Writer<File>,
    average_runtime_writer: &mut Writer<File>,
    per_run_runtime_writer: &mut Writer<File>,
    number_of_runs_per_graph: usize,
) -> Result<(), Box<dyn Error>> {
    let mut average_bound_data: Vec<usize> = Vec::new();
    let mut average_runtime_data: Vec<usize> = Vec::new();
    let mut offset_counter = 0;
    let mut average_runtime = 0;
    let mut average_bound = 0;

    for i in 0..per_run_bound_data.len() {
        if i == 0 {
            average_bound_data.push(
                *per_run_bound_data
                    .get(i)
                    .expect("Index should be in bound by loop invariant"),
            );
            average_runtime_data.push(
                *per_run_runtime_data
                    .get(i)
                    .expect("Index should be in bound by loop invariant"),
            );
        } else {
            if offset_counter == number_of_runs_per_graph {
                average_bound_data.push(average_bound);
                average_runtime_data.push(average_runtime);
                average_bound = *per_run_bound_data
                    .get(i)
                    .expect("Index should be in bound by loop invariant");
                average_runtime = *per_run_runtime_data
                    .get(i)
                    .expect("Index should be in bound by loop invariant");

                offset_counter = 1;
            } else {
                average_bound += per_run_bound_data
                    .get(i)
                    .expect("Index should be in bound by loop invariant");
                average_runtime += per_run_runtime_data
                    .get(i)
                    .expect("Index should be in bound by loop invariant");
                offset_counter += 1;

                if i == per_run_bound_data.len() - 1 {
                    average_bound_data.push(average_bound);
                    average_runtime_data.push(average_runtime);
                }
            }
        }
    }

    let per_run_bound_data: Vec<_> = per_run_bound_data.iter().map(|u| u.to_string()).collect();
    let per_run_runtime_data: Vec<_> = per_run_runtime_data.iter().map(|u| u.to_string()).collect();

    let average_bound_data: Vec<_> = average_bound_data.iter().map(|u| u.to_string()).collect();
    let average_runtime_data: Vec<_> = average_runtime_data.iter().map(|u| u.to_string()).collect();

    per_run_bound_writer.write_record(per_run_bound_data)?;
    per_run_runtime_writer.write_record(per_run_runtime_data)?;

    average_bound_writer.write_record(average_bound_data)?;
    average_runtime_writer.write_record(average_runtime_data)?;

    Ok(())
}

/// Per Run Bound&Runtime Data Header should be of the form
/// Graph HeuristicVersion1 HeuristicVersion1 ... HeuristicVersion1 (number_of_runs_per_graph often) HeuristicVersion2 ...
pub fn write_header_to_csv(
    per_run_bound_data: &mut Vec<String>,
    per_run_runtime_data: &mut Vec<String>,
    average_bound_writer: &mut Writer<File>,
    per_run_bound_writer: &mut Writer<File>,
    average_runtime_writer: &mut Writer<File>,
    per_run_runtime_writer: &mut Writer<File>,
    number_of_runs_per_graph: usize,
) -> Result<(), Box<dyn Error>> {
    let mut average_bound_data_header: Vec<_> = Vec::new();
    let mut average_runtime_data_header: Vec<_> = Vec::new();
    let mut offset_counter = 0;

    for i in 0..per_run_bound_data.len() {
        if i == 0 {
            average_bound_data_header.push(
                per_run_bound_data
                    .get(i)
                    .expect("Index should be in bound by loop invariant")
                    .to_owned(),
            );
            average_runtime_data_header.push(
                per_run_runtime_data
                    .get(i)
                    .expect("Index should be in bound by loop invariant")
                    .to_owned(),
            );
        } else {
            if offset_counter == number_of_runs_per_graph {
                average_bound_data_header.push(
                    per_run_bound_data
                        .get(i)
                        .expect("Index should be in bound by loop invariant")
                        .to_owned(),
                );
                average_runtime_data_header.push(
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

    per_run_bound_writer.write_record(per_run_bound_data)?;
    per_run_runtime_writer.write_record(per_run_runtime_data)?;

    average_bound_writer.write_record(average_bound_data_header)?;
    average_runtime_writer.write_record(average_runtime_data_header)?;

    Ok(())
}
