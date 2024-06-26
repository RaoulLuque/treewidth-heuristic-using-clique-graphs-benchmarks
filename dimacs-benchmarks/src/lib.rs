use treewidth_heuristic_using_clique_graphs::TreewidthComputationMethod::*;

#[derive(Debug)]
pub enum HeuristicTypes {
    // MstTree = Minimum spanning tree then fill using tree structure
    // FillWhile = Fill while building minimum spanning tree
    // Ni = Negative Intersection
    MstTreeNi,

    FillWhileNi,
    // Ld = Least difference
    MstTreeLd,

    FillWhileLd,
    // T = Then
    MstTreeNiTLd,
    FillWhileNiTLd,
    FillWhileUpNiTLd,
    FillWhileTreeNiTLd,

    MstTreeLdTNi,
    FillWhileLdTNi,
    // Bag = See [TreewidthComputationMethod::FillWhilstMSTBagSize]
    FillWhileBag,
    // BC = Bounded cliques
    MstTreeNiTLdBC(usize),
    FillWhileNiTLdBC(usize),
    FillWhileTreeNiTLdBC(usize),
}

impl std::fmt::Display for HeuristicTypes {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let display_string = match self {
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

use std::{collections::HashSet, hash::BuildHasher};

use petgraph::graph::NodeIndex;
use HeuristicTypes::*;

pub const HEURISTICS_BEING_TESTED: [HeuristicTypes; 8] = [
    MstTreeNiTLd,
    MstTreeNiTLdBC(2),
    MstTreeNiTLdBC(3),
    MstTreeNiTLdBC(10),
    FillWhileNiTLd,
    FillWhileNiTLdBC(2),
    FillWhileNiTLdBC(3),
    FillWhileNiTLdBC(10),
];

pub fn heuristic_to_edge_weight_heuristic<S: BuildHasher + Default>(
    heuristic: &HeuristicTypes,
) -> EdgeWeightTypes<S> {
    use treewidth_heuristic_using_clique_graphs::*;
    use HeuristicTypes::*;
    match heuristic {
        MstTreeNi => EdgeWeightTypes::ReturnI32(negative_intersection),
        FillWhileNi => EdgeWeightTypes::ReturnI32(negative_intersection),
        MstTreeLd => EdgeWeightTypes::ReturnI32(least_difference),
        FillWhileLd => EdgeWeightTypes::ReturnI32(least_difference),
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
) -> treewidth_heuristic_using_clique_graphs::TreewidthComputationMethod {
    match heuristic {
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
