// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Splitting deep circuit into major stages at global level indices.
//!
//! This is crucial in efficiently handling large and deep circuits
//! with a limited processing element width.

use indexmap::IndexSet;
use crate::aig::{AIG, EndpointGroup, DriverType};

/// A struct representing the boundaries of a staged AIG.
pub struct StagedAIG {
    /// the staged primary inputs from previous levels.
    pub primary_inputs: Option<IndexSet<usize>>,
    /// the staged primary output pins for next levels.
    ///
    /// these pins are active nodes at the level split.
    pub primary_output_pins: Vec<usize>,
    /// the endpoint indices of original AIG fulfilled by current level.
    pub endpoints: Vec<usize>,
}

impl StagedAIG {
    /// Get the number of endpoint groups that should be fulfilled
    /// with this staged AIG.
    ///
    /// This mimics the interface given by a raw AIG.
    pub fn num_endpoint_groups(&self) -> usize {
        self.primary_output_pins.len() + self.endpoints.len()
    }

    /// Get the virtual endpoint group with an index.
    ///
    /// This mimics the interface given by a raw AIG.
    pub fn get_endpoint_group<'aig>(&self, aig: &'aig AIG, endpt_id: usize) -> EndpointGroup<'aig> {
        if endpt_id < self.primary_output_pins.len() {
            EndpointGroup::StagedIOPin(self.primary_output_pins[endpt_id])
        }
        else {
            aig.get_endpoint_group(self.endpoints[endpt_id - self.primary_output_pins.len()])
        }
    }

    /// build a staged AIG that consists of all levels.
    pub fn from_full_aig(aig: &AIG) -> Self {
        StagedAIG {
            primary_inputs: None,
            primary_output_pins: vec![],
            endpoints: (0..aig.num_endpoint_groups()).collect()
        }
    }

    /// build a staged AIG by horizontal splitting given a subset
    /// of endpoints.
    ///
    /// return built StagedAIG.
    /// the endpoints are given as a slice of endpoint group indices,
    /// that must have all staged primary output groups at the front
    /// and original endpoints following. otherwise we panic.
    ///
    /// the result guarantees that the endpoint `i` corresponds to
    /// the original staged's endpoint `endpoint_subset[i]`.
    pub fn to_endpoint_subset(
        &self,
        endpoint_subset: &[usize]
    ) -> StagedAIG {
        let mut staged_sub = StagedAIG {
            primary_inputs: self.primary_inputs.clone(),
            primary_output_pins: vec![],
            endpoints: vec![],
        };
        for &endpt_i in endpoint_subset {
            if endpt_i < self.primary_output_pins.len() {
                staged_sub.primary_output_pins.push(
                    self.primary_output_pins[endpt_i]
                );
                assert!(staged_sub.endpoints.is_empty(),
                        "endpoint subset must be in order!");
            }
            else {
                staged_sub.endpoints.push(
                    self.endpoints[endpt_i - self.primary_output_pins.len()]
                );
            }
        }
        staged_sub
    }

    /// build a staged AIG by vertical splitting at the given level id.
    ///
    /// return built StagedAIG.
    /// the active middle nodes at split can be obtained from the
    /// StagedAIG::primary_output_pins.
    /// if this is empty, it means all endpoints are already satisfied
    /// after this stage.
    pub fn from_split(
        aig: &AIG,
        unrealized_orig_endpoints: &IndexSet<usize>,
        primary_inputs: Option<&IndexSet<usize>>,
        split_at_level: usize,
    ) -> Self {
        let mut unrealized_endpoint_nodes = Vec::new();
        for &endpt in unrealized_orig_endpoints {
            aig.get_endpoint_group(endpt).for_each_input(|i| {
                unrealized_endpoint_nodes.push(i);
            });
        }
        assert!(!unrealized_endpoint_nodes.is_empty());
        let order = aig.topo_traverse_generic(
            Some(&unrealized_endpoint_nodes),
            primary_inputs
        );
        let mut num_fanouts = vec![0; aig.num_aigpins + 1];
        let mut level_id = vec![0; aig.num_aigpins + 1];
        for &i in &order {
            if matches!(primary_inputs, Some(pi) if pi.contains(&i)) {
                continue
            }
            if let DriverType::AndGate(a, b) = aig.drivers[i] {
                if a >= 2 {
                    num_fanouts[a >> 1] += 1;
                    level_id[i] = level_id[i].max(level_id[a >> 1] + 1);
                }
                if b >= 2 {
                    num_fanouts[b >> 1] += 1;
                    level_id[i] = level_id[i].max(level_id[b >> 1] + 1);
                }
            }
        }
        let mut endpt_level_id = vec![0; aig.num_endpoint_groups()];
        for &endpt in unrealized_orig_endpoints {
            aig.get_endpoint_group(endpt).for_each_input(|i| {
                num_fanouts[i] += 1;
                endpt_level_id[endpt] = endpt_level_id[endpt].max(level_id[i]);
            });
        }
        let mut nodes_at_split = IndexSet::new();
        for &i in &order {
            if level_id[i] > split_at_level { continue }
            nodes_at_split.insert(i);
            if matches!(primary_inputs, Some(pi) if pi.contains(&i)) {
                continue
            }
            if let DriverType::AndGate(a, b) = aig.drivers[i] {
                if a >= 2 {
                    num_fanouts[a >> 1] -= 1;
                    if num_fanouts[a >> 1] == 0 {
                        assert!(nodes_at_split.swap_remove(&(a >> 1)));
                    }
                }
                if b >= 2 {
                    num_fanouts[b >> 1] -= 1;
                    if num_fanouts[b >> 1] == 0 {
                        assert!(nodes_at_split.swap_remove(&(b >> 1)));
                    }
                }
            }
        }
        let mut endpoints_before_split = Vec::new();
        for &endpt in unrealized_orig_endpoints {
            if endpt_level_id[endpt] > split_at_level { continue }
            endpoints_before_split.push(endpt);
            aig.get_endpoint_group(endpt).for_each_input(|i| {
                num_fanouts[i] -= 1;
                if num_fanouts[i] == 0 {
                    assert!(nodes_at_split.swap_remove(&i));
                }
            });
        }

        StagedAIG {
            primary_inputs: primary_inputs.cloned(),
            primary_output_pins: nodes_at_split.iter().copied()
                .filter(|po| !matches!(primary_inputs, Some(pi) if pi.contains(po)))
                .collect(),
            endpoints: endpoints_before_split
        }
    }
}

/// Given the level split points, return a list of split stages.
///
/// For example, given [10, 20], will return a list like this:
/// [(0, 10, stage0_10), (10, 20, stage10_20), (20, MAX, stage20_MAX)]
///
/// If the netlist ends early before all split points, the length might be
/// shorter than expected.
pub fn build_staged_aigs(
    aig: &AIG, level_split: &[usize]
) -> Vec<(usize, usize, StagedAIG)> {
    let mut ret = Vec::new();
    let mut unrealized_orig_endpoints = (0..aig.num_endpoint_groups()).collect::<IndexSet<_>>();
    let mut primary_inputs: Option<IndexSet<usize>> = None;

    for i in 0..level_split.len() {
        let cur_split = level_split[i];
        let last_split = match i {
            0 => 0,
            i @ _ => level_split[i - 1]
        };
        let staged = StagedAIG::from_split(
            aig, &unrealized_orig_endpoints, primary_inputs.as_ref(),
            cur_split - last_split
        );
        for &endpt in &staged.endpoints {
            assert!(unrealized_orig_endpoints.swap_remove(&endpt));
        }
        let primary_inputs = primary_inputs.get_or_insert_with(|| Default::default());
        for &inp in &staged.primary_output_pins {
            primary_inputs.insert(inp);
        }
        if staged.primary_output_pins.is_empty() {
            ret.push((last_split, usize::MAX, staged));
            return ret
        }
        ret.push((last_split, cur_split, staged));
    }

    let last_split = match level_split.len() {
        0 => 0,
        i @ _ => level_split[i - 1]
    };
    ret.push((last_split, usize::MAX, StagedAIG::from_split(
        aig, &unrealized_orig_endpoints, primary_inputs.as_ref(),
        usize::MAX
    )));

    ret
}
