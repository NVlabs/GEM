// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Partition executor

use crate::aig::{DriverType, AIG, EndpointGroup};
use crate::staging::StagedAIG;
use indexmap::{IndexMap, IndexSet};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::PathBuf;
use rayon::prelude::*;

/// The number of boomerang stages.
///
/// This determines the shuffle width, i.e., kernel width.
/// `kernel width = (1 << BOOMERANG_NUM_STAGES)`.
pub const BOOMERANG_NUM_STAGES: usize = 13;

const BOOMERANG_MAX_WRITEOUTS: usize = 1 << (BOOMERANG_NUM_STAGES - 5);

/// One Boomerang stage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoomerangStage {
    /// the boomerang hierarchy, 8192 -> 4096 -> ... -> 1.
    ///
    /// each element is an aigpin index (without iv).
    /// its parent indices should either be a passthrough or an
    /// and gate mapping.
    pub hier: Vec<Vec<usize>>,
    /// the 32-packed elements in the hierarchy where there should be
    /// a pass-through.
    pub write_outs: Vec<usize>,
}

/// One partitioned block: a basic execution unit on GPU.
///
/// A block is mapped to a GPU block with the following resource
/// constraints:
/// 1. the number of unique inputs should not exceed 8191.
/// 2. the number of unique outputs should not exceed 8191.
///    for srams and dffs, outputs include all enable pins and bus pins.
///    there might be unusable holes but the effective capacity is at least
///    4095.
/// 3. the number of intermediate pins alive at each stage should not
 ///    exceed 4095.
/// 4. the number of SRAM output groups should not exceed 64.
///    64 = 8192 / (32 * 4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Partition {
    /// the endpoints that are realized by this partition.
    pub endpoints: Vec<usize>,
    /// the boomerang stages.
    ///
    /// between stages there will automatically be shuffles.
    pub stages: Vec<BoomerangStage>,
}

/// build a single boomerang stage given the current inputs and
/// outputs.
fn build_one_boomerang_stage(
    aig: &AIG,
    unrealized_comb_outputs: &mut IndexSet<usize>,
    realized_inputs: &mut IndexSet<usize>,
    total_write_outs: &mut usize,
    num_reserved_writeouts: usize,
) -> Option<BoomerangStage> {
    let mut hier = Vec::new();
    for i in 0..=BOOMERANG_NUM_STAGES {
        hier.push(vec![usize::MAX; 1 << (BOOMERANG_NUM_STAGES - i)]);
    }

    // first discover the (remaining) subgraph to implement.
    let order = aig.topo_traverse_generic(
        Some(
            &unrealized_comb_outputs.iter().copied().collect()
        ),
        Some(&realized_inputs)
    );
    let id2order: IndexMap<_, _> = order.iter().copied().enumerate()
        .map(|(order_i, i)| (i, order_i))
        .collect();
    let mut level = vec![0; order.len()];
    for (order_i, i) in order.iter().copied().enumerate() {
        if realized_inputs.contains(&i) { continue }
        let mut lvli: usize = 0;
        if let DriverType::AndGate(a, b) = aig.drivers[i] {
            if a >= 2 {
                lvli = lvli.max(level[*id2order.get(&(a >> 1)).unwrap()] + 1);
            }
            if b >= 2 {
                lvli = lvli.max(level[*id2order.get(&(b >> 1)).unwrap()] + 1);
            }
        }
        level[order_i] = lvli;
    }
    let max_level = level.iter().copied().max().unwrap();
    clilog::trace!("boomerang current max level: {}", max_level);

    fn place_bit(
        aig: &AIG,
        hier: &mut Vec<Vec<usize>>,
        hier_visited_nodes_count: &mut IndexMap<usize, usize>,
        level: &Vec<usize>,
        id2order: &IndexMap<usize, usize>,
        hi: usize, j: usize, nd: usize
    ) {
        hier[hi][j] = nd;
        if hi == 0 { return }
        *hier_visited_nodes_count.entry(nd).or_default() += 1;
        let lvlnd = level[*id2order.get(&nd).unwrap()];
        assert!(lvlnd <= hi);
        if lvlnd != hi {
            place_bit(aig, hier, hier_visited_nodes_count,
                      level, id2order,
                      hi - 1, j, nd);
        }
        else {
            let (a, b) = match aig.drivers[nd] {
                DriverType::AndGate(a, b) => (a, b),
                _ => panic!()
            };
            let hier_hi_len = hier[hi].len();
            place_bit(aig, hier, hier_visited_nodes_count,
                      level, id2order,
                      hi - 1, j, a >> 1);
            place_bit(aig, hier, hier_visited_nodes_count,
                      level, id2order,
                      hi - 1, j + hier_hi_len, b >> 1);
        }
    }

    fn purge_bit(
        aig: &AIG,
        hier: &mut Vec<Vec<usize>>,
        hier_visited_nodes_count: &mut IndexMap<usize, usize>,
        level: &Vec<usize>,
        id2order: &IndexMap<usize, usize>,
        hi: usize, j: usize
    ) {
        if hier[hi][j] == usize::MAX { return }
        let nd = hier[hi][j];
        hier[hi][j] = usize::MAX;
        if hi == 0 { return }
        let hvc = hier_visited_nodes_count.get_mut(&nd).unwrap();
        *hvc -= 1;
        if *hvc == 0 {
            hier_visited_nodes_count.swap_remove(&nd);
        }
        let hier_hi_len = hier[hi].len();
        purge_bit(aig, hier, hier_visited_nodes_count,
                  level, id2order,
                  hi - 1, j);
        purge_bit(aig, hier, hier_visited_nodes_count,
                  level, id2order,
                  hi - 1, j + hier_hi_len);
    }

    // the nodes that are implemented in the hierarchy.
    // we only count for hierarchy[1 and more], [0] is not counted.
    let mut hier_visited_nodes_count: IndexMap<usize, usize> = IndexMap::new();
    let mut selected_level = max_level.min(BOOMERANG_NUM_STAGES);

    /// compute the maximum number of steps needed from this node
    /// to reach an endpoint node.
    ///
    /// during this path, except the starting point, no node should
    /// already be inside the boomerang hierarchy.
    fn compute_reverse_level(
        order: &Vec<usize>,
        id2order: &IndexMap<usize, usize>,
        unrealized_comb_outputs: &IndexSet<usize>,
        realized_inputs: &IndexSet<usize>,
        hier_visited_nodes_count: &IndexMap<usize, usize>,
        aig: &AIG
    ) -> Vec<usize> {
        let mut reverse_level = vec![usize::MAX; order.len()];
        for &i in unrealized_comb_outputs.iter() {
            reverse_level[*id2order.get(&i).unwrap()] = 0;
        }
        for (order_i, i) in order.iter().copied().enumerate().rev() {
            if realized_inputs.contains(&i) ||
                hier_visited_nodes_count.contains_key(&i)
            {
                continue
            }
            let rlvli = reverse_level[order_i];
            if let DriverType::AndGate(a, b) = aig.drivers[i] {
                if a >= 2 {
                    let a = *id2order.get(&(a >> 1)).unwrap();
                    let rlvla = &mut reverse_level[a];
                    if *rlvla == usize::MAX || *rlvla < rlvli + 1 {
                        *rlvla = rlvli + 1;
                    }
                }
                if b >= 2 {
                    let b = *id2order.get(&(b >> 1)).unwrap();
                    let rlvlb = &mut reverse_level[b];
                    if *rlvlb == usize::MAX || *rlvlb < rlvli + 1 {
                        *rlvlb = rlvli + 1;
                    }
                }
            }
        }
        reverse_level
    }

    /// compute the set of nodes that must be implemented in level 1
    /// in addition to the current hierarchy.
    ///
    /// the necessary_level1 nodes can only come from level 0 or
    /// level 1.
    /// a level 1 node is necessary if it is not already
    /// implemented, and it still drives a downstream endpoint.
    /// a level 0 node is necessary if it is not already implemented,
    /// and it either (1) is needed by a level>=2 node, or (2) is
    /// itself an unrealized endpoint.
    fn compute_lvl1_necessary_nodes(
        order: &Vec<usize>,
        id2order: &IndexMap<usize, usize>,
        level: &Vec<usize>,
        reverse_level: &Vec<usize>,
        aig: &AIG,
        unrealized_comb_outputs: &IndexSet<usize>,
        hier_visited_nodes_count: &IndexMap<usize, usize>,
    ) -> IndexSet<usize> {
        let mut lvl1_necessary_nodes = IndexSet::new();
        for order_i in 0..order.len() {
            if hier_visited_nodes_count.contains_key(&order[order_i]) {
                continue
            }
            if reverse_level[order_i] == usize::MAX { continue }
            if level[order_i] == 0 {
                if unrealized_comb_outputs.contains(&order[order_i]) {
                    lvl1_necessary_nodes.insert(order[order_i]);
                }
                continue
            }
            if level[order_i] == 1 {
                lvl1_necessary_nodes.insert(order[order_i]);
            }
            else {
                let (a, b) = match aig.drivers[order[order_i]] {
                    DriverType::AndGate(a, b) => (a, b),
                    _ => panic!()
                };
                if a >= 2 &&
                    level[*id2order.get(&(a >> 1)).unwrap()] == 0 &&
                    !hier_visited_nodes_count.contains_key(&(a >> 1))
                {
                    lvl1_necessary_nodes.insert(a >> 1);
                }
                if b >= 2 &&
                    level[*id2order.get(&(b >> 1)).unwrap()] == 0 &&
                    !hier_visited_nodes_count.contains_key(&(b >> 1))
                {
                    lvl1_necessary_nodes.insert(b >> 1);
                }
            }
        }
        lvl1_necessary_nodes
    }

    let mut reverse_level = compute_reverse_level(
        &order, &id2order,
        unrealized_comb_outputs, realized_inputs,
        &hier_visited_nodes_count, aig
    );

    let mut last_lvl1_necessary_nodes = IndexSet::new();

    while selected_level >= 2 {
        // find a valid slot to place a high level bit
        let mut slot_at_level = usize::MAX;
        for i in 0..hier[selected_level].len() {
            if hier[selected_level][i] == usize::MAX {
                slot_at_level = i;
                break
            }
        }
        if slot_at_level == usize::MAX {
            clilog::trace!("no space at level {}", selected_level);
            selected_level -= 1;
            continue
        }

        // find a valuable node to put into the above slot
        let mut selected_node_ord = usize::MAX;
        for order_i in 0..order.len() {
            if level[order_i] != selected_level { continue }
            if hier_visited_nodes_count.contains_key(&order[order_i]) || reverse_level[order_i] == usize::MAX {
                continue
            }
            if selected_node_ord == usize::MAX ||
                reverse_level[selected_node_ord] < reverse_level[order_i]
            {
                selected_node_ord = order_i;
            }
        }
        if selected_node_ord == usize::MAX {
            clilog::trace!("no node at level {}", selected_level);
            selected_level -= 1;
            continue
        }
        let selected_node = order[selected_node_ord];

        place_bit(
            aig, &mut hier, &mut hier_visited_nodes_count,
            &level, &id2order,
            selected_level, slot_at_level, selected_node
        );

        let reverse_level_upd = compute_reverse_level(
            &order, &id2order,
            unrealized_comb_outputs, realized_inputs,
            &hier_visited_nodes_count, aig
        );

        // store the nodes that need to be put on the 1-level
        // (simple ands).
        // they are periodically checked to ensure they have space.
        let lvl1_necessary_nodes = compute_lvl1_necessary_nodes(
            &order, &id2order, &level,
            &reverse_level_upd, aig, &unrealized_comb_outputs,
            &hier_visited_nodes_count
        );

        let num_lvl1_hier_taken =
            hier[1].iter().filter(|i| **i != usize::MAX).count();

        clilog::trace!(
            "taken one node at level {}, used 1-level space {}, hier visited unique {}, num nodes necessary in lvl1 {}",
            selected_level, num_lvl1_hier_taken,
            hier_visited_nodes_count.len(), lvl1_necessary_nodes.len()
        );

        if lvl1_necessary_nodes.len() +
            num_lvl1_hier_taken.max(hier_visited_nodes_count.len())
            >= (1 << (BOOMERANG_NUM_STAGES - 1))
        {
            clilog::trace!("REVERSED the plan due to overflow");
            purge_bit(
                aig, &mut hier, &mut hier_visited_nodes_count,
                &level, &id2order,
                selected_level, slot_at_level
            );
            selected_level -= 1;
            continue
        }

        reverse_level = reverse_level_upd;
        last_lvl1_necessary_nodes = lvl1_necessary_nodes;
    }

    if last_lvl1_necessary_nodes.is_empty() {
        last_lvl1_necessary_nodes = compute_lvl1_necessary_nodes(
            &order, &id2order, &level,
            &reverse_level, aig, &unrealized_comb_outputs,
            &hier_visited_nodes_count
        );
    }

    // the hierarchy is now constructed except all 1-level nodes.
    // it's time to place them. during this process, we heuristically collect
    // endpoint nodes into consecutive space for early write-out.
    //
    // we first try to finalize all endpoints that have to appear in
    // level 1.
    // after that, we will try if we can write out all others scattered.
    let mut endpoints_lvl1 = Vec::new();
    let mut endpoints_untouched = Vec::new();
    let mut endpoints_hier = IndexSet::new();
    for &endpt in unrealized_comb_outputs.iter() {
        if hier_visited_nodes_count.contains_key(&endpt) {
            endpoints_hier.insert(endpt);
        }
        else if last_lvl1_necessary_nodes.contains(&endpt) {
            endpoints_lvl1.push(endpt);
        }
        else {
            endpoints_untouched.push(endpt);
        }
    }

    // collect all 32-consecutive level 1 spaces.
    // (num occupied, i), will be sorted later.
    let mut spaces = Vec::new();
    for i in 0..hier[1].len() / 32 {
        let mut num_occupied = 0u8;
        for j in i * 32..(i + 1) * 32 {
            if hier[1][j] != usize::MAX {
                num_occupied += 1;
            }
        }
        if num_occupied < 10 {
            spaces.push((num_occupied, i * 32))
        }
    }
    spaces.sort();
    let mut spaces_j = 0;
    let mut endpt_lvl1_i = 0;
    let mut realized_endpoints = IndexSet::new();
    let mut write_outs = Vec::new();
    // heuristically push level 1 endpoints.
    while spaces_j < spaces.len() &&
        (endpoints_untouched.is_empty() || // if we can try all
         endpoints_lvl1.len() - endpt_lvl1_i >= (32 - spaces[spaces_j].0) as usize)
    {
        let i = spaces[spaces_j].1;
        for j in i..i + 32 {
            if endpt_lvl1_i >= endpoints_lvl1.len() { break }
            if hier[1][j] == usize::MAX {
                let endpt_i = endpoints_lvl1[endpt_lvl1_i];
                place_bit(
                    aig, &mut hier, &mut hier_visited_nodes_count,
                    &level, &id2order,
                    1, j, endpt_i
                );
                realized_endpoints.insert(endpt_i);
                endpt_lvl1_i += 1;
            }
            else if unrealized_comb_outputs.contains(&hier[1][j]) {
                realized_endpoints.insert(hier[1][j]);
            }
        }
        *total_write_outs += 1;
        write_outs.push((i + hier[1].len()) / 32);
        spaces_j += 1;
    }

    if *total_write_outs > BOOMERANG_MAX_WRITEOUTS - num_reserved_writeouts {
        clilog::trace!("boomerang: write out overflowed");
        return None
    }

    // then place all remaining lvl1 nodes in any order.
    // clilog::debug!("last_lvl1_necessary: {}, hier visited: {}, realized endpts: {}", last_lvl1_necessary_nodes.len(), hier_visited_nodes_count.len(), realized_endpoints.len());
    let mut hier1_j = 0;
    for &nd in &last_lvl1_necessary_nodes {
        if hier_visited_nodes_count.contains_key(&nd) ||
            realized_endpoints.contains(&nd)
        {
            continue
        }
        while hier[1][hier1_j] != usize::MAX {
            hier1_j += 1;
            if hier1_j >= hier[1].len() {
                clilog::trace!("boomerang: overflow putting lvl1");
                return None
            }
        }
        place_bit(
            aig, &mut hier, &mut hier_visited_nodes_count,
            &level, &id2order,
            1, hier1_j, nd
        );
    }
    while hier[1][hier1_j] != usize::MAX {
        hier1_j += 1;
        if hier1_j >= hier[1].len() {
            clilog::trace!("boomerang: overflow putting lvl1 (just a zero pin..)");
            return None
        }
    }

    // check if we can make this the last stage.
    if endpoints_untouched.is_empty() {
        let mut add_write_outs = IndexSet::new();
        for hi in 1..=BOOMERANG_NUM_STAGES {
            for j in 0..hier[hi].len() {
                let nd = hier[hi][j];
                if endpoints_hier.contains(&nd) && !realized_endpoints.contains(&nd) {
                    add_write_outs.insert((j + hier[hi].len()) / 32);
                    if add_write_outs.len() + *total_write_outs > BOOMERANG_MAX_WRITEOUTS - num_reserved_writeouts {
                        break
                    }
                }
            }
        }
        if add_write_outs.len() + *total_write_outs <= BOOMERANG_MAX_WRITEOUTS - num_reserved_writeouts {
            for wo in add_write_outs {
                write_outs.push(wo);
                *total_write_outs += 1;
            }
            for endpt in endpoints_hier {
                realized_endpoints.insert(endpt);
            }
        }
    }

    for (&i, _) in &hier_visited_nodes_count {
        realized_inputs.insert(i);
    }
    for &i in &realized_endpoints {
        assert!(unrealized_comb_outputs.swap_remove(&i));
    }

    Some(BoomerangStage {
        hier,
        write_outs
    })
}

impl Partition {
    /// build one partition given a set of endpoints to realize.
    ///
    /// if the resource is overflowed, None will be returned.
    /// see [Partition] for resource constraints.
    pub fn build_one(
        aig: &AIG,
        staged: &StagedAIG,
        endpoints: &Vec<usize>
    ) -> Option<Partition> {
        let mut unrealized_comb_outputs = IndexSet::new();
        let mut realized_inputs = staged.primary_inputs.as_ref()
            .cloned().unwrap_or_default();
        let mut num_srams = 0;
        let mut comb_outputs_activations = IndexMap::<usize, IndexSet<usize>>::new();
        for &endpt_i in endpoints {
            let edg = staged.get_endpoint_group(aig, endpt_i);
            edg.for_each_input(|i| {
                unrealized_comb_outputs.insert(i);
            });
            match edg {
                EndpointGroup::DFF(dff) => {
                    comb_outputs_activations.entry(dff.d_iv >> 1).or_default().insert(dff.en_iv << 1 | (dff.d_iv & 1));
                },
                EndpointGroup::PrimaryOutput(pin) => {
                    comb_outputs_activations.entry(pin >> 1).or_default().insert(2 | (pin & 1));
                },
                EndpointGroup::RAMBlock(_) => {
                    num_srams += 1;
                },
                EndpointGroup::StagedIOPin(pin) => {
                    comb_outputs_activations.entry(pin).or_default().insert(2);
                },
            }
        }
        let num_output_dups = comb_outputs_activations.iter()
            .map(|(_, ckens)| ckens.len() - 1)
            .sum::<usize>();
        let num_reserved_writeouts = num_srams + (num_output_dups + 31) / 32;
        if num_reserved_writeouts >= BOOMERANG_MAX_WRITEOUTS ||
            num_srams * 4 + num_output_dups > BOOMERANG_MAX_WRITEOUTS
        {
            // overflowed writeout
            return None
        }
        let mut stages = Vec::<BoomerangStage>::new();
        let mut total_write_outs = 0;
        while !unrealized_comb_outputs.is_empty() {
            let stage = build_one_boomerang_stage(
                aig, &mut unrealized_comb_outputs,
                &mut realized_inputs, &mut total_write_outs,
                num_reserved_writeouts
            )?;
            stages.push(stage);
        }
        Some(Partition {
            endpoints: endpoints.clone(),
            stages
        })
    }
}

/// Given an initial clustering solution of endpoints, generate and map a
/// refined solution.
///
/// The refined solution will have smaller number of partitions
/// as we aggressively merge the partitions when possible.
pub fn process_partitions(
    aig: &AIG,
    staged: &StagedAIG,
    mut parts: Vec<Vec<usize>>,
    max_stage_degrad: usize,
) -> Option<Vec<Partition>> {
    let cnt_nodes = parts.par_iter().map(|v| {
        let mut comb_outputs = Vec::new();
        for &endpt_i in v {
            staged.get_endpoint_group(aig, endpt_i).for_each_input(|i| {
                comb_outputs.push(i);
            });
        }
        let order = aig.topo_traverse_generic(
            Some(&comb_outputs),
            staged.primary_inputs.as_ref(),
        );
        order.len()
    }).collect::<Vec<_>>();

    let all_original_parts = parts.par_iter().enumerate().map(|(i, v)| {
        let part = Partition::build_one(aig, staged, v);
        if part.is_none() {
            clilog::error!("Partition {} exceeds resource constraint.", i);
        }
        part
    }).collect::<Vec<_>>();
    let all_original_parts = all_original_parts.into_iter().collect::<Option<Vec<_>>>()?;
    let max_original_nstages = all_original_parts.iter()
        .map(|p| p.stages.len()).max().unwrap();

    let mut effective_parts = Vec::<Partition>::new();
    let max_trials = (all_original_parts.len() / 8).max(20);
    for (i, mut partition_self) in all_original_parts.into_iter().enumerate() {
        if parts[i].is_empty() {
            continue
        }
        let mut merge_blacklist = HashSet::<usize>::new();
        let mut cnt_node_i = cnt_nodes[i];
        loop {
            let mut comb_outputs = Vec::new();
            for &endpt_i in &parts[i] {
                staged.get_endpoint_group(aig, endpt_i).for_each_input(|i| {
                    comb_outputs.push(i);
                });
            }

            let mut merge_choices = parts[i + 1..parts.len()].par_iter().enumerate().filter_map(|(j, v)| {
                if v.is_empty() { return None }
                if merge_blacklist.contains(&(i + j + 1)) {
                    return None
                }
                let mut comb_outputs = comb_outputs.clone();
                for &endpt_i in v {
                    staged.get_endpoint_group(aig, endpt_i).for_each_input(|i| {
                        comb_outputs.push(i);
                    });
                }
                let order = aig.topo_traverse_generic(
                    Some(&comb_outputs),
                    staged.primary_inputs.as_ref(),
                );
                Some((order.len() - cnt_nodes[i + j + 1].max(cnt_node_i),
                      order.len(),
                      i + j + 1))
            }).collect::<Vec<_>>();
            merge_choices.sort();
            let mut merged = false;

            #[derive(Clone)]
            struct PartsPartitions {
                parts_ij: Vec<usize>,
                partition_ij: Option<Partition>,
            }
            let mut merge_trials: Vec<Option<PartsPartitions>> =
                vec![None; merge_choices.len()];
            let mut parallel_trial_stride = 4;

            for (merge_i, &(_cnt_diff, cnt_new, j)) in merge_choices.iter().enumerate() {
                if merge_trials[merge_i].is_none() {
                    if merge_i > max_trials {
                        break   // do not try too more
                    }
                    let rhs = merge_trials.len().min(
                        merge_i + parallel_trial_stride);
                    merge_trials[merge_i..rhs].par_iter_mut().enumerate().for_each(|(merge_j, trial)| {
                        let j = merge_choices[merge_i + merge_j].2;
                        let parts_ij = parts[i].iter().chain(parts[j].iter()).copied().collect();
                        let partition_ij = Partition::build_one(aig, staged, &parts_ij);
                        *trial = Some(PartsPartitions {
                            parts_ij, partition_ij
                        });
                    });
                    parallel_trial_stride *= 2;
                }

                let PartsPartitions {
                    parts_ij, partition_ij
                } = merge_trials[merge_i].take().unwrap();

                match partition_ij {
                    None => {
                        merge_blacklist.insert(j);
                    }
                    Some(partition) if partition.stages.len() >
                        max_original_nstages + max_stage_degrad =>
                    {
                        clilog::debug!("skipped merging {} with {} due to nstage degradation: \
                                        {} > {}", i, j, partition.stages.len(),
                                       max_original_nstages + max_stage_degrad);
                        merge_blacklist.insert(j);
                    }
                    Some(partition) => {
                        clilog::info!("merged partition {} with {}", i, j);
                        parts[i] = parts_ij;
                        parts[j] = vec![];
                        partition_self = partition;
                        merged = true;
                        cnt_node_i = cnt_new;
                        break
                    },
                }
            }
            if !merged { break }
        }

        clilog::info!("part {}: #stages {}",
                      i, partition_self.stages.len());
        effective_parts.push(partition_self);
    }
    effective_parts.sort_by_key(|p| usize::MAX - p.stages.len());
    Some(effective_parts)
}

/// Read a cluster solution from hgr.part.xx file.
/// Then call [process_partitions].
pub fn process_partitions_from_hgr_parts_file(
    aig: &AIG,
    staged: &StagedAIG,
    hgr_parts_file: &PathBuf,
    max_stage_degrad: usize,
) -> Option<Vec<Partition>> {
    use std::io::{BufRead, BufReader};
    use std::fs::File;

    let mut parts = Vec::<Vec<usize>>::new();
    let f_parts = File::open(&hgr_parts_file).unwrap();
    let f_parts = BufReader::new(f_parts);
    for (i, line) in f_parts.lines().enumerate() {
        let line = line.unwrap();
        if line.is_empty() { continue }
        let part_id = line.parse::<usize>().unwrap();
        while parts.len() <= part_id {
            parts.push(vec![]);
        }
        parts[part_id].push(i);
    }
    clilog::info!("read parts file {} with {} parts",
                  hgr_parts_file.display(), parts.len());

    process_partitions(aig, staged, parts, max_stage_degrad)
}
