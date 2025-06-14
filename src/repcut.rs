// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
/// RepCut implementation

use crate::aig::{DriverType, AIG};
use crate::staging::StagedAIG;
use indexmap::{IndexMap, IndexSet};
use cachedhash::CachedHash;
use std::collections::HashMap;
use std::sync::Arc;
use std::fmt;
use rayon::prelude::*;
use rand::prelude::*;
use rand_chacha::ChaCha20Rng;

const REPCUT_HYPERGRAPH_EDGE_SIZE_LIMIT: usize = 1000;
const REPCUT_BITSET_BLOCK_SIZE: usize = 4096;

#[derive(Hash, PartialEq, Eq, Debug)]
struct EndpointSetSegment {
    bs_set: [u64; REPCUT_BITSET_BLOCK_SIZE / 64],
}

impl Default for EndpointSetSegment {
    fn default() -> Self {
        EndpointSetSegment {
            bs_set: [0; REPCUT_BITSET_BLOCK_SIZE / 64]
        }
    }
}

#[derive(Hash, PartialEq, Eq, Debug)]
struct EndpointSet {
    s: Vec<Option<Arc<CachedHash<EndpointSetSegment>>>>,
}

pub struct RCHyperGraph {
    num_vertices: usize,
    clusters: IndexMap<CachedHash<EndpointSet>, usize>,
    endpoint_weights: Vec<u64>,
}

impl EndpointSet {
    fn popcount(&self) -> usize {
        self.s.iter().map(|o| {
            match o {
                Some(ess) =>
                    ess.bs_set.iter().map(|u| u.count_ones())
                    .sum::<u32>() as usize,
                None => 0
            }
        }).sum()
    }
}

impl RCHyperGraph {
    pub fn from_staged_aig(aig: &AIG, staged: &StagedAIG) -> RCHyperGraph {
        let timer_repcut_endpoint_process = clilog::stimer!("repcut endpoint process");
        let num_blocks = (
            staged.num_endpoint_groups() + REPCUT_BITSET_BLOCK_SIZE - 1
        ) / REPCUT_BITSET_BLOCK_SIZE;
        let mut segments_blockid_nodeid = vec![
            Vec::<Option<Arc<CachedHash<EndpointSetSegment>>>>::new();
            num_blocks
        ];
        segments_blockid_nodeid.par_iter_mut().enumerate().for_each(|(i_block, vs)| {
            *vs = vec![None; aig.num_aigpins + 1];
            let endpoint_block_st = i_block * REPCUT_BITSET_BLOCK_SIZE;
            let endpoint_block_ed = staged.num_endpoint_groups()
                .min(endpoint_block_st + REPCUT_BITSET_BLOCK_SIZE);
            let mut endpoint_pins = Vec::new();
            for endpt_i in endpoint_block_st..endpoint_block_ed {
                staged.get_endpoint_group(aig, endpt_i).for_each_input(|i| {
                    endpoint_pins.push(i);
                });
            }
            let order_blk = aig.topo_traverse_generic(
                Some(&endpoint_pins),
                staged.primary_inputs.as_ref()
            );
            let mut unique_segs =
                IndexSet::<Arc<CachedHash<EndpointSetSegment>>>::new();
            let mut ess_init: HashMap<usize, EndpointSetSegment> =
                HashMap::new();
            for endpt_i in endpoint_block_st..endpoint_block_ed {
                let idx_offset = endpt_i - endpoint_block_st;
                staged.get_endpoint_group(aig, endpt_i).for_each_input(|i| {
                    let ess = ess_init.entry(i).or_default();
                    ess.bs_set[idx_offset / 64] |= 1 << (idx_offset % 64);
                });
            }
            for order_i in (0..order_blk.len()).rev() {
                let i = order_blk[order_i];
                let mut ess =
                    ess_init.remove(&i).unwrap_or_default();
                let fs = aig.fanouts_start[i];
                let fe = aig.fanouts_start[i + 1];
                for fi in fs..fe {
                    let j = aig.fanouts[fi];
                    if let Some(vj) = &mut vs[j] {
                        for bs_k in 0..REPCUT_BITSET_BLOCK_SIZE / 64 {
                            ess.bs_set[bs_k] |= vj.bs_set[bs_k];
                        }
                    }
                }
                let ess = Arc::new(
                    CachedHash::new(ess)
                );
                let (idx, _) = unique_segs.insert_full(ess);
                vs[i] = Some(unique_segs.get_index(idx).unwrap().clone());
            }
            // println!("vs: {:?}", vs);
        });
        // println!("sbn: {:?}", segments_blockid_nodeid);
        let mut clusters = IndexMap::<_, usize>::new();
        for i in 1..aig.num_aigpins {
            let es = CachedHash::new(EndpointSet {
                s: (0..num_blocks)
                    .map(|k| segments_blockid_nodeid[k][i]
                         .clone()).collect()
            });
            if es.popcount() >= 2 {
                *clusters.entry(es).or_default() += 1;
            }
        }
        clilog::finish!(timer_repcut_endpoint_process);

        let mut endpoint_pins_all = Vec::new();
        for endpt_i in 0..staged.num_endpoint_groups() {
            staged.get_endpoint_group(aig, endpt_i).for_each_input(|i| {
                endpoint_pins_all.push(i);
            });
        }
        let order_all = aig.topo_traverse_generic(
            Some(&endpoint_pins_all),
            staged.primary_inputs.as_ref()
        );
        let mut node_weights = vec![0.0f32; aig.num_aigpins + 1];
        for &i in &order_all {
            node_weights[i] = 1.;
            if let DriverType::AndGate(a, b) = aig.drivers[i] {
                if (a >> 1) != 0 {
                    node_weights[i] += node_weights[a >> 1] / ((
                        aig.fanouts_start[(a >> 1) + 1] - aig.fanouts_start[a >> 1]
                    ) as f32);
                }
                if (b >> 1) != 0 {
                    node_weights[i] += node_weights[b >> 1] / ((
                        aig.fanouts_start[(b >> 1) + 1] - aig.fanouts_start[b >> 1]
                    ) as f32);
                }
            }
        }
        let mut num_fanouts_to_endpt = vec![0usize; aig.num_aigpins + 1];
        for endpt_i in 0..staged.num_endpoint_groups() {
            staged.get_endpoint_group(aig, endpt_i).for_each_input(|i| {
                num_fanouts_to_endpt[i] += 1;
            });
        }
        let endpoint_weights = (0..staged.num_endpoint_groups()).map(|endpt_i| {
            let mut tot = 0.0;
            staged.get_endpoint_group(aig, endpt_i).for_each_input(|i| {
                tot += node_weights[i] / (num_fanouts_to_endpt[i] as f32)
            });
            (tot + 0.5) as u64
        }).collect();

        // println!("clusters: {:#?}, endpoint_weights: {:#?}", clusters, endpoint_weights);
        RCHyperGraph {
            num_vertices: staged.num_endpoint_groups(),
            clusters, endpoint_weights
        }
    }

    /// Make an edge list.
    ///
    /// (weight, node indices)
    pub fn to_edges(&self) -> Vec<(usize, Vec<usize>)> {
        self.clusters.par_iter().enumerate().map(|(i, (s, v))| {
            let mut rng = ChaCha20Rng::seed_from_u64(8026727 + i as u64);
            let mut edgend = Vec::<usize>::new();
            let mut num_prev_nodes = 0;
            for segment_i in 0..s.s.len() {
                let bs_set = match &s.s[segment_i] {
                    Some(seg) => &seg.bs_set,
                    None => continue
                };
                for bs_i in 0..REPCUT_BITSET_BLOCK_SIZE / 64 {
                    if bs_set[bs_i] == 0 {
                        continue
                    }
                    for k in 0..64 {
                        if (bs_set[bs_i] >> k & 1) != 0 {
                            let nd = segment_i * REPCUT_BITSET_BLOCK_SIZE + bs_i * 64 + k;
                            if edgend.len() < REPCUT_HYPERGRAPH_EDGE_SIZE_LIMIT {
                                edgend.push(nd);
                            }
                            else if rng.gen_range(0..num_prev_nodes) < REPCUT_HYPERGRAPH_EDGE_SIZE_LIMIT {
                                edgend[rng.gen_range(0..REPCUT_HYPERGRAPH_EDGE_SIZE_LIMIT)] = nd;
                            }
                            num_prev_nodes += 1;
                        }
                    }
                }
            }
            (*v, edgend)
        }).collect()
    }

    /// Run mt-kahypar to partition this hypergraph.
    pub fn partition(&self, num_parts: usize) -> Vec<usize> {
        let ctx = mt_kahypar::Context::builder()
            .preset(mt_kahypar::Preset::Deterministic)
            .k(num_parts as i32)
            .epsilon(0.2)
            .objective(mt_kahypar::Objective::Soed)
            .verbose(true)
            .build().unwrap();
        let edges = self.to_edges();
        let mut hyperedge_indices = Vec::with_capacity(edges.len() + 1);
        hyperedge_indices.push(0);
        hyperedge_indices.extend(edges.iter().scan(0, |acc, (_, edgend)| {
            *acc += edgend.len();
            Some(*acc)
        }));
        let mut hyperedges = Vec::with_capacity(hyperedge_indices[edges.len()]);
        hyperedges.extend(edges.iter().flat_map(|(_, edgend)| {
            edgend.iter().copied()
        }));
        let hyperedge_weights = edges.iter().map(|(v, _)| TryInto::<i32>::try_into(*v).unwrap()).collect::<Vec<_>>();
        let vertex_weights = self.endpoint_weights.iter().map(|v| TryInto::<i32>::try_into(*v).unwrap()).collect::<Vec<_>>();
        let hg = mt_kahypar::Hypergraph::from_adjacency(
            &ctx,
            self.num_vertices,
            &hyperedge_indices,
            &hyperedges,
            Some(&hyperedge_weights),
            Some(&vertex_weights)
        ).unwrap();
        let parts = hg.partition().unwrap();
        parts.extract_partition().into_iter().map(|i| i as usize).collect()
    }
}

impl fmt::Display for RCHyperGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} {} 11", self.clusters.len(), self.endpoint_weights.len())?;
        for (v, edgend) in self.to_edges() {
            write!(f, "{}", v)?;
            for nd in edgend {
                write!(f, " {}", nd + 1)?;
            }
            writeln!(f)?;
        }
        for w in &self.endpoint_weights {
            writeln!(f, "{}", w)?;
        }
        Ok(())
    }
}
