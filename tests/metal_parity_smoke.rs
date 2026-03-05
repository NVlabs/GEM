// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]

use gem::aig::{AIG, DriverType};
use gem::aigpdk::{AIGPDKLeafPins, AIGPDK_SRAM_SIZE};
use gem::flatten::FlattenedScriptV1;
use gem::gpu::backend::GpuBackendV1;
use gem::gpu::metal_backend::MetalBackend;
use gem::pe::Partition;
use gem::staging::build_staged_aigs;
use netlistdb::NetlistDB;
use std::path::PathBuf;
use ulib::UVec;

fn fixture_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel)
}

#[derive(Debug, Clone)]
struct BaselineManifestCase {
    name: String,
    netlist_verilog: String,
    gemparts: String,
    num_blocks: usize,
    level_split: Vec<usize>,
}

#[derive(Debug, Default)]
struct BaselineManifestBuilder {
    name: Option<String>,
    netlist_verilog: Option<String>,
    gemparts: Option<String>,
    num_blocks: Option<usize>,
    level_split: Option<Vec<usize>>,
}

fn parse_manifest_string(raw: &str) -> Option<String> {
    let value = raw.trim();
    if value.len() < 2 || !value.starts_with('"') || !value.ends_with('"') {
        return None;
    }
    Some(value[1..value.len() - 1].to_string())
}

fn parse_manifest_usize(raw: &str) -> Option<usize> {
    raw.trim().parse::<usize>().ok()
}

fn parse_manifest_usize_list(raw: &str) -> Option<Vec<usize>> {
    let value = raw.trim();
    if value == "[]" {
        return Some(Vec::new());
    }
    if !value.starts_with('[') || !value.ends_with(']') {
        return None;
    }
    let inner = value[1..value.len() - 1].trim();
    if inner.is_empty() {
        return Some(Vec::new());
    }
    let mut list = Vec::new();
    for token in inner.split(',') {
        list.push(token.trim().parse::<usize>().ok()?);
    }
    Some(list)
}

fn finish_manifest_entry(builder: &mut BaselineManifestBuilder, out: &mut Vec<BaselineManifestCase>) {
    let Some(name) = builder.name.take() else { return };
    let netlist_verilog = builder
        .netlist_verilog
        .take()
        .expect("baseline manifest entry missing netlist_verilog");
    let gemparts = builder
        .gemparts
        .take()
        .expect("baseline manifest entry missing gemparts");
    let num_blocks = builder.num_blocks.take().unwrap_or(1);
    let level_split = builder.level_split.take().unwrap_or_default();
    out.push(BaselineManifestCase {
        name,
        netlist_verilog,
        gemparts,
        num_blocks,
        level_split,
    });
}

fn parse_baseline_manifest_cases() -> Vec<BaselineManifestCase> {
    let manifest_path = fixture_path("baseline/manifest.toml");
    let content = std::fs::read_to_string(&manifest_path).expect("cannot read baseline manifest");
    let mut entries = Vec::new();
    let mut builder = BaselineManifestBuilder::default();

    for line in content.lines() {
        let without_comment = line.split('#').next().unwrap_or_default().trim();
        if without_comment.is_empty() {
            continue;
        }
        let Some((key, value)) = without_comment.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim();

        if key == "name" && builder.name.is_some() {
            finish_manifest_entry(&mut builder, &mut entries);
        }

        match key {
            "name" => builder.name = parse_manifest_string(value),
            "netlist_verilog" => builder.netlist_verilog = parse_manifest_string(value),
            "gemparts" => builder.gemparts = parse_manifest_string(value),
            "num_blocks" => builder.num_blocks = parse_manifest_usize(value),
            "level_split" => builder.level_split = parse_manifest_usize_list(value),
            _ => {}
        }
    }

    finish_manifest_entry(&mut builder, &mut entries);
    assert!(
        !entries.is_empty(),
        "baseline manifest parsed no entries: {}",
        manifest_path.display()
    );
    entries
}

fn build_script_from_artifacts(
    netlist_rel: &str,
    gemparts_rel: &str,
    num_blocks: usize,
    level_split: &[usize],
) -> FlattenedScriptV1 {
    let netlist_path = fixture_path(netlist_rel);
    let gemparts_path = fixture_path(gemparts_rel);

    let netlistdb = NetlistDB::from_sverilog_file(&netlist_path, None, &AIGPDKLeafPins())
        .expect("cannot build netlist");
    let aig = AIG::from_netlistdb(&netlistdb);
    let stageds = build_staged_aigs(&aig, level_split);

    let f = std::fs::File::open(gemparts_path).expect("cannot open gemparts");
    let mut buf = std::io::BufReader::new(f);
    let parts_in_stages: Vec<Vec<Partition>> =
        serde_bare::from_reader(&mut buf).expect("cannot decode gemparts");

    let mut input_layout = Vec::new();
    for (i, driv) in aig.drivers.iter().enumerate() {
        if let DriverType::InputPort(_) | DriverType::InputClockFlag(_, _) = driv {
            input_layout.push(i);
        }
    }

    FlattenedScriptV1::from(
        &aig,
        &stageds
            .iter()
            .map(|(_, _, staged)| staged)
            .collect::<Vec<_>>(),
        &parts_in_stages
            .iter()
            .map(|ps| ps.as_slice())
            .collect::<Vec<_>>(),
        num_blocks,
        input_layout,
    )
}

fn build_tiny_script(num_blocks: usize) -> FlattenedScriptV1 {
    build_script_from_artifacts("baseline/tiny_gatelevel.gv", "baseline/tiny.gemparts", num_blocks, &[])
}

fn simulate_block_v1_reference(
    script: &[u32],
    input_state: &[u32],
    output_state: &mut [u32],
    sram_data: &mut [u32],
) {
    let mut script_pi = 0usize;
    loop {
        let num_stages = script[script_pi];
        let is_last_part = script[script_pi + 1];
        let num_ios = script[script_pi + 2];
        let io_offset = script[script_pi + 3];
        let num_srams = script[script_pi + 4];
        let sram_offset = script[script_pi + 5];
        let num_global_read_rounds = script[script_pi + 6];
        let num_output_duplicates = script[script_pi + 7];

        let mut writeout_hooks = vec![0u32; 256];
        for i in 0..128 {
            let t = script[script_pi + 128 + i];
            writeout_hooks[i * 2] = t & 0xffff;
            writeout_hooks[i * 2 + 1] = t >> 16;
        }

        script_pi += 256;
        if num_stages == 0 {
            break;
        }

        let mut writeouts = vec![0u32; num_ios as usize];
        let mut state = vec![0u32; 256];

        for _ in 0..num_global_read_rounds {
            for i in 0..256 {
                let mut cur_state = state[i];
                let idx = script[script_pi + i * 2];
                let mut mask = script[script_pi + i * 2 + 1];
                if mask == 0 {
                    continue;
                }

                let value = if (idx >> 31) == 0 {
                    input_state[idx as usize]
                } else {
                    output_state[(idx ^ (1 << 31)) as usize]
                };

                while mask != 0 {
                    cur_state <<= 1;
                    let lowbit = mask & (-(mask as i32)) as u32;
                    if (value & lowbit) != 0 {
                        cur_state |= 1;
                    }
                    mask ^= lowbit;
                }
                state[i] = cur_state;
            }
            script_pi += 256 * 2;
        }

        for bs_i in 0..num_stages {
            let mut hier_inputs = vec![0u32; 256];
            let mut hier_flag_xora = vec![0u32; 256];
            let mut hier_flag_xorb = vec![0u32; 256];
            let mut hier_flag_orb = vec![0u32; 256];

            for k_outer in 0..4 {
                for i in 0..256 {
                    for k_inner in 0..4 {
                        let k = k_outer * 4 + k_inner;
                        let t_shuffle = script[script_pi + i * 4 + k_inner];
                        let t_shuffle_1_idx = t_shuffle & 0xffff;
                        let t_shuffle_2_idx = t_shuffle >> 16;
                        hier_inputs[i] |=
                            (state[(t_shuffle_1_idx >> 5) as usize] >> (t_shuffle_1_idx & 31)
                                & 1)
                                << (k * 2);
                        hier_inputs[i] |=
                            (state[(t_shuffle_2_idx >> 5) as usize] >> (t_shuffle_2_idx & 31)
                                & 1)
                                << (k * 2 + 1);
                    }
                }
                script_pi += 256 * 4;
            }

            for i in 0..256 {
                hier_flag_xora[i] = script[script_pi + i * 4];
                hier_flag_xorb[i] = script[script_pi + i * 4 + 1];
                hier_flag_orb[i] = script[script_pi + i * 4 + 2];
            }
            script_pi += 256 * 4;

            for i in 0..128 {
                let a = hier_inputs[i];
                let b = hier_inputs[128 + i];
                let xora = hier_flag_xora[128 + i];
                let xorb = hier_flag_xorb[128 + i];
                let orb = hier_flag_orb[128 + i];
                hier_inputs[128 + i] = (a ^ xora) & ((b ^ xorb) | orb);
            }

            for hi in 1..=7 {
                let hier_width = 1 << (7 - hi);
                for i in 0..hier_width {
                    let a = hier_inputs[hier_width * 2 + i];
                    let b = hier_inputs[hier_width * 3 + i];
                    let xora = hier_flag_xora[hier_width + i];
                    let xorb = hier_flag_xorb[hier_width + i];
                    let orb = hier_flag_orb[hier_width + i];
                    hier_inputs[hier_width + i] = (a ^ xora) & ((b ^ xorb) | orb);
                }
            }

            let v1 = hier_inputs[1];
            let xora = hier_flag_xora[0];
            let xorb = hier_flag_xorb[0];
            let orb = hier_flag_orb[0];
            let r8 = ((v1 << 16) ^ xora) & ((v1 ^ xorb) | orb) & 0xffff0000;
            let r9 = ((r8 >> 8) ^ xora) & (((r8 >> 16) ^ xorb) | orb) & 0xff00;
            let r10 = ((r9 >> 4) ^ xora) & (((r9 >> 8) ^ xorb) | orb) & 0xf0;
            let r11 = ((r10 >> 2) ^ xora) & (((r10 >> 4) ^ xorb) | orb) & 0b1100;
            let r12 = ((r11 >> 1) ^ xora) & (((r11 >> 2) ^ xorb) | orb) & 0b10;
            hier_inputs[0] = r8 | r9 | r10 | r11 | r12;

            state = hier_inputs;

            for i in 0..256 {
                let hook_i = writeout_hooks[i];
                if (hook_i >> 8) == bs_i {
                    writeouts[i] = state[(hook_i & 255) as usize];
                }
            }
        }

        let mut sram_duplicate_perm = vec![0u32; (num_srams * 4 + num_output_duplicates) as usize];
        for k_outer in 0..4 {
            for i in 0..(num_srams * 4 + num_output_duplicates) {
                for k_inner in 0..4 {
                    let k = k_outer * 4 + k_inner;
                    let t_shuffle = script[script_pi + (i * 4 + k_inner) as usize];
                    let t_shuffle_1_idx = t_shuffle & 0xffff;
                    let t_shuffle_2_idx = t_shuffle >> 16;
                    sram_duplicate_perm[i as usize] |=
                        (writeouts[(t_shuffle_1_idx >> 5) as usize] >> (t_shuffle_1_idx & 31)
                            & 1)
                            << (k * 2);
                    sram_duplicate_perm[i as usize] |=
                        (writeouts[(t_shuffle_2_idx >> 5) as usize] >> (t_shuffle_2_idx & 31)
                            & 1)
                            << (k * 2 + 1);
                }
            }
            script_pi += 256 * 4;
        }

        for i in 0..(num_srams * 4 + num_output_duplicates) as usize {
            sram_duplicate_perm[i] &= !script[script_pi + i * 4 + 1];
            sram_duplicate_perm[i] ^= script[script_pi + i * 4];
        }
        script_pi += 256 * 4;

        for sram_i_u32 in 0..num_srams {
            let sram_i = sram_i_u32 as usize;
            let addrs = sram_duplicate_perm[sram_i * 4];
            let port_r_addr_iv = addrs & 0xffff;
            let port_w_addr_iv = (addrs & 0xffff0000) >> 16;
            let port_w_wr_en = sram_duplicate_perm[sram_i * 4 + 1];
            let port_w_wr_data_iv = sram_duplicate_perm[sram_i * 4 + 2];

            let sram_st = sram_offset as usize + sram_i * AIGPDK_SRAM_SIZE;
            let sram_ed = sram_st + AIGPDK_SRAM_SIZE;
            let ram = &mut sram_data[sram_st..sram_ed];
            let r = ram[port_r_addr_iv as usize];
            let w0 = ram[port_w_addr_iv as usize];
            writeouts[(num_ios - num_srams + sram_i_u32) as usize] = r;
            ram[port_w_addr_iv as usize] = (w0 & !port_w_wr_en) | (port_w_wr_data_iv & port_w_wr_en);
        }

        for i in 0..num_output_duplicates {
            writeouts[(num_ios - num_srams - num_output_duplicates + i) as usize] =
                sram_duplicate_perm[(num_srams * 4 + i) as usize];
        }

        let mut clken_perm = vec![0u32; num_ios as usize];
        let writeouts_for_clken = writeouts.clone();
        for k_outer in 0..4 {
            for i in 0..num_ios {
                for k_inner in 0..4 {
                    let k = k_outer * 4 + k_inner;
                    let t_shuffle = script[script_pi + (i * 4 + k_inner) as usize];
                    let t_shuffle_1_idx = t_shuffle & 0xffff;
                    let t_shuffle_2_idx = t_shuffle >> 16;
                    clken_perm[i as usize] |=
                        (writeouts_for_clken[(t_shuffle_1_idx >> 5) as usize]
                            >> (t_shuffle_1_idx & 31)
                            & 1)
                            << (k * 2);
                    clken_perm[i as usize] |=
                        (writeouts_for_clken[(t_shuffle_2_idx >> 5) as usize]
                            >> (t_shuffle_2_idx & 31)
                            & 1)
                            << (k * 2 + 1);
                }
            }
            script_pi += 256 * 4;
        }

        for i in 0..num_ios as usize {
            clken_perm[i] &= !script[script_pi + i * 4 + 1];
            clken_perm[i] ^= script[script_pi + i * 4];
            writeouts[i] ^= script[script_pi + i * 4 + 2];
        }
        script_pi += 256 * 4;

        for i in 0..num_ios {
            let old_wo = input_state[(io_offset + i) as usize];
            let clken = clken_perm[i as usize];
            let wo = (old_wo & !clken) | (writeouts[i as usize] & clken);
            output_state[(io_offset + i) as usize] = wo;
        }

        if is_last_part != 0 {
            break;
        }
    }
    assert_eq!(script_pi, script.len());
}

fn run_reference(
    script: &FlattenedScriptV1,
    num_cycles: usize,
    states: &mut [u32],
    sram_storage: &mut [u32],
) {
    run_reference_raw(
        script.num_blocks,
        script.num_major_stages,
        &script.blocks_start,
        &script.blocks_data,
        num_cycles,
        script.reg_io_state_size as usize,
        states,
        sram_storage,
    );
}

fn run_reference_raw(
    num_blocks: usize,
    num_major_stages: usize,
    blocks_start: &[usize],
    blocks_data: &[u32],
    num_cycles: usize,
    state_size: usize,
    states: &mut [u32],
    sram_storage: &mut [u32],
) {
    assert_eq!(blocks_start.len(), num_blocks * num_major_stages + 1);
    assert_eq!(states.len(), state_size * (num_cycles + 1));

    for cycle_i in 0..num_cycles {
        let input_st = cycle_i * state_size;
        let output_st = (cycle_i + 1) * state_size;
        let (left, right) = states.split_at_mut(output_st);
        let input_state = &left[input_st..output_st];
        let output_state = &mut right[..state_size];

        for stage_i in 0..num_major_stages {
            for blk_i in 0..num_blocks {
                let script_st = blocks_start[stage_i * num_blocks + blk_i];
                let script_ed = blocks_start[stage_i * num_blocks + blk_i + 1];
                simulate_block_v1_reference(
                    &blocks_data[script_st..script_ed],
                    input_state,
                    output_state,
                    sram_storage,
                );
            }
        }
    }
}

fn fill_pattern(words: &mut [u32], salt: u32) {
    for (i, w) in words.iter_mut().enumerate() {
        let mut x = (i as u32).wrapping_mul(1664525).wrapping_add(1013904223 ^ salt);
        x ^= x.rotate_left((i % 31) as u32);
        *w = x;
    }
}

fn mix32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

#[derive(Clone, Copy)]
struct ScriptRng {
    state: u64,
}

impl ScriptRng {
    fn new(seed: u64) -> Self {
        let init = if seed == 0 { 0x9e37_79b9_7f4a_7c15 } else { seed };
        Self { state: init }
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 16) as u32
    }

    fn range_u32(&mut self, upper: u32) -> u32 {
        if upper == 0 {
            0
        } else {
            self.next_u32() % upper
        }
    }

    fn bitmask_31(&mut self) -> u32 {
        let mut mask = 0u32;
        let bit_count = self.range_u32(4);
        for _ in 0..bit_count {
            let bit = self.range_u32(31);
            mask |= 1u32 << bit;
        }
        mask
    }
}

struct RandomCudaCase {
    num_blocks: usize,
    num_major_stages: usize,
    num_cycles: usize,
    state_size: usize,
    sram_words: usize,
    blocks_start: Vec<usize>,
    blocks_data: Vec<u32>,
}

fn random_state_bit_index(rng: &mut ScriptRng) -> u32 {
    (rng.range_u32(256) << 5) | rng.range_u32(32)
}

fn random_writeout_bit_index(rng: &mut ScriptRng, num_ios: u32) -> u32 {
    let words = if num_ios == 0 { 1 } else { num_ios };
    (rng.range_u32(words) << 5) | rng.range_u32(32)
}

fn build_random_cuda_part(
    rng: &mut ScriptRng,
    state_size: usize,
    sram_words: usize,
    is_last_part: bool,
) -> Vec<u32> {
    let num_stages = rng.range_u32(3) + 1; // [1, 3]
    let num_global_read_rounds = rng.range_u32(4); // [0, 3]
    let mut num_ios = rng.range_u32(24) + 1; // [1, 24]
    if num_ios > 255 {
        num_ios = 255;
    }

    let sram_blocks_total = (sram_words / AIGPDK_SRAM_SIZE) as u32;
    let mut num_srams = rng.range_u32(3 + 1); // [0, 3]
    if num_srams > num_ios {
        num_srams = num_ios;
    }
    if num_srams > sram_blocks_total {
        num_srams = sram_blocks_total;
    }
    let dup_cap = num_ios - num_srams;
    let num_output_duplicates = if dup_cap == 0 { 0 } else { rng.range_u32(dup_cap + 1) };

    let io_room = (state_size as u32).saturating_sub(num_ios);
    let io_offset = if io_room == 0 { 0 } else { rng.range_u32(io_room + 1) };

    let sram_offset = if num_srams == 0 || sram_blocks_total == 0 {
        0
    } else {
        let max_start_block = sram_blocks_total - num_srams;
        rng.range_u32(max_start_block + 1) * (AIGPDK_SRAM_SIZE as u32)
    };

    let part_len =
        256usize + (num_global_read_rounds as usize) * 512usize + (num_stages as usize) * 5120usize + 5120usize + 5120usize;
    let mut block = vec![0u32; part_len];

    block[0] = num_stages;
    block[1] = if is_last_part { 1 } else { 0 };
    block[2] = num_ios;
    block[3] = io_offset;
    block[4] = num_srams;
    block[5] = sram_offset;
    block[6] = num_global_read_rounds;
    block[7] = num_output_duplicates;

    for i in 0..128usize {
        let hook_idx0 = i * 2;
        let hook_idx1 = hook_idx0 + 1;
        let hook0 = if hook_idx0 < (num_ios as usize) && rng.range_u32(3) != 0 {
            ((rng.range_u32(num_stages) & 0xff) << 8) | (rng.range_u32(256) & 0xff)
        } else {
            0xffffu32
        };
        let hook1 = if hook_idx1 < (num_ios as usize) && rng.range_u32(3) != 0 {
            ((rng.range_u32(num_stages) & 0xff) << 8) | (rng.range_u32(256) & 0xff)
        } else {
            0xffffu32
        };
        block[128 + i] = hook0 | (hook1 << 16);
    }

    let mut pi = 256usize;

    for _ in 0..num_global_read_rounds {
        for tid in 0..256usize {
            let mut idx = rng.range_u32(state_size as u32);
            if (rng.next_u32() & 1) != 0 {
                idx |= 1u32 << 31;
            }
            block[pi + tid * 2] = idx;
            block[pi + tid * 2 + 1] = rng.bitmask_31();
        }
        pi += 512;
    }

    for _ in 0..num_stages {
        for _ in 0..4usize {
            for tid in 0..256usize {
                for k_inner in 0..4usize {
                    let idx1 = random_state_bit_index(rng);
                    let idx2 = random_state_bit_index(rng);
                    block[pi + tid * 4 + k_inner] = (idx2 << 16) | idx1;
                }
            }
            pi += 1024;
        }

        for tid in 0..256usize {
            let seed = rng.next_u32() ^ ((tid as u32).wrapping_mul(0x9e37_79b9));
            block[pi + tid * 4] = seed;
            block[pi + tid * 4 + 1] = seed.rotate_left(7);
            block[pi + tid * 4 + 2] = seed.rotate_right(11) ^ 0x5a5a_5a5a;
            block[pi + tid * 4 + 3] = 0;
        }
        pi += 1024;
    }

    let sram_duplicate_count = (num_srams * 4 + num_output_duplicates) as usize;

    for _ in 0..4usize {
        for i in 0..256usize {
            for k_inner in 0..4usize {
                if i < sram_duplicate_count {
                    let idx1 = random_writeout_bit_index(rng, num_ios);
                    let idx2 = random_writeout_bit_index(rng, num_ios);
                    block[pi + i * 4 + k_inner] = (idx2 << 16) | idx1;
                } else {
                    block[pi + i * 4 + k_inner] = 0;
                }
            }
        }
        pi += 1024;
    }

    for i in 0..256usize {
        let base = pi + i * 4;
        if i < sram_duplicate_count {
            if i < (num_srams as usize) * 4 && (i & 3) == 0 {
                // Keep SRAM addresses in-range [0, 8191] to preserve valid script semantics.
                let raddr = rng.range_u32(AIGPDK_SRAM_SIZE as u32);
                let waddr = rng.range_u32(AIGPDK_SRAM_SIZE as u32);
                block[base] = (waddr << 16) | (raddr & 0xffff);
                block[base + 1] = 0xffff_ffff;
            } else {
                block[base] = rng.next_u32();
                block[base + 1] = rng.next_u32();
            }
        } else {
            block[base] = 0;
            block[base + 1] = 0;
        }
        block[base + 2] = 0;
        block[base + 3] = 0;
    }
    pi += 1024;

    for _ in 0..4usize {
        for i in 0..256usize {
            for k_inner in 0..4usize {
                if (i as u32) < num_ios {
                    let idx1 = random_writeout_bit_index(rng, num_ios);
                    let idx2 = random_writeout_bit_index(rng, num_ios);
                    block[pi + i * 4 + k_inner] = (idx2 << 16) | idx1;
                } else {
                    block[pi + i * 4 + k_inner] = 0;
                }
            }
        }
        pi += 1024;
    }

    for i in 0..256usize {
        let base = pi + i * 4;
        if (i as u32) < num_ios {
            block[base] = rng.next_u32();
            block[base + 1] = rng.next_u32();
            block[base + 2] = rng.next_u32();
        } else {
            block[base] = 0;
            block[base + 1] = 0;
            block[base + 2] = 0;
        }
        block[base + 3] = 0;
    }
    pi += 1024;

    assert_eq!(pi, part_len);
    block
}

fn build_random_cuda_case(seed: u64) -> RandomCudaCase {
    let mut rng = ScriptRng::new(seed);
    let num_blocks = 1usize;
    let num_major_stages = (rng.range_u32(2) + 1) as usize; // [1, 2]
    let num_cycles = (rng.range_u32(3) + 1) as usize; // [1, 3]
    let state_size = 192usize + (rng.range_u32(64) as usize);
    let sram_words = AIGPDK_SRAM_SIZE * 4;

    let mut blocks_start = Vec::with_capacity(num_blocks * num_major_stages + 1);
    let mut blocks_data = Vec::new();
    blocks_start.push(0usize);

    for _stage_i in 0..num_major_stages {
        for _block_i in 0..num_blocks {
            let num_parts = (rng.range_u32(2) + 1) as usize; // [1, 2]
            for part_i in 0..num_parts {
                let is_last_part = part_i + 1 == num_parts;
                let part = build_random_cuda_part(&mut rng, state_size, sram_words, is_last_part);
                blocks_data.extend_from_slice(&part);
            }
            blocks_start.push(blocks_data.len());
        }
    }

    RandomCudaCase {
        num_blocks,
        num_major_stages,
        num_cycles,
        state_size,
        sram_words,
        blocks_start,
        blocks_data,
    }
}

fn build_sram_duplicate_script() -> (Vec<usize>, Vec<u32>, usize) {
    // one block, one major stage, one part:
    // metadata + stage(16x+4x) + sram(16x+4x) + clken(16x+4x)
    let mut block = vec![0u32; 256 + 5120 + 5120 + 5120];

    // metadata
    block[0] = 1; // num_stages
    block[1] = 1; // is_last_part
    block[2] = 3; // num_ios
    block[3] = 0; // io_offset
    block[4] = 1; // num_srams
    block[5] = 0; // sram_offset
    block[6] = 0; // num_global_read_rounds
    block[7] = 1; // num_output_duplicates
    for i in 128..256 {
        block[i] = 0xffff_ffff; // disable all hooks by default
    }
    block[128] = 0xffff_0000; // hook[0] = stage0/state0, hook[1] disabled

    let sram_inv_offset = 9472usize;
    let clken_inv_offset = 14592usize;

    // SRAM path constants: raddr=1, waddr=2, wr_en=all1, wr_data=0xa5a55a5a
    block[sram_inv_offset] = (2u32 << 16) | 1u32;
    block[sram_inv_offset + 1] = 0xffff_ffff;
    block[sram_inv_offset + 4] = 0xffff_ffff;
    block[sram_inv_offset + 5] = 0xffff_ffff;
    block[sram_inv_offset + 8] = 0xa5a5_5a5a;
    block[sram_inv_offset + 9] = 0xffff_ffff;

    // duplicate output constant lives at slot num_srams*4 + 0 => 4
    block[sram_inv_offset + 16] = 0xdead_beef;
    block[sram_inv_offset + 17] = 0xffff_ffff;

    // clock-enable: force-write all 3 ios; no data invert
    for i in 0..3 {
        let base = clken_inv_offset + i * 4;
        block[base] = 0xffff_ffff; // clken_inv
        block[base + 1] = 0xffff_ffff; // clken_set0 (zeros perm first)
        block[base + 2] = 0; // data_inv
    }

    (vec![0, block.len()], block, AIGPDK_SRAM_SIZE)
}

fn build_hierarchy_stress_script() -> (Vec<usize>, Vec<u32>, usize) {
    let num_stages = 3usize;
    let num_ios = 16usize;
    let num_global_read_rounds = 2usize;
    let block_len = 256usize + num_global_read_rounds * 512usize + num_stages * 5120usize + 5120usize + 5120usize;
    let mut block = vec![0u32; block_len];

    block[0] = num_stages as u32;
    block[1] = 1;
    block[2] = num_ios as u32;
    block[3] = 0;
    block[4] = 0;
    block[5] = 0;
    block[6] = num_global_read_rounds as u32;
    block[7] = 0;

    for i in 128..256 {
        block[i] = 0xffff_ffff;
    }
    for i in 0..num_ios {
        let stage = (i % num_stages) as u32;
        let state_idx = mix32(0x9e37_79b9u32.wrapping_add(i as u32)) & 0xff;
        let hook = (stage << 8) | state_idx;
        let hook_slot = 128 + i / 2;
        if (i & 1) == 0 {
            block[hook_slot] = (block[hook_slot] & 0xffff_0000) | hook;
        } else {
            block[hook_slot] = (block[hook_slot] & 0x0000_ffff) | (hook << 16);
        }
    }

    let mut pi = 256usize;
    for round in 0..num_global_read_rounds {
        for tid in 0..256usize {
            let idx = (mix32(((round * 257) + tid) as u32) % (num_ios as u32)) as u32;
            let bit0 = mix32(((round * 1021) + tid) as u32) % 31;
            let bit1 = (bit0 + 11) % 31;
            let bit2 = (bit0 + 23) % 31;
            let mask = (1u32 << bit0) | (1u32 << bit1) | (1u32 << bit2);
            block[pi + tid * 2] = idx;
            block[pi + tid * 2 + 1] = mask;
        }
        pi += 512;
    }

    for stage in 0..num_stages {
        for k_outer in 0..4usize {
            for tid in 0..256usize {
                for k_inner in 0..4usize {
                    let mix = mix32(
                        (stage * 65_537 + k_outer * 4_099 + tid * 73 + k_inner * 7) as u32
                            ^ 0x1234_5678,
                    );
                    let idx1 = mix & 0x1fff;
                    let idx2 = mix.rotate_left(11) & 0x1fff;
                    block[pi + tid * 4 + k_inner] = (idx2 << 16) | idx1;
                }
            }
            pi += 1024;
        }

        for tid in 0..256usize {
            let seed = mix32((stage * 257 + tid) as u32 ^ 0xa5a5_5a5a);
            block[pi + tid * 4] = seed;
            block[pi + tid * 4 + 1] = seed.rotate_left(7);
            block[pi + tid * 4 + 2] = seed.rotate_right(5) ^ 0x3c3c_3c3c;
            block[pi + tid * 4 + 3] = 0;
        }
        pi += 1024;
    }

    pi += 5120;

    for k_outer in 0..4usize {
        for i in 0..num_ios {
            for k_inner in 0..4usize {
                let mix = mix32(
                    (k_outer * 3_571 + i * 151 + k_inner * 19) as u32 ^ 0xbeef_cafe,
                );
                let idx1 = mix % ((num_ios as u32) * 32);
                let idx2 = mix.rotate_left(9) % ((num_ios as u32) * 32);
                block[pi + i * 4 + k_inner] = (idx2 << 16) | idx1;
            }
        }
        pi += 1024;
    }
    for i in 0..num_ios {
        let base = pi + i * 4;
        block[base] = 0xffff_ffff;
        block[base + 1] = 0xffff_ffff;
        block[base + 2] = mix32(i as u32 ^ 0xfeed_face);
        block[base + 3] = 0;
    }
    pi += 1024;

    assert_eq!(pi, block_len);
    (vec![0, block_len], block, 0)
}

fn build_sram_ordering_duplicate_clken_stress_script() -> (Vec<usize>, Vec<u32>, usize, usize, u32, u32) {
    let num_stages = 1usize;
    let num_ios = 6usize;
    let num_srams = 1usize;
    let num_output_duplicates = 2usize;
    let sram_addr = 7u32;
    let write_value = 0x5566_7788u32;
    let duplicate0 = 0xdead_beefu32;
    let duplicate1 = 0xcafe_babeu32;

    let block_len = 256usize + num_stages * 5120usize + 5120usize + 5120usize;
    let mut block = vec![0u32; block_len];

    block[0] = num_stages as u32;
    block[1] = 1;
    block[2] = num_ios as u32;
    block[3] = 0;
    block[4] = num_srams as u32;
    block[5] = 0;
    block[6] = 0;
    block[7] = num_output_duplicates as u32;
    for i in 128..256 {
        block[i] = 0xffff_ffff;
    }

    let sram_inv_offset = 256usize + num_stages * 5120usize + 4096usize;
    let clken_inv_offset = 256usize + num_stages * 5120usize + 5120usize + 4096usize;

    block[sram_inv_offset] = (sram_addr << 16) | sram_addr;
    block[sram_inv_offset + 1] = 0xffff_ffff;
    block[sram_inv_offset + 4] = 0xffff_ffff;
    block[sram_inv_offset + 5] = 0xffff_ffff;
    block[sram_inv_offset + 8] = write_value;
    block[sram_inv_offset + 9] = 0xffff_ffff;
    block[sram_inv_offset + 16] = duplicate0;
    block[sram_inv_offset + 17] = 0xffff_ffff;
    block[sram_inv_offset + 20] = duplicate1;
    block[sram_inv_offset + 21] = 0xffff_ffff;

    for i in 0..num_ios {
        let base = clken_inv_offset + i * 4;
        block[base] = if i == 0 { 0 } else { 0xffff_ffff };
        block[base + 1] = 0xffff_ffff;
        block[base + 2] = if i == 3 { 0x0f0f_0f0f } else { 0 };
        block[base + 3] = 0;
    }

    (vec![0, block_len], block, AIGPDK_SRAM_SIZE, sram_addr as usize, write_value, duplicate0 ^ 0x0f0f_0f0f)
}

fn build_sram_stage_script_part(
    read_addr: u32,
    write_addr: u32,
    write_en: u32,
    write_data: u32,
    clken_enable: bool,
    is_last_part: bool,
) -> Vec<u32> {
    build_sram_stage_script_part_with_offsets(
        read_addr,
        write_addr,
        write_en,
        write_data,
        clken_enable,
        is_last_part,
        1,
        0,
        0,
    )
}

fn build_sram_stage_script_part_with_offsets(
    read_addr: u32,
    write_addr: u32,
    write_en: u32,
    write_data: u32,
    clken_enable: bool,
    is_last_part: bool,
    num_ios: u32,
    io_offset: u32,
    sram_offset: u32,
) -> Vec<u32> {
    // one part:
    // metadata (256) + boomerang section (5120) + sram section (5120) + clken section (5120)
    let mut block = vec![0u32; 256 + 5120 + 5120 + 5120];
    block[0] = 1; // num_stages
    block[1] = if is_last_part { 1 } else { 0 };
    block[2] = num_ios.max(1); // num_ios
    block[3] = io_offset; // io_offset
    block[4] = 1; // num_srams
    block[5] = sram_offset; // sram_offset
    block[6] = 0; // num_global_read_rounds
    block[7] = 0; // num_output_duplicates
    for i in 128..256 {
        block[i] = 0xffff_ffff;
    }

    let sram_inv_offset = 9472usize;
    let clken_inv_offset = 14592usize;

    // sram_duplicate_perm slot0 => addresses (read + write)
    block[sram_inv_offset] = (write_addr << 16) | (read_addr & 0xffff);
    block[sram_inv_offset + 1] = 0xffff_ffff;

    // slot1 => write enable mask
    block[sram_inv_offset + 4] = write_en;
    block[sram_inv_offset + 5] = 0xffff_ffff;

    // slot2 => write data
    block[sram_inv_offset + 8] = write_data;
    block[sram_inv_offset + 9] = 0xffff_ffff;

    // clken is force constant: 0 or all-ones independent of permutation payload.
    let clken_inv = if clken_enable { 0xffff_ffff } else { 0 };
    block[clken_inv_offset] = clken_inv;
    block[clken_inv_offset + 1] = 0xffff_ffff;
    block[clken_inv_offset + 2] = 0;

    block
}

fn build_sram_stage_script(
    read_addr: u32,
    write_addr: u32,
    write_en: u32,
    write_data: u32,
    clken_enable: bool,
) -> Vec<u32> {
    build_sram_stage_script_part(read_addr, write_addr, write_en, write_data, clken_enable, true)
}

fn build_multiblock_multistage_disjoint_script() -> (Vec<usize>, Vec<u32>, usize, u32, u32) {
    let value_a = 0x1357_9bdfu32;
    let value_b = 0x2468_ace1u32;

    // stage0/block0: write bank0 addr3
    let s00 = build_sram_stage_script_part_with_offsets(
        3,
        3,
        0xffff_ffff,
        value_a,
        false,
        true,
        1,
        0,
        0,
    );
    // stage0/block1: write bank1 addr7
    let s01 = build_sram_stage_script_part_with_offsets(
        7,
        7,
        0xffff_ffff,
        value_b,
        false,
        true,
        1,
        16,
        AIGPDK_SRAM_SIZE as u32,
    );
    // stage1/block0: read bank0 addr3 -> io_offset 0
    let s10 = build_sram_stage_script_part_with_offsets(
        3,
        3,
        0,
        0,
        true,
        true,
        1,
        0,
        0,
    );
    // stage1/block1: read bank1 addr7 -> io_offset 16
    let s11 = build_sram_stage_script_part_with_offsets(
        7,
        7,
        0,
        0,
        true,
        true,
        1,
        16,
        AIGPDK_SRAM_SIZE as u32,
    );

    let mut blocks_data = Vec::with_capacity(s00.len() + s01.len() + s10.len() + s11.len());
    blocks_data.extend_from_slice(&s00);
    let p1 = blocks_data.len();
    blocks_data.extend_from_slice(&s01);
    let p2 = blocks_data.len();
    blocks_data.extend_from_slice(&s10);
    let p3 = blocks_data.len();
    blocks_data.extend_from_slice(&s11);
    let p4 = blocks_data.len();

    // num_blocks=2, num_major_stages=2 => starts length 5
    let blocks_start = vec![0, p1, p2, p3, p4];
    (blocks_start, blocks_data, AIGPDK_SRAM_SIZE * 2, value_a, value_b)
}

fn build_multipart_sram_dependency_script() -> (Vec<usize>, Vec<u32>, usize, u32) {
    let written_value = 0x2468_ace1u32;
    let part0 = build_sram_stage_script_part(3, 5, 0xffff_ffff, written_value, false, false);
    let part1 = build_sram_stage_script_part(5, 5, 0, 0, true, true);

    let mut blocks_data = Vec::with_capacity(part0.len() + part1.len());
    blocks_data.extend_from_slice(&part0);
    blocks_data.extend_from_slice(&part1);
    let blocks_start = vec![0, blocks_data.len()];
    (blocks_start, blocks_data, AIGPDK_SRAM_SIZE, written_value)
}

fn build_multi_sram_boundary_script() -> (Vec<usize>, Vec<u32>, usize, usize, usize, u32, u32) {
    // one block, one major stage, one part with two SRAMs and high num_ios.
    let num_stages = 1usize;
    let num_ios = 12usize;
    let num_srams = 2usize;
    let num_output_duplicates = 2usize;
    let sram_blocks = 4usize;
    let sram_words = AIGPDK_SRAM_SIZE * sram_blocks;
    let sram0_addr = 15u32;
    let sram1_addr = 23u32;
    let sram0_write = 0x0bad_f00du32;
    let sram1_write = 0xface_c001u32;

    let block_len = 256usize + num_stages * 5120usize + 5120usize + 5120usize;
    let mut block = vec![0u32; block_len];

    block[0] = num_stages as u32;
    block[1] = 1;
    block[2] = num_ios as u32;
    block[3] = 0;
    block[4] = num_srams as u32;
    block[5] = AIGPDK_SRAM_SIZE as u32; // offset by one SRAM block
    block[6] = 0;
    block[7] = num_output_duplicates as u32;
    for i in 128..256 {
        block[i] = 0xffff_ffff;
    }

    let sram_inv_offset = 256usize + num_stages * 5120usize + 4096usize;
    let clken_inv_offset = 256usize + num_stages * 5120usize + 5120usize + 4096usize;

    // SRAM0 tuple
    block[sram_inv_offset] = (sram0_addr << 16) | sram0_addr;
    block[sram_inv_offset + 1] = 0xffff_ffff;
    block[sram_inv_offset + 4] = 0xffff_ffff;
    block[sram_inv_offset + 5] = 0xffff_ffff;
    block[sram_inv_offset + 8] = sram0_write;
    block[sram_inv_offset + 9] = 0xffff_ffff;

    // SRAM1 tuple (slot index 4)
    let s1 = sram_inv_offset + 16;
    block[s1] = (sram1_addr << 16) | sram1_addr;
    block[s1 + 1] = 0xffff_ffff;
    block[s1 + 4] = 0xffff_ffff;
    block[s1 + 5] = 0xffff_ffff;
    block[s1 + 8] = sram1_write;
    block[s1 + 9] = 0xffff_ffff;

    // duplicate outputs
    block[sram_inv_offset + 32] = 0x1111_2222;
    block[sram_inv_offset + 33] = 0xffff_ffff;
    block[sram_inv_offset + 36] = 0x3333_4444;
    block[sram_inv_offset + 37] = 0xffff_ffff;

    // Clken: keep io0 unchanged, enable others, and invert one duplicate output lane.
    for i in 0..num_ios {
        let base = clken_inv_offset + i * 4;
        block[base] = if i == 0 { 0 } else { 0xffff_ffff };
        block[base + 1] = 0xffff_ffff;
        block[base + 2] = if i == 8 { 0x00ff_00ff } else { 0 };
        block[base + 3] = 0;
    }

    (
        vec![0, block_len],
        block,
        sram_words,
        (AIGPDK_SRAM_SIZE + sram0_addr as usize),
        (AIGPDK_SRAM_SIZE * 2 + sram1_addr as usize),
        sram0_write,
        sram1_write,
    )
}

fn build_multistage_sram_dependency_script() -> (Vec<usize>, Vec<u32>, usize, u32) {
    let written_value = 0x1357_9bdfu32;
    let stage0 = build_sram_stage_script(0, 2, 0xffff_ffff, written_value, false);
    let stage1 = build_sram_stage_script(2, 2, 0, 0, true);

    let mut blocks_data = Vec::with_capacity(stage0.len() + stage1.len());
    blocks_data.extend_from_slice(&stage0);
    blocks_data.extend_from_slice(&stage1);
    let blocks_start = vec![0, stage0.len(), stage0.len() + stage1.len()];
    (blocks_start, blocks_data, AIGPDK_SRAM_SIZE, written_value)
}

#[test]
fn metal_matches_reference_on_tiny_script() {
    let num_blocks = 1usize;
    let num_cycles = 3usize;
    let script = build_tiny_script(num_blocks);

    let state_size = script.reg_io_state_size as usize;
    let state_words = state_size * (num_cycles + 1);

    let mut init_states = vec![0u32; state_words];
    let mut init_sram = vec![0u32; script.sram_storage_size as usize];
    fill_pattern(&mut init_states, 0x55aa_7733);
    fill_pattern(&mut init_sram, 0x127f_e210);

    let mut ref_states = init_states.clone();
    let mut ref_sram = init_sram.clone();
    run_reference(&script, num_cycles, &mut ref_states, &mut ref_sram);

    let mut gpu_states: UVec<u32> = init_states.into();
    let mut gpu_sram: UVec<u32> = init_sram.into();

    let backend = MetalBackend::new().expect("Metal backend unavailable for parity test");
    backend.simulate_v1_noninteractive_simple_scan(
        script.num_blocks,
        script.num_major_stages,
        &script.blocks_start,
        &script.blocks_data,
        &mut gpu_sram,
        num_cycles,
        state_size,
        &mut gpu_states,
    );
    let stats = backend.last_stats();
    assert_eq!(
        stats.dispatch_count,
        (num_cycles * script.num_major_stages) as u64
    );
    assert!(stats.gpu_dispatch_count > 0);
    assert!(stats.gpu_dispatch_count <= stats.dispatch_count);
    assert!(stats.total_ns >= stats.encode_ns);
    assert!(stats.total_ns >= stats.wait_ns);

    assert_eq!(&gpu_states[..], &ref_states[..]);
    assert_eq!(&gpu_sram[..], &ref_sram[..]);
}

#[test]
fn metal_matches_reference_on_manifest_baseline_corpus() {
    let cases = parse_baseline_manifest_cases();
    let backend = MetalBackend::new().expect("Metal backend unavailable for parity test");

    for (idx, case) in cases.iter().enumerate() {
        let script = build_script_from_artifacts(
            &case.netlist_verilog,
            &case.gemparts,
            case.num_blocks,
            &case.level_split,
        );
        let num_cycles = 3usize;
        let state_size = script.reg_io_state_size as usize;
        let state_words = state_size * (num_cycles + 1);

        let mut init_states = vec![0u32; state_words];
        let mut init_sram = vec![0u32; script.sram_storage_size as usize];
        fill_pattern(
            &mut init_states,
            0x73a5_1c9d ^ ((idx as u32).wrapping_mul(0x9e37_79b9)),
        );
        fill_pattern(
            &mut init_sram,
            0x1f20_3d4e ^ ((idx as u32).wrapping_mul(0x85eb_ca6b)),
        );

        let mut ref_states = init_states.clone();
        let mut ref_sram = init_sram.clone();
        run_reference(&script, num_cycles, &mut ref_states, &mut ref_sram);

        let mut gpu_states: UVec<u32> = init_states.into();
        let mut gpu_sram: UVec<u32> = init_sram.into();

        backend.simulate_v1_noninteractive_simple_scan(
            script.num_blocks,
            script.num_major_stages,
            &script.blocks_start,
            &script.blocks_data,
            &mut gpu_sram,
            num_cycles,
            state_size,
            &mut gpu_states,
        );
        let stats = backend.last_stats();
        assert_eq!(
            stats.dispatch_count,
            (num_cycles * script.num_major_stages) as u64,
            "dispatch_count mismatch for baseline entry {}",
            case.name
        );
        assert!(stats.gpu_dispatch_count > 0);
        assert!(stats.gpu_dispatch_count <= stats.dispatch_count);
        assert!(stats.total_ns >= stats.encode_ns);
        assert!(stats.total_ns >= stats.wait_ns);

        assert_eq!(
            &gpu_states[..],
            &ref_states[..],
            "state parity mismatch for baseline entry {}",
            case.name
        );
        assert_eq!(
            &gpu_sram[..],
            &ref_sram[..],
            "sram parity mismatch for baseline entry {}",
            case.name
        );
    }
}

#[test]
fn metal_matches_reference_on_sram_duplicate_case() {
    let (blocks_start, blocks_data, sram_size) = build_sram_duplicate_script();
    let num_blocks = 1usize;
    let num_major_stages = 1usize;
    let num_cycles = 1usize;
    let state_size = 8usize;

    let mut init_states = vec![0u32; state_size * (num_cycles + 1)];
    let mut init_sram = vec![0u32; sram_size];
    fill_pattern(&mut init_states, 0x33aa_7788);
    init_states[0] = 0x1111_1111;
    init_states[1] = 0x2222_2222;
    init_states[2] = 0x3333_3333;
    init_sram[1] = 0x1122_3344;
    init_sram[2] = 0x5566_7788;

    let mut ref_states = init_states.clone();
    let mut ref_sram = init_sram.clone();
    run_reference_raw(
        num_blocks,
        num_major_stages,
        &blocks_start,
        &blocks_data,
        num_cycles,
        state_size,
        &mut ref_states,
        &mut ref_sram,
    );

    let mut gpu_states: UVec<u32> = init_states.into();
    let mut gpu_sram: UVec<u32> = init_sram.into();
    let blocks_start_uvec: UVec<usize> = blocks_start.into();
    let blocks_data_uvec: UVec<u32> = blocks_data.into();

    let backend = MetalBackend::new().expect("Metal backend unavailable for parity test");
    backend.simulate_v1_noninteractive_simple_scan(
        num_blocks,
        num_major_stages,
        &blocks_start_uvec,
        &blocks_data_uvec,
        &mut gpu_sram,
        num_cycles,
        state_size,
        &mut gpu_states,
    );
    let stats = backend.last_stats();
    assert_eq!(stats.dispatch_count, (num_cycles * num_major_stages) as u64);
    assert!(stats.gpu_dispatch_count > 0);
    assert!(stats.gpu_dispatch_count <= stats.dispatch_count);
    assert!(stats.total_ns >= stats.encode_ns);
    assert!(stats.total_ns >= stats.wait_ns);

    assert_eq!(&gpu_states[..], &ref_states[..]);
    assert_eq!(&gpu_sram[..], &ref_sram[..]);
    assert_eq!(gpu_states[state_size], 0);
    assert_eq!(gpu_states[state_size + 1], 0xdead_beef);
    assert_eq!(gpu_states[state_size + 2], 0x1122_3344);
    assert_eq!(gpu_sram[2], 0xa5a5_5a5a);
}

#[test]
fn metal_matches_reference_on_multistage_sram_dependency_case() {
    let (blocks_start, blocks_data, sram_size, written_value) =
        build_multistage_sram_dependency_script();
    let num_blocks = 1usize;
    let num_major_stages = 2usize;
    let num_cycles = 2usize;
    let state_size = 8usize;

    let mut init_states = vec![0u32; state_size * (num_cycles + 1)];
    let mut init_sram = vec![0u32; sram_size];
    fill_pattern(&mut init_states, 0x44bb_7788);
    fill_pattern(&mut init_sram, 0x1234_5678);
    init_sram[2] = 0xaaaa_0000;

    let mut ref_states = init_states.clone();
    let mut ref_sram = init_sram.clone();
    run_reference_raw(
        num_blocks,
        num_major_stages,
        &blocks_start,
        &blocks_data,
        num_cycles,
        state_size,
        &mut ref_states,
        &mut ref_sram,
    );

    let mut gpu_states: UVec<u32> = init_states.into();
    let mut gpu_sram: UVec<u32> = init_sram.into();
    let blocks_start_uvec: UVec<usize> = blocks_start.into();
    let blocks_data_uvec: UVec<u32> = blocks_data.into();

    let backend = MetalBackend::new().expect("Metal backend unavailable for parity test");
    backend.simulate_v1_noninteractive_simple_scan(
        num_blocks,
        num_major_stages,
        &blocks_start_uvec,
        &blocks_data_uvec,
        &mut gpu_sram,
        num_cycles,
        state_size,
        &mut gpu_states,
    );
    let stats = backend.last_stats();
    assert_eq!(stats.dispatch_count, (num_cycles * num_major_stages) as u64);
    assert!(stats.gpu_dispatch_count > 0);
    assert!(stats.gpu_dispatch_count <= stats.dispatch_count);
    assert!(stats.total_ns >= stats.encode_ns);
    assert!(stats.total_ns >= stats.wait_ns);

    assert_eq!(&gpu_states[..], &ref_states[..]);
    assert_eq!(&gpu_sram[..], &ref_sram[..]);
    assert_eq!(gpu_states[state_size], written_value);
    assert_eq!(gpu_states[state_size * 2], written_value);
    assert_eq!(gpu_sram[2], written_value);
}

#[test]
fn metal_matches_reference_on_hierarchy_stress_case() {
    let (blocks_start, blocks_data, sram_size) = build_hierarchy_stress_script();
    let num_blocks = 1usize;
    let num_major_stages = 1usize;
    let num_cycles = 4usize;
    let state_size = 64usize;

    let mut init_states = vec![0u32; state_size * (num_cycles + 1)];
    let init_sram = vec![0u32; sram_size];
    fill_pattern(&mut init_states, 0x66cc_11aa);

    let mut ref_states = init_states.clone();
    let mut ref_sram = init_sram.clone();
    run_reference_raw(
        num_blocks,
        num_major_stages,
        &blocks_start,
        &blocks_data,
        num_cycles,
        state_size,
        &mut ref_states,
        &mut ref_sram,
    );

    let mut gpu_states: UVec<u32> = init_states.into();
    let mut gpu_sram: UVec<u32> = init_sram.into();
    let blocks_start_uvec: UVec<usize> = blocks_start.into();
    let blocks_data_uvec: UVec<u32> = blocks_data.into();

    let backend = MetalBackend::new().expect("Metal backend unavailable for parity test");
    backend.simulate_v1_noninteractive_simple_scan(
        num_blocks,
        num_major_stages,
        &blocks_start_uvec,
        &blocks_data_uvec,
        &mut gpu_sram,
        num_cycles,
        state_size,
        &mut gpu_states,
    );

    assert_eq!(&gpu_states[..], &ref_states[..]);
    assert_eq!(&gpu_sram[..], &ref_sram[..]);
}

#[test]
fn metal_matches_reference_on_multicycle_sram_duplicate_clken_stress_case() {
    let (blocks_start, blocks_data, sram_size, sram_addr, write_value, expected_dup0_cycle1) =
        build_sram_ordering_duplicate_clken_stress_script();
    let num_blocks = 1usize;
    let num_major_stages = 1usize;
    let num_cycles = 3usize;
    let state_size = 12usize;

    let mut init_states = vec![0u32; state_size * (num_cycles + 1)];
    let mut init_sram = vec![0u32; sram_size];
    fill_pattern(&mut init_states, 0x1188_22ff);
    init_sram[sram_addr] = 0x1122_3344;
    let held_output0 = init_states[0];

    let mut ref_states = init_states.clone();
    let mut ref_sram = init_sram.clone();
    run_reference_raw(
        num_blocks,
        num_major_stages,
        &blocks_start,
        &blocks_data,
        num_cycles,
        state_size,
        &mut ref_states,
        &mut ref_sram,
    );

    let mut gpu_states: UVec<u32> = init_states.into();
    let mut gpu_sram: UVec<u32> = init_sram.into();
    let blocks_start_uvec: UVec<usize> = blocks_start.into();
    let blocks_data_uvec: UVec<u32> = blocks_data.into();

    let backend = MetalBackend::new().expect("Metal backend unavailable for parity test");
    backend.simulate_v1_noninteractive_simple_scan(
        num_blocks,
        num_major_stages,
        &blocks_start_uvec,
        &blocks_data_uvec,
        &mut gpu_sram,
        num_cycles,
        state_size,
        &mut gpu_states,
    );

    assert_eq!(&gpu_states[..], &ref_states[..]);
    assert_eq!(&gpu_sram[..], &ref_sram[..]);
    assert_eq!(gpu_states[state_size], held_output0);
    assert_eq!(gpu_states[state_size + 3], expected_dup0_cycle1);
    assert_eq!(gpu_states[state_size + 5], 0x1122_3344);
    assert_eq!(gpu_states[state_size * 2 + 5], write_value);
    assert_eq!(gpu_sram[sram_addr], write_value);
}

#[test]
fn metal_matches_reference_on_multipart_sram_dependency_case() {
    let (blocks_start, blocks_data, sram_size, written_value) = build_multipart_sram_dependency_script();
    let num_blocks = 1usize;
    let num_major_stages = 1usize;
    let num_cycles = 2usize;
    let state_size = 16usize;

    let mut init_states = vec![0u32; state_size * (num_cycles + 1)];
    let mut init_sram = vec![0u32; sram_size];
    fill_pattern(&mut init_states, 0x5aa5_21f0);
    fill_pattern(&mut init_sram, 0x7788_99aa);
    init_sram[5] = 0x0102_0304;

    let mut ref_states = init_states.clone();
    let mut ref_sram = init_sram.clone();
    run_reference_raw(
        num_blocks,
        num_major_stages,
        &blocks_start,
        &blocks_data,
        num_cycles,
        state_size,
        &mut ref_states,
        &mut ref_sram,
    );

    let mut gpu_states: UVec<u32> = init_states.into();
    let mut gpu_sram: UVec<u32> = init_sram.into();
    let blocks_start_uvec: UVec<usize> = blocks_start.into();
    let blocks_data_uvec: UVec<u32> = blocks_data.into();

    let backend = MetalBackend::new().expect("Metal backend unavailable for parity test");
    backend.simulate_v1_noninteractive_simple_scan(
        num_blocks,
        num_major_stages,
        &blocks_start_uvec,
        &blocks_data_uvec,
        &mut gpu_sram,
        num_cycles,
        state_size,
        &mut gpu_states,
    );

    assert_eq!(&gpu_states[..], &ref_states[..]);
    assert_eq!(&gpu_sram[..], &ref_sram[..]);
    assert_eq!(gpu_states[state_size], written_value);
    assert_eq!(gpu_states[state_size * 2], written_value);
    assert_eq!(gpu_sram[5], written_value);
}

#[test]
fn metal_matches_reference_on_multi_sram_boundary_case() {
    let (blocks_start, blocks_data, sram_size, sram0_word, sram1_word, sram0_write, sram1_write) =
        build_multi_sram_boundary_script();
    let num_blocks = 1usize;
    let num_major_stages = 1usize;
    let num_cycles = 2usize;
    let state_size = 32usize;

    let mut init_states = vec![0u32; state_size * (num_cycles + 1)];
    let mut init_sram = vec![0u32; sram_size];
    fill_pattern(&mut init_states, 0x2121_4545);
    fill_pattern(&mut init_sram, 0x8383_1717);
    init_sram[sram0_word] = 0xaaaa_0001;
    init_sram[sram1_word] = 0xbbbb_0002;

    let mut ref_states = init_states.clone();
    let mut ref_sram = init_sram.clone();
    run_reference_raw(
        num_blocks,
        num_major_stages,
        &blocks_start,
        &blocks_data,
        num_cycles,
        state_size,
        &mut ref_states,
        &mut ref_sram,
    );

    let mut gpu_states: UVec<u32> = init_states.into();
    let mut gpu_sram: UVec<u32> = init_sram.into();
    let blocks_start_uvec: UVec<usize> = blocks_start.into();
    let blocks_data_uvec: UVec<u32> = blocks_data.into();

    let backend = MetalBackend::new().expect("Metal backend unavailable for parity test");
    backend.simulate_v1_noninteractive_simple_scan(
        num_blocks,
        num_major_stages,
        &blocks_start_uvec,
        &blocks_data_uvec,
        &mut gpu_sram,
        num_cycles,
        state_size,
        &mut gpu_states,
    );

    assert_eq!(&gpu_states[..], &ref_states[..]);
    assert_eq!(&gpu_sram[..], &ref_sram[..]);
    assert_eq!(gpu_sram[sram0_word], sram0_write);
    assert_eq!(gpu_sram[sram1_word], sram1_write);
}

#[test]
fn metal_matches_reference_on_multiblock_multistage_disjoint_case() {
    let (blocks_start, blocks_data, sram_size, value_a, value_b) =
        build_multiblock_multistage_disjoint_script();
    let num_blocks = 2usize;
    let num_major_stages = 2usize;
    let num_cycles = 2usize;
    let state_size = 48usize;

    let mut init_states = vec![0u32; state_size * (num_cycles + 1)];
    let mut init_sram = vec![0u32; sram_size];
    fill_pattern(&mut init_states, 0x4242_1919);
    fill_pattern(&mut init_sram, 0x7676_2828);
    init_sram[3] = 0x1111_0001;
    init_sram[AIGPDK_SRAM_SIZE + 7] = 0x2222_0002;

    let mut ref_states = init_states.clone();
    let mut ref_sram = init_sram.clone();
    run_reference_raw(
        num_blocks,
        num_major_stages,
        &blocks_start,
        &blocks_data,
        num_cycles,
        state_size,
        &mut ref_states,
        &mut ref_sram,
    );

    let mut gpu_states: UVec<u32> = init_states.into();
    let mut gpu_sram: UVec<u32> = init_sram.into();
    let blocks_start_uvec: UVec<usize> = blocks_start.into();
    let blocks_data_uvec: UVec<u32> = blocks_data.into();

    let backend = MetalBackend::new().expect("Metal backend unavailable for parity test");
    backend.simulate_v1_noninteractive_simple_scan(
        num_blocks,
        num_major_stages,
        &blocks_start_uvec,
        &blocks_data_uvec,
        &mut gpu_sram,
        num_cycles,
        state_size,
        &mut gpu_states,
    );

    assert_eq!(&gpu_states[..], &ref_states[..]);
    assert_eq!(&gpu_sram[..], &ref_sram[..]);
    assert_eq!(gpu_states[state_size], value_a);
    assert_eq!(gpu_states[state_size + 16], value_b);
    assert_eq!(gpu_states[state_size * 2], value_a);
    assert_eq!(gpu_states[state_size * 2 + 16], value_b);
    assert_eq!(gpu_sram[3], value_a);
    assert_eq!(gpu_sram[AIGPDK_SRAM_SIZE + 7], value_b);
}

#[test]
fn metal_matches_reference_on_randomized_cuda_shape_cases() {
    let backend = MetalBackend::new().expect("Metal backend unavailable for parity test");
    let seeds: [u64; 12] = [
        0x0000_0000_0000_1001,
        0x0000_0000_0000_2003,
        0x0000_0000_0000_3007,
        0x0000_0000_0000_400d,
        0x0000_0000_0000_5011,
        0x0000_0000_0000_601f,
        0x0000_0000_0000_702b,
        0x0000_0000_0000_8039,
        0x0000_0000_0000_9043,
        0x0000_0000_0000_a04f,
        0x0000_0000_0000_b05d,
        0x0000_0000_0000_c06b,
    ];

    for (case_idx, seed) in seeds.iter().enumerate() {
        let case = build_random_cuda_case(*seed);
        let mut init_states = vec![0u32; case.state_size * (case.num_cycles + 1)];
        let mut init_sram = vec![0u32; case.sram_words];
        fill_pattern(&mut init_states, (*seed as u32) ^ 0x5a5a_33cc);
        fill_pattern(&mut init_sram, ((*seed >> 32) as u32) ^ 0x33cc_5a5a);

        let mut ref_states = init_states.clone();
        let mut ref_sram = init_sram.clone();
        run_reference_raw(
            case.num_blocks,
            case.num_major_stages,
            &case.blocks_start,
            &case.blocks_data,
            case.num_cycles,
            case.state_size,
            &mut ref_states,
            &mut ref_sram,
        );

        let mut gpu_states: UVec<u32> = init_states.into();
        let mut gpu_sram: UVec<u32> = init_sram.into();
        let blocks_start_uvec: UVec<usize> = case.blocks_start.clone().into();
        let blocks_data_uvec: UVec<u32> = case.blocks_data.clone().into();

        backend.simulate_v1_noninteractive_simple_scan(
            case.num_blocks,
            case.num_major_stages,
            &blocks_start_uvec,
            &blocks_data_uvec,
            &mut gpu_sram,
            case.num_cycles,
            case.state_size,
            &mut gpu_states,
        );

        if &gpu_states[..] != &ref_states[..] {
            let first_diff = gpu_states
                .iter()
                .zip(ref_states.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            panic!(
                "state mismatch for randomized CUDA-shape case {} seed=0x{:016x} first_diff={} gpu={} ref={} num_blocks={} num_major_stages={} num_cycles={} state_size={}",
                case_idx,
                seed,
                first_diff,
                gpu_states[first_diff],
                ref_states[first_diff],
                case.num_blocks,
                case.num_major_stages,
                case.num_cycles,
                case.state_size
            );
        }
        if &gpu_sram[..] != &ref_sram[..] {
            let first_diff = gpu_sram
                .iter()
                .zip(ref_sram.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            panic!(
                "sram mismatch for randomized CUDA-shape case {} seed=0x{:016x} first_diff={} gpu={} ref={}",
                case_idx,
                seed,
                first_diff,
                gpu_sram[first_diff],
                ref_sram[first_diff]
            );
        }
    }
}
