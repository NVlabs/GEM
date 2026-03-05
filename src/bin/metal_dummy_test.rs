// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! this binary mirrors cuda_dummy_test flow but targets Metal backend.

use gem::aig::{AIG, DriverType};
use gem::aigpdk::AIGPDKLeafPins;
use gem::flatten::FlattenedScriptV1;
use gem::gpu::backend::GpuBackendV1;
use gem::gpu::metal_backend::MetalBackend;
use gem::pe::Partition;
use gem::staging::build_staged_aigs;
use netlistdb::NetlistDB;
use std::path::PathBuf;
use ulib::{Device, UVec};

#[derive(clap::Parser, Debug)]
struct SimulatorArgs {
    /// Gate-level verilog path synthesized in our provided library.
    netlist_verilog: PathBuf,
    /// Top module type in netlist to analyze.
    #[clap(long)]
    top_module: Option<String>,
    /// Level split thresholds.
    #[clap(long, value_delimiter = ',')]
    level_split: Vec<usize>,
    /// Input path for the serialized partitions.
    gemparts: PathBuf,
    /// The number of blocks to map and execute with.
    num_blocks: usize,
    /// The number of dummy cycles to execute.
    num_dummy_cycles: usize,
}

fn main() {
    clilog::init_stderr_color_debug();
    clilog::enable_timer("metal_dummy_test");
    clilog::enable_timer("gem");
    clilog::set_max_print_count(clilog::Level::Warn, "NL_SV_LIT", 1);
    let args = <SimulatorArgs as clap::Parser>::parse();
    clilog::info!("Simulator args:\n{:#?}", args);

    let backend = match MetalBackend::new() {
        Ok(backend) => backend,
        Err(err) => {
            eprintln!("metal_dummy_test: {err}");
            std::process::exit(2);
        }
    };
    println!(
        "metal_dummy_test: compiled metallib at {}",
        backend.metallib_path().display()
    );

    let netlistdb = NetlistDB::from_sverilog_file(
        &args.netlist_verilog,
        args.top_module.as_deref(),
        &AIGPDKLeafPins(),
    )
    .expect("cannot build netlist");

    let aig = AIG::from_netlistdb(&netlistdb);

    let order = aig.topo_traverse_generic(None, None);
    let mut level_id = vec![0; aig.num_aigpins + 1];
    for &i in &order {
        if let DriverType::AndGate(a, b) = aig.drivers[i] {
            if a >= 2 {
                level_id[i] = level_id[i].max(level_id[a >> 1] + 1);
            }
            if b >= 2 {
                level_id[i] = level_id[i].max(level_id[b >> 1] + 1);
            }
        }
    }
    let max_level = level_id.iter().copied().max().unwrap();
    println!(
        "netlist has {} pins, {} aig pins, {} and gates",
        netlistdb.num_pins,
        aig.num_aigpins,
        aig.and_gate_cache.len()
    );
    println!("netlist logic depth: {}", max_level);

    let stageds = build_staged_aigs(&aig, &args.level_split);

    let f = std::fs::File::open(&args.gemparts).unwrap();
    let mut buf = std::io::BufReader::new(f);
    let parts_in_stages: Vec<Vec<Partition>> = serde_bare::from_reader(&mut buf).unwrap();
    clilog::info!(
        "# of effective partitions in each stage: {:?}",
        parts_in_stages.iter().map(|ps| ps.len()).collect::<Vec<_>>()
    );

    let mut input_layout = Vec::new();
    for (i, driv) in aig.drivers.iter().enumerate() {
        if let DriverType::InputPort(_) | DriverType::InputClockFlag(_, _) = driv {
            input_layout.push(i);
        }
    }

    let script = FlattenedScriptV1::from(
        &aig,
        &stageds
            .iter()
            .map(|(_, _, staged)| staged)
            .collect::<Vec<_>>(),
        &parts_in_stages
            .iter()
            .map(|ps| ps.as_slice())
            .collect::<Vec<_>>(),
        args.num_blocks,
        input_layout,
    );

    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut s = DefaultHasher::new();
    script.blocks_data.hash(&mut s);
    println!("Script hash: {}", s.finish());

    clilog::info!("total number of cycles: {}", args.num_dummy_cycles);
    let device = Device::CPU;
    let mut input_states_uvec = UVec::new_zeroed(
        script.reg_io_state_size as usize * (args.num_dummy_cycles + 1),
        device,
    );
    let mut sram_storage = UVec::new_zeroed(script.sram_storage_size as usize, device);
    backend.synchronize();

    backend.simulate_v1_noninteractive_simple_scan(
        args.num_blocks,
        script.num_major_stages,
        &script.blocks_start,
        &script.blocks_data,
        &mut sram_storage,
        args.num_dummy_cycles,
        script.reg_io_state_size as usize,
        &mut input_states_uvec,
    );

    let stats = backend.last_stats();
    if stats.total_ns > 0 {
        let cycles_per_sec = args.num_dummy_cycles as f64 * 1_000_000_000.0 / stats.total_ns as f64;
        println!(
            "metal_dummy_test: logical_dispatches={} gpu_dispatches={} encode_ms={:.3} wait_ms={:.3} total_ms={:.3} cycles_per_sec={:.2}",
            stats.dispatch_count,
            stats.gpu_dispatch_count,
            stats.encode_ns as f64 / 1_000_000.0,
            stats.wait_ns as f64 / 1_000_000.0,
            stats.total_ns as f64 / 1_000_000.0,
            cycles_per_sec
        );
    }
}
