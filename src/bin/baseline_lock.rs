// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Deterministic baseline harness for flatten-script output.

use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use gem::aig::{AIG, DriverType};
use gem::aigpdk::AIGPDKLeafPins;
use gem::flatten::FlattenedScriptV1;
use gem::pe::Partition;
use gem::staging::build_staged_aigs;
use netlistdb::NetlistDB;

#[derive(clap::Parser, Debug)]
struct BaselineArgs {
    /// Gate-level verilog path synthesized in AIGPDK.
    netlist_verilog: PathBuf,
    /// Input path for serialized partitions.
    gemparts: PathBuf,
    /// Number of blocks used during flattening.
    num_blocks: usize,
    /// Top module type in netlist to analyze.
    #[clap(long)]
    top_module: Option<String>,
    /// Level split thresholds.
    #[clap(long, value_delimiter = ',')]
    level_split: Vec<usize>,
    /// Expected script hash (decimal). If provided, mismatch is an error.
    #[clap(long)]
    expected_script_hash: Option<u64>,
}

fn hash_of<T: Hash>(x: &T) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    x.hash(&mut hasher);
    hasher.finish()
}

fn main() {
    clilog::init_stderr_color_debug();
    clilog::set_max_print_count(clilog::Level::Warn, "NL_SV_LIT", 1);
    let args = <BaselineArgs as clap::Parser>::parse();
    clilog::info!("Baseline args:\n{:#?}", args);

    let netlistdb = NetlistDB::from_sverilog_file(
        &args.netlist_verilog,
        args.top_module.as_deref(),
        &AIGPDKLeafPins(),
    ).expect("cannot build netlist");
    let aig = AIG::from_netlistdb(&netlistdb);
    let stageds = build_staged_aigs(&aig, &args.level_split);

    let f = std::fs::File::open(&args.gemparts).expect("cannot open gemparts");
    let mut buf = std::io::BufReader::new(f);
    let parts_in_stages: Vec<Vec<Partition>> =
        serde_bare::from_reader(&mut buf).expect("cannot decode gemparts");

    let mut input_layout = Vec::new();
    for (i, driv) in aig.drivers.iter().enumerate() {
        if let DriverType::InputPort(_) | DriverType::InputClockFlag(_, _) = driv {
            input_layout.push(i);
        }
    }

    let script = FlattenedScriptV1::from(
        &aig,
        &stageds.iter().map(|(_, _, staged)| staged).collect::<Vec<_>>(),
        &parts_in_stages
            .iter()
            .map(|ps| ps.as_slice())
            .collect::<Vec<_>>(),
        args.num_blocks,
        input_layout,
    );

    let script_hash = hash_of(&script.blocks_data);
    println!("baseline.script_hash={script_hash}");
    println!(
        "baseline.summary=blocks:{} major_stages:{} reg_io_state_size:{} sram_storage_size:{} script_words:{}",
        script.num_blocks,
        script.num_major_stages,
        script.reg_io_state_size,
        script.sram_storage_size,
        script.blocks_data.len(),
    );

    if let Some(expected) = args.expected_script_hash {
        if expected != script_hash {
            eprintln!(
                "baseline mismatch: expected script hash {}, got {}",
                expected, script_hash
            );
            std::process::exit(2);
        }
    }
}
