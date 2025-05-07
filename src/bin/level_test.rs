// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use std::path::PathBuf;
use gem::aigpdk::AIGPDKLeafPins;
use gem::aig::{AIG, DriverType};
use netlistdb::NetlistDB;

#[derive(clap::Parser, Debug)]
struct SimulatorArgs {
    /// Gate-level verilog path synthesized in our provided library.
    ///
    /// If your design is still at RTL level, you should synthesize it
    /// in yosys first.
    netlist_verilog: PathBuf,
    /// Top module type in netlist to analyze.
    ///
    /// If not specified, we will guess it from the hierarchy.
    #[clap(long)]
    top_module: Option<String>,
}

fn main() {
    clilog::init_stderr_color_debug();
    clilog::set_max_print_count(clilog::Level::Warn, "NL_SV_LIT", 1);
    let args = <SimulatorArgs as clap::Parser>::parse();
    clilog::info!("Simulator args:\n{:#?}", args);

    let netlistdb = NetlistDB::from_sverilog_file(
        &args.netlist_verilog,
        args.top_module.as_deref(),
        &AIGPDKLeafPins()
    ).expect("cannot build netlist");

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
    let mut num_nodes_in_level = vec![0; max_level + 1];
    for &i in &order {
        num_nodes_in_level[level_id[i]] += 1;
    }

    println!("Number of levels: {}", max_level);
    for (i, &num_lvlnd) in num_nodes_in_level.iter().enumerate() {
        print!("[{i}]: {num_lvlnd},  ");
        if i % 6 == 5 {
            println!();
        }
    }
    println!();
}
