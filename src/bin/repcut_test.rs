// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use std::path::PathBuf;
use gem::aigpdk::AIGPDKLeafPins;
use gem::aig::AIG;
use gem::repcut::RCHyperGraph;
use gem::staging::build_staged_aigs;
use netlistdb::NetlistDB;
use std::io::Write;
use std::fs;

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
    /// Level split thresholds.
    #[clap(long, value_delimiter=',')]
    level_split: Vec<usize>,
    /// Output directory for hypergraph files.
    hgr_output_dir: PathBuf,
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
    println!("netlist has {} pins, {} aig pins, {} and gates",
             netlistdb.num_pins, aig.num_aigpins, aig.and_gate_cache.len());

    let stageds = build_staged_aigs(&aig, &args.level_split);

    if !args.hgr_output_dir.exists() {
        fs::create_dir_all(&args.hgr_output_dir).unwrap();
    }
    for &(l, r, ref staged) in &stageds {
        let hg = RCHyperGraph::from_staged_aig(&aig, staged);

        let filename = format!("{}.stage.{}-{}.hgr", netlistdb.name, l, match r {
            usize::MAX => "max".to_string(),
            r @ _ => format!("{}", r)
        });
        println!("writing {}", filename);
        let path = args.hgr_output_dir.join(filename);

        let f = std::fs::File::create(&path).unwrap();
        let mut buf = std::io::BufWriter::new(f);
        write!(buf, "{}", hg).unwrap();
    }
}
