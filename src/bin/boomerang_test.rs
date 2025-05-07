// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use std::path::PathBuf;
use gem::aigpdk::AIGPDKLeafPins;
use gem::aig::AIG;
use gem::staging::build_staged_aigs;
use gem::pe::process_partitions_from_hgr_parts_file;
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
    /// Level split thresholds.
    #[clap(long, value_delimiter=',')]
    level_split: Vec<usize>,
    /// Input path for the partition result.
    parts_dir: PathBuf,
    #[clap(long, value_delimiter=',')]
    parts_suffixes: Vec<usize>,
    /// Output path for the serialized partitions.
    parts_out: PathBuf,
    /// The maximum allowance of layers for merging-induced degradations.
    ///
    /// By default is 0, meaning no degradation is allowed.
    #[clap(long, default_value_t=0)]
    max_stage_degrad: usize,
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

    let stageds = build_staged_aigs(&aig, &args.level_split);

    assert_eq!(stageds.len(), args.parts_suffixes.len(), "incorrect number of parts suffixes given");

    let stages_effective_parts = stageds.iter().zip(args.parts_suffixes.iter()).map(|(&(l, r, ref staged), &suffix)| {
        let filename = format!("{}.stage.{}-{}.hgr.part.{}", netlistdb.name, l, match r {
            usize::MAX => "max".to_string(),
            r @ _ => format!("{}", r)
        }, suffix);
        let effective_parts = process_partitions_from_hgr_parts_file(
            &aig, staged, &args.parts_dir.join(&filename),
            args.max_stage_degrad,
        ).expect("some partition failed to map. please increase granularity.");

        clilog::info!("# of effective partitions in {}: {}", filename, effective_parts.len());
        effective_parts
    }).collect::<Vec<_>>();

    let f = std::fs::File::create(&args.parts_out).unwrap();
    let mut buf = std::io::BufWriter::new(f);
    serde_bare::to_writer(&mut buf, &stages_effective_parts).unwrap();
}
