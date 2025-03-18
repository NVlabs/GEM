//! This is an experimental interactive cutting-then-mapping
//! implementation.
//!
//! The key idea is to only repartition the endpoint groups that
//! are unable to be mapped.

use std::path::{Path, PathBuf};
use gem::repcut::RCHyperGraph;
use gem::aigpdk::AIGPDKLeafPins;
use gem::aig::AIG;
use gem::staging::build_staged_aigs;
use gem::pe::{process_partitions, Partition};
use netlistdb::NetlistDB;
use rayon::prelude::*;

/// Call an external hypergraph partitioner
fn run_par(hmetis_bin: &Path, hg: &RCHyperGraph, num_parts: usize) -> Vec<Vec<usize>> {
    clilog::debug!("invoking partitioner (#parts {})", num_parts);
    use std::io::{BufRead, BufReader, BufWriter, Write};
    use std::fs::File;

    let tmp_dir = tempdir::TempDir::new("gemtemp").unwrap();
    std::fs::create_dir_all(tmp_dir.path()).unwrap();
    let hgr_path = tmp_dir.path().join("graph.hgr");
    println!("hgr_path: {}", hgr_path.display());
    let f = File::create(&hgr_path).unwrap();
    let mut buf = BufWriter::new(f);
    write!(buf, "{}", hg).unwrap();
    buf.into_inner().unwrap().sync_all().unwrap();

    std::process::Command::new(hmetis_bin)
        .arg(&hgr_path)
        .arg(format!("{}", num_parts))
        .spawn()
        .expect("hmetis failed!")
        .wait().unwrap();

    let path_parts = tmp_dir.path()
        .join(format!("graph.hgr.part.{}", num_parts));
    let mut parts = Vec::<Vec<usize>>::new();
    let f_parts = File::open(&path_parts).unwrap();
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
                  path_parts.display(), parts.len());
    parts
}

#[derive(clap::Parser, Debug)]
struct SimulatorArgs {
    /// Path to hmetis (or compatible partitioner) binary.
    /// We will launch it with `/path/to/binary graph.hgr NUM_PARTS` and
    /// expect a partition result with file name `graph.hgr.part.NUM_PARTS`.
    ///
    /// E.g.: `"/path/to/hmetis-2.0pre1/Linux-x86_64/hmetis2.0pre1"`
    hmetis_bin: PathBuf,
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
    println!("netlist has {} pins, {} aig pins, {} and gates",
             netlistdb.num_pins, aig.num_aigpins, aig.and_gate_cache.len());

    let stageds = build_staged_aigs(&aig, &args.level_split);

    let stages_effective_parts = stageds.iter().map(|&(l, r, ref staged)| {
        clilog::info!("interactive partitioning stage {}-{}", l, match r {
            usize::MAX => "max".to_string(),
            r @ _ => format!("{}", r)
        });

        let mut parts_indices_good = Vec::new();
        // always made sure that staged output pins are at fronts.
        let mut unrealized_endpoints = (0..staged.num_endpoint_groups()).collect::<Vec<_>>();
        let mut division = 600;

        while !unrealized_endpoints.is_empty() {
            division = (division / 2).max(1);
            let num_parts = (unrealized_endpoints.len() + division - 1) / division;
            clilog::info!("current: {} endpoints, try {} parts", unrealized_endpoints.len(), num_parts);
            let staged_ur = staged.to_endpoint_subset(&unrealized_endpoints);
            let hg_ur = RCHyperGraph::from_staged_aig(&aig, &staged_ur);
            let mut parts_indices = run_par(&args.hmetis_bin, &hg_ur, num_parts);
            for idcs in &mut parts_indices {
                for i in idcs {
                    *i = unrealized_endpoints[*i];
                }
            }
            let parts_try = parts_indices.par_iter()
                .map(|endpts| Partition::build_one(&aig, staged, endpts))
                .collect::<Vec<_>>();
            let mut new_unrealized_endpoints = Vec::new();
            for (idx, part_opt) in parts_indices.into_iter().zip(parts_try.into_iter()) {
                match part_opt {
                    Some(_part) => {
                        parts_indices_good.push(idx);
                    }
                    None => {
                        if idx.len() == 1 {
                            panic!("A single endpoint still cannot map, you need to increase level cut granularity.");
                        }
                        for endpt_i in idx {
                            new_unrealized_endpoints.push(endpt_i);
                        }
                    }
                }
            }
            new_unrealized_endpoints.sort_unstable();
            unrealized_endpoints = new_unrealized_endpoints;
        }

        clilog::info!("interactive partition completed: {} in total. merging started.",
                      parts_indices_good.len());

        let effective_parts = process_partitions(
            &aig, staged, parts_indices_good, args.max_stage_degrad
        ).unwrap();
        clilog::info!("after merging: {} parts.", effective_parts.len());
        effective_parts
    }).collect::<Vec<_>>();

    let f = std::fs::File::create(&args.parts_out).unwrap();
    let mut buf = std::io::BufWriter::new(f);
    serde_bare::to_writer(&mut buf, &stages_effective_parts).unwrap();
}
