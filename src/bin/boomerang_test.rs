use std::path::PathBuf;
use gem::aigpdk::AIGPDKLeafPins;
use gem::aig::AIG;
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
    /// Input path for the partition result.
    parts: PathBuf,
    /// Output path for the serialized partitions.
    parts_out: PathBuf,
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
    let effective_parts = process_partitions_from_hgr_parts_file(
        &aig, &args.parts
    ).expect("some partition failed to map. please increase granularity.");

    clilog::info!("# of effective partitions: {}", effective_parts.len());

    let f = std::fs::File::create(&args.parts_out).unwrap();
    let mut buf = std::io::BufWriter::new(f);
    serde_bare::to_writer(&mut buf, &effective_parts).unwrap();
}
