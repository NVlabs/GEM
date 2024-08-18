use std::path::PathBuf;
use gem::aigpdk::AIGPDKLeafPins;
use gem::aig::AIG;
use gem::repcut::RCHyperGraph;
use netlistdb::NetlistDB;
use std::io::Write;

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
    /// Output path for hypergraph file.
    hgr_output: PathBuf,
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

    let hg = RCHyperGraph::from_aig(&aig);

    let f = std::fs::File::create(&args.hgr_output).unwrap();
    let mut buf = std::io::BufWriter::new(f);
    write!(buf, "{}", hg).unwrap();
}
