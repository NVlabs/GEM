use std::collections::HashSet;
use std::path::PathBuf;
use gem::aigpdk::AIGPDKLeafPins;
use gem::aig::{DriverType, EndpointGroup, AIG};
use gem::pe::Partition;
use gem::flatten::FlattenedScriptV1;
use netlistdb::{NetlistDB, GeneralPinName};

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
    /// Input path for the serialized partitions.
    gemparts: PathBuf,
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
    let f = std::fs::File::open(&args.gemparts).unwrap();
    let mut buf = std::io::BufReader::new(f);
    let effective_parts: Vec<Partition> = serde_bare::from_reader(&mut buf).unwrap();
    clilog::info!("# of effective partitions: {}", effective_parts.len());

    for (i, part) in effective_parts.iter().enumerate() {
        println!("Part {}: #stages {}", i, part.stages.len());
        println!("  num occupied inputs: {}", part.stages[0].hier[0].iter().filter(|x| **x != usize::MAX).count());
        println!("  num unique inputs: {}", part.stages[0].hier[0].iter().copied().collect::<HashSet<_>>().len());
        println!("  num effective outputs: {}", part.endpoints.iter().map(|e| {
            match aig.get_endpoint_group(*e) {
                EndpointGroup::PrimaryOutput(_) | EndpointGroup::DFF(_) => 1usize,
                EndpointGroup::RAMBlock(_) => 32usize
            }
        }).sum::<usize>());
        println!("  num srams to sim: {}", part.endpoints.iter().map(|e| {
            match aig.get_endpoint_group(*e) {
                EndpointGroup::PrimaryOutput(_) | EndpointGroup::DFF(_) => 0usize,
                EndpointGroup::RAMBlock(_) => 1usize
            }
        }).sum::<usize>());
    }

    let mut input_layout = Vec::new();
    for (i, driv) in aig.drivers.iter().enumerate() {
        if let DriverType::InputPort(_) | DriverType::InputClockFlag(_, _) = driv {
            input_layout.push(i);
        }
    }

    for (netlist_pin, aigpin) in aig.pin2aigpin_iv.iter().enumerate() {
        if (aigpin >> 1) == 23011 {
            clilog::debug!("test: netlist_pin {} ({}), aigpin {}",
                           netlist_pin, netlistdb.pinnames[netlist_pin].dbg_fmt_pin(), aigpin);
        }
    }
    for (cellid, dff) in &aig.dffs {
        if (dff.q >> 1) == 23011 {
            clilog::debug!("test: dff cellid {}: {:?}", cellid, dff);
        }
    }

    let _ = FlattenedScriptV1::from(
        &aig, &effective_parts, 108, input_layout
    );
}
