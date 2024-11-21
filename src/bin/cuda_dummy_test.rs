//! this binary only measures performance for 10000 cycles. it does not
//! input or output actual VCD.

use std::path::PathBuf;
use gem::aigpdk::AIGPDKLeafPins;
use gem::aig::{DriverType, AIG};
use gem::staging::build_staged_aigs;
use gem::pe::Partition;
use gem::flatten::FlattenedScriptV1;
use netlistdb::NetlistDB;
use ulib::{Device, UVec};

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
    /// Input path for the serialized partitions.
    gemparts: PathBuf,
    /// the number of CUDA blocks to map and execute with.
    ///
    /// should not exceed GPU maximum simutaneous occupancy.
    num_blocks: usize,
    /// the number of dummy cycles to execute.
    num_dummy_cycles: usize,
}

mod ucci {
    include!(concat!(env!("OUT_DIR"), "/uccbind/kernel_v1.rs"));
}

fn main() {
    clilog::init_stderr_color_debug();
    clilog::enable_timer("cuda_dummy_test");
    clilog::enable_timer("gem");
    clilog::set_max_print_count(clilog::Level::Warn, "NL_SV_LIT", 1);
    let args = <SimulatorArgs as clap::Parser>::parse();
    clilog::info!("Simulator args:\n{:#?}", args);

    let netlistdb = NetlistDB::from_sverilog_file(
        &args.netlist_verilog,
        args.top_module.as_deref(),
        &AIGPDKLeafPins()
    ).expect("cannot build netlist");

    let aig = AIG::from_netlistdb(&netlistdb);

    // print some statistics for listing
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
    println!("netlist has {} pins, {} aig pins, {} and gates",
             netlistdb.num_pins, aig.num_aigpins, aig.and_gate_cache.len());
    println!("netlist logic depth: {}", max_level);

    let stageds = build_staged_aigs(&aig, &args.level_split);

    let f = std::fs::File::open(&args.gemparts).unwrap();
    let mut buf = std::io::BufReader::new(f);
    let parts_in_stages: Vec<Vec<Partition>> = serde_bare::from_reader(&mut buf).unwrap();
    clilog::info!("# of effective partitions in each stage: {:?}",
                  parts_in_stages.iter().map(|ps| ps.len()).collect::<Vec<_>>());

    let mut input_layout = Vec::new();
    for (i, driv) in aig.drivers.iter().enumerate() {
        if let DriverType::InputPort(_) | DriverType::InputClockFlag(_, _) = driv {
            input_layout.push(i);
        }
    }

    let script = FlattenedScriptV1::from(
        &aig, &stageds.iter().map(|(_, _, staged)| staged).collect::<Vec<_>>(),
        &parts_in_stages.iter().map(|ps| ps.as_slice()).collect::<Vec<_>>(),
        args.num_blocks, input_layout
    );

    use std::hash::{DefaultHasher, Hash, Hasher};
    let mut s = DefaultHasher::new();
    script.blocks_data.hash(&mut s);
    println!("Script hash: {}", s.finish());

    // do simulation
    clilog::info!("total number of cycles: {}", args.num_dummy_cycles);
    let device = Device::CUDA(0);
    let mut input_states_uvec = UVec::new_zeroed(script.reg_io_state_size as usize * (args.num_dummy_cycles + 1), device);
    let mut sram_storage = UVec::new_zeroed(script.sram_storage_size as usize, device);
    device.synchronize();
    let timer_sim = clilog::stimer!("simulation (warm up)");
    ucci::simulate_v1_noninteractive_simple_scan(
        args.num_blocks,
        script.num_major_stages,
        &script.blocks_start, &script.blocks_data,
        &mut sram_storage,
        args.num_dummy_cycles,
        script.reg_io_state_size as usize,
        &mut input_states_uvec,
        device
    );
    device.synchronize();
    clilog::finish!(timer_sim);
    let timer_sim = clilog::stimer!("simulation");
    ucci::simulate_v1_noninteractive_simple_scan(
        args.num_blocks,
        script.num_major_stages,
        &script.blocks_start, &script.blocks_data,
        &mut sram_storage,
        args.num_dummy_cycles,
        script.reg_io_state_size as usize,
        &mut input_states_uvec,
        device
    );
    device.synchronize();
    clilog::finish!(timer_sim);
}
