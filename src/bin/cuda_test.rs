use std::path::PathBuf;
use gem::aigpdk::{AIGPDKLeafPins, AIGPDK_SRAM_SIZE};
use gem::aig::{DriverType, AIG};
use gem::staging::build_staged_aigs;
use gem::pe::Partition;
use gem::flatten::FlattenedScriptV1;
use netlistdb::{Direction, GeneralPinName, NetlistDB};
use sverilogparse::SVerilogRange;
use compact_str::CompactString;
use ulib::{AsUPtrMut, Device, UVec};
use std::fs::File;
use std::io::{BufReader, BufWriter, Seek, SeekFrom};
use std::hash::Hash;
use std::rc::Rc;
use std::collections::{HashMap, HashSet};
use vcd_ng::{Parser, ScopeItem, Var, Scope, FastFlow, FastFlowToken, FFValueChange, Writer, SimulationCommand};

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
    /// VCD input signal path
    input_vcd: String,
    /// The scope path of top module in the input VCD.
    ///
    /// If not specified, we will use a flat view.
    /// (this view is often incorrect..)
    #[clap(long)]
    input_vcd_scope: Option<String>,
    /// Output VCD path (must be writable)
    output_vcd: String,
    /// The scope path of top module in the output VCD.
    ///
    /// If not specified, we will use `gem_top_module`.
    #[clap(long)]
    output_vcd_scope: Option<String>,
    /// the number of CUDA blocks to map and execute with.
    ///
    /// should not exceed GPU maximum simutaneous occupancy.
    num_blocks: usize,
    /// Whether to run a sanity check against CPU baseline on finish.
    #[clap(long)]
    check_with_cpu: bool,
    /// Limit the number of simulated cycles to no more than this.
    #[clap(long)]
    max_cycles: Option<usize>,
}

/// Hierarchical name representation in VCD.
#[derive(PartialEq, Eq, Clone, Debug)]
struct VCDHier {
    cur: CompactString,
    prev: Option<Rc<VCDHier>>
}

/// Reverse iterator of a [`VCDHier`], yielding cell names
/// from the bottom to the top module.
struct VCDHierRevIter<'i>(Option<&'i VCDHier>);

impl<'i> Iterator for VCDHierRevIter<'i> {
    type Item = &'i CompactString;

    #[inline]
    fn next(&mut self) -> Option<&'i CompactString> {
        let name = self.0?;
        if name.cur.is_empty() {
            return None
        }
        let ret = &name.cur;
        self.0 = name.prev.as_ref().map(|a| a.as_ref());
        Some(ret)
    }
}

impl<'i> IntoIterator for &'i VCDHier {
    type Item = &'i CompactString;
    type IntoIter = VCDHierRevIter<'i>;

    #[inline]
    fn into_iter(self) -> VCDHierRevIter<'i> {
        VCDHierRevIter(Some(self))
    }
}

impl Hash for VCDHier {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for s in self.iter() {
            s.hash(state);
        }
    }
}

#[allow(dead_code)]
impl VCDHier {
    #[inline]
    fn single(cur: CompactString) -> Self {
        VCDHier { cur, prev: None }
    }

    #[inline]
    fn empty() -> Self {
        VCDHier { cur: "".into(), prev: None }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.cur.as_str() == "" && self.prev.is_none()
    }

    #[inline]
    fn iter(&self) -> VCDHierRevIter {
        (&self).into_iter()
    }
}

/// Try to match one component in a scope.
/// If succeed, returns the remaining scope (can be None itself indicating
/// all paths matched).
/// If fails, return None.
fn match_scope_path<'i>(mut scope: &'i str, cur: &str) -> Option<&'i str> {
    if scope.len() == 0 { return Some("") }
    if scope.starts_with('/') {
        scope = &scope[1..];
    }
    if scope.len() == 0 { Some("") }
    else if scope.starts_with(cur) {
        if scope.len() == cur.len() { Some("") }
        else if scope.as_bytes()[cur.len()] == b'/' {
            Some(&scope[cur.len() + 1..])
        }
        else { None }
    }
    else { None }
}

fn find_top_scope<'i>(
    items: &'i [ScopeItem], top_scope: &'_ str
) -> Option<&'i Scope> {
    for item in items {
        if let ScopeItem::Scope(scope) = item {
            if let Some(s1) = match_scope_path(
                top_scope, scope.identifier.as_str()
            ) {
                return match s1 {
                    "" => Some(scope),
                    _ => find_top_scope(&scope.children[..], s1)
                };
            }
        }
    }
    None
}

/// CPU prototype partition executor for script version 1.
fn simulate_block_v1(
    script: &[u32],
    input_state: &[u32], output_state: &mut [u32],
    sram_data: &mut [u32],
    debug_verbose: bool,
) {
    let mut script_pi = 0;
    loop {
        let num_stages = script[script_pi];
        let is_last_part = script[script_pi + 1];
        let num_ios = script[script_pi + 2];
        let io_offset = script[script_pi + 3];
        let num_srams = script[script_pi + 4];
        let sram_offset = script[script_pi + 5];
        let num_global_read_rounds = script[script_pi + 6];
        let num_output_duplicates = script[script_pi + 7];
        let mut writeout_hooks = vec![0; 256];
        for i in 0..128 {
            let t = script[script_pi + 128 + i];
            writeout_hooks[i * 2] = (t & ((1 << 16) - 1)) as u16;
            writeout_hooks[i * 2 + 1] = (t >> 16) as u16;
        }
        if num_stages == 0 {
            script_pi += 256;
            break
        }
        // assert_eq!(part.stages.len(), num_stages as usize);
        // assert_eq!(part.stages.iter().map(|s| s.write_outs.len()).sum::<usize>(), (num_ios - num_srams - num_output_duplicates) as usize);
        script_pi += 256;
        let mut writeouts = vec![0u32; num_ios as usize];

        let mut state = vec![0u32; 256];
        for _gr_i in 0..num_global_read_rounds {
            for i in 0..256 {
                let mut cur_state = state[i];
                let idx = script[script_pi + (i * 2)];
                let mut mask = script[script_pi + (i * 2 + 1)];
                if mask == 0 { continue }
                let value = match (idx >> 31) != 0 {
                    false => input_state[idx as usize],
                    true => output_state[(idx ^ (1 << 31)) as usize]
                };
                while mask != 0 {
                    cur_state <<= 1;
                    let lowbit = mask & (-(mask as i32)) as u32;
                    if (value & lowbit) != 0 {
                        cur_state |= 1;
                    }
                    mask ^= lowbit;
                }
                state[i] = cur_state;
            }
            script_pi += 256 * 2;
        }

        if debug_verbose {
            println!("debug_verbose STAGE 0");
            println!("global read states:");
            for i in 0..256 {
                println!(" [{}] = {}", i, state[i]);
            }
        }

        for bs_i in 0..num_stages {
            let mut hier_inputs = vec![0; 256];
            let mut hier_flag_xora = vec![0; 256];
            let mut hier_flag_xorb = vec![0; 256];
            let mut hier_flag_orb = vec![0; 256];
            for k_outer in 0..4 {
                for i in 0..256 {
                    for k_inner in 0..4 {
                        let k = k_outer * 4 + k_inner;
                        let t_shuffle = script[script_pi + i * 4 + k_inner];
                        let t_shuffle_1_idx = (t_shuffle & ((1 << 16) - 1)) as u16;
                        let t_shuffle_2_idx = (t_shuffle >> 16) as u16;
                        hier_inputs[i] |= (state[(t_shuffle_1_idx >> 5) as usize] >> (t_shuffle_1_idx & 31) & 1) << (k * 2);
                        hier_inputs[i] |= (state[(t_shuffle_2_idx >> 5) as usize] >> (t_shuffle_2_idx & 31) & 1) << (k * 2 + 1);
                    }
                }
                script_pi += 256 * 4;
            }
            for i in 0..256 {
                hier_flag_xora[i] = script[script_pi + i * 4];
                hier_flag_xorb[i] = script[script_pi + i * 4 + 1];
                hier_flag_orb[i] = script[script_pi + i * 4 + 2];
            }
            script_pi += 256 * 4;

            if debug_verbose {
                println!("debug_verbose STAGE 1.1 bs_i {bs_i}");
                println!("after local shuffle:");
                for i in 0..256 {
                    println!(" [{}] = {}", i, hier_inputs[i]);
                }
            }

            // hier[0]
            for i in 0..128 {
                let a = hier_inputs[i];
                let b = hier_inputs[128 + i];
                let xora = hier_flag_xora[128 + i];
                let xorb = hier_flag_xorb[128 + i];
                let orb = hier_flag_orb[128 + i];
                let ret = (a ^ xora) & ((b ^ xorb) | orb);
                hier_inputs[128 + i] = ret;
            }
            // hier 1 to 7
            for hi in 1..=7 {
                let hier_width = 1 << (7 - hi);
                for i in 0..hier_width {
                    let a = hier_inputs[hier_width * 2 + i];
                    let b = hier_inputs[hier_width * 3 + i];
                    let xora = hier_flag_xora[hier_width + i];
                    let xorb = hier_flag_xorb[hier_width + i];
                    let orb = hier_flag_orb[hier_width + i];
                    let ret = (a ^ xora) & ((b ^ xorb) | orb);
                    // for k in 0..32 {
                    //     let apin = part.stages[bs_i as usize].hier[hi][i * 32 + k];
                    //     let bpin = part.stages[bs_i as usize].hier[hi][part.stages[bs_i as usize].hier[hi + 1].len() + i * 32 + k];
                    //     let opin = part.stages[bs_i as usize].hier[hi + 1][i * 32 + k];
                    //     if [21876 / 2].contains(&opin) {
                    //         println!("Got ai gate at part {} bs_i {bs_i} hi {hi} i {i} k {k} (pos {} put {}): {opin}={} <- f[{apin}={} ^{}, {bpin}={} ^{}|{}]", parts_indices[part_i_dbg - 1], i * 32 + k, hier_width * 32 + i * 32 + k, ret >> k & 1, a >> k & 1, xora >> k & 1, b >> k & 1, xorb >> k & 1, orb >> k & 1);
                    //     }
                    // }
                    hier_inputs[hier_width + i] = ret;
                }
            }
            // hier 8,9,10,11,12
            let v1 = hier_inputs[1];
            let xora = hier_flag_xora[0];
            let xorb = hier_flag_xorb[0];
            let orb = hier_flag_orb[0];
            let r8 = ((v1 << 16) ^ xora) & ((v1 ^ xorb) | orb) & 0xffff0000;
            let r9 = ((r8 >> 8) ^ xora) & (((r8 >> 16) ^ xorb) | orb) & 0xff00;
            let r10 = ((r9 >> 4) ^ xora) & (((r9 >> 8) ^ xorb) | orb) & 0xf0;
            let r11 = ((r10 >> 2) ^ xora) & (((r10 >> 4) ^ xorb) | orb) & 0b1100;
            let r12 = ((r11 >> 1) ^ xora) & (((r11 >> 2) ^ xorb) | orb) & 0b10;
            hier_inputs[0] = r8 | r9 | r10 | r11 | r12;

            state = hier_inputs;

            if debug_verbose {
                println!("debug_verbose STAGE 1.2 bs_i {bs_i}");
                println!("after and-invert:");
                for i in 0..256 {
                    println!(" [{}] = {}", i, state[i]);
                }
            }

            for i in 0..256 {
                let hooki = writeout_hooks[i];
                if (hooki >> 8) as u32 == bs_i {
                    writeouts[i] = state[(hooki & 255) as usize];
                }
            }
        }

        let mut sram_duplicate_perm = vec![0u32; (num_srams * 4 + num_output_duplicates) as usize];
        for k_outer in 0..4 {
            for i in 0..(num_srams * 4 + num_output_duplicates) {
                for k_inner in 0..4 {
                    let k = k_outer * 4 + k_inner;
                    let t_shuffle = script[script_pi + (i * 4 + k_inner) as usize];
                    let t_shuffle_1_idx = (t_shuffle & ((1 << 16) - 1)) as u32;
                    let t_shuffle_2_idx = (t_shuffle >> 16) as u32;
                    sram_duplicate_perm[i as usize] |= (writeouts[(t_shuffle_1_idx >> 5) as usize] >> (t_shuffle_1_idx & 31) & 1) << (k * 2);
                    sram_duplicate_perm[i as usize] |= (writeouts[(t_shuffle_2_idx >> 5) as usize] >> (t_shuffle_2_idx & 31) & 1) << (k * 2 + 1);
                }
            }
            script_pi += 256 * 4;
        }
        for i in 0..(num_srams * 4 + num_output_duplicates) as usize {
            sram_duplicate_perm[i] &= !script[script_pi + i * 4 + 1];
            sram_duplicate_perm[i] ^= script[script_pi + i * 4];
        }
        script_pi += 256 * 4;

        for sram_i_u32 in 0..num_srams {
            let sram_i = sram_i_u32 as usize;
            let addrs = sram_duplicate_perm[sram_i * 4];
            let port_r_addr_iv = addrs & 0xffff;
            let port_w_addr_iv = (addrs & 0xffff0000) >> 16;
            let port_w_wr_en = sram_duplicate_perm[sram_i * 4 + 1];
            let port_w_wr_data_iv = sram_duplicate_perm[sram_i * 4 + 2];

            let sram_st = sram_offset as usize + sram_i * AIGPDK_SRAM_SIZE;
            let sram_ed = sram_st + AIGPDK_SRAM_SIZE;
            let ram = &mut sram_data[sram_st..sram_ed];
            let r = ram[port_r_addr_iv as usize];
            let w0 = ram[port_w_addr_iv as usize];
            writeouts[(num_ios - num_srams + sram_i_u32) as usize] = r;
            ram[port_w_addr_iv as usize] = (w0 & !port_w_wr_en) | (port_w_wr_data_iv & port_w_wr_en);
            // println!("sram for part id {} index {sram_i_u32}: port_r_addr_iv {port_r_addr_iv} port_w_addr_iv {port_w_addr_iv} port_w_wr_en {port_w_wr_en} port_w_wr_data_iv {port_w_wr_data_iv}", parts_indices[part_i_dbg - 1]);
        }

        for i in 0..num_output_duplicates {
            writeouts[(num_ios - num_srams - num_output_duplicates + i) as usize] =
                sram_duplicate_perm[(num_srams * 4 + i) as usize];
        }

        if debug_verbose {
            println!("debug_verbose STAGE 2");
            println!("before writeout_inv:");
            for i in 0..256 {
                println!(" [{}] = {}", i, if i < num_ios as usize {
                    writeouts[i]
                } else {
                    0
                });
            }
        }

        let mut clken_perm = vec![0u32; num_ios as usize];
        let writeouts_for_clken = writeouts.clone();
        for k_outer in 0..4 {
            for i in 0..num_ios {
                for k_inner in 0..4 {
                    let k = k_outer * 4 + k_inner;
                    let t_shuffle = script[script_pi + (i * 4 + k_inner) as usize];
                    let t_shuffle_1_idx = (t_shuffle & ((1 << 16) - 1)) as u32;
                    let t_shuffle_2_idx = (t_shuffle >> 16) as u32;
                    clken_perm[i as usize] |= (writeouts_for_clken[(t_shuffle_1_idx >> 5) as usize] >> (t_shuffle_1_idx & 31) & 1) << (k * 2);
                    clken_perm[i as usize] |= (writeouts_for_clken[(t_shuffle_2_idx >> 5) as usize] >> (t_shuffle_2_idx & 31) & 1) << (k * 2 + 1);
                }
            }
            script_pi += 256 * 4;
        }
        for i in 0..num_ios as usize {
            clken_perm[i] &= !script[script_pi + i * 4 + 1];
            clken_perm[i] ^= script[script_pi + i * 4];
            writeouts[i] ^= script[script_pi + i * 4 + 2];
        }
        script_pi += 256 * 4;
        // println!("test: clken_perm {:?}", clken_perm);

        for i in 0..num_ios {
            let old_wo = input_state[(io_offset + i) as usize];
            let clken = clken_perm[i as usize];
            let wo = (old_wo & !clken) | (writeouts[i as usize] & clken);
            output_state[(io_offset + i) as usize] = wo;
        }

        if debug_verbose {
            println!("debug_verbose STAGE 3");
            println!("final writeout:");
            for i in 0..num_ios {
                println!(" [{}] [global {}] = {}", i, io_offset + i, output_state[(io_offset + i) as usize]);
            }
        }

        if is_last_part != 0 {
            break
        }
    }
    assert_eq!(script_pi, script.len());
}

mod ucci {
    include!(concat!(env!("OUT_DIR"), "/uccbind/kernel_v1.rs"));
}

fn main() {
    clilog::init_stderr_color_debug();
    clilog::enable_timer("cuda_test");
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

    // simulate with the script.
    let input_vcd = File::open(&args.input_vcd).unwrap();
    let mut bufrd = BufReader::with_capacity(65536, input_vcd);
    let mut vcd_parser = Parser::new(&mut bufrd);
    let header = vcd_parser.parse_header().unwrap();
    drop(vcd_parser);
    let mut vcd_file = bufrd.into_inner();
    vcd_file.seek(SeekFrom::Start(0)).unwrap();
    let mut vcdflow = FastFlow::new(vcd_file, 65536);

    let top_scope = find_top_scope(
        &header.items[..],
        args.input_vcd_scope.as_deref().unwrap_or("")
    ).expect("Specified top scope not found in VCD.");

    let mut vcd2inp = HashMap::new();
    let mut inp_port_given = HashSet::new();

    let mut match_one_input = |var: &Var, i: Option<isize>, vcd_pos: usize| {
        let key = (VCDHier::empty(), var.reference.as_str(), i);
        if let Some(&id) = netlistdb.pinname2id.get(
            &key as &dyn GeneralPinName
        ) {
            if netlistdb.pindirect[id] != Direction::O { return }
            vcd2inp.insert((var.code.0, vcd_pos), id);
            inp_port_given.insert(id);
        }
    };
    for scope_item in &top_scope.children[..] {
        if let ScopeItem::Var(var) = scope_item {
            use vcd_ng::ReferenceIndex::*;
            match var.index {
                None => match var.size {
                    1 => match_one_input(var, None, 0),
                    w @ _ => {
                        for (pos, i) in (0..w).rev()
                            .enumerate()
                        {
                            match_one_input(
                                var, Some(i as isize), pos)
                        }
                    }
                },
                Some(BitSelect(i)) => match_one_input(
                    var, Some(i as isize), 0),
                Some(Range(a, b)) => {
                    for (pos, i) in SVerilogRange(
                        a as isize, b as isize).enumerate()
                    {
                        match_one_input(var, Some(i), pos);
                    }
                }
            }
        }
    }
    for i in netlistdb.cell2pin.iter_set(0) {
        if netlistdb.pindirect[i] != Direction::I &&
            !inp_port_given.contains(&i)
        {
            clilog::warn!(
                GATESIM_VCDI_MISSING_PI,
                "Primary input port {:?} not present in \
                 the VCD input",
                netlistdb.pinnames[i]);
        }
    }

    // open out
    let write_buf = File::create(&args.output_vcd).unwrap();
    let write_buf = BufWriter::new(write_buf);
    let mut writer = Writer::new(write_buf);
    if let Some((ratio, unit)) = header.timescale {
        writer.timescale(ratio, unit).unwrap();
    }
    let output_vcd_scope = args.output_vcd_scope.as_deref().unwrap_or("gem_top_module");
    let output_vcd_scope = output_vcd_scope.split('/').collect::<Vec<_>>();
    for &scope in &output_vcd_scope {
        writer.add_module(scope).unwrap();
    }
    let out2vcd = netlistdb.cell2pin.iter_set(0).filter_map(|i| {
        if netlistdb.pindirect[i] == Direction::I {
            let aigpin = aig.pin2aigpin_iv[i];
            if matches!(aig.drivers[aigpin >> 1], DriverType::InputPort(_)) {
                clilog::info!("skipped output for port {} as it is a pass-through of input port.", netlistdb.pinnames[i].dbg_fmt_pin());
                return None
            }
            if aigpin <= 1 {
                return Some((aigpin, u32::MAX, writer.add_wire(
                    1, &format!("{}", netlistdb.pinnames[i].dbg_fmt_pin())).unwrap()))
            }
            Some((aigpin, *script.output_map.get(&aigpin).unwrap(), writer.add_wire(
                1, &format!("{}", netlistdb.pinnames[i].dbg_fmt_pin())).unwrap()))
        }
        else { None }
    }).collect::<Vec<_>>();

    for _ in 0..output_vcd_scope.len() {
        writer.upscope().unwrap();
    }
    writer.enddefinitions().unwrap();
    writer.begin(SimulationCommand::Dumpvars).unwrap();

    // do simulation
    let mut state = vec![0; script.reg_io_state_size as usize];

    // the simulator keeps 2 previous timestamps.
    // vcd_time: the last seen timestamp.
    // vcd_time_last_active: the last timestamp strictly before vcd_time that has
    // active events (e.g., watched clock posedge).
    //
    // when a new timestamp arrives and vcd_time has active events, we simulate
    // the circuit with {actived edge flags from vcd_time}, but do NOT include the
    // input port value changes. then, we associate the result output port values to
    // vcd_time_last_active.
    //
    // the above complexity rises from the necessity to emulate {update, then propagate}
    // behavior from our actual {propagate, then update} implementation.
    let mut vcd_time_last_active = u64::MAX;
    let mut vcd_time = 0;
    let mut last_vcd_time_active = true;
    let mut delayed_bit_changes = HashSet::new();

    let mut input_states = Vec::new();
    let mut offsets_timestamps = Vec::new();

    while let Some(tok) = vcdflow.next_token().unwrap() {
        match tok {
            FastFlowToken::Timestamp(t) => {
                if t == vcd_time { continue }
                if last_vcd_time_active {
                    // clilog::debug!("simulating t={}", vcd_time);
                    input_states.extend(state.iter().copied());
                    offsets_timestamps.push((input_states.len(), vcd_time_last_active));
                    // reset for next timestamp
                    for (_, &(pe, ne)) in &aig.clock_pin2aigpins {
                        if pe != usize::MAX {
                            let p = *script.input_map.get(&pe).unwrap();
                            state[p as usize >> 5] &= !(1 << (p & 31));
                        }
                        if ne != usize::MAX {
                            let p = *script.input_map.get(&ne).unwrap();
                            state[p as usize >> 5] &= !(1 << (p & 31));
                        }
                    }
                    if let Some(max_cycles) = args.max_cycles {
                        if offsets_timestamps.len() >= max_cycles {
                            clilog::info!("reached maximum cycles, stop reading input vcd");
                            break
                        }
                    }
                }
                if last_vcd_time_active {
                    vcd_time_last_active = vcd_time;
                }
                vcd_time = t;
                last_vcd_time_active = false;

                for pos in std::mem::take(&mut delayed_bit_changes) {
                    state[(pos >> 5) as usize] ^= 1u32 << (pos & 31);
                }
            },
            FastFlowToken::Value(FFValueChange { id, bits }) => {
                for (pos, b) in bits.iter().enumerate() {
                    if let Some(&pin) = vcd2inp.get(
                        &(id.0, pos)
                    ) {
                        let aigpin = aig.pin2aigpin_iv[pin];
                        assert_eq!(aigpin & 1, 0);
                        let aigpin = aigpin >> 1;
                        let pos = match script.input_map.get(&aigpin).copied() {
                            Some(pos) => pos,
                            None => {
                                panic!("input pin {:?} (netlist id {}, aigpin {}) not found in output map.", netlistdb.pinnames[pin].dbg_fmt_pin(), pin, aigpin);
                            }
                        };
                        let old_value = state[(pos >> 5) as usize] >> (pos & 31) & 1;
                        if old_value == match b { b'1' => 1, _ => 0 } {
                            continue
                        }
                        if let Some((pe, ne)) = aig.clock_pin2aigpins.get(&pin).copied() {
                            if pe != usize::MAX && old_value == 0 {
                                last_vcd_time_active = true;
                                let p = *script.input_map.get(&pe).unwrap();
                                state[p as usize >> 5] |= 1 << (p & 31);
                            }
                            if ne != usize::MAX && old_value == 1 {
                                last_vcd_time_active = true;
                                let p = *script.input_map.get(&ne).unwrap();
                                state[p as usize >> 5] |= 1 << (p & 31);
                            }
                        }
                        delayed_bit_changes.insert(pos);
                    }
                }
            }
        }
    }
    input_states.extend(state.iter().copied());
    clilog::info!("total number of cycles: {}", offsets_timestamps.len());
    let mut input_states_uvec: UVec<_> = input_states.clone().into();
    let device = Device::CUDA(0);
    input_states_uvec.as_mut_uptr(device);
    let mut sram_storage = UVec::new_zeroed(script.sram_storage_size as usize, device);
    device.synchronize();
    let timer_sim = clilog::stimer!("simulation");
    ucci::simulate_v1_noninteractive_simple_scan(
        args.num_blocks,
        script.num_major_stages,
        &script.blocks_start, &script.blocks_data,
        &mut sram_storage,
        offsets_timestamps.len(),
        script.reg_io_state_size as usize,
        &mut input_states_uvec,
        device
    );
    device.synchronize();
    clilog::finish!(timer_sim);

    // sanity check.
    if args.check_with_cpu {
        let mut sram_storage_sanity = vec![0; script.sram_storage_size as usize * AIGPDK_SRAM_SIZE];
        let mut input_states_sanity = input_states.clone();
        clilog::info!("running sanity test");
        for i in 0..offsets_timestamps.len() {
            let mut output_state = vec![0; script.reg_io_state_size as usize];
            output_state.copy_from_slice(&input_states_sanity[((i + 1) * script.reg_io_state_size as usize)..((i + 2) * script.reg_io_state_size as usize)]);
            for stage_i in 0..script.num_major_stages {
                for blk_i in 0..script.num_blocks {
                    simulate_block_v1(
                        &script.blocks_data[script.blocks_start[stage_i * script.num_blocks + blk_i]..script.blocks_start[stage_i * script.num_blocks + blk_i + 1]],
                        &input_states_sanity[(i * script.reg_io_state_size as usize)..((i + 1) * script.reg_io_state_size as usize)],
                        &mut output_state,
                        &mut sram_storage_sanity,
                        false
                    );
                }
            }
            input_states_sanity[((i + 1) * script.reg_io_state_size as usize)..((i + 2) * script.reg_io_state_size as usize)].copy_from_slice(&output_state);
            if output_state != input_states_uvec[((i + 1) * script.reg_io_state_size as usize)..((i + 2) * script.reg_io_state_size as usize)] {
                println!("sanity check fail at cycle {i}.\ncpu good: {:?}\ngpu bad: {:?}", output_state, &input_states_uvec[((i + 1) * script.reg_io_state_size as usize)..((i + 2) * script.reg_io_state_size as usize)]);
                panic!()
            }
        }
        clilog::info!("sanity test passed!");
    }

    // output...
    clilog::info!("write out vcd");
    let mut last_val = vec![2; out2vcd.len()];
    for &(offset, timestamp) in &offsets_timestamps {
        if timestamp == u64::MAX {
            continue
        }
        writer.timestamp(timestamp).unwrap();
        for (i, &(output_aigpin, output_pos, vid)) in out2vcd.iter().enumerate() {
            use vcd_ng::Value;
            let value_new = match output_pos {
                u32::MAX => {
                    assert!(output_aigpin <= 1);
                    output_aigpin as u32
                },
                output_pos @ _ => {
                    let value_new_output = input_states_uvec[offset + (output_pos >> 5) as usize] >> (output_pos & 31) & 1;
                    value_new_output
                },
            };
            if value_new == last_val[i] {
                continue
            }
            last_val[i] = value_new;
            writer.change_scalar(vid, match value_new {
                1 => Value::V1,
                _ => Value::V0
            }).unwrap();
        }
    }
}
