//! And-inverter graph format
//!
//! An AIG is derived from netlistdb synthesized in AIGPDK.

use netlistdb::{NetlistDB, GeneralPinName, Direction};
use indexmap::{IndexMap, IndexSet};

/// A DFF.
#[derive(Debug, Default, Clone)]
pub struct DFF {
    /// The D input pin with invert (last bit)
    pub d_iv: usize,
    /// If the DFF is enabled, i.e., if the clock, S, or R is active.
    pub en: usize,
    /// The Q pin output.
    pub q: usize,
}

/// A ram block resembling the interface of `$__RAMGEM_SYNC_`.
#[derive(Debug, Default, Clone)]
pub struct RAMBlock {
    pub port_r_addr_iv: [usize; 13],

    /// controls whether r_rd_data should update. (from read clock)
    pub port_r_en_iv: usize,
    pub port_r_rd_data: [usize; 32],

    pub port_w_addr_iv: [usize; 13],
    /// controls whether memory should be updated.
    ///
    /// this is a combination of write enable and write clock.
    pub port_w_wr_en_iv: [usize; 32],
    pub port_w_wr_data_iv: [usize; 32],
}

/// The driver type of an AIG pin.
#[derive(Debug, Clone)]
pub enum DriverType {
    /// Driven by an and gate.
    ///
    /// The inversion bit is stored as the last bits in
    /// two input indices.
    ///
    /// Only this type has combinational fan-in.
    AndGate(usize, usize),
    /// Driven by a primary input port (with its netlistdb id).
    InputPort(usize),
    /// Driven by a clock flag (with clock port netlistdb id, and pos/negedge)
    InputClockFlag(usize, u8),
    /// Driven by a DFF (with its index)
    DFF(usize),
    /// Driven by a 13-bit by 32-bit RAM block (with its index)
    SRAM(usize),
    /// Tie0: tied to zero. Only the 0-th aig pin is allowed to have this.
    Tie0
}

/// An AIG associated with a netlistdb.
#[derive(Debug, Default)]
pub struct AIG {
    /// The number of AIG pins.
    ///
    /// This number might be smaller than num_pins in netlistdb,
    /// because inverters and buffers are merged when possible.
    /// It might also be larger because we may add mux circuits.
    ///
    /// AIG pins are numbered from 1 to num_aigpins inclusive.
    /// The AIG pin id zero (0) is tied to 0.
    ///
    /// AIG pins are guaranteed to have topological order.
    pub num_aigpins: usize,
    /// The combinational inputs in AIG.
    ///
    /// They might be:
    /// 1. input ports in netlistdb,
    /// 2. Q pins of DFF,
    /// 3. read out ports of RAMs.
    pub comb_inputs: Vec<usize>,
    /// The combinational outputs in AIG.
    ///
    /// They might be:
    /// 1. output ports in netlistdb,
    /// 2. D pins of DFF
    /// 3. input address, write data, and write enable ports
    ///    of RAMs.
    pub comb_outputs: IndexSet<usize>,
    /// The mapping from a netlistdb pin to an AIG pin.
    ///
    /// The inversion bit is stored as the last bit.
    /// E.g., `pin2aigpin_iv[pin_id] = aigpin_id << 1 | invert`.
    pub pin2aigpin_iv: Vec<usize>,
    /// The clock pins map. Every clock pin has a pair of flag pins
    /// showing if they are posedge/negedge.
    ///
    /// The flag pin can be empty which means the circuit is not
    /// active with that edge.
    pub clock_pin2aigpins: IndexMap<usize, (usize, usize)>,
    /// The driver types of AIG pins.
    pub drivers: Vec<DriverType>,
    /// A cache for identical and gates.
    pub and_gate_cache: IndexMap<(usize, usize), usize>,
    /// The D flip-flops (DFFs), indexed by cell id
    pub dffs: IndexMap<usize, DFF>,
    /// The SRAMs, indexed by cell id
    pub srams: IndexMap<usize, RAMBlock>,
    /// The fanout CSR start array.
    pub fanouts_start: Vec<usize>,
    /// The fanout CSR array.
    pub fanouts: Vec<usize>,
}

impl AIG {
    fn add_aigpin(&mut self, driver: DriverType) -> usize {
        self.num_aigpins += 1;
        self.drivers.push(driver);
        self.num_aigpins
    }

    fn add_and_gate(&mut self, a: usize, b: usize) -> usize {
        if a == 0 || b == 0 {
            return 0
        }
        if a == 1 {
            return b
        }
        if b == 1 {
            return a
        }
        let (a, b) = if a < b { (a, b) } else { (b, a) };
        if let Some(o) = self.and_gate_cache.get(&(a, b)) {
            return o << 1;
        }
        let aigpin = self.add_aigpin(DriverType::AndGate(a, b));
        self.and_gate_cache.insert((a, b), aigpin);
        aigpin << 1
    }

    /// given a clock pin, trace back to clock root and return its
    /// enable signal (with invert bit).
    ///
    /// if result is 0, that means the pin is dangled.
    ///
    /// we currently does not deal with negedge yet.
    fn trace_clock_pin(&mut self, netlistdb: &NetlistDB, pinid: usize) -> usize {
        let netid = netlistdb.pin2net[pinid];
        if Some(netid) == netlistdb.net_zero || Some(netid) == netlistdb.net_one {
            return 0
        }
        let root = netlistdb.net2pin.items[
            netlistdb.net2pin.start[netid]
        ];
        if netlistdb.pin2cell[root] != 0 {
            panic!("A sequential cell in {} is driven by non-port pin {}: this pattern is not yet supported. please disable clock gating.",
                   netlistdb.cellnames[netlistdb.pin2cell[pinid]],
                   netlistdb.pinnames[root].dbg_fmt_pin());
        }
        let clkentry = self.clock_pin2aigpins.entry(root)
            .or_insert((usize::MAX, usize::MAX));
        if clkentry.0 != usize::MAX {
            return clkentry.0 << 1
        }
        let aigpin = self.add_aigpin(DriverType::InputClockFlag(root, 0));
        self.clock_pin2aigpins.get_mut(&root).unwrap().0 = aigpin;
        aigpin << 1
    }

    /// recursively add aig pins for netlistdb pins
    ///
    /// for sequential logics like DFF and RAM,
    /// 1. their netlist pin inputs are not patched,
    /// 2. their aig pin inputs (in dffs and srams arrays) will be
    ///    patched to include mux -- but not inside this function.
    /// 3. their netlist/aig outputs are directly built here,
    ///    with possible patches (only in the future for DFFSR).
    fn dfs_netlistdb_build_aig(
        &mut self,
        netlistdb: &NetlistDB,
        topo_vis: &mut Vec<bool>,
        topo_instack: &mut Vec<bool>,
        pinid: usize
    ) {
        if topo_instack[pinid] {
            panic!("circuit has a loop around pin {}",
                   netlistdb.pinnames[pinid].dbg_fmt_pin());
        }
        if topo_vis[pinid] {
            return
        }
        topo_vis[pinid] = true;
        topo_instack[pinid] = true;
        let netid = netlistdb.pin2net[pinid];
        let cellid = netlistdb.pin2cell[pinid];
        if netlistdb.pindirect[pinid] == Direction::I {
            if Some(netid) == netlistdb.net_zero {
                self.pin2aigpin_iv[pinid] = 0;
            }
            else if Some(netid) == netlistdb.net_one {
                self.pin2aigpin_iv[pinid] = 1;
            }
            else {
                let root = netlistdb.net2pin.items[
                    netlistdb.net2pin.start[netid]
                ];
                self.dfs_netlistdb_build_aig(
                    netlistdb, topo_vis, topo_instack,
                    root
                );
                self.pin2aigpin_iv[pinid] = self.pin2aigpin_iv[root];
            }
            if cellid == 0 && self.pin2aigpin_iv[pinid] >= 2 {
                self.comb_outputs.insert(self.pin2aigpin_iv[pinid] >> 1);
            }
        }
        else if cellid == 0 {
            let aigpin = self.add_aigpin(
                DriverType::InputPort(pinid)
            );
            self.pin2aigpin_iv[pinid] = aigpin << 1;
            self.comb_inputs.push(aigpin);
        }
        else if matches!(netlistdb.celltypes[cellid].as_str(), "DFF" | "DFFSR") {
            let q = self.add_aigpin(DriverType::DFF(cellid));
            self.pin2aigpin_iv[pinid] = q << 1;
            self.comb_inputs.push(q);
            let mut ap_s_iv = 1;
            let mut ap_r_iv = 1;
            let mut q_out = q;
            for pinid in netlistdb.cell2pin.iter_set(cellid) {
                if !matches!(netlistdb.pinnames[pinid].1.as_str(), "S" | "R") {
                    continue
                }
                self.dfs_netlistdb_build_aig(
                    netlistdb, topo_vis, topo_instack, pinid
                );
                let prev = self.pin2aigpin_iv[pinid];
                match netlistdb.pinnames[pinid].1.as_str() {
                    "S" => ap_s_iv = prev,
                    "R" => ap_r_iv = prev,
                    _ => unreachable!()
                }
            }
            q_out = self.add_and_gate(q_out ^ 1, ap_s_iv) ^ 1;
            q_out = self.add_and_gate(q_out, ap_r_iv) ^ 1;
            let dff = self.dffs.entry(cellid).or_default();
            dff.q = q_out;
        }
        else if netlistdb.celltypes[cellid].as_str() == "$__RAMGEM_SYNC_" {
            let o = self.add_aigpin(DriverType::SRAM(cellid));
            self.pin2aigpin_iv[pinid] = o << 1;
            self.comb_inputs.push(o);
            assert_eq!(netlistdb.pinnames[pinid].1.as_str(),
                       "PORT_R_RD_DATA");
            let sram = self.srams.entry(cellid).or_default();
            sram.port_r_rd_data[netlistdb.pinnames[pinid].2.unwrap() as usize] = o;
        }
        else {
            let mut prev_a = usize::MAX;
            let mut prev_b = usize::MAX;
            for pinid in netlistdb.cell2pin.iter_set(cellid) {
                match netlistdb.pinnames[pinid].1.as_str() {
                    "A" => prev_a = pinid,
                    "B" => prev_b = pinid,
                    _ => {}
                }
            }
            for prev in [prev_a, prev_b] {
                if prev != usize::MAX {
                    self.dfs_netlistdb_build_aig(
                        netlistdb, topo_vis, topo_instack,
                        prev
                    );
                }
            }
            match netlistdb.celltypes[cellid].as_str() {
                "AND2_00_0" | "AND2_01_0" | "AND2_10_0" | "AND2_11_0" | "AND2_11_1" => {
                    assert_ne!(prev_a, usize::MAX);
                    assert_ne!(prev_b, usize::MAX);
                    let name = netlistdb.celltypes[cellid].as_bytes();
                    let iv_a = name[5] - b'0';
                    let iv_b = name[6] - b'0';
                    let iv_y = name[8] - b'0';
                    let apid = self.add_and_gate(
                        self.pin2aigpin_iv[prev_a] ^ (iv_a as usize),
                        self.pin2aigpin_iv[prev_b] ^ (iv_b as usize),
                    ) ^ (iv_y as usize);
                    self.pin2aigpin_iv[pinid] = apid;
                },
                "INV" => {
                    assert_ne!(prev_a, usize::MAX);
                    self.pin2aigpin_iv[pinid] = self.pin2aigpin_iv[prev_a] ^ 1;
                },
                "BUF" => {
                    assert_ne!(prev_a, usize::MAX);
                    self.pin2aigpin_iv[pinid] = self.pin2aigpin_iv[prev_a];
                },
                _ => unreachable!()
            }
        }
        topo_instack[pinid] = false;
    }

    pub fn from_netlistdb(netlistdb: &NetlistDB) -> AIG {
        let mut aig = AIG {
            num_aigpins: 0,
            pin2aigpin_iv: vec![usize::MAX; netlistdb.num_pins],
            drivers: vec![DriverType::Tie0],
            ..Default::default()
        };

        for cellid in 1..netlistdb.num_cells {
            if !matches!(netlistdb.celltypes[cellid].as_str(),
                         "DFF" | "$__RAMGEM_SYNC_") {
                continue
            }
            for pinid in netlistdb.cell2pin.iter_set(cellid) {
                if !matches!(netlistdb.pinnames[pinid].1.as_str(),
                            "CLK" | "PORT_R_CLK" | "PORT_W_CLK") {
                    continue
                }
                aig.trace_clock_pin(netlistdb, pinid);
            }
        }
        for (&clk, &(flagr, flagf)) in &aig.clock_pin2aigpins {
            clilog::info!(
                "inferred clock port {} ({})",
                netlistdb.pinnames[clk].dbg_fmt_pin(),
                match (flagr, flagf) {
                    (_, usize::MAX) => "posedge",
                    (usize::MAX, _) => "negedge",
                    _ => "posedge & negedge"
                }
            );
        }

        let mut topo_vis = vec![false; netlistdb.num_pins];
        let mut topo_instack = vec![false; netlistdb.num_pins];

        for pinid in 0..netlistdb.num_pins {
            aig.dfs_netlistdb_build_aig(
                netlistdb, &mut topo_vis, &mut topo_instack,
                pinid
            );
        }

        for cellid in 0..netlistdb.num_cells {
            if matches!(netlistdb.celltypes[cellid].as_str(), "DFF" | "DFFSR") {
                let mut ap_s_iv = 1;
                let mut ap_r_iv = 1;
                let mut ap_d_iv = 0;
                let mut ap_clken_iv = 0;
                for pinid in netlistdb.cell2pin.iter_set(cellid) {
                    let pin_iv = aig.pin2aigpin_iv[pinid];
                    match netlistdb.pinnames[pinid].1.as_str() {
                        "D" => ap_d_iv = pin_iv,
                        "S" => ap_s_iv = pin_iv,
                        "R" => ap_r_iv = pin_iv,
                        "CLK" => ap_clken_iv = aig.trace_clock_pin(netlistdb, pinid),
                        _ => {}
                    }
                }
                let mut d_in = ap_d_iv;

                d_in = aig.add_and_gate(d_in ^ 1, ap_s_iv) ^ 1;
                ap_clken_iv = aig.add_and_gate(ap_clken_iv ^ 1, ap_s_iv) ^ 1;
                d_in = aig.add_and_gate(d_in, ap_r_iv);
                ap_clken_iv = aig.add_and_gate(ap_clken_iv ^ 1, ap_r_iv) ^ 1;
                let dff = aig.dffs.entry(cellid).or_default();
                dff.en = ap_clken_iv;
                dff.d_iv = d_in;
                assert_ne!(dff.q, 0);
                if dff.d_iv >= 2 {
                    aig.comb_outputs.insert(dff.d_iv >> 1);
                }
            }
            else if netlistdb.celltypes[cellid].as_str() == "$__RAMGEM_SYNC_" {
                let mut sram = aig.srams.entry(cellid).or_default().clone();
                let mut write_clken_iv = 0;
                for pinid in netlistdb.cell2pin.iter_set(cellid) {
                    let bit = netlistdb.pinnames[pinid].2.map(|i| i as usize);
                    let pin_iv = aig.pin2aigpin_iv[pinid];
                    match netlistdb.pinnames[pinid].1.as_str() {

                        "PORT_R_ADDR" => {
                            sram.port_r_addr_iv[bit.unwrap()] = pin_iv;
                            if pin_iv >= 2 {
                                aig.comb_outputs.insert(pin_iv >> 1);
                            }
                        },
                        "PORT_R_CLK" => {
                            sram.port_r_en_iv = aig.trace_clock_pin(netlistdb, pinid);
                        },
                        "PORT_W_ADDR" => {
                            sram.port_w_addr_iv[bit.unwrap()] = pin_iv;
                            if pin_iv >= 2 {
                                aig.comb_outputs.insert(pin_iv >> 1);
                            }
                        }
                        "PORT_W_CLK" => {
                            write_clken_iv = aig.trace_clock_pin(netlistdb, pinid);
                        },
                        "PORT_W_WR_DATA" => {
                            sram.port_w_wr_data_iv[bit.unwrap()] = pin_iv;
                            if pin_iv >= 2 {
                                aig.comb_outputs.insert(pin_iv >> 1);
                            }
                        },
                        "PORT_W_WR_EN" => {
                            sram.port_w_wr_en_iv[bit.unwrap()] = pin_iv;
                        },
                        _ => {}
                    }
                }
                for i in 0..32 {
                    let or_en = sram.port_w_wr_en_iv[i];
                    let or_en = aig.add_and_gate(
                        or_en, write_clken_iv
                    );
                    sram.port_w_wr_en_iv[i] = or_en;
                    if or_en >= 2 {
                        aig.comb_outputs.insert(or_en >> 1);
                    }
                }
                *aig.srams.get_mut(&cellid).unwrap() = sram;
            }
        }

        aig.fanouts_start = vec![0; aig.num_aigpins + 2];
        for (_i, driver) in aig.drivers.iter().enumerate() {
            if let DriverType::AndGate(a, b) = *driver {
                if (a >> 1) != 0 {
                    aig.fanouts_start[a >> 1] += 1;
                }
                if (b >> 1) != 0 {
                    aig.fanouts_start[b >> 1] += 1;
                }
            }
        }
        for i in 1..aig.num_aigpins + 2 {
            aig.fanouts_start[i] += aig.fanouts_start[i - 1];
        }
        aig.fanouts = vec![0; aig.fanouts_start[aig.num_aigpins + 1]];
        for (i, driver) in aig.drivers.iter().enumerate() {
            if let DriverType::AndGate(a, b) = *driver {
                if (a >> 1) != 0 {
                    let st = aig.fanouts_start[a >> 1] - 1;
                    aig.fanouts_start[a >> 1] = st;
                    aig.fanouts[st] = i;
                }
                if (b >> 1) != 0 {
                    let st = aig.fanouts_start[b >> 1] - 1;
                    aig.fanouts_start[b >> 1] = st;
                    aig.fanouts[st] = i;
                }
            }
        }

        aig
    }

    pub fn topo_traverse_generic(
        &self,
        endpoints: Option<&Vec<usize>>,
        is_primary_input: Option<&IndexSet<usize>>,
    ) -> Vec<usize> {
        let mut vis = IndexSet::new();
        let mut ret = Vec::new();
        fn dfs_topo(aig: &AIG, vis: &mut IndexSet<usize>, ret: &mut Vec<usize>, is_primary_input: Option<&IndexSet<usize>>, u: usize) {
            if vis.contains(&u) {
                return
            }
            vis.insert(u);
            if let DriverType::AndGate(a, b) = aig.drivers[u] {
                if is_primary_input.map(|s| s.contains(&u)) != Some(true) {
                    if (a >> 1) != 0 {
                        dfs_topo(aig, vis, ret, is_primary_input, a >> 1);
                    }
                    if (b >> 1) != 0 {
                        dfs_topo(aig, vis, ret, is_primary_input, b >> 1);
                    }
                }
            }
            ret.push(u);
        }
        if let Some(endpoints) = endpoints {
            for &endpoint in endpoints {
                dfs_topo(self, &mut vis, &mut ret, is_primary_input, endpoint);
            }
        }
        else {
            for &endpoint in &self.comb_outputs {
                dfs_topo(self, &mut vis, &mut ret, is_primary_input, endpoint);
            }
        }
        ret
    }
}
