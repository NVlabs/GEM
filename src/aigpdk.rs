// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! AIGPDK is a special artificial cell library used in GEM.

use netlistdb::{Direction, LeafPinProvider};
use compact_str::CompactString;
use sverilogparse::SVerilogRange;

/// This implements direction and width providers for
/// AIG PDK cells.
///
/// You can use it in netlistdb construction.
pub struct AIGPDKLeafPins();

/// The addr width of an SRAM.
///
/// The word width is always 32.
/// If you change this, make sure to change all other occurences in this
/// project as well as the definitions in PDK libraries.
pub const AIGPDK_SRAM_ADDR_WIDTH: usize = 13;

pub const AIGPDK_SRAM_SIZE: usize = 1 << 13;

impl LeafPinProvider for AIGPDKLeafPins {
    fn direction_of(
        &self,
        macro_name: &CompactString,
        pin_name: &CompactString, pin_idx: Option<isize>
    ) -> Direction {
        match (macro_name.as_str(), pin_name.as_str(), pin_idx) {
            ("INV" | "BUF", "A", None) => Direction::I,
            ("INV" | "BUF", "Y", None) => Direction::O,

            ("AND2_00_0" | "AND2_01_0" | "AND2_10_0" | "AND2_11_0" |
             "AND2_11_1", "A" | "B", None) => Direction::I,
            ("AND2_00_0" | "AND2_01_0" | "AND2_10_0" | "AND2_11_0" |
             "AND2_11_1", "Y", None) => Direction::O,

            ("DFF" | "LATCH", "CLK" | "D", None) => Direction::I,
            ("DFFSR", "CLK" | "D" | "S" | "R", None) => Direction::I,
            ("DFF" | "DFFSR" | "LATCH", "Q", None) => Direction::O,

            ("CKLNQD", "CP" | "E", None) => Direction::I,
            ("CKLNQD", "Q", None) => Direction::O,

            ("$__RAMGEM_ASYNC_", _, _) => {
                panic!("Async RAM (lib cell {}) not supported yet in GEM.", macro_name);
            },

            ("$__RAMGEM_SYNC_",
             "PORT_R_CLK" | "PORT_W_CLK",
             None) => Direction::I,
            ("$__RAMGEM_SYNC_",
             "PORT_R_ADDR" | "PORT_W_ADDR",
             Some(0..=12)) => Direction::I,
            ("$__RAMGEM_SYNC_",
             "PORT_W_WR_EN" | "PORT_W_WR_DATA",
             Some(0..=31)) => Direction::I,
            ("$__RAMGEM_SYNC_",
             "PORT_R_RD_DATA",
             Some(0..=31)) => Direction::O,

            _ => {
                use netlistdb::{GeneralPinName, HierName};
                panic!("Cannot recognize pin type {}, please make sure the verilog netlist is synthesized in GEM's aigpdk.",
                       (HierName::single(macro_name.clone()),
                        pin_name, pin_idx).dbg_fmt_pin());
            }
        }
    }

    fn width_of(
        &self,
        macro_name: &CompactString,
        pin_name: &CompactString
    ) -> Option<SVerilogRange> {
        match (macro_name.as_str(), pin_name.as_str()) {
            ("INV" | "BUF", "A" | "Y") => None,
            ("AND2_00_0" | "AND2_01_0" | "AND2_10_0" | "AND2_11_0" |
             "AND2_11_1", "A" | "B" | "Y") => None,
            ("DFF" | "DFFSR" | "LATCH", "CLK" | "D" | "Q" | "S" | "R") => None,
            ("CKLNQD", "CP" | "E" | "Q") => None,
            ("$__RAMGEM_SYNC_",
             "PORT_R_CLK" | "PORT_W_CLK") => None,
            ("$__RAMGEM_SYNC_",
             "PORT_R_ADDR" | "PORT_W_ADDR")
                => Some(SVerilogRange(12, 0)),
            ("$__RAMGEM_SYNC_",
             "PORT_W_WR_EN" | "PORT_W_WR_DATA" | "PORT_R_RD_DATA")
                => Some(SVerilogRange(31, 0)),
            _ => None
        }
    }
}
