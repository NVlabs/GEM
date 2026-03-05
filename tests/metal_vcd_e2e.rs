// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]

use std::path::PathBuf;
use std::process::Command;

fn fixture_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel)
}

#[test]
fn metal_test_matches_flatten_test_vcd_output() {
    let netlist = fixture_path("baseline/tiny_gatelevel.gv");
    let gemparts = fixture_path("baseline/tiny.gemparts");
    let input_vcd = fixture_path("baseline/tiny_input.vcd");

    let out_dir = fixture_path("target/metal_test_e2e");
    std::fs::create_dir_all(&out_dir).expect("cannot create output directory");
    let cpu_vcd = out_dir.join("tiny_output_cpu.vcd");
    let metal_vcd = out_dir.join("tiny_output_metal.vcd");

    let flatten_status = Command::new(env!("CARGO_BIN_EXE_flatten_test"))
        .arg(&netlist)
        .arg(&gemparts)
        .arg(&input_vcd)
        .arg(&cpu_vcd)
        .status()
        .expect("failed to run flatten_test");
    assert!(flatten_status.success(), "flatten_test failed");

    let metal_status = Command::new(env!("CARGO_BIN_EXE_metal_test"))
        .arg(&netlist)
        .arg(&gemparts)
        .arg(&input_vcd)
        .arg(&metal_vcd)
        .arg("5")
        .status()
        .expect("failed to run metal_test");
    assert!(metal_status.success(), "metal_test failed");

    let cpu = std::fs::read(&cpu_vcd).expect("cannot read cpu vcd");
    let metal = std::fs::read(&metal_vcd).expect("cannot read metal vcd");
    assert_eq!(cpu, metal, "metal_test output differs from flatten_test");
}
