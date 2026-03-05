// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::gpu::backend::GpuBackendV1;
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use ulib::{AsUPtr, AsUPtrMut, Device, UVec};

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct MetalSimStatsV1 {
    pub dispatch_count: u64,
    pub gpu_dispatch_count: u64,
    pub encode_ns: u64,
    pub wait_ns: u64,
    pub total_ns: u64,
}

unsafe extern "C" {
    fn simulate_v1_noninteractive_simple_scan_metal(
        metallib_path: *const std::os::raw::c_char,
        num_blocks: usize,
        num_major_stages: usize,
        blocks_start: *const usize,
        blocks_data: *const u32,
        sram_data: *mut u32,
        sram_size: usize,
        num_cycles: usize,
        state_size: usize,
        states_noninteractive: *mut u32,
        stats_out: *mut MetalSimStatsV1,
    ) -> i32;
}

pub struct MetalBackend {
    metallib_path: PathBuf,
    last_stats: Mutex<MetalSimStatsV1>,
}

impl MetalBackend {
    pub fn new() -> Result<Self, String> {
        if !cfg!(target_os = "macos") {
            return Err("Metal backend requires macOS.".to_string());
        }
        if !cfg!(target_arch = "aarch64") {
            return Err(
                "Metal backend currently supports Apple Silicon (aarch64) only.".to_string(),
            );
        }
        let check = std::process::Command::new("xcrun")
            .arg("-f")
            .arg("metal")
            .output();
        match check {
            Ok(out) if out.status.success() => Ok(()),
            Ok(_) => Err(
                "Metal toolchain not found: `xcrun -f metal` failed. Install Xcode command line tools."
                    .to_string(),
            ),
            Err(err) => Err(format!(
                "Unable to run `xcrun` for Metal toolchain probe: {err}"
            )),
        }?;

        let metallib_path = Self::compile_kernel_library()?;
        Ok(Self {
            metallib_path,
            last_stats: Mutex::new(MetalSimStatsV1::default()),
        })
    }

    pub fn metallib_path(&self) -> &Path {
        &self.metallib_path
    }

    pub fn last_stats(&self) -> MetalSimStatsV1 {
        *self.last_stats.lock().expect("failed to lock Metal stats")
    }

    fn compile_kernel_library() -> Result<PathBuf, String> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let metal_src = manifest_dir.join("msrc/kernel_v1.metal");
        if !metal_src.exists() {
            return Err(format!(
                "Metal kernel source not found: {}",
                metal_src.display()
            ));
        }

        let out_dir = manifest_dir.join("target").join("metal");
        std::fs::create_dir_all(&out_dir).map_err(|err| {
            format!(
                "Failed to create metal output directory {}: {err}",
                out_dir.display()
            )
        })?;

        let air_path = out_dir.join("kernel_v1.air");
        let metallib_path = out_dir.join("kernel_v1.metallib");

        let metal_out = Command::new("xcrun")
            .arg("metal")
            .arg("-std=metal3.1")
            .arg("-O3")
            .arg("-c")
            .arg(&metal_src)
            .arg("-o")
            .arg(&air_path)
            .output()
            .map_err(|err| format!("Failed to invoke `xcrun metal`: {err}"))?;
        if !metal_out.status.success() {
            return Err(format!(
                "`xcrun metal` failed: {}",
                String::from_utf8_lossy(&metal_out.stderr)
            ));
        }

        let metallib_out = Command::new("xcrun")
            .arg("metallib")
            .arg(&air_path)
            .arg("-o")
            .arg(&metallib_path)
            .output()
            .map_err(|err| format!("Failed to invoke `xcrun metallib`: {err}"))?;
        if !metallib_out.status.success() {
            return Err(format!(
                "`xcrun metallib` failed: {}",
                String::from_utf8_lossy(&metallib_out.stderr)
            ));
        }

        Ok(metallib_path)
    }
}

impl GpuBackendV1 for MetalBackend {
    fn synchronize(&self) {}

    fn simulate_v1_noninteractive_simple_scan(
        &self,
        num_blocks: usize,
        num_major_stages: usize,
        blocks_start: &UVec<usize>,
        blocks_data: &UVec<u32>,
        sram_data: &mut UVec<u32>,
        num_cycles: usize,
        state_size: usize,
        states_noninteractive: &mut UVec<u32>,
    ) {
        let metallib = CString::new(self.metallib_path.to_string_lossy().to_string())
            .expect("metallib path contains interior NUL byte");
        let mut stats = MetalSimStatsV1::default();
        let status = unsafe {
            simulate_v1_noninteractive_simple_scan_metal(
                metallib.as_ptr(),
                num_blocks,
                num_major_stages,
                blocks_start.as_uptr(Device::CPU),
                blocks_data.as_uptr(Device::CPU),
                sram_data.as_mut_uptr(Device::CPU),
                sram_data.len(),
                num_cycles,
                state_size,
                states_noninteractive.as_mut_uptr(Device::CPU),
                &mut stats as *mut MetalSimStatsV1,
            )
        };
        *self.last_stats.lock().expect("failed to lock Metal stats") = stats;
        if status != 0 {
            panic!("Metal dispatch failed with status code {status}");
        }
        if std::env::var("GEM_METAL_PROFILE")
            .map(|v| v != "0")
            .unwrap_or(false)
        {
            let total_ms = stats.total_ns as f64 / 1_000_000.0;
            let encode_ms = stats.encode_ns as f64 / 1_000_000.0;
            let wait_ms = stats.wait_ns as f64 / 1_000_000.0;
            let cycles_per_sec = if stats.total_ns == 0 {
                0.0
            } else {
                num_cycles as f64 * 1_000_000_000.0 / stats.total_ns as f64
            };
            eprintln!(
                "[gem-metal] logical_dispatches={} gpu_dispatches={} encode_ms={:.3} wait_ms={:.3} total_ms={:.3} cycles_per_sec={:.2}",
                stats.dispatch_count,
                stats.gpu_dispatch_count,
                encode_ms,
                wait_ms,
                total_ms,
                cycles_per_sec
            );
        }
    }
}
