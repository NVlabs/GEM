// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::gpu::backend::GpuBackendV1;
use ulib::{Device, UVec};

mod ucci {
    include!(concat!(env!("OUT_DIR"), "/uccbind/kernel_v1.rs"));
}

pub struct CudaBackend {
    device: Device,
}

impl CudaBackend {
    pub fn new(device: Device) -> Self {
        Self { device }
    }
}

impl GpuBackendV1 for CudaBackend {
    fn synchronize(&self) {
        self.device.synchronize();
    }

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
        ucci::simulate_v1_noninteractive_simple_scan(
            num_blocks,
            num_major_stages,
            blocks_start,
            blocks_data,
            sram_data,
            num_cycles,
            state_size,
            states_noninteractive,
            self.device,
        );
    }
}
