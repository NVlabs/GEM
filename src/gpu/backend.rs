// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use ulib::UVec;

/// Backend contract for GEM v1 flattened-script execution.
pub trait GpuBackendV1 {
    fn synchronize(&self);

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
    );
}

#[cfg(feature = "cuda")]
const _: fn() = || {
    fn assert_backend_impl<T: GpuBackendV1>() {}
    assert_backend_impl::<crate::gpu::cuda_backend::CudaBackend>();
};

#[cfg(feature = "metal")]
const _: fn() = || {
    fn assert_backend_impl<T: GpuBackendV1>() {}
    assert_backend_impl::<crate::gpu::metal_backend::MetalBackend>();
};
