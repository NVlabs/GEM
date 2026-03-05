// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

struct KernelParamsV1 {
  uint64_t num_blocks;
  uint64_t num_major_stages;
  uint64_t state_size;
  uint64_t cycle_i;
  uint64_t cycle_count;
  uint64_t stage_i;
  uint64_t sram_size;
};

struct MetalSimStatsV1 {
  uint64_t dispatch_count;
  uint64_t gpu_dispatch_count;
  uint64_t encode_ns;
  uint64_t wait_ns;
  uint64_t total_ns;
};

static id<MTLBuffer> make_shared_copy_buffer(
  id<MTLDevice> device,
  const void *src,
  size_t bytes
) {
  if (bytes == 0) {
    return [device newBufferWithLength:sizeof(uint32_t)
                               options:MTLResourceStorageModeShared];
  }
  return [device newBufferWithBytes:src
                             length:bytes
                            options:MTLResourceStorageModeShared];
}

static inline uint64_t elapsed_ns(
  const std::chrono::steady_clock::time_point &st,
  const std::chrono::steady_clock::time_point &ed
) {
  return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
           ed - st)
    .count();
}

extern "C" int simulate_v1_noninteractive_simple_scan_metal(
  const char *metallib_path,
  size_t num_blocks,
  size_t num_major_stages,
  const size_t *blocks_start,
  const uint32_t *blocks_data,
  uint32_t *sram_data,
  size_t sram_size,
  size_t num_cycles,
  size_t state_size,
  uint32_t *states_noninteractive,
  MetalSimStatsV1 *stats_out
) {
  using std::chrono::steady_clock;
  if (stats_out != nullptr) {
    memset(stats_out, 0, sizeof(MetalSimStatsV1));
  }

  @autoreleasepool {
    const auto total_t0 = steady_clock::now();

    if (metallib_path == nullptr) {
      fprintf(stderr, "[gem-metal] metallib path is null\n");
      return 1;
    }
    if (blocks_start == nullptr || blocks_data == nullptr ||
        states_noninteractive == nullptr) {
      fprintf(stderr, "[gem-metal] null required input pointer\n");
      return 2;
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      fprintf(stderr, "[gem-metal] failed to create default Metal device\n");
      return 3;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
      fprintf(stderr, "[gem-metal] failed to create Metal command queue\n");
      return 4;
    }

    NSError *error = nil;
    NSString *libPath = [NSString stringWithUTF8String:metallib_path];
    if (libPath == nil) {
      fprintf(stderr, "[gem-metal] invalid metallib path string\n");
      return 5;
    }

    NSURL *libURL = [NSURL fileURLWithPath:libPath];
    id<MTLLibrary> library = [device newLibraryWithURL:libURL error:&error];
    if (library == nil) {
      fprintf(stderr, "[gem-metal] failed to load metallib: %s\n",
              error.localizedDescription.UTF8String);
      return 6;
    }

    id<MTLFunction> function =
      [library newFunctionWithName:@"simulate_v1_noninteractive_simple_scan"];
    if (function == nil) {
      fprintf(stderr, "[gem-metal] kernel function not found in metallib\n");
      return 7;
    }

    id<MTLComputePipelineState> pipeline =
      [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
      fprintf(stderr, "[gem-metal] failed to create compute pipeline: %s\n",
              error.localizedDescription.UTF8String);
      return 8;
    }

    const size_t blocks_start_count = num_blocks * num_major_stages + 1;
    const size_t blocks_start_bytes = blocks_start_count * sizeof(size_t);
    const size_t blocks_data_count = blocks_start[blocks_start_count - 1];
    const size_t blocks_data_bytes = blocks_data_count * sizeof(uint32_t);
    const size_t states_count = (num_cycles + 1) * state_size;
    const size_t states_bytes = states_count * sizeof(uint32_t);
    const size_t sram_bytes = sram_size * sizeof(uint32_t);

    id<MTLBuffer> blocks_start_buffer =
      make_shared_copy_buffer(device, blocks_start, blocks_start_bytes);
    id<MTLBuffer> blocks_data_buffer =
      make_shared_copy_buffer(device, blocks_data, blocks_data_bytes);
    id<MTLBuffer> states_buffer =
      make_shared_copy_buffer(device, states_noninteractive, states_bytes);
    id<MTLBuffer> sram_buffer =
      make_shared_copy_buffer(device, sram_data, sram_bytes);

    if (blocks_start_buffer == nil || blocks_data_buffer == nil ||
        states_buffer == nil || sram_buffer == nil) {
      fprintf(
        stderr,
        "[gem-metal] failed to allocate one or more MTLBuffer objects\n");
      return 9;
    }

    const NSUInteger blocks_grid =
      (NSUInteger)(num_blocks == 0 ? 1 : num_blocks);
    const NSUInteger tg_width = 256;
    if (pipeline.maxTotalThreadsPerThreadgroup < tg_width) {
      fprintf(
        stderr,
        "[gem-metal] pipeline supports only %lu threads/threadgroup, need %lu\n",
        (unsigned long)pipeline.maxTotalThreadsPerThreadgroup,
        (unsigned long)tg_width);
      return 13;
    }

    MTLSize tg_count = MTLSizeMake(blocks_grid, 1, 1);
    MTLSize tg = MTLSizeMake(tg_width, 1, 1);

    const size_t logical_dispatches = num_cycles * num_major_stages;
    const size_t kDispatchesPerCommandBuffer = 4096;
    size_t dispatch_linear = 0;
    size_t gpu_dispatches = 0;
    uint64_t encode_ns_acc = 0;
    uint64_t wait_ns_acc = 0;

    if (num_major_stages == 1 && num_cycles > 0) {
      id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
      if (commandBuffer == nil) {
        fprintf(stderr, "[gem-metal] failed to create command buffer\n");
        return 10;
      }

      id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
      if (encoder == nil) {
        fprintf(stderr, "[gem-metal] failed to create command encoder\n");
        return 11;
      }

      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:blocks_start_buffer offset:0 atIndex:0];
      [encoder setBuffer:blocks_data_buffer offset:0 atIndex:1];
      [encoder setBuffer:sram_buffer offset:0 atIndex:2];
      [encoder setBuffer:states_buffer offset:0 atIndex:3];

      const auto encode_t0 = steady_clock::now();
      KernelParamsV1 params;
      params.num_blocks = (uint64_t)num_blocks;
      params.num_major_stages = (uint64_t)num_major_stages;
      params.state_size = (uint64_t)state_size;
      params.cycle_i = 0;
      params.cycle_count = (uint64_t)num_cycles;
      params.stage_i = 0;
      params.sram_size = (uint64_t)sram_size;
      [encoder setBytes:&params length:sizeof(KernelParamsV1) atIndex:4];
      [encoder dispatchThreadgroups:tg_count threadsPerThreadgroup:tg];
      dispatch_linear = logical_dispatches;
      gpu_dispatches += 1;
      [encoder endEncoding];
      const auto encode_t1 = steady_clock::now();
      encode_ns_acc += elapsed_ns(encode_t0, encode_t1);

      const auto wait_t0 = steady_clock::now();
      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];
      const auto wait_t1 = steady_clock::now();
      wait_ns_acc += elapsed_ns(wait_t0, wait_t1);

      if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
        fprintf(
          stderr,
          "[gem-metal] command buffer failed at cycle=%zu stage=%zu status=%lu\n",
          (size_t)0,
          (size_t)0,
          (unsigned long)commandBuffer.status);
        return 12;
      }
    } else {
      while (dispatch_linear < logical_dispatches) {
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        if (commandBuffer == nil) {
          fprintf(stderr, "[gem-metal] failed to create command buffer\n");
          return 10;
        }

        id<MTLComputeCommandEncoder> encoder =
          [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
          fprintf(stderr, "[gem-metal] failed to create command encoder\n");
          return 11;
        }

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:blocks_start_buffer offset:0 atIndex:0];
        [encoder setBuffer:blocks_data_buffer offset:0 atIndex:1];
        [encoder setBuffer:sram_buffer offset:0 atIndex:2];
        [encoder setBuffer:states_buffer offset:0 atIndex:3];

        const size_t chunk_end =
          std::min(dispatch_linear + kDispatchesPerCommandBuffer,
                   logical_dispatches);
        const auto encode_t0 = steady_clock::now();
        while (dispatch_linear < chunk_end) {
          const size_t cycle_i = dispatch_linear / num_major_stages;
          const size_t stage_i = dispatch_linear % num_major_stages;

          KernelParamsV1 params;
          params.num_blocks = (uint64_t)num_blocks;
          params.num_major_stages = (uint64_t)num_major_stages;
          params.state_size = (uint64_t)state_size;
          params.cycle_i = (uint64_t)cycle_i;
          params.cycle_count = 1;
          params.stage_i = (uint64_t)stage_i;
          params.sram_size = (uint64_t)sram_size;

          [encoder setBytes:&params length:sizeof(KernelParamsV1) atIndex:4];
          [encoder dispatchThreadgroups:tg_count threadsPerThreadgroup:tg];
          gpu_dispatches += 1;
          ++dispatch_linear;
        }
        [encoder endEncoding];
        const auto encode_t1 = steady_clock::now();
        encode_ns_acc += elapsed_ns(encode_t0, encode_t1);

        const auto wait_t0 = steady_clock::now();
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        const auto wait_t1 = steady_clock::now();
        wait_ns_acc += elapsed_ns(wait_t0, wait_t1);

        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
          const size_t bad_dispatch = dispatch_linear - 1;
          const size_t bad_cycle = bad_dispatch / num_major_stages;
          const size_t bad_stage = bad_dispatch % num_major_stages;
          fprintf(
            stderr,
            "[gem-metal] command buffer failed at cycle=%zu stage=%zu status=%lu\n",
            bad_cycle,
            bad_stage,
            (unsigned long)commandBuffer.status);
          return 12;
        }
      }
    }

    if (states_bytes > 0) {
      memcpy(states_noninteractive, [states_buffer contents], states_bytes);
    }
    if (sram_bytes > 0) {
      memcpy(sram_data, [sram_buffer contents], sram_bytes);
    }

    const auto total_t1 = steady_clock::now();
    if (stats_out != nullptr) {
      stats_out->dispatch_count = (uint64_t)logical_dispatches;
      stats_out->gpu_dispatch_count = (uint64_t)gpu_dispatches;
      stats_out->encode_ns = encode_ns_acc;
      stats_out->wait_ns = wait_ns_acc;
      stats_out->total_ns = elapsed_ns(total_t0, total_t1);
    }
  }

  return 0;
}
