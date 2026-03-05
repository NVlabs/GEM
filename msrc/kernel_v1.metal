// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

constant uint GEM_NUM_THREADS_V1 = 256u;
constant uint GEM_SRAM_WORDS = 1u << 13;

struct KernelParamsV1 {
  ulong num_blocks;
  ulong num_major_stages;
  ulong state_size;
  ulong cycle_i;
  ulong cycle_count;
  ulong stage_i;
  ulong sram_size;
};

inline void simulate_block_v1(
  device const uint *script,
  ulong script_size,
  device const uint *input_state,
  device uint *output_state,
  device uint *sram_data,
  ulong sram_size,
  uint tid,
  threadgroup uint *tg_writeout_hooks,
  threadgroup uint *tg_writeouts,
  threadgroup uint *tg_state,
  threadgroup uint *tg_hier_inputs,
  threadgroup uint *tg_hier_flag_xora,
  threadgroup uint *tg_hier_flag_xorb,
  threadgroup uint *tg_hier_flag_orb,
  threadgroup uint *tg_sram_duplicate
) {
  ulong script_pi = 0;
  while (true) {
    if (script_pi + GEM_NUM_THREADS_V1 > script_size) {
      return;
    }

    uint num_stages = script[script_pi + 0];
    uint is_last_part = script[script_pi + 1];
    uint num_ios = script[script_pi + 2];
    uint io_offset = script[script_pi + 3];
    uint num_srams = script[script_pi + 4];
    uint sram_offset = script[script_pi + 5];
    uint num_global_read_rounds = script[script_pi + 6];
    uint num_output_duplicates = script[script_pi + 7];

    if (num_ios > GEM_NUM_THREADS_V1) {
      return;
    }

    if (tid < 128u) {
      uint t = script[script_pi + 128u + tid];
      tg_writeout_hooks[tid * 2u] = t & 0xffffu;
      tg_writeout_hooks[tid * 2u + 1u] = t >> 16;
    }

    script_pi += GEM_NUM_THREADS_V1;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (num_stages == 0u) {
      break;
    }

    tg_writeouts[tid] = 0u;
    tg_state[tid] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint gr_i = 0; gr_i < num_global_read_rounds; ++gr_i) {
      if (script_pi + GEM_NUM_THREADS_V1 * 2u > script_size) {
        return;
      }
      uint idx = script[script_pi + tid * 2u + 0u];
      uint mask = script[script_pi + tid * 2u + 1u];
      uint cur_state = tg_state[tid];
      if (mask != 0u) {
        uint value = ((idx >> 31) == 0u)
          ? input_state[(ulong)idx]
          : output_state[(ulong)(idx ^ (1u << 31))];
        while (mask != 0u) {
          cur_state <<= 1u;
          uint lowbit = mask & (0u - mask);
          if ((value & lowbit) != 0u) {
            cur_state |= 1u;
          }
          mask ^= lowbit;
        }
      }
      tg_state[tid] = cur_state;
      script_pi += GEM_NUM_THREADS_V1 * 2u;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint bs_i = 0; bs_i < num_stages; ++bs_i) {
      if (script_pi + GEM_NUM_THREADS_V1 * 4u * 5u > script_size) {
        return;
      }

      uint hier_input = 0u;
      ulong stage_base = script_pi;
      for (uint k_outer = 0; k_outer < 4u; ++k_outer) {
        ulong base = stage_base + (ulong)k_outer * GEM_NUM_THREADS_V1 * 4u + (ulong)tid * 4u;
        for (uint k_inner = 0; k_inner < 4u; ++k_inner) {
          uint k = k_outer * 4u + k_inner;
          uint t_shuffle = script[base + k_inner];
          uint t_shuffle_1_idx = t_shuffle & 0xffffu;
          uint t_shuffle_2_idx = t_shuffle >> 16;
          hier_input |=
            ((tg_state[t_shuffle_1_idx >> 5] >> (t_shuffle_1_idx & 31u)) & 1u) << (k * 2u);
          hier_input |=
            ((tg_state[t_shuffle_2_idx >> 5] >> (t_shuffle_2_idx & 31u)) & 1u)
            << (k * 2u + 1u);
        }
      }
      tg_hier_inputs[tid] = hier_input;

      ulong flags_base = stage_base + GEM_NUM_THREADS_V1 * 4u * 4u;
      ulong fi = flags_base + (ulong)tid * 4u;
      tg_hier_flag_xora[tid] = script[fi + 0u];
      tg_hier_flag_xorb[tid] = script[fi + 1u];
      tg_hier_flag_orb[tid] = script[fi + 2u];
      script_pi = stage_base + GEM_NUM_THREADS_V1 * 4u * 5u;

      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (tid < 128u) {
        uint a = tg_hier_inputs[tid];
        uint b = tg_hier_inputs[128u + tid];
        uint xora = tg_hier_flag_xora[128u + tid];
        uint xorb = tg_hier_flag_xorb[128u + tid];
        uint orb = tg_hier_flag_orb[128u + tid];
        tg_hier_inputs[128u + tid] = (a ^ xora) & ((b ^ xorb) | orb);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      for (uint hi = 1u; hi <= 7u; ++hi) {
        uint hier_width = 1u << (7u - hi);
        if (tid < hier_width) {
          uint a = tg_hier_inputs[hier_width * 2u + tid];
          uint b = tg_hier_inputs[hier_width * 3u + tid];
          uint xora = tg_hier_flag_xora[hier_width + tid];
          uint xorb = tg_hier_flag_xorb[hier_width + tid];
          uint orb = tg_hier_flag_orb[hier_width + tid];
          tg_hier_inputs[hier_width + tid] = (a ^ xora) & ((b ^ xorb) | orb);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }

      if (tid == 0u) {
        uint v1 = tg_hier_inputs[1u];
        uint xora = tg_hier_flag_xora[0u];
        uint xorb = tg_hier_flag_xorb[0u];
        uint orb = tg_hier_flag_orb[0u];
        uint r8 = ((v1 << 16) ^ xora) & ((v1 ^ xorb) | orb) & 0xffff0000u;
        uint r9 = ((r8 >> 8) ^ xora) & (((r8 >> 16) ^ xorb) | orb) & 0xff00u;
        uint r10 = ((r9 >> 4) ^ xora) & (((r9 >> 8) ^ xorb) | orb) & 0xf0u;
        uint r11 = ((r10 >> 2) ^ xora) & (((r10 >> 4) ^ xorb) | orb) & 0x0cu;
        uint r12 = ((r11 >> 1) ^ xora) & (((r11 >> 2) ^ xorb) | orb) & 0x02u;
        tg_hier_inputs[0u] = r8 | r9 | r10 | r11 | r12;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      tg_state[tid] = tg_hier_inputs[tid];
      uint hook_i = tg_writeout_hooks[tid];
      if ((hook_i >> 8) == bs_i) {
        tg_writeouts[tid] = tg_state[hook_i & 255u];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint sram_duplicate_count = num_srams * 4u + num_output_duplicates;
    if (sram_duplicate_count > GEM_NUM_THREADS_V1) {
      return;
    }

    if (script_pi + GEM_NUM_THREADS_V1 * 4u * 10u > script_size) {
      return;
    }

    ulong sram_perm_base = script_pi;
    ulong sram_inv_base = sram_perm_base + GEM_NUM_THREADS_V1 * 4u * 4u;
    ulong clken_perm_base = sram_inv_base + GEM_NUM_THREADS_V1 * 4u;
    ulong clken_inv_base = clken_perm_base + GEM_NUM_THREADS_V1 * 4u * 4u;
    script_pi += GEM_NUM_THREADS_V1 * 4u * 10u;

    uint sram_duplicate_t = 0u;
    if (tid < sram_duplicate_count) {
      for (uint k_outer = 0; k_outer < 4u; ++k_outer) {
        ulong base = sram_perm_base + (ulong)k_outer * GEM_NUM_THREADS_V1 * 4u;
        ulong base_i = base + (ulong)tid * 4u;
        for (uint k_inner = 0; k_inner < 4u; ++k_inner) {
          uint k = k_outer * 4u + k_inner;
          uint t_shuffle = script[base_i + k_inner];
          uint t_shuffle_1_idx = t_shuffle & 0xffffu;
          uint t_shuffle_2_idx = t_shuffle >> 16;
          sram_duplicate_t |=
            ((tg_writeouts[t_shuffle_1_idx >> 5] >> (t_shuffle_1_idx & 31u)) & 1u)
            << (k * 2u);
          sram_duplicate_t |=
            ((tg_writeouts[t_shuffle_2_idx >> 5] >> (t_shuffle_2_idx & 31u)) & 1u)
            << (k * 2u + 1u);
        }
      }

      ulong inv_i = sram_inv_base + (ulong)tid * 4u;
      sram_duplicate_t = (sram_duplicate_t & ~script[inv_i + 1u]) ^ script[inv_i + 0u];
    }
    tg_sram_duplicate[tid] = sram_duplicate_t;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_srams * 4u && (tid & 3u) == 0u) {
      uint sram_i = tid >> 2;
      uint addrs = tg_sram_duplicate[tid];
      uint port_r_addr_iv = addrs & 0xffffu;
      uint port_w_addr_iv = addrs >> 16;
      uint port_w_wr_en = tg_sram_duplicate[tid + 1u];
      uint port_w_wr_data_iv = tg_sram_duplicate[tid + 2u];

      ulong sram_st = (ulong)sram_offset + (ulong)sram_i * (ulong)GEM_SRAM_WORDS;
      ulong sram_ed = sram_st + (ulong)GEM_SRAM_WORDS;
      if (sram_ed > sram_size) {
        return;
      }

      device uint *ram = sram_data + sram_st;
      uint r = ram[port_r_addr_iv];
      uint w0 = ram[port_w_addr_iv];
      tg_writeouts[num_ios - num_srams + sram_i] = r;
      ram[port_w_addr_iv] = (w0 & ~port_w_wr_en) | (port_w_wr_data_iv & port_w_wr_en);
    }
    if (tid < num_output_duplicates) {
      tg_writeouts[num_ios - num_srams - num_output_duplicates + tid] =
        tg_sram_duplicate[num_srams * 4u + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < num_ios) {
      uint clken_perm = 0u;
      for (uint k_outer = 0; k_outer < 4u; ++k_outer) {
        ulong base_i = clken_perm_base + (ulong)k_outer * GEM_NUM_THREADS_V1 * 4u + (ulong)tid * 4u;
        for (uint k_inner = 0; k_inner < 4u; ++k_inner) {
          uint k = k_outer * 4u + k_inner;
          uint t_shuffle = script[base_i + k_inner];
          uint t_shuffle_1_idx = t_shuffle & 0xffffu;
          uint t_shuffle_2_idx = t_shuffle >> 16;
          clken_perm |=
            ((tg_writeouts[t_shuffle_1_idx >> 5] >> (t_shuffle_1_idx & 31u)) & 1u) << (k * 2u);
          clken_perm |=
            ((tg_writeouts[t_shuffle_2_idx >> 5] >> (t_shuffle_2_idx & 31u)) & 1u)
            << (k * 2u + 1u);
        }
      }

      ulong inv_i = clken_inv_base + (ulong)tid * 4u;
      clken_perm = (clken_perm & ~script[inv_i + 1u]) ^ script[inv_i + 0u];
      uint writeout_inv = tg_writeouts[tid] ^ script[inv_i + 2u];
      uint old_wo = input_state[(ulong)io_offset + tid];
      output_state[(ulong)io_offset + tid] = (old_wo & ~clken_perm) | (writeout_inv & clken_perm);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (is_last_part != 0u) {
      break;
    }
  }
}

kernel void simulate_v1_noninteractive_simple_scan(
  device const ulong *blocks_start [[buffer(0)]],
  device const uint *blocks_data [[buffer(1)]],
  device uint *sram_data [[buffer(2)]],
  device uint *states_noninteractive [[buffer(3)]],
  constant KernelParamsV1 &params [[buffer(4)]],
  uint3 tg_pos [[threadgroup_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]]
) {
  uint block_i = tg_pos.x;
  if ((ulong)block_i >= params.num_blocks) {
    return;
  }

  ulong script_block_i = params.stage_i * params.num_blocks + (ulong)block_i;
  ulong script_start = blocks_start[script_block_i];
  ulong script_end = blocks_start[script_block_i + 1u];
  if (script_end < script_start) {
    return;
  }

  ulong script_size = script_end - script_start;
  threadgroup uint tg_writeout_hooks[GEM_NUM_THREADS_V1];
  threadgroup uint tg_writeouts[GEM_NUM_THREADS_V1];
  threadgroup uint tg_state[GEM_NUM_THREADS_V1];
  threadgroup uint tg_hier_inputs[GEM_NUM_THREADS_V1];
  threadgroup uint tg_hier_flag_xora[GEM_NUM_THREADS_V1];
  threadgroup uint tg_hier_flag_xorb[GEM_NUM_THREADS_V1];
  threadgroup uint tg_hier_flag_orb[GEM_NUM_THREADS_V1];
  threadgroup uint tg_sram_duplicate[GEM_NUM_THREADS_V1];

  ulong cycle_count = params.cycle_count == 0 ? 1 : params.cycle_count;
  for (ulong cycle_delta = 0; cycle_delta < cycle_count; ++cycle_delta) {
    ulong cycle_i = params.cycle_i + cycle_delta;
    ulong input_offset = cycle_i * params.state_size;
    ulong output_offset = (cycle_i + 1u) * params.state_size;

    simulate_block_v1(
      blocks_data + script_start,
      script_size,
      states_noninteractive + input_offset,
      states_noninteractive + output_offset,
      sram_data,
      params.sram_size,
      tid,
      tg_writeout_hooks,
      tg_writeouts,
      tg_state,
      tg_hier_inputs,
      tg_hier_flag_xora,
      tg_hier_flag_xorb,
      tg_hier_flag_orb,
      tg_sram_duplicate
    );
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
