#include <crates/ulib/includes.hpp>
#include <cooperative_groups.h>

struct alignas(8) VectorRead2 {
  u32 c1, c2;

  __device__ __forceinline__ void read(const VectorRead2 *t) {
    *this = *t;
  }
};

struct alignas(16) VectorRead4 {
  u32 c1, c2, c3, c4;

  __device__ __forceinline__ void read(const VectorRead4 *t) {
    *this = *t;
  }
};

__device__ void simulate_block_v1(
  const u32 *__restrict__ script,
  usize script_size,
  const u32 *__restrict__ input_state,
  u32 *__restrict__ output_state,
  u32 *__restrict__ sram_data,
  u32 *__restrict__ shared_metadata,
  u32 *__restrict__ shared_writeouts,
  u32 *__restrict__ shared_state
  )
{
  int script_pi = 0;
  while(true) {
    shared_metadata[threadIdx.x] = script[script_pi + threadIdx.x];
    __syncthreads();
    int num_stages = shared_metadata[0];
    if(!num_stages) {
      script_pi += 256;
      break;
    }
    int is_last_part = shared_metadata[1];
    int num_ios = shared_metadata[2];
    int io_offset = shared_metadata[3];
    int num_srams = shared_metadata[4];
    int sram_offset = shared_metadata[5];
    int num_global_read_rounds = shared_metadata[6];
    int num_output_duplicates = shared_metadata[7];
    u32 writeout_hook_i = shared_metadata[128 + threadIdx.x / 2];
    if(threadIdx.x % 2 == 0) {
      writeout_hook_i = writeout_hook_i & ((1 << 16) - 1);
    }
    else {
      writeout_hook_i = writeout_hook_i >> 16;
    }
    script_pi += 256;

    u32 t_global_rd_state = 0;
    for(int gr_i = 0; gr_i < num_global_read_rounds; ++gr_i) {
      VectorRead2 idx_mask;
      idx_mask.read(((const VectorRead2 *)(script + script_pi)) + threadIdx.x);
      u32 idx = idx_mask.c1;
      u32 mask = idx_mask.c2;
      if(mask) {
        u32 value = input_state[idx];
        while(mask) {
          t_global_rd_state <<= 1;
          u32 lowbit = mask & -mask;
          if(value & lowbit) t_global_rd_state |= 1;
          mask ^= lowbit;
        }
      }
      script_pi += 256 * 2;
    }
    shared_state[threadIdx.x] = t_global_rd_state;
    __syncthreads();

    for(int bs_i = 0; bs_i < num_stages; ++bs_i) {
      u32 hier_input = 0, hier_flag_xora = 0, hier_flag_xorb = 0, hier_flag_orb = 0;
      for(int k_outer = 0; k_outer < 4; ++k_outer) {
        VectorRead4 t_sf;
        t_sf.read(((const VectorRead4 *)(script + script_pi)) + threadIdx.x);
#define GEMV1_SHUF_INPUT_K(k_inner, t_shuffle) {                      \
          u32 k = k_outer * 4 + k_inner;                              \
          u32 t_shuffle_1 = t_shuffle & ((1 << 16) - 1);              \
          u32 t_shuffle_2 = t_shuffle >> 16;                          \
          u32 t_shuffle_1_idx = t_shuffle_1 & ((1 << 13) - 1);        \
          u32 t_shuffle_2_idx = t_shuffle_2 & ((1 << 13) - 1);        \
                                                                      \
          hier_input |= (shared_state[t_shuffle_1_idx >> 5] >>        \
                         (t_shuffle_1_idx & 31) & 1) << (k * 2);      \
          hier_input |= (shared_state[t_shuffle_2_idx >> 5] >>        \
                         (t_shuffle_2_idx & 31) & 1) << (k * 2 + 1);  \
          hier_flag_xora |= (t_shuffle_1 >> 14 & 1) << (k * 2);       \
          hier_flag_xora |= (t_shuffle_2 >> 14 & 1) << (k * 2 + 1);   \
          hier_flag_xorb |= (t_shuffle_1 >> 13 & 1) << (k * 2);       \
          hier_flag_xorb |= (t_shuffle_2 >> 13 & 1) << (k * 2 + 1);   \
          hier_flag_orb |= (t_shuffle_1 >> 15) << (k * 2);            \
          hier_flag_orb |= (t_shuffle_2 >> 15) << (k * 2 + 1);        \
        }
        GEMV1_SHUF_INPUT_K(0, t_sf.c1);
        GEMV1_SHUF_INPUT_K(1, t_sf.c2);
        GEMV1_SHUF_INPUT_K(2, t_sf.c3);
        GEMV1_SHUF_INPUT_K(3, t_sf.c4);
#undef GEMV1_SHUF_INPUT_K
        script_pi += 256 * 4;
      }
      __syncthreads();
      shared_state[threadIdx.x] = hier_input;
      __syncthreads();

      // hier[0]
      if(threadIdx.x >= 128) {
        u32 hier_input_a = shared_state[threadIdx.x - 128];
        u32 hier_input_b = hier_input;
        u32 ret = (hier_input_a ^ hier_flag_xora) & ((hier_input_b ^ hier_flag_xorb) | hier_flag_orb);
        shared_state[threadIdx.x] = ret;
      }
      __syncthreads();

      // hier[1..7]
      for(int hi = 1; hi <= 7; ++hi) {
        int hier_width = 1 << (7 - hi);
        if(threadIdx.x >= hier_width && threadIdx.x < hier_width * 2) {
          u32 hier_input_a = shared_state[threadIdx.x + hier_width];
          u32 hier_input_b = shared_state[threadIdx.x + hier_width * 2];
          u32 ret = (hier_input_a ^ hier_flag_xora) & ((hier_input_b ^ hier_flag_xorb) | hier_flag_orb);
          shared_state[threadIdx.x] = ret;
        }
        __syncthreads();
      }

      // hier[8..12]
      if(threadIdx.x == 0) {
        u32 v1 = shared_state[1];
        u32 r8 = ((v1 << 16) ^ hier_flag_xora) & ((v1 ^ hier_flag_xorb) | hier_flag_orb) & 0xffff0000;
        u32 r9 = ((r8 >> 8) ^ hier_flag_xora) & (((r8 >> 16) ^ hier_flag_xorb) | hier_flag_orb) & 0xff00;
        u32 r10 = ((r9 >> 4) ^ hier_flag_xora) & (((r9 >> 8) ^ hier_flag_xorb) | hier_flag_orb) & 0xf0;
        u32 r11 = ((r10 >> 2) ^ hier_flag_xora) & (((r10 >> 4) ^ hier_flag_xorb) | hier_flag_orb) & 12 /* 0b1100 */;
        u32 r12 = ((r11 >> 1) ^ hier_flag_xora) & (((r11 >> 2) ^ hier_flag_xorb) | hier_flag_orb) & 2 /* 0b10 */;
        shared_state[0] = r8 | r9 | r10 | r11 | r12;
      }
      __syncthreads();

      // write out
      if((writeout_hook_i >> 8) == bs_i) {
        shared_writeouts[threadIdx.x] = shared_state[writeout_hook_i & 255];
      }
    }
    __syncthreads();

    // sram & duplicate permutation
    u32 sram_duplicate_t = 0;
    for(int k_outer = 0; k_outer < 4; ++k_outer) {
      VectorRead4 t_sf;
      t_sf.read(((const VectorRead4 *)(script + script_pi)) + threadIdx.x);
#define GEMV1_SHUF_SRAM_DUPL_K(k_inner, t_shuffle) {          \
        u32 k = k_outer * 4 + k_inner;                        \
        u32 t_shuffle_1 = t_shuffle & ((1 << 16) - 1);        \
        u32 t_shuffle_2 = t_shuffle >> 16;                    \
        u32 t_shuffle_1_idx = t_shuffle_1 & ((1 << 13) - 1);  \
        u32 t_shuffle_2_idx = t_shuffle_2 & ((1 << 13) - 1);  \
                                                              \
        sram_duplicate_t |= (                                 \
          ((shared_writeouts[t_shuffle_1_idx >> 5] >>         \
            (t_shuffle_1_idx & 31) & 1)                       \
           & ~(t_shuffle_1 >> 14)) ^ (t_shuffle_1 >> 13 & 1)  \
          ) << (k * 2);                                       \
        sram_duplicate_t |= (                                 \
          ((shared_writeouts[t_shuffle_2_idx >> 5] >>         \
            (t_shuffle_2_idx & 31) & 1)                       \
           & ~(t_shuffle_2 >> 14)) ^ (t_shuffle_2 >> 13 & 1)  \
          ) << (k * 2 + 1);                                   \
      }
      if(threadIdx.x < num_srams * 4 + num_output_duplicates) {
        GEMV1_SHUF_SRAM_DUPL_K(0, t_sf.c1);
        GEMV1_SHUF_SRAM_DUPL_K(1, t_sf.c2);
        GEMV1_SHUF_SRAM_DUPL_K(2, t_sf.c3);
        GEMV1_SHUF_SRAM_DUPL_K(3, t_sf.c4);
      }
#undef GEMV1_SHUF_SRAM_DUPL_K
      script_pi += 256 * 4;
    }

    if(threadIdx.x < num_srams * 4) {
      u32 addrs = sram_duplicate_t;
      u32 last_tid = 32 + threadIdx.x / 32 * 32;
      u32 mask = (last_tid <= num_srams * 4)
        ? 0xffffffff : (0xffffffff >> (last_tid - num_srams * 4));
      u32 port_w_wr_en = __shfl_down_sync(mask, sram_duplicate_t, 1);
      u32 port_w_wr_data_iv = __shfl_down_sync(mask, sram_duplicate_t, 2);
      if(threadIdx.x % 4 == 0) {
        u32 sram_i = threadIdx.x / 4;
        u32 sram_st = sram_offset + sram_i * (1 << 13);
        // u32 sram_ed = sram_st + (1 << 13);
        u32 port_r_addr_iv = addrs & 0xffff;
        u32 port_w_addr_iv = addrs >> 16;

        u32 *ram = sram_data + sram_st;
        u32 r = ram[port_r_addr_iv], w0 = ram[port_w_addr_iv];
        shared_writeouts[num_ios - num_srams + sram_i] = r;
        ram[port_w_addr_iv] = (w0 & ~port_w_wr_en) | (port_w_wr_data_iv & port_w_wr_en);
      }
    }
    else if(threadIdx.x < num_srams * 4 + num_output_duplicates) {
      shared_writeouts[num_ios - num_srams - num_output_duplicates + (threadIdx.x - num_srams * 4)] = sram_duplicate_t;
    }
    __syncthreads();

    // clock enable permutation
    u32 clken_perm = 0;
    u32 writeout_inv = shared_writeouts[threadIdx.x];
    for(int k_outer = 0; k_outer < 4; ++k_outer) {
      VectorRead4 t_sf;
      t_sf.read(((const VectorRead4 *)(script + script_pi)) + threadIdx.x);
#define GEMV1_SHUF_CLKEN_K(k_inner, t_shuffle) {              \
        u32 k = k_outer * 4 + k_inner;                        \
        u32 t_shuffle_1 = t_shuffle & ((1 << 16) - 1);        \
        u32 t_shuffle_2 = t_shuffle >> 16;                    \
        u32 t_shuffle_1_idx = t_shuffle_1 & ((1 << 13) - 1);  \
        u32 t_shuffle_2_idx = t_shuffle_2 & ((1 << 13) - 1);  \
                                                              \
        clken_perm |= (                                       \
          ((shared_writeouts[t_shuffle_1_idx >> 5] >>         \
            (t_shuffle_1_idx & 31) & 1)                       \
           & ~(t_shuffle_1 >> 14)) ^ (t_shuffle_1 >> 13 & 1)  \
          ) << (k * 2);                                       \
        clken_perm |= (                                       \
          ((shared_writeouts[t_shuffle_2_idx >> 5] >>         \
            (t_shuffle_2_idx & 31) & 1)                       \
           & ~(t_shuffle_2 >> 14)) ^ (t_shuffle_2 >> 13 & 1)  \
          ) << (k * 2 + 1);                                   \
        writeout_inv ^= (t_shuffle_1 >> 15) << (k * 2);       \
        writeout_inv ^= (t_shuffle_2 >> 15) << (k * 2 + 1);   \
      }
      if(threadIdx.x < num_ios) {
        GEMV1_SHUF_CLKEN_K(0, t_sf.c1);
        GEMV1_SHUF_CLKEN_K(1, t_sf.c2);
        GEMV1_SHUF_CLKEN_K(2, t_sf.c3);
        GEMV1_SHUF_CLKEN_K(3, t_sf.c4);
      }
#undef GEMV1_SHUF_CLKEN_K
      script_pi += 256 * 4;
    }

    if(threadIdx.x < num_ios) {
      u32 old_wo = input_state[io_offset + threadIdx.x];
      u32 wo = (old_wo & ~clken_perm) | (writeout_inv & clken_perm);
      output_state[io_offset + threadIdx.x] = wo;
    }

    if(is_last_part) break;
  }
  assert(script_size == script_pi);
}

__global__ void __maxnreg__(128) simulate_v1_noninteractive_simple_scan(
  usize num_blocks,
  const usize *__restrict__ blocks_start,
  const u32 *__restrict__ blocks_data,
  u32 *__restrict__ sram_data,
  usize num_cycles,
  usize state_size,
  u32 *__restrict__ states_noninteractive
  )
{
  assert(num_blocks == gridDim.x);
  assert(256 == blockDim.x);
  __shared__ u32 shared_metadata[256];
  __shared__ u32 shared_writeouts[256];
  __shared__ u32 shared_state[256];
  usize script_start = blocks_start[blockIdx.x];
  usize script_size = blocks_start[blockIdx.x + 1] - script_start;
  for(usize cycle_i = 0; cycle_i < num_cycles; ++cycle_i) {
    simulate_block_v1(
      blocks_data + script_start,
      script_size,
      states_noninteractive + cycle_i * state_size,
      states_noninteractive + (cycle_i + 1) * state_size,
      sram_data,
      shared_metadata, shared_writeouts, shared_state
      );
    cooperative_groups::this_grid().sync();
  }
}
