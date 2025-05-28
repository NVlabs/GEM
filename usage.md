# Getting Started to use GEM

**Caveats**: currently GEM only supports non-interactive testbenches. This means the input to the circuit needs to be a static waveform (e.g., VCD). Registers and clock gates inside the circuit are allowed, but latches and other asynchronous sequential logics are currently unsupported.

**Dataset**: Some (namely, netlists after AIG transformation in Steps 1-2 below, and reference VCDs) input data is available [here](https://drive.google.com/drive/folders/1M42vFoVZhG4ZjyD1hqYD0Hrw8F1rwNXd?usp=drive_link) .

## Step 0. Download the AIG Process Kit
Go to [aigpdk](./aigpdk) directory where you can download `aigpdk.lib`, `aigpdk.v`, and `memlib_yosys.txt`. You will need them later in the flow.

Before continuing, make sure your design contains only synchronous logic.
If your design has clock gates implemented in your RTL code, you need to replace them manually with instantiations to the `CKLNQD` module in `aigpdk.v`.
Also, you are advised to be familiar with where memory blocks (e.g., caches) are implemented in your design so you can check that the memory blocks are mapped correctly later.

## Step 1. Memory Synthesis with Yosys
This step makes use of the open-source [Yosys](https://github.com/YosysHQ/yosys) synthesizer to recognize and map the memory blocks automatically.

Download and compile the latest version of Yosys. Then run yosys shell with the following synthesis script.

``` tcl
# replace this with paths to your RTL code, and add `-I`, `-D`, `-sv` etc when necessary
read_verilog xx.v yy.v top.v

# replace TOP_MODULE with your top module name
hierarchy -check -top TOP_MODULE

# simplify design before mapping
proc;;
opt_expr; opt_dff; opt_clean
memory -nomap

# map the rams
# point -lib path to your downloaded memlib_yosys.txt
memory_libmap -lib path/to/memlib_yosys.txt -logic-cost-rom 100 -logic-cost-ram 100
```

The `memory_libmap` command will output a list of RAMs it found and mapped.

- If you see `$__RAMGEM_SYNC_`, it means the mapping is successful.
- If you see `$__RAMGEM_ASYNC_`, it means this RAM is found to have asynchronous READ port. You need to confirm if it is the case.
  - If it is a synchronous one but accidentally recognized as asynchronous, you might need to patch the RTL code to fix it. There might be multiple reasons it cannot be recognized as synchronous. For example, [when the read and write clocks are different](https://github.com/YosysHQ/yosys/issues/4521).
  - If it is indeed asynchronous, check its size. If its size is very small and affordable to be synthesized using registers and mux trees (which is *very* expensive for large RAM banks), you can remove the `$__RAMGEM_ASYNC_` block in `memlib_yosys.txt`, re-run Yosys to force the use of registers.
- If you see `using FF mapping for memory`, it means the memory is recognized, but due to it being nonstandard (e.g., special global reset or nontrivial initialization), GEM will fall back to registers and mux trees. If the size of the memory is small, this is usually not an issue. Otherwise, you are advised to try other implementations.

After a successful mapping, use the following command to write out the mapped RTL as a single Verilog file.
``` tcl
write_verilog memory_mapped.v
```

Check the correctness of this step by simulating `memory_mapped.v` with your reference CPU simulator.

## Step 2. Logic Synthesis
This step maps all combinational and sequential logic into a special set of standard cells we defined in `aigpdk.lib`.
The quality of synthesis is directly tied to GEM's final performance, so we suggest you use a commercial synthesis tool like DC. You can also use Yosys to complete this if you do not have access to a commercial synthesis tool.

Check the correctness of this step by simulating `gatelevel.gv` with your reference CPU simulator.

### Use Synopsys DC
First, you need to compile `aigpdk.lib` to `aigpdk.db` using Library Compiler.

With that, you synthesize the `memory_mapped.v` obtained before under `aigpdk.db`.

Some key commands you may use on top of your existing DC flow:

``` tcl
# change path/to/aigpdk.db to a correct path. same for other commands.
set_app_var link_path path/to/aigpdk.db
set_app_var target_library path/to/aigpdk.db
read_file -format db $target_library

# elaborate TOP_MODULE
# current_design TOP_MODULE

# timing settings like create_clock ... are recommended. GEM benefits from timing-driven synthesis.

compile_ultra -no_seq_output_inversion -no_autoungroup
optimize_netlist -area

write -format verilog -hierarchy -out gatelevel.gv
```

### Use Yosys: Example script
``` tcl
# if you exited Yosys in step 2, you can read back in your memory_mapped.v yourself.
# read_verilog memory_mapped.v
# hierarchy -check -top TOP_MODULE

# synthesis
synth -flatten
delete t:$print

# change path/to/aigpdk.lib to a correct path. same for other commands.
dfflibmap -liberty path/to/aigpdk.lib
opt_clean -purge
abc -liberty /home/zzscratch1/scratch/aigpdk/aigpdk.lib
opt_clean -purge
techmap
abc -liberty /home/zzscratch1/scratch/aigpdk/aigpdk.lib
opt_clean -purge

# write out
write_verilog gatelevel.gv
```

## Step 3. Download and Compile GEM
Make sure CUDA is installed on your Linux machine.

Download and install Rust toolchain. This is as simple as a one-liner in your terminal. We recommend [https://rustup.rs](https://rustup.rs/).

Clone GEM along with its dependency.
``` sh
git clone https://github.com/NVlabs/GEM.git
cd GEM
git submodule update --init --recursive
```

GEM comes with a `cut_map_interactive` command and a `cuda_test` command, that correspond to `compile` and `simulate` steps of a classical CPU simulator. See their help usage with the following command under `GEM`:
``` sh
cargo run -r --features cuda --bin cut_map_interactive -- --help

cargo run -r --features cuda --bin cuda_test -- --help
```

## Map the Design with GEM
GEM depends on an external hypergraph partitioner binary. We recommend hmetis 2.0. You can download its binary and put it in a proper location.

Then, run the following command to start the Boolean processor mapping.

``` sh
cargo run -r --features cuda --bin cut_map_interactive -- path/to/hmetis/Linux-x86_64/hmetis2.0pre1 path/to/gatelevel.gv path/to/result.gemparts
```

The mapped result will be stored in a binary file `result.gemparts`.

If the mapping failed due to failure to partition deep circuits (which often shows as trying to partition a circuit with only 0 or 1 endpoints), try adding a `--level-split` option to force a stage split. For example `--level-split 30` or `--level-split 20,40`. If you used this, remember to add the same `--level-split` option when you simulate.

## Simulate the Design
Run the following. Replace `NUM_BLOCKS` with twice the number of physical streaming multiprocessors (SMs) of your GPU. If ports in your `input.vcd` are not in top-level, add a `--input-vcd-scope` to specify it.
``` sh
cargo run -r --features cuda --bin cuda_test -- path/to/gatelevel.gv path/to/result.gemparts path/to/input.vcd path/to/output.vcd NUM_BLOCKS
```

The simulated output ports value will be stored in `output.vcd`.

**Caveat**: The actual GPU simulation runtime will also be outputted. You might see a long time before GPU enters due to reading and parsing `input.vcd`. You are recommended to develop your own pipeline to feed the input waveform into GEM CUDA kernels.
