# Welcome to GEM
GEM is an open-source RTL logic simulator with CUDA acceleration, developed and maintained by NVIDIA Research.
GEM can deliver up to 5--40X speed-up compared to CPU-based leading RTL simulators.
A summary of the work with paper can be found [here](https://research.nvidia.com/publication/2025-06_gem-gpu-accelerated-emulator-inspired-rtl-simulation).

## Compile and Run Your Design with GEM
GEM works in a way similar to an FPGA-based RTL emulator.
It first synthesizes your design with a special and-inverter graph (AIG) process, and then map the synthesized gate-level netlist to a virtual manycore Boolean processor which can be emulated with CUDA-compatible GPUs.

The synthesis and mapping is slower than the compiling/elaboration process of CPU-based simulators. But it is a one-time cost and your design can be simulated under different testbenches without re-running the synthesis or mapping.

**See [usage.md](./usage.md) for usage documentation.**

## Citation
Please cite our paper if you find GEM useful.

``` bibtex
@inproceedings{gem,
 author = {Guo, Zizheng and Zhang, Yanqing and Wang, Runsheng and Lin, Yibo and Ren, Haoxing},
 booktitle = {Proceedings of the 62nd Annual Design Automation Conference 2025},
 organization = {IEEE},
 title = {{GEM}: {GPU}-Accelerated Emulator-Inspired {RTL} Simulation},
 year = {2025}
}
```
