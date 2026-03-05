//! this build script compiles GEM kernels
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

fn main() {
    println!("Building cuda source files for GEM...");
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-changed=msrc");

    #[cfg(feature = "cuda")] {
        let csrc_headers = ucc::import_csrc();
        let mut cl_cuda = ucc::cl_cuda();
        cl_cuda.ccbin(false);
        cl_cuda.flag("-lineinfo");
        cl_cuda.flag("-maxrregcount=128");
        cl_cuda.debug(false).opt_level(3)
            .include(&csrc_headers)
            .files(["csrc/kernel_v1.cu"]);
        cl_cuda.compile("gemcu");
        println!("cargo:rustc-link-lib=static=gemcu");
        println!("cargo:rustc-link-lib=dylib=cudart");
        ucc::bindgen(["csrc/kernel_v1.cu"], "kernel_v1.rs");
        ucc::export_csrc();
        ucc::make_compile_commands(&[&cl_cuda]);
    }

    #[cfg(feature = "metal")] {
        let mut cl_metal = cc::Build::new();
        cl_metal
            .cpp(true)
            .flag("-std=c++17")
            .flag("-fobjc-arc")
            .file("csrc/kernel_v1_metal.mm");
        cl_metal.compile("gemmetal");

        println!("cargo:rustc-link-lib=static=gemmetal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
    }
}
