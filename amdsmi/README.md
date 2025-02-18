# amd_smi_sys

* only linux

Rust wrapper of AMD System Management Interface (AMD SMI) Library (https://github.com/ROCm/amdsmi)

The original goal of this library was to provide a simple means of viewing the metrics of a GPU (memory usage, temperatures, ...),
in the future, time permitting, it may become a complete package to be used in Rust, and published on crates.io

similar to how this crate does https://github.com/Umio-Yasuno/amdgpu_top


| ROCM Version | Cargo Feature |          
|:-------------|:--------------|
| 6.2.2        | rocm_6_2_2    |

# Testing

test
```bash
cargo test -- --test-threads=1
```

Error if tests are run in parallel
```bash
cargo test
```
