# amd_smi_sys

* only linux

Rust bindings of AMD System Management Interface (AMD SMI) Library (https://github.com/ROCm/amdsmi)

| ROCM Version | Cargo Feature |          
|:-------------|:--------------|
| 6.2.2        | rocm_6_2_2    |


Bindings generated with: https://github.com/rust-lang/rust-bindgen
```
bindgen /opt/rocm/include/amd_smi/amdsmi.h -o src/bindings.rs
```

# Testing

test
```bash
cargo test -- --test-threads=1
```

Error if tests are run in parallel
```bash
cargo test
```
