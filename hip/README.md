# AMD HIP

Rust wrapper of AMD Heterogeneous-computing Interface for Portability (HIP) (https://github.com/ROCm/amdsmi)

| ROCM Version | Cargo Feature |          
|:-------------|:--------------|
| 6.2.2        | rocm_6_2_2    |

# Testing

test
```bash
cargo test -- --test-threads=1
```

test
```bash
cargo test
```

test_module_api
```bash
hipcc --genco --offload-arch=gfx1032 ./resources/module_api/module.hip -o module_gfx1032.co
```

test_global_array

v1
```bash
/opt/rocm/bin/hipcc global_array.cpp --cuda-device-only -c -emit-llvm --offload-arch=gfx1032
```

v2
```bash
/opt/rocm/bin/hipcc global_array.cpp -fno-optimize-sibling-calls -fno-strict-aliasing --cuda-device-only -c -emit-llvm --offload-arch=gfx1032
```

```bash
/opt/rocm/llvm/bin/llvm-dis global_array-hip-amdgcn-amd-amdhsa-gfx1032.bc -o global_array-hip-amdgcn-amd-amdhsa_gfx1032.ll
```

```bash
/opt/rocm/llvm/bin/clang -target amdgcn-amd-amdhsa -mcpu=gfx1032 global_array-hip-amdgcn-amd-amdhsa_gfx1032.ll -o global_array-hip-amdgcn-amd-amdhsa_gfx1032.o
```


other

https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_rtc.html

https://github.com/ROCm/HIP/blob/amd-staging/docs/how-to/faq.md

https://rocm.docs.amd.com/projects/HIP/en/latest/index.html

https://github.com/ROCm/HIP/tree/amd-staging
