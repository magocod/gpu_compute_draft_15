# GPU COMPUTE DRAFT

(Important) All these examples of programs with GPU have only been tested in:

OS
* Ubuntu Ubuntu 22.04.5 LTS 64-bit

GPU
* Radeon rx 6500xt
* Radeon rx 6600

CPU
* ryzen 5700G

ROCM
* 6.2.2.60202-116~22.04

Other
* At the moment it mostly requires the explanation of the code and its final objectives (I seek to somehow justify a certain madness of the present code)


The main goal of this project is to create a draft of what it would be like to have data structures controlled and managed 100% on the GPU.

goals:
- rewrite hsa runtime from c++ to rust
- write an alternative to d-bus (https://www.freedesktop.org/wiki/Software/dbus/) controlled 100% by a gpu (it is possible, as it would look like)


After simple practices of opencl, hip, hsa and cuda:

questions:
- All the execution control of the device has to be done at host side. ?
- The device has to care only about completing the queued jobs, as fast as possible. ?
- It is actually possible for a GPU to execute complex algorithms without help from the processor, such as controlling other peripherals.
- Can you imagine programming on a GPU as if it were a microcontroller?


Inspired by the following libraries:
* https://github.com/stotko/stdgpu 
* https://github.com/nvidia/cccl
* https://github.com/rust-gpu/rust-gpu
* https://github.com/prsyahmi/GpuRamDrive.git
* https://github.com/tracel-ai/cubecl-hip
* https://github.com/tracel-ai/cubecl


thanks.
* https://github.com/Umio-Yasuno/amdgpu_top
* https://github.com/kenba/opencl3

---

# Vision

* https://github.com/crossbeam-rs/crossbeam
* https://serde.rs/ (in rust there was a standard for gpu, at the same level as serde)
* https://docs.rust-embedded.org/discovery/microbit/index.html (Can you imagine programming on a GPU as if it were a microcontroller?, with more freedoms)
* https://docs.rust-embedded.org/book/intro/index.html (What are we if we don't dream, programming a GPU with a part of its private source code (driver) and a public one?)

---

# Opencl crates

---

## opencl-sys

Rust bindings - OpenCL C 2.0 (ROCM CL headers)

## opencl

Rust wrapper - OpenCL C 2.0

The goal of this library is to provide a small wrapper around the opencl 2.0 API,
in most cases trying to preserve the flow and names of each function of the original api,
only intended to be used in conjunction with gpus to maintain the greater simplicity.

## opencl-examples

simple opencl practices, quickly observe what can cause a fatal error on an amd gpu,...

## opencl-collections

* At the time of making these examples, I was a complete newbie to opencl, I chose to write the cl code in text, 
at first it seemed like a good idea (It was not for advanced cases)

Implementations that seek to provide containers the same (similar) to those provided by rust std and c++
* https://doc.rust-lang.org/std/collections/
* https://cplusplus.com/reference/stl/

* https://github.com/stotko/stdgpu
* https://github.com/NVIDIA/cuCollections

## opencl-collections-examples

Examples of use of collections (opencl-collections) and comparisons with existing libraries in rust (at least what was wanted)

## opencl-node

implementing some type of algorithm with gpu in typescript, is the objective

* https://github.com/napi-rs/napi-rs
* https://github.com/neon-bindings/neon

---

# ROCM crates

---

## hip-sys

Rust bindings of AMD Heterogeneous-computing Interface for Portability (HIP) (https://github.com/ROCm/HIP)

## hip

Rust wrapper of AMD Heterogeneous-computing Interface for Portability (HIP) (https://github.com/ROCm/amdsmi)

## hsa-sys

Rust bindings of AMD HSA Runtime (hsa-runtime) (https://github.com/ROCm/ROCR-Runtime)

## hsa

Rust wrapper of AMD HSA Runtime (hsa-runtime) (https://github.com/ROCm/ROCR-Runtime)

## hsakmt-rs (TODO)

ROCt Thunk Library (`libhsakmt`) rewrite from C to Rust 
TODO add repo url

## hsa-rs (TODO)

The HSA Runtime (`hsa-runtime`) rewrite from c++ to Rust
TODO add repo url

## amdsmi-sys

Rust bindings of AMD System Management Interface (AMD SMI) Library (https://github.com/ROCm/amdsmi)

## amdsmi

Rust wrapper of AMD System Management Interface (AMD SMI) Library (https://github.com/ROCm/amdsmi)

---

# Other crates

---

## gpu-fs

use gpu like fs? (https://github.com/prsyahmi/GpuRamDrive.git)

## gpu-ipc

use gpu as ipc (Interprocess Communication) ? (https://www.freedesktop.org/wiki/Software/dbus/)

* https://dbus.freedesktop.org/doc/dbus-specification.html
* https://github.com/dbus2/zbus
* https://github.com/diwic/dbus-rs

---

# Docs

I hope at some point to write here how the programming journey on a GPU was, what the learning process is like from my point of view.

```bash
mdbook serve gpu-compute-book
```

---

# Testing

test
```bash
cargo test -- --test-threads=1
```

// texts

All the execution control of the device has to be done at host side. The device has to care only about completing the queued jobs, as fast as possible.
That's all. If you intent to slow down the execution then do it at host side.


https://github.com/tracel-ai/cubecl-hip

https://github.com/tracel-ai/cubecl

https://github.com/ROCm/ROCm/issues/419
apt show rocm-libs -a

https://youtrack.jetbrains.com/issue/CPP-30059
