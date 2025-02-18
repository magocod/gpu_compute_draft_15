# Opencl-sys

Rust bindings - OpenCL C 2.0

Bindings generated with: https://github.com/rust-lang/rust-bindgen

- add #[link(name = "OpenCL")]
- FIXME bindings generation
```
bindgen /usr/include/CL/cl.h -o src/bindings.rs
```

Below we can see some quite complete crates that can be a direct replacement for this library:

* OpenCL C FFI are quite well-prepared.
  * https://github.com/kenba/opencl-sys-rs
  * https://github.com/cogciprocate/ocl/tree/master/cl-sys

# Testing

test
```bash
cargo test -- --test-threads=1
```

test
```bash
cargo test
```

TODO 
* FIXME bindings generation (...)
