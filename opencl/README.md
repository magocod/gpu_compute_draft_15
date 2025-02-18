# Opencl-sys

Rust wrapper - OpenCL C 2.0

The goal of this library is to provide a small wrapper around the opencl 2.0 API,
in most cases trying to preserve the flow and names of each function of the original api,
only intended to be used in conjunction with gpus to maintain the greater simplicity.

Below we can see some quite complete crates that can be a direct replacement for this library:

* https://github.com/kenba/opencl3, It offers a lot of OpenCL capability, however, more than required and the abstraction levels are a bit high.

* https://github.com/cogciprocate/ocl, Good OpenCL implementations in rust, however it renames and overly abbreviates several of the flows in a simple OpenCL program. I think it is difficult to understand for someone who is starting out in OpenCL.


# Testing

test
```bash
cargo test -- --test-threads=1
```

test
```bash
cargo test
```
