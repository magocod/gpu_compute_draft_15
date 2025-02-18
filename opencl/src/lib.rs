//! # opencl_sys
//!
//! ...
//!
//! https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html
//!

pub mod error;
pub mod utils;

pub mod unsafe_wrapper;
pub mod wrapper;

// re-export
pub use opencl_sys;
pub use utilities;
