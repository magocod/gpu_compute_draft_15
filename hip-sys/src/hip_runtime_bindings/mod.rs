#[cfg(feature = "rocm_6_2_2")]
mod hip_runtime_6_2_2;
#[cfg(feature = "rocm_6_2_2")]
pub use hip_runtime_6_2_2::*;

#[cfg(feature = "rocm_6_2_2")]
pub const HIP_LAUNCH_PARAM_BUFFER_POINTER: i32 = 0x01;

#[cfg(feature = "rocm_6_2_2")]
pub const HIP_LAUNCH_PARAM_BUFFER_SIZE: i32 = 0x02;

#[cfg(feature = "rocm_6_2_2")]
pub const HIP_LAUNCH_PARAM_END: i32 = 0x03;
