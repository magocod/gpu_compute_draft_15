//! # Hip error
//!
//! ...
//!

use hip_sys::hip_runtime_bindings::hipError_t_hipSuccess;
use hip_sys::hiprtc_bindings::hiprtcResult_HIPRTC_SUCCESS;
use io::Error as IoError;
use std::io;

#[derive(Debug, PartialEq)]
pub enum HipError {
    // hip_runtime error code
    Runtime(u32),
    // hiprtc error code
    Rtc(u32),
}

pub type HipResult<T> = Result<T, HipError>;

// HIP_CHECK
pub fn hip_check(status: u32) -> HipResult<()> {
    if hipError_t_hipSuccess == status {
        Ok(())
    } else {
        Err(HipError::Runtime(status))
    }
}

// HIPRTC_CHECK
pub fn hiprtc_check(status: u32) -> HipResult<()> {
    if hiprtcResult_HIPRTC_SUCCESS == status {
        Ok(())
    } else {
        Err(HipError::Rtc(status))
    }
}

impl From<HipError> for IoError {
    fn from(e: HipError) -> Self {
        let error_kind = io::ErrorKind::Other;

        match e {
            HipError::Runtime(code) => Self::new(error_kind, format!("hip error code: {code}")),
            HipError::Rtc(code) => Self::new(error_kind, format!("hiprtc error code: {code}")),
        }
    }
}
