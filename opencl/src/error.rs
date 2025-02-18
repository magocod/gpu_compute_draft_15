//! # opencl error
//!
//! ...
//!

use io::Error as IoError;
use opencl_sys::bindings::{cl_int, CL_SUCCESS};
use std::io;

// opencl wrapper errors

pub const CL_WRAPPER_FIRST_PLATFORM_NOT_FOUND: cl_int = -100;

pub const CL_WRAPPER_EMPTY_PROGRAM_SOURCE: cl_int = -101;

#[derive(Debug, PartialEq)]
pub enum OclError {
    // opencl error code
    Code(i32),
    // opencl wrapper error code
    Wrapper(i32),
}

pub type OclResult<T> = Result<T, OclError>;

// CL_CHECK | HIP_ASSERT
pub fn cl_check(status: cl_int) -> OclResult<()> {
    if CL_SUCCESS as cl_int == status {
        Ok(())
    } else {
        Err(OclError::Code(status))
    }
}

impl From<OclError> for IoError {
    fn from(e: OclError) -> Self {
        let error_kind = io::ErrorKind::Other;

        match e {
            OclError::Code(code) => Self::new(error_kind, format!("opencl error code: {code}")),
            OclError::Wrapper(code) => {
                Self::new(error_kind, format!("opencl wrapper error code: {code}"))
            }
        }
    }
}
