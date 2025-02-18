//! # Hsa error
//!
//! ...
//!

use hsa_sys::bindings::hsa_status_t_HSA_STATUS_SUCCESS;
use io::Error as IoError;
use std::io;

#[derive(Debug, PartialEq)]
pub enum HsaError {
    // hsa runtime error code
    Code(u32),
}

pub type HsaResult<T> = Result<T, HsaError>;

pub fn hsa_check(status: u32) -> HsaResult<()> {
    if hsa_status_t_HSA_STATUS_SUCCESS == status {
        Ok(())
    } else {
        Err(HsaError::Code(status))
    }
}

impl From<HsaError> for IoError {
    fn from(e: HsaError) -> Self {
        let error_kind = io::ErrorKind::Other;

        match e {
            HsaError::Code(code) => Self::new(error_kind, format!("hsa error code: {code}")),
        }
    }
}
