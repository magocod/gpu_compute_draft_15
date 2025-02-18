// TODO complete handle errors

use amdsmi_sys::bindings::{amdsmi_status_t, amdsmi_status_t_AMDSMI_STATUS_SUCCESS};

#[derive(Debug, PartialEq)]
pub enum AmdSmiError {
    Status(u32),
    // ...
}

impl From<u32> for AmdSmiError {
    fn from(value: u32) -> Self {
        Self::Status(value)
    }
}

impl From<AmdSmiError> for u32 {
    fn from(value: AmdSmiError) -> Self {
        match value {
            AmdSmiError::Status(status) => status,
        }
    }
}

pub type AmdSmiResult<T> = Result<T, AmdSmiError>;

// CHK_AMDSMI_RET
pub fn chk_amd_smi_ret(status: amdsmi_status_t) -> AmdSmiResult<()> {
    if amdsmi_status_t_AMDSMI_STATUS_SUCCESS == status {
        Ok(())
    } else {
        Err(AmdSmiError::Status(status))
    }
}
