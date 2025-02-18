use crate::error::{hsa_check, HsaResult};
use hsa_sys::bindings::{
    hsa_amd_signal_create, hsa_signal_destroy, hsa_signal_t, hsa_signal_value_t,
    hsa_status_t_HSA_STATUS_SUCCESS,
};

// TODO complete hsa Signal

#[derive(Debug)]
pub struct Signal {
    signal: hsa_signal_t,
}

impl Signal {
    pub fn new(initial_value: hsa_signal_value_t, attributes: u64) -> HsaResult<Self> {
        let mut signal = hsa_signal_t { handle: 0 };

        // let ret = unsafe { hsa_signal_create(initial_value, 0, std::ptr::null_mut(), &mut signal) };
        let ret = unsafe {
            hsa_amd_signal_create(
                initial_value,
                0,
                std::ptr::null_mut(),
                attributes,
                &mut signal,
            )
        };
        hsa_check(ret)?;

        Ok(Signal { signal })
    }

    pub fn get_hsa_signal_t(&self) -> hsa_signal_t {
        self.signal
    }
}

impl Drop for Signal {
    fn drop(&mut self) {
        unsafe {
            let err = hsa_signal_destroy(self.signal);
            if err != hsa_status_t_HSA_STATUS_SUCCESS {
                panic!("hsa_signal_destroy error: {}", err);
            }
        }
    }
}

impl PartialEq for Signal {
    fn eq(&self, other: &Self) -> bool {
        self.signal.handle.eq(&other.signal.handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::System;

    #[test]
    fn test_signal_create() {
        let _system = System::new().unwrap();

        let _signal = Signal::new(1, 0).unwrap();

        // TODO assert test
    }

    // TODO tests
}
