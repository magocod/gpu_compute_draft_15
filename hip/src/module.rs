use crate::error::{hiprtc_check, HipResult};

use hip_sys::hip_runtime_bindings::{
    hipFunction_t, hipModuleGetFunction, hipModuleLoadData, hipModule_t,
};
use hip_sys::hiprtc_bindings::{
    hiprtcCompileProgram, hiprtcCreateProgram, hiprtcDestroyProgram, hiprtcGetCode,
    hiprtcGetCodeSize, hiprtcGetProgramLog, hiprtcGetProgramLogSize, hiprtcProgram,
    hiprtcResult_HIPRTC_SUCCESS,
};
use std::ffi::CString;
use utilities::helper_functions::buf_i8_to_string;

#[derive(Debug, PartialEq)]
pub struct HipModule {
    code_size: Option<usize>,
    hip_module: hipModule_t,
}

impl HipModule {
    pub fn new(hip_module: hipModule_t, code_size: Option<usize>) -> Self {
        Self {
            hip_module,
            code_size,
        }
    }

    pub fn get_hip_module_t(&self) -> hipModule_t {
        self.hip_module
    }

    pub fn get_code_size(&self) -> Option<usize> {
        self.code_size
    }

    pub fn create(kernel_source: &str) -> HipResult<Self> {
        let mut prog: hiprtcProgram = std::ptr::null_mut();

        let k_source = CString::new(kernel_source).unwrap();
        let mut module: hipModule_t = std::ptr::null_mut();

        unsafe {

            let rtc_ret_code = hiprtcCreateProgram(
                &mut prog,            // HIPRTC program handle
                k_source.as_ptr(),    // kernel source string
                std::ptr::null(),     // Name of the file
                0,                    // Number of headers
                std::ptr::null_mut(), // Header sources
                std::ptr::null_mut(), // Name of header file
            );
            hiprtc_check(rtc_ret_code)?;

            // hipGetDeviceProperties

            let rtc_ret_code = hiprtcCompileProgram(
                prog,                 // hiprtcProgram
                0,                    // Number of options
                std::ptr::null_mut(), // Clang Options
            );

            if rtc_ret_code != hiprtcResult_HIPRTC_SUCCESS {
                let mut log_size = 0;
                let rtc_ret_code = hiprtcGetProgramLogSize(prog, &mut log_size);
                hiprtc_check(rtc_ret_code)?;

                if log_size > 0 {
                    let mut log_vec = vec![0; log_size];

                    let rtc_ret_code = hiprtcGetProgramLog(prog, log_vec.as_mut_ptr());
                    hiprtc_check(rtc_ret_code)?;

                    let text = buf_i8_to_string(&log_vec).expect("Failed to convert log vec to string");
                    panic!("Compilation failed with: {}", text);
                }
            }

            let mut code_size = 0;
            let rtc_ret_code = hiprtcGetCodeSize(prog, &mut code_size);
            hiprtc_check(rtc_ret_code)?;

            let mut kernel_binary = vec![0; code_size];
            let rtc_ret_code = hiprtcGetCode(prog, kernel_binary.as_mut_ptr());
            hiprtc_check(rtc_ret_code)?;

            let rtc_ret_code = hiprtcDestroyProgram(&mut prog);
            hiprtc_check(rtc_ret_code)?;

            let rtc_ret_code = hipModuleLoadData(
                &mut module,
                kernel_binary.as_ptr() as *const std::os::raw::c_void,
            );
            hiprtc_check(rtc_ret_code)?;
        }

        Ok(Self {
            hip_module: module,
            code_size: Some(code_size),
        })
    }

    pub fn create_kernel(&self, kernel_name: &str) -> HipResult<hipFunction_t> {
        let k_name = CString::new(kernel_name).expect("kernel_name CString failed");
        let mut kernel: hipFunction_t = std::ptr::null_mut();

        let rtc_ret_code =
            unsafe { hipModuleGetFunction(&mut kernel, self.hip_module, k_name.as_ptr()) };
        hiprtc_check(rtc_ret_code)?;

        Ok(kernel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::HipError;
    use hip_sys::hip_runtime_bindings::hipError_t_hipErrorNotFound;
    use hip_sys::hiprtc_bindings::hiprtcResult_HIPRTC_ERROR_INVALID_INPUT;

    const KERNEL_SOURCE: &str = r#"
        extern "C" __global__ void vector_add(float* output, float* input1, float* input2, size_t size) {
            int i = threadIdx.x;
            if (i < size) {
                output[i] = input1[i] + input2[i];
            }
        }
    "#;

    const INVALID_KERNEL_SOURCE: &str = r#"
        extern "C" __global__ void vector_add(float* output, float* input1, float* input2, size_t size) {
            int i = threadIdx.x;
            if (i < size) {
                output[i] = input1[i] + inp[i];
            }
        }
    "#;

    #[test]
    fn test_hip_module_create_with_valid_kernel_source() {
        let hip_module = HipModule::create(KERNEL_SOURCE).unwrap();

        assert_eq!(hip_module.get_code_size(), Some(3656));
    }

    #[test]
    #[should_panic]
    fn test_hip_module_create_with_invalid_kernel_source() {
        let _hip_module = HipModule::create(INVALID_KERNEL_SOURCE).unwrap();

        unreachable!();
    }

    #[test]
    fn test_hip_module_create_with_invalid_kernel_source_2() {
        let result = HipModule::create("");
        assert_eq!(
            result,
            Err(HipError::Rtc(hiprtcResult_HIPRTC_ERROR_INVALID_INPUT))
        );
    }

    #[test]
    fn test_hip_module_create_kernel() {
        let hip_module = HipModule::create(KERNEL_SOURCE).unwrap();

        let _f = hip_module.create_kernel("vector_add").unwrap();
        // TODO assert test
    }

    #[test]
    fn test_hip_module_create_kernel_with_invalid_name() {
        let hip_module = HipModule::create(KERNEL_SOURCE).unwrap();

        let result = hip_module.create_kernel("invalid_name");
        assert_eq!(result, Err(HipError::Rtc(hipError_t_hipErrorNotFound)));
    }
}
