#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(improper_ctypes)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::ptr_offset_with_cast)]
#![allow(clippy::useless_transmute)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::approx_constant)]

#[cfg(feature = "rocm_6_2_2")]
pub mod hip_runtime_bindings;

#[cfg(feature = "rocm_6_2_2")]
pub mod hiprtc_bindings;

#[cfg(test)]
mod tests {
    use crate::hip_runtime_bindings::{
        hipError_t_hipSuccess, hipFree, hipFunction_t, hipMalloc, hipMemcpy,
        hipMemcpyKind_hipMemcpyDeviceToHost, hipMemcpyKind_hipMemcpyHostToDevice,
        hipModuleGetFunction, hipModuleLaunchKernel, hipModuleLoadData, hipModule_t,
        hipRuntimeGetVersion, HIP_LAUNCH_PARAM_BUFFER_POINTER, HIP_LAUNCH_PARAM_BUFFER_SIZE,
        HIP_LAUNCH_PARAM_END,
    };
    use crate::hiprtc_bindings::{
        hiprtcCompileProgram, hiprtcCreateProgram, hiprtcDestroyProgram, hiprtcGetCode,
        hiprtcGetCodeSize, hiprtcGetProgramLog, hiprtcGetProgramLogSize, hiprtcProgram,
        hiprtcResult_HIPRTC_SUCCESS,
    };
    use std::ffi::{c_float, c_uint, CString};
    use utilities::helper_functions::buf_i8_to_string;

    #[test]
    fn test_hip_runtime_get_version() {
        let mut version = 0;
        let ret = unsafe { hipRuntimeGetVersion(&mut version) };

        assert_eq!(ret, hipError_t_hipSuccess);

        println!("ret: {:?}", ret);
        println!("version: {:?}", version);
    }

    // https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_rtc.html
    // resources/hiprtc_vector_add.cpp
    #[test]
    fn test_hiprtc_vector_add() {
        let kernel_source: &str = r#"

            extern "C" __global__ void vector_add(float* output, float* input1, float* input2, size_t size) {
                int i = threadIdx.x;
                if (i < size) {
                    output[i] = input1[i] + input2[i];
                }
            }

        "#;

        let mut prog: hiprtcProgram = std::ptr::null_mut();

        let k_source = CString::new(kernel_source).unwrap();
        // let file_name = CString::new("vector_add.cpp").unwrap();

        let rtc_ret_code = unsafe {
            hiprtcCreateProgram(
                &mut prog,            // HIPRTC program handle
                k_source.as_ptr(),    // kernel source string
                std::ptr::null(),     // Name of the file
                0,                    // Number of headers
                std::ptr::null_mut(), // Header sources
                std::ptr::null_mut(), // Name of header file
            )
        };
        assert_eq!(rtc_ret_code, hiprtcResult_HIPRTC_SUCCESS);

        // let props: hipDeviceProp_t;
        // int device = 0;
        // HIP_CHECK(hipGetDeviceProperties(&props, device));s
        // std::string sarg = std::string("--gpu-architecture=") +
        //     props.gcnArchName;  // device for which binary is to be generated
        //
        // const char* options[] = {sarg.c_str()};

        let rtc_ret_code = unsafe {
            hiprtcCompileProgram(
                prog,                 // hiprtcProgram
                0,                    // Number of options
                std::ptr::null_mut(), // Clang Options
            )
        };
        assert_eq!(rtc_ret_code, hiprtcResult_HIPRTC_SUCCESS);

        let mut log_size = 0;
        let rtc_ret_code = unsafe { hiprtcGetProgramLogSize(prog, &mut log_size) };
        assert_eq!(rtc_ret_code, hiprtcResult_HIPRTC_SUCCESS);

        println!("hiprtcGetProgramLogSize: {:?}", log_size);

        if log_size > 0 {
            println!("compilation failed");
            let mut log_vec = vec![0; log_size];

            let rtc_ret_code = unsafe { hiprtcGetProgramLog(prog, log_vec.as_mut_ptr()) };
            assert_eq!(rtc_ret_code, hiprtcResult_HIPRTC_SUCCESS);

            let text = buf_i8_to_string(&log_vec).unwrap();

            panic!("Compilation failed with: {}", text);
        }

        let mut code_size = 0;
        let rtc_ret_code = unsafe { hiprtcGetCodeSize(prog, &mut code_size) };
        assert_eq!(rtc_ret_code, hiprtcResult_HIPRTC_SUCCESS);

        let mut kernel_binary = vec![0; code_size];
        let rtc_ret_code = unsafe { hiprtcGetCode(prog, kernel_binary.as_mut_ptr()) };
        assert_eq!(rtc_ret_code, hiprtcResult_HIPRTC_SUCCESS);

        let rtc_ret_code = unsafe { hiprtcDestroyProgram(&mut prog) };
        assert_eq!(rtc_ret_code, hiprtcResult_HIPRTC_SUCCESS);

        let mut module: hipModule_t = std::ptr::null_mut();
        let mut kernel: hipFunction_t = std::ptr::null_mut();

        let rtc_ret_code = unsafe {
            hipModuleLoadData(
                &mut module,
                kernel_binary.as_ptr() as *const std::os::raw::c_void,
            )
        };
        assert_eq!(rtc_ret_code, hiprtcResult_HIPRTC_SUCCESS);

        let k_name = CString::new("vector_add").unwrap();
        let rtc_ret_code = unsafe { hipModuleGetFunction(&mut kernel, module, k_name.as_ptr()) };
        assert_eq!(rtc_ret_code, hiprtcResult_HIPRTC_SUCCESS);

        let ele_size = 256; // total number of items to add

        let mut hinput: Vec<c_float> = Vec::with_capacity(ele_size);
        let mut output: Vec<c_float> = vec![0.0; ele_size];

        for i in 0..ele_size {
            hinput.push(i as c_float + 1f32);
        }

        let mut dinput1: *mut std::os::raw::c_void = std::ptr::null_mut();
        let mut dinput2: *mut std::os::raw::c_void = std::ptr::null_mut();
        let mut doutput: *mut std::os::raw::c_void = std::ptr::null_mut();

        let input_size = std::mem::size_of::<f32>() * ele_size;

        unsafe {
            let ret = hipMalloc(&mut dinput1, input_size);
            assert_eq!(ret, hipError_t_hipSuccess);

            let ret = hipMalloc(&mut dinput2, input_size);
            assert_eq!(ret, hipError_t_hipSuccess);

            let ret = hipMalloc(&mut doutput, input_size);
            assert_eq!(ret, hipError_t_hipSuccess);

            let ret = hipMemcpy(
                dinput1,
                hinput.as_ptr() as *const std::os::raw::c_void,
                input_size,
                hipMemcpyKind_hipMemcpyHostToDevice,
            );
            assert_eq!(ret, hipError_t_hipSuccess);

            let ret = hipMemcpy(
                dinput2,
                hinput.as_ptr() as *const std::os::raw::c_void,
                input_size,
                hipMemcpyKind_hipMemcpyHostToDevice,
            );
            assert_eq!(ret, hipError_t_hipSuccess);
        }

        #[repr(C)]
        #[derive(Debug, Copy, Clone)]
        struct Args {
            output: *mut std::os::raw::c_void,
            input1: *mut std::os::raw::c_void,
            input2: *mut std::os::raw::c_void,
            size: usize,
        }

        let args = Args {
            output: doutput,
            input1: dinput1,
            input2: dinput2,
            size: ele_size,
        };

        let size = std::mem::size_of_val(&args);

        let mut config = [
            HIP_LAUNCH_PARAM_BUFFER_POINTER as *mut std::os::raw::c_void,
            &args as *const _ as *mut std::os::raw::c_void,
            HIP_LAUNCH_PARAM_BUFFER_SIZE as *mut std::os::raw::c_void,
            &size as *const _ as *mut std::os::raw::c_void,
            HIP_LAUNCH_PARAM_END as *mut std::os::raw::c_void,
        ];

        unsafe {
            let ret = hipModuleLaunchKernel(
                kernel,
                1,
                1,
                1,
                ele_size as c_uint,
                1,
                1,
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                config.as_mut_ptr(),
            );
            assert_eq!(ret, hipError_t_hipSuccess);

            let ret = hipMemcpy(
                output.as_mut_ptr() as *mut std::os::raw::c_void,
                doutput,
                input_size,
                hipMemcpyKind_hipMemcpyDeviceToHost,
            );
            assert_eq!(ret, hipError_t_hipSuccess);
        }

        for i in 0..ele_size {
            println!(
                "index: {} -> hinput {} + hinput {} = output {}",
                i, hinput[i], hinput[i], output[i]
            );
            if (hinput[i] + hinput[i]) != output[i] {
                panic!(
                    "Failed in validation: {} - {}",
                    hinput[i] + hinput[i],
                    output[i]
                );
            }
        }

        println!("Passed");

        unsafe {
            let ret = hipFree(dinput1);
            assert_eq!(ret, hipError_t_hipSuccess);

            let ret = hipFree(dinput2);
            assert_eq!(ret, hipError_t_hipSuccess);

            let ret = hipFree(doutput);
            assert_eq!(ret, hipError_t_hipSuccess);
        }
    }
}
