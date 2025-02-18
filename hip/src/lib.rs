//! # HIP
//!
//! ...
//!

pub mod error;
pub mod memory;
pub mod module;

pub mod collections;

pub mod utils;

// re-export
pub use hip_sys;

#[cfg(test)]
mod tests {
    use crate::error::hip_check;
    use crate::memory::HipBuffer;
    use crate::module::HipModule;
    use hip_sys::hip_runtime_bindings::{
        hipModuleLaunchKernel, HIP_LAUNCH_PARAM_BUFFER_POINTER, HIP_LAUNCH_PARAM_BUFFER_SIZE,
        HIP_LAUNCH_PARAM_END,
    };
    use std::ffi::{c_float, c_uint};

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

        let module = HipModule::create(kernel_source).unwrap();
        let kernel = module.create_kernel("vector_add").unwrap();

        let ele_size = 256; // total number of items to add

        let mut hinput: Vec<c_float> = Vec::with_capacity(ele_size);
        // let mut output: Vec<c_float> = vec![0.0; ele_size];

        for i in 0..ele_size {
            hinput.push(i as c_float + 1f32);
        }

        let dinput1: HipBuffer<c_float> = HipBuffer::new(ele_size).unwrap();
        let dinput2: HipBuffer<c_float> = HipBuffer::new(ele_size).unwrap();
        let doutput: HipBuffer<c_float> = HipBuffer::new(ele_size).unwrap();

        dinput1.memcpy_host_to_device(&hinput).unwrap();
        dinput2.memcpy_host_to_device(&hinput).unwrap();

        #[repr(C)]
        #[derive(Debug, Copy, Clone)]
        struct Args {
            output: *mut std::os::raw::c_void,
            input1: *mut std::os::raw::c_void,
            input2: *mut std::os::raw::c_void,
            size: usize,
        }

        let args = Args {
            output: doutput.get_mem_ptr(),
            input1: dinput1.get_mem_ptr(),
            input2: dinput2.get_mem_ptr(),
            size: ele_size,
        };

        let size = std::mem::size_of_val(&args);
        println!("args size: {}", size);

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
            hip_check(ret).unwrap();
        }

        let output = doutput.memcpy_device_to_host().unwrap();

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

        // HipBuffer traits hipFree
    }
}
