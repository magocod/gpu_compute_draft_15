use hip::error::hip_check;
use hip::module::HipModule;
use hip::utils::ceiling_div;
use hip_sys::hip_runtime_bindings::{
    hipFree, hipGetLastError, hipMalloc, hipMemcpy, hipMemcpyKind_hipMemcpyDeviceToHost,
    hipMemcpyKind_hipMemcpyHostToDevice, hipModuleLaunchKernel, HIP_LAUNCH_PARAM_BUFFER_POINTER,
    HIP_LAUNCH_PARAM_BUFFER_SIZE, HIP_LAUNCH_PARAM_END,
};
use std::ffi::c_float;

const DEVICE_ARRAY_SIZE: usize = 16;

const GLOBAL_VALUE: c_float = 42.0;

fn test_globals_reference(input: &[c_float], global_array: &[c_float]) -> Vec<c_float> {
    input
        .iter()
        .enumerate()
        .map(|(i, x)| x + GLOBAL_VALUE + global_array[i % DEVICE_ARRAY_SIZE])
        .collect()
}

const PROGRAM_SOURCE: &str = r#"
    constexpr unsigned int device_array_size = 16;
    
    /// A test global variable of a single element, that will later be set from the host.
    __device__ float global;
    
    /// A test global variable of \p device_array_size elements that will be set from the host.
    __device__ float global_array[device_array_size];
    
    /// \brief A simple test kernel, that reads from <tt>in</tt>, <tt>global</tt>, and
    /// <tt>global_array</tt>. The result will be written to <tt>out</tt>.
    extern "C" __global__ void test_globals_kernel(float* out, const float* in, const size_t size)
    {
        const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if(tid < size)
        {
            out[tid] = in[tid] + global + global_array[tid % device_array_size];
        }
    }
    
    // extra
    extern "C" __global__ void set_global_kernel() {
        global = 42;
    }
    
    extern "C" __global__ void set_global_array_kernel() {
        const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        global_array[tid] = tid * 1000;
    }
    
    "#;

#[test]
fn test_device_globals() {
    let mut global_array = vec![0.0; DEVICE_ARRAY_SIZE];

    // The size of the input and output vectors.
    let size: u32 = 64;

    // The total number of bytes in the input and output vectors.
    let size_bytes = size as usize * std::mem::size_of::<c_float>();

    // Number of threads per kernel block.
    let block_size: u32 = size;

    // Number of blocks per kernel grid. The expression below calculates ceil(size/block_size).
    let grid_size: u32 = ceiling_div(size, block_size);

    // Allocate host vectors for the input and output.
    let h_in: Vec<c_float> = (1..=size).map(|i| i as f32).collect();
    let mut h_out: Vec<c_float> = vec![0.0; size as usize];

    // Allocate and copy vectors to device memory.
    let mut d_in: *mut std::os::raw::c_void = std::ptr::null_mut();
    let mut d_out: *mut std::os::raw::c_void = std::ptr::null_mut();

    let hip_module = HipModule::create(PROGRAM_SOURCE).unwrap();

    let test_globals_kernel = hip_module.create_kernel("test_globals_kernel").unwrap();

    let set_global_kernel = hip_module.create_kernel("set_global_kernel").unwrap();

    let set_global_array_kernel = hip_module.create_kernel("set_global_array_kernel").unwrap();

    unsafe {
        hip_check(hipMalloc(&mut d_in, size_bytes)).unwrap();
        hip_check(hipMalloc(&mut d_out, size_bytes)).unwrap();

        hip_check(hipMemcpy(
            d_in,
            h_in.as_ptr() as *const std::os::raw::c_void,
            size_bytes,
            hipMemcpyKind_hipMemcpyHostToDevice,
        ))
        .unwrap();

        // Fetch a device pointer to the device variable "global". We can pass the relevant
        // symbol directly to this function.
        // ??? HIP_SYMBOL

        let ret = hipModuleLaunchKernel(
            set_global_kernel,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        hip_check(ret).unwrap();

        // This pointer is a regular device pointer, and so we may use it in the same ways
        // as pointers allocated using `hipMalloc`.
        // ???

        #[allow(clippy::needless_range_loop)]
        for i in 0..DEVICE_ARRAY_SIZE {
            global_array[i] = i as c_float * 1000.0;
        }

        let ret = hipModuleLaunchKernel(
            set_global_array_kernel,
            1,
            1,
            1,
            block_size,
            1,
            1,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        hip_check(ret).unwrap();

        // Launch the kernel on the default stream and with the above configuration.
        #[repr(C)]
        #[derive(Debug, Copy, Clone)]
        struct Args {
            out: *mut std::os::raw::c_void,
            input: *mut std::os::raw::c_void,
            size: usize,
        }

        let args = Args {
            out: d_out,
            input: d_in,
            size: size as usize,
        };

        let args_size = std::mem::size_of_val(&args);

        let mut config = [
            HIP_LAUNCH_PARAM_BUFFER_POINTER as *mut std::os::raw::c_void,
            &args as *const _ as *mut std::os::raw::c_void,
            HIP_LAUNCH_PARAM_BUFFER_SIZE as *mut std::os::raw::c_void,
            &args_size as *const _ as *mut std::os::raw::c_void,
            HIP_LAUNCH_PARAM_END as *mut std::os::raw::c_void,
        ];

        // Launch the kernel on the default stream and with the above configuration.
        let ret = hipModuleLaunchKernel(
            test_globals_kernel,
            grid_size,
            1,
            1,
            block_size,
            1,
            1,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            config.as_mut_ptr(),
        );
        hip_check(ret).unwrap();

        // Check if the kernel launch was successful.
        hip_check(hipGetLastError()).unwrap();

        // Copy the results back to the host. This call blocks the host's execution until the copy is finished.
        hip_check(hipMemcpy(
            h_out.as_mut_ptr() as *mut std::os::raw::c_void,
            d_out,
            size_bytes,
            hipMemcpyKind_hipMemcpyDeviceToHost,
        ))
        .unwrap();

        // Free device memory.
        hip_check(hipFree(d_in)).unwrap();
        hip_check(hipFree(d_out)).unwrap();
    }

    // Compute the expected values on the host.
    let reference = test_globals_reference(&h_in, &global_array);

    // Check the results' validity.
    let mut errors = 0;

    for (i, v) in reference.into_iter().enumerate() {
        println!("index: {}, h_out: {}, reference: {}", i, h_out[i], v);
        if h_out[i] != v {
            errors += 1;
        }
    }

    assert_eq!(errors, 0, "Validation failed. Errors: {errors}");
    println!("Validation passed.");
}
