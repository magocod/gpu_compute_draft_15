use hip::error::hip_check;
use hip::utils::ceiling_div;
use hip_sys::hip_runtime_bindings::{
    hipFree, hipFunction_t, hipMalloc, hipMemcpy, hipMemcpyKind_hipMemcpyDeviceToHost,
    hipMemcpyKind_hipMemcpyHostToDevice, hipModuleGetFunction, hipModuleLaunchKernel,
    hipModuleLoad, hipModule_t, hipStreamDefault, HIP_LAUNCH_PARAM_BUFFER_POINTER,
    HIP_LAUNCH_PARAM_BUFFER_SIZE, HIP_LAUNCH_PARAM_END,
};
use std::env;
use std::ffi::{c_float, CString};

#[test]
fn test_module_api() {
    // The module file that contains the kernel that we want to invoke. This
    // file is expected to be in the same directory as the executable.
    let module_file_name = CString::new("module_gfx1032.co").unwrap();

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
    }

    // Compute an absolute path to the module that we are going to load.
    // To do that, find the directory where the example executable is placed in from the 0th argument.
    // Note that this does not always work (the executable may be invoked with a completely different
    // value for argv[0]), but works for the purposes of this example.
    let mut module_path = env::current_dir().unwrap();
    module_path.push(module_file_name.to_str().unwrap());
    println!("module_path: {:?}", module_path);

    let module_path_cstring = CString::new(module_path.to_str().unwrap()).unwrap();

    // Load the module from the path that we just constructed.
    // If the module does not exist, this function will return an error.
    let mut module: hipModule_t = std::ptr::null_mut();

    unsafe {
        let ret = hipModuleLoad(&mut module, module_path_cstring.as_ptr());
        hip_check(ret).unwrap();

        // Fetch a reference to the kernel that we are going to invoke.
        let mut kernel: hipFunction_t = std::ptr::null_mut();
        let k_name = CString::new("test_module_api_kernel").unwrap();

        let ret = hipModuleGetFunction(&mut kernel, module, k_name.as_ptr());
        hip_check(ret).unwrap();

        // Create and fill array with kernel arguments.
        #[repr(C)]
        #[derive(Debug, Copy, Clone)]
        struct Args {
            out: *mut std::os::raw::c_void,
            input: *mut std::os::raw::c_void,
        }

        let args = Args {
            out: d_out,
            input: d_in,
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
            kernel,
            grid_size,
            1,
            1,
            block_size,
            1,
            1,
            0,
            hipStreamDefault as *mut _,
            std::ptr::null_mut(),
            config.as_mut_ptr(),
        );
        hip_check(ret).unwrap();

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

    // Check the results' validity.
    let mut errors = 0;

    for i in 0..size as usize {
        println!("index: {}, h_out: {}, h_in: {}", i, h_out[i], h_in[i]);
        if h_out[i] != h_in[i] {
            errors += 1;
        }
    }

    assert_eq!(errors, 0, "Validation failed. Errors: {errors}");
    println!("Validation passed.");
}
