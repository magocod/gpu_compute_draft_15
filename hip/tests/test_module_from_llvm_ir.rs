use hip::error::hip_check;
use hip::memory::HipBuffer;
use hip_sys::hip_runtime_bindings::{
    hipFunction_t, hipModuleGetFunction, hipModuleLaunchKernel, hipModuleLoad, hipModule_t,
    HIP_LAUNCH_PARAM_BUFFER_POINTER, HIP_LAUNCH_PARAM_BUFFER_SIZE, HIP_LAUNCH_PARAM_END,
};
use std::env;
use std::ffi::CString;

const BLOCK_DIM_X: usize = 32;

fn global_array_put(module: hipModule_t, input: &[i32]) {
    let input_buf: HipBuffer<i32> = HipBuffer::new(BLOCK_DIM_X).unwrap();

    input_buf.memcpy_host_to_device(input).unwrap();

    unsafe {
        // Fetch a reference to the kernel that we are going to invoke.
        let mut kernel: hipFunction_t = std::ptr::null_mut();
        let k_name = CString::new("_Z16global_array_putPi").unwrap();

        let ret = hipModuleGetFunction(&mut kernel, module, k_name.as_ptr());
        hip_check(ret).unwrap();

        // Create and fill array with kernel arguments.
        #[repr(C)]
        #[derive(Debug, Copy, Clone)]
        struct Args {
            input: *mut std::os::raw::c_void,
        }

        let args = Args {
            input: input_buf.get_mem_ptr(),
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
            1,
            1,
            1,
            BLOCK_DIM_X as u32,
            1,
            1,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            config.as_mut_ptr(),
        );
        hip_check(ret).unwrap();
    }
}

fn global_array_get(module: hipModule_t) -> Vec<i32> {
    let output: HipBuffer<i32> = HipBuffer::new(BLOCK_DIM_X).unwrap();

    unsafe {
        // Fetch a reference to the kernel that we are going to invoke.
        let mut kernel: hipFunction_t = std::ptr::null_mut();
        let k_name = CString::new("_Z16global_array_getPi").unwrap();

        let ret = hipModuleGetFunction(&mut kernel, module, k_name.as_ptr());
        hip_check(ret).unwrap();

        // Create and fill array with kernel arguments.
        #[repr(C)]
        #[derive(Debug, Copy, Clone)]
        struct Args {
            output: *mut std::os::raw::c_void,
        }

        let args = Args {
            output: output.get_mem_ptr(),
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
            1,
            1,
            1,
            BLOCK_DIM_X as u32,
            1,
            1,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            config.as_mut_ptr(),
        );
        hip_check(ret).unwrap();
    }

    output.memcpy_device_to_host().unwrap()
}

fn global_array_increase(module: hipModule_t) -> Vec<i32> {
    let output: HipBuffer<i32> = HipBuffer::new(BLOCK_DIM_X).unwrap();

    unsafe {
        // Fetch a reference to the kernel that we are going to invoke.
        let mut kernel: hipFunction_t = std::ptr::null_mut();
        let k_name = CString::new("_Z21global_array_increasev").unwrap();

        let ret = hipModuleGetFunction(&mut kernel, module, k_name.as_ptr());
        hip_check(ret).unwrap();

        // Create and fill array with kernel arguments.
        #[repr(C)]
        #[derive(Debug, Copy, Clone)]
        struct Args {
            output: *mut std::os::raw::c_void,
        }

        let args = Args {
            output: output.get_mem_ptr(),
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
            1,
            1,
            1,
            BLOCK_DIM_X as u32,
            1,
            1,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            config.as_mut_ptr(),
        );
        hip_check(ret).unwrap();
    }

    output.memcpy_device_to_host().unwrap()
}

fn load_module(path: &str) -> hipModule_t {
    let module_path_cstring = CString::new(path).unwrap();

    // Load the module from the path that we just constructed.
    // If the module does not exist, this function will return an error.
    let mut module: hipModule_t = std::ptr::null_mut();

    unsafe {
        let ret = hipModuleLoad(&mut module, module_path_cstring.as_ptr());
        hip_check(ret).unwrap();
    }

    module
}

#[test]
fn test_module_api() {
    let module_file_name = "global_array-hip-amdgcn-amd-amdhsa_gfx1032.o";

    let mut module_path = env::current_dir().unwrap();
    module_path.push(module_file_name);
    println!("module_path: {:?}", module_path);

    // Load the module from the path that we just constructed.
    // If the module does not exist, this function will return an error.
    let module_1 = load_module(module_path.to_str().unwrap());

    let module_2 = load_module(module_path.to_str().unwrap());

    let input = vec![2; BLOCK_DIM_X];
    global_array_put(module_1, &input);

    let output = global_array_get(module_1);
    println!("output 1 {:?}", output);
    assert_eq!(output, input);

    global_array_increase(module_1);

    let output = global_array_get(module_1);
    println!("output 1 {:?}", output);
    assert_eq!(output, vec![3; BLOCK_DIM_X]);

    global_array_increase(module_1);
    global_array_increase(module_1);

    let output = global_array_get(module_1);
    println!("output 1 {:?}", output);
    assert_eq!(output, vec![5; BLOCK_DIM_X]);

    let output = global_array_get(module_2);
    println!("output 2 {:?}", output);
    assert_eq!(output, vec![0; BLOCK_DIM_X]);
}
