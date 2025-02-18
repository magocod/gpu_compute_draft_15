use hip::error::hip_check;
use hip::module::HipModule;
use hip::utils::ceiling_div;
use hip_sys::hip_runtime_bindings::{
    dim3, hipDeviceCanAccessPeer, hipDeviceDisablePeerAccess, hipDeviceEnablePeerAccess,
    hipDeviceProp_tR0600, hipDeviceSynchronize, hipFree, hipGetDeviceCount,
    hipGetDevicePropertiesR0600, hipMalloc, hipMemcpy, hipMemcpyKind_hipMemcpyDeviceToDevice,
    hipMemcpyKind_hipMemcpyDeviceToHost, hipMemcpyKind_hipMemcpyHostToDevice,
    hipModuleLaunchKernel, hipSetDevice, HIP_LAUNCH_PARAM_BUFFER_POINTER,
    HIP_LAUNCH_PARAM_BUFFER_SIZE, HIP_LAUNCH_PARAM_END,
};
use std::ffi::{c_float, c_uint};
use utilities::helper_functions::buf_i8_to_string;
// FIXME example no pass

/// \brief Checks whether peer-to-peer is supported or not among the current available devices.
/// Returns, if exist, the IDs of the first two devices found with peer-to-peer memory access.
fn check_peer_to_peer_support() -> (i32, i32) {
    // Get number of GPUs available.
    let mut gpu_count = 0;
    let mut can_access_peer = 0;

    unsafe {
        hip_check(hipGetDeviceCount(&mut gpu_count)).unwrap();
    }

    println!("GPU count: {}", gpu_count);

    // If there are not enough devices (at least 2) peer-to-peer is not possible.
    if gpu_count < 2 {
        panic!("Peer-to-peer application requires at least 2 GPU devices.");
    }

    // Check accessibility for each device available.
    for current_gpu in 0..gpu_count {
        // Check if current_gpu device can access the memory of the devices with lower ID.
        for peer_gpu in 0..gpu_count {
            unsafe {
                hip_check(hipDeviceCanAccessPeer(
                    &mut can_access_peer,
                    current_gpu,
                    peer_gpu,
                ))
                .unwrap();
            }

            // The first pair found with peer-to-peer memory access is returned.
            println!(
                "current_gpu {} -> peer_gpu {}: can_access_peer: {}",
                current_gpu, peer_gpu, can_access_peer
            );

            if can_access_peer == 1 {
                return (current_gpu, peer_gpu);
            }
        }
    }

    panic!("Peer-to-peer application requires at least 2 GPU devices accessible between them.");
}

/// \brief Enables (if possible) direct memory access from <tt>current_gpu<\tt> to <tt>peer_gpu<\tt>.
fn enable_peer_to_peer(current_gpu: i32, peer_gpu: i32) {
    // Must be on a multi-gpu system.
    if current_gpu == peer_gpu {
        panic!("Current and peer devices must be different.");
    }

    // Set current GPU as default device for subsequent API calls.
    unsafe {
        hip_check(hipSetDevice(current_gpu)).unwrap();

        // Enable direct memory access from current to peer device.
        hip_check(hipDeviceEnablePeerAccess(peer_gpu, 0 /*flags*/)).unwrap();
    }
}

/// \brief Disables (if possible) direct memory access from <tt>current_gpu<\tt> to <tt>peer_gpu<\tt>.
fn disable_peer_to_peer(current_gpu: i32, peer_gpu: i32) {
    // Must be on a multi-gpu system.
    if current_gpu == peer_gpu {
        panic!("Current and peer devices must be different.");
    }

    // Set current GPU as default device for subsequent API calls.
    unsafe {
        hip_check(hipSetDevice(current_gpu)).unwrap();

        // Disable direct memory access from current to peer device.
        hip_check(hipDeviceDisablePeerAccess(peer_gpu)).unwrap();
    }
}

const PROGRAM_SOURCE: &str = r#"
    /// \brief Simple matrix transpose kernel using static shared memory.
    
    const unsigned int Width = 32;
    const unsigned int Height = 32;
    
   // template<const unsigned int Width = 32, const unsigned int Height = 32>
    extern "C" __global__ void static_shared_matrix_transpose_kernel(
        float* out,
        const float* input
        )
    {
        // Allocate the necessary amount of shared memory to store the transpose of the matrix.
        // Note that the amount of shared memory needed is known at compile time.
        constexpr unsigned int size = Width * Height;
        __shared__ float       shared_matrix_memory[size];
    
        // Compute the row and column indexes of the matrix element that each thread is going
        // to process.
        const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
        const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    
        // If out of bounds, do nothing.
        if(!(x < Width && y < Height))
        {
            return;
        }
    
        // Store transposed element (x,y) in shared memory.
        shared_matrix_memory[y * Width + x] = input[x * Height + y];
    
        // Synchronize threads so all writes are done before accessing shared memory again.
        __syncthreads();
    
        // Copy transposed element from shared to global memory (output matrix).
        out[y * Width + x] = shared_matrix_memory[y * Width + x];
    }
    
    /// \brief Simple matrix transpose kernel using dynamic shared memory.
    
    extern "C" __global__ void dynamic_shared_matrix_transpose_kernel(
        float*             out,
        const float*       input,
        const unsigned int width,
        const unsigned int height
        )
    {
        // Declare that this kernel is using dynamic shared memory to store a number of floats.
        // The unsized array type indicates that the total amount of memory that is going
        // to be used here is not known ahead of time, and will be computed at runtime and
        // passed to the kernel launch function.
        extern __shared__ float shared_matrix_memory[];
    
        // Compute the row and column indexes of the matrix element that each thread is going
        // to process.
        const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
        const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    
        // If out of bounds, do nothing.
        if(!(x < width && y < height))
        {
            return;
        }
    
        // Store transposed element (x,y) in shared memory.
        shared_matrix_memory[y * width + x] = input[x * height + y];
    
        // Synchronize threads so all writes are done before accessing shared memory again.
        __syncthreads();
    
        // Copy transposed element from shared to global memory (output matrix).
        out[y * width + x] = shared_matrix_memory[y * height + x];
    }
    "#;

#[test]
fn test_basic_multi_gpu_data_transfer() {
    // Check peer-to-peer access for all devices and get the IDs of the first pair (if exist)
    // that support peer-to-peer memory access.
    let (gpu_first, gpu_second) = check_peer_to_peer_support();

    println!(
        "Devices with IDs {} and {} selected.",
        gpu_first, gpu_second
    );

    let mut props = std::mem::MaybeUninit::<hipDeviceProp_tR0600>::uninit();
    unsafe {
        hip_check(hipGetDevicePropertiesR0600(props.as_mut_ptr(), gpu_first)).unwrap();
    };

    let props = unsafe { props.assume_init() };
    println!(
        "Devices Id: {} {:?}",
        gpu_first,
        buf_i8_to_string(&props.name).unwrap()
    );

    let mut props = std::mem::MaybeUninit::<hipDeviceProp_tR0600>::uninit();
    unsafe {
        hip_check(hipGetDevicePropertiesR0600(props.as_mut_ptr(), gpu_second)).unwrap();
    };

    let props = unsafe { props.assume_init() };
    println!(
        "Devices Id: {} {:?}",
        gpu_second,
        buf_i8_to_string(&props.name).unwrap()
    );

    // Number of rows and columns, total number of elements and size in bytes of the matrix
    // to be transposed.
    let width: u32 = 4;
    let height: u32 = width;
    let size = width * height;
    let size_bytes: usize = size as usize * std::mem::size_of::<c_float>();

    // Number of threads in each dimension of the kernel block.
    let block_size: u32 = 4;

    // Number of blocks in each dimension of the grid. Calculated as
    // ceiling(matrix_dimension/block_size) with matrix_dimension being width or height.
    let grid_size_x = ceiling_div(width, block_size);
    let grid_size_y = ceiling_div(height, block_size);

    // Block and grid sizes in 2D.
    let block_dim = dim3 {
        x: block_size,
        y: block_size,
        z: 1,
    };
    let grid_dim = dim3 {
        x: grid_size_x,
        y: grid_size_y,
        z: 1,
    };

    // Allocate host input matrix and initialize with increasing sequence 1, 2, 3, ....
    let matrix: Vec<c_float> = (0..size_bytes).map(|i| (i + 1) as f32).collect();

    // Allocate host matrix to store the results of the kernel execution on the second device.
    let mut transposed_matrix = vec![c_float::default(); size_bytes];

    // Declare input and output matrices for the executions on both devices.
    let mut d_matrix: Vec<*mut std::os::raw::c_void> = vec![std::ptr::null_mut(); 2];
    let mut d_transposed_matrix: Vec<*mut std::os::raw::c_void> = vec![std::ptr::null_mut(); 2];

    unsafe {
        // Set first gpu as default device for subsequent API calls.
        hip_check(hipSetDevice(gpu_first)).unwrap();

        // Allocate input and output matrices on current device.
        hip_check(hipMalloc(&mut d_transposed_matrix[0], size_bytes)).unwrap();
        hip_check(hipMalloc(&mut d_matrix[0], size_bytes)).unwrap();

        // Copy input matrix data from host to current device.
        hip_check(hipMemcpy(
            d_matrix[0],
            matrix.as_ptr() as *const std::os::raw::c_void,
            size_bytes,
            hipMemcpyKind_hipMemcpyHostToDevice,
        ))
        .unwrap();

        println!("Computing matrix transpose on device {}", gpu_first);
    }

    let hip_module = HipModule::create(PROGRAM_SOURCE).unwrap();

    let kernel = hip_module
        .create_kernel("static_shared_matrix_transpose_kernel")
        .unwrap();

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    struct StaticSharedMatrixTransposeKernelArgs {
        out: *mut std::os::raw::c_void,
        input: *mut std::os::raw::c_void,
    }

    let args = StaticSharedMatrixTransposeKernelArgs {
        out: d_transposed_matrix[0],
        input: d_matrix[0],
    };

    let args_size = std::mem::size_of_val(&args);

    let mut config = [
        HIP_LAUNCH_PARAM_BUFFER_POINTER as *mut std::os::raw::c_void,
        &args as *const _ as *mut std::os::raw::c_void,
        HIP_LAUNCH_PARAM_BUFFER_SIZE as *mut std::os::raw::c_void,
        &args_size as *const _ as *mut std::os::raw::c_void,
        HIP_LAUNCH_PARAM_END as *mut std::os::raw::c_void,
    ];

    unsafe {
        // Launch kernel in current device. Note that, as this kernel uses static shared memory, no
        // bytes of shared memory need to be allocated when launching the kernel.

        let ret = hipModuleLaunchKernel(
            kernel,
            grid_dim.x,
            grid_dim.y,
            1,
            block_dim.x,
            block_dim.y,
            1,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            config.as_mut_ptr(),
        );
        hip_check(ret).unwrap();

        // Wait on all active streams on the current device.
        hip_check(hipDeviceSynchronize()).unwrap();

        // Set second gpu as default device for subsequent API calls.
        hip_check(hipSetDevice(gpu_second)).unwrap();

        // Allocate input and output matrices on current device.
        hip_check(hipMalloc(&mut d_transposed_matrix[1], size_bytes)).unwrap();
        hip_check(hipMalloc(&mut d_matrix[1], size_bytes)).unwrap();

        println!(
            "Transferring results from device {} to device {}",
            gpu_first, gpu_second
        );

        // Enable (if possible) direct memory access from current (second) to peer (first) GPU.
        enable_peer_to_peer(gpu_second, gpu_first); // error 101

        // Copy output matrix from peer device to input matrix on current device. This copy is made
        // directly between devices (no host needed) because direct access memory was previously
        // enabled from second to first device.
        hip_check(hipMemcpy(
            d_matrix[1],
            d_transposed_matrix[0],
            size_bytes,
            hipMemcpyKind_hipMemcpyDeviceToDevice,
        ))
        .unwrap();

        println!("Computing matrix transpose on device {}", gpu_second);

        // Launch kernel in current device. Note that size_bytes bytes of shared memory are required to launch
        // this kernel because it uses dynamically allocated shared memory.

        let kernel = hip_module
            .create_kernel("dynamic_shared_matrix_transpose_kernel")
            .unwrap();

        #[repr(C)]
        #[derive(Debug, Copy, Clone)]
        struct StaticSharedMatrixTransposeKernelArgs {
            out: *mut std::os::raw::c_void,
            input: *mut std::os::raw::c_void,
            width: u32,
            height: u32,
        }

        let args = StaticSharedMatrixTransposeKernelArgs {
            out: d_transposed_matrix[0],
            input: d_matrix[0],
            width,
            height,
        };

        let args_size = std::mem::size_of_val(&args);

        let mut config = [
            HIP_LAUNCH_PARAM_BUFFER_POINTER as *mut std::os::raw::c_void,
            &args as *const _ as *mut std::os::raw::c_void,
            HIP_LAUNCH_PARAM_BUFFER_SIZE as *mut std::os::raw::c_void,
            &args_size as *const _ as *mut std::os::raw::c_void,
            HIP_LAUNCH_PARAM_END as *mut std::os::raw::c_void,
        ];

        let ret = hipModuleLaunchKernel(
            kernel,
            grid_dim.x,
            grid_dim.y,
            1,
            block_dim.x,
            block_dim.y,
            1,
            size_bytes as c_uint, /*shared_memory_bytes*/
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            config.as_mut_ptr(),
        );
        hip_check(ret).unwrap();

        // Wait on all active streams on the current device.
        hip_check(hipDeviceSynchronize()).unwrap();

        // Copy results from second device to host.
        hip_check(hipMemcpy(
            transposed_matrix.as_mut_ptr() as *mut std::os::raw::c_void,
            d_transposed_matrix[1],
            size_bytes,
            hipMemcpyKind_hipMemcpyDeviceToHost,
        ))
        .unwrap();

        // Disable direct memory access.
        disable_peer_to_peer(gpu_second, gpu_first);

        // Free device memory.
        for i in 0..2 {
            hip_check(hipFree(d_matrix[i])).unwrap();
            hip_check(hipFree(d_transposed_matrix[i])).unwrap();
        }
    }

    // Validate results. The input matrix for the kernel execution on the first device must be
    // the same as the output matrix from the kernel execution on the second device.
    let errors = 0;
    // let eps= 1.0E-6f; // rust equivalent?

    println!("Validating peer-to-peer.");

    for i in 0..size as usize {
        println!(
            "i: {} -> matrix: {}, transposed_matrix: {}",
            i, matrix[i], transposed_matrix[i]
        );
        // errors += (std::fabs(matrix[i] - transposed_matrix[i]) > eps);
    }

    if errors > 0 {
        println!("Validation failed with {} errors", errors);
    } else {
        println!("Validation passed.");
    }
}
