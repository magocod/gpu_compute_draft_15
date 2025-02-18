use hsa_sys::bindings::{
    hsa_agent_get_info, hsa_agent_info_t_HSA_AGENT_INFO_DEVICE, hsa_agent_t,
    hsa_amd_agent_iterate_memory_pools, hsa_amd_agents_allow_access, hsa_amd_memory_fill,
    hsa_amd_memory_pool_allocate, hsa_amd_memory_pool_free, hsa_amd_memory_pool_get_info,
    hsa_amd_memory_pool_global_flag_s_HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT,
    hsa_amd_memory_pool_info_t_HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
    hsa_amd_memory_pool_info_t_HSA_AMD_MEMORY_POOL_INFO_SEGMENT, hsa_amd_memory_pool_t,
    hsa_amd_segment_t, hsa_amd_segment_t_HSA_AMD_SEGMENT_GLOBAL,
    hsa_code_object_reader_create_from_file, hsa_code_object_reader_t,
    hsa_default_float_rounding_mode_t_HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, hsa_device_type_t,
    hsa_device_type_t_HSA_DEVICE_TYPE_CPU, hsa_device_type_t_HSA_DEVICE_TYPE_GPU,
    hsa_executable_create_alt, hsa_executable_freeze, hsa_executable_get_symbol,
    hsa_executable_load_agent_code_object, hsa_executable_symbol_get_info,
    hsa_executable_symbol_info_t_HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
    hsa_executable_symbol_info_t_HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
    hsa_executable_symbol_info_t_HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
    hsa_executable_symbol_info_t_HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
    hsa_executable_symbol_t, hsa_executable_t, hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM,
    hsa_file_t, hsa_init, hsa_iterate_agents, hsa_kernel_dispatch_packet_t,
    hsa_packet_header_t_HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE,
    hsa_packet_header_t_HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE,
    hsa_packet_type_t_HSA_PACKET_TYPE_KERNEL_DISPATCH, hsa_profile_t_HSA_PROFILE_FULL,
    hsa_queue_add_write_index_relaxed, hsa_queue_create, hsa_queue_destroy,
    hsa_queue_load_write_index_relaxed, hsa_queue_store_write_index_relaxed, hsa_queue_t,
    hsa_queue_type_t_HSA_QUEUE_TYPE_MULTI, hsa_shut_down,
    hsa_signal_condition_t_HSA_SIGNAL_CONDITION_LT, hsa_signal_create, hsa_signal_destroy,
    hsa_signal_store_relaxed, hsa_signal_store_screlease, hsa_signal_t, hsa_signal_value_t,
    hsa_signal_wait_scacquire, hsa_status_t, hsa_status_t_HSA_STATUS_ERROR,
    hsa_status_t_HSA_STATUS_INFO_BREAK, hsa_status_t_HSA_STATUS_SUCCESS,
    hsa_wait_state_t_HSA_WAIT_STATE_BLOCKED,
};
use libc::{close, intptr_t, memcpy, memset, open, size_t, uintptr_t, O_RDONLY};
use std::ffi::CString;
use std::{env, thread};

// based on the binary search example changing the code object for a simpler one

// Hold all the info specific to binary search
#[derive(Debug)]
struct BinarySearch {
    // Binary Search parameters
    // length: u32,
    work_group_size: u32,
    work_grid_size: u32,
    // num_sub_divisions: u32,
    // find_me: u32,

    // Buffers needed for this application
    input: *mut std::os::raw::c_void, // *mut i32
    // input_arr: *mut std::os::raw::c_void, // *mut i32
    output: *mut std::os::raw::c_void, // *mut i32

    // Keneral argument buffers and addresses
    kern_arg_buffer: *mut std::os::raw::c_void, // Begin of allocated memory
    //  this pointer to be deallocated
    kern_arg_address: *mut std::os::raw::c_void, // Properly aligned address to be used in aql
    // packet (don't use for deallocation)

    // Kernel code
    kernel_file_name: String,
    kernel_name: String,
    kernarg_size: u32,
    kernarg_align: u32,

    // HSA/RocR objects needed for this application
    gpu_dev: hsa_agent_t,
    cpu_dev: hsa_agent_t,
    signal: hsa_signal_t,
    queue: *mut hsa_queue_t,
    cpu_pool: hsa_amd_memory_pool_t,
    gpu_pool: hsa_amd_memory_pool_t,
    kern_arg_pool: hsa_amd_memory_pool_t,

    // Other items we need to populate AQL packet
    kernel_object: u64,
    group_segment_size: u32,   // < Kernel group seg size
    private_segment_size: u32, // < Kernel private seg size
}

fn initialize_binary_search() -> BinarySearch {
    BinarySearch {
        // length: 0,
        work_group_size: 0,
        work_grid_size: 0,
        // num_sub_divisions: 0,
        // find_me: 0,
        input: std::ptr::null_mut(),
        output: std::ptr::null_mut(),
        kern_arg_buffer: std::ptr::null_mut(),
        kern_arg_address: std::ptr::null_mut(),
        kernel_file_name: "global_array-hip-amdgcn-amd-amdhsa_gfx1032.o".to_string(),
        kernel_name: "_Z19global_array_insertPiS_.kd".to_string(),
        kernarg_size: 0,
        kernarg_align: 0,
        gpu_dev: hsa_agent_t { handle: 0 },
        cpu_dev: hsa_agent_t { handle: 0 },
        signal: hsa_signal_t { handle: 0 },
        queue: std::ptr::null_mut(),
        cpu_pool: hsa_amd_memory_pool_t { handle: 0 },
        gpu_pool: hsa_amd_memory_pool_t { handle: 0 },
        kern_arg_pool: hsa_amd_memory_pool_t { handle: 0 },
        kernel_object: 0,
        group_segment_size: 0,
        private_segment_size: 0,
    }
}

// This function is called by the call-back functions used to find an agent of
// the specified hsa_device_type_t. Note that it cannot be called directly from
// hsa_iterate_agents() as it does not match the prototype of the call-back
// function. It must be wrapped by a function with the correct prototype.
//
// Return values:
//  HSA_STATUS_INFO_BREAK -- "agent" is of the specified type (dev_type)
//  HSA_STATUS_SUCCESS -- "agent" is not of the specified type
//  Other -- Some error occurred
unsafe fn find_agent(
    agent: hsa_agent_t,
    data: *mut std::os::raw::c_void,
    dev_type: hsa_device_type_t,
) -> hsa_status_t {
    // See if the provided agent matches the input type (dev_type)
    let mut hsa_device_type: hsa_device_type_t = 0;

    let err = hsa_agent_get_info(
        agent,
        hsa_agent_info_t_HSA_AGENT_INFO_DEVICE,
        &mut hsa_device_type as *mut _ as *mut std::os::raw::c_void,
    );
    if err != hsa_status_t_HSA_STATUS_SUCCESS {
        return err;
    }

    if hsa_device_type == dev_type {
        let ag = &mut *(data as *mut hsa_agent_t);
        *ag = agent;

        return hsa_status_t_HSA_STATUS_INFO_BREAK;
    }

    hsa_status_t_HSA_STATUS_SUCCESS
}

// This is the call-back function used to find a GPU type agent. Note that the
// prototype of this function is dictated by the HSA specification
unsafe extern "C" fn find_gpu_device(
    agent: hsa_agent_t,
    data: *mut std::os::raw::c_void,
) -> hsa_status_t {
    find_agent(agent, data, hsa_device_type_t_HSA_DEVICE_TYPE_GPU)
}

// This is the call-back function used to find a CPU type agent. Note that the
// prototype of this function is dictated by the HSA specification
unsafe extern "C" fn find_cpu_device(
    agent: hsa_agent_t,
    data: *mut std::os::raw::c_void,
) -> hsa_status_t {
    find_agent(agent, data, hsa_device_type_t_HSA_DEVICE_TYPE_CPU)
}

// Find the CPU and GPU agents we need to run this sample, and save them in the
// BinarySearch structure for later use.
unsafe fn find_devices(bs: &mut BinarySearch) -> hsa_status_t {
    let gpu_ptr = &mut bs.gpu_dev as *mut _ as *mut std::os::raw::c_void;
    let cpu_ptr = &mut bs.cpu_dev as *mut _ as *mut std::os::raw::c_void;

    // Note that hsa_iterate_agents iterate through all known agents until
    // HSA_STATUS_SUCCESS is not returned. The call-backs are implemented such
    // that HSA_STATUS_INFO_BREAK means we found an agent of the specified type.
    // This value is returned by hsa_iterate_agents.
    let err = hsa_iterate_agents(Some(find_gpu_device), gpu_ptr);

    if err != hsa_status_t_HSA_STATUS_INFO_BREAK {
        return hsa_status_t_HSA_STATUS_ERROR;
    }

    if 0 == bs.gpu_dev.handle {
        println!("GPU Device is not Created properly!");
        if err != hsa_status_t_HSA_STATUS_SUCCESS {
            return err;
        }
    }

    let err = hsa_iterate_agents(Some(find_cpu_device), cpu_ptr);

    if err != hsa_status_t_HSA_STATUS_INFO_BREAK {
        return hsa_status_t_HSA_STATUS_ERROR;
    }

    if 0 == bs.cpu_dev.handle {
        println!("CPU Device is not Created properly!");
        if err != hsa_status_t_HSA_STATUS_SUCCESS {
            return err;
        }
    }

    hsa_status_t_HSA_STATUS_SUCCESS
}

// This function checks to see if the provided
// pool has the HSA_AMD_SEGMENT_GLOBAL property. If the kern_arg flag is true,
// the function adds an additional requirement that the pool have the
// HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT property. If kern_arg is false,
// pools must NOT have this property.
// Upon finding a pool that meets these conditions, HSA_STATUS_INFO_BREAK is
// returned. HSA_STATUS_SUCCESS is returned if no errors were encountered, but
// no pool was found meeting the requirements. If an error is encountered, we
// return that error.

// Note that this function does not match the required prototype for the
// hsa_amd_agent_iterate_memory_pools call back function, and therefore must be
// wrapped by a function with the correct prototype.
unsafe fn find_global_pool(
    pool: hsa_amd_memory_pool_t,
    data: *mut std::os::raw::c_void,
    kern_arg: bool,
) -> hsa_status_t {
    let mut segment: hsa_amd_segment_t = 0;
    let mut flag: u32 = 0;

    let ret = hsa_amd_memory_pool_get_info(
        pool,
        hsa_amd_memory_pool_info_t_HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
        &mut segment as *mut _ as *mut std::os::raw::c_void,
    );

    if ret != hsa_status_t_HSA_STATUS_SUCCESS {
        return ret;
    }

    if hsa_amd_segment_t_HSA_AMD_SEGMENT_GLOBAL != segment {
        return hsa_status_t_HSA_STATUS_SUCCESS;
    }

    let ret = hsa_amd_memory_pool_get_info(
        pool,
        hsa_amd_memory_pool_info_t_HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
        &mut flag as *mut _ as *mut std::os::raw::c_void,
    );

    if ret != hsa_status_t_HSA_STATUS_SUCCESS {
        return ret;
    }

    let karg_st: u32 =
        flag & hsa_amd_memory_pool_global_flag_s_HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT;

    if (karg_st == 0 && kern_arg) || (karg_st != 0 && !kern_arg) {
        return hsa_status_t_HSA_STATUS_SUCCESS;
    }

    let p = data as *mut hsa_amd_memory_pool_t;
    *p = pool;

    hsa_status_t_HSA_STATUS_INFO_BREAK
}

// This is the call-back function for hsa_amd_agent_iterate_memory_pools() that
// finds a pool with the properties of HSA_AMD_SEGMENT_GLOBAL and that is NOT
// HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT
unsafe extern "C" fn find_standard_pool(
    pool: hsa_amd_memory_pool_t,
    data: *mut ::std::os::raw::c_void,
) -> hsa_status_t {
    find_global_pool(pool, data, false)
}

// This is the call-back function for hsa_amd_agent_iterate_memory_pools() that
// finds a pool with the properties of HSA_AMD_SEGMENT_GLOBAL and that IS
// HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT
unsafe extern "C" fn find_kern_arg_pool(
    pool: hsa_amd_memory_pool_t,
    data: *mut ::std::os::raw::c_void,
) -> hsa_status_t {
    find_global_pool(pool, data, true)
}

// Find memory pools that we will need to allocate from for this sample
// application. We will need memory associated with the host CPU, the GPU
// executing the kernels, and for kernel arguments. This function will
// save the found pools to the BinarySearch structure for use elsewhere
// in this program.
unsafe fn find_pools(bs: &mut BinarySearch) -> hsa_status_t {
    let err = hsa_amd_agent_iterate_memory_pools(
        bs.cpu_dev,
        Some(find_standard_pool),
        &mut bs.cpu_pool as *mut _ as *mut std::os::raw::c_void,
    );

    if err != hsa_status_t_HSA_STATUS_INFO_BREAK {
        return hsa_status_t_HSA_STATUS_ERROR;
    }

    let err = hsa_amd_agent_iterate_memory_pools(
        bs.gpu_dev,
        Some(find_standard_pool),
        &mut bs.gpu_pool as *mut _ as *mut std::os::raw::c_void,
    );

    if err != hsa_status_t_HSA_STATUS_INFO_BREAK {
        return hsa_status_t_HSA_STATUS_ERROR;
    }

    let err = hsa_amd_agent_iterate_memory_pools(
        bs.cpu_dev,
        Some(find_kern_arg_pool),
        &mut bs.kern_arg_pool as *mut _ as *mut std::os::raw::c_void,
    );

    if err != hsa_status_t_HSA_STATUS_INFO_BREAK {
        return hsa_status_t_HSA_STATUS_ERROR;
    }

    hsa_status_t_HSA_STATUS_SUCCESS
}

// Once the needed memory pools have been found and the BinarySearch structure
// has been updated with these handles, this function is then used to allocate
// memory from those pools.
// Devices with which a pool is associated already have access to the pool.
// However, other devices may also need to read or write to that memory. Below,
// we see how we can grant access to other devices to address this issue.
unsafe fn allocate_and_init_buffers(bs: &mut BinarySearch) -> hsa_status_t {
    let out_length = 32 * std::mem::size_of::<i32>();
    let in_length = out_length;

    // In all of these examples, we want both the cpu and gpu to have access to
    // the buffer in question. We use the array of agents below in the susequent
    // calls to hsa_amd_agents_allow_access() for this purpose.
    let ag_list = [bs.gpu_dev, bs.cpu_dev];

    let ret = hsa_amd_memory_pool_allocate(bs.cpu_pool, in_length, 0, &mut bs.input);
    assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

    let ret = hsa_amd_agents_allow_access(2, ag_list.as_ptr(), std::ptr::null_mut(), bs.input);
    assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

    // memset(bs.input, 0, in_length);

    let ret = hsa_amd_memory_fill(bs.input, 4, 32);
    assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

    let ret = hsa_amd_memory_pool_allocate(bs.cpu_pool, out_length, 0, &mut bs.output);
    assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

    let ret = hsa_amd_agents_allow_access(2, ag_list.as_ptr(), std::ptr::null_mut(), bs.output);
    assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

    memset(bs.output, 0, in_length);

    ret
}

// The code in this function illustrates how to load a kernel from
// pre-compiled code. The goal is to get a handle that can be later
// used in an AQL packet and also to extract information about kernel
// that we will need. All of the information hand kernel handle will
// be saved to the BinarySearch structure. It will be used when we
// populate the AQL packet.
unsafe fn load_kernel_from_obj_file(bs: &mut BinarySearch) -> hsa_status_t {
    let mut code_obj_rdr = hsa_code_object_reader_t { handle: 0 };
    let mut executable = hsa_executable_t { handle: 0 };

    let mut file_path = env::current_dir().unwrap();
    file_path.push(&bs.kernel_file_name);
    println!("file_path: {:?}", file_path);

    let f_n = CString::new(file_path.to_string_lossy().to_string()).unwrap();
    let file_handle: hsa_file_t = open(f_n.as_ptr(), O_RDONLY);

    if file_handle == -1 {
        panic!("failed to open file");
    }

    let err = hsa_code_object_reader_create_from_file(file_handle, &mut code_obj_rdr);
    close(file_handle);
    assert_eq!(err, hsa_status_t_HSA_STATUS_SUCCESS);

    let err = hsa_executable_create_alt(
        hsa_profile_t_HSA_PROFILE_FULL,
        hsa_default_float_rounding_mode_t_HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
        std::ptr::null_mut(),
        &mut executable,
    );
    assert_eq!(err, hsa_status_t_HSA_STATUS_SUCCESS);

    let err = hsa_executable_load_agent_code_object(
        executable,
        bs.gpu_dev,
        code_obj_rdr,
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    );
    assert_eq!(err, hsa_status_t_HSA_STATUS_SUCCESS);

    let err = hsa_executable_freeze(executable, std::ptr::null_mut());
    assert_eq!(err, hsa_status_t_HSA_STATUS_SUCCESS);

    let mut kern_sym = hsa_executable_symbol_t { handle: 0 };
    let k_n = CString::new(bs.kernel_name.as_str()).unwrap();
    let err = hsa_executable_get_symbol(
        executable,
        std::ptr::null_mut(),
        k_n.as_ptr(),
        bs.gpu_dev,
        0,
        &mut kern_sym,
    );
    assert_eq!(err, hsa_status_t_HSA_STATUS_SUCCESS);

    let err = hsa_executable_symbol_get_info(
        kern_sym,
        hsa_executable_symbol_info_t_HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
        &mut bs.kernel_object as *mut _ as *mut std::os::raw::c_void,
    );
    assert_eq!(err, hsa_status_t_HSA_STATUS_SUCCESS);

    let err = hsa_executable_symbol_get_info(
        kern_sym,
        hsa_executable_symbol_info_t_HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
        &mut bs.private_segment_size as *mut _ as *mut std::os::raw::c_void,
    );
    assert_eq!(err, hsa_status_t_HSA_STATUS_SUCCESS);

    let err = hsa_executable_symbol_get_info(
        kern_sym,
        hsa_executable_symbol_info_t_HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
        &mut bs.group_segment_size as *mut _ as *mut std::os::raw::c_void,
    );
    assert_eq!(err, hsa_status_t_HSA_STATUS_SUCCESS);

    // Remaining queries not supported on code object v3.
    let err = hsa_executable_symbol_get_info(
        kern_sym,
        hsa_executable_symbol_info_t_HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
        &mut bs.kernarg_size as *mut _ as *mut std::os::raw::c_void,
    );
    assert_eq!(err, hsa_status_t_HSA_STATUS_SUCCESS);

    let err = hsa_executable_symbol_get_info(
        kern_sym,
        hsa_executable_symbol_info_t_HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
        &mut bs.kernarg_align as *mut _ as *mut std::os::raw::c_void,
    );
    assert_eq!(err, hsa_status_t_HSA_STATUS_SUCCESS);

    assert!(
        bs.kernarg_align >= 16,
        "Reported kernarg size is too small."
    );

    err
}

// AlignDown and AlignUp are 2 utility functions we use to find an aligned
// boundary either below or above a given value (address). The function will
// return a value that has the specified alignment.
fn _align_down(value: intptr_t, alignment: size_t) -> intptr_t {
    let b = alignment != 0;
    assert!(b, "Zero alignment");
    value & (alignment - 1) as intptr_t
}

fn _align_up(value: *mut std::os::raw::c_void, alignment: size_t) -> *mut std::os::raw::c_void {
    let r = _align_down(value as intptr_t + alignment as intptr_t - 1, alignment);
    r as *mut std::os::raw::c_void
}

// This function populates the AQL patch with the information
// we have collected and stored in the BinarySearch structure thus far.
unsafe fn populate_aqlpacket(bs: &BinarySearch) -> hsa_kernel_dispatch_packet_t {
    hsa_kernel_dispatch_packet_t {
        header: 0,
        setup: 1,
        workgroup_size_x: bs.work_group_size as u16,
        workgroup_size_y: 1,
        workgroup_size_z: 1,
        reserved0: 0,
        grid_size_x: bs.work_grid_size,
        grid_size_y: 1,
        grid_size_z: 1,
        private_segment_size: bs.private_segment_size,
        group_segment_size: bs.group_segment_size,
        kernel_object: bs.kernel_object,
        kernarg_address: bs.kern_arg_address,
        reserved2: 0,
        completion_signal: bs.signal,
    }
}

// This function allocates memory from the kern_arg pool we already found, and
// then sets the argument values needed by the kernel code.
unsafe fn alloc_and_set_kern_args(
    bs: &mut BinarySearch,
    args: *mut std::os::raw::c_void,
    arg_size: usize,
) -> hsa_status_t {
    let aql_buf_ptr: *mut *mut std::os::raw::c_void = &mut bs.kern_arg_address;
    let mut kern_arg_buf: *mut std::os::raw::c_void = std::ptr::null_mut();

    // The kernel code must be written to memory at the correct alignment. We
    // already queried the executable to get the correct alignment, which is
    // stored in bs->kernarg_align. In case the memory returned from
    // hsa_amd_memory_pool is not of the correct alignment, we request a little
    // more than what we need in case we need to adjust.
    let req_align: usize = bs.kernarg_align as usize;
    // Allocate enough extra space for alignment adjustments if ncessary
    let buf_size = arg_size + (req_align << 1);

    println!("arg_size: {}", arg_size);
    println!("buf_size: {}", buf_size);
    println!("req_align: {}", req_align);

    let err = hsa_amd_memory_pool_allocate(bs.kern_arg_pool, arg_size, 0, &mut kern_arg_buf);
    assert_eq!(err, hsa_status_t_HSA_STATUS_SUCCESS);

    // Address of the allocated buffer
    bs.kern_arg_buffer = kern_arg_buf;

    // Addr. of kern arg start.
    // bs.kern_arg_address = align_up(kern_arg_buf, req_align);
    bs.kern_arg_address = bs.kern_arg_buffer;

    println!("arg_size: {}, kernarg_size: {}", arg_size, bs.kernarg_size);
    // assert!(arg_size >= bs.kernarg_size as usize);

    let n_1 = bs.kern_arg_address as uintptr_t + arg_size;
    let n_2 = bs.kern_arg_buffer as uintptr_t + buf_size;
    assert!(n_1 < n_2);

    memcpy(bs.kern_arg_address, args, arg_size);
    assert_eq!(err, hsa_status_t_HSA_STATUS_SUCCESS);

    // Make sure both the CPU and GPU can access the kernel arguments
    let ag_list = [bs.gpu_dev, bs.cpu_dev];
    let err = hsa_amd_agents_allow_access(
        2,
        ag_list.as_ptr(),
        std::ptr::null_mut(),
        bs.kern_arg_buffer,
    );
    assert_eq!(err, hsa_status_t_HSA_STATUS_SUCCESS);

    // Save this info in our BinarySearch structure for later.
    *aql_buf_ptr = bs.kern_arg_address;

    hsa_status_t_HSA_STATUS_SUCCESS
}

/*
 * Write everything in the provided AQL packet to the queue except the first 32
 * bits which include the header and setup fields. That should be done
 * last.
 */
unsafe fn write_aqlto_queue(in_aql: &hsa_kernel_dispatch_packet_t, q: *mut hsa_queue_t) {
    let queue = &mut *(q as *mut hsa_queue_t);

    let queue_base = queue.base_address;

    let queue_mask = queue.size - 1;
    let que_idx = hsa_queue_add_write_index_relaxed(q, 1);

    let n = que_idx & queue_mask as u64;

    // let base = queue_base as *mut hsa_kernel_dispatch_packet_t;

    println!("que_idx: {:?}", que_idx);
    println!("que_idx & queue_mask : {:?}", n);

    // let slice = std::slice::from_raw_parts(base, queue.size as usize);
    // println!("slice {:?}", slice);

    // let queue_aql_packet = queue_base.add(n as usize) as *mut hsa_kernel_dispatch_packet_t;
    let queue_aql_packet = (queue_base as *mut hsa_kernel_dispatch_packet_t).add(n as usize);

    // println!("aql 2: {:#?}", queue_aql_packet);
    let packet = &mut *(queue_aql_packet as *mut hsa_kernel_dispatch_packet_t);

    packet.workgroup_size_x = in_aql.workgroup_size_x;
    packet.workgroup_size_y = in_aql.workgroup_size_y;
    packet.workgroup_size_z = in_aql.workgroup_size_z;
    packet.grid_size_x = in_aql.grid_size_x;
    packet.grid_size_y = in_aql.grid_size_y;
    packet.grid_size_z = in_aql.grid_size_z;
    packet.private_segment_size = in_aql.private_segment_size;
    packet.group_segment_size = in_aql.group_segment_size;
    packet.kernel_object = in_aql.kernel_object;
    packet.kernarg_address = in_aql.kernarg_address;
    packet.completion_signal = in_aql.completion_signal;
}

// TODO update atomic store
// This wrapper atomically writes the provided header and setup to the
// provided AQL packet. The provided AQL packet address should be in the
// queue memory space.
unsafe fn atomic_set_packet_header(
    header: u16,
    setup: u16,
    queue_packet: *mut hsa_kernel_dispatch_packet_t,
) {
    let n = queue_packet as *mut u32;
    let v = header as u32 | ((setup as u32) << 16);

    *n = v;
    // __atomic_store_n(reinterpret_cast<uint32_t*>(queue_packet), header | (setup << 16), __ATOMIC_RELEASE);
}

// Once all the required data for kernel execution is collected (in this
// application it is stored in the BinarySearch structure) we can put it in
// an AQL packet and ring the queue door bell to tell the command processor to
// execute it.
unsafe fn run(bs: &mut BinarySearch) -> hsa_status_t {
    println!("Executing kernel {}", bs.kernel_name);

    // Adjust the size of workgroup
    // This is mostly application specific.
    bs.work_group_size = 32;
    bs.work_grid_size = 32;

    // Setup the kernel args
    #[repr(C)]
    #[derive(Debug)]
    struct LocalArgsT {
        input: *mut std::os::raw::c_void,
        output: *mut std::os::raw::c_void,
    }

    let mut local_args = LocalArgsT {
        input: bs.input,
        output: bs.output,
    };

    // Copy the kernel args structure into kernel arg memory
    let ret = alloc_and_set_kern_args(
        bs,
        &mut local_args as *mut _ as *mut std::os::raw::c_void,
        std::mem::size_of_val(&local_args),
    );
    assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

    // Populate an AQL packet with the info we've gathered
    let aql = populate_aqlpacket(bs);

    // Copy kernel parameter from system memory to local memory
    // let err = AgentMemcpy(reinterpret_cast<uint8_t*>(bs->input_arr_local),
    // reinterpret_cast<uint8_t*>(bs->input_arr),
    // in_length, bs->gpu_dev, bs->cpu_dev);
    //
    // RET_IF_HSA_ERR(err);

    // Dispatch kernel with global work size, work group size with ONE dimesion
    // and wait for kernel to complete

    // Compute the write index of queue and copy Aql packet into it
    let que_idx = hsa_queue_load_write_index_relaxed(bs.queue);

    let queue = &mut *(bs.queue as *mut hsa_queue_t);

    let mask = queue.size - 1;

    // This function simply copies the data we've collected so far into our
    // local AQL packet, except the the setup and header fields.
    write_aqlto_queue(&aql, bs.queue);

    let mut aql_header = hsa_packet_type_t_HSA_PACKET_TYPE_KERNEL_DISPATCH;
    aql_header |= hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM
        << hsa_packet_header_t_HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
    aql_header |= hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM
        << hsa_packet_header_t_HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;

    // Set the packet's type, acquire and release fences. This should be done
    // atomically after all the other fields have been set, using release
    // memory ordering to ensure all the fields are set when the door bell
    // signal is activated.
    let q_base = queue.base_address;

    let n = que_idx as u32 & mask;
    // let queue_packet = q_base.add(n as usize) as *mut hsa_kernel_dispatch_packet_t;
    let queue_packet = (q_base as *mut hsa_kernel_dispatch_packet_t).add(n as usize);

    atomic_set_packet_header(aql_header as u16, aql.setup, queue_packet);

    // Increment the write index and ring the doorbell to dispatch kernel.
    hsa_queue_store_write_index_relaxed(bs.queue, que_idx + 1);
    hsa_signal_store_relaxed(queue.doorbell_signal, que_idx as hsa_signal_value_t);

    // Wait on the dispatch signal until the kernel is finished.
    // Modify the wait condition to HSA_WAIT_STATE_ACTIVE (instead of
    // HSA_WAIT_STATE_BLOCKED) if polling is needed instead of blocking, as we
    // have below.
    // The call below will block until the condition is met. Below we have said
    // the condition is that the signal value (initiailzed to 1) associated with
    // the queue is less than 1. When the kernel associated with the queued AQL
    // packet has completed execution, the signal value is automatically
    // decremented by the packet processor.
    let value = hsa_signal_wait_scacquire(
        bs.signal,
        hsa_signal_condition_t_HSA_SIGNAL_CONDITION_LT,
        1,
        u64::MAX,
        hsa_wait_state_t_HSA_WAIT_STATE_BLOCKED,
    );

    // value should be 0, or we timed-out
    if value != 0 {
        panic!(
            "Timed out waiting for kernel to complete?, hsa_signal_wait_scacquire {}",
            value
        );
    }

    // Reset the signal to its initial value for the next iteration
    hsa_signal_store_screlease(bs.signal, 1);

    hsa_status_t_HSA_STATUS_SUCCESS
}

// Release all the RocR resources we have acquired in this application.
unsafe fn clean_up(bs: &BinarySearch) -> hsa_status_t {
    let err = hsa_amd_memory_pool_free(bs.input);
    assert_eq!(err, hsa_status_t_HSA_STATUS_SUCCESS);

    let err = hsa_amd_memory_pool_free(bs.output);
    assert_eq!(err, hsa_status_t_HSA_STATUS_SUCCESS);

    let err = hsa_amd_memory_pool_free(bs.kern_arg_buffer);
    assert_eq!(err, hsa_status_t_HSA_STATUS_SUCCESS);

    let err = hsa_queue_destroy(bs.queue);
    assert_eq!(err, hsa_status_t_HSA_STATUS_SUCCESS);

    let err = hsa_signal_destroy(bs.signal);
    assert_eq!(err, hsa_status_t_HSA_STATUS_SUCCESS);

    let err = hsa_shut_down();
    assert_eq!(err, hsa_status_t_HSA_STATUS_SUCCESS);

    hsa_status_t_HSA_STATUS_SUCCESS
}

unsafe extern "C" fn print_queue_error(
    status: hsa_status_t,
    _source: *mut hsa_queue_t,
    _data: *mut ::std::os::raw::c_void,
) {
    println!("queue status: {:?}", status);
}

#[test]
fn binary_search() {
    // Set some working values specific to this application
    let mut bs = initialize_binary_search();

    unsafe {
        // hsa_init() initializes internal data structures and causes devices
        // (agents), memory pools and other resources to be discovered.
        let ret = hsa_init();
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        // Find the agents needed for the sample
        let ret = find_devices(&mut bs);
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        // Create the completion signal used when dispatching a packet
        let ret = hsa_signal_create(1, 0, std::ptr::null_mut(), &mut bs.signal);
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        // Create a queue to submit our binary search AQL packets
        let ret = hsa_queue_create(
            bs.gpu_dev,
            128,
            hsa_queue_type_t_HSA_QUEUE_TYPE_MULTI,
            Some(print_queue_error),
            std::ptr::null_mut(),
            u32::MAX,
            u32::MAX,
            &mut bs.queue,
        );
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        // Find the HSA memory pools we need to run this sample
        let ret = find_pools(&mut bs);
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        // Allocate memory from the correct memory pool, and initialize them as
        // neeeded for the algorihm.
        let ret = allocate_and_init_buffers(&mut bs);
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        // Create a kernel object from the pre-compiled kernel, and read some
        // attributes associated with the kernel that we will need.
        let ret = load_kernel_from_obj_file(&mut bs);
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        // Fill in the AQL packet, assign the kernel arguments, enqueue the packet,
        // "ring" the doorbell, and wait for completion.
        let ret = run(&mut bs);
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        let slice = std::slice::from_raw_parts(bs.input as *mut i32, 32);
        println!("input {:?}", slice);

        let slice = std::slice::from_raw_parts(bs.output as *mut i32, 32);
        println!("output {:?}", slice);

        thread::sleep(std::time::Duration::from_secs(0));

        let ret = hsa_amd_memory_fill(bs.input, 6, 32);
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        let ret = run(&mut bs);
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        let slice = std::slice::from_raw_parts(bs.input as *mut i32, 32);
        println!("input {:?}", slice);

        let slice = std::slice::from_raw_parts(bs.output as *mut i32, 32);
        println!("output {:?}", slice);

        // Release all the RocR resources we've acquired and shutdown HSA.
        let ret = clean_up(&bs);
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);
    }

    // println!("{:#?}", bs);
}
