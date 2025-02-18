use hsa_sys::bindings::{
    hsa_agent_get_info, hsa_agent_info_t_HSA_AGENT_INFO_DEVICE,
    hsa_agent_info_t_HSA_AGENT_INFO_NAME, hsa_agent_s, hsa_agent_t,
    hsa_amd_agent_info_s_HSA_AMD_AGENT_INFO_BDFID, hsa_amd_agent_iterate_memory_pools,
    hsa_amd_agent_memory_pool_get_info,
    hsa_amd_agent_memory_pool_info_t_HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS,
    hsa_amd_agents_allow_access, hsa_amd_ipc_memory_attach, hsa_amd_ipc_memory_create,
    hsa_amd_ipc_memory_detach, hsa_amd_ipc_memory_t, hsa_amd_ipc_signal_attach,
    hsa_amd_ipc_signal_create, hsa_amd_ipc_signal_t, hsa_amd_memory_async_copy,
    hsa_amd_memory_fill, hsa_amd_memory_pool_access_t,
    hsa_amd_memory_pool_access_t_HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED,
    hsa_amd_memory_pool_allocate, hsa_amd_memory_pool_free, hsa_amd_memory_pool_get_info,
    hsa_amd_memory_pool_info_t_HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
    hsa_amd_memory_pool_info_t_HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
    hsa_amd_memory_pool_info_t_HSA_AMD_MEMORY_POOL_INFO_SEGMENT, hsa_amd_memory_pool_t,
    hsa_amd_segment_t, hsa_amd_segment_t_HSA_AMD_SEGMENT_GLOBAL,
    hsa_amd_signal_attribute_t_HSA_AMD_SIGNAL_IPC, hsa_amd_signal_create, hsa_device_type_t,
    hsa_device_type_t_HSA_DEVICE_TYPE_CPU, hsa_device_type_t_HSA_DEVICE_TYPE_GPU, hsa_init,
    hsa_iterate_agents, hsa_shut_down, hsa_signal_condition_t_HSA_SIGNAL_CONDITION_LT,
    hsa_signal_condition_t_HSA_SIGNAL_CONDITION_NE, hsa_signal_create, hsa_signal_destroy,
    hsa_signal_store_relaxed, hsa_signal_store_release, hsa_signal_t, hsa_signal_wait_acquire,
    hsa_signal_wait_relaxed, hsa_status_t, hsa_status_t_HSA_STATUS_ERROR,
    hsa_status_t_HSA_STATUS_INFO_BREAK, hsa_status_t_HSA_STATUS_SUCCESS,
    hsa_wait_state_t_HSA_WAIT_STATE_BLOCKED,
};
use libc::{
    fork, mmap, munmap, sched_yield, waitpid, MAP_ANONYMOUS, MAP_FAILED, MAP_SHARED, PROT_READ,
    PROT_WRITE,
};
use std::ffi::CString;
use utilities::helper_functions::buf_u8_remove_zero_to_string;

#[derive(Debug)]
struct CallbackArgs {
    host: hsa_agent_t,
    device: hsa_agent_t,
    cpu_pool: hsa_amd_memory_pool_t,
    gpu_pool: hsa_amd_memory_pool_t,
    gpu_mem_granule: usize,
}

// This function will test whether the provided memory pool is 1) in the
// GLOBAL segment, 2) allows allocation and 3) is accessible by the provided
// agent. If the provided pool meets these criteria, HSA_STATUS_INFO_BREAK is
// returned
unsafe fn find_pool(in_pool: hsa_amd_memory_pool_t, agent: hsa_agent_t) -> hsa_status_t {
    let mut segment: hsa_amd_segment_t = 0;

    let err = hsa_amd_memory_pool_get_info(
        in_pool,
        hsa_amd_memory_pool_info_t_HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
        &mut segment as *mut _ as *mut std::os::raw::c_void,
    );
    if err != hsa_status_t_HSA_STATUS_SUCCESS {
        return err;
    }

    if segment != hsa_amd_segment_t_HSA_AMD_SEGMENT_GLOBAL {
        return hsa_status_t_HSA_STATUS_SUCCESS;
    }

    let mut can_alloc: bool = false;

    let err = hsa_amd_memory_pool_get_info(
        in_pool,
        hsa_amd_memory_pool_info_t_HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
        &mut can_alloc as *mut _ as *mut std::os::raw::c_void,
    );
    if err != hsa_status_t_HSA_STATUS_SUCCESS {
        return err;
    }

    if !can_alloc {
        return hsa_status_t_HSA_STATUS_SUCCESS;
    }

    let mut access: hsa_amd_memory_pool_access_t =
        hsa_amd_memory_pool_access_t_HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;

    let err = hsa_amd_agent_memory_pool_get_info(
        agent,
        in_pool,
        hsa_amd_agent_memory_pool_info_t_HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS,
        &mut access as *mut _ as *mut std::os::raw::c_void,
    );
    if err != hsa_status_t_HSA_STATUS_SUCCESS {
        return err;
    }

    if access == hsa_amd_memory_pool_access_t_HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED {
        return hsa_status_t_HSA_STATUS_SUCCESS;
    }

    hsa_status_t_HSA_STATUS_INFO_BREAK
}

// Callback function for hsa_amd_agent_iterate_memory_pools(). If the provided
// pool is suitable (see comments for FindPool()), HSA_STATUS_INFO_BREAK is
// returned. The input parameter "data" should point to memory for a "struct
// callback_args", which includes a gpu pool and a granule field.  These fields
// will be filled in by this function if the provided pool meets all the
// requirements.
unsafe extern "C" fn find_device_pool(
    pool: hsa_amd_memory_pool_t,
    data: *mut std::os::raw::c_void,
) -> hsa_status_t {
    let args = &mut *(data as *mut CallbackArgs);

    let err = find_pool(pool, args.device);

    if err == hsa_status_t_HSA_STATUS_INFO_BREAK {
        args.gpu_pool = pool;

        let err = hsa_amd_memory_pool_get_info(
            args.gpu_pool,
            hsa_amd_memory_pool_info_t_HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
            &mut args.gpu_mem_granule as *mut _ as *mut std::os::raw::c_void,
        );

        if err != hsa_status_t_HSA_STATUS_SUCCESS {
            return err;
        }

        // We found what we were looking for, so return HSA_STATUS_INFO_BREAK
        return hsa_status_t_HSA_STATUS_INFO_BREAK;
    }

    hsa_status_t_HSA_STATUS_SUCCESS
}

// Callback function for hsa_amd_agent_iterate_memory_pools(). If the provided
// pool is suitable (see comments for FindPool()), HSA_STATUS_INFO_BREAK is
// returned. The input parameter "data" should point to memory for a "struct
// callback_args", which includes a cpu pool. This field will be filled in by
// this function if the provided pool meets all the requirements.
unsafe extern "C" fn find_cpu_pool(
    pool: hsa_amd_memory_pool_t,
    data: *mut std::os::raw::c_void,
) -> hsa_status_t {
    let args = &mut *(data as *mut CallbackArgs);

    let err = find_pool(pool, args.host);

    if err == hsa_status_t_HSA_STATUS_INFO_BREAK {
        args.cpu_pool = pool;
    }

    err
}

// This function is meant to be a call-back to hsa_iterate_agents. Find the
// first GPU agent that has memory accessible by CPU
// Return values:
//  HSA_STATUS_INFO_BREAK -- 2 GPU agents have been found and stored. Iterator
//    should stop iterating
//  HSA_STATUS_SUCCESS -- 2 GPU agents have not yet been found; iterator
//    should keep iterating
//  Other -- Some error occurred
unsafe extern "C" fn find_gpu(agent: hsa_agent_t, data: *mut std::os::raw::c_void) -> hsa_status_t {
    let mut hsa_device_type: hsa_device_type_t = 0;

    let err = hsa_agent_get_info(
        agent,
        hsa_agent_info_t_HSA_AGENT_INFO_DEVICE,
        &mut hsa_device_type as *mut _ as *mut std::os::raw::c_void,
    );
    if err != hsa_status_t_HSA_STATUS_SUCCESS {
        return err;
    }

    if hsa_device_type != hsa_device_type_t_HSA_DEVICE_TYPE_GPU {
        return hsa_status_t_HSA_STATUS_SUCCESS;
    }

    let args = &mut *(data as *mut CallbackArgs);

    // Make sure GPU device has pool host can access
    args.device = agent;
    let err = hsa_amd_agent_iterate_memory_pools(agent, Some(find_device_pool), data);

    if err == hsa_status_t_HSA_STATUS_INFO_BREAK {
        // We were looking for, so return HSA_STATUS_INFO_BREAK
        return hsa_status_t_HSA_STATUS_INFO_BREAK;
    } else {
        args.device = hsa_agent_t { handle: 0 };
    }

    if err != hsa_status_t_HSA_STATUS_SUCCESS {
        return err;
    }

    // Returning HSA_STATUS_SUCCESS tells the calling iterator to keep iterating
    hsa_status_t_HSA_STATUS_SUCCESS
}

// This function is meant to be a call-back to hsa_iterate_agents. For each
// input agent the iterator provides as input, this function will check to
// see if the input agent is a CPU. If so, it will update the callback_args
// structure pointed to by the input parameter "data".

// Return values:
//  HSA_STATUS_INFO_BREAK -- CPU agent has been found and stored. Iterator
//    should stop iterating
//  HSA_STATUS_SUCCESS -- CPU agent has not yet been found; iterator
//    should keep iterating
//  Other -- Some error occurred
unsafe extern "C" fn find_cpu_device(
    agent: hsa_agent_t,
    data: *mut std::os::raw::c_void,
) -> hsa_status_t {
    let mut hsa_device_type: hsa_device_type_t = 0;

    let err = hsa_agent_get_info(
        agent,
        hsa_agent_info_t_HSA_AGENT_INFO_DEVICE,
        &mut hsa_device_type as *mut _ as *mut std::os::raw::c_void,
    );
    if err != hsa_status_t_HSA_STATUS_SUCCESS {
        return err;
    }

    if hsa_device_type == hsa_device_type_t_HSA_DEVICE_TYPE_CPU {
        let args = &mut *(data as *mut CallbackArgs);

        args.host = agent;

        let err = hsa_amd_agent_iterate_memory_pools(agent, Some(find_cpu_pool), data);

        return if err == hsa_status_t_HSA_STATUS_INFO_BREAK {
            // we found what we were looking for
            hsa_status_t_HSA_STATUS_INFO_BREAK
        } else {
            args.host = hsa_agent_s { handle: 0 };
            err
        };
    }

    // Returning HSA_STATUS_SUCCESS tells the calling iterator to keep iterating
    hsa_status_t_HSA_STATUS_SUCCESS
}

// This function will test whether the gpu-local buffer has been filled
// with an expected value and return an error if not. The expected value is
// also replaced with a new value.
// Implementation notes: We create a buffer in system memory and copy
// the gpu-local data buffer to be tested to this system memory buffer.
// We also write the system memory buffer with the new value, and then copy
// it back the gpu-local buffer.
unsafe fn check_and_fill_buffer(
    args: &mut CallbackArgs,
    gpu_src_ptr: *mut std::os::raw::c_void,
    exp_cur_val: u32,
    new_val: u32,
    print_text: &str,
) -> hsa_status_t {
    let mut copy_signal = hsa_signal_t { handle: 0 };
    let sz = args.gpu_mem_granule;
    let cpu_ag = args.host;
    let gpu_ag = args.device;

    let ret = hsa_signal_create(1, 0, std::ptr::null_mut(), &mut copy_signal);
    assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

    let mut sys_buf: *mut std::os::raw::c_void = std::ptr::null_mut();

    let ret = hsa_amd_memory_pool_allocate(args.cpu_pool, sz, 0, &mut sys_buf);
    assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

    let ag_list = [args.device, args.host];
    let ret = hsa_amd_agents_allow_access(2, ag_list.as_ptr(), std::ptr::null_mut(), sys_buf);
    assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

    let ret = hsa_amd_memory_async_copy(
        sys_buf,
        cpu_ag,
        gpu_src_ptr,
        gpu_ag,
        sz,
        0,
        std::ptr::null_mut(),
        copy_signal,
    );
    assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

    let r = hsa_signal_wait_relaxed(
        copy_signal,
        hsa_signal_condition_t_HSA_SIGNAL_CONDITION_LT,
        1,
        u64::MAX,
        hsa_wait_state_t_HSA_WAIT_STATE_BLOCKED,
    );

    println!(
        "{}: check_and_fill_buffer - hsa_signal_wait_relaxed {}",
        print_text, r
    );

    if r != 0 {
        println!("{}: check_and_fill_buffer - hsa_signal_wait_relaxed - Async copy returned error value - {}", print_text, r);
        return hsa_status_t_HSA_STATUS_ERROR;
    }

    let count = sz / std::mem::size_of::<u32>();

    // let slice = std::slice::from_raw_parts(sys_buf as *mut u32, sz / count);
    // println!("{}: slice {:?}", print_text, slice);
    //
    // for v in slice.iter() {
    //     if *v != exp_cur_val {
    //         panic!("Expected {} but got {} in buffer.", exp_cur_val, v)
    //     }
    //     // v = new_val;
    // }

    // let ret = hsa_amd_memory_fill(sys_buf, new_val, count);
    // hsa_check(ret).unwrap();

    let slice = std::slice::from_raw_parts_mut(sys_buf as *mut u32, sz / count);
    println!("{}: slice {:?}", print_text, slice);

    for v in slice.iter_mut() {
        if *v != exp_cur_val {
            panic!("Expected {} but got {} in buffer.", exp_cur_val, v)
        }
        *v = new_val;
    }

    hsa_signal_store_relaxed(copy_signal, 1);

    let ret = hsa_amd_memory_async_copy(
        gpu_src_ptr,
        gpu_ag,
        sys_buf,
        cpu_ag,
        sz,
        0,
        std::ptr::null_mut(),
        copy_signal,
    );
    assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

    let r = hsa_signal_wait_relaxed(
        copy_signal,
        hsa_signal_condition_t_HSA_SIGNAL_CONDITION_LT,
        1,
        u64::MAX,
        hsa_wait_state_t_HSA_WAIT_STATE_BLOCKED,
    );
    println!(
        "{}: check_and_fill_buffer - hsa_signal_wait_relaxed (2) {}",
        print_text, r
    );

    if r != 0 {
        println!("{}: check_and_fill_buffer - hsa_signal_wait_relaxed (2) - Async copy returned error value - {}", print_text, r);
        return hsa_status_t_HSA_STATUS_ERROR;
    }

    let ret = hsa_signal_destroy(copy_signal);
    assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

    let ret = hsa_amd_memory_pool_free(sys_buf);
    assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

    hsa_status_t_HSA_STATUS_SUCCESS
}

// See if the other process wrote an error value to the token; if not, write
// the newVal to the token.
fn check_and_set_token(token: &mut i32, new_val: i32) {
    if *token == -1 {
        panic!("Error in other process.");
    } else {
        *token = new_val;
    }
}

// Summary of this IPC Sample:
// This program demonstrates the IPC apis. Run it by executing 2 instances
// of the program.
// The first process will allocate some gpu-local memory and fill it with
// 1's. This HSA buffer will be made shareable with hsa_amd_ipc_memory_create()
// The 2nd process will access this shared buffer with
// hsa_amd_ipc_memory_attach(), verify that 1's were written, and then fill
// the buffer with 2's. Finally, the first process will then read the
// gpu-local buffer and verify that the 2's were indeed written. The main
// point is to show how hsa memory buffer handles can be shared among
// processes.
//
// Implementation Notes:
// -Standard linux shared memory is used in this sample program as a way
// of sharing info and  synchronizing the 2 processes. This is independent
// of RocR IPC and should not be confused with it.
fn main() {
    // IPC test
    #[derive(Debug)]
    struct Shared {
        token: i32,
        count: i32,
        size: usize,
        handle: hsa_amd_ipc_memory_t,
        signal_handle: hsa_amd_ipc_signal_t,
    }
    let shared_size = std::mem::size_of::<Shared>();

    unsafe {
        // Allocate linux shared memory.
        let shared = mmap(
            std::ptr::null_mut(),
            shared_size,
            PROT_READ | PROT_WRITE,
            MAP_SHARED | MAP_ANONYMOUS,
            -1,
            0,
        );

        if shared == MAP_FAILED {
            panic!("Unable to allocate shared memory. Exiting.");
        }

        let shared_values = &mut *(shared as *mut Shared);
        // println!("Shared: {:?}", shared_values);

        // "token" is used to signal state changes between the 2 processes.
        shared_values.token = 0;

        let mut process_one: bool = false;

        // Spawn second process and verify communication
        let child = fork();

        if child == -1 {
            panic!("Fork failed");
        }

        if child != 0 {
            process_one = true;

            // Signal to other process we are waiting, and then wait...
            shared_values.token = 1;

            while shared_values.token == 1 {
                sched_yield();
            }

            println!("Second process observed, handshake...");

            shared_values.token = 1;

            while shared_values.token == 1 {
                sched_yield();
            }
        } else {
            process_one = false;

            println!("Second process running.");

            while shared_values.token == 0 {
                sched_yield();
            }

            check_and_set_token(&mut shared_values.token, 0);

            // Wait for handshake
            while shared_values.token == 0 {
                sched_yield();
            }

            check_and_set_token(&mut shared_values.token, 0);
            println!("Handshake complete.");
        }

        let ret = hsa_init();
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        let mut args = CallbackArgs {
            host: hsa_agent_t { handle: 0 },
            device: hsa_agent_t { handle: 0 },
            cpu_pool: hsa_amd_memory_pool_t { handle: 0 },
            gpu_pool: hsa_amd_memory_pool_t { handle: 0 },
            gpu_mem_granule: 0,
        };

        let ret = hsa_iterate_agents(
            Some(find_cpu_device),
            &mut args as *mut _ as *mut std::os::raw::c_void,
        );
        assert_eq!(ret, hsa_status_t_HSA_STATUS_INFO_BREAK);

        let ret = hsa_iterate_agents(
            Some(find_gpu),
            &mut args as *mut _ as *mut std::os::raw::c_void,
        );
        assert_eq!(ret, hsa_status_t_HSA_STATUS_INFO_BREAK);

        // Print out name of the device.
        let name_1 = CString::new(vec![32; 63]).unwrap();
        let name_2 = CString::new(vec![32; 63]).unwrap();

        let ret = hsa_agent_get_info(
            args.host,
            hsa_agent_info_t_HSA_AGENT_INFO_NAME,
            name_1.as_ptr() as *mut std::os::raw::c_void,
        );
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        let ret = hsa_agent_get_info(
            args.device,
            hsa_agent_info_t_HSA_AGENT_INFO_NAME,
            name_2.as_ptr() as *mut std::os::raw::c_void,
        );
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        let mut loc_1: u32 = 0;
        let mut loc_2: u32 = 0;

        let ret = hsa_agent_get_info(
            args.host,
            hsa_amd_agent_info_s_HSA_AMD_AGENT_INFO_BDFID,
            &mut loc_1 as *mut _ as *mut std::os::raw::c_void,
        );
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        let ret = hsa_agent_get_info(
            args.device,
            hsa_amd_agent_info_s_HSA_AMD_AGENT_INFO_BDFID,
            &mut loc_2 as *mut _ as *mut std::os::raw::c_void,
        );
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        println!(
            "Using: {} ({}) and {} ({})\n",
            buf_u8_remove_zero_to_string(name_1.as_bytes()).unwrap(),
            loc_1,
            buf_u8_remove_zero_to_string(name_2.as_bytes()).unwrap(),
            loc_2
        );

        // Get signal for async copy
        let mut copy_signal = hsa_signal_t { handle: 0 };
        let ret = hsa_signal_create(1, 0, std::ptr::null_mut(), &mut copy_signal);
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        let ag_list = [args.device, args.host];

        if process_one {
            // Allocate some VRAM and fill it with 1's
            let mut gpu_buf: *mut std::os::raw::c_void = std::ptr::null_mut();

            let ret =
                hsa_amd_memory_pool_allocate(args.gpu_pool, args.gpu_mem_granule, 0, &mut gpu_buf);
            assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

            println!("process #1: Allocated local memory buffer at {:?}", gpu_buf);

            let ret =
                hsa_amd_agents_allow_access(2, ag_list.as_ptr(), std::ptr::null_mut(), gpu_buf);
            assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

            let ret =
                hsa_amd_ipc_memory_create(gpu_buf, args.gpu_mem_granule, &mut shared_values.handle);
            println!("process #1: Created IPC handle associated with gpu-local buffer at P0 address {:?}", gpu_buf);

            assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

            let count = args.gpu_mem_granule / std::mem::size_of::<u32>();
            shared_values.size = args.gpu_mem_granule;
            shared_values.count = count as i32;

            let ret = hsa_amd_memory_fill(gpu_buf, 1, count);
            assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

            // Get IPC capable signal
            let mut ipc_signal = hsa_signal_t { handle: 0 };
            let ret = hsa_amd_signal_create(
                1,
                0,
                std::ptr::null_mut(),
                hsa_amd_signal_attribute_t_HSA_AMD_SIGNAL_IPC as u64,
                &mut ipc_signal,
            );
            assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

            println!("process #1: Created IPC handle associated with ipc_signal");

            let ret = hsa_amd_ipc_signal_create(ipc_signal, &mut shared_values.signal_handle);
            assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

            // Signal Process 2 that the gpu buffer is ready to read.
            check_and_set_token(&mut shared_values.token, 1);

            println!("process #1: Allocated buffer and filled it with 1's. Wait for P1...");

            let r = hsa_signal_wait_acquire(
                ipc_signal,
                hsa_signal_condition_t_HSA_SIGNAL_CONDITION_NE,
                1,
                u64::MAX,
                hsa_wait_state_t_HSA_WAIT_STATE_BLOCKED,
            );

            println!("process #1: hsa_signal_wait_acquire {}", r);

            if r != 2 {
                hsa_signal_store_release(ipc_signal, -1);
                return ();
            }

            let ret = check_and_fill_buffer(&mut args, gpu_buf, 2, 0, "process #1");
            assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

            println!("process #1: Confirmed P1 filled buffer with 2");
            println!("process #1: PASSED on P0");

            hsa_signal_store_relaxed(ipc_signal, 0);

            let ret = hsa_signal_destroy(ipc_signal);
            assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

            let ret = hsa_amd_memory_pool_free(gpu_buf);
            assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

            waitpid(child, std::ptr::null_mut(), 0);

            println!("process #1: {:?}", args);
        } else {
            // "ProcessTwo"
            println!("process #2: Waiting for process 0 to write 1 to token...");

            while shared_values.token == 0 {
                sched_yield();
            }

            if shared_values.token != 1 {
                shared_values.token = -1;
                return ();
            }

            // Attach shared VRAM
            let mut ptr: *mut std::os::raw::c_void = std::ptr::null_mut();

            let ret = hsa_amd_ipc_memory_attach(
                &mut shared_values.handle,
                shared_values.size,
                1,
                ag_list.as_ptr(),
                &mut ptr,
            );
            assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

            println!(
                "process #2: Attached to IPC handle; P1 buffer address gpu-local memory is {:?}",
                ptr
            );

            // Attach shared signal
            let mut ipc_signal = hsa_signal_t { handle: 0 };
            let ret = hsa_amd_ipc_signal_attach(&shared_values.signal_handle, &mut ipc_signal);
            assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

            println!("process #2: Attached to signal IPC handle");

            let ret = check_and_fill_buffer(&mut args, ptr, 1, 2, "process #2");
            assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

            println!("process #2: Confirmed P0 filled buffer with 1; P1 re-filled buffer with 2");
            println!("process #2: PASSED on P1");

            hsa_signal_store_release(ipc_signal, 2);

            let ret = hsa_amd_ipc_memory_detach(ptr);
            assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

            hsa_signal_wait_relaxed(
                ipc_signal,
                hsa_signal_condition_t_HSA_SIGNAL_CONDITION_NE,
                2,
                u64::MAX,
                hsa_wait_state_t_HSA_WAIT_STATE_BLOCKED,
            );

            let ret = hsa_signal_destroy(ipc_signal);
            assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

            println!("process #2: {:?}", args);
        }

        let ret = hsa_signal_destroy(copy_signal);
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        // println!("{:#?}", shared_values);

        munmap(shared, shared_size);

        let ret = hsa_shut_down();
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);
    }
}
