use hsa_sys::bindings::{
    hsa_agent_get_info, hsa_agent_info_t_HSA_AGENT_INFO_DEVICE,
    hsa_agent_info_t_HSA_AGENT_INFO_NAME, hsa_agent_s, hsa_agent_t,
    hsa_amd_agent_iterate_memory_pools, hsa_amd_agent_memory_pool_get_info,
    hsa_amd_agent_memory_pool_info_t_HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS,
    hsa_amd_agents_allow_access, hsa_amd_memory_async_copy, hsa_amd_memory_fill,
    hsa_amd_memory_pool_access_t,
    hsa_amd_memory_pool_access_t_HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED,
    hsa_amd_memory_pool_allocate, hsa_amd_memory_pool_free, hsa_amd_memory_pool_get_info,
    hsa_amd_memory_pool_info_t_HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
    hsa_amd_memory_pool_info_t_HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
    hsa_amd_memory_pool_info_t_HSA_AMD_MEMORY_POOL_INFO_SEGMENT, hsa_amd_memory_pool_s,
    hsa_amd_memory_pool_t, hsa_amd_segment_t, hsa_amd_segment_t_HSA_AMD_SEGMENT_GLOBAL,
    hsa_device_type_t, hsa_device_type_t_HSA_DEVICE_TYPE_CPU,
    hsa_device_type_t_HSA_DEVICE_TYPE_GPU, hsa_init, hsa_iterate_agents, hsa_shut_down,
    hsa_signal_condition_t_HSA_SIGNAL_CONDITION_LT, hsa_signal_create, hsa_signal_store_screlease,
    hsa_signal_t, hsa_signal_wait_relaxed, hsa_status_t, hsa_status_t_HSA_STATUS_INFO_BREAK,
    hsa_status_t_HSA_STATUS_SUCCESS, hsa_wait_state_t_HSA_WAIT_STATE_BLOCKED,
};
use std::ffi::CString;
use utilities::helper_functions::buf_u8_remove_zero_to_string;

const K_TEST_FILL_VALUE1: u32 = 0xabcdef12;
const K_TEST_FILL_VALUE2: u32 = 0xba5eba11;
const K_TEST_FILL_VALUE3: u32 = 0xfeed5a1e;
const K_TEST_INIT_VALUE: u32 = 0xbaadf00d;

// This structure holds an agent pointer and associated memory pool to be used
// for this test program.

#[derive(Debug, Clone)]
struct AsyncMemCpyAgent {
    dev: hsa_agent_t,
    pool: hsa_amd_memory_pool_t,
    granule: usize,
    ptr: *mut std::os::raw::c_void,
}

struct AsyncMemCpyPoolQuery<'a> {
    pool_info: &'a mut AsyncMemCpyAgent,
    peer_device: hsa_agent_t,
}

#[derive(Debug)]
struct CallbackArgs {
    cpu: AsyncMemCpyAgent,
    gpu1: AsyncMemCpyAgent,
    gpu2: AsyncMemCpyAgent,
}

// Find the least common multiple of 2 numbers
fn lcm(a: u32, b: u32) -> u32 {
    let mut tmp_a = a as i32;
    let mut tmp_b = b as i32;

    while tmp_a != tmp_b {
        if tmp_a < tmp_b {
            tmp_a += a as i32;
        } else {
            tmp_b += b as i32;
        }
    }

    tmp_a as u32
}

// This function is a callback for hsa_amd_agent_iterate_memory_pools()
// and will test whether the provided memory pool is 1) in the GLOBAL
// segment, 2) allows allocation and 3) is accessible by the provided
// agent. The "data" input parameter is assumed to be pointing to a
// struct async_mem_cpy_agent. If the provided pool meets these criteria,
// HSA_STATUS_INFO_BREAK is returned.
unsafe extern "C" fn find_pool(
    in_pool: hsa_amd_memory_pool_t,
    data: *mut std::os::raw::c_void,
) -> hsa_status_t {
    let mut segment: hsa_amd_segment_t = 0;

    let args = &mut *(data as *mut AsyncMemCpyPoolQuery);

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

    if args.peer_device.handle != 0 {
        let mut access: hsa_amd_memory_pool_access_t =
            hsa_amd_memory_pool_access_t_HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;

        let err = hsa_amd_agent_memory_pool_get_info(
            args.peer_device,
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
    }

    let err = hsa_amd_memory_pool_get_info(
        in_pool,
        hsa_amd_memory_pool_info_t_HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
        &mut args.pool_info.granule as *mut _ as *mut std::os::raw::c_void,
    );
    if err != hsa_status_t_HSA_STATUS_SUCCESS {
        return err;
    }

    args.pool_info.pool = in_pool;

    hsa_status_t_HSA_STATUS_INFO_BREAK
}

// This function is meant to be a callback to hsa_iterate_agents. For each
// input agent the iterator provides as input, this function will check to
// see if the input agent is a CPU agent. If so, it will update the
// async_mem_cpy_agent structure pointed to by the input parameter "data".

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
        let args = &mut *(data as *mut AsyncMemCpyAgent);

        args.dev = agent;

        let mut pool_query = AsyncMemCpyPoolQuery {
            pool_info: args,
            peer_device: hsa_agent_s { handle: 0 },
        };

        // pool_query.peer_device.handle = 0;

        let err = hsa_amd_agent_iterate_memory_pools(
            agent,
            Some(find_pool),
            &mut pool_query as *mut _ as *mut std::os::raw::c_void,
        );

        return if err == hsa_status_t_HSA_STATUS_INFO_BREAK {
            // we found what we were looking for

            // args.pool = pool_query.pool_info.pool;
            // args.granule = pool_query.pool_info.granule;

            hsa_status_t_HSA_STATUS_INFO_BREAK
        } else {
            args.dev = hsa_agent_s { handle: 0 };
            err
        };
    }

    // Returning HSA_STATUS_SUCCESS tells the calling iterator to keep iterating
    hsa_status_t_HSA_STATUS_SUCCESS
}

// This function is meant to be a callback to hsa_iterate_agents. It will
// attempt to find 2, or at least 1 GPU agent suitable for our test. The data
// input parameter should point to a callback_args struct. The 2 GPU fields
// will be updated as GPUs are discovered.
// Return values:
//  HSA_STATUS_INFO_BREAK -- 2 GPU agents have been found and stored. Iterator
//    should stop iterating
//  HSA_STATUS_SUCCESS -- 2 GPU agents have not yet been found; 0 or 1 may
//    have been found; iterator function should keep iterating
//  Other -- Some error occurred
unsafe extern "C" fn find_gpus(
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

    if hsa_device_type != hsa_device_type_t_HSA_DEVICE_TYPE_GPU {
        return hsa_status_t_HSA_STATUS_SUCCESS;
    }

    let args = &mut *(data as *mut CallbackArgs);

    let mut pool_query = AsyncMemCpyPoolQuery {
        pool_info: &mut AsyncMemCpyAgent {
            dev: hsa_agent_s { handle: 0 },
            pool: hsa_amd_memory_pool_s { handle: 0 },
            granule: 0,
            ptr: std::ptr::null_mut(),
        },
        peer_device: hsa_agent_s { handle: 0 },
    };

    let gpu = if args.gpu1.dev.handle == 0 {
        &mut args.gpu1
    } else {
        // Check that gpu1 has peer access into the selected pool.
        pool_query.peer_device = args.gpu1.dev;
        &mut args.gpu2
    };

    // Make sure GPU device has pool host can access
    gpu.dev = agent;
    pool_query.pool_info = gpu;

    let err = hsa_amd_agent_iterate_memory_pools(
        agent,
        Some(find_pool),
        &mut pool_query as *mut _ as *mut std::os::raw::c_void,
    );

    if err == hsa_status_t_HSA_STATUS_INFO_BREAK {
        return if gpu.dev.handle == args.gpu2.dev.handle {
            // We found 2 gpu's
            hsa_status_t_HSA_STATUS_INFO_BREAK
        } else {
            // Keep looking for another gpu
            hsa_status_t_HSA_STATUS_SUCCESS
        };
    } else {
        gpu.dev = hsa_agent_s { handle: 0 };
    }

    if err != hsa_status_t_HSA_STATUS_SUCCESS {
        return err;
    }

    // Returning HSA_STATUS_SUCCESS tells the calling iterator to keep iterating
    hsa_status_t_HSA_STATUS_SUCCESS
}

// This is the main test, showing various paths of async. copy. Source and
// destination agents and their respective pools should already be discovered.
// Additionally, buffer from the pools should already be allocated and availble
// from the input parameters.
unsafe fn async_cpy_test(
    dst: &AsyncMemCpyAgent,
    src: &AsyncMemCpyAgent,
    args: &CallbackArgs,
    sz: usize,
    val: u32,
) -> hsa_status_t {
    let mut copy_signal = hsa_signal_t { handle: 0 };

    // Initialize the system and destination buffers with a value so we can later validate it has
    // been overwritten
    let sys_ptr = args.cpu.ptr;

    let size_u32 = std::mem::size_of::<u32>();

    let ret = hsa_amd_memory_fill(sys_ptr, K_TEST_INIT_VALUE, sz / size_u32);
    assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

    if dst.ptr != sys_ptr {
        let ret = hsa_amd_memory_fill(dst.ptr, K_TEST_INIT_VALUE, sz / size_u32);
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);
    }

    // Fill the source buffer with the provided uint32_t value
    let ret = hsa_amd_memory_fill(src.ptr, val, sz / size_u32);
    assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

    // Make sure the target and destination agents have access to the buffer.
    let ag_list = [dst.dev, src.dev];
    let ret = hsa_amd_agents_allow_access(2, ag_list.as_ptr(), std::ptr::null_mut(), dst.ptr);
    assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

    // Create a signal that will be used to inform us when the copy is done
    let ret = hsa_signal_create(1, 0, std::ptr::null_mut(), &mut copy_signal);
    assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

    // Do the copy...
    let ret = hsa_amd_memory_async_copy(
        dst.ptr,
        dst.dev,
        src.ptr,
        src.dev,
        sz,
        0,
        std::ptr::null_mut(),
        copy_signal,
    );
    assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

    // Here we do a blocking wait. Alternatively, we could also use a
    // non-blocking wait in a loop, and do other work while waiting.
    let r = hsa_signal_wait_relaxed(
        copy_signal,
        hsa_signal_condition_t_HSA_SIGNAL_CONDITION_LT,
        1,
        u64::MAX,
        hsa_wait_state_t_HSA_WAIT_STATE_BLOCKED,
    );

    if r != 0 {
        // println!("Async copy returned error value 1 -> {}", r);
        // return hsa_status_t_HSA_STATUS_ERROR;
        panic!("Async copy returned error value 1 -> {}", r);
    }

    // Verify the copy was successful; copy from the dst buffer to the sysBuf,
    // (if the result is not already in sys. mem.) and check the sysBuf values
    if dst.ptr != sys_ptr {
        if src.ptr != sys_ptr {
            // In this case, we need to give the gpu dev that owns dst->ptr access
            // to the system memory we are going to copy to.
            let ag_list_ck = [dst.dev, args.cpu.dev];
            let ret =
                hsa_amd_agents_allow_access(2, ag_list_ck.as_ptr(), std::ptr::null_mut(), sys_ptr);
            assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);
        }

        // Reset signal to 1
        hsa_signal_store_screlease(copy_signal, 1);

        let ret = hsa_amd_memory_async_copy(
            sys_ptr,
            args.cpu.dev,
            dst.ptr,
            dst.dev,
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

        if r != 0 {
            // println!("Async copy returned error value 2");
            // return hsa_status_t_HSA_STATUS_ERROR;
            panic!("Async copy returned error value 2 -> {}", r);
        }
    }

    let slice = std::slice::from_raw_parts(sys_ptr as *mut u32, sz / size_u32);
    println!("slice {:?}", slice);

    // Check that the contents of the buffer are what is expected.
    for p in slice {
        // println!("index {:?}: , value: {:?}", i, p);
        assert_eq!(*p, val);
    }

    hsa_status_t_HSA_STATUS_SUCCESS
}

#[test]
fn test_async_mem_copy() {
    let mut args = CallbackArgs {
        cpu: AsyncMemCpyAgent {
            dev: hsa_agent_s { handle: 0 },
            pool: hsa_amd_memory_pool_s { handle: 0 },
            granule: 0,
            ptr: std::ptr::null_mut(),
        },
        gpu1: AsyncMemCpyAgent {
            dev: hsa_agent_s { handle: 0 },
            pool: hsa_amd_memory_pool_s { handle: 0 },
            granule: 0,
            ptr: std::ptr::null_mut(),
        },
        gpu2: AsyncMemCpyAgent {
            dev: hsa_agent_s { handle: 0 },
            pool: hsa_amd_memory_pool_s { handle: 0 },
            granule: 0,
            ptr: std::ptr::null_mut(),
        },
    };

    let mut two_gpus = false;

    unsafe {
        let ret = hsa_init();
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        // First, find the cpu agent and associated pool
        let ret = hsa_iterate_agents(
            Some(find_cpu_device),
            &mut args.cpu as *mut _ as *mut std::os::raw::c_void,
        );
        assert_eq!(ret, hsa_status_t_HSA_STATUS_INFO_BREAK);

        // Now, find 1 or 2 (if possible) GPUs and associated pool(s) for our test
        let ret = hsa_iterate_agents(
            Some(find_gpus),
            &mut args as *mut _ as *mut std::os::raw::c_void,
        );

        if ret == hsa_status_t_HSA_STATUS_INFO_BREAK {
            two_gpus = true;
        } else {
            // See if we at least have 1 GPU
            if args.gpu1.dev.handle == 0 {
                panic!("GPU with accessible VRAM not found; at least 1 required. Exiting");
            }
            println!("Only 1 GPU found with required VRAM. Peer-to-Peer copy will be skipped");
        }

        // We will use the smallest amount of allocatable memory that works for all
        // potential sources and destinations of the copy
        let mut sz = lcm(args.cpu.granule as u32, args.gpu1.granule as u32) as usize;

        // Allocate memory on each source/destination
        if two_gpus {
            sz = lcm(sz as u32, args.gpu2.granule as u32) as usize;

            let ret = hsa_amd_memory_pool_allocate(args.gpu2.pool, sz, 0, &mut args.gpu2.ptr);
            assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);
        }

        let ret = hsa_amd_memory_pool_allocate(args.cpu.pool, sz, 0, &mut args.cpu.ptr);
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        let ret = hsa_amd_memory_pool_allocate(args.gpu1.pool, sz, 0, &mut args.gpu1.ptr);
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        let name = CString::new(vec![32; 63]).unwrap();
        let ret = hsa_agent_get_info(
            args.cpu.dev,
            hsa_agent_info_t_HSA_AGENT_INFO_NAME,
            name.as_ptr() as *mut std::os::raw::c_void,
        );
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        println!(
            "CPU is {}",
            buf_u8_remove_zero_to_string(name.as_bytes()).unwrap()
        );

        let ret = hsa_agent_get_info(
            args.gpu1.dev,
            hsa_agent_info_t_HSA_AGENT_INFO_NAME,
            name.as_ptr() as *mut std::os::raw::c_void,
        );
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        println!(
            "GPU1 is {}",
            buf_u8_remove_zero_to_string(name.as_bytes()).unwrap()
        );

        if two_gpus {
            let ret = hsa_agent_get_info(
                args.gpu2.dev,
                hsa_agent_info_t_HSA_AGENT_INFO_NAME,
                name.as_ptr() as *mut std::os::raw::c_void,
            );
            assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

            println!(
                "GPU2 is {:?}",
                buf_u8_remove_zero_to_string(name.as_bytes()).unwrap()
            );
        }

        println!("K_TEST_INIT_VALUE {}", K_TEST_INIT_VALUE);

        println!("Copying {} bytes from gpu1 memory to system memory...", sz);
        println!("K_TEST_FILL_VALUE1 {}", K_TEST_FILL_VALUE1);

        let ret = async_cpy_test(&args.cpu, &args.gpu1, &args, sz, K_TEST_FILL_VALUE1);
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        println!("Success!");

        println!("Copying {} bytes from system memory to gpu1 memory...", sz);
        println!("K_TEST_FILL_VALUE2 {}", K_TEST_FILL_VALUE2);

        let ret = async_cpy_test(&args.gpu1, &args.cpu, &args, sz, K_TEST_FILL_VALUE2);
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        println!("Success!");

        if two_gpus {
            println!(
                "Copying {} bytes from from gpu1 memory to gpu2 memory...",
                sz
            );
            println!("K_TEST_FILL_VALUE3 {}", K_TEST_FILL_VALUE3);

            let ret = async_cpy_test(&args.gpu2, &args.gpu1, &args, sz, K_TEST_FILL_VALUE3);
            assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

            println!("Success!");
        }

        // Clean up
        let ret = hsa_amd_memory_pool_free(args.cpu.ptr);
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        let ret = hsa_amd_memory_pool_free(args.gpu1.ptr);
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        if two_gpus {
            let ret = hsa_amd_memory_pool_free(args.gpu2.ptr);
            assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);
        }

        let ret = hsa_shut_down();
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);
    }

    println!("two_gpus {}", two_gpus);
    println!("{:#?}", args);
}
