use hsa_sys::bindings::{
    hsa_agent_feature_t_HSA_AGENT_FEATURE_KERNEL_DISPATCH, hsa_agent_get_info,
    hsa_agent_info_t_HSA_AGENT_INFO_FEATURE, hsa_agent_iterate_regions, hsa_agent_t,
    hsa_executable_symbol_info_t_HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
    hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM, hsa_init, hsa_iterate_agents,
    hsa_kernel_dispatch_packet_setup_t_HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS,
    hsa_kernel_dispatch_packet_t, hsa_memory_allocate,
    hsa_packet_header_t_HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE,
    hsa_packet_header_t_HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE,
    hsa_packet_header_t_HSA_PACKET_HEADER_TYPE, hsa_packet_type_t,
    hsa_packet_type_t_HSA_PACKET_TYPE_KERNEL_DISPATCH, hsa_queue_add_write_index_relaxed,
    hsa_queue_create, hsa_queue_destroy, hsa_queue_t, hsa_queue_type_t_HSA_QUEUE_TYPE_SINGLE,
    hsa_region_get_info, hsa_region_global_flag_t,
    hsa_region_global_flag_t_HSA_REGION_GLOBAL_FLAG_KERNARG,
    hsa_region_info_t_HSA_REGION_INFO_GLOBAL_FLAGS, hsa_region_info_t_HSA_REGION_INFO_SEGMENT,
    hsa_region_segment_t, hsa_region_segment_t_HSA_REGION_SEGMENT_GLOBAL, hsa_region_t,
    hsa_shut_down, hsa_signal_create, hsa_signal_destroy, hsa_signal_store_screlease, hsa_signal_t,
    hsa_status_t, hsa_status_t_HSA_STATUS_INFO_BREAK, hsa_status_t_HSA_STATUS_SUCCESS,
};

// hsa_status_t get_kernel_agent(hsa_agent_t agent, void* data) { uint32_t features = 0;
//     hsa_agent_get_info(agent, HSA_AGENT_INFO_FEATURE, &features);
//     if (features& HSA_AGENT_FEATURE_KERNEL_DISPATCH) {
//         // Store kernel agent in the application-provided buffer and return
//         hsa_agent_t*ret = (hsa_agent_t*) data;
//         *ret = agent;
//         return HSA_STATUS_INFO_BREAK;
//     }
//     // Keep iterating
//     return HSA_STATUS_SUCCESS;
// }

unsafe extern "C" fn get_kernel_agent(
    agent: hsa_agent_t,
    data: *mut std::os::raw::c_void,
) -> hsa_status_t {
    let mut features = 0;

    let err = hsa_agent_get_info(
        agent,
        hsa_agent_info_t_HSA_AGENT_INFO_FEATURE,
        &mut features as *mut _ as *mut std::os::raw::c_void,
    );
    if err != hsa_status_t_HSA_STATUS_SUCCESS {
        return err;
    }

    let b = features & hsa_agent_feature_t_HSA_AGENT_FEATURE_KERNEL_DISPATCH;

    if b == 1 {
        // Store kernel agent in the application-provided buffer and return
        let ret = data as *mut hsa_agent_t;
        *ret = agent;

        return hsa_status_t_HSA_STATUS_INFO_BREAK;
    }

    // Keep iterating
    hsa_status_t_HSA_STATUS_SUCCESS
}

// void initialize_packet(hsa_kernel_dispatch_packet_t* packet) {
//     // Reserved fields, private and group memory, and completion signal are all set to 0.
//     memset(((uint8_t*) packet) + 4, 0, sizeof(hsa_kernel_dispatch_packet_t) - 4);
//     packet->workgroup_size_x = 256;
//     packet->workgroup_size_y = 1;
//     packet->workgroup_size_z = 1;
//     packet->grid_size_x = 256;
//     packet->grid_size_y = 1;
//     packet->grid_size_z = 1;
//     // Indicate which executable code to run.
//     // The application is expected to have finalized a kernel (for example, using the finalization API).
//     // We will assume that the kernel object containing the executable code is stored in KERNEL_OBJECT
//     packet->kernel_object = KERNEL_OBJECT;
//     // Assume our kernel receives no arguments
//     packet->kernarg_address = NULL;
// }

// void packet_store_release(uint32_t* packet, uint16_t header, uint16_t rest) {
//     __atomic_store_n(packet, header | (rest << 16), __ATOMIC_RELEASE);
// }

// uint16_t header(hsa_packet_type_t type) {
//     uint16_t header = type << HSA_PACKET_HEADER_TYPE;
//     header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
//     header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
//     return header;
// }

// uint16_t kernel_dispatch_setup() {
//     return 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
// }

// void simple_dispatch() {
//     // Initialize the runtime
//     hsa_init();
//     // Retrieve the kernel agent
//     hsa_agent_t kernel_agent;
//     hsa_iterate_agents(get_kernel_agent, &kernel_agent);
//     // Create a queue in the kernel agent. The queue can hold 4 packets, and has no callback or service queue associated with it
//     hsa_queue_t *queue;
//     hsa_queue_create(kernel_agent, 4, HSA_QUEUE_TYPE_SINGLE, NULL, NULL, 0, 0, &queue);
//     // Since no packets have been enqueued yet, we use zero as the packet ID and bump the write index accordingly
//     hsa_queue_add_write_index_relaxed(queue, 1);
//     uint64_t packet_id =0;
//     // Calculate the virtual address where to place the packet
//     hsa_kernel_dispatch_packet_t* packet = (hsa_kernel_dispatch_packet_t*) queue->base_address + packet_id;
//     // Populate fields in kernel dispatch packet, except for the header, the setup, and the completion signal fields
//     initialize_packet(packet);
//     // Create a signal with an initial value of one to monitor the task completion
//     hsa_signal_create(1, 0, NULL, &packet->completion_signal);
//     // Notify the queue that the packet is ready to be processed
//     packet_store_release((uint32_t*) packet, header(HSA_PACKET_TYPE_KERNEL_DISPATCH), kernel_dispatch_setup());
//     hsa_signal_store_screlease(queue->doorbell_signal, packet_id);
//     // Wait for the task to finish, which is the same as waiting for the value of the completion signal to be zero
//     while (hsa_signal_wait_scacquire(packet->completion_signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_
//     STATE_ACTIVE) != 0);
//     // Done! The kernel has completed. Time to cleanup resources and leave
//     hsa_signal_destroy(packet->completion_signal);
//     hsa_queue_destroy(queue);
//     hsa_shut_down();
// }

fn initialize_packet(packet_ptr: *mut hsa_kernel_dispatch_packet_t) {
    // Reserved fields, private and group memory, and completion signal are all set to 0.
    let p = &mut unsafe { *(packet_ptr) };

    p.workgroup_size_x = 256;
    p.workgroup_size_y = 1;
    p.workgroup_size_z = 1;
    p.grid_size_x = 256;
    p.grid_size_y = 1;
    p.grid_size_z = 1;

    // Indicate which executable code to run.
    // The application is expected to have finalized a kernel (for example, using the finalization API).
    // We will assume that the kernel object containing the executable code is stored in KERNEL_OBJECT
    p.kernel_object = hsa_executable_symbol_info_t_HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT as u64;
    // Assume our kernel receives no arguments
    p.kernarg_address = std::ptr::null_mut();
}

// not atomic
fn packet_store_release(packet: *mut u32, header: u16, rest: u16) {
    let h = header as u32;
    let r = rest as u32;
    unsafe {
        *packet = h | (r << 16);
    }
}

fn header(packet_type: hsa_packet_type_t) -> u16 {
    let mut header = packet_type << hsa_packet_header_t_HSA_PACKET_HEADER_TYPE;

    header |= hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM
        << hsa_packet_header_t_HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
    header |= hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM
        << hsa_packet_header_t_HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;

    header as u16
}

fn kernel_dispatch_setup() -> u16 {
    1 << hsa_kernel_dispatch_packet_setup_t_HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS
}

unsafe extern "C" fn check_queue(
    status: hsa_status_t,
    _source: *mut hsa_queue_t,
    _data: *mut ::std::os::raw::c_void,
) {
    println!("check_queue: status: {:?}", status);
}

// HSA-Runtime-1.2.pdf
// 2.5.2 Example: a simple dispatch

#[test]
fn test_simple_dispatch() {
    unsafe {
        // Initialize the runtime
        let ret = hsa_init();
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        // Retrieve the kernel agent
        let mut kernel_agent = hsa_agent_t { handle: 0 };

        let ret = hsa_iterate_agents(
            Some(get_kernel_agent),
            &mut kernel_agent as *mut _ as *mut std::os::raw::c_void,
        );
        assert_eq!(ret, hsa_status_t_HSA_STATUS_INFO_BREAK);

        // Create a queue in the kernel agent. The queue can hold 4 packets,
        // and has no callback or service queue associated with it
        let mut queue: *mut hsa_queue_t = std::ptr::null_mut();

        let ret = hsa_queue_create(
            kernel_agent,
            64, // TODO check minimun size of agent
            hsa_queue_type_t_HSA_QUEUE_TYPE_SINGLE,
            Some(check_queue),
            std::ptr::null_mut(),
            // &mut count as *mut _ as *mut std::os::raw::c_void,
            0,
            0,
            &mut queue,
        );
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        let queue_st = &mut *(queue);

        // Since no packets have been enqueued yet, we use zero as the packet ID
        // and bump the write index accordingly
        let write_index = hsa_queue_add_write_index_relaxed(queue, 1);
        let packet_id = write_index;

        // Calculate the virtual address where to place the packet
        let packet_ptr =
            (queue_st.base_address as *mut hsa_kernel_dispatch_packet_t).add(packet_id as usize);

        // Populate fields in kernel dispatch packet, except for the header, the setup, and the completion signal fields
        initialize_packet(packet_ptr);
        let packet_st = &mut *(packet_ptr);

        // Create a signal with an initial value of one to monitor the task completion
        let ret = hsa_signal_create(1, 0, std::ptr::null_mut(), &mut packet_st.completion_signal);
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        // Notify the queue that the packet is ready to be processed
        packet_store_release(
            packet_ptr as *mut u32,
            header(hsa_packet_type_t_HSA_PACKET_TYPE_KERNEL_DISPATCH),
            kernel_dispatch_setup(),
        );

        hsa_signal_store_screlease(queue_st.doorbell_signal, packet_id as i64);

        // Wait for the task to finish, which is the same as waiting for the value of the completion signal to be zero
        // while
        //     hsa_signal_wait_scacquire(
        //         packet_st.completion_signal,
        //         hsa_signal_condition_t_HSA_SIGNAL_CONDITION_EQ,
        //         0,
        //         u64::MAX,
        //         hsa_wait_state_t_HSA_WAIT_STATE_ACTIVE
        //     ) == 0 {
        //     // pass
        // }

        // error
        // loop {
        //     let ret = hsa_signal_wait_scacquire(
        //         packet_st.completion_signal,
        //         hsa_signal_condition_t_HSA_SIGNAL_CONDITION_EQ,
        //         0,
        //         u64::MAX,
        //         hsa_wait_state_t_HSA_WAIT_STATE_ACTIVE,
        //     );
        //
        //     println!("hsa_signal_wait_scacquire: {:#?}", ret);
        //
        //     if ret == 0 {
        //         break;
        //     }
        // }

        // Done! The kernel has completed. Time to cleanup resources and leave
        let ret = hsa_signal_destroy(packet_st.completion_signal);
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        let ret = hsa_queue_destroy(queue);
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        let ret = hsa_shut_down();
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);
    }

    // hsa_shut_down
    // ...
}

// Place the signal the argument buffer
// hsa_signal_t* buffer = (hsa_signal_t*) packet->kernarg_address;
// assert(buffer != NULL);
// hsa_signal_t signal;
// hsa_signal_create(128, 1, &kernel_agent, &signal);
// *buffer = signal;

// hsa_status_t get_kernarg(hsa_region_t region, void* data) {
//     hsa_region_segment_t segment;
//     hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment);
//     if (segment != HSA_REGION_SEGMENT_GLOBAL) {
//     return HSA_STATUS_SUCCESS;
//     }
//     hsa_region_global_flag_t flags;
//     hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
//     if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG) {
//     hsa_region_t* ret = (hsa_region_t*) data;
//     *ret = region;
//     return HSA_STATUS_INFO_BREAK;
//     }
//     return HSA_STATUS_SUCCESS;
// }

unsafe extern "C" fn get_kernarg(
    region: hsa_region_t,
    data: *mut std::os::raw::c_void,
) -> hsa_status_t {
    let mut segment: hsa_region_segment_t = 0;

    hsa_region_get_info(
        region,
        hsa_region_info_t_HSA_REGION_INFO_SEGMENT,
        &mut segment as *mut _ as *mut std::os::raw::c_void,
    );

    if segment != hsa_region_segment_t_HSA_REGION_SEGMENT_GLOBAL {
        return hsa_status_t_HSA_STATUS_SUCCESS;
    }

    let mut flags: hsa_region_global_flag_t = 0;

    hsa_region_get_info(
        region,
        hsa_region_info_t_HSA_REGION_INFO_GLOBAL_FLAGS,
        &mut flags as *mut _ as *mut std::os::raw::c_void,
    );

    let f = flags & hsa_region_global_flag_t_HSA_REGION_GLOBAL_FLAG_KERNARG;
    if f == 1 {
        let ret = data as *mut hsa_region_t;
        *ret = region;
        return hsa_status_t_HSA_STATUS_INFO_BREAK;
    }

    hsa_status_t_HSA_STATUS_SUCCESS
}

// HSA-Runtime-1.2.pdf
// 2.7.1.1 Example: passing arguments to a kernel
#[allow(unused_assignments)]
#[test]
fn test_passing_arguments_to_a_kernel() {
    unsafe {
        // Initialize the runtime
        let ret = hsa_init();
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        // Retrieve the kernel agent
        let mut kernel_agent = hsa_agent_t { handle: 0 };

        let ret = hsa_iterate_agents(
            Some(get_kernel_agent),
            &mut kernel_agent as *mut _ as *mut std::os::raw::c_void,
        );
        assert_eq!(ret, hsa_status_t_HSA_STATUS_INFO_BREAK);

        let mut queue: *mut hsa_queue_t = std::ptr::null_mut();

        let ret = hsa_queue_create(
            kernel_agent,
            64,
            hsa_queue_type_t_HSA_QUEUE_TYPE_SINGLE,
            Some(check_queue),
            std::ptr::null_mut(),
            0,
            0,
            &mut queue,
        );
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        let queue_st = &mut *(queue);

        // Since no packets have been enqueued yet, we use zero as the packet ID and bump the write index accordingly
        let write_index = hsa_queue_add_write_index_relaxed(queue, 1);
        let packet_id = write_index;

        // Calculate the virtual address where to place the packet
        let packet_ptr =
            queue_st.base_address.add(packet_id as usize) as *mut hsa_kernel_dispatch_packet_t;

        // Populate fields in kernel dispatch packet, except for the header, the setup, and the completion signal fields
        initialize_packet(packet_ptr);
        let packet_st = &mut *(packet_ptr);

        // searches for a memory region that can be used to allocate backing storage for the kernarg
        let mut region = hsa_region_t { handle: 0 };

        let ret = hsa_agent_iterate_regions(
            kernel_agent,
            Some(get_kernarg),
            &mut region as *mut _ as *mut std::os::raw::c_void,
        );
        assert_eq!(ret, hsa_status_t_HSA_STATUS_INFO_BREAK);

        // Allocate a buffer where to place the kernel arguments.
        let ret = hsa_memory_allocate(
            region,
            std::mem::size_of::<hsa_signal_t>(),
            &mut packet_st.kernarg_address,
        );
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        // Place the signal the argument buffer
        let mut buffer: *mut hsa_signal_t = packet_st.kernarg_address as *mut hsa_signal_t;

        assert!(!buffer.is_null());

        // let mut signal: *mut hsa_signal_t = std::ptr::null_mut();
        let mut signal = hsa_signal_t { handle: 0 };

        // let k = &kernel_agent as *const _ as *const hsa_agent_t;

        let ret = hsa_signal_create(1, 1, &kernel_agent, &mut signal);
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        buffer = &mut signal;

        println!("packet: {:#?}", packet_st);

        let ret = hsa_shut_down();
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);
    }
}
