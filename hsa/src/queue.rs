use crate::error::{hsa_check, HsaResult};
use crate::system::Agent;
use hsa_sys::bindings::{
    hsa_kernel_dispatch_packet_t, hsa_queue_add_write_index_relaxed, hsa_queue_create,
    hsa_queue_destroy, hsa_queue_store_write_index_relaxed, hsa_queue_t,
    hsa_queue_type_t_HSA_QUEUE_TYPE_MULTI, hsa_signal_condition_t_HSA_SIGNAL_CONDITION_LT,
    hsa_signal_store_relaxed, hsa_signal_store_screlease, hsa_signal_t, hsa_signal_value_t,
    hsa_signal_wait_scacquire, hsa_status_t, hsa_status_t_HSA_STATUS_SUCCESS,
    hsa_wait_state_t_HSA_WAIT_STATE_BLOCKED,
};

// This wrapper atomically writes the provided header and setup to the
// provided AQL packet. The provided AQL packet address should be in the
// queue memory space.

/// ...
///
///
/// # Safety
///
/// TODO safety function explain
pub unsafe fn atomic_set_packet_header(
    header: u16,
    setup: u16,
    queue_packet: *mut hsa_kernel_dispatch_packet_t,
) {
    let n = queue_packet as *mut u32;
    // println!("queue_packet: {:?}", queue_packet);

    let v = header as u32 | ((setup as u32) << 16);
    // println!("atomic_set_packet_header v: {}", v);

    *n = v;

    // __atomic_store_n(reinterpret_cast<uint32_t*>(queue_packet), header | (setup << 16), __ATOMIC_RELEASE);
}

unsafe extern "C" fn handle_queue_error(
    status: hsa_status_t,
    _source: *mut hsa_queue_t,
    _data: *mut std::os::raw::c_void,
) {
    println!("queue status: {:?}", status);
}

#[derive(Debug)]
pub struct WriteAqlPacket {
    pub packet_ptr: *mut hsa_kernel_dispatch_packet_t,
    pub que_idx: u64,
    pub queue_mask: u32,
    pub packet_id: u64,
}

#[derive(Debug, PartialEq)]
pub struct Queue {
    hsa_queue_ptr: *mut hsa_queue_t,
}

impl Queue {
    pub fn new(agent: &Agent) -> HsaResult<Self> {
        let mut hsa_queue_ptr: *mut hsa_queue_t = std::ptr::null_mut();

        unsafe {
            let ret = hsa_queue_create(
                agent.get_hsa_agent_t(),
                128, // TODO check agent queue size
                hsa_queue_type_t_HSA_QUEUE_TYPE_MULTI,
                Some(handle_queue_error),
                std::ptr::null_mut(),
                u32::MAX,
                u32::MAX,
                &mut hsa_queue_ptr,
            );
            hsa_check(ret)?;
        }

        Ok(Queue { hsa_queue_ptr })
    }

    /*
     * Write everything in the provided AQL packet to the queue except the first 32
     * bits which include the header and setup fields. That should be done
     * last.
     */

    /// ...
    ///
    ///
    /// # Safety
    ///
    /// TODO safety function explain
    pub unsafe fn write_aql_packet(
        &self,
        aql_packet: hsa_kernel_dispatch_packet_t,
    ) -> WriteAqlPacket {
        let queue = &mut *(self.hsa_queue_ptr as *mut hsa_queue_t);

        let queue_base = queue.base_address;

        let queue_mask = queue.size - 1;
        let que_idx = hsa_queue_add_write_index_relaxed(self.hsa_queue_ptr, 1);

        let packet_id = que_idx & queue_mask as u64;

        // println!("que_idx: {:?}", que_idx);
        // println!("queue_mask: {:?}", queue_mask);
        // println!("packet_id: {:?}", packet_id);

        let queue_aql_packet =
            (queue_base as *mut hsa_kernel_dispatch_packet_t).add(packet_id as usize);
        let packet = &mut *(queue_aql_packet as *mut hsa_kernel_dispatch_packet_t);

        packet.workgroup_size_x = aql_packet.workgroup_size_x;
        packet.workgroup_size_y = aql_packet.workgroup_size_y;
        packet.workgroup_size_z = aql_packet.workgroup_size_z;
        packet.grid_size_x = aql_packet.grid_size_x;
        packet.grid_size_y = aql_packet.grid_size_y;
        packet.grid_size_z = aql_packet.grid_size_z;
        packet.private_segment_size = aql_packet.private_segment_size;
        packet.group_segment_size = aql_packet.group_segment_size;
        packet.kernel_object = aql_packet.kernel_object;
        packet.kernarg_address = aql_packet.kernarg_address;
        packet.completion_signal = aql_packet.completion_signal;

        WriteAqlPacket {
            packet_ptr: queue_aql_packet,
            que_idx,
            queue_mask,
            packet_id,
        }
    }

    /// ...
    ///
    ///
    /// # Safety
    ///
    /// TODO safety function explain
    pub unsafe fn dispatch(&self, packet: WriteAqlPacket, signal: hsa_signal_t) {
        let queue = &mut *(self.hsa_queue_ptr as *mut hsa_queue_t);

        // Increment the write index and ring the doorbell to dispatch kernel.
        hsa_queue_store_write_index_relaxed(self.hsa_queue_ptr, packet.que_idx + 1);
        hsa_signal_store_relaxed(queue.doorbell_signal, packet.que_idx as hsa_signal_value_t);

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
            signal,
            hsa_signal_condition_t_HSA_SIGNAL_CONDITION_LT,
            1,
            u64::MAX,
            hsa_wait_state_t_HSA_WAIT_STATE_BLOCKED,
        );

        println!("hsa_signal_wait_scacquire value: {}", value);

        // value should be 0, or we timed-out
        if value != 0 {
            panic!("Timed out waiting for kernel to complete?");
        }

        // Reset the signal to its initial value for the next iteration
        hsa_signal_store_screlease(signal, 1);
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        unsafe {
            let ret = hsa_queue_destroy(self.hsa_queue_ptr);
            if ret != hsa_status_t_HSA_STATUS_SUCCESS {
                panic!("hsa_queue_destroy error {}", ret);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::System;

    #[test]
    fn test_queue_create() {
        let system = System::new().unwrap();
        let gpu_agent = system.get_first_gpu().unwrap();

        let _queue = Queue::new(gpu_agent);

        // TODO assert test
    }

    // TODO tests
}
