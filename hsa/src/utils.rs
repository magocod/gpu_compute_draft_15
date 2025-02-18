use hsa_sys::bindings::{hsa_amd_ipc_memory_t, hsa_amd_ipc_signal_t};

#[repr(C)]
#[derive(Debug)]
pub struct SharedMemory {
    pub mem_handle: hsa_amd_ipc_memory_t,
    pub signal_handle: hsa_amd_ipc_signal_t,
}
