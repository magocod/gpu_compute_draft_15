use crate::error::{hsa_check, HsaResult};
use crate::system::{Agent, MemoryPool};
use hsa_sys::bindings::{
    hsa_agent_t, hsa_amd_agents_allow_access, hsa_amd_memory_async_copy,
    hsa_amd_memory_pool_allocate, hsa_amd_memory_pool_free,
    hsa_signal_condition_t_HSA_SIGNAL_CONDITION_LT, hsa_signal_create, hsa_signal_destroy,
    hsa_signal_t, hsa_signal_wait_relaxed, hsa_status_t_HSA_STATUS_SUCCESS,
    hsa_wait_state_t_HSA_WAIT_STATE_BLOCKED,
};

/// ...
///
///
/// # Safety
///
/// TODO safety function explain
pub unsafe fn memory_copy_async(
    dst: *mut ::std::os::raw::c_void,
    dst_agent: hsa_agent_t,
    src: *const ::std::os::raw::c_void,
    src_agent: hsa_agent_t,
    size: usize,
) -> HsaResult<()> {
    let mut copy_signal = hsa_signal_t { handle: 0 };

    // Create a signal that will be used to inform us when the copy is done
    let ret = hsa_signal_create(1, 0, std::ptr::null_mut(), &mut copy_signal);
    hsa_check(ret)?;

    // Do the copy...
    let ret = hsa_amd_memory_async_copy(
        dst,
        dst_agent,
        src,
        src_agent,
        size,
        0,
        std::ptr::null_mut(),
        copy_signal,
    );
    hsa_check(ret)?;

    // Here we do a blocking wait. Alternatively, we could also use a
    // non-blocking wait in a loop, and do other work while waiting.
    let r = hsa_signal_wait_relaxed(
        copy_signal,
        hsa_signal_condition_t_HSA_SIGNAL_CONDITION_LT,
        1,
        u64::MAX,
        hsa_wait_state_t_HSA_WAIT_STATE_BLOCKED,
    );

    println!("hsa_signal_wait_relaxed -> {}", r);

    if r != 0 {
        let ret = hsa_signal_destroy(copy_signal);
        hsa_check(ret)?;

        panic!("Async copy returned error value -> {}", r);
        // return Err(HsaError::Code(hsa_status_t_HSA_STATUS_ERROR))
    }

    let ret = hsa_signal_destroy(copy_signal);
    hsa_check(ret)?;

    Ok(())
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct HsaBuffer<T: Copy + Clone + Default> {
    size: usize,
    size_bytes: usize,
    mem_ptr: *mut std::os::raw::c_void,
    phantom: std::marker::PhantomData<T>,
}

impl<T: Copy + Clone + Default> HsaBuffer<T> {
    pub fn new(
        size: usize,
        mem_pool: &MemoryPool,
        allow_access_agents: &[&Agent],
    ) -> HsaResult<HsaBuffer<T>> {
        let size_bytes = std::mem::size_of::<T>() * size;

        let mut mem_ptr: *mut std::os::raw::c_void = std::ptr::null_mut();

        let ag_list: Vec<hsa_agent_t> = allow_access_agents
            .iter()
            .map(|x| x.get_hsa_agent_t())
            .collect();

        unsafe {
            let ret = hsa_amd_memory_pool_allocate(
                mem_pool.get_hsa_amd_memory_pool_t(),
                size_bytes,
                0,
                &mut mem_ptr,
            );
            hsa_check(ret)?;

            let ret = hsa_amd_agents_allow_access(
                ag_list.len() as u32,
                ag_list.as_ptr(),
                std::ptr::null_mut(),
                mem_ptr,
            );
            hsa_check(ret)?;
        }

        Ok(Self {
            size,
            size_bytes,
            mem_ptr,
            phantom: std::marker::PhantomData,
        })
    }

    pub fn get_mem_ptr(&self) -> *mut std::os::raw::c_void {
        self.mem_ptr
    }

    pub fn get_size_bytes(&self) -> usize {
        self.size_bytes
    }
}

impl<T: Copy + Clone + Default> Drop for HsaBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            let ret = hsa_amd_memory_pool_free(self.mem_ptr);
            if ret != hsa_status_t_HSA_STATUS_SUCCESS {
                panic!("hsa_amd_memory_pool_free error {}", ret);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::HsaError;
    use crate::system::System;
    use hsa_sys::bindings::{
        hsa_amd_ipc_memory_create, hsa_amd_ipc_memory_t,
        hsa_status_t_HSA_STATUS_ERROR_INVALID_ARGUMENT,
    };

    #[test]
    fn test_hsa_buffer_create() {
        let system = System::new().unwrap();

        let size = 32;

        let hsa_buffer: HsaBuffer<i32> =
            HsaBuffer::new(size, system.get_kern_arg_pool(), &system.get_agents()).unwrap();

        assert_eq!(hsa_buffer.size, 32);
        assert_eq!(hsa_buffer.size_bytes, std::mem::size_of::<i32>() * size);
    }

    #[test]
    fn test_memory_ipc_success() {
        let system = System::new().unwrap();

        let gpu_agent = system.get_first_gpu().unwrap();
        let gpu_mem_pool = gpu_agent.get_standard_pool().unwrap();

        let buffer: HsaBuffer<u32> = HsaBuffer::new(32, gpu_mem_pool, &[gpu_agent]).unwrap();

        let mut handle = hsa_amd_ipc_memory_t { handle: [0; 8] };

        unsafe {
            let ret = hsa_amd_ipc_memory_create(
                buffer.get_mem_ptr(),
                buffer.get_size_bytes(),
                &mut handle,
            );
            hsa_check(ret).unwrap();
        }
    }

    #[test]
    fn test_memory_ipc_error() {
        let system = System::new().unwrap();

        let cpu_agent = system.get_first_cpu().unwrap();
        let cpu_mem_pool = cpu_agent.get_standard_pool().unwrap();

        let buffer: HsaBuffer<u32> = HsaBuffer::new(32, cpu_mem_pool, &[cpu_agent]).unwrap();

        let mut handle = hsa_amd_ipc_memory_t { handle: [0; 8] };

        unsafe {
            let ret = hsa_amd_ipc_memory_create(
                buffer.get_mem_ptr(),
                buffer.get_size_bytes(),
                &mut handle,
            );
            let result = hsa_check(ret);
            assert_eq!(
                result,
                Err(HsaError::Code(
                    hsa_status_t_HSA_STATUS_ERROR_INVALID_ARGUMENT
                ))
            );
        }
    }
}
