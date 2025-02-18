use hsa::error::hsa_check;
use hsa::memory::{memory_copy_async, HsaBuffer};
use hsa::signal::Signal;
use hsa::system::System;
use hsa::utils::SharedMemory;
use hsa_sys::bindings::{
    hsa_amd_ipc_memory_create, hsa_amd_ipc_signal_create, hsa_amd_memory_fill,
    hsa_amd_signal_attribute_t_HSA_AMD_SIGNAL_IPC, hsa_signal_condition_t_HSA_SIGNAL_CONDITION_NE,
    hsa_signal_wait_scacquire, hsa_wait_state_t_HSA_WAIT_STATE_BLOCKED,
};
use libc::{shmat, shmget, IPC_CREAT, IPC_EXCL, S_IRGRP, S_IRUSR, S_IWGRP, S_IWUSR};

const WORK_GROUP_SIZE_X: usize = 32;

#[allow(dead_code)]
#[derive(Debug)]
struct HsaModule<'a> {
    system: &'a System,
    output_buf: HsaBuffer<i32>,
    ipc_mem_buf: HsaBuffer<i32>,
    ipc_signal: Signal,
}

impl<'a> HsaModule<'a> {
    pub fn new(system: &'a System) -> Self {
        let cpu_agent = system.get_first_cpu().unwrap();
        let gpu_agent = system.get_first_gpu().unwrap();

        let cpu_mem_pool = cpu_agent.get_standard_pool().unwrap();
        let gpu_mem_pool = gpu_agent.get_standard_pool().unwrap();

        let output_buf: HsaBuffer<i32> =
            HsaBuffer::new(WORK_GROUP_SIZE_X, cpu_mem_pool, &[cpu_agent, gpu_agent]).unwrap();

        let ipc_mem_buf: HsaBuffer<i32> =
            HsaBuffer::new(WORK_GROUP_SIZE_X, gpu_mem_pool, &[gpu_agent, cpu_agent]).unwrap();

        let shared_size = std::mem::size_of::<SharedMemory>();

        let ipc_signal =
            Signal::new(1, hsa_amd_signal_attribute_t_HSA_AMD_SIGNAL_IPC as u64).unwrap();

        unsafe {
            // Allocate linux shared memory.
            let shm_id = shmget(
                0,
                shared_size,
                IPC_CREAT
                    | IPC_EXCL
                    | S_IRUSR as i32
                    | S_IWUSR as i32
                    | S_IRGRP as i32
                    | S_IWGRP as i32,
            );

            println!("shm_id {}", shm_id);

            // shmat to attach to shared memory
            let shme = shmat(shm_id, std::ptr::null(), 0) as *mut SharedMemory;

            let shared_values = &mut *(shme);

            let ret = hsa_amd_memory_fill(ipc_mem_buf.get_mem_ptr(), 3, WORK_GROUP_SIZE_X);
            hsa_check(ret).unwrap();

            let ret = hsa_amd_ipc_memory_create(
                ipc_mem_buf.get_mem_ptr(),
                ipc_mem_buf.get_size_bytes(),
                &mut shared_values.mem_handle,
            );
            hsa_check(ret).unwrap();

            let ret = hsa_amd_ipc_signal_create(
                ipc_signal.get_hsa_signal_t(),
                &mut shared_values.signal_handle,
            );
            hsa_check(ret).unwrap();
        }

        Self {
            system,
            output_buf,
            ipc_mem_buf,
            ipc_signal,
        }
    }

    pub fn print_output(&self) {
        let cpu_agent = self.system.get_first_cpu().unwrap();
        let gpu_agent = self.system.get_first_gpu().unwrap();

        let ipc_mem_size = WORK_GROUP_SIZE_X * std::mem::size_of::<i32>();

        let slice = unsafe {
            memory_copy_async(
                self.output_buf.get_mem_ptr(),
                cpu_agent.get_hsa_agent_t(),
                self.ipc_mem_buf.get_mem_ptr(),
                gpu_agent.get_hsa_agent_t(),
                ipc_mem_size,
            )
            .expect("memory_copy_async error");

            std::slice::from_raw_parts(self.output_buf.get_mem_ptr() as *mut i32, WORK_GROUP_SIZE_X)
        };
        println!("ipc output {:?}", slice);
    }

    pub fn wait_for_signal(&self) {
        println!("waiting for signal");

        unsafe {
            let r = hsa_signal_wait_scacquire(
                self.ipc_signal.get_hsa_signal_t(),
                hsa_signal_condition_t_HSA_SIGNAL_CONDITION_NE,
                1,
                u64::MAX,
                hsa_wait_state_t_HSA_WAIT_STATE_BLOCKED,
            );

            println!("hsa_signal_wait_scacquire {}", r);
        }

        self.print_output();
    }
}

fn main() {
    let system = System::new().unwrap();

    let module_1 = HsaModule::new(&system);
    module_1.print_output();

    module_1.wait_for_signal();
    // loop {
    //
    // }
}
