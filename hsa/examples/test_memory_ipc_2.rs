use hsa::error::hsa_check;
use hsa::memory::{memory_copy_async, HsaBuffer};
use hsa::system::System;
use hsa::utils::SharedMemory;
use hsa_sys::bindings::{
    hsa_amd_ipc_memory_attach, hsa_amd_ipc_memory_detach, hsa_amd_ipc_signal_attach,
    hsa_signal_destroy, hsa_signal_store_screlease, hsa_signal_t, hsa_status_t_HSA_STATUS_SUCCESS,
};
use libc::shmat;

const WORK_GROUP_SIZE_X: usize = 32;

#[allow(dead_code)]
#[derive(Debug)]
struct HsaModule<'a> {
    system: &'a System,
    output_buf: HsaBuffer<i32>,
    ipc_mem_ptr: *mut std::os::raw::c_void,
    ipc_signal: hsa_signal_t,
}

impl<'a> HsaModule<'a> {
    pub fn new(system: &'a System, shm_id: i32) -> Self {
        let cpu_agent = system.get_first_cpu().unwrap();
        let gpu_agent = system.get_first_gpu().unwrap();

        let cpu_mem_pool = cpu_agent.get_standard_pool().unwrap();
        // let gpu_mem_pool = gpu_agent.get_standard_pool().unwrap();

        let output_buf: HsaBuffer<i32> =
            HsaBuffer::new(WORK_GROUP_SIZE_X, cpu_mem_pool, &[cpu_agent, gpu_agent]).unwrap();

        let mut ipc_mem_ptr: *mut std::os::raw::c_void = std::ptr::null_mut();
        let ipc_mem_size = WORK_GROUP_SIZE_X * std::mem::size_of::<i32>();

        let mut ipc_signal = hsa_signal_t { handle: 0 };

        unsafe {
            // shmat to attach to shared memory
            let shme = shmat(shm_id, std::ptr::null_mut(), 0) as *mut SharedMemory;

            let shared_values = &mut *(shme);
            println!("shared_values: {:?}", shared_values);

            let ag_list = &[gpu_agent.get_hsa_agent_t()];

            let ret = hsa_amd_ipc_memory_attach(
                &mut shared_values.mem_handle,
                ipc_mem_size,
                ag_list.len() as u32,
                ag_list.as_ptr(),
                &mut ipc_mem_ptr,
            );
            hsa_check(ret).unwrap();

            // Attach shared signal
            let ret = hsa_amd_ipc_signal_attach(&shared_values.signal_handle, &mut ipc_signal);
            hsa_check(ret).unwrap();
        }

        Self {
            system,
            output_buf,
            ipc_mem_ptr,
            ipc_signal,
        }
    }

    pub fn write_in_ipc_memory(&self, input: &[i32]) {
        // allocate_and_init_buffers
        let cpu_agent = self.system.get_first_cpu().unwrap();
        let gpu_agent = self.system.get_first_gpu().unwrap();

        let src_ptr = input.as_ptr() as *mut std::os::raw::c_void;
        let input_size = input.len() * std::mem::size_of::<u32>();

        unsafe {
            memory_copy_async(
                self.output_buf.get_mem_ptr(),
                cpu_agent.get_hsa_agent_t(),
                src_ptr,
                cpu_agent.get_hsa_agent_t(),
                input_size,
            )
            .expect("memory_copy_async error");

            memory_copy_async(
                self.ipc_mem_ptr,
                gpu_agent.get_hsa_agent_t(),
                self.output_buf.get_mem_ptr(),
                cpu_agent.get_hsa_agent_t(),
                input_size,
            )
            .expect("memory_copy_async error");
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
                self.ipc_mem_ptr,
                gpu_agent.get_hsa_agent_t(),
                ipc_mem_size,
            )
            .expect("memory_copy_async error");

            std::slice::from_raw_parts(self.output_buf.get_mem_ptr() as *mut i32, WORK_GROUP_SIZE_X)
        };
        println!("ipc output {:?}", slice);
    }

    pub fn update_signal(&self) {
        unsafe {
            hsa_signal_store_screlease(self.ipc_signal, 2);
        }
    }
}

impl Drop for HsaModule<'_> {
    fn drop(&mut self) {
        unsafe {
            let ret = hsa_amd_ipc_memory_detach(self.ipc_mem_ptr);
            if ret != hsa_status_t_HSA_STATUS_SUCCESS {
                panic!("hsa_amd_ipc_memory_detach error: {:?}", ret);
            }

            let ret = hsa_signal_destroy(self.ipc_signal);
            if ret != hsa_status_t_HSA_STATUS_SUCCESS {
                panic!("hsa_signal_destroy error: {:?}", ret);
            }
        }
    }
}

fn main() {
    let shm_id = 524298;
    let system = System::new().unwrap();

    let module_1 = HsaModule::new(&system, shm_id);
    module_1.print_output();

    let input = vec![7; WORK_GROUP_SIZE_X];
    module_1.write_in_ipc_memory(&input);

    let module_2 = HsaModule::new(&system, shm_id);
    module_2.print_output();

    module_1.update_signal();
}
