use opencl::error::OclResult;
use opencl::opencl_sys::bindings::cl_int;
use opencl::wrapper::system::{OpenclCommonOperation, System};

pub const DEFAULT_DEVICE_INDEX: usize = 0;

// 1 GPU

pub struct SingleGpuExample {
    pub system: System,
}

impl SingleGpuExample {
    pub fn new(program_src: &str) -> Self {
        let system = System::new(DEFAULT_DEVICE_INDEX, program_src).unwrap();

        Self { system }
    }

    pub fn get_tmp_arr(&self, tmp_arr_len: usize) -> OclResult<Vec<cl_int>> {
        let global_work_size = tmp_arr_len;
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let output_buf = self.system.create_output_buffer(global_work_size)?;

        let mut kernel = self.system.create_kernel("get_tmp_arr")?;

        unsafe {
            kernel.set_arg(&output_buf)?;

            kernel.enqueue_nd_range_kernel(
                self.system.get_host_command_queue(),
                &[],
                &[global_work_size],
                &[local_work_size],
                &[],
            )?;
        };

        let output =
            self.system
                .blocking_enqueue_read_buffer(global_work_size, &output_buf, &[])?;

        println!("tmp_arr len {} - {:?}", output.len(), output);

        Ok(output)
    }
}

// 2 GPU

// ...
