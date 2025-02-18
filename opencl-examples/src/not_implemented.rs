// cl_khr_kernel_clock
// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_C.html#kernel-clock-functions

use crate::utils::SingleGpuExample;
use opencl::error::OclResult;
use opencl::opencl_sys::bindings::cl_ulong;
use opencl::wrapper::system::OpenclCommonOperation;

pub const PROGRAM_SRC: &str = r#"

kernel void main_task(
    global ulong* output
    ) {
    int i = get_global_id(0);

    ulong v = clock_read_device();
    
    output[i] = v;
}

"#;

impl SingleGpuExample {
    pub fn not_implemented_1(&self) -> OclResult<Vec<cl_ulong>> {
        let global_work_size = 1024;
        let local_work_size = 256;

        let output_buf = self.system.create_output_buffer(global_work_size)?;

        let mut kernel = self.system.create_kernel("main_task")?;

        unsafe {
            kernel.set_arg(&output_buf)?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        };

        let output =
            self.system
                .blocking_enqueue_read_buffer(global_work_size, &output_buf, &[])?;

        println!("output len {} - {:?}", output.len(), output);

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn case_1() {
        let ocl_program = SingleGpuExample::new(PROGRAM_SRC);

        let _result = ocl_program.not_implemented_1().unwrap();
    }
}
