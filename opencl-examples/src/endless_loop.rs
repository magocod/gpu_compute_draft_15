use crate::utils::SingleGpuExample;
use opencl::error::OclResult;
use opencl::wrapper::system::OpenclCommonOperation;

pub const PROGRAM_SRC: &str = r#"

kernel void main_task() {
    int i = get_global_id(0);

    while(1) {
        // ...
    }
}

"#;

impl SingleGpuExample {
    pub fn endless_loop_1(&self) -> OclResult<()> {
        let kernel = self.system.create_kernel("main_task")?;

        unsafe {
            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                1024,
                256,
                &[],
            )?;
        };

        Ok(())
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//
//     #[test]
//     fn case_1() {
//         let device = SingleGpuExample::new(PROGRAM_SRC);
//
//         device.endless_loop_1().unwrap();
//     }
// }
