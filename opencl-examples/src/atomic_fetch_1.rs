//! # opencl atomic_fetch
//!
//! TODO explain
//!

use crate::utils::SingleGpuExample;
use opencl::error::OclResult;
use opencl::opencl_sys::bindings::cl_int;
use opencl::wrapper::system::OpenclCommonOperation;

pub const TMP_ARR_LEN: usize = 16;

pub const PROGRAM_SRC: &str = r#"

global int tmp_arr[16];

kernel void get_tmp_arr(
    global int* output
    ) {
    int i = get_global_id(0);

    output[i] = tmp_arr[i];

}

kernel void increase_entries_v1(
    const int value,
    global int* output
    ) {
    int i = get_global_id(0);
    
    output[i] = atomic_fetch_add(&tmp_arr[i], value);

}

kernel void decrease_entries_v1(
    const int value,
    global int* output
    ) {
    int i = get_global_id(0);
    
    output[i] = atomic_fetch_sub(&tmp_arr[i], value);

}

"#;

impl SingleGpuExample {
    pub fn increase_entries_v1(&self, elements: usize, value: cl_int) -> OclResult<Vec<cl_int>> {
        let global_work_size = elements;
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let output_buf = self.system.create_output_buffer(global_work_size)?;

        let mut kernel = self.system.create_kernel("increase_entries_v1")?;

        unsafe {
            kernel.set_arg(&value)?;
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
        println!("inc output: {:?}", output);

        Ok(output)
    }

    pub fn decrease_entries_v1(&self, elements: usize, value: cl_int) -> OclResult<Vec<cl_int>> {
        let global_work_size = elements;
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let output_buf = self.system.create_output_buffer(global_work_size)?;

        let mut kernel = self.system.create_kernel("decrease_entries_v1")?;

        unsafe {
            kernel.set_arg(&value)?;
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
        println!("dec output: {:?}", output);

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_increase_entries_v1_case_1() {
        let ocl_program = SingleGpuExample::new(PROGRAM_SRC);

        let tmp_arr = ocl_program.get_tmp_arr(TMP_ARR_LEN).unwrap();
        assert_eq!(tmp_arr, vec![0; TMP_ARR_LEN]);

        let result = ocl_program.increase_entries_v1(16, 1).unwrap();
        assert_eq!(result, vec![0; TMP_ARR_LEN]);

        let tmp_arr = ocl_program.get_tmp_arr(TMP_ARR_LEN).unwrap();
        assert_eq!(tmp_arr, vec![1; TMP_ARR_LEN]);
    }

    #[test]
    fn test_increase_entries_v1_case_2() {
        let ocl_program = SingleGpuExample::new(PROGRAM_SRC);

        let tmp_arr = ocl_program.get_tmp_arr(TMP_ARR_LEN).unwrap();
        assert_eq!(tmp_arr, vec![0; TMP_ARR_LEN]);

        let result = ocl_program.increase_entries_v1(16, 1).unwrap();
        assert_eq!(result, vec![0; TMP_ARR_LEN]);

        let result = ocl_program.increase_entries_v1(4, 2).unwrap();
        assert_eq!(result, vec![1; 4]);

        let tmp_arr = ocl_program.get_tmp_arr(TMP_ARR_LEN).unwrap();

        let mut expected = vec![3; 4];
        expected.append(&mut vec![1; 12]);

        assert_eq!(tmp_arr, expected);
    }

    #[test]
    fn test_decrease_entries_v1() {
        let ocl_program = SingleGpuExample::new(PROGRAM_SRC);

        let tmp_arr = ocl_program.get_tmp_arr(TMP_ARR_LEN).unwrap();
        assert_eq!(tmp_arr, vec![0; TMP_ARR_LEN]);

        let result = ocl_program.decrease_entries_v1(16, 3).unwrap();
        assert_eq!(result, vec![0; TMP_ARR_LEN]);

        let tmp_arr = ocl_program.get_tmp_arr(TMP_ARR_LEN).unwrap();
        assert_eq!(tmp_arr, vec![-3; TMP_ARR_LEN]);
    }

    #[test]
    fn test_increase_and_decrease_entries() {
        let ocl_program = SingleGpuExample::new(PROGRAM_SRC);

        let tmp_arr = ocl_program.get_tmp_arr(TMP_ARR_LEN).unwrap();
        assert_eq!(tmp_arr, vec![0; TMP_ARR_LEN]);

        let result = ocl_program.increase_entries_v1(16, 2).unwrap();
        assert_eq!(result, vec![0; TMP_ARR_LEN]);

        let result = ocl_program.decrease_entries_v1(16, 3).unwrap();
        assert_eq!(result, vec![2; TMP_ARR_LEN]);

        let tmp_arr = ocl_program.get_tmp_arr(TMP_ARR_LEN).unwrap();
        assert_eq!(tmp_arr, vec![-1; TMP_ARR_LEN]);
    }
}
