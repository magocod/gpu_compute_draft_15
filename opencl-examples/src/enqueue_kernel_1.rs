use crate::utils::SingleGpuExample;
use opencl::error::OclResult;
use opencl::opencl_sys::bindings::cl_int;
use opencl::wrapper::system::OpenclCommonOperation;

pub const PROGRAM_SRC: &str = r#"

global int tmp_arr[TMP_ARR_LEN];

kernel void get_tmp_arr(global int* output) {
    int i = get_global_id(0);
    output[i] = tmp_arr[i];
}

kernel void add_1() {
    int i = get_global_id(0);
    tmp_arr[i] += 1;
}

kernel void add_2_and_get_value(
    global int* output
    ) {
    int i = get_global_id(0);
    tmp_arr[i] += 2;
    output[i] = tmp_arr[i];
}

const int CMQ_ADD_1 = 0;
const int CMQ_ADD_2_AND_GET_VALUE = 1;

kernel void main_task(
    queue_t q0,
    global int* output,
    global int* enqueue_kernel_output
    ) {

    clk_event_t evt0;

    enqueue_kernel_output[CMQ_ADD_1] = enqueue_kernel(
        q0,
        CLK_ENQUEUE_FLAGS_NO_WAIT,
        ndrange_1D(TMP_ARR_LEN, DEVICE_LOCAL_WORK_SIZE),
        0,
        NULL,
        &evt0,
        ^{
           add_1();
        }
    );

    enqueue_kernel_output[CMQ_ADD_2_AND_GET_VALUE] = enqueue_kernel(
        q0,
        CLK_ENQUEUE_FLAGS_NO_WAIT,
        ndrange_1D(TMP_ARR_LEN, DEVICE_LOCAL_WORK_SIZE),
        1,
        &evt0,
        NULL,
        ^{
           add_2_and_get_value(output);
        }
    );

    release_event(evt0);
}

"#;

impl SingleGpuExample {
    pub fn enqueue_kernel_example_1(&self, tmp_arr_len: usize) -> OclResult<Vec<cl_int>> {
        let enqueue_kernel_output_capacity = 2;

        let output_buf = self.system.create_output_buffer(tmp_arr_len)?;
        let enqueue_kernel_output_buf = self
            .system
            .create_output_buffer(enqueue_kernel_output_capacity)?;

        let mut kernel = self.system.create_kernel("main_task")?;

        let q0 = self
            .system
            .get_device_command_queue_0()
            .get_cl_command_queue();

        let global_work_size = 1;
        let local_work_size = 1;

        unsafe {
            kernel.set_arg(&q0)?;
            kernel.set_arg(&output_buf)?;
            kernel.set_arg(&enqueue_kernel_output_buf)?;

            kernel.enqueue_nd_range_kernel(
                self.system.get_host_command_queue(),
                &[],
                &[global_work_size],
                &[local_work_size],
                &[],
            )?;
        };

        let output = self
            .system
            .blocking_enqueue_read_buffer(tmp_arr_len, &output_buf, &[])?;

        println!("example_1 output len {} - {:?}", output.len(), output);

        self.system.assert_device_enqueue_kernel(
            enqueue_kernel_output_capacity,
            enqueue_kernel_output_buf,
            &[],
        )?;

        Ok(output)
    }

    pub fn add_1(&self, tmp_arr_len: usize) -> OclResult<()> {
        let global_work_size = tmp_arr_len;
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let kernel = self.system.create_kernel("add_1")?;

        unsafe {
            kernel.enqueue_nd_range_kernel(
                self.system.get_host_command_queue(),
                &[],
                &[global_work_size],
                &[local_work_size],
                &[],
            )?;
        };

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_case_1() {
        let tmp_arr_len = 32;
        let device_local_work_size = 32;

        let program_src = PROGRAM_SRC
            .replace("TMP_ARR_LEN", &tmp_arr_len.to_string())
            .replace(
                "DEVICE_LOCAL_WORK_SIZE",
                &device_local_work_size.to_string(),
            );
        let ocl_program = SingleGpuExample::new(&program_src);

        let tmp_arr = ocl_program.get_tmp_arr(tmp_arr_len).unwrap();
        assert_eq!(tmp_arr, vec![0; tmp_arr_len]);

        let result = ocl_program.enqueue_kernel_example_1(tmp_arr_len).unwrap();
        assert_eq!(result, vec![3; tmp_arr_len]);

        let tmp_arr = ocl_program.get_tmp_arr(tmp_arr_len).unwrap();
        assert_eq!(tmp_arr, vec![3; tmp_arr_len]);
    }

    #[test]
    fn test_case_2() {
        let tmp_arr_len = 1024;
        let device_local_work_size = 256;

        let program_src = PROGRAM_SRC
            .replace("TMP_ARR_LEN", &tmp_arr_len.to_string())
            .replace(
                "DEVICE_LOCAL_WORK_SIZE",
                &device_local_work_size.to_string(),
            );
        let ocl_program = SingleGpuExample::new(&program_src);

        let tmp_arr = ocl_program.get_tmp_arr(tmp_arr_len).unwrap();
        assert_eq!(tmp_arr, vec![0; tmp_arr_len]);

        let result = ocl_program.enqueue_kernel_example_1(tmp_arr_len).unwrap();
        assert_eq!(result, vec![3; tmp_arr_len]);

        let result = ocl_program.add_1(tmp_arr_len);
        assert!(result.is_ok());

        let tmp_arr = ocl_program.get_tmp_arr(tmp_arr_len).unwrap();
        assert_eq!(tmp_arr, vec![4; tmp_arr_len]);

        let result = ocl_program.enqueue_kernel_example_1(tmp_arr_len).unwrap();
        assert_eq!(result, vec![7; tmp_arr_len]);

        let tmp_arr = ocl_program.get_tmp_arr(tmp_arr_len).unwrap();
        assert_eq!(tmp_arr, vec![7; tmp_arr_len]);
    }
}
