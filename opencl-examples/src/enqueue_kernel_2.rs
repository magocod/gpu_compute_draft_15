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

kernel void add_2() {
    int i = get_global_id(0);
    tmp_arr[i] += 2;
}

const int CMQ_SUB_TASK = 0;

const int CMB_SUB_TASK_1 = 1;
const int CMB_SUB_TASK_2 = 2;
const int CMB_SUB_TASK_3 = 3;
const int CMB_SUB_TASK_4 = 4;

const int CMQ_GET_TMP = 5;

kernel void sub_task(
    queue_t q0,
    global int* enqueue_kernel_output
    ) {

    enqueue_kernel_output[CMB_SUB_TASK_1] = enqueue_kernel(
        q0,
        CLK_ENQUEUE_FLAGS_NO_WAIT,
        ndrange_1D(TMP_ARR_LEN, DEVICE_LOCAL_WORK_SIZE),
        ^{
           add_1();
        }
    );

    enqueue_kernel_output[CMB_SUB_TASK_2] = enqueue_kernel(
        q0,
        CLK_ENQUEUE_FLAGS_NO_WAIT,
        ndrange_1D(TMP_ARR_LEN, DEVICE_LOCAL_WORK_SIZE),
        ^{
           add_1();
        }
    );

    enqueue_kernel_output[CMB_SUB_TASK_3] = enqueue_kernel(
        q0,
        CLK_ENQUEUE_FLAGS_NO_WAIT,
        ndrange_1D(TMP_ARR_LEN, DEVICE_LOCAL_WORK_SIZE),
        ^{
           add_2();
        }
    );

    enqueue_kernel_output[CMB_SUB_TASK_4] = enqueue_kernel(
        q0,
        CLK_ENQUEUE_FLAGS_NO_WAIT,
        ndrange_1D(TMP_ARR_LEN, DEVICE_LOCAL_WORK_SIZE),
        ^{
           add_2();
        }
    );

}

kernel void main_task(
    queue_t q0,
    global int* output,
    global int* enqueue_kernel_output
    ) {
    int i = get_global_id(0);

    clk_event_t evt0;

    enqueue_kernel_output[CMQ_SUB_TASK] = enqueue_kernel(
        q0,
        CLK_ENQUEUE_FLAGS_NO_WAIT,
        ndrange_1D(1, 1),
        0,
        NULL,
        &evt0,
        ^{
           sub_task(
               q0,
               enqueue_kernel_output
           );
        }
    );

    enqueue_kernel_output[CMQ_GET_TMP] = enqueue_kernel(
        q0,
        CLK_ENQUEUE_FLAGS_NO_WAIT,
        ndrange_1D(TMP_ARR_LEN, DEVICE_LOCAL_WORK_SIZE),
        1,
        &evt0,
        NULL,
        ^{
           get_tmp_arr(output);
        }
    );

    release_event(evt0);
}

"#;

impl SingleGpuExample {
    pub fn enqueue_kernel_example_2(&self, tmp_arr_len: usize) -> OclResult<Vec<cl_int>> {
        let enqueue_kernel_output_capacity = 6;

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

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        };

        let output = self
            .system
            .blocking_enqueue_read_buffer(tmp_arr_len, &output_buf, &[])?;

        println!("output len {} - {:?}", output.len(), output);

        self.system.assert_device_enqueue_kernel(
            enqueue_kernel_output_capacity,
            enqueue_kernel_output_buf,
            &[],
        )?;

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn case_1() {
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

        let result = ocl_program.enqueue_kernel_example_2(tmp_arr_len).unwrap();
        assert_eq!(result, vec![6; tmp_arr_len]);

        let tmp_arr = ocl_program.get_tmp_arr(tmp_arr_len).unwrap();
        assert_eq!(tmp_arr, vec![6; tmp_arr_len]);
    }

    #[test]
    fn case_2() {
        let tmp_arr_len = 1024;
        let device_local_work_size = 256;
        let program_src = PROGRAM_SRC
            .replace("TMP_ARR_LEN", &tmp_arr_len.to_string())
            .replace(
                "DEVICE_LOCAL_WORK_SIZE",
                &device_local_work_size.to_string(),
            );
        let ocl_program = SingleGpuExample::new(&program_src);

        let result = ocl_program.enqueue_kernel_example_2(tmp_arr_len).unwrap();
        assert_eq!(result, vec![6; tmp_arr_len]);

        let tmp_arr = ocl_program.get_tmp_arr(tmp_arr_len).unwrap();
        assert_eq!(tmp_arr, vec![6; tmp_arr_len]);

        let result = ocl_program.enqueue_kernel_example_2(tmp_arr_len).unwrap();
        assert_eq!(result, vec![12; tmp_arr_len]);

        let tmp_arr = ocl_program.get_tmp_arr(tmp_arr_len).unwrap();
        assert_eq!(tmp_arr, vec![12; tmp_arr_len]);

        let result = ocl_program.enqueue_kernel_example_2(tmp_arr_len).unwrap();
        assert_eq!(result, vec![18; tmp_arr_len]);

        let tmp_arr = ocl_program.get_tmp_arr(tmp_arr_len).unwrap();
        assert_eq!(tmp_arr, vec![18; tmp_arr_len]);
    }
}
