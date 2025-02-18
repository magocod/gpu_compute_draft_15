use crate::utils::SingleGpuExample;
use opencl::error::OclResult;
use opencl::opencl_sys::bindings::cl_int;
use opencl::wrapper::system::OpenclCommonOperation;

pub const PROGRAM_SRC: &str = r#"

const int CMQ_GROUP_TASKS = 0;
const int CMQ_GET_TMP = 1;

const int CMQ_ADD_2 = 0;
const int CMQ_SUB_TASK_A = 1;

const int CMQ_SUB_TASK_A_ADD_2 = 2;
const int CMQ_SUB_TASK_B = 3;

const int CMQ_SUB_TASK_B_ADD_1 = 4;
const int CMQ_SUB_TASK_B_ADD_2 = 5;

global int tmp_arr[TMP_ARR_LEN];

kernel void get_tmp_arr(global int* output) {
    int i = get_global_id(0);
    output[i] = tmp_arr[i];
}

kernel void add_1(
    const int parent_global_index
    ) {

    tmp_arr[parent_global_index] += 1;
}

kernel void add_2(
    const int parent_global_index
    ) {

    tmp_arr[parent_global_index] += 2;
}

kernel void sub_task_b(
    queue_t q0,
    const int parent_global_index,
    const int enqueue_kernel_output_index,
    global int* enqueue_kernel_output
    ) {
    clk_event_t evt0;

    enqueue_kernel_output[CMQ_SUB_TASK_B_ADD_1 + enqueue_kernel_output_index] = enqueue_kernel(
        q0,
        CLK_ENQUEUE_FLAGS_NO_WAIT,
        ndrange_1D(1, 1),
        0,
        NULL,
        &evt0,
        ^{
           add_1(parent_global_index);
        }
    );

    enqueue_kernel_output[CMQ_SUB_TASK_B_ADD_2 + enqueue_kernel_output_index] = enqueue_kernel(
        q0,
        CLK_ENQUEUE_FLAGS_NO_WAIT,
        ndrange_1D(1, 1),
        1,
        &evt0,
        NULL,
        ^{
           add_2(parent_global_index);
        }
    );

    release_event(evt0);
}

kernel void sub_task_a(
    queue_t q0,
    const int parent_global_index,
    const int enqueue_kernel_output_index,
    global int* enqueue_kernel_output
    ) {
    clk_event_t evt0;

    enqueue_kernel_output[CMQ_SUB_TASK_A_ADD_2 + enqueue_kernel_output_index] = enqueue_kernel(
        q0,
        CLK_ENQUEUE_FLAGS_NO_WAIT,
        ndrange_1D(1, 1),
        0,
        NULL,
        &evt0,
        ^{
           add_2(parent_global_index);
        }
    );

    enqueue_kernel_output[CMQ_SUB_TASK_B + enqueue_kernel_output_index] = enqueue_kernel(
        q0,
        CLK_ENQUEUE_FLAGS_NO_WAIT,
        ndrange_1D(1, 1),
        1,
        &evt0,
        NULL,
        ^{
           sub_task_b(
                q0,
                parent_global_index,
                enqueue_kernel_output_index,
                enqueue_kernel_output
           );
        }
    );

    release_event(evt0);
}

kernel void group_tasks(
    queue_t q0,
    global int* enqueue_kernel_output
    ) {

    int i = get_global_id(0);
    int parent_global_index = i;

    clk_event_t evt0;

    // 6 enqueue kernel per kernel
    int enqueue_kernel_output_index = i * 6;

    enqueue_kernel_output[CMQ_ADD_2 + enqueue_kernel_output_index] = enqueue_kernel(
        q0,
        CLK_ENQUEUE_FLAGS_NO_WAIT,
        ndrange_1D(1, 1),
        0,
        NULL,
        &evt0,
        ^{
           add_2(parent_global_index);
        }
    );

    enqueue_kernel_output[CMQ_SUB_TASK_A + enqueue_kernel_output_index] = enqueue_kernel(
        q0,
        CLK_ENQUEUE_FLAGS_NO_WAIT,
        ndrange_1D(1, 1),
        1,
        &evt0,
        NULL,
        ^{
           sub_task_a(
                q0,
                parent_global_index,
                enqueue_kernel_output_index,
                enqueue_kernel_output
           );
        }
    );

    release_event(evt0);
}

kernel void main_task(
    queue_t q0,
    global int* output,
    global int* init_enqueue_kernel_output,
    global int* enqueue_kernel_output
    ) {
    clk_event_t evt0;

    init_enqueue_kernel_output[CMQ_GROUP_TASKS] = enqueue_kernel(
        q0,
        CLK_ENQUEUE_FLAGS_NO_WAIT,
        ndrange_1D(TMP_ARR_LEN, DEVICE_LOCAL_WORK_SIZE),
        0,
        NULL,
        &evt0,
        ^{
           group_tasks(
                q0,
                enqueue_kernel_output
           );
        }
    );

    init_enqueue_kernel_output[CMQ_GET_TMP] = enqueue_kernel(
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
    pub fn enqueue_kernel_example_3(&self, tmp_arr_len: usize) -> OclResult<Vec<cl_int>> {
        let init_enqueue_kernel_output_capacity = 2;

        // 6 enqueue kernel per thread
        let enqueue_kernel_output_capacity = 6 * tmp_arr_len;

        let output_buf = self.system.create_output_buffer(tmp_arr_len)?;

        let init_enqueue_kernel_output_buf = self
            .system
            .create_output_buffer(init_enqueue_kernel_output_capacity)?;
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
            kernel.set_arg(&init_enqueue_kernel_output_buf)?;
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

        println!("example_3 output {} - {:?}", output.len(), output);

        self.system.assert_device_enqueue_kernel(
            init_enqueue_kernel_output_capacity,
            init_enqueue_kernel_output_buf,
            &[],
        )?;

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
        let tmp_arr_len = 512;
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

        let result = ocl_program.enqueue_kernel_example_3(tmp_arr_len).unwrap();
        assert_eq!(result, vec![7; tmp_arr_len]);

        let tmp_arr = ocl_program.get_tmp_arr(tmp_arr_len).unwrap();
        assert_eq!(tmp_arr, vec![7; tmp_arr_len]);
    }

    // fatal error
    // #[test]
    // fn case_1_2() {
    //     let tmp_arr_len = 1024;
    //     let device_local_work_size = 256;
    //     let program_src = PROGRAM_SRC
    //         .replace("TMP_ARR_LEN", &tmp_arr_len.to_string())
    //         .replace(
    //             "DEVICE_LOCAL_WORK_SIZE",
    //             &device_local_work_size.to_string(),
    //         );
    //     let ocl_program = SingleGpuExample::new(&program_src);
    //
    //     let tmp_arr = ocl_program.get_tmp_arr(tmp_arr_len).unwrap();
    //     assert_eq!(tmp_arr, vec![0; tmp_arr_len]);
    //
    //     let result = ocl_program.enqueue_kernel_example_3(tmp_arr_len).unwrap();
    //     assert_eq!(result, vec![7; tmp_arr_len]);
    //
    //     let result = ocl_program.enqueue_kernel_example_3(tmp_arr_len).unwrap();
    //     assert_eq!(result, vec![14; tmp_arr_len]);
    //
    //     let tmp_arr = ocl_program.get_tmp_arr(tmp_arr_len).unwrap();
    //     assert_eq!(tmp_arr, vec![14; tmp_arr_len]);
    // }
}
