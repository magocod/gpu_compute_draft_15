use crate::utils::SingleGpuExample;
use opencl::error::OclResult;
use opencl::opencl_sys::bindings::{cl_int, cl_uint};
use opencl::wrapper::system::OpenclCommonOperation;

pub const PROGRAM_SRC: &str = r#"

const int CMQ_ADD_1 = 0;
const int CMQ_CONFIRM = 1;
const int CMQ_SUB_TASK = 2;

global int tmp_arr[TMP_ARR_LEN];

kernel void sub_task(
    queue_t q0,
    global int* stop_input,
    global int* stop_output,
    global uint* recursive_count_output,
    global int* enqueue_kernel_output
);

kernel void get_tmp_arr(global int* output) {
    int i = get_global_id(0);
    output[i] = tmp_arr[i];
}

kernel void add_1(
    global int* stop_input,
    global int* stop_output
    ) {
    int i = get_global_id(0);

    if (tmp_arr[i] >= stop_input[i]) {
        stop_output[i] = 0;
        return;
    }

    tmp_arr[i] += 1;
}

kernel void confirm(
    queue_t q0,
    global int* stop_input,
    global int* stop_output,
    global uint* recursive_count_output,
    global int* enqueue_kernel_output
    ) {

    bool continue_execution = false;
    int enqueue_kernel_result = 0;

    for (int index = 0; index < TMP_ARR_LEN; index++) {
        if (stop_output[index] != 0) {
            continue_execution = true;
            break;
        }
    }

    if (continue_execution == true) {
        recursive_count_output[0] += 1;

        enqueue_kernel_result = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(1, 1),
            ^{
               sub_task(
                   q0,
                   stop_input,
                   stop_output,
                   recursive_count_output,
                   enqueue_kernel_output
               );
            }
        );
    }

    enqueue_kernel_output[CMQ_SUB_TASK] = enqueue_kernel_result;
}

kernel void sub_task(
    queue_t q0,
    global int* stop_input,
    global int* stop_output,
    global uint* recursive_count_output,
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
           add_1(
                stop_input,
                stop_output
           );
        }
    );

    enqueue_kernel_output[CMQ_CONFIRM] = enqueue_kernel(
        q0,
        CLK_ENQUEUE_FLAGS_NO_WAIT,
        ndrange_1D(1, 1),
        1,
        &evt0,
        NULL,
        ^{
           confirm(
               q0,
               stop_input,
               stop_output,
               recursive_count_output,
               enqueue_kernel_output
           );
        }
    );

    release_event(evt0);
}

kernel void main_task(
    queue_t q0,
    global int* stop_input,
    global int* stop_output,
    global uint* recursive_count_output,
    global int* enqueue_kernel_output
    ) {

    clk_event_t evt0;

    // init
    for (int index = 0; index < TMP_ARR_LEN; index++) {
        stop_output[index] = -1;
    }
    recursive_count_output[0] = 0;

    enqueue_kernel_output[CMQ_ADD_1] = enqueue_kernel(
        q0,
        CLK_ENQUEUE_FLAGS_NO_WAIT,
        ndrange_1D(TMP_ARR_LEN, DEVICE_LOCAL_WORK_SIZE),
        0,
        NULL,
        &evt0,
        ^{
           add_1(
                stop_input,
                stop_output
           );
        }
    );

    enqueue_kernel_output[CMQ_CONFIRM] = enqueue_kernel(
        q0,
        CLK_ENQUEUE_FLAGS_NO_WAIT,
        ndrange_1D(1, 1),
        1,
        &evt0,
        NULL,
        ^{
           confirm(
               q0,
               stop_input,
               stop_output,
               recursive_count_output,
               enqueue_kernel_output
           );
        }
    );

    release_event(evt0);
}

"#;

impl SingleGpuExample {
    pub fn enqueue_kernel_example_5(
        &self,
        tmp_arr_len: usize,
        stop_input: &[cl_int],
    ) -> OclResult<(Vec<cl_int>, Vec<cl_uint>)> {
        let recursive_count_output_capacity = 1;
        let enqueue_kernel_output_capacity = 3;

        let stop_input_buf = self.system.blocking_prepare_input_buffer(stop_input)?;
        let stop_output_buf = self.system.create_output_buffer(tmp_arr_len)?;
        let recursive_count_output_buf = self
            .system
            .create_output_buffer(recursive_count_output_capacity)?;

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
            kernel.set_arg(&stop_input_buf)?;
            kernel.set_arg(&stop_output_buf)?;
            kernel.set_arg(&recursive_count_output_buf)?;
            kernel.set_arg(&enqueue_kernel_output_buf)?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        };

        let stop_output =
            self.system
                .blocking_enqueue_read_buffer(tmp_arr_len, &stop_output_buf, &[])?;

        println!("stop_output len {} - {stop_output:?}", stop_output.len());

        let recursive_count_output = self.system.blocking_enqueue_read_buffer(
            recursive_count_output_capacity,
            &recursive_count_output_buf,
            &[],
        )?;

        println!(
            "recursive_count_output len {} - {recursive_count_output:?}",
            recursive_count_output.len()
        );

        self.system.assert_device_enqueue_kernel(
            enqueue_kernel_output_capacity,
            enqueue_kernel_output_buf,
            &[],
        )?;

        Ok((stop_output, recursive_count_output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn case_1() {
        let tmp_arr_len: usize = 1024;
        let device_local_work_size = 256;

        let stop_number = 256;

        let program_src = PROGRAM_SRC
            .replace("TMP_ARR_LEN", &tmp_arr_len.to_string())
            .replace(
                "DEVICE_LOCAL_WORK_SIZE",
                &device_local_work_size.to_string(),
            );
        let ocl_program = SingleGpuExample::new(&program_src);

        let tmp_arr = ocl_program.get_tmp_arr(tmp_arr_len).unwrap();
        assert_eq!(tmp_arr, vec![0; tmp_arr_len]);

        let input = vec![stop_number; tmp_arr_len];
        let (result, recursive_call) = ocl_program
            .enqueue_kernel_example_5(tmp_arr_len, &input)
            .unwrap();

        assert_eq!(result, vec![0; tmp_arr_len]);
        assert_eq!(recursive_call, vec![stop_number as cl_uint]);

        let tmp_arr = ocl_program.get_tmp_arr(tmp_arr_len).unwrap();
        assert_eq!(tmp_arr, vec![stop_number; tmp_arr_len]);

        let input = vec![stop_number; tmp_arr_len];
        let (result, recursive_call) = ocl_program
            .enqueue_kernel_example_5(tmp_arr_len, &input)
            .unwrap();

        assert_eq!(result, vec![0; tmp_arr_len]);
        assert_eq!(recursive_call, vec![0]);

        let tmp_arr = ocl_program.get_tmp_arr(tmp_arr_len).unwrap();
        assert_eq!(tmp_arr, vec![stop_number; tmp_arr_len]);
    }

    // fatal error
    // #[test]
    // fn case_1_error() {
    //     let tmp_arr_len: usize = 1024;
    //     let device_local_work_size = 256;
    //
    //     let stop_number = 1024;
    //
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
    //     let input = vec![stop_number; tmp_arr_len];
    //     let (result, recursive_call) = ocl_program.enqueue_kernel_example_5(tmp_arr_len, &input).unwrap();
    //
    //     assert_eq!(result, vec![0; tmp_arr_len]);
    //     assert_eq!(recursive_call, vec![stop_number as cl_uint]);
    //
    //     let tmp_arr = ocl_program.get_tmp_arr(tmp_arr_len).unwrap();
    //     assert_eq!(tmp_arr, vec![stop_number; tmp_arr_len]);
    //
    //     let input = vec![stop_number; tmp_arr_len];
    //     let (result, recursive_call) = ocl_program.enqueue_kernel_example_5(tmp_arr_len, &input).unwrap();
    //
    //     assert_eq!(result, vec![0; tmp_arr_len]);
    //     assert_eq!(recursive_call, vec![0]);
    //
    //     let tmp_arr = ocl_program.get_tmp_arr(tmp_arr_len).unwrap();
    //     assert_eq!(tmp_arr, vec![stop_number; tmp_arr_len]);
    // }

    #[test]
    fn case_2() {
        let tmp_arr_len: usize = 1024;
        let device_local_work_size = 256;

        let program_src = PROGRAM_SRC
            .replace("TMP_ARR_LEN", &tmp_arr_len.to_string())
            .replace(
                "DEVICE_LOCAL_WORK_SIZE",
                &device_local_work_size.to_string(),
            );
        let ocl_program = SingleGpuExample::new(&program_src);

        let input = vec![0; tmp_arr_len];
        let (result, recursive_call) = ocl_program
            .enqueue_kernel_example_5(tmp_arr_len, &input)
            .unwrap();
        assert_eq!(result, vec![0; tmp_arr_len]);
        assert_eq!(recursive_call, vec![0]);

        let tmp_arr = ocl_program.get_tmp_arr(tmp_arr_len).unwrap();
        assert_eq!(tmp_arr, vec![0; tmp_arr_len]);
    }

    #[test]
    fn case_3() {
        let tmp_arr_len: usize = 1024;
        let device_local_work_size = 256;

        let program_src = PROGRAM_SRC
            .replace("TMP_ARR_LEN", &tmp_arr_len.to_string())
            .replace(
                "DEVICE_LOCAL_WORK_SIZE",
                &device_local_work_size.to_string(),
            );
        let ocl_program = SingleGpuExample::new(&program_src);

        let input: Vec<cl_int> = (0..tmp_arr_len as cl_int).collect();
        let (result, recursive_call) = ocl_program
            .enqueue_kernel_example_5(tmp_arr_len, &input)
            .unwrap();
        assert_eq!(result, vec![0; tmp_arr_len]);
        assert_eq!(recursive_call, vec![(tmp_arr_len as cl_uint) - 1]);

        let tmp_arr = ocl_program.get_tmp_arr(tmp_arr_len).unwrap();
        assert_eq!(tmp_arr, input);
    }

    // fatal error
    // #[test]
    // fn case_3_error() {
    //     let tmp_arr_len: usize = 1024 * 2;
    //     let device_local_work_size = 256;
    //
    //     let program_src = PROGRAM_SRC
    //         .replace("TMP_ARR_LEN", &tmp_arr_len.to_string())
    //         .replace(
    //             "DEVICE_LOCAL_WORK_SIZE",
    //             &device_local_work_size.to_string(),
    //         );
    //     let ocl_program = SingleGpuExample::new(&program_src);
    //
    //     let input: Vec<cl_int> = (0..tmp_arr_len as cl_int).collect();
    //     let (result, recursive_call) = ocl_program.enqueue_kernel_example_5(tmp_arr_len, &input).unwrap();
    //     assert_eq!(result, vec![0; tmp_arr_len]);
    //     assert_eq!(recursive_call, vec![(tmp_arr_len as cl_uint) - 1]);
    //
    //     let tmp_arr = ocl_program.get_tmp_arr(tmp_arr_len).unwrap();
    //     assert_eq!(tmp_arr, input);
    // }
}
