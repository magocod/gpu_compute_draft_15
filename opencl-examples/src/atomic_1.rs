use crate::utils::SingleGpuExample;
use opencl::error::OclResult;
use opencl::opencl_sys::bindings::{cl_int, cl_uint};
use opencl::wrapper::system::OpenclCommonOperation;

pub const PROGRAM_SRC: &str = r#"

global int tmp_count = 0;

kernel void get_tmp_count(
    global int* output
    ) {

    output[0] = tmp_count;

}

kernel void increment_value_v1(
    global int* output
    ) {

    int i = get_global_id(0);
    output[i] = atomic_fetch_add(&tmp_count, 1);

}

// ERROR
kernel void increment_value_v1_2(
    global int* output
    ) {
    int i = get_global_id(0);

    output[i] = tmp_count++;
}

kernel void increment_value_v2(
    queue_t q0,
    const uint increment_global_work_size,
    const uint increment_local_work_size,
    global int* output,
    global int* tmp_count_output,
    global int* queue_output
    ) {

    clk_event_t evt0;

    queue_output[0] = enqueue_kernel(
        q0,
        CLK_ENQUEUE_FLAGS_NO_WAIT,
        ndrange_1D(increment_global_work_size, increment_local_work_size),
        0,
        NULL,
        &evt0,
        ^{
            int i = get_global_id(0);
            output[i] = atomic_fetch_add(&tmp_count, 1);
        }
    );

    queue_output[1] = enqueue_kernel(
        q0,
        CLK_ENQUEUE_FLAGS_NO_WAIT,
        ndrange_1D(1, 1),
        1,
        &evt0,
        NULL,
        ^{
           get_tmp_count(tmp_count_output);
        }
    );

    release_event(evt0);
}

// error
kernel void increment_value_v2_2(
    queue_t q0,
    const uint increment_global_work_size,
    const uint increment_local_work_size,
    global int* output,
    global int* tmp_count_output,
    global int* queue_output
    ) {

    clk_event_t evt0;

    queue_output[0] = enqueue_kernel(
        q0,
        CLK_ENQUEUE_FLAGS_NO_WAIT,
        ndrange_1D(increment_global_work_size, increment_local_work_size),
        0,
        NULL,
        &evt0,
        ^{
            int i = get_global_id(0);
            output[i] = tmp_count++;
        }
    );

    queue_output[1] = enqueue_kernel(
        q0,
        CLK_ENQUEUE_FLAGS_NO_WAIT,
        ndrange_1D(1, 1),
        1,
        &evt0,
        NULL,
        ^{
           get_tmp_count(tmp_count_output);
        }
    );

    release_event(evt0);
}

// CL_QUEUE_ON_DEVICE - Indicates that this is a device queue. If CL_QUEUE_ON_DEVICE is set, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE must also be set.
kernel void increment_value_v3(
    queue_t q0,
    const uint increment_global_work_size,
    global int* output,
    global int* queue_output
    ) {

    for (int i = 0; i < increment_global_work_size; i++) {
        queue_output[i] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(1, 1),
            ^{            
                output[i] = tmp_count++;
            }
        );
    }

}

// CL_QUEUE_ON_DEVICE - Indicates that this is a device queue. If CL_QUEUE_ON_DEVICE is set, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE must also be set.
kernel void increment_value_v4(
    queue_t q0,
    global int* output,
    global int* queue_output
    ) {
    
    int i = get_global_id(0);

    queue_output[i] = enqueue_kernel(
        q0,
        CLK_ENQUEUE_FLAGS_NO_WAIT,
        ndrange_1D(1, 1),
        ^{
            output[i] = tmp_count++;
        }
    );

}

"#;

impl SingleGpuExample {
    pub fn get_tmp_count(&self) -> OclResult<Vec<cl_int>> {
        let global_work_size = 1;
        let local_work_size = 1;

        let output_buf = self.system.create_output_buffer(global_work_size)?;

        let mut kernel = self.system.create_kernel("get_tmp_count")?;

        unsafe {
            kernel.set_arg(&output_buf)?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        };

        let output: Vec<cl_int> =
            self.system
                .blocking_enqueue_read_buffer(global_work_size, &output_buf, &[])?;
        println!("tmp_count: {:?}", output);

        Ok(output)
    }

    pub fn increment_value_v1(&self, value: usize) -> OclResult<Vec<cl_int>> {
        let global_work_size = value;
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let output_buf = self.system.create_output_buffer(global_work_size)?;

        let mut kernel = self.system.create_kernel("increment_value_v1")?;

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
        println!("output: {:?}", output);

        Ok(output)
    }

    pub fn increment_value_v1_2(&self, value: usize) -> OclResult<Vec<cl_int>> {
        let global_work_size = value;
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let output_buf = self.system.create_output_buffer(global_work_size)?;

        let mut kernel = self.system.create_kernel("increment_value_v1_2")?;

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
        println!("output: {:?}", output);

        Ok(output)
    }

    pub fn increment_value_v2(&self, value: usize) -> OclResult<Vec<cl_int>> {
        let global_work_size = 1;
        let local_work_size = 1;

        let tmp_count_output_capacity = 1;
        let enqueue_kernel_output_capacity = 2;

        let output_buf = self.system.create_output_buffer(value)?;
        let tmp_count_output_buf = self
            .system
            .create_output_buffer(tmp_count_output_capacity)?;
        let enqueue_kernel_output_buf = self
            .system
            .create_output_buffer(enqueue_kernel_output_capacity)?;

        let mut kernel = self.system.create_kernel("increment_value_v2")?;

        let q0 = self
            .system
            .get_device_command_queue_0()
            .get_cl_command_queue();

        let increment_global_work_size = value as cl_uint;
        let increment_local_work_size =
            self.system.first_device_check_local_work_size(value) as cl_uint;

        unsafe {
            kernel.set_arg(&q0)?;
            kernel.set_arg(&increment_global_work_size)?;
            kernel.set_arg(&increment_local_work_size)?;
            kernel.set_arg(&output_buf)?;
            kernel.set_arg(&tmp_count_output_buf)?;
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
            .blocking_enqueue_read_buffer(value, &output_buf, &[])?;
        println!("output: {:?}", output);

        let tmp_count_output: Vec<cl_int> = self.system.blocking_enqueue_read_buffer(
            tmp_count_output_capacity,
            &tmp_count_output_buf,
            &[],
        )?;
        println!("tmp_count: {:?}", tmp_count_output);

        self.system.assert_device_enqueue_kernel(
            enqueue_kernel_output_capacity,
            enqueue_kernel_output_buf,
            &[],
        )?;

        Ok(output)
    }

    pub fn increment_value_v2_2(&self, value: usize) -> OclResult<Vec<cl_int>> {
        let global_work_size = 1;
        let local_work_size = 1;

        let tmp_count_output_capacity = 1;
        let enqueue_kernel_output_capacity = 2;

        let output_buf = self.system.create_output_buffer(global_work_size)?;
        let tmp_count_output_buf = self
            .system
            .create_output_buffer(tmp_count_output_capacity)?;
        let enqueue_kernel_output_buf = self
            .system
            .create_output_buffer(enqueue_kernel_output_capacity)?;

        let mut kernel = self.system.create_kernel("increment_value_v2_2")?;

        let q0 = self
            .system
            .get_device_command_queue_0()
            .get_cl_command_queue();

        let increment_global_work_size = value as cl_uint;
        let increment_local_work_size =
            self.system.first_device_check_local_work_size(value) as cl_uint;

        unsafe {
            kernel.set_arg(&q0)?;
            kernel.set_arg(&increment_global_work_size)?;
            kernel.set_arg(&increment_local_work_size)?;
            kernel.set_arg(&output_buf)?;
            kernel.set_arg(&tmp_count_output_buf)?;
            kernel.set_arg(&enqueue_kernel_output_buf)?;

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
        println!("output: {:?}", output);

        let tmp_count_output: Vec<cl_int> = self.system.blocking_enqueue_read_buffer(
            tmp_count_output_capacity,
            &tmp_count_output_buf,
            &[],
        )?;
        println!("tmp_count: {:?}", tmp_count_output);

        self.system.assert_device_enqueue_kernel(
            enqueue_kernel_output_capacity,
            enqueue_kernel_output_buf,
            &[],
        )?;

        Ok(output)
    }

    pub fn increment_value_v3(&self, value: usize) -> OclResult<Vec<cl_int>> {
        let global_work_size = 1;
        let local_work_size = 1;

        let output_capacity = value;
        let enqueue_kernel_output_capacity = value;

        let output_buf = self.system.create_output_buffer(output_capacity)?;
        let enqueue_kernel_output_buf = self
            .system
            .create_output_buffer(enqueue_kernel_output_capacity)?;

        let mut kernel = self.system.create_kernel("increment_value_v3")?;

        let q0 = self
            .system
            .get_device_command_queue_0()
            .get_cl_command_queue();

        let increment_global_work_size = value as cl_uint;

        unsafe {
            kernel.set_arg(&q0)?;
            kernel.set_arg(&increment_global_work_size)?;
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
            .blocking_enqueue_read_buffer(output_capacity, &output_buf, &[])?;
        println!("output: {:?}", output);

        let ek_codes = self.system.assert_device_enqueue_kernel(
            enqueue_kernel_output_capacity,
            enqueue_kernel_output_buf,
            &[],
        )?;
        println!("enqueue_kernel {:?}", ek_codes);

        Ok(output)
    }

    pub fn increment_value_v4(&self, value: usize) -> OclResult<Vec<cl_int>> {
        let global_work_size = value;
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let output_capacity = value;
        let enqueue_kernel_output_capacity = value;

        let output_buf = self.system.create_output_buffer(output_capacity)?;
        let enqueue_kernel_output_buf = self
            .system
            .create_output_buffer(enqueue_kernel_output_capacity)?;

        let mut kernel = self.system.create_kernel("increment_value_v4")?;

        let q0 = self
            .system
            .get_device_command_queue_0()
            .get_cl_command_queue();

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
            .blocking_enqueue_read_buffer(output_capacity, &output_buf, &[])?;
        println!("output: {:?}", output);

        let ek_codes = self.system.assert_device_enqueue_kernel(
            enqueue_kernel_output_capacity,
            enqueue_kernel_output_buf,
            &[],
        )?;
        println!("ek_codes {:?}", ek_codes);

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_increment_value_v1() {
        let value = 1024;

        let ocl_program = SingleGpuExample::new(PROGRAM_SRC);

        let result = ocl_program.increment_value_v1(value).unwrap();
        let tmp_count = ocl_program.get_tmp_count().unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        let expected: Vec<i32> = (0..value as cl_int).collect();
        assert_eq!(result_sorted, expected);

        assert_eq!(tmp_count, vec![value as cl_int]);
    }

    // error
    #[test]
    fn test_increment_value_v1_2() {
        let value = 1024;

        let ocl_program = SingleGpuExample::new(PROGRAM_SRC);

        let result = ocl_program.increment_value_v1_2(value).unwrap();
        let tmp_count = ocl_program.get_tmp_count().unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        // let expected: Vec<i32> = (0..value as cl_int).collect();
        assert_eq!(result_sorted, vec![0; value]);

        // assert_eq!(tmp_count, vec![value as cl_int]);
        assert_eq!(tmp_count, vec![1]);
    }

    #[test]
    fn test_increment_value_v2() {
        let value = 1024;

        let ocl_program = SingleGpuExample::new(PROGRAM_SRC);

        let result = ocl_program.increment_value_v2(value).unwrap();
        let tmp_count = ocl_program.get_tmp_count().unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        let expected: Vec<i32> = (0..value as cl_int).collect();
        assert_eq!(result_sorted, expected);

        assert_eq!(tmp_count, vec![value as cl_int]);
    }

    #[test]
    fn test_increment_value_v3_success() {
        let value = 128;

        let ocl_program = SingleGpuExample::new(PROGRAM_SRC);

        let result = ocl_program.increment_value_v3(value).unwrap();
        let tmp_count = ocl_program.get_tmp_count().unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        let expected: Vec<i32> = (0..value as cl_int).collect();
        assert_eq!(result_sorted, expected);

        assert_eq!(tmp_count, vec![value as cl_int]);
    }

    // error
    #[test]
    #[should_panic]
    fn test_increment_value_v3_error() {
        let value = 1024 * 3;

        let ocl_program = SingleGpuExample::new(PROGRAM_SRC);

        let result = ocl_program.increment_value_v3(value).unwrap();
        let tmp_count = ocl_program.get_tmp_count().unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        let expected: Vec<i32> = (0..value as cl_int).collect();
        assert_eq!(result_sorted, expected);

        assert_eq!(tmp_count, vec![value as cl_int]);
    }

    #[test]
    fn test_increment_value_v4_success() {
        let value = 1024;

        let ocl_program = SingleGpuExample::new(PROGRAM_SRC);

        let result = ocl_program.increment_value_v4(value).unwrap();
        let tmp_count = ocl_program.get_tmp_count().unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        let expected: Vec<i32> = (0..value as cl_int).collect();
        assert_eq!(result_sorted, expected);

        assert_eq!(tmp_count, vec![value as cl_int]);
    }

    // error
    #[test]
    #[should_panic]
    fn test_increment_value_v4_error() {
        let value = 1024 * 3; // error

        let ocl_program = SingleGpuExample::new(PROGRAM_SRC);

        let result = ocl_program.increment_value_v4(value).unwrap();
        let tmp_count = ocl_program.get_tmp_count().unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        let expected: Vec<i32> = (0..value as cl_int).collect();
        assert_eq!(result_sorted, expected);

        assert_eq!(tmp_count, vec![value as cl_int]);
    }
}
