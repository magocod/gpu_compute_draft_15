use crate::utils::SingleGpuExample;
use opencl::error::OclResult;
use opencl::opencl_sys::bindings::{cl_int, cl_uint};
use opencl::wrapper::memory::Pipe;
use opencl::wrapper::system::OpenclCommonOperation;

pub const PROGRAM_SRC: &str = r#"

kernel void get_ordered_pipe_content(
    __read_only pipe int pipe_0,
    const uint pipe_elements,
    global int* output
    ) {

    for (int i = 0; i < pipe_elements; i++) {
       int pi = -1;

       // TODO check read_pipe return
       read_pipe(pipe_0, &pi);

       output[i] = pi;
    }
}

kernel void get_pipe_info(
    __read_only pipe int pipe_0,
    global uint* output
    ) {

    output[0] = get_pipe_num_packets(pipe_0);
    output[1] = get_pipe_max_packets(pipe_0);
}

kernel void write_in_pipe(
    __write_only pipe int pipe_0,
    global int* values_input,
    global int* pipe_output
    ) {
    
    int i = get_global_id(0);

    int v = values_input[i];
    pipe_output[i] = write_pipe(pipe_0, &v);

}

kernel void read_in_pipe(
    __read_only pipe int pipe_0,
    global int* values_output,
    global int* pipe_output
    ) {
    
    int i = get_global_id(0);

    int pi = -1;
    pipe_output[i] = read_pipe(pipe_0, &pi);
    values_output[i] = pi;

}

"#;

impl SingleGpuExample {
    pub fn get_ordered_pipe_content(
        &self,
        pipe_elements: usize,
        pipe: &Pipe<cl_int>,
    ) -> OclResult<Vec<cl_int>> {
        let global_work_size = 1;
        let local_work_size = 1;

        let output_buf = self.system.create_output_buffer(pipe_elements)?;

        let mut kernel = self.system.create_kernel("get_ordered_pipe_content")?;

        let pipe0 = pipe.get_cl_mem();
        let p_e = pipe_elements as cl_uint;

        let _event = unsafe {
            kernel.set_arg(&pipe0)?;
            kernel.set_arg(&p_e)?;
            kernel.set_arg(&output_buf)?;

            kernel.enqueue_nd_range_kernel(
                self.system.get_host_command_queue(),
                &[],
                &[global_work_size],
                &[local_work_size],
                &[],
            )?
        };

        let output = self
            .system
            .blocking_enqueue_read_buffer(pipe_elements, &output_buf, &[])?;

        println!("pipe len {} - {:?}", output.len(), output);

        Ok(output)
    }

    pub fn get_pipe_info(&self, pipe: &Pipe<cl_int>) -> OclResult<(cl_uint, cl_uint)> {
        let global_work_size = 1;
        let local_work_size = 1;

        let output_len = 2;

        let output_buf = self.system.create_output_buffer(output_len)?;

        let mut kernel = self.system.create_kernel("get_pipe_info")?;

        let pipe0 = pipe.get_cl_mem();

        let _event = unsafe {
            kernel.set_arg(&pipe0)?;
            kernel.set_arg(&output_buf)?;

            kernel.enqueue_nd_range_kernel(
                self.system.get_host_command_queue(),
                &[],
                &[global_work_size],
                &[local_work_size],
                &[],
            )?
        };

        let output = self
            .system
            .blocking_enqueue_read_buffer(output_len, &output_buf, &[])?;

        let num_packets = output[0];
        let max_packets = output[1];
        println!(
            "pipe - num_packets: {}, max_packets: {:?}",
            num_packets, max_packets
        );

        Ok((num_packets, max_packets))
    }

    pub fn write_in_pipe(&self, pipe: &Pipe<cl_int>, input: &[cl_int]) -> OclResult<Vec<cl_int>> {
        let global_work_size = input.len();
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let values_input_buf = self.system.blocking_prepare_input_buffer(input)?;

        let pipe_output_buf = self.system.create_output_buffer(global_work_size)?;

        let mut kernel = self.system.create_kernel("write_in_pipe")?;

        let pipe0 = pipe.get_cl_mem();

        let _event = unsafe {
            kernel.set_arg(&pipe0)?;
            kernel.set_arg(&values_input_buf)?;
            kernel.set_arg(&pipe_output_buf)?;

            kernel.enqueue_nd_range_kernel(
                self.system.get_host_command_queue(),
                &[],
                &[global_work_size],
                &[local_work_size],
                &[],
            )?
        };

        let pipe_output =
            self.system
                .blocking_enqueue_read_buffer(global_work_size, &pipe_output_buf, &[])?;

        println!(
            "write_pipe_output len {} - {:?}",
            pipe_output.len(),
            pipe_output
        );

        Ok(pipe_output)
    }

    pub fn write_in_pipe_with_fatal_error(
        &self,
        pipe: &Pipe<cl_int>,
        input: &[cl_int],
    ) -> OclResult<Vec<cl_int>> {
        let global_work_size = input.len();
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        // let values_input_buf = self.system.prepare_input_buffer(input)?;

        let pipe_output_buf = self.system.create_output_buffer(global_work_size)?;

        let mut kernel = self.system.create_kernel("write_in_pipe")?;

        let pipe0 = pipe.get_cl_mem();

        let _event = unsafe {
            kernel.set_arg(&pipe0)?;
            kernel.set_arg(&input.len())?;
            kernel.set_arg(&pipe_output_buf)?;

            kernel.enqueue_nd_range_kernel(
                self.system.get_host_command_queue(),
                &[],
                &[global_work_size],
                &[local_work_size],
                &[],
            )?
        };

        let pipe_output =
            self.system
                .blocking_enqueue_read_buffer(global_work_size, &pipe_output_buf, &[])?;

        println!(
            "write_pipe_output len {} - {:?}",
            pipe_output.len(),
            pipe_output
        );

        Ok(pipe_output)
    }

    // (pipe_result, values)
    pub fn read_in_pipe(
        &self,
        pipe: &Pipe<cl_int>,
        pipe_elements: usize,
    ) -> OclResult<(Vec<cl_int>, Vec<cl_int>)> {
        let global_work_size = pipe_elements;
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let values_output_buf = self.system.create_output_buffer(global_work_size)?;

        let pipe_output_buf = self.system.create_output_buffer(global_work_size)?;

        let mut kernel = self.system.create_kernel("read_in_pipe")?;

        let pipe0 = pipe.get_cl_mem();

        let _event = unsafe {
            kernel.set_arg(&pipe0)?;
            kernel.set_arg(&values_output_buf)?;
            kernel.set_arg(&pipe_output_buf)?;

            kernel.enqueue_nd_range_kernel(
                self.system.get_host_command_queue(),
                &[],
                &[global_work_size],
                &[local_work_size],
                &[],
            )?
        };

        let pipe_output =
            self.system
                .blocking_enqueue_read_buffer(global_work_size, &pipe_output_buf, &[])?;

        let values_output =
            self.system
                .blocking_enqueue_read_buffer(global_work_size, &values_output_buf, &[])?;

        println!(
            "read_pipe_output len {} - {:?}",
            pipe_output.len(),
            pipe_output
        );

        println!(
            "values_output len {} - {:?}",
            values_output.len(),
            values_output
        );

        Ok((pipe_output, values_output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_in_pipe_case_1() {
        let pipe_elements: usize = 32;

        let ocl_program = SingleGpuExample::new(PROGRAM_SRC);

        let input: Vec<cl_int> = (0..pipe_elements).map(|i| i as i32).collect();
        println!("input {input:?}");

        let pipe = Pipe::new(ocl_program.system.get_context(), pipe_elements as cl_uint).unwrap();

        let result = ocl_program.write_in_pipe(&pipe, &input).unwrap();
        assert_eq!(result, vec![0; pipe_elements]);

        ocl_program.get_pipe_info(&pipe).unwrap();

        let result = ocl_program
            .get_ordered_pipe_content(pipe_elements, &pipe)
            .unwrap();
        assert_eq!(result, input);

        ocl_program.get_pipe_info(&pipe).unwrap();
    }

    #[test]
    fn test_write_in_pipe_case_2() {
        let pipe_elements: usize = 1024;

        let ocl_program = SingleGpuExample::new(PROGRAM_SRC);

        let input: Vec<cl_int> = (0..pipe_elements).map(|i| i as i32).collect();
        println!("input {input:?}");

        let pipe = Pipe::new(ocl_program.system.get_context(), pipe_elements as cl_uint).unwrap();

        let result = ocl_program.write_in_pipe(&pipe, &input).unwrap();
        assert_eq!(result, vec![0; pipe_elements]);

        let result = ocl_program
            .get_ordered_pipe_content(pipe_elements, &pipe)
            .unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();
        assert_eq!(result_sorted, input);
    }

    #[test]
    fn test_write_in_pipe_case_3() {
        let pipe_elements: usize = 32;

        let ocl_program = SingleGpuExample::new(PROGRAM_SRC);

        let input: Vec<cl_int> = (0..(pipe_elements / 2)).map(|i| i as i32).collect();
        println!("input {input:?}");

        let pipe = Pipe::new(ocl_program.system.get_context(), pipe_elements as cl_uint).unwrap();

        let result = ocl_program.write_in_pipe(&pipe, &input).unwrap();
        assert_eq!(result, vec![0; pipe_elements / 2]);

        ocl_program.get_pipe_info(&pipe).unwrap();

        let result = ocl_program
            .get_ordered_pipe_content(pipe_elements, &pipe)
            .unwrap();

        let mut expected = input.clone();
        expected.append(&mut vec![-1; pipe_elements / 2]);

        assert_eq!(result, expected);

        ocl_program.get_pipe_info(&pipe).unwrap();
    }

    #[test]
    fn test_read_in_pipe_case_1() {
        let pipe_elements: usize = 32;

        let ocl_program = SingleGpuExample::new(PROGRAM_SRC);

        let input: Vec<cl_int> = (0..pipe_elements).map(|i| i as i32).collect();
        println!("input {input:?}");

        let pipe = Pipe::new(ocl_program.system.get_context(), pipe_elements as cl_uint).unwrap();

        let result = ocl_program.write_in_pipe(&pipe, &input).unwrap();
        assert_eq!(result, vec![0; pipe_elements]);

        let (pipe_result, values) = ocl_program.read_in_pipe(&pipe, pipe_elements).unwrap();
        assert_eq!(pipe_result, vec![0; pipe_elements]);
        assert_eq!(values, input);
    }

    #[test]
    fn test_read_in_pipe_case_2() {
        let pipe_elements: usize = 1024;

        let ocl_program = SingleGpuExample::new(PROGRAM_SRC);

        let input: Vec<cl_int> = (0..pipe_elements).map(|i| i as i32).collect();
        println!("input {input:?}");

        let pipe = Pipe::new(ocl_program.system.get_context(), pipe_elements as cl_uint).unwrap();

        let result = ocl_program.write_in_pipe(&pipe, &input).unwrap();
        assert_eq!(result, vec![0; pipe_elements]);

        let (pipe_result, values) = ocl_program.read_in_pipe(&pipe, pipe_elements).unwrap();
        assert_eq!(pipe_result, vec![0; pipe_elements]);
        let mut values_sorted = values.clone();
        values_sorted.sort();
        assert_eq!(values_sorted, input);
    }

    #[test]
    fn test_read_in_pipe_case_3() {
        let pipe_elements: usize = 1024;

        let ocl_program = SingleGpuExample::new(PROGRAM_SRC);

        let pipe = Pipe::new(ocl_program.system.get_context(), pipe_elements as cl_uint).unwrap();

        let (pipe_result, values) = ocl_program.read_in_pipe(&pipe, pipe_elements).unwrap();
        assert_eq!(pipe_result, vec![-1; pipe_elements]);
        assert_eq!(values, vec![-1; pipe_elements]);
    }
}

// #[cfg(test)]
// mod tests_fatal_errors {
//     use super::*;
//
//     #[test]
//     fn test_fatal_write_in_pipe_with_fatal_error() {
//         let pipe_elements: usize = 1024;
//
//         let ocl_program = SingleGpuExample::new(PROGRAM_SRC);
//
//         let input: Vec<cl_int> = (0..pipe_elements).map(|i| i as i32).collect();
//         println!("input {input:?}");
//
//         let pipe = Pipe::new(ocl_program.system.get_context(), pipe_elements as cl_uint).unwrap();
//
//         let result = ocl_program.write_in_pipe_with_fatal_error(&pipe, &input).unwrap();
//         assert_eq!(result, vec![0; pipe_elements]);
//
//         let result = ocl_program.get_ordered_pipe_content(pipe_elements, &pipe).unwrap();
//
//         let mut result_sorted = result.clone();
//         result_sorted.sort();
//         assert_eq!(result_sorted, input);
//     }
// }
