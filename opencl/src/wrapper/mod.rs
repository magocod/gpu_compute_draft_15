//! # Opencl safe wrapper
//!
//! ...
//!

pub mod context;
pub mod memory;
pub mod platform;
pub mod program;
pub mod system;

#[cfg(test)]
mod tests {
    use crate::wrapper::context::{CommandQueue, Context};
    use crate::wrapper::memory::Buffer;
    use crate::wrapper::platform::Platform;
    use crate::wrapper::program::{Kernel, Program};
    use opencl_sys::bindings::{
        cl_mem_flags, cl_queue_properties, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY,
        CL_QUEUE_PROFILING_ENABLE, CL_TRUE,
    };

    const PROGRAM_SRC: &str = r#"
        __kernel void vecAdd(
            __global int *input_a,
            __global int *input_b,
            __global int *output_c
        ) {
            int id = get_global_id(0);

            output_c[id] = input_a[id] + input_b[id];
        }

    "#;

    // similar to resources/basic_vector_add_1.c
    #[test]
    fn test_example_basic_vector_add() {
        // Create the two input vectors
        let list_size = 1024;

        let mut input_a: Vec<i32> = vec![0; list_size];
        let mut input_b: Vec<i32> = vec![0; list_size];

        for i in 0..list_size {
            input_a[i] = i as i32;
            input_b[i] = list_size as i32 - i as i32;
            // input_b[i] = i as i32;
        }

        // Load the kernel source code into the array source_str
        // PROGRAM_SRC

        // Get platform and device information

        let platform = Platform::first().unwrap();
        println!("{:#?}", platform.info);

        let devices = platform.get_gpu_devices().unwrap();
        let device = devices[0];
        println!("{:#?}", device.info().unwrap());

        // Create an OpenCL context
        let context = Context::new(&[device]).unwrap();
        println!("{:#?}", context.info().unwrap());

        // Create a command queue
        let command_queue = CommandQueue::new(
            &context,
            &device,
            CL_QUEUE_PROFILING_ENABLE as cl_queue_properties,
        )
        .unwrap();
        println!("{:#?}", command_queue.get_info().unwrap());

        // Create memory buffers on the device for each vector
        let input_a_buf: Buffer<i32> =
            Buffer::new(&context, CL_MEM_READ_ONLY as cl_mem_flags, list_size).unwrap();
        println!("{:#?}", input_a_buf.info().unwrap());

        let input_b_buf: Buffer<i32> =
            Buffer::new(&context, CL_MEM_READ_ONLY as cl_mem_flags, list_size).unwrap();

        let output_c_buf: Buffer<i32> =
            Buffer::new(&context, CL_MEM_WRITE_ONLY as cl_mem_flags, list_size).unwrap();

        // Copy the lists A and B to their respective memory buffers
        unsafe {
            command_queue
                .enqueue_write_buffer(&input_a_buf, CL_TRUE, &input_a, &[])
                .unwrap();

            command_queue
                .enqueue_write_buffer(&input_b_buf, CL_TRUE, &input_b, &[])
                .unwrap();
        }

        // Create a program from the kernel source
        let program = Program::new(&context, PROGRAM_SRC).unwrap();

        // Build the program
        // * is done inside the Program::new(...)

        println!("{:#?}", program.get_info().unwrap());

        let program_build_info = program.get_build_info(&device).unwrap();
        println!(
            "global_variable_total_size: {}",
            program_build_info.global_variable_total_size
        );
        println!(
            "log len: {} -> {}",
            program_build_info.log.len(),
            program_build_info.log
        );

        // Create the OpenCL kernel
        let mut kernel = Kernel::new(&program, "vecAdd").unwrap();

        println!("{:#?}", kernel.get_info().unwrap());

        println!("{:#?}", kernel.get_work_group_info(&device).unwrap());

        // Set the arguments of the kernel
        unsafe {
            kernel.set_arg(&input_a_buf).unwrap();
            kernel.set_arg(&input_b_buf).unwrap();
            kernel.set_arg(&output_c_buf).unwrap();
        }

        // Execute the OpenCL kernel on the list
        let global_work_size = list_size; // Process the entire lists
        let local_work_size = 64; // Process in groups of 64

        unsafe {
            kernel
                .enqueue_nd_range_kernel(
                    &command_queue,
                    &[],
                    &[global_work_size],
                    &[local_work_size],
                    &[],
                )
                .unwrap();
        };

        // Read the memory buffer C on the device to the local variable C

        let mut output_c: Vec<i32> = vec![0; list_size];
        unsafe {
            command_queue
                .enqueue_read_buffer(&output_c_buf, CL_TRUE, &mut output_c, &[])
                .unwrap();
        };

        // Display the result to the screen
        for i in 0..list_size {
            println!(
                "index: {} -> A ({}) + B ({}) = C ({})",
                i, input_a[i], input_b[i], output_c[i]
            );
        }

        // Clean up
        // TODO command_queue.flush()
        // TODO command_queue.finish()

        // Kernel - drop
        // Program - drop

        // input_a_buf - drop
        // input_b_buf - drop
        // output_c_buf - drop

        // CommandQueue - drop

        // Context - drop

        assert!(output_c.iter().all(|&x| x == list_size as i32));
    }

    // TODO add sample code https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_sample_code_5
}
