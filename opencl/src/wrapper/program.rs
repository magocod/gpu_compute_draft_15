//! # Opencl program, kernel safe wrapper (FFI - Foreign Function Interface)
//!
//! Program
//!
//! ...
//!
//! Kernel
//!
//! ...

use crate::error::{OclError, OclResult, CL_WRAPPER_EMPTY_PROGRAM_SOURCE};
use crate::unsafe_wrapper::{
    cl_build_program, cl_create_kernel, cl_create_program_with_source, cl_enqueue_nd_range_kernel,
    cl_get_kernel_info, cl_get_kernel_work_group_info, cl_get_program_build_info,
    cl_get_program_info, cl_release_kernel, cl_release_program, cl_set_kernel_arg, KernelInfo,
    KernelWorkGroupInfo, ProgramBuildInfo, ProgramInfo,
};
use crate::wrapper::context::{CommandQueue, Context};
use crate::wrapper::platform::Device;
use opencl_sys::bindings::{cl_event, cl_kernel, cl_program, cl_uint};

#[derive(Debug, PartialEq)]
pub struct Program {
    cl_prog: cl_program,
    src_len: usize,
}

impl Program {
    pub fn new(context: &Context, source: &str) -> OclResult<Self> {
        // fatal error (signal: 11, SIGSEGV: invalid memory reference)
        if source.is_empty() {
            return Err(OclError::Wrapper(CL_WRAPPER_EMPTY_PROGRAM_SOURCE));
        }

        let cl_prog = unsafe {
            let prog = cl_create_program_with_source(context.get_cl_context(), source)?;

            cl_build_program(prog, context.get_cl_device_ids())?;

            prog
        };

        Ok(Self {
            cl_prog,
            src_len: source.len(),
        })
    }

    pub fn get_cl_program(&self) -> cl_program {
        self.cl_prog
    }

    pub fn get_info(&self) -> OclResult<ProgramInfo> {
        let program_info = unsafe { cl_get_program_info(self.cl_prog, self.src_len)? };

        Ok(program_info)
    }

    pub fn get_build_info(&self, device: &Device) -> OclResult<ProgramBuildInfo> {
        let program_info =
            unsafe { cl_get_program_build_info(self.cl_prog, device.get_cl_device_id())? };

        Ok(program_info)
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        unsafe { cl_release_program(self.cl_prog).expect("Error: clReleaseProgram") };
    }
}

#[derive(Debug, PartialEq)]
pub struct Kernel {
    cl_k: cl_kernel,
    count_args: cl_uint,
}

impl Kernel {
    pub fn new(program: &Program, name: &str) -> OclResult<Self> {
        let cl_k = unsafe { cl_create_kernel(program.get_cl_program(), name)? };

        Ok(Self {
            cl_k,
            count_args: 0,
        })
    }

    pub fn get_cl_kernel(&self) -> cl_kernel {
        self.cl_k
    }

    pub fn get_info(&self) -> OclResult<KernelInfo> {
        let kernel_info = unsafe { cl_get_kernel_info(self.cl_k)? };

        Ok(kernel_info)
    }

    pub fn get_work_group_info(&self, device: &Device) -> OclResult<KernelWorkGroupInfo> {
        let kernel_info =
            unsafe { cl_get_kernel_work_group_info(self.cl_k, device.get_cl_device_id())? };

        Ok(kernel_info)
    }

    /// ...
    ///
    ///
    /// # Safety
    ///
    /// TODO safety function explain
    pub unsafe fn set_arg_to<T>(&self, arg_index: cl_uint, value: &T) -> OclResult<()> {
        unsafe {
            cl_set_kernel_arg(self.cl_k, arg_index, value)?;
        }

        Ok(())
    }

    /// ...
    ///
    ///
    /// # Safety
    ///
    /// TODO safety function explain
    pub unsafe fn set_arg<T>(&mut self, value: &T) -> OclResult<()> {
        self.set_arg_to(self.count_args, value)?;
        self.count_args += 1;

        Ok(())
    }

    /// ...
    ///
    ///
    /// # Safety
    ///
    /// TODO safety function explain
    pub unsafe fn enqueue_nd_range_kernel(
        &self,
        command_queue: &CommandQueue,
        global_work_offsets: &[usize],
        global_work_sizes: &[usize],
        local_work_sizes: &[usize],
        event_wait_list: &[cl_event],
    ) -> OclResult<cl_event> {
        cl_enqueue_nd_range_kernel(
            command_queue.get_cl_command_queue(),
            self.cl_k,
            global_work_offsets,
            global_work_sizes,
            local_work_sizes,
            event_wait_list,
        )
    }

    /// ...
    ///
    ///
    /// # Safety
    ///
    /// TODO safety function explain
    pub unsafe fn enqueue_nd_range_kernel_dim_1(
        &self,
        command_queue: &CommandQueue,
        global_work_size: usize,
        local_work_size: usize,
        event_wait_list: &[cl_event],
    ) -> OclResult<cl_event> {
        cl_enqueue_nd_range_kernel(
            command_queue.get_cl_command_queue(),
            self.cl_k,
            &[],
            &[global_work_size],
            &[local_work_size],
            event_wait_list,
        )
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        unsafe { cl_release_kernel(self.cl_k).expect("Error: clReleaseKernel") };
    }
}

#[cfg(test)]
mod tests_program {
    use super::*;
    use crate::unsafe_wrapper::{cl_get_device_ids, cl_get_platform_ids};
    use crate::wrapper::platform::Device;

    const VALID_PROGRAM_SRC: &str = r#"
        __kernel void vecAdd(
            __global int *input_a,
            __global int *input_b,
            __global int *output_c
        ) {
            int id = get_global_id(0);
            
            output_c[id] = input_a[id] + input_b[id];
        }
        
        __kernel void example(
            __global int *input_a,
            __global int *output_c
        ) {
            int id = get_global_id(0);
            
            output_c[id] = input_a[id];
        }
    "#;

    const INVALID_PROGRAM_SRC: &str = r#"
        __kernel void vecAdd(
            __global int *input_a,
            __global int *input_b,
            __global int *output_c
        ) {
            int id = get_global_id(0);
            
            output_c[id] = input_a[id] + input[id];
        }
    "#;

    #[test]
    fn test_program_new() {
        let platforms = cl_get_platform_ids().unwrap();
        let platform_id = platforms[0];

        let devices = unsafe { cl_get_device_ids(platform_id).unwrap() };
        let device_id = devices[0];

        let device = Device::new(device_id).unwrap();

        // valid context reference
        let context = Context::new(&[device]).unwrap();

        let program = Program::new(&context, VALID_PROGRAM_SRC).unwrap();

        let program_info =
            unsafe { cl_get_program_info(program.get_cl_program(), VALID_PROGRAM_SRC.len()) }
                .unwrap();
        println!("{:#?}", program_info);

        assert_eq!(
            program_info.kernel_names,
            vec!["example".to_string(), "vecAdd".to_string()]
        );
        assert_eq!(program_info.reference_count, 1);
        assert_eq!(program_info.num_devices, 1);
    }

    #[test]
    #[should_panic]
    fn test_program_invalid_source() {
        let platforms = cl_get_platform_ids().unwrap();
        let platform_id = platforms[0];

        let devices = unsafe { cl_get_device_ids(platform_id).unwrap() };
        let device_id = devices[0];

        let device = Device::new(device_id).unwrap();

        // valid context reference
        let context = Context::new(&[device]).unwrap();

        Program::new(&context, INVALID_PROGRAM_SRC).unwrap();
    }

    #[test]
    fn test_program_empty_source() {
        let platforms = cl_get_platform_ids().unwrap();
        let platform_id = platforms[0];

        let devices = unsafe { cl_get_device_ids(platform_id).unwrap() };
        let device_id = devices[0];

        let device = Device::new(device_id).unwrap();

        // valid context reference
        let context = Context::new(&[device]).unwrap();

        let result = Program::new(&context, "");
        assert_eq!(
            result,
            Err(OclError::Wrapper(CL_WRAPPER_EMPTY_PROGRAM_SOURCE))
        );
    }

    // #[test]
    // fn test_program_drop() {
    //     let platforms = cl_get_platform_ids().unwrap();
    //     let platform_id = platforms[0];
    //
    //     let devices = unsafe { cl_get_device_ids(platform_id).unwrap() };
    //     let device_id = devices[0];
    //
    //     let device = Device::new(device_id).unwrap();
    //
    //     // valid context reference
    //     let context = Context::new(&[device]).unwrap();
    //
    //     let program = Program::new(&context, VALID_PROGRAM_SRC).unwrap();
    //
    //     let prog = program.get_cl_program();
    //     drop(program);
    //
    //     // fatal error
    //     let k = unsafe { cl_create_kernel(prog, "vecAdd") }.unwrap();
    //     println!("{:#?}", k);
    // }
}

#[cfg(test)]
mod tests_kernel {
    use super::*;
    use crate::error::OclError;
    use crate::unsafe_wrapper::{
        cl_create_buffer, cl_create_command_queue_with_properties, cl_enqueue_read_buffer,
        cl_enqueue_write_buffer, cl_get_device_ids, cl_get_platform_ids,
    };
    use crate::wrapper::platform::Device;
    use opencl_sys::bindings::{
        cl_command_queue, cl_context, cl_device_id, cl_int, cl_mem, cl_mem_flags,
        cl_queue_properties, CL_INVALID_ARG_INDEX, CL_INVALID_ARG_SIZE, CL_INVALID_COMMAND_QUEUE,
        CL_INVALID_DEVICE_QUEUE, CL_INVALID_KERNEL_NAME, CL_INVALID_WORK_GROUP_SIZE,
        CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY, CL_QUEUE_ON_DEVICE, CL_QUEUE_PROFILING_ENABLE,
        CL_TRUE,
    };

    const PROGRAM_SRC: &str = r#"
        __kernel void vecAdd(
            const int value,
            __global int *input_a,
            __global int *output_c
        ) {
            int id = get_global_id(0);

            output_c[id] = input_a[id] + value;
        }

        __kernel void increase(
            const int value,
            __global int *output_c
        ) {
            int id = get_global_id(0);

            output_c[id] =+ value;
        }

        __kernel void exampleNoArgs() {
            int id = get_global_id(0);
            // pass
        }

        __kernel void example(
            __global int *input_a,
            __global int *output_c
        ) {
            int id = get_global_id(0);

            output_c[id] = input_a[id];
        }
        
        __kernel void commandQueueArgs(
            queue_t q0
            ) {
            int id = get_global_id(0);
            
            enqueue_kernel(
                q0,
                CLK_ENQUEUE_FLAGS_NO_WAIT,
                ndrange_1D(32, 32),
                ^{
                   int cid = get_global_id(0);
                   // pass
                }
            );
        }
    "#;

    unsafe fn prepare_args(
        context: cl_context,
        device: cl_device_id,
        buf_len: usize,
        input_a_value: i32,
    ) -> (cl_mem, cl_mem, cl_command_queue) {
        let command_queue = cl_create_command_queue_with_properties(
            context,
            device,
            CL_QUEUE_PROFILING_ENABLE as cl_queue_properties,
        )
        .unwrap();

        let input_size = buf_len * std::mem::size_of::<i32>();
        let input_a_buf =
            cl_create_buffer(context, CL_MEM_READ_ONLY as cl_mem_flags, input_size).unwrap();

        let input_a = vec![input_a_value; buf_len];

        let _write_event =
            cl_enqueue_write_buffer(command_queue, input_a_buf, CL_TRUE, &input_a, &[]).unwrap();

        let output_c_buf =
            cl_create_buffer(context, CL_MEM_WRITE_ONLY as cl_mem_flags, input_size).unwrap();

        (input_a_buf, output_c_buf, command_queue)
    }

    unsafe fn assert_kernel_result(
        output_c_buf: cl_mem,
        command_queue: cl_command_queue,
        buf_len: usize,
        expected: Option<Vec<i32>>,
    ) {
        let mut output_c: Vec<i32> = vec![0; buf_len];
        cl_enqueue_read_buffer(command_queue, output_c_buf, CL_TRUE, &mut output_c, &[]).unwrap();

        println!("output_c = {:?}", output_c);
        if let Some(vec_expected) = expected {
            assert_eq!(output_c, vec_expected)
        }
    }

    #[test]
    fn test_kernel_new() {
        let platforms = cl_get_platform_ids().unwrap();
        let platform_id = platforms[0];

        let devices = unsafe { cl_get_device_ids(platform_id).unwrap() };
        let device_id = devices[0];

        let device = Device::new(device_id).unwrap();

        // valid context reference
        let context = Context::new(&[device]).unwrap();

        let program = Program::new(&context, PROGRAM_SRC).unwrap();

        let kernel = Kernel::new(&program, "vecAdd").unwrap();
        let kernel_info = unsafe { cl_get_kernel_info(kernel.get_cl_kernel()).unwrap() };
        println!("{:#?}", kernel_info);

        assert_eq!(
            kernel_info,
            KernelInfo {
                num_args: 3,
                reference_count: 1
            }
        );

        let kernel = Kernel::new(&program, "exampleNoArgs").unwrap();
        let kernel_info = kernel.get_info().unwrap();
        println!("{:#?}", kernel_info);

        assert_eq!(
            kernel_info,
            KernelInfo {
                num_args: 0,
                reference_count: 1
            }
        );

        let result = Kernel::new(&program, "invalid_name");
        assert_eq!(result, Err(OclError::Code(CL_INVALID_KERNEL_NAME)));
    }

    #[test]
    fn test_kernel_drop() {
        let platforms = cl_get_platform_ids().unwrap();
        let platform_id = platforms[0];

        let devices = unsafe { cl_get_device_ids(platform_id).unwrap() };
        let device_id = devices[0];

        let device = Device::new(device_id).unwrap();

        // valid context reference
        let context = Context::new(&[device]).unwrap();

        let program = Program::new(&context, PROGRAM_SRC).unwrap();

        let kernel = Kernel::new(&program, "vecAdd").unwrap();
        let cl_k = kernel.get_cl_kernel();
        drop(kernel);

        let kernel_info = unsafe { cl_get_kernel_info(cl_k).unwrap() };
        println!("{:#?}", kernel_info);

        assert_ne!(
            kernel_info,
            KernelInfo {
                num_args: 3,
                reference_count: 1
            }
        );
    }

    #[test]
    fn test_kernel_set_arg() {
        let platforms = cl_get_platform_ids().unwrap();
        let platform_id = platforms[0];

        let devices = unsafe { cl_get_device_ids(platform_id).unwrap() };
        let device_id = devices[0];

        let device = Device::new(device_id).unwrap();

        // valid context reference
        let context = Context::new(&[device]).unwrap();

        let program = Program::new(&context, PROGRAM_SRC).unwrap();

        let value: cl_int = 2;
        let input_a_value = 10;
        let (input_a_buf, output_c_buf, _) = unsafe {
            prepare_args(
                context.get_cl_context(),
                device.get_cl_device_id(),
                32,
                input_a_value,
            )
        };

        unsafe {
            // ok
            let mut kernel = Kernel::new(&program, "vecAdd").unwrap();

            let result = kernel.set_arg(&value);
            assert_eq!(result, Ok(()));
            let result = kernel.set_arg(&input_a_buf);
            assert_eq!(result, Ok(()));
            let result = kernel.set_arg(&output_c_buf);
            assert_eq!(result, Ok(()));

            // error
            let mut kernel = Kernel::new(&program, "vecAdd").unwrap();

            let result = kernel.set_arg(&value);
            assert_eq!(result, Ok(()));
            let result = kernel.set_arg(&input_a_buf);
            assert_eq!(result, Ok(()));
            let result = kernel.set_arg(&output_c_buf);
            assert_eq!(result, Ok(()));
            let result = kernel.set_arg(&output_c_buf);
            assert_eq!(result, Err(OclError::Code(CL_INVALID_ARG_INDEX)));

            // error
            let mut kernel = Kernel::new(&program, "vecAdd").unwrap();

            let result = kernel.set_arg(&input_a_buf);
            assert_eq!(result, Err(OclError::Code(CL_INVALID_ARG_SIZE)));
            kernel.set_arg(&value).unwrap();
            // fatal error
            // let result = kernel.set_arg(&value);
            // assert_eq!(result, Err(OclError::Code(CL_INVALID_ARG_SIZE)));

            // ok, when executed, undefined behavior will occur
            let mut kernel = Kernel::new(&program, "vecAdd").unwrap();

            kernel.set_arg(&value).unwrap();
            let result = kernel.set_arg(&output_c_buf);
            assert_eq!(result, Ok(()));

            // error
            let kernel = Kernel::new(&program, "vecAdd").unwrap();

            let result = kernel.set_arg_to(5, &value);
            assert_eq!(result, Err(OclError::Code(CL_INVALID_ARG_INDEX)));

            // error
            let mut kernel = Kernel::new(&program, "exampleNoArgs").unwrap();

            let result = kernel.set_arg(&value);
            assert_eq!(result, Err(OclError::Code(CL_INVALID_ARG_INDEX)));
        }
    }

    #[test]
    fn test_kernel_enqueue_nd_range_kernel_1_ok() {
        let platforms = cl_get_platform_ids().unwrap();
        let platform_id = platforms[0];

        let devices = unsafe { cl_get_device_ids(platform_id).unwrap() };
        let device_id = devices[0];

        let device = Device::new(device_id).unwrap();

        // valid context reference
        let context = Context::new(&[device]).unwrap();

        let program = Program::new(&context, PROGRAM_SRC).unwrap();

        let buf_len = 32;

        let value: cl_int = 2;
        let input_a_value = 10;

        unsafe {
            let (input_a_buf, output_c_buf, _) = prepare_args(
                context.get_cl_context(),
                device.get_cl_device_id(),
                buf_len,
                input_a_value,
            );

            let command_queue = CommandQueue::new(
                &context,
                &device,
                CL_QUEUE_PROFILING_ENABLE as cl_queue_properties,
            )
            .unwrap();

            // ok
            let mut kernel = Kernel::new(&program, "vecAdd").unwrap();

            let result = kernel.set_arg(&value);
            assert_eq!(result, Ok(()));
            let result = kernel.set_arg(&input_a_buf);
            assert_eq!(result, Ok(()));
            let result = kernel.set_arg(&output_c_buf);
            assert_eq!(result, Ok(()));

            kernel
                .enqueue_nd_range_kernel(&command_queue, &[], &[buf_len], &[buf_len], &[])
                .unwrap();

            assert_kernel_result(
                output_c_buf,
                command_queue.get_cl_command_queue(),
                buf_len,
                Some(vec![input_a_value + value; buf_len]),
            );
        }
    }

    // #[test]
    // fn test_kernel_enqueue_nd_range_kernel_2_fatal_error() {
    //     let platforms = cl_get_platform_ids().unwrap();
    //     let platform_id = platforms[0];
    //
    //     let devices = unsafe { cl_get_device_ids(platform_id).unwrap() };
    //     let device_id = devices[0];
    //
    //     let device = Device::new(device_id).unwrap();
    //
    //     // valid context reference
    //     let context = Context::new(&[device]).unwrap();
    //
    //     let program = Program::new(&context, PROGRAM_SRC).unwrap();
    //
    //     let buf_len = 32;
    //
    //     let value: cl_int = 2;
    //     let input_a_value = 10;
    //
    //     unsafe {
    //         // input_a_buf values = 10
    //         let (input_a_buf, output_c_buf, _) = prepare_args(context.get_cl_context(), device.get_cl_device_id(), buf_len, input_a_value);
    //
    //         let command_queue = CommandQueue::new(&context, &device, CL_QUEUE_PROFILING_ENABLE as cl_queue_properties).unwrap();
    //
    //         // fatal error (global work size bigger than buffer)
    //         let mut kernel = Kernel::new(&program, "vecAdd").unwrap();
    //
    //         let result = kernel.set_arg(&value);
    //         assert_eq!(result, Ok(()));
    //         let result = kernel.set_arg(&input_a_buf);
    //         assert_eq!(result, Ok(()));
    //         let result = kernel.set_arg(&output_c_buf);
    //         assert_eq!(result, Ok(()));
    //
    //             kernel.enqueue_nd_range_kernel(
    //                 &command_queue,
    //                 &[],
    //                 &[1024 * 1024],
    //                 &[buf_len],
    //                 &[]
    //             ).unwrap();
    //
    //     }
    // }

    #[test]
    fn test_kernel_enqueue_nd_range_kernel_3_ok_and_undefined_behavior() {
        let platforms = cl_get_platform_ids().unwrap();
        let platform_id = platforms[0];

        let devices = unsafe { cl_get_device_ids(platform_id).unwrap() };
        let device_id = devices[0];

        let device = Device::new(device_id).unwrap();

        // valid context reference
        let context = Context::new(&[device]).unwrap();

        let program = Program::new(&context, PROGRAM_SRC).unwrap();

        let buf_len = 32;

        let value: cl_int = 2;
        let input_a_value = 10;

        unsafe {
            let (input_a_buf, output_c_buf, _) = prepare_args(
                context.get_cl_context(),
                device.get_cl_device_id(),
                buf_len,
                input_a_value,
            );

            let command_queue = CommandQueue::new(
                &context,
                &device,
                CL_QUEUE_PROFILING_ENABLE as cl_queue_properties,
            )
            .unwrap();

            // ok, when executed, undefined behavior will occur
            let mut kernel = Kernel::new(&program, "vecAdd").unwrap();

            let result = kernel.set_arg(&value);
            assert_eq!(result, Ok(()));
            let result = kernel.set_arg(&output_c_buf);
            assert_eq!(result, Ok(()));
            let result = kernel.set_arg(&input_a_buf);
            assert_eq!(result, Ok(()));

            kernel
                .enqueue_nd_range_kernel(&command_queue, &[], &[buf_len], &[buf_len], &[])
                .unwrap();

            assert_kernel_result(
                output_c_buf,
                command_queue.get_cl_command_queue(),
                buf_len,
                None,
            );
        }
    }

    #[test]
    fn test_kernel_enqueue_nd_range_kernel_4_error_work_group_size() {
        let platforms = cl_get_platform_ids().unwrap();
        let platform_id = platforms[0];

        let devices = unsafe { cl_get_device_ids(platform_id).unwrap() };
        let device_id = devices[0];

        let device = Device::new(device_id).unwrap();

        // valid context reference
        let context = Context::new(&[device]).unwrap();

        let program = Program::new(&context, PROGRAM_SRC).unwrap();

        let buf_len = 32;

        unsafe {
            let command_queue = CommandQueue::new(
                &context,
                &device,
                CL_QUEUE_PROFILING_ENABLE as cl_queue_properties,
            )
            .unwrap();

            // ok
            let kernel = Kernel::new(&program, "exampleNoArgs").unwrap();

            kernel
                .enqueue_nd_range_kernel(&command_queue, &[], &[buf_len], &[buf_len], &[])
                .unwrap();

            // error invalid local work size
            let kernel = Kernel::new(&program, "exampleNoArgs").unwrap();

            let result =
                kernel.enqueue_nd_range_kernel(&command_queue, &[], &[buf_len], &[2048], &[]);
            assert_eq!(result, Err(OclError::Code(CL_INVALID_WORK_GROUP_SIZE)));
        }
    }

    #[test]
    fn test_kernel_enqueue_nd_range_kernel_5_ok_multiple_execution() {
        let platforms = cl_get_platform_ids().unwrap();
        let platform_id = platforms[0];

        let devices = unsafe { cl_get_device_ids(platform_id).unwrap() };
        let device_id = devices[0];

        let device = Device::new(device_id).unwrap();

        // valid context reference
        let context = Context::new(&[device]).unwrap();

        let program = Program::new(&context, PROGRAM_SRC).unwrap();

        let buf_len = 32;

        let value: cl_int = 2;
        let input_a_value = 10;

        unsafe {
            let (input_a_buf, output_c_buf, _) = prepare_args(
                context.get_cl_context(),
                device.get_cl_device_id(),
                buf_len,
                input_a_value,
            );

            let command_queue = CommandQueue::new(
                &context,
                &device,
                CL_QUEUE_PROFILING_ENABLE as cl_queue_properties,
            )
            .unwrap();

            // ok
            let mut kernel = Kernel::new(&program, "vecAdd").unwrap();

            let result = kernel.set_arg(&value);
            assert_eq!(result, Ok(()));
            let result = kernel.set_arg(&input_a_buf);
            assert_eq!(result, Ok(()));
            let result = kernel.set_arg(&output_c_buf);
            assert_eq!(result, Ok(()));

            kernel
                .enqueue_nd_range_kernel(&command_queue, &[], &[buf_len], &[buf_len], &[])
                .unwrap();

            assert_kernel_result(
                output_c_buf,
                command_queue.get_cl_command_queue(),
                buf_len,
                Some(vec![input_a_value + value; buf_len]),
            );

            kernel
                .enqueue_nd_range_kernel(&command_queue, &[], &[buf_len], &[buf_len], &[])
                .unwrap();

            assert_kernel_result(
                output_c_buf,
                command_queue.get_cl_command_queue(),
                buf_len,
                Some(vec![input_a_value + value; buf_len]),
            );
        }
    }

    #[test]
    fn test_kernel_enqueue_nd_range_kernel_6_ok_command_queue() {
        let platforms = cl_get_platform_ids().unwrap();
        let platform_id = platforms[0];

        let devices = unsafe { cl_get_device_ids(platform_id).unwrap() };
        let device_id = devices[0];

        let device = Device::new(device_id).unwrap();

        let context = Context::new(&[device]).unwrap();

        let program = Program::new(&context, PROGRAM_SRC).unwrap();

        unsafe {
            let command_queue = CommandQueue::new(
                &context,
                &device,
                CL_QUEUE_PROFILING_ENABLE as cl_queue_properties,
            )
            .unwrap();

            let device_command_queue =
                CommandQueue::new(&context, &device, CL_QUEUE_ON_DEVICE as cl_queue_properties)
                    .unwrap();

            // ok
            let mut kernel = Kernel::new(&program, "commandQueueArgs").unwrap();

            let q0 = command_queue.get_cl_command_queue();
            let result = kernel.set_arg(&q0);
            assert_eq!(result, Err(OclError::Code(CL_INVALID_DEVICE_QUEUE)));

            let q0 = device_command_queue.get_cl_command_queue();
            let result = kernel.set_arg(&q0);
            assert_eq!(result, Ok(()));

            kernel
                .enqueue_nd_range_kernel(&command_queue, &[], &[32], &[32], &[])
                .unwrap();

            let result =
                kernel.enqueue_nd_range_kernel(&device_command_queue, &[], &[32], &[32], &[]);
            assert_eq!(result, Err(OclError::Code(CL_INVALID_COMMAND_QUEUE)));
        }
    }

    #[test]
    fn test_kernel_enqueue_nd_range_kernel_6_fatal_error_command_queue() {
        let platforms = cl_get_platform_ids().unwrap();
        let platform_id = platforms[0];

        let devices = unsafe { cl_get_device_ids(platform_id).unwrap() };
        let device_id = devices[0];

        let device = Device::new(device_id).unwrap();

        let context = Context::new(&[device]).unwrap();

        let program = Program::new(&context, PROGRAM_SRC).unwrap();

        unsafe {
            let command_queue = CommandQueue::new(
                &context,
                &device,
                CL_QUEUE_PROFILING_ENABLE as cl_queue_properties,
            )
            .unwrap();

            let device_command_queue =
                CommandQueue::new(&context, &device, CL_QUEUE_ON_DEVICE as cl_queue_properties)
                    .unwrap();

            // ok
            let mut kernel = Kernel::new(&program, "commandQueueArgs").unwrap();

            let q0 = device_command_queue;
            let result = kernel.set_arg(&q0);
            assert_eq!(result, Ok(()));

            kernel
                .enqueue_nd_range_kernel(&command_queue, &[], &[32], &[32], &[])
                .unwrap();
        }
    }
}
