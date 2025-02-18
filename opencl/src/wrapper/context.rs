//! # Opencl context, command_queue safe wrapper (FFI - Foreign Function Interface)
//!
//! Context
//!
//! ...
//!
//! CommandQueue
//!
//! ...
//!

use crate::error::OclResult;
use crate::unsafe_wrapper::{
    cl_create_command_queue_with_properties, cl_create_context, cl_enqueue_read_buffer,
    cl_enqueue_write_buffer, cl_get_command_queue_info, cl_get_context_info,
    cl_release_command_queue, cl_release_context, CommandQueueInfo, ContextInfo,
};
use crate::wrapper::memory::Buffer;
use crate::wrapper::platform::Device;
use opencl_sys::bindings::{
    cl_bool, cl_command_queue, cl_context, cl_device_id, cl_event, cl_queue_properties,
    CL_QUEUE_ON_DEVICE,
};

#[derive(Debug, PartialEq)]
pub struct Context {
    cl_ctx: cl_context,
    cl_device_ids: Vec<cl_device_id>,
}

impl Context {
    pub fn new(devices: &[Device]) -> OclResult<Self> {
        let cl_device_ids: Vec<_> = devices.iter().map(|d| d.get_cl_device_id()).collect();
        // SAFETY: Device (Struct)eEnsures (in most scenarios) that the device reference is valid.
        let cl_ctx = unsafe { cl_create_context(&cl_device_ids)? };

        Ok(Self {
            cl_ctx,
            cl_device_ids,
        })
    }

    pub fn get_cl_context(&self) -> cl_context {
        self.cl_ctx
    }

    pub fn get_cl_device_ids(&self) -> &[cl_device_id] {
        &self.cl_device_ids
    }

    pub fn info(&self) -> OclResult<ContextInfo> {
        // SAFETY: ...
        unsafe { cl_get_context_info(self.cl_ctx) }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: ...
        unsafe { cl_release_context(self.cl_ctx).expect("Error: clReleaseContext") };
    }
}

#[derive(Debug)]
pub struct CommandQueue {
    cl_cmd_queue: cl_command_queue,
}

impl CommandQueue {
    pub fn new(
        context: &Context,
        device: &Device,
        queue_property: cl_queue_properties,
    ) -> OclResult<Self> {
        // SAFETY: ...
        let cl_cmd_queue = unsafe {
            cl_create_command_queue_with_properties(
                context.get_cl_context(),
                device.get_cl_device_id(),
                queue_property,
            )?
        };

        Ok(Self { cl_cmd_queue })
    }

    pub fn get_cl_command_queue(&self) -> cl_command_queue {
        self.cl_cmd_queue
    }

    pub fn get_info(&self) -> OclResult<CommandQueueInfo> {
        // SAFETY: ...
        unsafe { cl_get_command_queue_info(self.cl_cmd_queue) }
    }

    /// ...
    ///
    /// # Safety
    ///
    /// TODO safety function explain
    pub unsafe fn enqueue_read_buffer<T>(
        &self,
        buffer: &Buffer<T>,
        blocking_read: cl_bool,
        data: &mut [T],
        event_wait_list: &[cl_event],
    ) -> OclResult<cl_event> {
        cl_enqueue_read_buffer(
            self.cl_cmd_queue,
            buffer.get_cl_mem(),
            blocking_read,
            data,
            event_wait_list,
        )
    }

    /// ...
    ///
    /// # Safety
    ///
    /// TODO safety function explain
    pub unsafe fn enqueue_write_buffer<T>(
        &self,
        buffer: &Buffer<T>,
        blocking_write: cl_bool,
        data: &[T],
        event_wait_list: &[cl_event],
    ) -> OclResult<cl_event> {
        cl_enqueue_write_buffer(
            self.cl_cmd_queue,
            buffer.get_cl_mem(),
            blocking_write,
            data,
            event_wait_list,
        )
    }
}

impl Drop for CommandQueue {
    fn drop(&mut self) {
        unsafe {
            cl_release_command_queue(self.cl_cmd_queue).expect("Error: clReleaseCommandQueue")
        };
    }
}

///
/// creates a command queue with the CL_QUEUE_ON_DEVICE property always set
///
/// ```rust no_run
/// use opencl::utils::{EXAMPLE_KERNEL_NAME, EXAMPLE_PROGRAM_SOURCE};
/// use opencl::wrapper::context::{Context, DeviceCommandQueue};
/// use opencl::wrapper::platform::Platform;
/// use opencl::wrapper::program::{Kernel, Program};
///
/// let platform = Platform::first().unwrap();
///
/// let devices = platform.get_gpu_devices().unwrap();
/// let device = devices[0];
///
/// let context = Context::new(&[device]).unwrap();
///
/// let device_command_queue = DeviceCommandQueue::new(&context, &device);
///
/// let program = Program::new(&context, EXAMPLE_PROGRAM_SOURCE).unwrap();
///
/// let kernel = Kernel::new(&program, EXAMPLE_KERNEL_NAME).unwrap();
///
/// // prevents this queue from being used in contexts where having that property (CL_QUEUE_ON_DEVICE) causes an error
/// // (when using the wrapper)
///
/// // unsafe {
/// //    kernel.enqueue_nd_range_kernel(
/// //        &device_command_queue, // error
/// //        &[],
/// //        &[32],
/// //        &[32],
/// //        &[]
/// //    )
/// // }
///
/// ```
#[derive(Debug)]
pub struct DeviceCommandQueue {
    cl_cmd_queue: cl_command_queue,
}

impl DeviceCommandQueue {
    pub fn new(context: &Context, device: &Device) -> OclResult<Self> {
        // SAFETY: ...
        let cl_cmd_queue = unsafe {
            cl_create_command_queue_with_properties(
                context.get_cl_context(),
                device.get_cl_device_id(),
                CL_QUEUE_ON_DEVICE as cl_queue_properties,
            )?
        };

        Ok(Self { cl_cmd_queue })
    }

    pub fn get_cl_command_queue(&self) -> cl_command_queue {
        self.cl_cmd_queue
    }
}

impl Drop for DeviceCommandQueue {
    fn drop(&mut self) {
        unsafe {
            cl_release_command_queue(self.cl_cmd_queue)
                .expect("Error: device - clReleaseCommandQueue")
        };
    }
}

#[cfg(test)]
mod tests_context {
    use super::*;
    use crate::error::OclError;
    use crate::unsafe_wrapper::{
        cl_create_command_queue_with_properties, cl_get_device_ids, cl_get_platform_ids,
    };
    use opencl_sys::bindings::{cl_queue_properties, CL_INVALID_DEVICE, CL_QUEUE_PROFILING_ENABLE};

    #[test]
    fn test_context_new() {
        let platforms = cl_get_platform_ids().unwrap();
        let platform_id = platforms[0];

        let devices = unsafe { cl_get_device_ids(platform_id).unwrap() };
        let device_id = devices[0];

        // valid device reference
        let valid_device = Device::new(device_id).unwrap();

        let context = Context::new(&[valid_device]).unwrap();
        let context_info = unsafe { cl_get_context_info(context.get_cl_context()) }.unwrap();

        assert_eq!(context_info.reference_count, 1);
        assert_eq!(context_info.num_devices, 1);

        // invalid device reference
        let device_id = std::ptr::null_mut() as *mut _ as cl_device_id;
        let invalid_device = Device::create(device_id, 256);

        let result = Context::new(&[valid_device, invalid_device]);
        assert_eq!(result, Err(OclError::Code(CL_INVALID_DEVICE)));
    }

    #[test]
    fn test_context_drop() {
        let platforms = cl_get_platform_ids().unwrap();
        let platform_id = platforms[0];

        let devices = unsafe { cl_get_device_ids(platform_id).unwrap() };
        let device_id = devices[0];

        // valid device reference
        let valid_device = Device::new(device_id).unwrap();

        let context = Context::new(&[valid_device]).unwrap();
        let context_info = unsafe { cl_get_context_info(context.get_cl_context()) }.unwrap();

        assert_eq!(context_info.reference_count, 1);
        assert_eq!(context_info.num_devices, 1);

        // not error after drop context
        // unsafe {
        //     cl_create_command_queue_with_properties(
        //         context.get_cl_context(),
        //         device_id,
        //         CL_QUEUE_PROFILING_ENABLE as cl_queue_properties,
        //     ).unwrap();
        // }

        let cl_ctx_clone = context.get_cl_context();
        drop(context);

        // error
        let context_info = unsafe { cl_get_context_info(cl_ctx_clone).unwrap() };
        println!("{:?}", context_info);

        assert_ne!(context_info.reference_count, 1);
        // assert_ne!(context_info.num_devices, 1);

        let result = unsafe {
            cl_create_command_queue_with_properties(
                cl_ctx_clone,
                device_id,
                CL_QUEUE_PROFILING_ENABLE as cl_queue_properties,
            )
        };
        assert_eq!(result, Err(OclError::Code(CL_INVALID_DEVICE)));
    }
}

#[cfg(test)]
mod tests_command_queue {
    use super::*;
    use crate::error::OclError;
    use crate::unsafe_wrapper::{
        cl_build_program, cl_create_buffer, cl_create_kernel, cl_create_program_with_source,
        cl_enqueue_nd_range_kernel, cl_get_command_queue_info, cl_get_device_ids,
        cl_get_platform_ids, cl_set_kernel_arg,
    };
    use opencl_sys::bindings::{
        cl_kernel, cl_mem, cl_mem_flags, cl_queue_properties, CL_INVALID_COMMAND_QUEUE,
        CL_INVALID_DEVICE_QUEUE, CL_MEM_READ_ONLY, CL_QUEUE_ON_DEVICE, CL_QUEUE_PROFILING_ENABLE,
        CL_TRUE,
    };

    const PROGRAM_SRC: &str = r#"
        __kernel void exampleNoArgs() {
            int id = get_global_id(0);
            // pass
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

    unsafe fn create_opencl_context() -> (Context, Device, cl_kernel, cl_kernel, cl_mem, Buffer<i32>)
    {
        let platforms = cl_get_platform_ids().unwrap();
        let platform_id = platforms[0];

        let devices = cl_get_device_ids(platform_id).unwrap();
        let device_id = devices[0];

        let device = Device::new(device_id).unwrap();

        let context = Context::new(&[device]).unwrap();

        let program = cl_create_program_with_source(context.get_cl_context(), PROGRAM_SRC).unwrap();

        cl_build_program(program, &[device_id]).unwrap();

        let kernel_example_no_args = cl_create_kernel(program, "exampleNoArgs").unwrap();

        let kernel_command_queue_args = cl_create_kernel(program, "commandQueueArgs").unwrap();

        let buf_size = 32 * std::mem::size_of::<i32>();
        let buffer = cl_create_buffer(
            context.get_cl_context(),
            CL_MEM_READ_ONLY as cl_mem_flags,
            buf_size,
        )
        .unwrap();

        let cl_buffer: Buffer<i32> =
            Buffer::new(&context, CL_MEM_READ_ONLY as cl_mem_flags, 32).unwrap();

        (
            context,
            device,
            kernel_example_no_args,
            kernel_command_queue_args,
            buffer,
            cl_buffer,
        )
    }

    #[test]
    fn test_command_queue_new() {
        let platforms = cl_get_platform_ids().unwrap();
        let platform_id = platforms[0];

        let devices = unsafe { cl_get_device_ids(platform_id).unwrap() };
        let device_id = devices[0];

        let device = Device::new(device_id).unwrap();

        // valid context reference
        let context = Context::new(&[device]).unwrap();

        let command_queue = CommandQueue::new(
            &context,
            &device,
            CL_QUEUE_PROFILING_ENABLE as cl_queue_properties,
        )
        .unwrap();

        let command_queue_info =
            unsafe { cl_get_command_queue_info(command_queue.get_cl_command_queue()).unwrap() };

        assert_eq!(command_queue_info.reference_count, 1);
    }

    #[test]
    fn test_command_queue_drop() {
        let platforms = cl_get_platform_ids().unwrap();
        let platform_id = platforms[0];

        let devices = unsafe { cl_get_device_ids(platform_id).unwrap() };
        let device_id = devices[0];

        let device = Device::new(device_id).unwrap();

        // valid context reference
        let context = Context::new(&[device]).unwrap();

        let command_queue = CommandQueue::new(
            &context,
            &device,
            CL_QUEUE_PROFILING_ENABLE as cl_queue_properties,
        )
        .unwrap();
        let cmq = command_queue.get_cl_command_queue();

        drop(command_queue);

        let command_queue_info = unsafe { cl_get_command_queue_info(cmq).unwrap() };

        assert_ne!(command_queue_info.reference_count, 1);
    }

    #[test]
    fn test_command_queue_cases() {
        let (context, device, kernel_example_no_args, kernel_command_queue_args, buffer, cl_buffer) =
            unsafe { create_opencl_context() };

        let command_queue = CommandQueue::new(
            &context,
            &device,
            CL_QUEUE_PROFILING_ENABLE as cl_queue_properties,
        )
        .unwrap();

        unsafe {
            // enqueue buffer
            let input = vec![10; 32];

            cl_enqueue_write_buffer(
                command_queue.get_cl_command_queue(),
                buffer,
                CL_TRUE,
                &input,
                &[],
            )
            .unwrap();

            command_queue
                .enqueue_write_buffer(&cl_buffer, CL_TRUE, &input, &[])
                .unwrap();

            // enqueue nd range kernel
            cl_enqueue_nd_range_kernel(
                command_queue.get_cl_command_queue(),
                kernel_example_no_args,
                &[],
                &[32],
                &[32],
                &[],
            )
            .unwrap();

            // enqueue nd range kernel with command queue arg

            let result = cl_set_kernel_arg(
                kernel_command_queue_args,
                0,
                &command_queue.get_cl_command_queue(),
            );
            assert_eq!(result, Err(OclError::Code(CL_INVALID_DEVICE_QUEUE)));
        }
    }

    #[test]
    fn test_command_queue_on_device_cases() {
        let (context, device, kernel_example_no_args, kernel_command_queue_args, buffer, cl_buffer) =
            unsafe { create_opencl_context() };

        let command_queue =
            CommandQueue::new(&context, &device, CL_QUEUE_ON_DEVICE as cl_queue_properties)
                .unwrap();

        unsafe {
            // enqueue buffer
            let input = vec![10; 32];

            let result = cl_enqueue_write_buffer(
                command_queue.get_cl_command_queue(),
                buffer,
                CL_TRUE,
                &input,
                &[],
            );
            assert_eq!(result, Err(OclError::Code(CL_INVALID_COMMAND_QUEUE)));

            let result = command_queue.enqueue_write_buffer(&cl_buffer, CL_TRUE, &input, &[]);
            assert_eq!(result, Err(OclError::Code(CL_INVALID_COMMAND_QUEUE)));

            // enqueue nd range kernel
            let result = cl_enqueue_nd_range_kernel(
                command_queue.get_cl_command_queue(),
                kernel_example_no_args,
                &[],
                &[32],
                &[32],
                &[],
            );
            assert_eq!(result, Err(OclError::Code(CL_INVALID_COMMAND_QUEUE)));

            // enqueue nd range kernel with command queue arg

            let result = cl_set_kernel_arg(
                kernel_command_queue_args,
                0,
                &command_queue.get_cl_command_queue(),
            );
            assert_eq!(result, Ok(()));
        }
    }
}

#[cfg(test)]
mod tests_device_command_queue {
    use super::*;
    use crate::error::OclError;
    use crate::unsafe_wrapper::{
        cl_build_program, cl_create_buffer, cl_create_kernel, cl_create_program_with_source,
        cl_enqueue_nd_range_kernel, cl_get_command_queue_info, cl_get_device_ids,
        cl_get_platform_ids, cl_set_kernel_arg,
    };
    use opencl_sys::bindings::{
        cl_kernel, cl_mem, cl_mem_flags, CL_INVALID_COMMAND_QUEUE, CL_MEM_READ_ONLY, CL_TRUE,
    };

    const PROGRAM_SRC: &str = r#"
        __kernel void exampleNoArgs() {
            int id = get_global_id(0);
            // pass
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

    unsafe fn create_opencl_context() -> (Context, Device, cl_kernel, cl_kernel, cl_mem, Buffer<i32>)
    {
        let platforms = cl_get_platform_ids().unwrap();
        let platform_id = platforms[0];

        let devices = cl_get_device_ids(platform_id).unwrap();
        let device_id = devices[0];

        let device = Device::new(device_id).unwrap();

        let context = Context::new(&[device]).unwrap();

        let program = cl_create_program_with_source(context.get_cl_context(), PROGRAM_SRC).unwrap();

        cl_build_program(program, &[device_id]).unwrap();

        let kernel_example_no_args = cl_create_kernel(program, "exampleNoArgs").unwrap();

        let kernel_command_queue_args = cl_create_kernel(program, "commandQueueArgs").unwrap();

        let buf_size = 32 * std::mem::size_of::<i32>();
        let buffer = cl_create_buffer(
            context.get_cl_context(),
            CL_MEM_READ_ONLY as cl_mem_flags,
            buf_size,
        )
        .unwrap();

        let cl_buffer: Buffer<i32> =
            Buffer::new(&context, CL_MEM_READ_ONLY as cl_mem_flags, 32).unwrap();

        (
            context,
            device,
            kernel_example_no_args,
            kernel_command_queue_args,
            buffer,
            cl_buffer,
        )
    }

    #[test]
    fn test_device_command_queue_new() {
        let platforms = cl_get_platform_ids().unwrap();
        let platform_id = platforms[0];

        let devices = unsafe { cl_get_device_ids(platform_id).unwrap() };
        let device_id = devices[0];

        let device = Device::new(device_id).unwrap();

        // valid context reference
        let context = Context::new(&[device]).unwrap();

        let command_queue = DeviceCommandQueue::new(&context, &device).unwrap();

        let command_queue_info =
            unsafe { cl_get_command_queue_info(command_queue.get_cl_command_queue()).unwrap() };

        assert_eq!(command_queue_info.reference_count, 1);
    }

    #[test]
    fn test_device_command_queue_drop() {
        let platforms = cl_get_platform_ids().unwrap();
        let platform_id = platforms[0];

        let devices = unsafe { cl_get_device_ids(platform_id).unwrap() };
        let device_id = devices[0];

        let device = Device::new(device_id).unwrap();

        // valid context reference
        let context = Context::new(&[device]).unwrap();

        let command_queue = DeviceCommandQueue::new(&context, &device).unwrap();
        let cmq = command_queue.get_cl_command_queue();

        drop(command_queue);

        let command_queue_info = unsafe { cl_get_command_queue_info(cmq).unwrap() };

        assert_ne!(command_queue_info.reference_count, 1);
    }

    #[test]
    fn test_device_command_queue_cases() {
        let (context, device, kernel_example_no_args, kernel_command_queue_args, buffer, _) =
            unsafe { create_opencl_context() };

        let command_queue = DeviceCommandQueue::new(&context, &device).unwrap();

        unsafe {
            // enqueue buffer
            let input = vec![10; 32];

            let result = cl_enqueue_write_buffer(
                command_queue.get_cl_command_queue(),
                buffer,
                CL_TRUE,
                &input,
                &[],
            );
            assert_eq!(result, Err(OclError::Code(CL_INVALID_COMMAND_QUEUE)));

            // enqueue nd range kernel
            let result = cl_enqueue_nd_range_kernel(
                command_queue.get_cl_command_queue(),
                kernel_example_no_args,
                &[],
                &[32],
                &[32],
                &[],
            );
            assert_eq!(result, Err(OclError::Code(CL_INVALID_COMMAND_QUEUE)));

            // enqueue nd range kernel with command queue arg

            let result = cl_set_kernel_arg(
                kernel_command_queue_args,
                0,
                &command_queue.get_cl_command_queue(),
            );
            assert_eq!(result, Ok(()));
        }
    }
}
