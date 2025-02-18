//! # Opencl block
//!
//! Function grouping (rust traits) of common operations in opencl function execution
//!
//! ...
//!

use crate::error::OclResult;
use crate::wrapper::context::{CommandQueue, Context, DeviceCommandQueue};
use crate::wrapper::memory::Buffer;
use crate::wrapper::platform::{Device, Platform};
use crate::wrapper::program::{Kernel, Program};
use opencl_sys::bindings::{
    cl_event, cl_int, cl_mem_flags, cl_queue_properties, CL_BLOCKING, CL_MEM_READ_ONLY,
    CL_MEM_WRITE_ONLY, CL_QUEUE_PROFILING_ENABLE, CL_SUCCESS,
};
use utilities::uuid::Uuid;

///
/// Grouping common operations when using the opencl wrapper
///
pub trait OpenclCommonOperation {
    fn get_context(&self) -> &Context;

    fn get_host_command_queue(&self) -> &CommandQueue;

    /// command queue created with CL_QUEUE_ON_DEVICE property
    fn get_device_command_queue_0(&self) -> &DeviceCommandQueue;

    ///
    /// 1 - Create a buffer CL_MEM_READ_ONLY (same capacity as the list to copy in this) (clCreateBuffer)
    ///
    /// 2 - Copy (blocking_write) a list of values (Vec) into this read-only buffer (clEnqueueWriteBuffer)
    ///
    fn blocking_prepare_input_buffer<T>(&self, data: &[T]) -> OclResult<Buffer<T>> {
        let cl_buf = Buffer::new(
            self.get_context(),
            CL_MEM_READ_ONLY as cl_mem_flags,
            data.len(),
        )?;

        let _write_event = unsafe {
            self.get_host_command_queue()
                .enqueue_write_buffer(&cl_buf, CL_BLOCKING, data, &[])?
        };

        Ok(cl_buf)
    }

    /// Create a buffer CL_MEM_WRITE_ONLY (clCreateBuffer)
    fn create_output_buffer<T>(&self, buf_len: usize) -> OclResult<Buffer<T>> {
        let cl_buf = Buffer::new(
            self.get_context(),
            CL_MEM_WRITE_ONLY as cl_mem_flags,
            buf_len,
        )?;

        Ok(cl_buf)
    }

    ///
    /// 1 - Create a vector (rust) to store the result of reading an opencl buffer
    ///
    /// 2 - Read (blocking_read) the contents of an opencl buffer and store the result in a vector (rust) (clEnqueueReadBuffer)
    ///
    fn blocking_enqueue_read_buffer<T: Default + Copy + Clone>(
        &self,
        output_buf_len: usize,
        cl_buffer: &Buffer<T>,
        event_wait_list: &[cl_event],
    ) -> OclResult<Vec<T>> {
        let mut output: Vec<T> = vec![T::default(); output_buf_len];

        let _read_event = unsafe {
            self.get_host_command_queue().enqueue_read_buffer(
                cl_buffer,
                CL_BLOCKING,
                &mut output,
                event_wait_list,
            )?
        };

        Ok(output)
    }

    /// To validate the result of an enqueue_kernel call, made from a kernel,
    /// a simple method can be to save the return of this function in a buffer
    /// and perform the respective validation from the host
    ///
    /// For more information about enqueue_kernel functions:
    /// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_C.html#enqueuing-kernels
    ///
    /// Given a buffer that stores the results of calling the enqueue_kernel function,
    /// this method does the following:
    ///
    /// 1 - With the length provided with the *enqueue_kernel_output_len parameter,
    /// create a vector (rust) to store the result of reading from an opencl buffer provided with the parameter *enqueue_kernel_output_buf
    ///
    /// 2 - Read (blocking_read) the contents of an enqueue_kernel_output_buf and store the result in a vector (rust) (clEnqueueReadBuffer)
    ///
    /// 3 - Verify that each result is equal to CL_SUCCESS. In case of error it will call panic! macro,
    /// otherwise it will return the vector with the results.
    ///
    fn assert_device_enqueue_kernel(
        &self,
        enqueue_kernel_output_len: usize,
        enqueue_kernel_output_buf: Buffer<cl_int>,
        events: &[cl_event],
    ) -> OclResult<Vec<cl_int>> {
        let mut enqueue_kernel_output = vec![-1; enqueue_kernel_output_len];

        let _read_event = unsafe {
            self.get_host_command_queue().enqueue_read_buffer(
                &enqueue_kernel_output_buf,
                CL_BLOCKING,
                &mut enqueue_kernel_output,
                events,
            )?
        };

        if enqueue_kernel_output
            .iter()
            .any(|&x| x != CL_SUCCESS as i32)
        {
            panic!("error device enqueue kernel: {enqueue_kernel_output:?}");
        }

        Ok(enqueue_kernel_output)
    }

    fn get_devices(&self) -> &Vec<Device>;

    // TODO replace this method for kernel clGetKernelWorkGroupInfo - CL_KERNEL_WORK_GROUP_SIZE
    ///
    /// quick access to the first device (mainly the only one in the program) to make
    /// the call to the method:
    ///
    /// Device.check_local_work_size(n)
    ///
    ///
    /// ```rust
    /// use opencl::wrapper::platform::Platform;
    ///
    /// let device_index = 0; // example radeon rx 6600 device
    ///
    /// let platform = Platform::first().unwrap();
    /// let devices = platform.get_gpu_devices().unwrap();
    ///
    /// // device CL_DEVICE_MAX_WORK_GROUP_SIZE = 256
    /// let device = devices[device_index];
    ///
    /// // work required
    /// let global_work_size = 1024;
    ///
    /// // return 256;
    /// let local_work_size = device.check_local_work_size(global_work_size);
    /// ```
    ///
    /// call method from struct impl this trait
    /// ```rust
    /// use opencl::wrapper::system::{System, OpenclCommonOperation};
    ///
    /// let device_index = 0; // example radeon rx 6600 device
    ///
    /// // work required
    /// let global_work_size = 1024;
    ///
    /// let system = System::new(device_index, "// source").unwrap();
    ///
    /// // return 256;
    /// let local_work_size = system
    ///     .first_device_check_local_work_size(global_work_size);
    ///
    /// ```
    fn first_device_check_local_work_size(&self, global_work_size: usize) -> usize {
        let device = self.get_devices().first().unwrap();
        device.check_local_work_size(global_work_size)
    }

    fn get_program(&self) -> &Program;

    fn create_kernel(&self, name: &str) -> OclResult<Kernel> {
        Kernel::new(self.get_program(), name)
    }

    // testing
    fn initialize_memory(&self) -> OclResult<()> {
        let program_info = self.get_program().get_info()?;
        let first_kernel_name = program_info
            .kernel_names
            .first()
            .expect("program kernel_names is empty");
        self.create_kernel(first_kernel_name)?;
        Ok(())
    }
}

///
/// Structure that stores the complete initialization of a basic opencl program,
/// intended for quick testing (just one device)
///
/// from
/// ```rust
///
/// use opencl_sys::bindings::{cl_queue_properties, CL_QUEUE_PROFILING_ENABLE};
/// use opencl::wrapper::context::{CommandQueue, Context, DeviceCommandQueue};
/// use opencl::wrapper::platform::Platform;
/// use opencl::wrapper::program::Program;
///
///  // Get platform and device information
///  let platform = Platform::first().unwrap();
///
///  let devices = platform.get_gpu_devices().unwrap();
///  // just one device
///  let device = devices[0];
///
///  // Create an OpenCL context
///  let context = Context::new(&[device]).unwrap();
///
///  // Create a command queue
///  let host_command_queue = CommandQueue::new(
///             &context,
///             &device,
///             CL_QUEUE_PROFILING_ENABLE as cl_queue_properties,
///         ).unwrap();
///
///  // in case of using enqueue_kernel
///  let device_command_queue = DeviceCommandQueue::new(&context, &device).unwrap();
///
///  // Create and build program from the kernel source
///  let example_src: &str = r#"
///         __kernel void vecAdd(
///             __global int *input_a,
///             __global int *input_b,
///             __global int *output_c
///         ) {
///             int id = get_global_id(0);
///
///             output_c[id] = input_a[id] + input_b[id];
///         }
///
///     "#;
///  let program = Program::new(&context, example_src).unwrap();
///
/// ```
///
/// to
///
/// ```rust
///
///  use opencl_sys::bindings::{cl_mem_flags, CL_MEM_WRITE_ONLY};
///  use opencl::wrapper::system::{System, OpenclCommonOperation};
///  use opencl::wrapper::memory::Buffer;
///
///  let example_src: &str = r#"
///         __kernel void vecAdd(
///             __global int *input_a,
///             __global int *input_b,
///             __global int *output_c
///         ) {
///             int id = get_global_id(0);
///
///             output_c[id] = input_a[id] + input_b[id];
///         }
///
///     "#;
///  // create Platform, Device, Context, CommandQueue, Program
///  let system = System::new(0, example_src).unwrap();
///
///  // access each element
///  let context = &system.context;
///
///  // do something ...
///  let buffer: Buffer<i32> = Buffer::new(
///      context,
///      CL_MEM_WRITE_ONLY as cl_mem_flags,
///      32,
///  ).unwrap();
///
///  // or use a common operation like (create buffer with CL_MEM_WRITE_ONLY)
///  let buffer: Buffer<i32> = system.create_output_buffer(32).unwrap();
///
/// ```
///
#[derive(Debug)]
pub struct System {
    id: Uuid,
    pub platform: Platform,
    pub devices: Vec<Device>,
    pub context: Context,
    /// required to run kernel, read and write cl buffers
    pub host_command_queue: CommandQueue,
    /// required to run kernel in device side (enqueue_kernel)
    pub device_command_queue_0: DeviceCommandQueue,
    /// required to create kernel
    pub program: Program,
}

impl System {
    pub fn new(device_index: usize, program_source: &str) -> OclResult<Self> {
        let platform = Platform::first()?;

        let devices = platform.get_gpu_devices()?;

        let device = devices[device_index];
        let info = device.info()?;
        println!("Device: ({}) - {}", info.name, info.board_name);

        let context = Context::new(&[device])?;

        let host_command_queue = CommandQueue::new(
            &context,
            &device,
            CL_QUEUE_PROFILING_ENABLE as cl_queue_properties,
        )?;

        // let device_command_queue_0 =
        //     CommandQueue::new(&context, &device, CL_QUEUE_ON_DEVICE as cl_queue_properties)?;
        let device_command_queue_0 = DeviceCommandQueue::new(&context, &device)?;

        let program = Program::new(&context, program_source)?;
        println!("program {:?}", program.get_cl_program());

        let id = Uuid::new_v4();

        Ok(Self {
            id,
            platform,
            devices,
            context,
            host_command_queue,
            device_command_queue_0,
            program,
        })
    }

    pub fn get_id(&self) -> Uuid {
        self.id
    }
}

unsafe impl Send for System {}

unsafe impl Sync for System {}

impl OpenclCommonOperation for System {
    fn get_context(&self) -> &Context {
        &self.context
    }

    fn get_host_command_queue(&self) -> &CommandQueue {
        &self.host_command_queue
    }

    fn get_device_command_queue_0(&self) -> &DeviceCommandQueue {
        &self.device_command_queue_0
    }

    fn get_devices(&self) -> &Vec<Device> {
        &self.devices
    }

    fn get_program(&self) -> &Program {
        &self.program
    }
}

#[cfg(test)]
mod tests_system {
    use super::*;

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
        // * is done inside the System::new(...)

        // Create an OpenCL context
        // * is done inside the System::new(...)

        // Create a command queue
        // * is done inside the System::new(...)

        let system = System::new(0, PROGRAM_SRC).unwrap();

        // Create memory buffers on the device for each vector
        // * is done inside the System.blocking_prepare_input_buffer(...)

        // Copy the lists A and B to their respective memory buffers
        // * is done inside the System.blocking_prepare_input_buffer(...)

        let input_a_buf = system.blocking_prepare_input_buffer(&input_a).unwrap();
        let input_b_buf = system.blocking_prepare_input_buffer(&input_b).unwrap();

        let output_c_buf = system.create_output_buffer::<i32>(list_size).unwrap();

        // Create a program from the kernel source
        // * is done inside the System::new(...)

        // Build the program
        // * is done inside the System::new(...)

        // Create the OpenCL kernel
        let mut kernel = system.create_kernel("vecAdd").unwrap();

        let _event = unsafe {
            // Set the arguments of the kernel
            kernel.set_arg(&input_a_buf).unwrap();
            kernel.set_arg(&input_b_buf).unwrap();
            kernel.set_arg(&output_c_buf).unwrap();

            // Execute the OpenCL kernel on the list
            let global_work_size = list_size; // Process the entire lists
            let local_work_size = 64; // Process in groups of 64

            kernel.enqueue_nd_range_kernel(
                system.get_host_command_queue(),
                &[],
                &[global_work_size],
                &[local_work_size],
                &[],
            )
        };

        // Read the memory buffer C on the device to the local variable C
        let output_c = system
            .blocking_enqueue_read_buffer(list_size, &output_c_buf, &[])
            .unwrap();

        // Display the result to the screen
        for i in 0..list_size {
            println!(
                "index: {} -> A ({}) + B ({}) = C ({})",
                i, input_a[i], input_b[i], output_c[i]
            );
        }

        // Clean up
        // * it is done when drop its call in System
        // Program.drop, Context.drop, ...

        assert!(output_c.iter().all(|&x| x == list_size as i32));
    }
}
