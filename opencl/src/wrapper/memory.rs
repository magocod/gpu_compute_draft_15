//! # Opencl buffer, pipe safe wrapper (FFI - Foreign Function Interface)
//!
//! Buffer
//!
//! ...
//!
//! Pipe
//!
//! ...
//!

use crate::error::OclResult;
use crate::unsafe_wrapper::{
    cl_create_buffer, cl_create_pipe, cl_get_mem_object_info, cl_release_mem_object, MemInfo,
};
use crate::wrapper::context::Context;
use opencl_sys::bindings::{cl_mem, cl_mem_flags, cl_uint};
use std::marker::PhantomData;

#[derive(Debug)]
pub struct Buffer<T> {
    mem: cl_mem,
    cl_type: PhantomData<T>,
}

impl<T> Buffer<T> {
    pub fn new(context: &Context, flags: cl_mem_flags, total_elements: usize) -> OclResult<Self> {
        let size = total_elements * std::mem::size_of::<T>();

        let mem = unsafe { cl_create_buffer(context.get_cl_context(), flags, size)? };

        Ok(Self {
            mem,
            cl_type: Default::default(),
        })
    }

    pub fn get_cl_mem(&self) -> cl_mem {
        self.mem
    }

    pub fn info(&self) -> OclResult<MemInfo> {
        // SAFETY: ...
        unsafe { cl_get_mem_object_info(self.mem) }
    }
}

impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        unsafe { cl_release_mem_object(self.mem).expect("Error: Buffer clReleaseMemObject") };
    }
}

#[derive(Debug)]
pub struct Pipe<T> {
    mem: cl_mem,
    cl_type: PhantomData<T>,
}

impl<T> Pipe<T> {
    pub fn new(context: &Context, pipe_max_packets: cl_uint) -> OclResult<Self> {
        let mem = unsafe { cl_create_pipe::<T>(context.get_cl_context(), pipe_max_packets)? };

        Ok(Self {
            mem,
            cl_type: Default::default(),
        })
    }

    pub fn get_cl_mem(&self) -> cl_mem {
        self.mem
    }
}

impl<T> Drop for Pipe<T> {
    fn drop(&mut self) {
        unsafe { cl_release_mem_object(self.mem).expect("Error: Pipe clReleaseMemObject") };
    }
}

// TODO tests_buffer
#[cfg(test)]
mod tests_buffer {
    use super::*;
    use crate::unsafe_wrapper::{cl_get_device_ids, cl_get_platform_ids};
    use crate::wrapper::platform::Device;
    use opencl_sys::bindings::{CL_MEM_OBJECT_BUFFER, CL_MEM_READ_ONLY};

    #[test]
    fn test_buffer_new() {
        let platforms = cl_get_platform_ids().unwrap();
        let platform_id = platforms[0];

        let devices = unsafe { cl_get_device_ids(platform_id).unwrap() };
        let device_id = devices[0];

        let device = Device::new(device_id).unwrap();

        // valid context reference
        let context = Context::new(&[device]).unwrap();

        let buffer: Buffer<i32> =
            Buffer::new(&context, CL_MEM_READ_ONLY as cl_mem_flags, 256).unwrap();

        let mem_info = unsafe { cl_get_mem_object_info(buffer.get_cl_mem()) }.unwrap();
        println!("{:#?}", mem_info);

        assert_eq!(
            mem_info,
            MemInfo {
                mem_type: CL_MEM_OBJECT_BUFFER,
                mem_size: 1024,
                reference_count: 1,
            }
        );

        let buffer: Buffer<i8> =
            Buffer::new(&context, CL_MEM_READ_ONLY as cl_mem_flags, 256).unwrap();

        let mem_info = unsafe { cl_get_mem_object_info(buffer.get_cl_mem()) }.unwrap();
        println!("{:#?}", mem_info);

        assert_eq!(
            mem_info,
            MemInfo {
                mem_type: CL_MEM_OBJECT_BUFFER,
                mem_size: 256,
                reference_count: 1,
            }
        );
    }

    #[test]
    fn test_buffer_drop() {
        let platforms = cl_get_platform_ids().unwrap();
        let platform_id = platforms[0];

        let devices = unsafe { cl_get_device_ids(platform_id).unwrap() };
        let device_id = devices[0];

        let device = Device::new(device_id).unwrap();

        // valid context reference
        let context = Context::new(&[device]).unwrap();

        let buffer: Buffer<i32> =
            Buffer::new(&context, CL_MEM_READ_ONLY as cl_mem_flags, 256).unwrap();

        let mem = buffer.get_cl_mem();
        drop(buffer);

        let mem_info = unsafe { cl_get_mem_object_info(mem) }.unwrap();
        println!("{:#?}", mem_info);

        assert_ne!(
            mem_info,
            MemInfo {
                mem_type: CL_MEM_OBJECT_BUFFER,
                mem_size: 1024,
                reference_count: 1,
            }
        );
    }
}

#[cfg(test)]
mod tests_pipe {
    use super::*;
    use crate::unsafe_wrapper::{cl_get_device_ids, cl_get_platform_ids};
    use crate::wrapper::platform::Device;
    use opencl_sys::bindings::{CL_MEM_OBJECT_BUFFER, CL_MEM_OBJECT_PIPE};

    #[test]
    fn test_pipe_new() {
        let platforms = cl_get_platform_ids().unwrap();
        let platform_id = platforms[0];

        let devices = unsafe { cl_get_device_ids(platform_id).unwrap() };
        let device_id = devices[0];

        let device = Device::new(device_id).unwrap();

        // valid context reference
        let context = Context::new(&[device]).unwrap();

        let pipe: Pipe<i32> = Pipe::new(&context, 10).unwrap();

        let mem_info = unsafe { cl_get_mem_object_info(pipe.get_cl_mem()) }.unwrap();
        println!("{:#?}", mem_info);

        assert_eq!(
            mem_info,
            MemInfo {
                mem_type: CL_MEM_OBJECT_PIPE,
                mem_size: 528,
                reference_count: 1,
            }
        );

        let pipe: Pipe<i8> = Pipe::new(&context, 32).unwrap();

        let mem_info = unsafe { cl_get_mem_object_info(pipe.get_cl_mem()) }.unwrap();
        println!("{:#?}", mem_info);

        assert_eq!(
            mem_info,
            MemInfo {
                mem_type: CL_MEM_OBJECT_PIPE,
                mem_size: 1152,
                reference_count: 1,
            }
        );
    }

    #[test]
    fn test_pipe_drop() {
        let platforms = cl_get_platform_ids().unwrap();
        let platform_id = platforms[0];

        let devices = unsafe { cl_get_device_ids(platform_id).unwrap() };
        let device_id = devices[0];

        let device = Device::new(device_id).unwrap();

        // valid context reference
        let context = Context::new(&[device]).unwrap();

        let buffer: Pipe<i32> = Pipe::new(&context, 16).unwrap();

        let mem = buffer.get_cl_mem();
        drop(buffer);

        let mem_info = unsafe { cl_get_mem_object_info(mem) }.unwrap();
        println!("{:#?}", mem_info);

        assert_ne!(
            mem_info,
            MemInfo {
                mem_type: CL_MEM_OBJECT_BUFFER,
                mem_size: 1152,
                reference_count: 1,
            }
        );
    }
}
