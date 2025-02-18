//! # opencl unsafe wrapper
//! This is a small unsafe wrapper around OpenCL FFI bindings:
//!
//! * convert OpenCL C API functions into Rust functions that return a rust Result.
//! ```rust no_run
//! use opencl_sys::bindings::{clFinish, cl_command_queue};
//! use opencl::error::OclResult;
//! use opencl::unsafe_wrapper::{cl_finish};
//!
//! unsafe {
//!
//!     let command_queue: cl_command_queue = std::mem::zeroed(); // example
//!
//!     // from
//!     let ret = clFinish(command_queue);
//!
//!     // to
//!     let result: OclResult<()> = cl_finish(command_queue);
//!
//! }
//! ```
//!
//! * shorten the call of certain OpenCL C API functions, performing memory allocations, parameter selection, within the same function.
//! ```rust no_run
//! use opencl_sys::bindings::{clGetDeviceIDs, clGetPlatformIDs, cl_device_id, cl_device_type, cl_platform_id, CL_DEVICE_TYPE_GPU};
//! use opencl::error::OclResult;
//! use opencl::unsafe_wrapper::{cl_get_device_ids, cl_get_platform_ids};
//!
//! // from
//! unsafe {
//!
//!     let mut platforms: Vec<cl_platform_id> = vec![std::ptr::null_mut(); 1];
//!
//!     let ret = clGetPlatformIDs(1, platforms.as_mut_ptr(), std::ptr::null_mut());
//!
//!     let num_devices = 1;
//!     let mut devices: Vec<cl_device_id> = vec![std::ptr::null_mut(); num_devices as usize];
//!
//!     let ret = clGetDeviceIDs(
//!         platforms[0],
//!         CL_DEVICE_TYPE_GPU as cl_device_type,
//!         num_devices,
//!         devices.as_mut_ptr(),
//!         std::ptr::null_mut(),
//!     );
//!
//! }
//!
//! // to
//!
//! // get all platforms
//! let platforms: Vec<cl_platform_id> = cl_get_platform_ids().unwrap();
//!
//! unsafe {
//!     
//!     // always DEVICE TYPE GPU
//!     let result_devices: OclResult<Vec<cl_device_id>> = cl_get_device_ids(platforms[0]);
//!
//! }
//!
//! ```

use crate::error::{cl_check, OclResult};
use crate::utils::get_board_name_amd;
use opencl_sys::bindings::{
    clBuildProgram, clCreateBuffer, clCreateCommandQueueWithProperties, clCreateContext,
    clCreateKernel, clCreatePipe, clCreateProgramWithSource, clEnqueueNDRangeKernel,
    clEnqueueReadBuffer, clEnqueueWriteBuffer, clFinish, clFlush, clGetCommandQueueInfo,
    clGetContextInfo, clGetDeviceIDs, clGetDeviceInfo, clGetKernelInfo, clGetKernelWorkGroupInfo,
    clGetMemObjectInfo, clGetPlatformIDs, clGetPlatformInfo, clGetProgramBuildInfo,
    clGetProgramInfo, clReleaseCommandQueue, clReleaseContext, clReleaseKernel, clReleaseMemObject,
    clReleaseProgram, clSetKernelArg, cl_bool, cl_char, cl_command_queue, cl_context, cl_device_id,
    cl_device_type, cl_event, cl_kernel, cl_mem, cl_mem_flags, cl_mem_object_type, cl_platform_id,
    cl_program, cl_queue_properties, cl_uint, cl_ulong, CL_BUILD_PROGRAM_FAILURE,
    CL_CONTEXT_NUM_DEVICES, CL_CONTEXT_REFERENCE_COUNT, CL_DEVICE_MAX_WORK_GROUP_SIZE,
    CL_DEVICE_NAME, CL_DEVICE_TYPE_GPU, CL_DEVICE_VENDOR, CL_INVALID_VALUE,
    CL_KERNEL_LOCAL_MEM_SIZE, CL_KERNEL_NUM_ARGS, CL_KERNEL_REFERENCE_COUNT,
    CL_KERNEL_WORK_GROUP_SIZE, CL_MEM_REFERENCE_COUNT, CL_MEM_SIZE, CL_MEM_TYPE, CL_PLATFORM_NAME,
    CL_PLATFORM_PROFILE, CL_PLATFORM_VERSION, CL_PROGRAM_BINARY_SIZES,
    CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE, CL_PROGRAM_BUILD_LOG, CL_PROGRAM_KERNEL_NAMES,
    CL_PROGRAM_NUM_DEVICES, CL_PROGRAM_REFERENCE_COUNT, CL_QUEUE_PROPERTIES,
    CL_QUEUE_REFERENCE_COUNT,
};
use std::ffi::{c_void, CString};
use std::{mem, ptr};
use utilities::helper_functions::buf_i8_to_string;

/// ...
///
/// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_querying_platform_info
pub fn cl_get_platform_ids_count() -> OclResult<cl_uint> {
    let mut num_platforms = 0;

    let ret = unsafe { clGetPlatformIDs(0, std::ptr::null_mut(), &mut num_platforms) };
    cl_check(ret)?;

    Ok(num_platforms)
}

/// ...
///
/// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_querying_platform_info
pub fn cl_get_platform_ids() -> OclResult<Vec<cl_platform_id>> {
    let num_entries = cl_get_platform_ids_count()?;

    if num_entries == 0 {
        return Ok(vec![]);
    }

    let mut platforms: Vec<cl_platform_id> = vec![ptr::null_mut(); num_entries as usize];

    let ret = unsafe { clGetPlatformIDs(num_entries, platforms.as_mut_ptr(), ptr::null_mut()) };
    cl_check(ret)?;

    Ok(platforms)
}

#[derive(Debug)]
pub struct PlatformInfo {
    pub profile: String,
    pub version: String,
    pub name: String,
}

/// ...
///
/// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_querying_platform_info
///
/// # Safety
///
/// TODO safety function explain
pub unsafe fn cl_get_platform_info(platform_id: cl_platform_id) -> OclResult<PlatformInfo> {
    // use CString ?

    let mut profile: Vec<i8> = vec![0; 32];
    let mut profile_param_value_size = mem::size_of::<i8>() * 32;

    let ret = clGetPlatformInfo(
        platform_id,
        CL_PLATFORM_PROFILE,
        profile_param_value_size,
        profile.as_mut_ptr() as *mut c_void,
        &mut profile_param_value_size,
    );

    cl_check(ret)?;

    let mut version: Vec<i8> = vec![0; 128];
    let mut version_param_value_size = mem::size_of::<i8>() * 128;

    let ret = clGetPlatformInfo(
        platform_id,
        CL_PLATFORM_VERSION,
        version_param_value_size,
        version.as_mut_ptr() as *mut c_void,
        &mut version_param_value_size,
    );

    cl_check(ret)?;

    let mut name: Vec<i8> = vec![0; 128];
    let mut name_param_value_size = mem::size_of::<i8>() * 128;

    let ret = clGetPlatformInfo(
        platform_id,
        CL_PLATFORM_NAME,
        name_param_value_size,
        name.as_mut_ptr() as *mut c_void,
        &mut name_param_value_size,
    );

    cl_check(ret)?;

    Ok(PlatformInfo {
        profile: buf_i8_to_string(&profile).unwrap(),
        version: buf_i8_to_string(&version).unwrap(),
        name: buf_i8_to_string(&name).unwrap(),
    })
}

/// ...
///
/// Only GPU type devices are required
/// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#platform-querying-devices
///
/// # Safety
///
/// TODO safety function explain
pub unsafe fn cl_get_device_count(platform_id: cl_platform_id) -> OclResult<cl_uint> {
    let mut num_devices = 0;

    let ret = clGetDeviceIDs(
        platform_id,
        CL_DEVICE_TYPE_GPU as cl_device_type,
        0,
        ptr::null_mut(),
        &mut num_devices,
    );
    cl_check(ret)?;

    Ok(num_devices)
}

///
/// Only GPU type devices are required
/// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#platform-querying-devices
///
/// # Safety
///
/// TODO safety function explain
pub unsafe fn cl_get_device_ids(platform_id: cl_platform_id) -> OclResult<Vec<cl_device_id>> {
    let num_devices = cl_get_device_count(platform_id)?;

    if num_devices == 0 {
        return Ok(vec![]);
    }

    let mut devices: Vec<cl_device_id> = vec![ptr::null_mut(); num_devices as usize];

    let ret = clGetDeviceIDs(
        platform_id,
        CL_DEVICE_TYPE_GPU as cl_device_type,
        num_devices,
        devices.as_mut_ptr(),
        ptr::null_mut(),
    );
    cl_check(ret)?;

    Ok(devices)
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub board_name: &'static str,
    pub name: String,
    pub vendor: String,
    pub max_work_group_size: usize,
}

/// ...
///
/// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#platform-querying-devices
///
/// # Safety
///
/// TODO safety function explain
pub unsafe fn cl_get_device_info(device: cl_device_id) -> OclResult<DeviceInfo> {
    // use CString ?

    let mut device_name: Vec<i8> = vec![0; 32];
    let mut device_name_param_value_size = mem::size_of::<i8>() * 32;

    let ret = clGetDeviceInfo(
        device,
        CL_DEVICE_NAME,
        device_name_param_value_size,
        device_name.as_mut_ptr() as *mut c_void,
        &mut device_name_param_value_size,
    );
    cl_check(ret)?;

    let mut vendor_name: Vec<i8> = vec![0; 128];
    let mut vendor_name_param_value_size = mem::size_of::<i8>() * 128;

    let ret = clGetDeviceInfo(
        device,
        CL_DEVICE_VENDOR,
        vendor_name_param_value_size,
        vendor_name.as_mut_ptr() as *mut c_void,
        &mut vendor_name_param_value_size,
    );
    cl_check(ret)?;

    let mut max_work_group_size = 0;
    let mut max_work_group_size_param_value_size = mem::size_of::<usize>();

    let ret = clGetDeviceInfo(
        device,
        CL_DEVICE_MAX_WORK_GROUP_SIZE,
        max_work_group_size_param_value_size,
        &mut max_work_group_size as *mut _ as *mut c_void,
        &mut max_work_group_size_param_value_size,
    );
    cl_check(ret)?;

    let name = buf_i8_to_string(&device_name).unwrap();

    Ok(DeviceInfo {
        board_name: get_board_name_amd(&name),
        name,
        vendor: buf_i8_to_string(&vendor_name).unwrap(),
        max_work_group_size,
    })
}

/// ...
///
/// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_contexts
///
/// # Safety
///
/// TODO safety function explain
pub unsafe fn cl_create_context(devices: &[cl_device_id]) -> OclResult<cl_context> {
    let num_devices = devices.len() as cl_uint;
    let mut ret = CL_INVALID_VALUE;

    let context = clCreateContext(
        ptr::null_mut(),
        num_devices,
        devices.as_ptr(),
        None,
        ptr::null_mut(),
        &mut ret,
    );
    cl_check(ret)?;

    Ok(context)
}

#[derive(Debug, Clone)]
pub struct ContextInfo {
    pub reference_count: cl_uint,
    pub num_devices: cl_uint,
}

/// ...
///
/// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_contexts
///
/// # Safety
///
/// TODO safety function explain
pub unsafe fn cl_get_context_info(context: cl_context) -> OclResult<ContextInfo> {
    let mut reference_count: cl_uint = 0;
    let mut reference_count_size_param_value_size = mem::size_of::<cl_uint>();

    let ret = clGetContextInfo(
        context,
        CL_CONTEXT_REFERENCE_COUNT,
        reference_count_size_param_value_size,
        &mut reference_count as *mut _ as *mut c_void,
        &mut reference_count_size_param_value_size,
    );
    cl_check(ret)?;

    let mut num_devices: cl_uint = 0;
    let mut num_devices_param_value_size = mem::size_of::<cl_uint>();

    let ret = clGetContextInfo(
        context,
        CL_CONTEXT_NUM_DEVICES,
        num_devices_param_value_size,
        &mut num_devices as *mut _ as *mut c_void,
        &mut num_devices_param_value_size,
    );
    cl_check(ret)?;

    Ok(ContextInfo {
        reference_count,
        num_devices,
    })
}

/// ...
///
/// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_contexts
///
/// # Safety
///
/// TODO safety function explain
pub unsafe fn cl_release_context(context: cl_context) -> OclResult<()> {
    let ret = clReleaseContext(context);
    cl_check(ret)
}

/// ...
///
/// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_command_queues
///
/// # Safety
///
/// TODO safety function explain
pub unsafe fn cl_create_command_queue_with_properties(
    context: cl_context,
    device: cl_device_id,
    queue_property: cl_queue_properties,
) -> OclResult<cl_command_queue> {
    let mut ret = CL_INVALID_VALUE;

    // TODO check
    let mut props: [cl_queue_properties; 3] = [0; 3];
    props[0] = CL_QUEUE_PROPERTIES as cl_queue_properties;
    props[1] = queue_property;

    let command_queue =
        clCreateCommandQueueWithProperties(context, device, props.as_ptr(), &mut ret);

    cl_check(ret)?;

    Ok(command_queue)
}

#[derive(Debug, Clone)]
pub struct CommandQueueInfo {
    pub reference_count: cl_uint,
}

/// ...
///
/// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_command_queues
///
/// # Safety
///
/// TODO safety function explain
pub unsafe fn cl_get_command_queue_info(
    command_queue: cl_command_queue,
) -> OclResult<CommandQueueInfo> {
    let mut reference_count: cl_uint = 0;
    let mut reference_count_param_value_size = mem::size_of::<cl_uint>();

    let ret = clGetCommandQueueInfo(
        command_queue,
        CL_QUEUE_REFERENCE_COUNT,
        reference_count_param_value_size,
        &mut reference_count as *mut _ as *mut c_void,
        &mut reference_count_param_value_size,
    );
    cl_check(ret)?;

    Ok(CommandQueueInfo { reference_count })
}

/// ...
///
/// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_command_queues
///
/// # Safety
///
/// TODO safety function explain
pub unsafe fn cl_release_command_queue(command_queue: cl_command_queue) -> OclResult<()> {
    let ret = clReleaseCommandQueue(command_queue);
    cl_check(ret)
}

/// ...
///
/// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_creating_buffer_objects
///
/// # Safety
///
/// TODO safety function explain
pub unsafe fn cl_create_buffer(
    context: cl_context,
    flags: cl_mem_flags,
    size: usize,
) -> OclResult<cl_mem> {
    let mut ret = CL_INVALID_VALUE;

    let cl_buffer = clCreateBuffer(context, flags, size, ptr::null_mut(), &mut ret);
    cl_check(ret)?;

    Ok(cl_buffer)
}

#[derive(Debug, Clone, PartialEq)]
pub struct MemInfo {
    pub mem_type: cl_mem_object_type,
    pub mem_size: usize,
    pub reference_count: cl_uint,
}

/// ...
///
/// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#memory-object-queries
///
/// # Safety
///
/// TODO safety function explain
pub unsafe fn cl_get_mem_object_info(mem_obj: cl_mem) -> OclResult<MemInfo> {
    let mut mem_type: cl_mem_object_type = 0;
    let mut mem_type_param_value_size = mem::size_of::<cl_mem_object_type>();

    let ret = clGetMemObjectInfo(
        mem_obj,
        CL_MEM_TYPE,
        mem_type_param_value_size,
        &mut mem_type as *mut _ as *mut c_void,
        &mut mem_type_param_value_size,
    );
    cl_check(ret)?;

    let mut mem_size: usize = 0;
    let mut mem_size_param_value_size = mem::size_of::<usize>();

    let ret = clGetMemObjectInfo(
        mem_obj,
        CL_MEM_SIZE,
        mem_size_param_value_size,
        &mut mem_size as *mut _ as *mut c_void,
        &mut mem_size_param_value_size,
    );
    cl_check(ret)?;

    let mut reference_count: cl_uint = 0;
    let mut reference_count_param_value_size = mem::size_of::<cl_uint>();

    let ret = clGetMemObjectInfo(
        mem_obj,
        CL_MEM_REFERENCE_COUNT,
        reference_count_param_value_size,
        &mut reference_count as *mut _ as *mut c_void,
        &mut reference_count_param_value_size,
    );
    cl_check(ret)?;

    Ok(MemInfo {
        mem_type,
        mem_size,
        reference_count,
    })
}

/// ...
///
/// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_retaining_and_releasing_memory_objects
///
/// # Safety
///
/// TODO safety function explain
pub unsafe fn cl_release_mem_object(mem_obj: cl_mem) -> OclResult<()> {
    let ret = clReleaseMemObject(mem_obj);
    cl_check(ret)
}

/// ...
///
/// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_reading_writing_and_copying_buffer_objects
///
/// # Safety
///
/// TODO safety function explain
pub unsafe fn cl_enqueue_read_buffer<T>(
    command_queue: cl_command_queue,
    buffer: cl_mem,
    blocking_read: cl_bool,
    data: &mut [T],
    event_wait_list: &[cl_event],
) -> OclResult<cl_event> {
    // let size = data.len() * mem::size_of::<i32>();
    let size = std::mem::size_of_val(data);

    let mut event: cl_event = ptr::null_mut();

    let ret = clEnqueueReadBuffer(
        command_queue,
        buffer,
        blocking_read,
        0,
        size,
        data.as_mut_ptr() as *mut c_void,
        event_wait_list.len() as cl_uint,
        if !event_wait_list.is_empty() {
            event_wait_list.as_ptr()
        } else {
            ptr::null()
        },
        &mut event,
    );

    cl_check(ret)?;

    Ok(event)
}

/// ...
///
/// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_reading_writing_and_copying_buffer_objects
///
/// # Safety
///
/// TODO safety function explain
pub unsafe fn cl_enqueue_write_buffer<T>(
    command_queue: cl_command_queue,
    buffer: cl_mem,
    blocking_write: cl_bool,
    data: &[T],
    event_wait_list: &[cl_event],
) -> OclResult<cl_event> {
    // let size = data.len() * mem::size_of::<i32>();
    let size = std::mem::size_of_val(data);

    let mut event: cl_event = ptr::null_mut();

    let ret = clEnqueueWriteBuffer(
        command_queue,
        buffer,
        blocking_write,
        0,
        size,
        data.as_ptr() as *const c_void,
        event_wait_list.len() as cl_uint,
        if !event_wait_list.is_empty() {
            event_wait_list.as_ptr()
        } else {
            ptr::null()
        },
        &mut event,
    );
    cl_check(ret)?;

    Ok(event)
}

/// ...
///
/// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_creating_program_objects
///
/// # Safety
///
/// TODO safety function explain
pub unsafe fn cl_create_program_with_source(
    context: cl_context,
    source: &str,
) -> OclResult<cl_program> {
    let count = 1;
    let mut strings = [source.as_ptr() as *const i8];
    let lengths = source.len();
    let mut ret = CL_INVALID_VALUE;

    let program =
        clCreateProgramWithSource(context, count, strings.as_mut_ptr(), &lengths, &mut ret);

    cl_check(ret)?;

    Ok(program)
}

/// ...
///
/// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_retaining_and_releasing_program_objects
///
/// # Safety
///
/// TODO safety function explain
pub unsafe fn cl_release_program(program: cl_program) -> OclResult<()> {
    let ret = clReleaseProgram(program);
    cl_check(ret)
}

pub const CL_STD_2_0: &str = "-cl-std=CL2.0 ";

/// ...
///
/// always opencl 2.0
///
/// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_building_program_executables
///
/// # Safety
///
/// TODO safety function explain
pub unsafe fn cl_build_program(program: cl_program, devices: &[cl_device_id]) -> OclResult<()> {
    let num_devices = devices.len() as cl_uint;
    let options = CString::new(CL_STD_2_0).unwrap();

    let ret = clBuildProgram(
        program,
        num_devices,
        devices.as_ptr(),
        options.as_ptr(),
        None,
        ptr::null_mut(),
    );

    if ret == CL_BUILD_PROGRAM_FAILURE {
        let device_id = devices[0];
        let program_build_info = cl_get_program_build_info(program, device_id)?;

        panic!(
            "CL_BUILD_PROGRAM_FAILURE ({}): {}",
            ret, program_build_info.log
        )
    }

    cl_check(ret)
}

#[derive(Debug, PartialEq)]
pub struct ProgramInfo {
    // pub src: String,
    pub kernel_names: Vec<String>,
    pub reference_count: cl_uint,
    pub num_devices: cl_uint,
    pub binary_sizes: Vec<usize>,
}

/// ...
///
/// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_program_object_queries
///
/// # Safety
///
/// TODO safety function explain
pub unsafe fn cl_get_program_info(
    program: cl_program,
    program_source_size: usize,
) -> OclResult<ProgramInfo> {
    // let program_src_size = program_source_size + 1;

    // use CString ?

    // let program_source: Vec<cl_char> = vec![0; program_src_size];
    // let mut program_source_param_size = program_src_size;
    //
    // let ret = clGetProgramInfo(
    //     program,
    //     CL_PROGRAM_SOURCE,
    //     program_source_param_size,
    //     program_source.as_ptr() as *mut c_void,
    //     &mut program_source_param_size,
    // );
    //
    // cl_check(ret, ())?;

    // The total size of the program kernel names has not been calculated;
    // For simplicity, half the size of the program's src is used.
    let kernel_names: Vec<cl_char> = vec![0; program_source_size / 2];
    let mut kernel_names_param_size = program_source_size / 2;

    let ret = clGetProgramInfo(
        program,
        CL_PROGRAM_KERNEL_NAMES,
        kernel_names_param_size,
        kernel_names.as_ptr() as *mut c_void,
        &mut kernel_names_param_size,
    );

    cl_check(ret)?;

    let mut reference_count: cl_uint = 0;
    let mut reference_count_param_value_size = mem::size_of::<cl_uint>();

    let ret = clGetProgramInfo(
        program,
        CL_PROGRAM_REFERENCE_COUNT,
        reference_count_param_value_size,
        &mut reference_count as *mut _ as *mut c_void,
        &mut reference_count_param_value_size,
    );
    cl_check(ret)?;

    let mut num_devices: cl_uint = 0;
    let mut num_devices_param_value_size = mem::size_of::<cl_uint>();

    let ret = clGetProgramInfo(
        program,
        CL_PROGRAM_NUM_DEVICES,
        num_devices_param_value_size,
        &mut num_devices as *mut _ as *mut c_void,
        &mut num_devices_param_value_size,
    );
    cl_check(ret)?;

    // CL_PROGRAM_BINARY_SIZES

    let binary_sizes: Vec<usize> = vec![0; 1];
    let mut binary_sizes_param_size = std::mem::size_of_val(&kernel_names);

    let ret = clGetProgramInfo(
        program,
        CL_PROGRAM_BINARY_SIZES,
        binary_sizes_param_size,
        binary_sizes.as_ptr() as *mut c_void,
        &mut binary_sizes_param_size,
    );
    cl_check(ret)?;

    // ...

    let kernel_names = buf_i8_to_string(&kernel_names)
        .unwrap()
        .split(";")
        .map(|s| s.to_string())
        .collect();

    Ok(ProgramInfo {
        // src: buf_i8_to_string(&program_source),
        kernel_names,
        reference_count,
        num_devices,
        binary_sizes,
    })
}

#[derive(Debug)]
pub struct ProgramBuildInfo {
    pub log: String,
    pub global_variable_total_size: usize,
}

/// ...
///
/// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_program_object_queries
///
/// # Safety
///
/// TODO safety function explain
pub unsafe fn cl_get_program_build_info(
    program: cl_program,
    device: cl_device_id,
) -> OclResult<ProgramBuildInfo> {
    // TODO update log size
    let log_size = 1024 * 1024;

    let param_program_build_log: Vec<cl_char> = vec![0; log_size];
    let mut param_program_build_log_size = log_size;

    let ret = clGetProgramBuildInfo(
        program,
        device,
        CL_PROGRAM_BUILD_LOG,
        param_program_build_log_size,
        param_program_build_log.as_ptr() as *mut c_void,
        &mut param_program_build_log_size,
    );

    cl_check(ret)?;

    let mut param_global_variable_total_size: usize = 0;
    let mut param_global_variable_total_size_ret = mem::size_of::<usize>();

    let ret = clGetProgramBuildInfo(
        program,
        device,
        CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE,
        param_global_variable_total_size_ret,
        &mut param_global_variable_total_size as *mut _ as *mut c_void,
        &mut param_global_variable_total_size_ret,
    );
    cl_check(ret)?;

    Ok(ProgramBuildInfo {
        log: buf_i8_to_string(&param_program_build_log).unwrap(),
        global_variable_total_size: param_global_variable_total_size,
    })
}

/// ...
///
/// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_creating_kernel_objects
///
/// # Safety
///
/// TODO safety function explain
pub unsafe fn cl_create_kernel(program: cl_program, kernel_name: &str) -> OclResult<cl_kernel> {
    let mut ret = CL_INVALID_VALUE;
    let k_n = kernel_name.as_bytes();
    let c = CString::new(k_n).unwrap();

    let kernel = clCreateKernel(program, c.as_ptr(), &mut ret);
    cl_check(ret)?;

    Ok(kernel)
}

#[derive(Debug, PartialEq)]
pub struct KernelInfo {
    pub num_args: cl_uint,
    pub reference_count: cl_uint,
}

/// ...
///
/// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_kernel_object_queries
///
/// # Safety
///
/// TODO safety function explain
pub unsafe fn cl_get_kernel_info(kernel: cl_kernel) -> OclResult<KernelInfo> {
    let mut num_args: cl_uint = 0;
    let mut num_args_param_value_size = mem::size_of::<cl_uint>();

    let ret = clGetKernelInfo(
        kernel,
        CL_KERNEL_NUM_ARGS,
        num_args_param_value_size,
        &mut num_args as *mut _ as *mut c_void,
        &mut num_args_param_value_size,
    );
    cl_check(ret)?;

    let mut reference_count: cl_uint = 0;
    let mut reference_count_param_value_size = mem::size_of::<cl_uint>();

    let ret = clGetKernelInfo(
        kernel,
        CL_KERNEL_REFERENCE_COUNT,
        reference_count_param_value_size,
        &mut reference_count as *mut _ as *mut c_void,
        &mut reference_count_param_value_size,
    );
    cl_check(ret)?;

    Ok(KernelInfo {
        num_args,
        reference_count,
    })
}

#[derive(Debug, PartialEq)]
pub struct KernelWorkGroupInfo {
    pub work_group_size: usize,
    pub local_mem_size: cl_ulong,
}

/// ...
///
/// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_kernel_object_queries
///
/// # Safety
///
/// TODO safety function explain
pub unsafe fn cl_get_kernel_work_group_info(
    kernel: cl_kernel,
    device_id: cl_device_id,
) -> OclResult<KernelWorkGroupInfo> {
    let mut work_group_size: usize = 0;
    let mut work_group_size_param_value_size = mem::size_of::<usize>();

    let ret = clGetKernelWorkGroupInfo(
        kernel,
        device_id,
        CL_KERNEL_WORK_GROUP_SIZE,
        work_group_size_param_value_size,
        &mut work_group_size as *mut _ as *mut c_void,
        &mut work_group_size_param_value_size,
    );
    cl_check(ret)?;

    let mut local_mem_size: cl_ulong = 0;
    let mut local_mem_size_param_value_size = mem::size_of::<cl_ulong>();

    let ret = clGetKernelWorkGroupInfo(
        kernel,
        device_id,
        CL_KERNEL_LOCAL_MEM_SIZE,
        local_mem_size_param_value_size,
        &mut local_mem_size as *mut _ as *mut c_void,
        &mut local_mem_size_param_value_size,
    );
    cl_check(ret)?;

    Ok(KernelWorkGroupInfo {
        work_group_size,
        local_mem_size,
    })
}

/// ...
///
/// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_creating_kernel_objects
///
/// # Safety
///
/// TODO safety function explain
pub unsafe fn cl_release_kernel(kernel: cl_kernel) -> OclResult<()> {
    let ret = clReleaseKernel(kernel);

    cl_check(ret)
}

/// ...
///
/// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#setting-kernel-arguments
///
/// # Safety
///
/// TODO safety function explain
pub unsafe fn cl_set_kernel_arg<T>(
    kernel: cl_kernel,
    arg_index: cl_uint,
    arg_value: &T,
) -> OclResult<()> {
    let ret = clSetKernelArg(
        kernel,
        arg_index,
        mem::size_of::<T>(),
        arg_value as *const _ as *const c_void,
    );

    cl_check(ret)
}

/// ...
///
/// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_executing_kernels
///
/// # Safety
///
/// TODO safety function explain
pub unsafe fn cl_enqueue_nd_range_kernel(
    command_queue: cl_command_queue,
    kernel: cl_kernel,
    global_work_offsets: &[usize],
    global_work_sizes: &[usize],
    local_work_sizes: &[usize],
    event_wait_list: &[cl_event],
) -> OclResult<cl_event> {
    // check len ? - global_work_offset & global_work_size & local_work_size

    let work_dim = global_work_sizes.len();

    let global_work_offset = if global_work_offsets.is_empty() {
        ptr::null()
    } else {
        global_work_offsets.as_ptr()
    };
    let global_work_size = global_work_sizes.as_ptr();

    let local_work_size = if local_work_sizes.is_empty() {
        ptr::null()
    } else {
        local_work_sizes.as_ptr()
    };

    let num_events_in_wait_list = event_wait_list.len();

    let mut event: cl_event = ptr::null_mut();

    let ret = clEnqueueNDRangeKernel(
        command_queue,
        kernel,
        work_dim as cl_uint,
        global_work_offset,
        global_work_size,
        local_work_size,
        num_events_in_wait_list as cl_uint,
        if !event_wait_list.is_empty() {
            event_wait_list.as_ptr()
        } else {
            ptr::null()
        },
        &mut event,
    );
    cl_check(ret)?;

    Ok(event)
}

/// ...
///
/// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_flush_and_finish
///
/// # Safety
///
/// TODO safety function explain
pub unsafe fn cl_flush(command_queue: cl_command_queue) -> OclResult<()> {
    let ret = unsafe { clFlush(command_queue) };
    cl_check(ret)
}

/// ...
///
/// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_flush_and_finish
///
/// # Safety
///
/// TODO safety function explain
pub unsafe fn cl_finish(command_queue: cl_command_queue) -> OclResult<()> {
    let ret = unsafe { clFinish(command_queue) };
    cl_check(ret)
}

/// ...
///
/// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_creating_pipe_objects
///
/// # Safety
///
/// TODO safety function explain
pub unsafe fn cl_create_pipe<T>(
    context: cl_context,
    pipe_max_packets: cl_uint,
) -> OclResult<cl_mem> {
    let mut ret = CL_INVALID_VALUE;

    let pipe_packet_size = mem::size_of::<T>() * pipe_max_packets as usize;

    let cl_pipe = clCreatePipe(
        context,
        0, // default
        pipe_packet_size as cl_uint,
        pipe_max_packets,
        ptr::null_mut(),
        &mut ret,
    );
    cl_check(ret)?;

    Ok(cl_pipe)
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencl_sys::bindings::{
        CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY, CL_QUEUE_PROFILING_ENABLE, CL_TRUE,
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
        unsafe {
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

            let platforms = cl_get_platform_ids().unwrap();
            let platform_id = platforms[0];

            let platform_info = cl_get_platform_info(platform_id).unwrap();
            println!("{:#?}", platform_info);

            let devices = cl_get_device_ids(platform_id).unwrap();
            let device_id = devices[0];

            let device_info = cl_get_device_info(device_id).unwrap();
            println!("{:#?}", device_info);

            // Create an OpenCL context
            let context = cl_create_context(&[device_id]).unwrap();

            let context_info = cl_get_context_info(context).unwrap();
            println!("{:#?}", context_info);

            // Create a command queue
            let command_queue = cl_create_command_queue_with_properties(
                context,
                device_id,
                CL_QUEUE_PROFILING_ENABLE as cl_queue_properties,
            )
            .unwrap();

            let command_queue_info = cl_get_command_queue_info(command_queue).unwrap();
            println!("{:#?}", command_queue_info);

            let input_size = list_size * mem::size_of::<i32>();

            // Create memory buffers on the device for each vector
            let input_a_buf =
                cl_create_buffer(context, CL_MEM_READ_ONLY as cl_mem_flags, input_size).unwrap();

            let mem_info = cl_get_mem_object_info(input_a_buf).unwrap();
            println!("{:#?}", mem_info);

            let input_b_buf =
                cl_create_buffer(context, CL_MEM_READ_ONLY as cl_mem_flags, input_size).unwrap();

            let output_c_buf =
                cl_create_buffer(context, CL_MEM_WRITE_ONLY as cl_mem_flags, input_size).unwrap();

            // Copy the lists A and B to their respective memory buffers
            let _write_event =
                cl_enqueue_write_buffer(command_queue, input_a_buf, CL_TRUE, &input_a, &[])
                    .unwrap();

            let _write_event =
                cl_enqueue_write_buffer(command_queue, input_b_buf, CL_TRUE, &input_b, &[])
                    .unwrap();

            // Create a program from the kernel source
            let program = cl_create_program_with_source(context, PROGRAM_SRC).unwrap();

            // Build the program
            cl_build_program(program, &[device_id]).unwrap();

            let program_info = cl_get_program_info(program, PROGRAM_SRC.len()).unwrap();
            println!("{:#?}", program_info);
            // println!("{}", program_info.src);
            // println!("kernel_names: {:?}", program_info.kernel_names);

            let program_build_info = cl_get_program_build_info(program, device_id).unwrap();
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
            let kernel = cl_create_kernel(program, "vecAdd").unwrap();

            let kernel_info = cl_get_kernel_info(kernel).unwrap();
            println!("{:#?}", kernel_info);

            let kernel_work_group_info = cl_get_kernel_work_group_info(kernel, device_id).unwrap();
            println!("{:#?}", kernel_work_group_info);

            // Set the arguments of the kernel
            cl_set_kernel_arg(kernel, 0, &input_a_buf).unwrap();
            cl_set_kernel_arg(kernel, 1, &input_b_buf).unwrap();
            cl_set_kernel_arg(kernel, 2, &output_c_buf).unwrap();

            // Execute the OpenCL kernel on the list
            let global_work_size = list_size; // Process the entire lists
            let local_work_size = 64; // Process in groups of 64

            let _event = cl_enqueue_nd_range_kernel(
                command_queue,
                kernel,
                &[],
                &[global_work_size],
                &[local_work_size],
                &[],
            )
            .unwrap();

            // Read the memory buffer C on the device to the local variable C

            let mut output_c: Vec<i32> = vec![0; list_size];
            cl_enqueue_read_buffer(command_queue, output_c_buf, CL_TRUE, &mut output_c, &[])
                .unwrap();

            // Display the result to the screen
            for i in 0..list_size {
                println!(
                    "index: {} -> A ({}) + B ({}) = C ({})",
                    i, input_a[i], input_b[i], output_c[i]
                );
            }

            // Clean up
            cl_flush(command_queue).unwrap();
            cl_finish(command_queue).unwrap();

            cl_release_kernel(kernel).unwrap();
            cl_release_program(program).unwrap();

            cl_release_mem_object(input_a_buf).unwrap();
            cl_release_mem_object(input_b_buf).unwrap();
            cl_release_mem_object(output_c_buf).unwrap();

            cl_release_command_queue(command_queue).unwrap();

            cl_release_context(context).unwrap();

            assert!(output_c.iter().all(|&x| x == list_size as i32));
        }
    }

    // TODO add sample code https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_sample_code_5
}
