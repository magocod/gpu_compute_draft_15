use crate::error::{hsa_check, HsaResult};
use crate::system::{Agent, MemoryPool};
use hsa_sys::bindings::{
    hsa_agent_t, hsa_amd_agents_allow_access, hsa_amd_memory_pool_allocate,
    hsa_amd_memory_pool_free, hsa_code_object_reader_create_from_file,
    hsa_code_object_reader_destroy, hsa_code_object_reader_t,
    hsa_default_float_rounding_mode_t_HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
    hsa_executable_create_alt, hsa_executable_destroy, hsa_executable_freeze,
    hsa_executable_get_symbol, hsa_executable_load_agent_code_object,
    hsa_executable_symbol_get_info,
    hsa_executable_symbol_info_t_HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
    hsa_executable_symbol_info_t_HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
    hsa_executable_symbol_info_t_HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
    hsa_executable_symbol_info_t_HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
    hsa_executable_symbol_t, hsa_executable_t, hsa_file_t, hsa_kernel_dispatch_packet_t,
    hsa_memory_copy, hsa_profile_t_HSA_PROFILE_FULL, hsa_signal_t, hsa_status_t_HSA_STATUS_SUCCESS,
};
use libc::{close, open, O_RDONLY};
use std::ffi::CString;
use std::path::Path;

#[derive(Debug)]
pub struct DispatchPacketOptions {
    // X dimension of work-group, in work-items. Must be greater than 0.
    pub workgroup_size_x: u16,
    // Y dimension of work-group, in work-items. Must be greater than\n 0. If the grid has 1 dimension, the only valid value is 1.
    pub workgroup_size_y: u16,
    // Z dimension of work-group, in work-items. Must be greater than\n 0. If the grid has 1 or 2 dimensions, the only valid value is 1.
    pub workgroup_size_z: u16,
    // X dimension of grid, in work-items. Must be greater than 0. Must\n not be smaller than @a workgroup_size_x.
    pub grid_size_x: u32,
    // Y dimension of grid, in work-items. Must be greater than 0. If the grid has\n 1 dimension, the only valid value is 1. Must not be smaller than @a\n workgroup_size_y.
    pub grid_size_y: u32,
    // Z dimension of grid, in work-items. Must be greater than 0. If the grid has\n 1 or 2 dimensions, the only valid value is 1. Must not be smaller than @a\n workgroup_size_z.
    pub grid_size_z: u32,
}

#[derive(Debug)]
pub struct AqlPacket {
    hsa_kernel_dispatch_packet: hsa_kernel_dispatch_packet_t,
    kern_arg_buffer: *mut std::os::raw::c_void,
    kernel: Kernel,
}

impl AqlPacket {
    pub fn get_hsa_kernel_dispatch_packet_t(&self) -> hsa_kernel_dispatch_packet_t {
        self.hsa_kernel_dispatch_packet
    }

    /// ...
    ///
    ///
    /// # Safety
    ///
    /// TODO safety function explain
    pub unsafe fn alloc_and_set_kern_args(
        &mut self,
        args: *mut std::os::raw::c_void,
        arg_size: usize,
        kern_arg_pool: &MemoryPool,
        allow_access_agents: &[&Agent],
    ) -> HsaResult<()> {
        let mut kern_arg_buf: *mut std::os::raw::c_void = std::ptr::null_mut();

        let req_align: usize = self.kernel.kern_arg_align as usize;
        // Allocate enough extra space for alignment adjustments if necessary
        let _buf_size = arg_size + (req_align << 1);

        // println!("arg_size: {}", arg_size);
        // println!("buf_size: {}", buf_size);
        // println!("req_align: {}", req_align);

        let err = hsa_amd_memory_pool_allocate(
            kern_arg_pool.get_hsa_amd_memory_pool_t(),
            arg_size,
            0,
            &mut kern_arg_buf,
        );
        hsa_check(err)?;

        // memcpy(bs.kern_arg_address, args, arg_size);
        let err = hsa_memory_copy(kern_arg_buf, args, arg_size);
        hsa_check(err)?;

        // Make sure both the CPU and GPU can access the kernel arguments
        let ag_list: Vec<hsa_agent_t> = allow_access_agents
            .iter()
            .map(|x| x.get_hsa_agent_t())
            .collect();

        let err = hsa_amd_agents_allow_access(
            ag_list.len() as u32,
            ag_list.as_ptr(),
            std::ptr::null_mut(),
            kern_arg_buf,
        );
        hsa_check(err)?;

        // Address of the allocated buffer
        self.kern_arg_buffer = kern_arg_buf;

        let aql_buf_ptr: *mut *mut std::os::raw::c_void =
            &mut self.hsa_kernel_dispatch_packet.kernarg_address;
        // Addr. of kern arg start.
        *aql_buf_ptr = kern_arg_buf;

        Ok(())
    }
}

impl Drop for AqlPacket {
    fn drop(&mut self) {
        if !self.hsa_kernel_dispatch_packet.kernarg_address.is_null() {
            unsafe {
                let ret = hsa_amd_memory_pool_free(self.kern_arg_buffer);
                if ret != hsa_status_t_HSA_STATUS_SUCCESS {
                    panic!("hsa_amd_memory_pool_free error: {:?}", ret);
                }
            }
        }
    }
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct Kernel {
    kernel_object: u64,
    private_segment_size: u32,
    group_segment_size: u32,
    kern_arg_size: u32,
    kern_arg_align: u32,
}

impl Kernel {
    pub fn create_empty() -> Self {
        Self {
            kernel_object: 0,
            private_segment_size: 0,
            group_segment_size: 0,
            kern_arg_size: 0,
            kern_arg_align: 0,
        }
    }

    pub fn populate_aql_packet(
        &self,
        options: DispatchPacketOptions,
        signal: hsa_signal_t,
    ) -> AqlPacket {
        let packet = hsa_kernel_dispatch_packet_t {
            header: 0,
            setup: 1,
            workgroup_size_x: options.workgroup_size_x,
            workgroup_size_y: 1,
            workgroup_size_z: 1,
            reserved0: 0,
            grid_size_x: options.grid_size_x,
            grid_size_y: 1,
            grid_size_z: 1,
            private_segment_size: self.private_segment_size,
            group_segment_size: self.group_segment_size,
            kernel_object: self.kernel_object,
            kernarg_address: std::ptr::null_mut(),
            reserved2: 0,
            completion_signal: signal,
        };

        AqlPacket {
            hsa_kernel_dispatch_packet: packet,
            kern_arg_buffer: std::ptr::null_mut(),
            kernel: *self,
        }
    }
}

#[derive(Debug)]
pub struct CodeObject {
    code_obj_rdr: hsa_code_object_reader_t,
    executable: hsa_executable_t,
    agent: Agent,
}

impl CodeObject {
    pub fn new<P: AsRef<Path>>(file_path: P, agent: &Agent) -> HsaResult<CodeObject> {
        let mut code_obj_rdr = hsa_code_object_reader_t { handle: 0 };
        let mut executable = hsa_executable_t { handle: 0 };

        let f_n = CString::new(file_path.as_ref().to_string_lossy().to_string()).unwrap();

        unsafe {
            let file_handle: hsa_file_t = open(f_n.as_ptr(), O_RDONLY);

            if file_handle == -1 {
                panic!("failed to open file {:?}", file_path.as_ref());
            }

            let err = hsa_code_object_reader_create_from_file(file_handle, &mut code_obj_rdr);
            close(file_handle);
            hsa_check(err)?;

            let err = hsa_executable_create_alt(
                hsa_profile_t_HSA_PROFILE_FULL,
                hsa_default_float_rounding_mode_t_HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                std::ptr::null_mut(),
                &mut executable,
            );
            hsa_check(err)?;

            let err = hsa_executable_load_agent_code_object(
                executable,
                agent.get_hsa_agent_t(),
                code_obj_rdr,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            hsa_check(err)?;

            let err = hsa_executable_freeze(executable, std::ptr::null_mut());
            hsa_check(err)?;
        }

        Ok(Self {
            code_obj_rdr,
            executable,
            agent: agent.clone(),
        })
    }

    pub fn get_agent(&self) -> &Agent {
        &self.agent
    }

    pub fn get_kernel(&self, kernel_name: &str) -> HsaResult<Kernel> {
        let mut kern_sym = hsa_executable_symbol_t { handle: 0 };
        let mut kernel_object: u64 = 0;
        let mut private_segment_size: u32 = 0;
        let mut group_segment_size: u32 = 0;
        let mut kern_arg_size: u32 = 0;
        let mut kern_arg_align: u32 = 0;

        unsafe {
            let k_n = CString::new(kernel_name).unwrap();

            let err = hsa_executable_get_symbol(
                self.executable,
                std::ptr::null_mut(),
                k_n.as_ptr(),
                self.agent.get_hsa_agent_t(),
                0,
                &mut kern_sym,
            );
            hsa_check(err)?;

            let err = hsa_executable_symbol_get_info(
                kern_sym,
                hsa_executable_symbol_info_t_HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                &mut kernel_object as *mut _ as *mut std::os::raw::c_void,
            );
            hsa_check(err)?;

            let err = hsa_executable_symbol_get_info(
                kern_sym,
                hsa_executable_symbol_info_t_HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                &mut private_segment_size as *mut _ as *mut std::os::raw::c_void,
            );
            hsa_check(err)?;

            let err = hsa_executable_symbol_get_info(
                kern_sym,
                hsa_executable_symbol_info_t_HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                &mut group_segment_size as *mut _ as *mut std::os::raw::c_void,
            );
            hsa_check(err)?;

            // Remaining queries not supported on code object v3.
            let err = hsa_executable_symbol_get_info(
                kern_sym,
                hsa_executable_symbol_info_t_HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
                &mut kern_arg_size as *mut _ as *mut std::os::raw::c_void,
            );
            hsa_check(err)?;

            let err = hsa_executable_symbol_get_info(
                kern_sym,
                hsa_executable_symbol_info_t_HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
                &mut kern_arg_align as *mut _ as *mut std::os::raw::c_void,
            );
            hsa_check(err)?;
        }

        Ok(Kernel {
            kernel_object,
            private_segment_size,
            group_segment_size,
            kern_arg_size,
            kern_arg_align,
        })
    }
}

impl PartialEq for CodeObject {
    fn eq(&self, other: &Self) -> bool {
        self.executable.handle.eq(&other.executable.handle)
    }
}

impl Drop for CodeObject {
    fn drop(&mut self) {
        unsafe {
            let ret = hsa_executable_destroy(self.executable);
            if ret != hsa_status_t_HSA_STATUS_SUCCESS {
                panic!("hsa_executable_destroy error: {:?}", ret);
            }

            let ret = hsa_code_object_reader_destroy(self.code_obj_rdr);
            if ret != hsa_status_t_HSA_STATUS_SUCCESS {
                panic!("hsa_code_object_reader_destroy error: {:?}", ret);
            }
        }
    }
}

#[cfg(test)]
mod tests_code_object {
    use super::*;
    use crate::error::HsaError;
    use crate::system::System;
    use hsa_sys::bindings::{
        hsa_status_t_HSA_STATUS_ERROR_INVALID_CODE_OBJECT,
        hsa_status_t_HSA_STATUS_ERROR_INVALID_SYMBOL_NAME,
    };
    use std::env;

    const CODE_OBJECT_FILE: &str = "global_array-hip-amdgcn-amd-amdhsa_gfx1032.o";

    #[test]
    fn test_code_object_create() {
        let system = System::new().unwrap();
        let gpu_agent = system.get_first_gpu().unwrap();

        let mut file_path = env::current_dir().unwrap();
        file_path.push(CODE_OBJECT_FILE);
        println!("file_path: {:?}", file_path);

        let code_object = CodeObject::new(file_path, gpu_agent).unwrap();

        let kernel = code_object
            .get_kernel("_Z21global_array_increasev.kd")
            .unwrap();

        assert_ne!(code_object.executable.handle, 0);
        assert_ne!(code_object.code_obj_rdr.handle, 0);

        assert_ne!(kernel, Kernel::create_empty());
    }

    #[test]
    fn test_code_object_create_invalid_file() {
        let system = System::new().unwrap();
        let gpu_agent = system.get_first_gpu().unwrap();

        let mut file_path = env::current_dir().unwrap();
        file_path.push("global_array-hip-amdgcn-amd-amdhsa_gfx1032.ll");

        let result = CodeObject::new(file_path, gpu_agent);
        assert_eq!(
            result,
            Err(HsaError::Code(
                hsa_status_t_HSA_STATUS_ERROR_INVALID_CODE_OBJECT
            ))
        );
    }

    #[test]
    fn test_code_object_create_kernel_error() {
        let system = System::new().unwrap();
        let gpu_agent = system.get_first_gpu().unwrap();

        let mut file_path = env::current_dir().unwrap();
        file_path.push(CODE_OBJECT_FILE);
        println!("file_path: {:?}", file_path);

        let code_object = CodeObject::new(file_path, gpu_agent).unwrap();

        // does not end with .kd
        let result = code_object.get_kernel("_Z21global_array_increasev");
        assert_eq!(
            result,
            Err(HsaError::Code(
                hsa_status_t_HSA_STATUS_ERROR_INVALID_SYMBOL_NAME
            ))
        );

        let result = code_object.get_kernel("invalid_.kd");
        assert_eq!(
            result,
            Err(HsaError::Code(
                hsa_status_t_HSA_STATUS_ERROR_INVALID_SYMBOL_NAME
            ))
        );
    }
}
