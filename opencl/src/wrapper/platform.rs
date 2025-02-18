//! # Opencl platform, devices safe wrapper (FFI - Foreign Function Interface)
//!
//! Platform
//!
//! ...
//!
//! Devices
//!
//! ...
//!

use crate::error::{OclError, OclResult, CL_WRAPPER_FIRST_PLATFORM_NOT_FOUND};
use crate::unsafe_wrapper::{
    cl_get_device_ids, cl_get_device_info, cl_get_platform_ids, cl_get_platform_info, DeviceInfo,
    PlatformInfo,
};
use opencl_sys::bindings::{cl_device_id, cl_platform_id};

#[derive(Debug)]
pub struct Platform {
    platform_id: cl_platform_id,
    pub info: PlatformInfo,
}

impl Platform {
    // FIXME defines whether only the declaration of this structure is allowed using the first method (make fn private)
    /// Using clGetPlatformInfo, verify that the reference to platform_id is valid (at least at creation time).
    fn new(platform_id: cl_platform_id) -> OclResult<Self> {
        // SAFETY: The platform_id reference is made within this same function,
        // all parameters are passed to clGetPlatformInfo (except platform_id),
        // none are obtained externally, guaranteed statically.
        let platform_info = unsafe { cl_get_platform_info(platform_id)? };

        Ok(Self {
            platform_id,
            info: platform_info,
        })
    }

    pub fn get_cl_platform_id(&self) -> cl_platform_id {
        self.platform_id
    }

    /// get the first available platform
    pub fn first() -> OclResult<Platform> {
        let platforms: Vec<cl_platform_id> = cl_get_platform_ids()?;

        if platforms.is_empty() {
            return Err(OclError::Wrapper(CL_WRAPPER_FIRST_PLATFORM_NOT_FOUND));
        }

        let platform_id = platforms[0];

        Self::new(platform_id)
    }

    /// get all devices type CL_DEVICE_TYPE_GPU
    pub fn get_gpu_devices(&self) -> OclResult<Vec<Device>> {
        // SAFETY: The parameters provided to clGetDeviceIDs are done statically (except platform_id).
        // the platform_id parameter the only current method that has this structure
        // is through the first method. !!! new fix case
        let device_ids = unsafe { cl_get_device_ids(self.platform_id)? };
        let mut devices = Vec::with_capacity(device_ids.len());

        for device_id in device_ids {
            devices.push(Device::new(device_id)?);
        }

        Ok(devices)
    }

    // ...
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Device {
    device_id: cl_device_id,
    max_work_group_size: usize,
}

impl Device {
    /// Using clGetDeviceInfo, verify that the reference to device_id is valid (at least at creation time).
    pub fn new(device_id: cl_device_id) -> OclResult<Self> {
        let mut d = Self {
            device_id,
            max_work_group_size: 0,
        };

        // check device_id ?
        // This function currently loads things that are not necessarily required in this context,
        // to maintain simplicity in the early stages of the project,
        // an optimization is omitted here
        let device_info = d.info()?;
        d.max_work_group_size = device_info.max_work_group_size;

        Ok(d)
    }

    #[cfg(test)]
    pub fn create(device_id: cl_device_id, max_work_group_size: usize) -> Self {
        Self {
            device_id,
            max_work_group_size,
        }
    }

    pub fn get_cl_device_id(&self) -> cl_device_id {
        self.device_id
    }

    pub fn info(&self) -> OclResult<DeviceInfo> {
        // SAFETY: The parameters provided to clGetDeviceIDs are done statically (except device_id).
        unsafe { cl_get_device_info(self.device_id) }
    }

    /// Verifies that a given total work (global_work_size) for a one-dimensional kernel (work_dim = 1)
    /// is not larger than the device capacity; If it is greater, it will return the
    /// CL_DEVICE_MAX_WORK_GROUP_SIZE value of the device, otherwise it will not modify the value
    /// (in future versions should it return the most optimal number for the operation?)
    ///
    /// ```rust
    /// use opencl::wrapper::platform::Platform;
    ///
    /// let platform = Platform::first().unwrap();
    /// let devices = platform.get_gpu_devices().unwrap();
    ///
    /// // device CL_DEVICE_MAX_WORK_GROUP_SIZE = 256
    /// let device = devices[0];
    ///
    /// // work required
    /// let global_work_size = 1024;
    ///
    /// // return 256;
    /// let local_work_size = device.check_local_work_size(global_work_size);
    ///
    /// // work required
    /// let global_work_size = 128;
    ///
    /// // return 128;
    /// let local_work_size = device.check_local_work_size(global_work_size);
    ///  
    /// ```
    pub fn check_local_work_size(&self, global_work_size: usize) -> usize {
        if global_work_size >= self.max_work_group_size {
            return self.max_work_group_size;
        }

        global_work_size
    }

    // ...
}

#[cfg(test)]
mod tests_platform {
    use super::*;
    use opencl_sys::bindings::{clGetDeviceIDs, cl_device_type, CL_DEVICE_TYPE_GPU, CL_SUCCESS};

    #[test]
    fn test_platform_first() {
        let platform = Platform::first().unwrap();
        let platform_id = platform.get_cl_platform_id();

        // check info
        assert!(!platform.info.profile.is_empty());
        assert!(!platform.info.version.is_empty());
        assert!(!platform.info.name.is_empty());

        // check platform id reference
        let ret = unsafe {
            let mut num_devices = 0;

            clGetDeviceIDs(
                platform_id,
                CL_DEVICE_TYPE_GPU as cl_device_type,
                0,
                std::ptr::null_mut(),
                &mut num_devices,
            )
        };

        assert_eq!(ret, CL_SUCCESS as i32);
    }

    // TODO test_get_gpu_devices
}

#[cfg(test)]
mod tests_device {
    use super::*;
    use crate::error::OclError;
    use crate::unsafe_wrapper::{cl_get_device_ids, cl_get_device_info, cl_get_platform_ids};
    use opencl_sys::bindings::{cl_device_id, CL_INVALID_DEVICE};

    #[test]
    fn test_device_new() {
        let platforms = cl_get_platform_ids().unwrap();
        let platform_id = platforms[0];

        let devices = unsafe { cl_get_device_ids(platform_id).unwrap() };
        let device_id = devices[0];

        // valid reference
        let device = Device::new(device_id).unwrap();
        assert!(device.max_work_group_size > 0);

        // invalid reference
        let device_id = std::ptr::null_mut() as *mut _ as cl_device_id;
        let result = Device::new(device_id);
        assert_eq!(result, Err(OclError::Code(CL_INVALID_DEVICE)));
    }

    #[test]
    fn test_device_check_local_work_size() {
        let platforms = cl_get_platform_ids().unwrap();
        let platform_id = platforms[0];

        let devices = unsafe { cl_get_device_ids(platform_id).unwrap() };
        let device_id = devices[0];

        let device_info = unsafe { cl_get_device_info(device_id).unwrap() };

        // CL_DEVICE_MAX_WORK_GROUP_SIZE
        let device_max_work_group_size = device_info.max_work_group_size;

        let device = Device::new(device_id).unwrap();
        assert_eq!(device.max_work_group_size, device_max_work_group_size);

        let global_work_size = device_max_work_group_size * 2;

        let local_work_size = device.check_local_work_size(global_work_size);
        assert_eq!(local_work_size, device_max_work_group_size);

        let global_work_size = device_max_work_group_size / 2;

        let local_work_size = device.check_local_work_size(global_work_size);
        assert_eq!(local_work_size, global_work_size);
    }
}
