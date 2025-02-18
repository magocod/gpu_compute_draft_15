use crate::error::{hip_check, HipResult};
use hip_sys::hip_runtime_bindings::{
    hipError_t_hipSuccess, hipFree, hipMalloc, hipMemcpy, hipMemcpyKind_hipMemcpyDeviceToHost,
    hipMemcpyKind_hipMemcpyHostToDevice,
};

#[derive(Debug)]
pub struct HipBuffer<T: Copy + Clone + Default> {
    size: usize,
    size_bytes: usize,
    mem_ptr: *mut std::os::raw::c_void,
    phantom: std::marker::PhantomData<T>,
}

impl<T: Copy + Clone + Default> HipBuffer<T> {
    pub fn new(size: usize) -> HipResult<HipBuffer<T>> {
        let size_bytes = std::mem::size_of::<T>() * size;

        let mut mem_ptr: *mut std::os::raw::c_void = std::ptr::null_mut();

        let ret = unsafe { hipMalloc(&mut mem_ptr, size_bytes) };
        hip_check(ret)?;

        Ok(Self {
            size,
            size_bytes,
            mem_ptr,
            phantom: std::marker::PhantomData,
        })
    }

    pub fn get_mem_ptr(&self) -> *mut std::os::raw::c_void {
        self.mem_ptr
    }

    pub fn memcpy_host_to_device(&self, host_input: &[T]) -> HipResult<()> {
        let ret = unsafe {
            hipMemcpy(
                self.mem_ptr,
                host_input.as_ptr() as *const std::os::raw::c_void,
                self.size_bytes,
                hipMemcpyKind_hipMemcpyHostToDevice,
            )
        };
        hip_check(ret)
    }

    pub fn memcpy_device_to_host(&self) -> HipResult<Vec<T>> {
        let mut output = vec![T::default(); self.size];

        let ret = unsafe {
            hipMemcpy(
                output.as_mut_ptr() as *mut std::os::raw::c_void,
                self.mem_ptr,
                self.size_bytes,
                hipMemcpyKind_hipMemcpyDeviceToHost,
            )
        };
        hip_check(ret)?;

        Ok(output)
    }
}

impl<T: Copy + Clone + Default> Drop for HipBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            let ret = hipFree(self.mem_ptr);
            if ret != hipError_t_hipSuccess {
                panic!("error: hipFree");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hip_buffer_create() {
        let size = 32;

        let hip_buffer: HipBuffer<i32> = HipBuffer::new(size).unwrap();

        assert_eq!(hip_buffer.size, 32);
        assert_eq!(hip_buffer.size_bytes, std::mem::size_of::<i32>() * size);
    }
}
