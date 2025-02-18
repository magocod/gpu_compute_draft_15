#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(improper_ctypes)]
#![allow(clippy::approx_constant)]
#![allow(clippy::useless_transmute)]

pub mod bindings;

#[cfg(test)]
mod tests {
    use crate::bindings::{clGetPlatformIDs, cl_int, CL_SUCCESS};

    #[test]
    fn test_example() {
        let mut num_platforms = 0;

        let ret = unsafe { clGetPlatformIDs(0, std::ptr::null_mut(), &mut num_platforms) };
        println!("num_platforms: {}", num_platforms);

        assert_eq!(ret, CL_SUCCESS as cl_int);
    }
}
