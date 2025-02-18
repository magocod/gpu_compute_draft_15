use crate::collections::stack::kernel::generate_stack_program_source;
use crate::error::{hip_check, HipResult};
use crate::memory::HipBuffer;
use crate::module::HipModule;
use hip_sys::hip_runtime_bindings::{
    hipModuleLaunchKernel, HIP_LAUNCH_PARAM_BUFFER_POINTER, HIP_LAUNCH_PARAM_BUFFER_SIZE,
    HIP_LAUNCH_PARAM_END,
};

#[derive(Debug, PartialEq)]
pub struct StackSnapshot {
    pub top: i32,
    pub items: Vec<i32>,
}

impl StackSnapshot {
    pub fn create_empty(capacity: usize) -> Self {
        Self {
            top: -1,
            items: vec![0; capacity],
        }
    }
}

#[derive(Debug)]
pub struct StackHandle {
    capacity: usize,
    hip_module: HipModule,
}

impl StackHandle {
    pub fn new(capacity: usize) -> Self {
        let src = generate_stack_program_source(capacity);
        let hip_module = HipModule::create(&src).unwrap();
        Self {
            capacity,
            hip_module,
        }
    }

    pub fn stack_debug(&self) -> HipResult<StackSnapshot> {
        let kernel = self.hip_module.create_kernel("stack_debug")?;

        let d_items_output: HipBuffer<i32> = HipBuffer::new(self.capacity)?;
        let d_meta_output: HipBuffer<i32> = HipBuffer::new(1)?;

        unsafe {
            #[repr(C)]
            #[derive(Debug, Copy, Clone)]
            struct Args {
                items_output: *mut std::os::raw::c_void,
                meta_output: *mut std::os::raw::c_void,
            }

            let args = Args {
                items_output: d_items_output.get_mem_ptr(),
                meta_output: d_meta_output.get_mem_ptr(),
            };

            let size = std::mem::size_of_val(&args);

            let mut config = [
                HIP_LAUNCH_PARAM_BUFFER_POINTER as *mut std::os::raw::c_void,
                &args as *const _ as *mut std::os::raw::c_void,
                HIP_LAUNCH_PARAM_BUFFER_SIZE as *mut std::os::raw::c_void,
                &size as *const _ as *mut std::os::raw::c_void,
                HIP_LAUNCH_PARAM_END as *mut std::os::raw::c_void,
            ];

            let ret = hipModuleLaunchKernel(
                kernel,
                1,
                1,
                1,
                self.capacity as u32,
                1,
                1,
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                config.as_mut_ptr(),
            );
            hip_check(ret)?;
        }

        let items = d_items_output.memcpy_device_to_host()?;
        let meta = d_meta_output.memcpy_device_to_host()?;

        Ok(StackSnapshot {
            items,
            top: meta[0],
        })
    }

    pub fn print(&self) -> HipResult<StackSnapshot> {
        let sn = self.stack_debug()?;
        println!("{sn:?}");
        Ok(sn)
    }

    pub fn write_to_stack(&self, input: &[i32]) -> HipResult<Vec<i32>> {
        let kernel = self.hip_module.create_kernel("write_to_stack")?;

        let input_size = input.len();

        let d_input: HipBuffer<i32> = HipBuffer::new(input_size)?;
        let d_output: HipBuffer<i32> = HipBuffer::new(input_size)?;

        d_input.memcpy_host_to_device(input)?;

        unsafe {
            #[repr(C)]
            #[derive(Debug, Copy, Clone)]
            struct Args {
                input: *mut std::os::raw::c_void,
                output: *mut std::os::raw::c_void,
            }

            let args = Args {
                input: d_input.get_mem_ptr(),
                output: d_output.get_mem_ptr(),
            };
            // let args_size = std::mem::size_of_val(&args);
            // let mut config = prepare_kernel_config(args);

            let size = std::mem::size_of_val(&args);

            let mut config = [
                HIP_LAUNCH_PARAM_BUFFER_POINTER as *mut std::os::raw::c_void,
                &args as *const _ as *mut std::os::raw::c_void,
                HIP_LAUNCH_PARAM_BUFFER_SIZE as *mut std::os::raw::c_void,
                &size as *const _ as *mut std::os::raw::c_void,
                HIP_LAUNCH_PARAM_END as *mut std::os::raw::c_void,
            ];

            let ret = hipModuleLaunchKernel(
                kernel,
                1,
                1,
                1,
                input_size as u32,
                1,
                1,
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                config.as_mut_ptr(),
            );
            hip_check(ret)?;
        }

        let output = d_output.memcpy_device_to_host()?;

        Ok(output)
    }

    pub fn read_on_stack(&self, take: usize) -> HipResult<Vec<i32>> {
        let kernel = self.hip_module.create_kernel("read_on_stack")?;

        let d_output: HipBuffer<i32> = HipBuffer::new(take)?;

        unsafe {
            #[repr(C)]
            #[derive(Debug, Copy, Clone)]
            struct Args {
                output: *mut std::os::raw::c_void,
            }

            let args = Args {
                output: d_output.get_mem_ptr(),
            };
            // let args_size = std::mem::size_of_val(&args);
            // let mut config = prepare_kernel_config(args);

            let size = std::mem::size_of_val(&args);

            let mut config = [
                HIP_LAUNCH_PARAM_BUFFER_POINTER as *mut std::os::raw::c_void,
                &args as *const _ as *mut std::os::raw::c_void,
                HIP_LAUNCH_PARAM_BUFFER_SIZE as *mut std::os::raw::c_void,
                &size as *const _ as *mut std::os::raw::c_void,
                HIP_LAUNCH_PARAM_END as *mut std::os::raw::c_void,
            ];

            let ret = hipModuleLaunchKernel(
                kernel,
                1,
                1,
                1,
                take as u32,
                1,
                1,
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                config.as_mut_ptr(),
            );
            hip_check(ret)?;
        }

        let output = d_output.memcpy_device_to_host()?;

        Ok(output)
    }
}

#[cfg(test)]
mod tests_stack_debug {
    use super::*;

    #[test]
    fn stack_is_empty() {
        let stack_capacity = 32;

        let st = StackHandle::new(stack_capacity);

        let st_sn = st.print().unwrap();

        assert_eq!(st_sn, StackSnapshot::create_empty(stack_capacity));
    }

    #[test]
    fn stack_is_full() {
        let stack_capacity = 32;

        let st = StackHandle::new(stack_capacity);

        let input: Vec<i32> = (0..stack_capacity as i32).collect();
        let _ = st.write_to_stack(&input).unwrap();
        let st_sn = st.print().unwrap();

        assert_eq!(st_sn.top, (stack_capacity as i32) - 1);
        assert_eq!(st_sn.items, input);
    }

    #[test]
    fn partially_full_stack() {
        let stack_capacity = 32;

        let st = StackHandle::new(stack_capacity);

        let input: Vec<i32> = (0..(stack_capacity / 2) as i32).collect();
        let _ = st.write_to_stack(&input).unwrap();

        let st_sn = st.print().unwrap();

        let mut expected = input.clone();
        expected.append(&mut vec![0; stack_capacity / 2]);

        assert_eq!(st_sn.top, (stack_capacity as i32 / 2) - 1);
        assert_eq!(st_sn.items, expected);
    }

    #[test]
    fn overflowing_stack() {
        let stack_capacity = 32;

        let st = StackHandle::new(stack_capacity);

        let input: Vec<i32> = (0..(stack_capacity * 2) as i32).collect();
        let _st_in = st.write_to_stack(&input).unwrap();
        let _st_out = st.read_on_stack(stack_capacity * 2).unwrap();

        let q_s = st.print().unwrap();

        assert_eq!(q_s.top, -1);
        assert_eq!(q_s.items, &input[0..stack_capacity]);
    }
}

#[cfg(test)]
mod tests_stack_push {
    use super::*;

    #[test]
    fn stack_is_empty() {
        let stack_capacity = 32;

        let st = StackHandle::new(stack_capacity);

        let input: Vec<i32> = (0..stack_capacity as i32).collect();
        let result = st.write_to_stack(&input).unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        assert_eq!(result_sorted, input);

        st.print().unwrap();
    }

    // FIXME rename test
    #[test]
    fn stack_is_empty_2() {
        let stack_capacity = 1024;

        let st = StackHandle::new(stack_capacity);

        let input: Vec<i32> = (0..stack_capacity as i32).collect();
        let result = st.write_to_stack(&input).unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        let expected_indices: Vec<i32> = (0..stack_capacity).map(|x| x as i32).collect();
        assert_eq!(result_sorted, expected_indices);

        st.print().unwrap();
    }

    // FIXME rename test, the test consists of inserting data into the stack and there is space left
    #[test]
    fn stack_is_empty_3() {
        let stack_capacity = 32;

        let st = StackHandle::new(stack_capacity);

        let input: Vec<i32> = (0..(stack_capacity / 2) as i32).collect();

        let result = st.write_to_stack(&input).unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        let expected_indices: Vec<i32> = (0..(stack_capacity / 2) as i32).collect();
        assert_eq!(result_sorted, expected_indices);

        st.print().unwrap();
    }

    #[test]
    fn stack_is_full() {
        let stack_capacity = 32;

        let st = StackHandle::new(stack_capacity);

        let input: Vec<i32> = (0..stack_capacity as i32).collect();
        let input_b: Vec<i32> = input.iter().map(|x| x + stack_capacity as i32).collect();

        let _ = st.write_to_stack(&input).unwrap();
        let result = st.write_to_stack(&input_b).unwrap();

        assert_eq!(result, vec![-1; stack_capacity]);

        st.print().unwrap();
    }

    #[test]
    fn stack_without_enough_space() {
        let stack_capacity = 512;

        let st = StackHandle::new(stack_capacity);

        let input: Vec<i32> = (0..(stack_capacity * 2) as i32).collect();

        let result = st.write_to_stack(&input).unwrap();

        assert_eq!(result.iter().filter(|&&x| x < 0).count(), stack_capacity);
        assert_eq!(result.iter().filter(|&&x| x >= 0).count(), stack_capacity);

        st.print().unwrap();
    }
}

#[cfg(test)]
mod tests_stack_pop {
    use super::*;

    #[test]
    fn stack_is_empty() {
        let stack_capacity = 32;

        let st = StackHandle::new(stack_capacity);

        let result = st.read_on_stack(stack_capacity).unwrap();

        assert_eq!(result, vec![-1; stack_capacity]);

        st.print().unwrap();
    }

    #[test]
    fn stack_is_full() {
        let stack_capacity = 32;

        let st = StackHandle::new(stack_capacity);

        let input: Vec<i32> = (0..stack_capacity as i32).collect();
        let _ = st.write_to_stack(&input).unwrap();

        let result = st.read_on_stack(stack_capacity).unwrap();
        let mut result_sorted = result.clone();
        result_sorted.sort();

        assert_eq!(result_sorted, input);

        st.print().unwrap();
    }

    #[test]
    fn stack_is_full_2() {
        let stack_capacity = 1024;

        let st = StackHandle::new(stack_capacity);

        let input: Vec<i32> = (0..stack_capacity as i32).collect();
        let _ = st.write_to_stack(&input).unwrap();
        let result = st.read_on_stack(stack_capacity).unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();
        assert_eq!(result_sorted, input);

        st.print().unwrap();
    }

    #[test]
    fn partially_full_stack() {
        let stack_capacity = 64;

        let st = StackHandle::new(stack_capacity);

        let input: Vec<i32> = (0..(stack_capacity / 2) as i32).collect();
        let _ = st.write_to_stack(&input).unwrap();

        let result = st.read_on_stack(stack_capacity).unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        let mut expected = vec![-1; stack_capacity / 2];
        expected.append(&mut input.clone());

        assert_eq!(result_sorted, expected);

        st.print().unwrap();
    }

    // FIXME rename tests - The test consists of removing data from the stack and not leaving it empty
    #[test]
    fn simple_case() {
        let stack_capacity = 128;

        let st = StackHandle::new(stack_capacity);

        let input: Vec<i32> = (0..stack_capacity as i32).collect();
        let _ = st.write_to_stack(&input).unwrap();

        let _result = st.read_on_stack(stack_capacity / 2).unwrap();

        // FIXME assert result
        st.print().unwrap();
    }

    // FIXME rename tests - The test consists of multiple calls to the stack until it is empty
    #[test]
    fn stack_emptied() {
        let stack_capacity = 32;

        let st = StackHandle::new(stack_capacity);

        let input: Vec<i32> = (0..stack_capacity as i32).collect();
        let _ = st.write_to_stack(&input).unwrap();
        let _ = st.read_on_stack(stack_capacity).unwrap();

        let result = st.read_on_stack(stack_capacity).unwrap();
        assert_eq!(result, vec![-1; 32]);

        let result = st.read_on_stack(stack_capacity).unwrap();
        assert_eq!(result, vec![-1; 32]);

        st.print().unwrap();
    }

    // FIXME rename tests - The test consists of multiple calls to the stack until it is empty
    #[test]
    fn stack_emptied_2() {
        let stack_capacity = 64;

        let st = StackHandle::new(stack_capacity);

        let input: Vec<i32> = (0..stack_capacity as i32).collect();
        let _ = st.write_to_stack(&input).unwrap();

        let result = st.read_on_stack(stack_capacity * 2).unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        let mut expected = vec![-1; stack_capacity];
        expected.append(&mut input.clone());

        assert_eq!(result_sorted, expected);

        let result = st.read_on_stack(stack_capacity).unwrap();
        assert_eq!(result, vec![-1; stack_capacity]);

        st.print().unwrap();
    }
}
