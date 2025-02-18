use crate::collections::set::kernel::generate_array_set_program_source;
use crate::error::{hip_check, HipResult};
use crate::memory::HipBuffer;
use crate::module::HipModule;
use hip_sys::hip_runtime_bindings::{
    hipModuleLaunchKernel, HIP_LAUNCH_PARAM_BUFFER_POINTER, HIP_LAUNCH_PARAM_BUFFER_SIZE,
    HIP_LAUNCH_PARAM_END,
};

#[derive(Debug, PartialEq)]
pub struct ArraySetSnapshot {
    pub items: Vec<i32>,
    pub entries: Vec<i32>,
}

impl ArraySetSnapshot {
    pub fn new(items: Vec<i32>, entries: Vec<i32>) -> Self {
        Self { items, entries }
    }

    pub fn create_empty(capacity: usize) -> Self {
        Self::new(vec![-1; capacity], vec![0; capacity])
    }

    pub fn sort(&mut self) {
        self.items.sort();
        self.entries.sort();
    }
}

#[derive(Debug)]
pub struct ArraySetHandle {
    capacity: usize,
    hip_module: HipModule,
}

impl ArraySetHandle {
    pub fn new(capacity: usize) -> Self {
        let src = generate_array_set_program_source(capacity);
        let hip_module = HipModule::create(&src).unwrap();
        Self {
            capacity,
            hip_module,
        }
    }

    pub fn array_set_debug(&self) -> HipResult<ArraySetSnapshot> {
        let kernel = self.hip_module.create_kernel("array_set_debug")?;

        let d_output: HipBuffer<i32> = HipBuffer::new(self.capacity * 2)?;

        unsafe {
            #[repr(C)]
            #[derive(Debug, Copy, Clone)]
            struct Args {
                output: *mut std::os::raw::c_void,
            }

            let args = Args {
                output: d_output.get_mem_ptr(),
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

        let output = d_output.memcpy_device_to_host()?;

        let items = output[0..self.capacity].to_vec();
        let entries = output[self.capacity..].to_vec();

        Ok(ArraySetSnapshot { items, entries })
    }

    pub fn array_set_reset(&self) -> HipResult<()> {
        let kernel = self.hip_module.create_kernel("array_set_reset")?;

        unsafe {
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
                std::ptr::null_mut(),
            );
            hip_check(ret)?;
        }

        Ok(())
    }

    pub fn initialize(&self) -> HipResult<()> {
        self.array_set_reset()
    }

    pub fn print(&self) -> HipResult<ArraySetSnapshot> {
        let sn = self.array_set_debug()?;
        println!("{sn:?}");
        Ok(sn)
    }

    pub fn write_in_array_set(&self, input: &[i32]) -> HipResult<Vec<i32>> {
        let kernel = self.hip_module.create_kernel("write_in_array_set")?;

        let input_size = input.len();

        let d_items_input: HipBuffer<i32> = HipBuffer::new(input_size)?;
        let d_indices_output: HipBuffer<i32> = HipBuffer::new(input_size)?;

        d_items_input.memcpy_host_to_device(input)?;

        unsafe {
            #[repr(C)]
            #[derive(Debug, Copy, Clone)]
            struct Args {
                items_input: *mut std::os::raw::c_void,
                indices_output: *mut std::os::raw::c_void,
            }

            let args = Args {
                items_input: d_items_input.get_mem_ptr(),
                indices_output: d_indices_output.get_mem_ptr(),
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

        let output = d_indices_output.memcpy_device_to_host()?;

        Ok(output)
    }

    pub fn remove_in_array_set(&self, input: &[i32]) -> HipResult<Vec<i32>> {
        let kernel = self.hip_module.create_kernel("remove_in_array_set")?;

        let input_size = input.len();

        let d_items_input: HipBuffer<i32> = HipBuffer::new(input_size)?;
        let d_indices_output: HipBuffer<i32> = HipBuffer::new(input_size)?;

        d_items_input.memcpy_host_to_device(input)?;

        unsafe {
            #[repr(C)]
            #[derive(Debug, Copy, Clone)]
            struct Args {
                items_input: *mut std::os::raw::c_void,
                indices_output: *mut std::os::raw::c_void,
            }

            let args = Args {
                items_input: d_items_input.get_mem_ptr(),
                indices_output: d_indices_output.get_mem_ptr(),
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

        let output = d_indices_output.memcpy_device_to_host()?;

        Ok(output)
    }
}

#[cfg(test)]
mod tests_array_set_debug {
    use super::*;

    #[test]
    fn array_set_is_empty() {
        let set_capacity = 32;

        let set = ArraySetHandle::new(set_capacity);

        let set_sn = set.print().unwrap();
        assert_eq!(
            set_sn,
            ArraySetSnapshot::new(vec![0; set_capacity], vec![0; set_capacity])
        );

        set.initialize().unwrap();

        let set_sn = set.print().unwrap();
        assert_eq!(set_sn, ArraySetSnapshot::create_empty(set_capacity));
    }

    #[test]
    fn array_set_is_full() {
        let set_capacity = 32;

        let set = ArraySetHandle::new(set_capacity);

        let input: Vec<i32> = (0..set_capacity as i32).collect();

        set.initialize().unwrap();
        let _ = set.write_in_array_set(&input).unwrap();

        let mut ns = set.print().unwrap();
        ns.sort();
        ns.entries.sort();

        let expected: Vec<i32> = (1..=set_capacity as i32).collect();

        assert_eq!(ns, ArraySetSnapshot::new(input.clone(), expected));
    }

    #[test]
    fn array_set_is_full_2() {
        let set_capacity = 1024;

        let set = ArraySetHandle::new(set_capacity);

        let input: Vec<i32> = (0..set_capacity as i32).collect();

        set.initialize().unwrap();
        let _ = set.write_in_array_set(&input).unwrap();

        let mut ns = set.print().unwrap();
        ns.sort();
        ns.entries.sort();

        assert_eq!(ns.items, input);
    }
}

#[cfg(test)]
mod tests_array_set_insert {
    use super::*;
    use utilities::helper_functions::has_unique_elements;

    #[test]
    fn array_set_is_empty() {
        let set_capacity = 32;

        let set = ArraySetHandle::new(set_capacity);

        let input: Vec<i32> = (0..set_capacity as i32).collect();

        set.initialize().unwrap();
        let indices = set.write_in_array_set(&input).unwrap();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        let mut ns = set.print().unwrap();
        ns.sort();

        assert_eq!(indices_sorted, input);
        assert_eq!(ns.items, input);
    }

    #[test]
    fn array_set_is_empty_2() {
        let set_capacity = 512;

        let set = ArraySetHandle::new(set_capacity);

        let input: Vec<i32> = (0..set_capacity as i32).collect();

        set.initialize().unwrap();
        let indices = set.write_in_array_set(&input).unwrap();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        let mut ns = set.print().unwrap();
        ns.sort();

        assert_eq!(indices_sorted, input);
        assert_eq!(ns.items, input);
    }

    #[test]
    fn array_set_is_full() {
        let set_capacity = 32;

        let set = ArraySetHandle::new(set_capacity);

        let input: Vec<i32> = (0..set_capacity as i32).collect();
        let input_2: Vec<i32> = input.iter().map(|x| x + set_capacity as i32).collect();

        set.initialize().unwrap();
        set.write_in_array_set(&input_2).unwrap();

        let indices = set.write_in_array_set(&input).unwrap();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        let mut ns = set.print().unwrap();
        ns.sort();

        assert_eq!(indices_sorted, vec![-1; set_capacity]);
        assert_eq!(ns.items, input_2);
    }

    #[test]
    fn partially_full_array_set() {
        let set_capacity = 32;

        let set = ArraySetHandle::new(set_capacity);

        let half = set_capacity as i32 / 2;

        let input: Vec<i32> = (0..half).collect();
        let input_2: Vec<i32> = (0..set_capacity as i32)
            .map(|x| x + set_capacity as i32)
            .collect();

        set.initialize().unwrap();
        set.write_in_array_set(&input).unwrap();

        let indices = set.write_in_array_set(&input_2).unwrap();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        let mut ns = set.print().unwrap();
        ns.sort();

        let expected: Vec<i32> = (half..(set_capacity as i32)).collect();

        let mut expected_indices = vec![-1; set_capacity / 2];
        expected_indices.append(&mut expected.clone());

        assert_eq!(indices_sorted, expected_indices);
        assert!(has_unique_elements(&ns.items));
    }

    #[test]
    fn partially_full_array_set_2() {
        let set_capacity = 512;

        let set = ArraySetHandle::new(set_capacity);

        let half = set_capacity as i32 / 2;

        let input: Vec<i32> = (0..half).collect();
        let input_2: Vec<i32> = (0..set_capacity as i32)
            .map(|x| x + set_capacity as i32)
            .collect();

        set.initialize().unwrap();
        set.write_in_array_set(&input).unwrap();

        let indices = set.write_in_array_set(&input_2).unwrap();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        let mut ns = set.print().unwrap();
        ns.sort();

        let expected: Vec<i32> = (half..(set_capacity as i32)).collect();

        let mut expected_indices = vec![-1; set_capacity / 2];
        expected_indices.append(&mut expected.clone());

        assert_eq!(indices_sorted, expected_indices);
        assert!(has_unique_elements(&ns.items));
    }

    #[test]
    fn all_values_exist() {
        let set_capacity = 32;

        let set = ArraySetHandle::new(set_capacity);

        let input: Vec<i32> = (0..set_capacity as i32).collect();

        set.initialize().unwrap();
        set.write_in_array_set(&input).unwrap();

        let indices = set.write_in_array_set(&input).unwrap();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        let mut ns = set.print().unwrap();
        ns.sort();

        assert_eq!(indices_sorted, input);
        assert_eq!(ns.items, input);
    }

    #[test]
    fn all_values_are_the_same() {
        let set_capacity = 32;

        let set = ArraySetHandle::new(set_capacity);

        let input_values: Vec<i32> = vec![1; set_capacity];

        set.initialize().unwrap();
        let _ = set.write_in_array_set(&input_values).unwrap();

        let ns = set.print().unwrap();

        let mut expected: Vec<i32> = vec![1];
        expected.append(&mut vec![-1; set_capacity - 1]);

        assert_eq!(ns.items, expected);
    }

    #[test]
    fn all_values_are_the_same_2() {
        // let set_capacity = 64;
        let set_capacity = 1024;

        let set = ArraySetHandle::new(set_capacity);

        let input_values: Vec<i32> = vec![1; set_capacity];

        set.initialize().unwrap();
        let _ = set.write_in_array_set(&input_values).unwrap();

        let ns = set.print().unwrap();

        let mut expected: Vec<i32> = vec![1];
        expected.append(&mut vec![-1; set_capacity - 1]);

        assert_eq!(ns.items, expected);
    }

    #[test]
    fn repeated_values_1() {
        let set_capacity = 32;

        let set = ArraySetHandle::new(set_capacity);

        let input_values: Vec<i32> = (0..set_capacity)
            .map(|x| if x % 2 == 0 { 1 } else { 2 })
            .collect();

        set.initialize().unwrap();
        let _ = set.write_in_array_set(&input_values).unwrap();

        let ns = set.print().unwrap();

        let mut expected: Vec<i32> = vec![1, 2];
        expected.append(&mut vec![-1; set_capacity - 2]);

        assert_eq!(ns.items, expected);
    }

    #[test]
    fn repeated_values_1_large() {
        let set_capacity = 1024;

        let set = ArraySetHandle::new(set_capacity);

        let input_values: Vec<i32> = (0..set_capacity)
            .map(|x| if x % 2 == 0 { 1 } else { 2 })
            .collect();

        set.initialize().unwrap();
        let _ = set.write_in_array_set(&input_values).unwrap();

        let ns = set.print().unwrap();

        let mut expected: Vec<i32> = vec![1, 2];
        expected.append(&mut vec![-1; set_capacity - 2]);

        assert_eq!(ns.items, expected);
    }

    #[test]
    fn repeated_values_2() {
        let set_capacity = 32;

        let set = ArraySetHandle::new(set_capacity);

        let mut input_values = vec![2; set_capacity / 2];
        input_values.append(&mut vec![3; set_capacity / 2]);

        set.initialize().unwrap();
        let _ = set.write_in_array_set(&input_values).unwrap();

        let ns = set.print().unwrap();

        let mut expected: Vec<i32> = vec![2, 3];
        expected.append(&mut vec![-1; set_capacity - 2]);

        assert_eq!(ns.items, expected);
    }

    #[test]
    fn repeated_values_2_large() {
        let set_capacity = 1024;

        let set = ArraySetHandle::new(set_capacity);

        let mut input_values = vec![2; set_capacity / 2];
        input_values.append(&mut vec![3; set_capacity / 2]);

        set.initialize().unwrap();
        let _ = set.write_in_array_set(&input_values).unwrap();

        let mut ns = set.print().unwrap();
        ns.sort();

        let mut expected: Vec<i32> = vec![-1; set_capacity - 2];
        expected.append(&mut vec![2, 3]);

        assert_eq!(ns.items, expected);
    }
}

#[cfg(test)]
mod tests_array_set_remove {
    use super::*;

    #[test]
    fn array_set_is_empty() {
        let set_capacity = 32;

        let set = ArraySetHandle::new(set_capacity);

        let input: Vec<i32> = (0..set_capacity as i32).collect();

        set.initialize().unwrap();
        let indices = set.remove_in_array_set(&input).unwrap();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        let mut ns = set.print().unwrap();
        ns.sort();

        assert_eq!(indices_sorted, vec![-1; set_capacity]);
        assert_eq!(ns, ArraySetSnapshot::create_empty(set_capacity));
    }

    #[test]
    fn none_of_the_values_exist() {
        let set_capacity = 32;

        let set = ArraySetHandle::new(set_capacity);

        let input: Vec<i32> = (0..set_capacity as i32).collect();
        let input_2: Vec<i32> = input.iter().map(|x| x + set_capacity as i32).collect();

        set.initialize().unwrap();
        set.write_in_array_set(&input_2).unwrap();

        let indices = set.remove_in_array_set(&input).unwrap();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        let mut ns = set.print().unwrap();
        ns.sort();

        assert_eq!(indices_sorted, vec![-1; set_capacity]);
        assert_eq!(ns.items, input_2);
    }

    #[test]
    fn all_values_exist() {
        let set_capacity = 32;

        let set = ArraySetHandle::new(set_capacity);

        let input: Vec<i32> = (0..set_capacity as i32).collect();

        set.initialize().unwrap();
        set.write_in_array_set(&input).unwrap();

        let indices = set.remove_in_array_set(&input).unwrap();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        let mut ns = set.print().unwrap();
        ns.sort();

        assert_eq!(indices_sorted, input);
        assert_eq!(ns, ArraySetSnapshot::create_empty(set_capacity));
    }

    #[test]
    fn all_values_are_the_same() {
        let set_capacity = 32;

        let set = ArraySetHandle::new(set_capacity);

        let input_values: Vec<i32> = vec![1; set_capacity];

        set.initialize().unwrap();
        set.write_in_array_set(&input_values).unwrap();

        let indices = set.remove_in_array_set(&input_values).unwrap();

        let mut ns = set.print().unwrap();
        ns.sort();

        assert_eq!(indices, vec![0; set_capacity]);
        assert_eq!(ns, ArraySetSnapshot::create_empty(set_capacity));
    }
}
