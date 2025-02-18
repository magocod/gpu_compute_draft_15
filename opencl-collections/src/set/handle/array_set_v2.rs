use crate::config::DEBUG_MODE;
use crate::error::OpenClResult;
use crate::set::config::SetConfig;
use crate::set::kernel::name::{
    get_set_kernel_name, ARRAY_SET_DEBUG, ARRAY_SET_RESET, REMOVE_IN_ARRAY_SET, WRITE_IN_ARRAY_SET,
};
use opencl::opencl_sys::bindings::cl_int;
use opencl::wrapper::system::OpenclCommonOperation;

#[derive(Debug, PartialEq)]
pub struct ArraySetSnapshot {
    pub items: Vec<cl_int>,
    pub entries: Vec<cl_int>,
}

impl ArraySetSnapshot {
    pub fn new(items: Vec<cl_int>, entries: Vec<cl_int>) -> Self {
        Self { items, entries }
    }

    pub fn create_empty(capacity: usize) -> Self {
        Self::new(vec![-1; capacity], vec![0; capacity])
    }

    pub fn sort(&mut self) {
        self.items.sort();
    }
}

#[derive(Debug)]
pub struct ArraySetHandle<T: OpenclCommonOperation> {
    pub config: SetConfig,
    pub(crate) system: T,
}

#[allow(dead_code)]
impl<T: OpenclCommonOperation> ArraySetHandle<T> {
    pub fn new(config: &SetConfig, system: T) -> Self {
        Self {
            config: config.clone(),
            system,
        }
    }

    pub fn get_id(&self) -> usize {
        self.config.id
    }

    pub fn debug(&self) -> OpenClResult<ArraySetSnapshot> {
        let global_work_size = self.config.capacity;
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let output_capacity = self.config.capacity * 2;

        let output_buf = self.system.create_output_buffer(output_capacity)?;

        let kernel_name = get_set_kernel_name(ARRAY_SET_DEBUG, self.get_id());
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&output_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        let output = self
            .system
            .blocking_enqueue_read_buffer(output_capacity, &output_buf, &[])?;

        let capacity = self.config.capacity;

        let items = output[0..capacity].to_vec();
        let entries = output[capacity..].to_vec();

        Ok(ArraySetSnapshot { items, entries })
    }

    pub fn print(&self) -> OpenClResult<ArraySetSnapshot> {
        let ns = self.debug()?;
        // println!("{q_s:?}");
        println!(
            "ArraySetSnapshot (
            items:   {:?},
            entries: {:?}
        )",
            ns.items, ns.entries
        );
        Ok(ns)
    }

    pub fn reset(&self) -> OpenClResult<()> {
        let global_work_size = self.config.capacity;
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let kernel_name = get_set_kernel_name(ARRAY_SET_RESET, self.get_id());
        let kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        Ok(())
    }

    pub fn initialize(&self) -> OpenClResult<()> {
        self.reset()
    }

    pub fn insert(&self, input: &[cl_int]) -> OpenClResult<Vec<cl_int>> {
        let global_work_size = input.len();
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let input_buf = self.system.blocking_prepare_input_buffer(input)?;
        let output_buf = self.system.create_output_buffer(global_work_size)?;

        let kernel_name = get_set_kernel_name(WRITE_IN_ARRAY_SET, self.get_id());
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&input_buf.get_cl_mem())?;
            kernel.set_arg(&output_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        let output =
            self.system
                .blocking_enqueue_read_buffer(global_work_size, &output_buf, &[])?;

        if DEBUG_MODE {
            println!("insert output {output:?}");
        }

        Ok(output)
    }

    pub fn remove(&self, input: &[cl_int]) -> OpenClResult<Vec<cl_int>> {
        let global_work_size = input.len();
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let input_buf = self.system.blocking_prepare_input_buffer(input)?;
        let output_buf = self.system.create_output_buffer(global_work_size)?;

        let kernel_name = get_set_kernel_name(REMOVE_IN_ARRAY_SET, self.get_id());
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&input_buf.get_cl_mem())?;
            kernel.set_arg(&output_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        let output =
            self.system
                .blocking_enqueue_read_buffer(global_work_size, &output_buf, &[])?;

        if DEBUG_MODE {
            println!("insert output {output:?}");
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests_array_set_debug {
    use super::*;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::set::config::{ArraySetVersion, SetSrc};
    use opencl::wrapper::system::System;

    #[test]
    fn array_set_is_empty() {
        let set_capacity = 32;

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V2)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

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

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V2)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input: Vec<cl_int> = (0..set_capacity as cl_int).collect();

        set.initialize().unwrap();
        let _ = set.insert(&input).unwrap();

        let mut ns = set.print().unwrap();
        ns.sort();
        ns.entries.sort();

        let expected: Vec<cl_int> = (1..=set_capacity as cl_int).collect();

        assert_eq!(ns, ArraySetSnapshot::new(input.clone(), expected));
    }

    #[test]
    fn array_set_is_full_2() {
        let set_capacity = 1024;

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V2)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input: Vec<cl_int> = (0..set_capacity as cl_int).collect();

        set.initialize().unwrap();
        let _ = set.insert(&input).unwrap();

        let mut ns = set.print().unwrap();
        ns.sort();
        ns.entries.sort();

        assert_eq!(ns.items, input);
    }
}

#[cfg(test)]
mod tests_array_set_insert {
    use super::*;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::set::config::{ArraySetVersion, SetSrc};
    use crate::utils::has_unique_elements;
    use opencl::wrapper::system::System;

    #[test]
    fn array_set_is_empty() {
        let set_capacity = 32;

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V2)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input: Vec<cl_int> = (0..set_capacity as cl_int).collect();

        set.initialize().unwrap();
        let indices = set.insert(&input).unwrap();

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

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V2)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input: Vec<cl_int> = (0..set_capacity as cl_int).collect();

        set.initialize().unwrap();
        let indices = set.insert(&input).unwrap();

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

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V2)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input: Vec<cl_int> = (0..set_capacity as cl_int).collect();
        let input_2: Vec<cl_int> = input.iter().map(|x| x + set_capacity as cl_int).collect();

        set.initialize().unwrap();
        set.insert(&input_2).unwrap();

        let indices = set.insert(&input).unwrap();

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

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V2)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let half = set_capacity as cl_int / 2;

        let input: Vec<cl_int> = (0..half).collect();
        let input_2: Vec<cl_int> = (0..set_capacity as cl_int)
            .map(|x| x + set_capacity as cl_int)
            .collect();

        set.initialize().unwrap();
        set.insert(&input).unwrap();

        let indices = set.insert(&input_2).unwrap();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        let mut ns = set.print().unwrap();
        ns.sort();

        let expected: Vec<cl_int> = (half..(set_capacity as cl_int)).collect();

        let mut expected_indices = vec![-1; set_capacity / 2];
        expected_indices.append(&mut expected.clone());

        assert_eq!(indices_sorted, expected_indices);
        assert!(has_unique_elements(&ns.items));
    }

    #[test]
    fn partially_full_array_set_2() {
        let set_capacity = 512;

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V2)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let half = set_capacity as cl_int / 2;

        let input: Vec<cl_int> = (0..half).collect();
        let input_2: Vec<cl_int> = (0..set_capacity as cl_int)
            .map(|x| x + set_capacity as cl_int)
            .collect();

        set.initialize().unwrap();
        set.insert(&input).unwrap();

        let indices = set.insert(&input_2).unwrap();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        let mut ns = set.print().unwrap();
        ns.sort();

        let expected: Vec<cl_int> = (half..(set_capacity as cl_int)).collect();

        let mut expected_indices = vec![-1; set_capacity / 2];
        expected_indices.append(&mut expected.clone());

        assert_eq!(indices_sorted, expected_indices);
        assert!(has_unique_elements(&ns.items));
    }

    #[test]
    fn all_values_exist() {
        let set_capacity = 32;

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V2)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input: Vec<cl_int> = (0..set_capacity as cl_int).collect();

        set.initialize().unwrap();
        set.insert(&input).unwrap();

        let indices = set.insert(&input).unwrap();

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

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V2)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input_values: Vec<cl_int> = vec![1; set_capacity];

        set.initialize().unwrap();
        let _ = set.insert(&input_values).unwrap();

        let ns = set.print().unwrap();

        let mut expected: Vec<cl_int> = vec![1];
        expected.append(&mut vec![-1; set_capacity - 1]);

        assert_eq!(ns.items, expected);
    }

    #[test]
    fn all_values_are_the_same_2() {
        // let set_capacity = 64;
        let set_capacity = 1024;

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V2)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input_values: Vec<cl_int> = vec![1; set_capacity];

        set.initialize().unwrap();
        let _ = set.insert(&input_values).unwrap();

        let ns = set.print().unwrap();

        let mut expected: Vec<cl_int> = vec![1];
        expected.append(&mut vec![-1; set_capacity - 1]);

        assert_eq!(ns.items, expected);
    }

    #[test]
    fn repeated_values_1() {
        let set_capacity = 32;

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V2)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input_values: Vec<cl_int> = (0..set_capacity)
            .map(|x| if x % 2 == 0 { 1 } else { 2 })
            .collect();

        set.initialize().unwrap();
        let _ = set.insert(&input_values).unwrap();

        let ns = set.print().unwrap();

        let mut expected: Vec<cl_int> = vec![1, 2];
        expected.append(&mut vec![-1; set_capacity - 2]);

        assert_eq!(ns.items, expected);
    }

    #[test]
    fn repeated_values_1_large() {
        let set_capacity = 1024;

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V2)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input_values: Vec<cl_int> = (0..set_capacity)
            .map(|x| if x % 2 == 0 { 1 } else { 2 })
            .collect();

        set.initialize().unwrap();
        let _ = set.insert(&input_values).unwrap();

        let ns = set.print().unwrap();

        let mut expected: Vec<cl_int> = vec![1, 2];
        expected.append(&mut vec![-1; set_capacity - 2]);

        assert_eq!(ns.items, expected);
    }

    #[test]
    fn repeated_values_2() {
        let set_capacity = 32;

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V2)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let mut input_values = vec![2; set_capacity / 2];
        input_values.append(&mut vec![3; set_capacity / 2]);

        set.initialize().unwrap();
        let _ = set.insert(&input_values).unwrap();

        let ns = set.print().unwrap();

        let mut expected: Vec<cl_int> = vec![2, 3];
        expected.append(&mut vec![-1; set_capacity - 2]);

        assert_eq!(ns.items, expected);
    }

    #[test]
    fn repeated_values_2_large() {
        let set_capacity = 1024;

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V2)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let mut input_values = vec![2; set_capacity / 2];
        input_values.append(&mut vec![3; set_capacity / 2]);

        set.initialize().unwrap();
        let _ = set.insert(&input_values).unwrap();

        let mut ns = set.print().unwrap();
        ns.sort();

        let mut expected: Vec<cl_int> = vec![-1; set_capacity - 2];
        expected.append(&mut vec![2, 3]);

        assert_eq!(ns.items, expected);
    }
}

#[cfg(test)]
mod tests_array_set_remove {
    use super::*;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::set::config::{ArraySetVersion, SetSrc};
    use opencl::wrapper::system::System;

    #[test]
    fn array_set_is_empty() {
        let set_capacity = 32;

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V2)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input: Vec<cl_int> = (0..set_capacity as cl_int).collect();

        set.initialize().unwrap();
        let indices = set.remove(&input).unwrap();

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

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V2)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input: Vec<cl_int> = (0..set_capacity as cl_int).collect();
        let input_2: Vec<cl_int> = input.iter().map(|x| x + set_capacity as cl_int).collect();

        set.initialize().unwrap();
        set.insert(&input_2).unwrap();

        let indices = set.remove(&input).unwrap();

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

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V2)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input: Vec<cl_int> = (0..set_capacity as cl_int).collect();

        set.initialize().unwrap();
        set.insert(&input).unwrap();

        let indices = set.remove(&input).unwrap();

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

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V2)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input_values: Vec<cl_int> = vec![1; set_capacity];

        set.initialize().unwrap();
        set.insert(&input_values).unwrap();

        let indices = set.remove(&input_values).unwrap();

        let mut ns = set.print().unwrap();
        ns.sort();

        assert_eq!(indices, vec![0; set_capacity]);
        assert_eq!(ns, ArraySetSnapshot::create_empty(set_capacity));
    }
}
