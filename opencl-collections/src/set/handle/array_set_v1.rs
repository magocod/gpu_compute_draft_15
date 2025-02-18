use crate::config::DEBUG_MODE;
use crate::error::OpenClResult;
use crate::set::config::SetConfig;
use crate::set::kernel::name::{
    get_set_kernel_name, ARRAY_SET_DEBUG, ARRAY_SET_RESET, REMOVE_IN_ARRAY_SET, WRITE_IN_ARRAY_SET,
    WRITE_WITH_CMQ_IN_ARRAY_SET, WRITE_WITH_SINGLE_THREAD_IN_ARRAY_SET,
};
use opencl::opencl_sys::bindings::{cl_int, cl_uint};
use opencl::wrapper::system::OpenclCommonOperation;

#[derive(Debug, PartialEq)]
pub struct ArraySetSnapshot {
    pub items: Vec<cl_int>,
}

impl ArraySetSnapshot {
    pub fn new(items: Vec<cl_int>) -> Self {
        Self { items }
    }

    pub fn create_empty(capacity: usize) -> Self {
        Self::new(vec![-1; capacity])
    }

    pub fn sort(&mut self) {
        self.items.sort();
    }
}

#[derive(Debug)]
pub struct ArraySetHandle<T: OpenclCommonOperation> {
    config: SetConfig,
    system: T,
}

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

        let items_output_buf = self.system.create_output_buffer(global_work_size)?;

        let kernel_name = get_set_kernel_name(ARRAY_SET_DEBUG, self.get_id());
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&items_output_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        let items_output =
            self.system
                .blocking_enqueue_read_buffer(global_work_size, &items_output_buf, &[])?;

        Ok(ArraySetSnapshot {
            items: items_output,
        })
    }

    pub fn print(&self) -> OpenClResult<ArraySetSnapshot> {
        let sn = self.debug()?;
        println!("{sn:?}");
        Ok(sn)
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

        // let local_work_size = self.system.first_device_check_local_work_size(global_work_size);
        let local_work_size = 1;

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

    pub fn insert_with_cmq(
        &self,
        input: &[cl_int],
        local_work_size: Option<usize>,
    ) -> OpenClResult<Vec<cl_int>> {
        let global_work_size = input.len();

        // error
        let local_work_size = match local_work_size {
            None => self
                .system
                .first_device_check_local_work_size(global_work_size),
            Some(v) => self.system.first_device_check_local_work_size(v),
        };

        let enqueue_kernel_output_capacity = global_work_size;

        let input_buf = self.system.blocking_prepare_input_buffer(input)?;
        let output_buf = self.system.create_output_buffer(global_work_size)?;

        let enqueue_kernel_output_buf = self
            .system
            .create_output_buffer(enqueue_kernel_output_capacity)?;

        let kernel_name = get_set_kernel_name(WRITE_WITH_CMQ_IN_ARRAY_SET, self.get_id());
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        let q0 = self
            .system
            .get_device_command_queue_0()
            .get_cl_command_queue();

        unsafe {
            kernel.set_arg(&q0)?;
            kernel.set_arg(&input_buf.get_cl_mem())?;
            kernel.set_arg(&output_buf.get_cl_mem())?;
            kernel.set_arg(&enqueue_kernel_output_buf.get_cl_mem())?;

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

        self.system.assert_device_enqueue_kernel(
            enqueue_kernel_output_capacity,
            enqueue_kernel_output_buf,
            &[],
        )?;

        Ok(output)
    }

    pub fn insert_with_single_thread(&self, input: &[cl_int]) -> OpenClResult<Vec<cl_int>> {
        let global_work_size = 1;
        let local_work_size = 1;

        let input_len = input.len();

        let input_buf = self.system.blocking_prepare_input_buffer(input)?;
        let output_buf = self.system.create_output_buffer(input_len)?;

        let kernel_name = get_set_kernel_name(WRITE_WITH_SINGLE_THREAD_IN_ARRAY_SET, self.get_id());
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        let input_global_work_size = input_len as cl_uint;

        unsafe {
            kernel.set_arg(&input_global_work_size)?;
            kernel.set_arg(&input_buf.get_cl_mem())?;
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
            .blocking_enqueue_read_buffer(input_len, &output_buf, &[])?;

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
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V1)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let set_sn = set.print().unwrap();
        assert_eq!(set_sn, ArraySetSnapshot::new(vec![0; set_capacity]));

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
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V1)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input: Vec<cl_int> = (0..set_capacity as cl_int).collect();

        set.initialize().unwrap();
        let _ = set.insert_with_single_thread(&input).unwrap();

        let mut ns = set.print().unwrap();
        ns.sort();

        assert_eq!(ns.items, input);
    }
}

#[cfg(test)]
mod tests_array_set_insert {
    use super::*;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::set::config::{ArraySetVersion, SetSrc};
    use opencl::wrapper::system::System;

    // all inserted in index 0
    #[test]
    fn array_set_is_empty() {
        let set_capacity = 32;

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V1)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input: Vec<cl_int> = (0..set_capacity as cl_int).collect();

        set.initialize().unwrap();
        let _ = set.insert(&input).unwrap();
        let mut ns = set.print().unwrap();
        ns.sort();

        assert_eq!(ns.items, input);
    }

    // the rest of the test results are very predictable (a disaster)

    // array_set_is_empty_2

    // array_set_is_full

    // all_values_exist

    // all_values_are_the_same

    // all_values_are_the_same_2
}

// local_work_size = 1
#[cfg(test)]
mod tests_array_set_insert_with_cmq_v1 {
    use super::*;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::set::config::{ArraySetVersion, SetSrc};
    use opencl::wrapper::system::System;

    const LOCAL_WORK_SIZE: Option<usize> = Some(1);

    #[test]
    fn array_set_is_empty() {
        let set_capacity = 32;

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V1)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input: Vec<cl_int> = (0..set_capacity as cl_int).collect();

        set.initialize().unwrap();
        let indices = set.insert_with_cmq(&input, LOCAL_WORK_SIZE).unwrap();

        let mut indices_sort = indices.clone();
        indices_sort.sort();

        let mut ns = set.print().unwrap();
        ns.sort();

        assert_eq!(indices_sort, input);
        assert_eq!(ns.items, input);
    }

    #[test]
    fn array_set_is_empty_2() {
        let set_capacity = 128;

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V1)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input: Vec<cl_int> = (0..set_capacity as cl_int).collect();

        set.initialize().unwrap();
        let indices = set.insert_with_cmq(&input, LOCAL_WORK_SIZE).unwrap();

        let mut indices_sort = indices.clone();
        indices_sort.sort();

        let mut ns = set.print().unwrap();
        ns.sort();

        assert_eq!(indices_sort, input);
        assert_eq!(ns.items, input);
    }

    #[test]
    fn array_set_is_full() {
        let set_capacity = 32;

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V1)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input: Vec<cl_int> = (0..set_capacity as cl_int).collect();
        let input_2: Vec<cl_int> = input.iter().map(|x| x + set_capacity as cl_int).collect();

        set.initialize().unwrap();
        set.insert_with_cmq(&input_2, LOCAL_WORK_SIZE).unwrap();

        let indices = set.insert_with_cmq(&input, LOCAL_WORK_SIZE).unwrap();

        let mut indices_sort = indices.clone();
        indices_sort.sort();

        let mut ns = set.print().unwrap();
        ns.sort();

        assert_eq!(indices_sort, vec![-1; set_capacity]);
        assert_eq!(ns.items, input_2);
    }

    #[test]
    fn all_values_exist() {
        let set_capacity = 32;

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V1)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input: Vec<cl_int> = (0..set_capacity as cl_int).collect();

        set.initialize().unwrap();
        set.insert_with_cmq(&input, LOCAL_WORK_SIZE).unwrap();

        let indices = set.insert_with_cmq(&input, LOCAL_WORK_SIZE).unwrap();

        let mut indices_sort = indices.clone();
        indices_sort.sort();

        let mut ns = set.print().unwrap();
        ns.sort();

        assert_eq!(indices_sort, input);
        assert_eq!(ns.items, input);
    }

    #[test]
    fn all_values_are_the_same() {
        let set_capacity = 32;

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V1)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input_values: Vec<cl_int> = vec![1; set_capacity];

        set.initialize().unwrap();
        let _ = set.insert_with_cmq(&input_values, LOCAL_WORK_SIZE).unwrap();

        let mut ns = set.print().unwrap();
        ns.sort();

        let mut expected: Vec<cl_int> = vec![1];
        expected.append(&mut vec![-1; set_capacity - 1]);
        expected.sort();

        assert_eq!(ns.items, expected);
    }

    #[test]
    fn all_values_are_the_same_2() {
        // let set_capacity = 64;
        let set_capacity = 128;

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V1)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input_values: Vec<cl_int> = vec![1; set_capacity];

        set.initialize().unwrap();
        let _ = set.insert_with_cmq(&input_values, LOCAL_WORK_SIZE).unwrap();

        let mut ns = set.print().unwrap();
        ns.sort();

        let mut expected: Vec<cl_int> = vec![1];
        expected.append(&mut vec![-1; set_capacity - 1]);
        expected.sort();

        assert_eq!(ns.items, expected);
    }
}

// local_work_size variable = disaster
#[cfg(test)]
mod tests_array_set_insert_with_cmq_v2 {
    use super::*;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::set::config::{ArraySetVersion, SetSrc};
    use opencl::wrapper::system::System;

    const LOCAL_WORK_SIZE: Option<usize> = None;

    #[test]
    fn array_set_is_empty() {
        let set_capacity = 32;

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V1)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input: Vec<cl_int> = (0..set_capacity as cl_int).collect();

        set.initialize().unwrap();
        let indices = set.insert_with_cmq(&input, LOCAL_WORK_SIZE).unwrap();

        let mut indices_sort = indices.clone();
        indices_sort.sort();

        let mut ns = set.print().unwrap();
        ns.sort();

        assert_eq!(indices_sort, input);
        assert_eq!(ns.items, input);
    }

    #[test]
    fn array_set_is_empty_2() {
        let set_capacity = 128;

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V1)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input: Vec<cl_int> = (0..set_capacity as cl_int).collect();

        set.initialize().unwrap();
        let indices = set.insert_with_cmq(&input, LOCAL_WORK_SIZE).unwrap();

        let mut indices_sort = indices.clone();
        indices_sort.sort();

        let mut ns = set.print().unwrap();
        ns.sort();

        assert_eq!(indices_sort, input);
        assert_eq!(ns.items, input);
    }

    #[test]
    fn array_set_is_full() {
        let set_capacity = 32;

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V1)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input: Vec<cl_int> = (0..set_capacity as cl_int).collect();
        let input_2: Vec<cl_int> = input.iter().map(|x| x + set_capacity as cl_int).collect();

        set.initialize().unwrap();
        set.insert_with_cmq(&input_2, LOCAL_WORK_SIZE).unwrap();

        let indices = set.insert_with_cmq(&input, LOCAL_WORK_SIZE).unwrap();

        let mut indices_sort = indices.clone();
        indices_sort.sort();

        let mut ns = set.print().unwrap();
        ns.sort();

        assert_eq!(indices_sort, vec![-1; set_capacity]);
        assert_eq!(ns.items, input_2);
    }

    #[test]
    fn all_values_exist() {
        let set_capacity = 32;

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V1)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input: Vec<cl_int> = (0..set_capacity as cl_int).collect();

        set.initialize().unwrap();
        set.insert_with_cmq(&input, LOCAL_WORK_SIZE).unwrap();

        let indices = set.insert_with_cmq(&input, LOCAL_WORK_SIZE).unwrap();

        let mut indices_sort = indices.clone();
        indices_sort.sort();

        let mut ns = set.print().unwrap();
        ns.sort();

        assert_eq!(indices_sort, input);
        assert_eq!(ns.items, input);
    }

    #[test]
    fn all_values_are_the_same() {
        let set_capacity = 32;

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V1)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input_values: Vec<cl_int> = vec![1; set_capacity];

        set.initialize().unwrap();
        let _ = set.insert_with_cmq(&input_values, LOCAL_WORK_SIZE).unwrap();

        let mut ns = set.print().unwrap();
        ns.sort();

        let mut expected: Vec<cl_int> = vec![1];
        expected.append(&mut vec![-1; set_capacity - 1]);
        expected.sort();

        assert_eq!(ns.items, expected);
    }

    #[test]
    fn all_values_are_the_same_2() {
        // let set_capacity = 64;
        let set_capacity = 128;

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V1)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input_values: Vec<cl_int> = vec![1; set_capacity];

        set.initialize().unwrap();
        let _ = set.insert_with_cmq(&input_values, LOCAL_WORK_SIZE).unwrap();

        let mut ns = set.print().unwrap();
        ns.sort();

        let mut expected: Vec<cl_int> = vec![1];
        expected.append(&mut vec![-1; set_capacity - 1]);
        expected.sort();

        assert_eq!(ns.items, expected);
    }
}

#[cfg(test)]
mod tests_array_set_insert_with_single_thread {
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
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V1)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input: Vec<cl_int> = (0..set_capacity as cl_int).collect();

        set.initialize().unwrap();
        let _ = set.insert_with_single_thread(&input).unwrap();

        let mut ns = set.print().unwrap();
        ns.sort();

        assert_eq!(ns.items, input);
    }

    #[test]
    fn array_set_is_empty_2() {
        let set_capacity = 64;

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V1)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input: Vec<cl_int> = (0..set_capacity as cl_int).collect();

        set.initialize().unwrap();
        let indices = set.insert_with_single_thread(&input).unwrap();

        let mut indices_sort = indices.clone();
        indices_sort.sort();

        let mut ns = set.print().unwrap();
        ns.sort();

        assert_eq!(indices_sort, input);
        assert_eq!(ns.items, input);
    }

    #[test]
    fn array_set_is_full() {
        let set_capacity = 32;

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V1)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input: Vec<cl_int> = (0..set_capacity as cl_int).collect();
        let input_2: Vec<cl_int> = input.iter().map(|x| x + set_capacity as cl_int).collect();

        set.initialize().unwrap();
        set.insert_with_single_thread(&input_2).unwrap();

        let indices = set.insert_with_single_thread(&input).unwrap();

        let mut indices_sort = indices.clone();
        indices_sort.sort();

        let mut ns = set.print().unwrap();
        ns.sort();

        assert_eq!(indices_sort, vec![-1; set_capacity]);
        assert_eq!(ns.items, input_2);
    }

    #[test]
    fn all_values_exist() {
        let set_capacity = 32;

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V1)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input: Vec<cl_int> = (0..set_capacity as cl_int).collect();

        set.initialize().unwrap();
        set.insert_with_single_thread(&input).unwrap();

        let indices = set.insert_with_single_thread(&input).unwrap();

        let mut indices_sort = indices.clone();
        indices_sort.sort();

        let mut ns = set.print().unwrap();
        ns.sort();

        assert_eq!(indices_sort, input);
        assert_eq!(ns.items, input);
    }

    #[test]
    fn all_values_are_the_same() {
        let set_capacity = 32;

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V1)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input_values: Vec<cl_int> = vec![1; set_capacity];

        set.initialize().unwrap();
        let indices = set.insert_with_single_thread(&input_values).unwrap();

        let mut ns = set.print().unwrap();
        ns.sort();

        let mut expected: Vec<cl_int> = vec![1];
        expected.append(&mut vec![-1; set_capacity - 1]);
        expected.sort();

        assert_eq!(indices, vec![0; set_capacity]);
        assert_eq!(ns.items, expected);
    }

    #[test]
    fn all_values_are_the_same_2() {
        let set_capacity = 128;

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V1)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input_values: Vec<cl_int> = vec![1; set_capacity];

        set.initialize().unwrap();
        let indices = set.insert_with_single_thread(&input_values).unwrap();

        let mut ns = set.print().unwrap();
        ns.sort();

        let mut expected: Vec<cl_int> = vec![1];
        expected.append(&mut vec![-1; set_capacity - 1]);
        expected.sort();

        assert_eq!(indices, vec![0; set_capacity]);
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
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V1)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input: Vec<cl_int> = (0..set_capacity as cl_int).collect();

        set.initialize().unwrap();
        let indices = set.remove(&input).unwrap();

        let mut indices_sort = indices.clone();
        indices_sort.sort();

        let mut ns = set.print().unwrap();
        ns.sort();

        assert_eq!(indices_sort, vec![-1; set_capacity]);
        assert_eq!(ns, ArraySetSnapshot::create_empty(set_capacity));
    }

    #[test]
    fn none_of_the_values_exist() {
        let set_capacity = 32;

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V1)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input: Vec<cl_int> = (0..set_capacity as cl_int).collect();
        let input_2: Vec<cl_int> = input.iter().map(|x| x + set_capacity as cl_int).collect();

        set.initialize().unwrap();
        set.insert_with_single_thread(&input_2).unwrap();

        let indices = set.remove(&input).unwrap();

        let mut indices_sort = indices.clone();
        indices_sort.sort();

        let mut ns = set.print().unwrap();
        ns.sort();

        assert_eq!(indices_sort, vec![-1; set_capacity]);
        assert_eq!(ns.items, input_2);
    }

    #[test]
    fn all_values_exist() {
        let set_capacity = 32;

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V1)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input: Vec<cl_int> = (0..set_capacity as cl_int).collect();

        set.initialize().unwrap();
        set.insert_with_cmq(&input, Some(1)).unwrap();

        let indices = set.remove(&input).unwrap();

        let mut indices_sort = indices.clone();
        indices_sort.sort();

        let mut ns = set.print().unwrap();
        ns.sort();

        assert_eq!(indices_sort, input);
        assert_eq!(ns, ArraySetSnapshot::create_empty(set_capacity));
    }

    #[test]
    fn all_values_are_the_same() {
        let set_capacity = 32;

        let mut set_src = SetSrc::new();
        set_src.add(set_capacity);

        let system =
            System::new(DEFAULT_DEVICE_INDEX, &set_src.build(ArraySetVersion::V1)).unwrap();

        let config = set_src.get_config_by_id(0).unwrap();
        let set = ArraySetHandle::new(config, system);

        let input_values: Vec<cl_int> = vec![1; set_capacity];

        set.initialize().unwrap();
        set.insert_with_single_thread(&input_values).unwrap();

        let indices = set.remove(&input_values).unwrap();

        let mut ns = set.print().unwrap();
        ns.sort();

        assert_eq!(indices, vec![0; set_capacity]);
        assert_eq!(ns, ArraySetSnapshot::create_empty(set_capacity));
    }
}
