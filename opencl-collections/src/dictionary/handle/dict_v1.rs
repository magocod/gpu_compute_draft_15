use crate::config::{ClTypeTrait, DEBUG_MODE};
use crate::dictionary::config::DictConfig;
use crate::dictionary::handle::{DictSnapshotSummary, DictSummary};
use crate::dictionary::kernel::name::{
    get_dict_kernel_name, DICT_DEBUG, DICT_GET_KEYS, DICT_GET_SUMMARY, DICT_RESET, READ_ON_DICT,
    READ_VALUE_SIZE_ON_DICT, REMOVE_FROM_DICT, VERIFY_AND_REMOVE_IN_DICT, VERIFY_AND_WRITE_IN_DICT,
    WRITE_TO_DICT,
};
use crate::error::OpenClResult;
use crate::utils::ensure_vec_size;
use opencl::opencl_sys::bindings::{cl_int, cl_uint};
use opencl::wrapper::system::OpenclCommonOperation;
use std::marker::PhantomData;

#[derive(Debug, PartialEq)]
pub struct DictSnapshot<T: ClTypeTrait> {
    config: DictConfig<T>,
    pub keys: Vec<Vec<T>>,
    pub values: Vec<Vec<T>>,
    pub entries: Vec<cl_int>,
}

impl<T: ClTypeTrait> DictSnapshot<T> {
    pub fn new(
        config: &DictConfig<T>,
        keys: Vec<Vec<T>>,
        values: Vec<Vec<T>>,
        entries: Vec<cl_int>,
    ) -> Self {
        Self {
            config: config.clone(),
            keys,
            values,
            entries,
        }
    }

    pub fn create_empty(config: &DictConfig<T>) -> Self {
        let keys = vec![vec![T::cl_default(); config.key_len]; config.capacity];
        let values = vec![vec![T::cl_default(); config.value_len]; config.capacity];

        Self::new(config, keys, values, vec![0; config.capacity])
    }

    // debug
    pub fn print_all_keys(&self) {
        for (i, key) in self.keys.iter().enumerate() {
            println!("index: {} - key: {:?}", i, key);
        }
    }

    pub fn print_all_values(&self) {
        for (i, value) in self.values.iter().enumerate() {
            println!("index: {} - value: {:?}", i, value);
        }
    }

    pub fn print_all_entries(&self) {
        for (i, key) in self.keys.iter().enumerate() {
            println!("index: {} - key:     {:?}", i, key);
            println!("index: {} - value:   {:?}", i, self.values[i]);
            println!("index: {} - entries: {:?}", i, self.entries[i]);
        }
    }

    pub fn summary(&self) -> DictSnapshotSummary {
        DictSnapshotSummary::new(
            self.keys
                .iter()
                .filter(|&x| x.iter().any(|&x| x != T::cl_default()))
                .count(),
            self.values
                .iter()
                .filter(|&x| x.iter().any(|&x| x != T::cl_default()))
                .count(),
        )
    }

    pub fn has_entry(&self, key: &Vec<T>, value: &Vec<T>) -> bool {
        match self.keys.iter().position(|x| x == key) {
            None => false,
            Some(index) => &self.values[index] == value,
        }
    }
}

#[derive(Debug)]
pub struct DictHandle<T: ClTypeTrait, D: OpenclCommonOperation> {
    config: DictConfig<T>,
    system: D,
    phantom: PhantomData<T>,
}

impl<T: ClTypeTrait, D: OpenclCommonOperation> DictHandle<T, D> {
    pub fn new(config: &DictConfig<T>, system: D) -> Self {
        Self {
            config: config.clone(),
            system,
            phantom: Default::default(),
        }
    }

    pub fn get_id(&self) -> usize {
        self.config.id
    }

    pub fn get_value_len(&self) -> usize {
        self.config.value_len
    }

    pub fn debug(&self) -> OpenClResult<DictSnapshot<T>> {
        let global_work_size = self.config.capacity;
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let keys_output_capacity = self.config.key_len * global_work_size;
        let values_output_capacity = self.config.value_len * global_work_size;
        let entries_output_capacity = global_work_size;

        let keys_output_buf = self.system.create_output_buffer(keys_output_capacity)?;
        let values_output_buf = self.system.create_output_buffer(values_output_capacity)?;

        let entries_output_buf = self.system.create_output_buffer(entries_output_capacity)?;

        let kernel_name = get_dict_kernel_name(DICT_DEBUG, self.get_id());
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&keys_output_buf.get_cl_mem())?;
            kernel.set_arg(&values_output_buf.get_cl_mem())?;
            kernel.set_arg(&entries_output_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        let keys_output = self.system.blocking_enqueue_read_buffer(
            keys_output_capacity,
            &keys_output_buf,
            &[],
        )?;

        let values_output = self.system.blocking_enqueue_read_buffer(
            values_output_capacity,
            &values_output_buf,
            &[],
        )?;

        let entries_output = self.system.blocking_enqueue_read_buffer(
            entries_output_capacity,
            &entries_output_buf,
            &[],
        )?;

        let keys: Vec<Vec<_>> = keys_output
            .chunks(self.config.key_len)
            .map(|x| x.to_vec())
            .collect();

        let values: Vec<Vec<_>> = values_output
            .chunks(self.config.value_len)
            .map(|x| x.to_vec())
            .collect();

        Ok(DictSnapshot::new(
            &self.config,
            keys,
            values,
            entries_output,
        ))
    }

    pub fn print(&self) -> OpenClResult<DictSnapshot<T>> {
        let ds = self.debug()?;

        // println!("keys");
        // ds.print_all_keys();
        //
        // println!("values");
        // ds.print_all_values();

        println!("dict entries");
        ds.print_all_entries();

        println!("entries: {:?}", ds.entries);

        Ok(ds)
    }

    pub fn reset(&self) -> OpenClResult<()> {
        let global_work_size = self.config.capacity;
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let kernel_name = get_dict_kernel_name(DICT_RESET, self.get_id());
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

    pub fn insert(&self, keys: &Vec<Vec<T>>, values: &Vec<Vec<T>>) -> OpenClResult<Vec<cl_int>> {
        if keys.len() != values.len() {
            panic!("error handle keys & values error len")
        }

        let global_work_size = keys.len();
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let keys_input_capacity = global_work_size * self.config.key_len;
        let values_input_capacity = global_work_size * self.config.value_len;
        let indices_output_capacity = global_work_size;

        let mut keys_input: Vec<_> = Vec::with_capacity(keys_input_capacity);

        for key in keys {
            let mut id_input = ensure_vec_size(key, self.config.key_len);
            keys_input.append(&mut id_input);
        }

        let mut values_input: Vec<_> = Vec::with_capacity(values_input_capacity);

        for b in values {
            let mut v = ensure_vec_size(b, self.config.value_len);
            values_input.append(&mut v);
        }

        let keys_input_buf = self.system.blocking_prepare_input_buffer(&keys_input)?;
        let values_input_buf = self.system.blocking_prepare_input_buffer(&values_input)?;

        let indices_output_buf = self.system.create_output_buffer(indices_output_capacity)?;

        let key_len = self.config.key_len as cl_uint;
        let value_len = self.config.value_len as cl_uint;

        let kernel_name = get_dict_kernel_name(WRITE_TO_DICT, self.get_id());
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&key_len)?;
            kernel.set_arg(&value_len)?;
            kernel.set_arg(&keys_input_buf.get_cl_mem())?;
            kernel.set_arg(&values_input_buf.get_cl_mem())?;
            kernel.set_arg(&indices_output_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        let indices_output = self.system.blocking_enqueue_read_buffer(
            indices_output_capacity,
            &indices_output_buf,
            &[],
        )?;

        if DEBUG_MODE {
            println!("insert indices_output {indices_output:?}");
        }

        Ok(indices_output)
    }

    pub fn insert_with_verification(
        &self,
        keys: &Vec<Vec<T>>,
        values: &Vec<Vec<T>>,
    ) -> OpenClResult<Vec<cl_int>> {
        if keys.len() <= 1 {
            panic!("error handle input len 1");
        }

        if keys.len() != values.len() {
            panic!("error handle keys & values error len")
        }

        let global_work_size = 1;
        let local_work_size = 1;

        let keys_len = keys.len();

        let keys_input_capacity = keys_len * self.config.key_len;
        let values_input_capacity = keys_len * self.config.value_len;
        let indices_output_capacity = keys_len;
        let enqueue_kernel_output_capacity = 2;

        let mut keys_input: Vec<_> = Vec::with_capacity(keys_input_capacity);

        for key in keys {
            let mut id_input = ensure_vec_size(key, self.config.key_len);
            keys_input.append(&mut id_input);
        }

        let mut values_input: Vec<_> = Vec::with_capacity(values_input_capacity);

        for b in values {
            let mut v = ensure_vec_size(b, self.config.value_len);
            values_input.append(&mut v);
        }

        let keys_input_buf = self.system.blocking_prepare_input_buffer(&keys_input)?;
        let values_input_buf = self.system.blocking_prepare_input_buffer(&values_input)?;

        let indices_output_buf = self.system.create_output_buffer(indices_output_capacity)?;

        let enqueue_kernel_output_buf = self
            .system
            .create_output_buffer(enqueue_kernel_output_capacity)?;

        let keys_global_work_size = keys_len as cl_uint;
        let keys_local_work_size =
            self.system.first_device_check_local_work_size(keys_len) as cl_uint;

        let q0 = self
            .system
            .get_device_command_queue_0()
            .get_cl_command_queue();

        let kernel_name = get_dict_kernel_name(VERIFY_AND_WRITE_IN_DICT, self.get_id());
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&q0)?;
            kernel.set_arg(&keys_global_work_size)?;
            kernel.set_arg(&keys_local_work_size)?;
            kernel.set_arg(&keys_input_buf.get_cl_mem())?;
            kernel.set_arg(&values_input_buf.get_cl_mem())?;
            kernel.set_arg(&indices_output_buf.get_cl_mem())?;
            kernel.set_arg(&enqueue_kernel_output_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        let indices_output = self.system.blocking_enqueue_read_buffer(
            indices_output_capacity,
            &indices_output_buf,
            &[],
        )?;

        if DEBUG_MODE {
            println!("insert indices_output {indices_output:?}");
        }

        self.system.assert_device_enqueue_kernel(
            enqueue_kernel_output_capacity,
            enqueue_kernel_output_buf,
            &[],
        )?;

        Ok(indices_output)
    }

    pub fn remove(&self, keys: &Vec<Vec<T>>) -> OpenClResult<Vec<cl_int>> {
        let global_work_size = keys.len();
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let keys_input_capacity = global_work_size * self.config.key_len;
        let indices_output_capacity = global_work_size;

        let mut keys_input: Vec<_> = Vec::with_capacity(keys_input_capacity);

        for key in keys {
            let mut id_input = ensure_vec_size(key, self.config.key_len);
            keys_input.append(&mut id_input);
        }

        let keys_input_buf = self.system.blocking_prepare_input_buffer(&keys_input)?;

        let indices_output_buf = self.system.create_output_buffer(indices_output_capacity)?;

        let kernel_name = get_dict_kernel_name(REMOVE_FROM_DICT, self.get_id());
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        let key_len = self.config.key_len as cl_uint;

        unsafe {
            kernel.set_arg(&key_len)?;
            kernel.set_arg(&keys_input_buf.get_cl_mem())?;
            kernel.set_arg(&indices_output_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        let indices_output = self.system.blocking_enqueue_read_buffer(
            indices_output_capacity,
            &indices_output_buf,
            &[],
        )?;

        if DEBUG_MODE {
            println!("remove indices_output {indices_output:?}");
        }

        Ok(indices_output)
    }

    pub fn remove_with_verification(&self, keys: &Vec<Vec<T>>) -> OpenClResult<Vec<cl_int>> {
        if keys.len() <= 1 {
            panic!("error handle input len 1");
        }

        let global_work_size = 1;
        let local_work_size = 1;

        let keys_len = keys.len();

        let keys_input_capacity = keys_len * self.config.key_len;
        let indices_output_capacity = keys_len;
        let enqueue_kernel_output_capacity = 2;

        let mut keys_input: Vec<_> = Vec::with_capacity(keys_input_capacity);

        for key in keys {
            let mut id_input = ensure_vec_size(key, self.config.key_len);
            keys_input.append(&mut id_input);
        }

        let keys_input_buf = self.system.blocking_prepare_input_buffer(&keys_input)?;

        let indices_output_buf = self.system.create_output_buffer(indices_output_capacity)?;

        let enqueue_kernel_output_buf = self
            .system
            .create_output_buffer(enqueue_kernel_output_capacity)?;

        let keys_global_work_size = keys_len as cl_uint;
        let keys_local_work_size =
            self.system.first_device_check_local_work_size(keys_len) as cl_uint;

        let q0 = self
            .system
            .get_device_command_queue_0()
            .get_cl_command_queue();

        let kernel_name = get_dict_kernel_name(VERIFY_AND_REMOVE_IN_DICT, self.get_id());
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&q0)?;
            kernel.set_arg(&keys_global_work_size)?;
            kernel.set_arg(&keys_local_work_size)?;
            kernel.set_arg(&keys_input_buf.get_cl_mem())?;
            kernel.set_arg(&indices_output_buf.get_cl_mem())?;
            kernel.set_arg(&enqueue_kernel_output_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        let indices_output = self.system.blocking_enqueue_read_buffer(
            indices_output_capacity,
            &indices_output_buf,
            &[],
        )?;

        if DEBUG_MODE {
            println!("remove indices_output {indices_output:?}");
        }

        self.system.assert_device_enqueue_kernel(
            enqueue_kernel_output_capacity,
            enqueue_kernel_output_buf,
            &[],
        )?;

        Ok(indices_output)
    }

    pub fn keys(&self) -> OpenClResult<Vec<Vec<T>>> {
        let global_work_size = self.config.capacity;
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let keys_output_capacity = self.config.key_len * global_work_size;
        let keys_output_buf = self.system.create_output_buffer(keys_output_capacity)?;

        let kernel_name = get_dict_kernel_name(DICT_GET_KEYS, self.get_id());
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&keys_output_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        let keys_output = self.system.blocking_enqueue_read_buffer(
            keys_output_capacity,
            &keys_output_buf,
            &[],
        )?;

        let keys: Vec<Vec<_>> = keys_output
            .chunks(self.config.key_len)
            .map(|x| x.to_vec())
            .collect();

        if DEBUG_MODE {
            for (i, value) in keys.iter().enumerate() {
                println!("index: {} - k: {:?}", i, value);
            }
        }

        Ok(keys)
    }

    pub fn get(&self, keys: &Vec<Vec<T>>) -> OpenClResult<(Vec<Vec<T>>, Vec<cl_int>)> {
        let global_work_size = self.config.capacity;
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let keys_input_capacity = self.config.key_len * global_work_size;
        let values_output_capacity = self.config.value_len * global_work_size;
        let indices_output_capacity = global_work_size;

        let mut keys_input: Vec<_> = Vec::with_capacity(keys_input_capacity);

        for key in keys {
            let mut id_input = ensure_vec_size(key, self.config.key_len);
            keys_input.append(&mut id_input);
        }

        let keys_input_buf = self.system.blocking_prepare_input_buffer(&keys_input)?;

        let values_output_buf = self.system.create_output_buffer(values_output_capacity)?;

        let indices_output_buf = self.system.create_output_buffer(indices_output_capacity)?;

        let key_len = self.config.key_len as cl_uint;
        let value_len = self.config.value_len as cl_uint;

        let kernel_name = get_dict_kernel_name(READ_ON_DICT, self.get_id());
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&key_len)?;
            kernel.set_arg(&value_len)?;
            kernel.set_arg(&keys_input_buf.get_cl_mem())?;
            kernel.set_arg(&values_output_buf.get_cl_mem())?;
            kernel.set_arg(&indices_output_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        let values_output = self.system.blocking_enqueue_read_buffer(
            values_output_capacity,
            &values_output_buf,
            &[],
        )?;

        let values: Vec<Vec<_>> = values_output
            .chunks(self.config.value_len)
            .map(|x| x.to_vec())
            .collect();

        let indices_output = self.system.blocking_enqueue_read_buffer(
            indices_output_capacity,
            &indices_output_buf,
            &[],
        )?;

        if DEBUG_MODE {
            for (i, value) in values.iter().enumerate() {
                println!("n: {} - i: {} - v: {:?}", i, indices_output[i], value);
            }
        }

        Ok((values, indices_output))
    }

    pub fn get_size(&self, keys: &Vec<Vec<T>>) -> OpenClResult<(Vec<cl_uint>, Vec<cl_int>)> {
        let global_work_size = self.config.capacity;
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let keys_input_capacity = self.config.key_len * global_work_size;
        let sizes_output_capacity = global_work_size;
        let indices_output_capacity = global_work_size;

        let mut keys_input: Vec<_> = Vec::with_capacity(keys_input_capacity);

        for key in keys {
            let mut id_input = ensure_vec_size(key, self.config.key_len);
            keys_input.append(&mut id_input);
        }

        let keys_input_buf = self.system.blocking_prepare_input_buffer(&keys_input)?;

        let sizes_output_buf = self.system.create_output_buffer(sizes_output_capacity)?;

        let indices_output_buf = self.system.create_output_buffer(indices_output_capacity)?;

        let key_len = self.config.key_len as cl_uint;

        let kernel_name = get_dict_kernel_name(READ_VALUE_SIZE_ON_DICT, self.get_id());
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&key_len)?;
            kernel.set_arg(&keys_input_buf.get_cl_mem())?;
            kernel.set_arg(&sizes_output_buf.get_cl_mem())?;
            kernel.set_arg(&indices_output_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        let sizes_output = self.system.blocking_enqueue_read_buffer(
            sizes_output_capacity,
            &sizes_output_buf,
            &[],
        )?;

        let indices_output = self.system.blocking_enqueue_read_buffer(
            indices_output_capacity,
            &indices_output_buf,
            &[],
        )?;

        if DEBUG_MODE {
            println!("sizes_output   {sizes_output:?}");
            println!("indices_output {indices_output:?}");
        }

        Ok((sizes_output, indices_output))
    }

    pub fn summary(&self) -> OpenClResult<DictSummary<T>> {
        let global_work_size = 1;
        let local_work_size = 1;

        let sizes_output_capacity = self.config.capacity * 2;
        // 0 = DictSummary.reserved
        let meta_output_capacity = 1;
        let enqueue_kernel_output_capacity = 2;

        let sizes_output_buf = self.system.create_output_buffer(sizes_output_capacity)?;

        let meta_output_buf = self
            .system
            .create_output_buffer::<cl_int>(meta_output_capacity)?;

        let enqueue_kernel_output_buf = self
            .system
            .create_output_buffer(enqueue_kernel_output_capacity)?;

        let q0 = self
            .system
            .get_device_command_queue_0()
            .get_cl_command_queue();

        let capacity_device_local_work_size =
            self.system
                .first_device_check_local_work_size(self.config.capacity) as cl_uint;

        let kernel_name = get_dict_kernel_name(DICT_GET_SUMMARY, self.get_id());
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&q0)?;
            kernel.set_arg(&capacity_device_local_work_size)?;
            kernel.set_arg(&sizes_output_buf.get_cl_mem())?;
            kernel.set_arg(&meta_output_buf.get_cl_mem())?;
            kernel.set_arg(&enqueue_kernel_output_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        let sizes_output = self.system.blocking_enqueue_read_buffer(
            sizes_output_capacity,
            &sizes_output_buf,
            &[],
        )?;

        let meta_output = self.system.blocking_enqueue_read_buffer(
            meta_output_capacity,
            &meta_output_buf,
            &[],
        )?;

        let keys_sizes = &sizes_output[..self.config.capacity];
        let values_sizes = &sizes_output[self.config.capacity..];

        let meta_key_available = meta_output[0] as usize;

        let summary = DictSummary::new(&self.config, keys_sizes, values_sizes, meta_key_available);

        if DEBUG_MODE {
            println!("config: {:#?}", summary.config);
            println!("key_sizes {:?}", summary.keys_sizes);
            println!("value_sizes {:?}", summary.values_sizes);
            println!("available {:#?}", summary.available);
        }

        // assert enqueue kernels
        self.system.assert_device_enqueue_kernel(
            enqueue_kernel_output_capacity,
            enqueue_kernel_output_buf,
            &[],
        )?;

        Ok(summary)
    }
}

#[cfg(test)]
mod tests_dict_reset {
    use super::*;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::dictionary::config::DictSrc;
    use opencl::wrapper::system::System;

    #[test]
    fn dict_is_empty() {
        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(32, 32, 8);
        dict_src.add(64, 64, 16);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);

        let keys = vec![vec![0; 32]; 8];
        let values = vec![vec![0; 32]; 8];
        let entries = vec![0; 8];

        assert_eq!(
            dict.print().unwrap(),
            DictSnapshot::new(config, keys, values, entries)
        );

        dict.reset().unwrap();

        assert_eq!(dict.print().unwrap(), DictSnapshot::create_empty(config));
    }
}

#[cfg(test)]
mod tests_dict_debug {
    use super::*;
    use crate::config::{ClTypeDefault, DEFAULT_DEVICE_INDEX};
    use crate::dictionary::config::DictSrc;
    use crate::test_utils::TestMatrix;
    use opencl::wrapper::system::System;

    #[test]
    fn dict_is_empty() {
        let dict_capacity = 64;
        let dict_key_len = 256;
        let dict_value_len = 256;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let ds = dict.print().unwrap();

        assert_eq!(ds, DictSnapshot::create_empty(config));
    }

    #[test]
    fn dict_is_full() {
        let dict_capacity = 64;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let test_matrix = TestMatrix::new(dict_capacity, dict_key_len, dict_value_len, 1, 10);

        let indices = dict.insert(&test_matrix.keys, &test_matrix.values).unwrap();
        let ds = dict.print().unwrap();

        for (i, di) in indices.into_iter().enumerate() {
            let entry_index = di as usize;
            assert_eq!(ds.keys[entry_index], test_matrix.keys[i]);
            assert_eq!(ds.values[entry_index], test_matrix.values[i]);
        }

        assert_eq!(ds.entries.iter().filter(|&&x| x == 0).count(), 0);
    }

    #[test]
    fn partially_full_dict() {
        let dict_capacity = 64;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let test_matrix = TestMatrix::new(dict_capacity / 2, dict_key_len, dict_value_len, 1, 10);

        let indices = dict.insert(&test_matrix.keys, &test_matrix.values).unwrap();
        let ds = dict.print().unwrap();

        for (i, di) in indices.iter().enumerate() {
            let entry_index = *di as usize;
            assert_eq!(ds.keys[entry_index], test_matrix.keys[i]);
            assert_eq!(ds.values[entry_index], test_matrix.values[i]);
        }

        for i in 0..dict_capacity {
            let i_i32 = i as cl_int;
            if !indices.contains(&i_i32) {
                assert_eq!(ds.keys[i], vec![i16::cl_default(); dict_key_len]);
                assert_eq!(ds.values[i], vec![i16::cl_default(); dict_key_len]);
            }
        }

        assert_eq!(
            ds.entries.iter().filter(|&&x| x > 0).count(),
            dict_capacity / 2
        );
        assert_eq!(
            ds.entries.iter().filter(|&&x| x == 0).count(),
            dict_capacity / 2
        );
    }
}

#[cfg(test)]
mod tests_dict_insert {
    use super::*;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::dictionary::config::DictSrc;
    use crate::dictionary::handle::KEYS_NOT_AVAILABLE;
    use crate::test_utils::TestMatrix;
    use crate::utils::is_all_same;
    use opencl::wrapper::system::System;

    #[test]
    fn dict_is_empty() {
        let dict_capacity = 64;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let test_matrix = TestMatrix::new(dict_capacity, dict_key_len, dict_value_len, 1, 10);

        let indices = dict.insert(&test_matrix.keys, &test_matrix.values).unwrap();

        let expected: Vec<cl_int> = (0..dict_capacity as cl_int).collect();
        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        let ds = dict.print().unwrap();

        assert_eq!(indices_sorted, expected);

        for (i, di) in indices.into_iter().enumerate() {
            let entry_index = di as usize;
            assert_eq!(ds.keys[entry_index], test_matrix.keys[i]);
            assert_eq!(ds.values[entry_index], test_matrix.values[i]);
        }
    }

    #[test]
    fn dict_is_empty_2() {
        let dict_capacity = 1024;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let test_matrix = TestMatrix::new(dict_capacity, dict_key_len, dict_value_len, 1, 10);

        let indices = dict.insert(&test_matrix.keys, &test_matrix.values).unwrap();

        let expected: Vec<cl_int> = (0..dict_capacity as cl_int).collect();
        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        let ds = dict.print().unwrap();

        assert_eq!(indices_sorted, expected);

        for (i, di) in indices.into_iter().enumerate() {
            let entry_index = di as usize;
            assert_eq!(ds.keys[entry_index], test_matrix.keys[i]);
            assert_eq!(ds.values[entry_index], test_matrix.values[i]);
        }
    }

    #[test]
    fn dict_is_full() {
        let dict_capacity = 64;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let test_matrix = TestMatrix::new(dict_capacity, dict_key_len, dict_value_len, 1, 10);
        let test_matrix_2 = TestMatrix::new(
            dict_capacity,
            dict_key_len,
            dict_value_len,
            1 + dict_capacity as i16,
            100,
        );

        let indices_1 = dict.insert(&test_matrix.keys, &test_matrix.values).unwrap();

        let indices_2 = dict
            .insert(&test_matrix_2.keys, &test_matrix_2.values)
            .unwrap();
        assert_eq!(
            indices_2,
            vec![KEYS_NOT_AVAILABLE; test_matrix_2.keys.len()]
        );

        let ds = dict.print().unwrap();

        for (i, di) in indices_1.into_iter().enumerate() {
            let entry_index = di as usize;
            assert_eq!(ds.keys[entry_index], test_matrix.keys[i]);
            assert_eq!(ds.values[entry_index], test_matrix.values[i]);
        }
    }

    #[test]
    fn partially_full_dict() {
        let dict_capacity = 64;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let test_matrix = TestMatrix::new(dict_capacity / 2, dict_key_len, dict_value_len, 1, 10);
        let test_matrix_2 = TestMatrix::new(
            dict_capacity,
            dict_key_len,
            dict_value_len,
            1 + dict_capacity as i16,
            20,
        );

        let indices_1 = dict.insert(&test_matrix.keys, &test_matrix.values).unwrap();

        let indices_2 = dict
            .insert(&test_matrix_2.keys, &test_matrix_2.values)
            .unwrap();
        assert_eq!(
            indices_2
                .iter()
                .filter(|&&x| x == KEYS_NOT_AVAILABLE)
                .count(),
            dict_capacity / 2
        );
        assert_eq!(
            indices_2.iter().filter(|&&x| x >= 0).count(),
            dict_capacity / 2
        );

        let ds = dict.print().unwrap();

        for (i, di) in indices_1.into_iter().enumerate() {
            let entry_index = di as usize;
            assert_eq!(ds.keys[entry_index], test_matrix.keys[i]);
            assert_eq!(ds.values[entry_index], test_matrix.values[i]);
        }

        for (i, di) in indices_2.into_iter().enumerate() {
            if di >= 0 {
                let entry_index = di as usize;
                assert_eq!(ds.keys[entry_index], test_matrix_2.keys[i]);
                assert_eq!(ds.values[entry_index], test_matrix_2.values[i]);
            }
        }
    }

    #[test]
    fn keys_already_exist() {
        let dict_capacity = 64;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let test_matrix = TestMatrix::new(dict_capacity, dict_key_len, dict_value_len, 1, 10);

        let _ = dict.insert(&test_matrix.keys, &test_matrix.values).unwrap();

        let test_matrix_2 = TestMatrix::new(dict_capacity, dict_key_len, dict_value_len, 1, 100);

        let indices = dict
            .insert(&test_matrix_2.keys, &test_matrix_2.values)
            .unwrap();

        let expected: Vec<cl_int> = (0..dict_capacity as cl_int).collect();
        let mut indices_sorted = indices.clone();
        indices_sorted.sort();
        assert_eq!(indices_sorted, expected);

        let ds = dict.print().unwrap();

        for (i, di) in indices.into_iter().enumerate() {
            let entry_index = di as usize;
            assert_eq!(ds.keys[entry_index], test_matrix.keys[i]);
            assert_eq!(ds.values[entry_index], test_matrix_2.values[i]);
        }
    }

    #[test]
    fn keys_already_exist_2() {
        let dict_capacity = 1024;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let test_matrix = TestMatrix::new(dict_capacity, dict_key_len, dict_value_len, 1, 10);

        let _ = dict.insert(&test_matrix.keys, &test_matrix.values).unwrap();

        let test_matrix_2 = TestMatrix::new(dict_capacity, dict_key_len, dict_value_len, 1, 100);

        let indices = dict
            .insert(&test_matrix_2.keys, &test_matrix_2.values)
            .unwrap();

        let expected: Vec<cl_int> = (0..dict_capacity as cl_int).collect();
        let mut indices_sorted = indices.clone();
        indices_sorted.sort();
        assert_eq!(indices_sorted, expected);

        let ds = dict.print().unwrap();

        for (i, di) in indices.into_iter().enumerate() {
            let entry_index = di as usize;
            assert_eq!(ds.keys[entry_index], test_matrix.keys[i]);
            assert_eq!(ds.values[entry_index], test_matrix_2.values[i]);
        }
    }

    #[test]
    fn all_keys_are_the_same() {
        let dict_capacity = 64;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let key = vec![1; dict_key_len];
        let value = vec![2; dict_value_len];

        let keys = vec![key.clone(); dict_capacity];
        let values = vec![value.clone(); dict_capacity];

        let indices = dict.insert(&keys, &values).unwrap();

        let ds = dict.print().unwrap();

        // TODO assert test
        assert!(ds.has_entry(&key, &value));
        assert!(is_all_same(&indices));
        assert_eq!(ds.summary(), DictSnapshotSummary::with(1));
    }
}

#[cfg(test)]
mod tests_dict_insert_with_verification {
    use super::*;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::dictionary::config::DictSrc;
    use crate::dictionary::handle::{DUPLICATE_KEY, KEYS_NOT_AVAILABLE};
    use crate::test_utils::TestMatrix;
    use opencl::wrapper::system::System;

    #[test]
    fn dict_is_empty() {
        let dict_capacity = 64;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let test_matrix = TestMatrix::new(dict_capacity, dict_key_len, dict_value_len, 1, 10);

        let indices = dict
            .insert_with_verification(&test_matrix.keys, &test_matrix.values)
            .unwrap();

        let expected: Vec<cl_int> = (0..dict_capacity as cl_int).collect();
        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        assert_eq!(indices_sorted, expected);

        let ds = dict.print().unwrap();

        for (i, di) in indices.into_iter().enumerate() {
            let entry_index = di as usize;
            assert_eq!(ds.keys[entry_index], test_matrix.keys[i]);
            assert_eq!(ds.values[entry_index], test_matrix.values[i]);
        }
    }

    #[test]
    fn dict_is_full() {
        let dict_capacity = 64;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let test_matrix = TestMatrix::new(dict_capacity, dict_key_len, dict_value_len, 1, 10);
        let test_matrix_2 = TestMatrix::new(
            dict_capacity,
            dict_key_len,
            dict_value_len,
            1 + dict_capacity as i16,
            100,
        );

        let indices_1 = dict
            .insert_with_verification(&test_matrix.keys, &test_matrix.values)
            .unwrap();

        let indices_2 = dict
            .insert_with_verification(&test_matrix_2.keys, &test_matrix_2.values)
            .unwrap();
        assert_eq!(
            indices_2,
            vec![KEYS_NOT_AVAILABLE; test_matrix_2.keys.len()]
        );

        let ds = dict.print().unwrap();

        for (i, di) in indices_1.into_iter().enumerate() {
            let entry_index = di as usize;
            assert_eq!(ds.keys[entry_index], test_matrix.keys[i]);
            assert_eq!(ds.values[entry_index], test_matrix.values[i]);
        }
    }

    #[test]
    fn partially_full_dict() {
        let dict_capacity = 64;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let test_matrix = TestMatrix::new(dict_capacity / 2, dict_key_len, dict_value_len, 1, 10);
        let test_matrix_2 = TestMatrix::new(
            dict_capacity,
            dict_key_len,
            dict_value_len,
            1 + dict_capacity as i16,
            20,
        );

        let indices_1 = dict
            .insert_with_verification(&test_matrix.keys, &test_matrix.values)
            .unwrap();

        let indices_2 = dict
            .insert_with_verification(&test_matrix_2.keys, &test_matrix_2.values)
            .unwrap();

        assert_eq!(
            indices_2
                .iter()
                .filter(|&&x| x == KEYS_NOT_AVAILABLE)
                .count(),
            dict_capacity / 2
        );
        assert_eq!(
            indices_2.iter().filter(|&&x| x >= 0).count(),
            dict_capacity / 2
        );

        let ds = dict.print().unwrap();

        for (i, di) in indices_1.into_iter().enumerate() {
            let entry_index = di as usize;
            assert_eq!(ds.keys[entry_index], test_matrix.keys[i]);
            assert_eq!(ds.values[entry_index], test_matrix.values[i]);
        }

        for (i, di) in indices_2.into_iter().enumerate() {
            if di >= 0 {
                let entry_index = di as usize;
                assert_eq!(ds.keys[entry_index], test_matrix_2.keys[i]);
                assert_eq!(ds.values[entry_index], test_matrix_2.values[i]);
            }
        }
    }

    #[test]
    fn keys_already_exist() {
        let dict_capacity = 64;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let test_matrix = TestMatrix::new(dict_capacity, dict_key_len, dict_value_len, 1, 10);

        let _ = dict
            .insert_with_verification(&test_matrix.keys, &test_matrix.values)
            .unwrap();

        let test_matrix_2 = TestMatrix::new(dict_capacity, dict_key_len, dict_value_len, 1, 100);

        let indices = dict
            .insert_with_verification(&test_matrix_2.keys, &test_matrix_2.values)
            .unwrap();

        let expected: Vec<cl_int> = (0..dict_capacity as cl_int).collect();
        let mut indices_sorted = indices.clone();
        indices_sorted.sort();
        assert_eq!(indices_sorted, expected);

        let ds = dict.print().unwrap();

        for (i, di) in indices.into_iter().enumerate() {
            let entry_index = di as usize;
            assert_eq!(ds.keys[entry_index], test_matrix.keys[i]);
            assert_eq!(ds.values[entry_index], test_matrix_2.values[i]);
        }
    }

    #[test]
    fn all_keys_are_the_same() {
        let dict_capacity = 64;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let key = vec![1; dict_key_len];
        let value = vec![2; dict_value_len];

        let keys = vec![key.clone(); dict_capacity];
        let values = vec![value.clone(); dict_capacity];

        let indices = dict.insert_with_verification(&keys, &values).unwrap();

        let ds = dict.print().unwrap();

        // TODO assert test
        assert!(ds.has_entry(&key, &value));
        // assert!(is_all_same(&indices));
        assert_eq!(indices.iter().filter(|&&x| x >= 0).count(), 1);
        assert_eq!(
            indices.iter().filter(|&&x| x == DUPLICATE_KEY).count(),
            dict_capacity - 1
        );
        assert_eq!(ds.summary(), DictSnapshotSummary::with(1));
    }
}

#[cfg(test)]
mod tests_dict_remove {
    use super::*;
    use crate::config::{ClTypeDefault, DEFAULT_DEVICE_INDEX};
    use crate::dictionary::config::DictSrc;
    use crate::dictionary::handle::KEY_NOT_EXIST;
    use crate::test_utils::TestMatrix;
    use opencl::wrapper::system::System;

    #[test]
    fn dict_is_empty() {
        let dict_capacity = 64;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let test_matrix = TestMatrix::new(dict_capacity, dict_key_len, dict_value_len, 1, 10);
        let indices = dict.remove(&test_matrix.keys).unwrap();

        assert_eq!(indices, vec![KEY_NOT_EXIST; test_matrix.keys.len()]);

        let ds = dict.print().unwrap();
        for key in ds.keys.into_iter() {
            assert_eq!(key, vec![i16::cl_default(); dict_key_len]);
        }
        for value in ds.values.into_iter() {
            assert_eq!(value, vec![i16::cl_default(); dict_value_len]);
        }
    }

    #[test]
    fn dict_is_full() {
        let dict_capacity = 64;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let test_matrix = TestMatrix::new(dict_capacity, dict_key_len, dict_value_len, 1, 10);

        let insert_indices = dict.insert(&test_matrix.keys, &test_matrix.values).unwrap();

        let keys = &test_matrix.keys[..(dict_capacity / 2)].to_vec();
        let indices = dict.remove(keys).unwrap();

        let mut expected = insert_indices[..(dict_capacity / 2)].to_vec();
        expected.sort();
        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        assert_eq!(indices_sorted, expected);

        let ds = dict.print().unwrap();

        for ei in insert_indices[..(dict_capacity / 2)].iter() {
            let entry_index = *ei as usize;
            assert_eq!(ds.keys[entry_index], vec![i16::cl_default(); dict_key_len]);
            assert_eq!(
                ds.values[entry_index],
                vec![i16::cl_default(); dict_value_len]
            );
        }

        for (i, ei) in insert_indices[(dict_capacity / 2)..].iter().enumerate() {
            let entry_index = *ei as usize;
            let index = i + dict_capacity / 2;
            assert_eq!(ds.keys[entry_index], test_matrix.keys[index]);
            assert_eq!(ds.values[entry_index], test_matrix.values[index]);
        }
    }

    #[test]
    fn keys_do_not_exist() {
        let dict_capacity = 64;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let test_matrix = TestMatrix::new(dict_capacity, dict_key_len, dict_value_len, 1, 10);
        let test_matrix_2 = TestMatrix::new(
            dict_capacity,
            dict_key_len,
            dict_value_len,
            1 + dict_capacity as i16,
            10,
        );

        let insert_indices = dict.insert(&test_matrix.keys, &test_matrix.values).unwrap();

        let indices = dict.remove(&test_matrix_2.keys).unwrap();

        assert_eq!(indices, vec![KEY_NOT_EXIST; dict_capacity]);

        let ds = dict.print().unwrap();

        for (i, di) in insert_indices.into_iter().enumerate() {
            let entry_index = di as usize;
            assert_eq!(ds.keys[entry_index], test_matrix.keys[i]);
            assert_eq!(ds.values[entry_index], test_matrix.values[i]);
        }
    }

    #[test]
    fn all_keys_are_the_same() {
        let dict_capacity = 32;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let test_matrix = TestMatrix::new(dict_capacity, dict_key_len, dict_value_len, 1, 10);
        let _ = dict.insert(&test_matrix.keys, &test_matrix.values).unwrap();

        let key = test_matrix.keys[0].clone();
        let value = test_matrix.values[0].clone();

        let keys = vec![key.clone(); dict_capacity];

        let _indices = dict.remove(&keys).unwrap();

        // TODO assert indices

        let ds = dict.print().unwrap();

        assert!(!ds.has_entry(&key, &value));
        assert_eq!(ds.summary(), DictSnapshotSummary::with(dict_capacity - 1));

        let summary = dict.summary().unwrap();
        assert_eq!(summary.available, 1);
    }
}

#[cfg(test)]
mod tests_dict_remove_with_verification {
    use super::*;
    use crate::config::{ClTypeDefault, DEFAULT_DEVICE_INDEX};
    use crate::dictionary::config::DictSrc;
    use crate::dictionary::handle::KEY_NOT_EXIST;
    use crate::test_utils::TestMatrix;
    use opencl::wrapper::system::System;

    #[test]
    fn dict_is_empty() {
        let dict_capacity = 64;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let test_matrix = TestMatrix::new(dict_capacity, dict_key_len, dict_value_len, 1, 10);
        let indices = dict.remove_with_verification(&test_matrix.keys).unwrap();

        assert_eq!(indices, vec![KEY_NOT_EXIST; test_matrix.keys.len()]);

        let ds = dict.print().unwrap();
        for key in ds.keys.into_iter() {
            assert_eq!(key, vec![i16::cl_default(); dict_key_len]);
        }
        for value in ds.values.into_iter() {
            assert_eq!(value, vec![i16::cl_default(); dict_value_len]);
        }
    }

    #[test]
    fn dict_is_full() {
        let dict_capacity = 64;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let test_matrix = TestMatrix::new(dict_capacity, dict_key_len, dict_value_len, 1, 10);

        let insert_indices = dict.insert(&test_matrix.keys, &test_matrix.values).unwrap();

        let keys = &test_matrix.keys[..(dict_capacity / 2)].to_vec();
        let indices = dict.remove_with_verification(keys).unwrap();

        let mut expected = insert_indices[..(dict_capacity / 2)].to_vec();
        expected.sort();
        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        assert_eq!(indices_sorted, expected);

        let ds = dict.print().unwrap();

        for ei in insert_indices[..(dict_capacity / 2)].iter() {
            let entry_index = *ei as usize;
            assert_eq!(ds.keys[entry_index], vec![i16::cl_default(); dict_key_len]);
            assert_eq!(
                ds.values[entry_index],
                vec![i16::cl_default(); dict_value_len]
            );
        }

        for (i, ei) in insert_indices[(dict_capacity / 2)..].iter().enumerate() {
            let entry_index = *ei as usize;
            let index = i + dict_capacity / 2;
            assert_eq!(ds.keys[entry_index], test_matrix.keys[index]);
            assert_eq!(ds.values[entry_index], test_matrix.values[index]);
        }
    }

    #[test]
    fn keys_do_not_exist() {
        let dict_capacity = 64;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let test_matrix = TestMatrix::new(dict_capacity, dict_key_len, dict_value_len, 1, 10);
        let test_matrix_2 = TestMatrix::new(
            dict_capacity,
            dict_key_len,
            dict_value_len,
            1 + dict_capacity as i16,
            10,
        );

        let insert_indices = dict.insert(&test_matrix.keys, &test_matrix.values).unwrap();

        let indices = dict.remove_with_verification(&test_matrix_2.keys).unwrap();

        assert_eq!(indices, vec![KEY_NOT_EXIST; dict_capacity]);

        let ds = dict.print().unwrap();

        for (i, di) in insert_indices.into_iter().enumerate() {
            let entry_index = di as usize;
            assert_eq!(ds.keys[entry_index], test_matrix.keys[i]);
            assert_eq!(ds.values[entry_index], test_matrix.values[i]);
        }
    }

    #[test]
    fn all_keys_are_the_same() {
        let dict_capacity = 32;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let test_matrix = TestMatrix::new(dict_capacity, dict_key_len, dict_value_len, 1, 10);
        let _ = dict.insert(&test_matrix.keys, &test_matrix.values).unwrap();

        let key = test_matrix.keys[0].clone();
        let value = test_matrix.values[0].clone();

        let keys = vec![key.clone(); dict_capacity];

        let _indices = dict.remove_with_verification(&keys).unwrap();

        // TODO assert indices

        let ds = dict.print().unwrap();

        assert!(!ds.has_entry(&key, &value));
        assert_eq!(ds.summary(), DictSnapshotSummary::with(dict_capacity - 1));

        let summary = dict.summary().unwrap();
        assert_eq!(summary.available, 1);
    }
}

#[cfg(test)]
mod tests_dict_keys {
    use super::*;
    use crate::config::{ClTypeDefault, DEFAULT_DEVICE_INDEX};
    use crate::dictionary::config::DictSrc;
    use crate::test_utils::TestMatrix;
    use opencl::wrapper::system::System;

    #[test]
    fn dict_is_empty() {
        let dict_capacity = 64;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let keys = dict.keys().unwrap();

        assert_eq!(
            keys,
            vec![vec![i16::cl_default(); dict_key_len]; dict_capacity]
        );
    }

    #[test]
    fn dict_is_full() {
        let dict_capacity = 64;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let test_matrix = TestMatrix::new(dict_capacity, dict_key_len, dict_value_len, 1, 10);

        let indices = dict.insert(&test_matrix.keys, &test_matrix.values).unwrap();
        let keys = dict.keys().unwrap();

        #[allow(clippy::needless_range_loop)]
        for i in 0..dict_capacity {
            let ri = indices[i] as usize;
            assert_eq!(keys[ri], test_matrix.keys[i]);
        }
    }

    #[test]
    fn partially_full_dict() {
        let dict_capacity = 64;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let test_matrix = TestMatrix::new(dict_capacity / 2, dict_key_len, dict_value_len, 1, 10);

        let indices = dict.insert(&test_matrix.keys, &test_matrix.values).unwrap();
        let keys = dict.keys().unwrap();

        #[allow(clippy::needless_range_loop)]
        for i in 0..(dict_capacity / 2) {
            let ri = indices[i] as usize;
            assert_eq!(keys[ri], test_matrix.keys[i]);
        }

        #[allow(clippy::needless_range_loop)]
        for i in 0..dict_capacity {
            let i_i32 = i as cl_int;
            if !indices.contains(&i_i32) {
                assert_eq!(keys[i], vec![i16::cl_default(); dict_key_len]);
            }
        }
    }
}

#[cfg(test)]
mod tests_dict_get {
    use super::*;
    use crate::config::{ClTypeDefault, DEFAULT_DEVICE_INDEX};
    use crate::dictionary::config::DictSrc;
    use crate::dictionary::handle::KEY_NOT_EXIST;
    use crate::test_utils::TestMatrix;
    use opencl::wrapper::system::System;

    #[test]
    fn dict_is_empty() {
        let dict_capacity = 64;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let test_matrix = TestMatrix::new(dict_capacity, dict_key_len, dict_value_len, 1, 10);

        let (values, indices) = dict.get(&test_matrix.keys).unwrap();

        assert_eq!(
            values,
            vec![vec![i16::cl_default(); dict_key_len]; dict_capacity]
        );
        assert_eq!(indices, vec![KEY_NOT_EXIST; dict_capacity]);
    }

    #[test]
    fn dict_is_full() {
        let dict_capacity = 64;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let test_matrix = TestMatrix::new(dict_capacity, dict_key_len, dict_value_len, 1, 10);

        let inserted_indices = dict.insert(&test_matrix.keys, &test_matrix.values).unwrap();

        let (values, indices) = dict.get(&test_matrix.keys).unwrap();

        for (i, v) in values.iter().enumerate() {
            assert_eq!(v, &test_matrix.values[i]);
        }

        assert_eq!(indices, inserted_indices);
    }

    #[test]
    fn partially_full_dict() {
        let dict_capacity = 64;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let test_matrix = TestMatrix::new(dict_capacity, dict_key_len, dict_value_len, 1, 10);

        let _inserted_indices = dict
            .insert(
                &test_matrix.keys[0..(dict_capacity / 2)].to_vec(),
                &test_matrix.values[0..(dict_capacity / 2)].to_vec(),
            )
            .unwrap();

        let (values, _indices) = dict.get(&test_matrix.keys).unwrap();

        for (i, v) in values.into_iter().enumerate() {
            if i >= (dict_capacity / 2) {
                assert_eq!(v, vec![i16::cl_default(); dict_value_len]);
                continue;
            }
            assert_eq!(v, test_matrix.values[i]);
        }

        // TODO assert indices
    }
}

#[cfg(test)]
mod tests_dict_get_size {
    use super::*;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::dictionary::config::DictSrc;
    use crate::test_utils::TestMatrix;
    use opencl::opencl_sys::bindings::cl_short;
    use opencl::wrapper::system::System;

    #[test]
    fn dict_is_empty() {
        let dict_capacity = 64;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let test_matrix = TestMatrix::new(dict_capacity, dict_key_len, dict_value_len, 1, 10);

        let (sizes, _indices) = dict.get_size(&test_matrix.keys).unwrap();

        assert_eq!(sizes, vec![0; dict_capacity]);
    }

    #[test]
    fn dict_is_full() {
        let dict_capacity = 64;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let mut keys = vec![];

        let test_matrix_1 = TestMatrix::new(dict_capacity / 2, dict_key_len, dict_value_len, 1, 10);
        let _ = dict
            .insert(&test_matrix_1.keys, &test_matrix_1.values)
            .unwrap();
        keys.append(&mut test_matrix_1.keys.clone());

        let test_matrix_2 = TestMatrix::new(
            dict_capacity / 2,
            dict_key_len,
            dict_value_len / 2,
            1 + dict_capacity as cl_short,
            10 + dict_capacity as cl_short,
        );
        let _ = dict
            .insert(&test_matrix_2.keys, &test_matrix_2.values)
            .unwrap();
        keys.append(&mut test_matrix_2.keys.clone());

        let (sizes, _indices) = dict.get_size(&keys).unwrap();

        assert_eq!(
            sizes[..(dict_capacity / 2)],
            vec![dict_capacity as cl_uint; dict_capacity / 2]
        );
        assert_eq!(
            sizes[(dict_capacity / 2)..],
            vec![(dict_value_len / 2) as cl_uint; dict_capacity / 2]
        );
    }

    // FIXME test name
    #[test]
    fn simple_cases() {
        let dict_capacity = 64;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let mut keys = vec![];

        let test_matrix_1 = TestMatrix::new(dict_capacity / 2, dict_key_len, 1, 1, 10);
        let _ = dict
            .insert(&test_matrix_1.keys, &test_matrix_1.values)
            .unwrap();
        keys.append(&mut test_matrix_1.keys.clone());

        let test_matrix_2 = TestMatrix::new(
            dict_capacity / 2,
            dict_key_len,
            dict_value_len - 2,
            1 + dict_capacity as cl_short,
            10 + dict_capacity as cl_short,
        );
        let _ = dict
            .insert(&test_matrix_2.keys, &test_matrix_2.values)
            .unwrap();
        keys.append(&mut test_matrix_2.keys.clone());

        let (sizes, _indices) = dict.get_size(&keys).unwrap();

        assert_eq!(sizes[..(dict_capacity / 2)], vec![1; dict_capacity / 2]);
        assert_eq!(
            sizes[(dict_capacity / 2)..],
            vec![(dict_value_len - 2) as cl_uint; dict_capacity / 2]
        );
    }
}

#[cfg(test)]
mod tests_dict_summary {
    use super::*;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::dictionary::config::DictSrc;
    use crate::test_utils::TestMatrix;
    use opencl::opencl_sys::bindings::cl_short;
    use opencl::wrapper::system::System;

    #[test]
    fn dict_is_empty() {
        let dict_capacity = 64;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let summary = dict.summary().unwrap();

        assert_eq!(summary, DictSummary::create_empty(config));
    }

    #[test]
    fn dict_is_full() {
        let dict_capacity = 64;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let test_matrix_1 = TestMatrix::new(dict_capacity / 2, dict_key_len, dict_value_len, 1, 10);
        let _ = dict
            .insert(&test_matrix_1.keys, &test_matrix_1.values)
            .unwrap();

        let test_matrix_2 = TestMatrix::new(
            dict_capacity / 2,
            dict_key_len,
            dict_value_len / 2,
            1 + dict_capacity as cl_short,
            10 + dict_capacity as cl_short,
        );
        let _ = dict
            .insert(&test_matrix_2.keys, &test_matrix_2.values)
            .unwrap();

        let summary = dict.summary().unwrap();

        let expected_keys_sizes = &vec![dict_key_len as cl_uint; dict_capacity];

        let mut expected_values_sizes = vec![dict_value_len as cl_uint; dict_capacity / 2];
        expected_values_sizes.append(&mut vec![
            (dict_value_len / 2) as cl_uint;
            dict_capacity / 2
        ]);

        assert_eq!(
            summary,
            DictSummary::new(config, expected_keys_sizes, &expected_values_sizes, 0)
        );
    }

    // FIXME test name
    #[test]
    fn simple_cases() {
        let dict_capacity = 64;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let test_matrix_1 = TestMatrix::new(dict_capacity / 2, 1, 1, 1, 10);
        let _ = dict
            .insert(&test_matrix_1.keys, &test_matrix_1.values)
            .unwrap();

        let test_matrix_2 = TestMatrix::new(
            dict_capacity / 2,
            dict_key_len - 2,
            dict_value_len - 2,
            1 + dict_capacity as cl_short,
            10 + dict_capacity as cl_short,
        );
        let _ = dict
            .insert(&test_matrix_2.keys, &test_matrix_2.values)
            .unwrap();

        let summary = dict.summary().unwrap();

        let mut expected_keys_sizes = vec![1; dict_capacity / 2];
        expected_keys_sizes.append(&mut vec![(dict_capacity - 2) as cl_uint; dict_capacity / 2]);

        let mut expected_values_sizes = vec![1; dict_capacity / 2];
        expected_values_sizes.append(&mut vec![(dict_capacity - 2) as cl_uint; dict_capacity / 2]);

        assert_eq!(
            summary,
            DictSummary::new(config, &expected_keys_sizes, &expected_values_sizes, 0)
        );
    }

    #[test]
    fn partially_filled_dict() {
        let dict_capacity = 64;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let test_matrix_1 = TestMatrix::new(dict_capacity / 2, dict_key_len, dict_value_len, 1, 10);
        let _ = dict
            .insert(&test_matrix_1.keys, &test_matrix_1.values)
            .unwrap();

        let summary = dict.summary().unwrap();

        let mut expected_keys_sizes = vec![dict_key_len as cl_uint; dict_capacity / 2];
        expected_keys_sizes.append(&mut vec![0; dict_capacity / 2]);

        let mut expected_values_sizes = vec![dict_key_len as cl_uint; dict_capacity / 2];
        expected_values_sizes.append(&mut vec![0; dict_capacity / 2]);

        assert_eq!(
            summary,
            DictSummary::new(config, &expected_keys_sizes, &expected_values_sizes, 32)
        );
    }

    // FIXME test name
    #[test]
    fn dict_with_one_available_key() {
        let dict_capacity = 64;
        let dict_key_len = 64;
        let dict_value_len = 64;

        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(dict_key_len, dict_value_len, dict_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &dict_src.build()).unwrap();

        let config = dict_src.get_config_by_id(0).unwrap();
        let dict = DictHandle::new(config, system);
        dict.initialize().unwrap();

        let test_matrix_1 = TestMatrix::new(dict_capacity - 1, dict_key_len, dict_value_len, 1, 10);
        let _ = dict
            .insert(&test_matrix_1.keys, &test_matrix_1.values)
            .unwrap();

        let summary = dict.summary().unwrap();

        let mut expected_keys_sizes = vec![dict_capacity as cl_uint; dict_capacity - 1];
        expected_keys_sizes.push(0);

        let mut expected_values_sizes = vec![dict_capacity as cl_uint; dict_capacity - 1];
        expected_values_sizes.push(0);

        assert_eq!(
            summary,
            DictSummary::new(config, &expected_keys_sizes, &expected_values_sizes, 1)
        );
    }
}
