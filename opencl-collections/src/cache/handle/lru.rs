use crate::cache::config::CacheConfig;
use crate::cache::handle::{CacheIndices, LruSummary};
use crate::cache::kernel::name::{
    get_cache_kernel_name, LRU_CACHE_ARRAY_SET_RESET, LRU_CACHE_DEBUG, LRU_CACHE_DEBUG_SORT,
    LRU_CACHE_GET_KEYS, LRU_CACHE_GET_SORTED_KEYS, LRU_CACHE_PUT, LRU_CACHE_RESET, LRU_CACHE_SORT,
    READ_ON_LRU_CACHE, WRITE_IN_LRU_CACHE,
};
use crate::config::{ClTypeDefault, DEBUG_MODE};
use crate::error::OpenClResult;
use crate::set::handle::array_set_v2::ArraySetSnapshot;
use crate::utils::ensure_vec_size;
use opencl::opencl_sys::bindings::{cl_int, cl_uint};
use opencl::wrapper::system::OpenclCommonOperation;
use std::sync::Arc;

#[derive(Debug, PartialEq)]
pub struct LRUCacheSnapshot {
    pub last_priority: cl_int,
    pub top: cl_int,
    pub keys: Vec<Vec<cl_int>>,
    pub values: Vec<Vec<cl_int>>,
    pub priorities: Vec<cl_int>,
    pub array_set: ArraySetSnapshot,
}

impl LRUCacheSnapshot {
    pub fn new(
        last_priority: cl_int,
        top: cl_int,
        keys: Vec<Vec<cl_int>>,
        values: Vec<Vec<cl_int>>,
        priorities: Vec<cl_int>,
        array_set: ArraySetSnapshot,
    ) -> Self {
        Self {
            last_priority,
            top,
            keys,
            values,
            priorities,
            array_set,
        }
    }

    pub fn create_empty(key_len: usize, value_len: usize, capacity: usize) -> Self {
        Self::new(
            1,
            0,
            vec![vec![i32::cl_default(); key_len]; capacity],
            vec![vec![i32::cl_default(); value_len]; capacity],
            vec![0; capacity],
            ArraySetSnapshot::create_empty(capacity),
        )
    }

    pub fn get_lower_priority_keys(&mut self, take: usize) -> Vec<Vec<cl_int>> {
        let mut keys: Vec<Vec<cl_int>> = vec![];

        for _ in 0..take {
            let key_index: Option<usize> = self
                .priorities
                .iter()
                .enumerate()
                .filter(|(_i, &x)| x != i32::cl_default())
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index);

            if let Some(i) = key_index {
                keys.push(self.keys[i].clone());
                self.keys.remove(i);
                self.values.remove(i);
                self.priorities.remove(i);
            }
        }

        keys
    }

    pub fn summary(&self) -> LruSummary {
        LruSummary::new(
            self.keys
                .iter()
                .filter(|&x| x.iter().any(|&x| x != i32::cl_default()))
                .count(),
            self.values
                .iter()
                .filter(|&x| x.iter().any(|&x| x != i32::cl_default()))
                .count(),
            self.priorities
                .iter()
                .filter(|&&x| x != i32::cl_default())
                .count(),
        )
    }

    pub fn sort(&mut self) {
        // self.keys.sort();
        // self.values.sort();
        self.priorities.sort();
    }

    pub fn has_entry(&self, key: &Vec<cl_int>, value: &Vec<cl_int>) -> bool {
        match self.keys.iter().position(|x| x == key) {
            None => false,
            Some(index) => &self.values[index] == value,
        }
    }

    // debug
    pub fn print_all_keys(&self) {
        for (i, key) in self.keys.iter().enumerate() {
            println!(
                "index: {} priority: {} - key: {:?}",
                i, self.priorities[i], key
            );
        }
    }

    pub fn print_all_values(&self) {
        for (i, value) in self.values.iter().enumerate() {
            println!(
                "index: {} priority: {} - value: {:?}",
                i, self.priorities[i], value
            );
        }
    }

    pub fn print_all_entries(&self) {
        for (i, key) in self.keys.iter().enumerate() {
            println!("i: {} - key:     {:?}", i, key);
            println!("     - value:   {:?}", self.values[i]);
            println!("     - priority: {:?}", self.priorities[i]);
        }
    }
}

#[derive(Debug)]
pub struct LRUCacheHandle<T: OpenclCommonOperation> {
    config: CacheConfig,
    system: Arc<T>,
}

#[derive(Debug)]
pub struct KeyPriority {
    pub priority: cl_int,
    pub key: Vec<cl_int>,
}

#[derive(Debug)]
pub struct SortEntry {
    pub priority: cl_int,
    // pub from_index: cl_int,
    pub to_index: cl_int,
}

pub type CacheKeys = Vec<Vec<cl_int>>;
pub type CacheValues = Vec<Vec<cl_int>>;

impl<T: OpenclCommonOperation> LRUCacheHandle<T> {
    pub fn new(config: &CacheConfig, system: Arc<T>) -> Self {
        Self {
            config: config.clone(),
            system,
        }
    }

    pub fn get_id(&self) -> usize {
        self.config.id
    }

    pub fn debug(&self) -> OpenClResult<LRUCacheSnapshot> {
        let global_work_size = self.config.capacity;
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let keys_output_capacity = self.config.key_len * global_work_size;
        let values_output_capacity = self.config.value_len * global_work_size;

        let priorities_output_capacity = global_work_size;
        let meta_output_capacity = 2;

        let set_items_output_capacity = global_work_size * 2;

        let keys_output_buf = self.system.create_output_buffer(keys_output_capacity)?;
        let values_output_buf = self.system.create_output_buffer(values_output_capacity)?;

        let priorities_output_buf = self
            .system
            .create_output_buffer(priorities_output_capacity)?;
        let meta_output_buf = self.system.create_output_buffer(meta_output_capacity)?;

        let set_items_output_buf = self
            .system
            .create_output_buffer(set_items_output_capacity)?;

        let kernel_name = get_cache_kernel_name(LRU_CACHE_DEBUG, self.get_id());
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&keys_output_buf.get_cl_mem())?;
            kernel.set_arg(&values_output_buf.get_cl_mem())?;
            kernel.set_arg(&priorities_output_buf.get_cl_mem())?;
            kernel.set_arg(&meta_output_buf.get_cl_mem())?;
            kernel.set_arg(&set_items_output_buf.get_cl_mem())?;

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

        let priorities_output = self.system.blocking_enqueue_read_buffer(
            priorities_output_capacity,
            &priorities_output_buf,
            &[],
        )?;

        let meta_output = self.system.blocking_enqueue_read_buffer(
            meta_output_capacity,
            &meta_output_buf,
            &[],
        )?;

        let set_items_output = self.system.blocking_enqueue_read_buffer(
            set_items_output_capacity,
            &set_items_output_buf,
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

        let items = set_items_output[0..self.config.capacity].to_vec();
        let entries = set_items_output[self.config.capacity..].to_vec();

        let array_set = ArraySetSnapshot::new(items, entries);

        Ok(LRUCacheSnapshot {
            last_priority: meta_output[0],
            top: meta_output[1],
            keys,
            values,
            priorities: priorities_output,
            array_set,
        })
    }

    pub fn print(&self) -> OpenClResult<LRUCacheSnapshot> {
        let cs = self.debug()?;
        // println!("{q_s:?}");
        println!(
            "
LRUCacheSnapshot (
   last_priority: {},
   top:           {}
   priorities: {:?}
   ArraySetSnapshot: (
     items:   {:?},
     entries: {:?}
   )
)
        ",
            cs.last_priority, cs.top, cs.priorities, cs.array_set.items, cs.array_set.entries
        );

        // println!("keys");
        // cs.print_all_keys();
        //
        // println!("values");
        // cs.print_all_values();

        cs.print_all_entries();

        Ok(cs)
    }

    pub fn reset(&self) -> OpenClResult<()> {
        let global_work_size = self.config.capacity;
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let kernel_name = get_cache_kernel_name(LRU_CACHE_RESET, self.get_id());
        let kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        };

        Ok(())
    }

    pub fn initialize(&self) -> OpenClResult<()> {
        self.reset()
    }

    pub fn insert(&self, keys: &CacheKeys, values: &CacheValues) -> OpenClResult<Vec<cl_int>> {
        if keys.len() != values.len() {
            panic!("error handle keys & values len")
        }

        let global_work_size = keys.len();
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let keys_input_capacity = self.config.key_len * global_work_size;
        let values_input_capacity = self.config.value_len * global_work_size;

        let indices_output_capacity = global_work_size;

        let mut keys_input: Vec<_> = Vec::with_capacity(keys_input_capacity);

        for key in keys {
            let mut k = ensure_vec_size(key, self.config.key_len);
            keys_input.append(&mut k);
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
        let buf_len = self.config.value_len as cl_uint;

        let kernel_name = get_cache_kernel_name(WRITE_IN_LRU_CACHE, self.get_id());
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&key_len)?;
            kernel.set_arg(&buf_len)?;
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
            println!("indices_output {indices_output:?}");
        }

        Ok(indices_output)
    }

    pub fn reset_array_set(&self) -> OpenClResult<()> {
        let global_work_size = self.config.capacity;
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let kernel_name = get_cache_kernel_name(LRU_CACHE_ARRAY_SET_RESET, self.get_id());
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

    pub fn add(&self, keys: &CacheKeys, values: &CacheValues) -> OpenClResult<Vec<cl_int>> {
        let r = self.insert(keys, values)?;
        self.reset_array_set()?;
        Ok(r)
    }

    pub fn put(
        &self,
        keys: &CacheKeys,
        values: &CacheValues,
        priorities: &[cl_int],
    ) -> OpenClResult<()> {
        if keys.len() != values.len() {
            panic!("error handle keys & values len")
        }

        if keys.len() != priorities.len() {
            panic!("error handle keys & priorities len")
        }

        let global_work_size = keys.len();
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let keys_input_capacity = self.config.key_len * global_work_size;
        let values_input_capacity = self.config.value_len * global_work_size;

        let mut keys_input: Vec<_> = Vec::with_capacity(keys_input_capacity);

        for key in keys {
            let mut k = ensure_vec_size(key, self.config.key_len);
            keys_input.append(&mut k);
        }

        let mut values_input: Vec<_> = Vec::with_capacity(values_input_capacity);

        for b in values {
            let mut v = ensure_vec_size(b, self.config.value_len);
            values_input.append(&mut v);
        }

        let keys_input_buf = self.system.blocking_prepare_input_buffer(&keys_input)?;
        let values_input_buf = self.system.blocking_prepare_input_buffer(&values_input)?;

        let priorities_input_buf = self.system.blocking_prepare_input_buffer(priorities)?;

        let kernel_name = get_cache_kernel_name(LRU_CACHE_PUT, self.get_id());
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&keys_input_buf.get_cl_mem())?;
            kernel.set_arg(&values_input_buf.get_cl_mem())?;
            kernel.set_arg(&priorities_input_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        Ok(())
    }

    pub fn get(&self, keys: &CacheKeys) -> OpenClResult<(CacheValues, CacheIndices)> {
        let global_work_size = keys.len();
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
        let buf_len = self.config.value_len as cl_uint;

        let kernel_name = get_cache_kernel_name(READ_ON_LRU_CACHE, self.get_id());
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&key_len)?;
            kernel.set_arg(&buf_len)?;
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

        let indices_output = self.system.blocking_enqueue_read_buffer(
            indices_output_capacity,
            &indices_output_buf,
            &[],
        )?;

        let values: Vec<Vec<_>> = values_output
            .chunks(self.config.value_len)
            .map(|x| x.to_vec())
            .collect();

        if DEBUG_MODE {
            println!("indices_output {indices_output:?}");
            for (i, k) in values.iter().enumerate() {
                println!("{i} - {k:?}");
            }
        }

        Ok((values, indices_output))
    }

    pub fn keys(&self) -> OpenClResult<Vec<KeyPriority>> {
        let global_work_size = self.config.capacity;
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let keys_output_capacity = self.config.key_len * global_work_size;
        let priorities_output_capacity = global_work_size;

        let keys_output_buf = self.system.create_output_buffer(keys_output_capacity)?;
        let priorities_output_buf = self
            .system
            .create_output_buffer(priorities_output_capacity)?;

        let kernel_name = get_cache_kernel_name(LRU_CACHE_GET_KEYS, self.get_id());
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&keys_output_buf.get_cl_mem())?;
            kernel.set_arg(&priorities_output_buf.get_cl_mem())?;

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

        let priorities_output = self.system.blocking_enqueue_read_buffer(
            priorities_output_capacity,
            &priorities_output_buf,
            &[],
        )?;

        let keys: Vec<KeyPriority> = keys_output
            .chunks(self.config.key_len)
            .enumerate()
            .filter_map(|(i, x)| -> Option<KeyPriority> {
                let priority = priorities_output[i];

                if priority == 0 {
                    return None;
                }

                Some(KeyPriority {
                    key: x.to_vec(),
                    priority: priorities_output[i],
                })
            })
            .collect();

        Ok(keys)
    }

    /// cpu sort
    pub fn sorted_keys(&self) -> OpenClResult<Vec<KeyPriority>> {
        let mut keys = self.keys()?;
        keys.sort_by(|a, b| b.priority.cmp(&a.priority));
        Ok(keys)
    }

    pub fn get_sorted_keys(&self) -> OpenClResult<Vec<KeyPriority>> {
        let global_work_size = 1;
        let local_work_size = 1;

        let output_capacity = self.config.capacity;

        // 4 - CMQ_SORT
        //    0 - CMQ_PREPARE
        //       1 - CMQ_COPY_VALUES
        //       2 - CMQ_INIT_SORT
        //    3 - CMQ_CONFIRM_SORT
        // 5 - CMQ_GET_KEYS
        let enqueue_kernel_output_capacity = 6;

        let keys_output_capacity = self.config.key_len * output_capacity;
        let priorities_output_capacity = output_capacity;

        let keys_output_buf = self.system.create_output_buffer(keys_output_capacity)?;
        let priorities_output_buf = self
            .system
            .create_output_buffer(priorities_output_capacity)?;

        let enqueue_kernel_output_buf = self
            .system
            .create_output_buffer(enqueue_kernel_output_capacity)?;

        let q0 = self
            .system
            .get_device_command_queue_0()
            .get_cl_command_queue();

        let kernel_name = get_cache_kernel_name(LRU_CACHE_GET_SORTED_KEYS, self.get_id());
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        let capacity_device_local_work_size =
            self.system
                .first_device_check_local_work_size(self.config.capacity) as cl_uint;

        unsafe {
            kernel.set_arg(&q0)?;
            kernel.set_arg(&capacity_device_local_work_size)?;
            kernel.set_arg(&keys_output_buf.get_cl_mem())?;
            kernel.set_arg(&priorities_output_buf.get_cl_mem())?;
            kernel.set_arg(&enqueue_kernel_output_buf.get_cl_mem())?;

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

        let priorities_output = self.system.blocking_enqueue_read_buffer(
            priorities_output_capacity,
            &priorities_output_buf,
            &[],
        )?;

        let keys: Vec<KeyPriority> = keys_output
            .chunks(self.config.key_len)
            .enumerate()
            .filter_map(|(i, x)| -> Option<KeyPriority> {
                let priority = priorities_output[i];

                if priority == 0 {
                    return None;
                }

                Some(KeyPriority {
                    key: x.to_vec(),
                    priority: priorities_output[i],
                })
            })
            .collect();

        self.system.assert_device_enqueue_kernel(
            enqueue_kernel_output_capacity,
            enqueue_kernel_output_buf,
            &[],
        )?;

        Ok(keys)
    }

    pub fn sort(&self) -> OpenClResult<()> {
        let global_work_size = 1;
        let local_work_size = 1;

        // 0 - CMQ_PREPARE
        //    1 - CMQ_COPY_VALUES
        //    2 - CMQ_INIT_SORT
        // 3 - CMQ_CONFIRM_SORT
        let enqueue_kernel_output_capacity = 4;

        let enqueue_kernel_output_buf = self
            .system
            .create_output_buffer(enqueue_kernel_output_capacity)?;

        let kernel_name = get_cache_kernel_name(LRU_CACHE_SORT, self.get_id());
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        let q0 = self
            .system
            .get_device_command_queue_0()
            .get_cl_command_queue();

        let capacity_device_local_work_size =
            self.system
                .first_device_check_local_work_size(self.config.capacity) as cl_uint;

        unsafe {
            kernel.set_arg(&q0)?;
            kernel.set_arg(&capacity_device_local_work_size)?;
            kernel.set_arg(&enqueue_kernel_output_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        self.system.assert_device_enqueue_kernel(
            enqueue_kernel_output_capacity,
            enqueue_kernel_output_buf,
            &[],
        )?;

        Ok(())
    }

    pub fn debug_sort(&self) -> OpenClResult<Vec<SortEntry>> {
        let global_work_size = self.config.capacity;
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        // index 0 priority
        // index 1 to_index
        let entry_len = 2;
        let output_capacity = global_work_size * entry_len;

        let output_buf = self.system.create_output_buffer(output_capacity)?;

        let kernel_name = get_cache_kernel_name(LRU_CACHE_DEBUG_SORT, self.get_id());
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

        let entries: Vec<SortEntry> = output
            .chunks(entry_len)
            .map(|x| SortEntry {
                priority: x[0],
                to_index: x[1],
            })
            .collect();

        if DEBUG_MODE {
            println!("output {output:?}");
            for (i, e) in entries.iter().enumerate() {
                println!("i:{} - {:?}", i, e);
            }
        }

        Ok(entries)
    }
}

#[cfg(test)]
mod tests_lru_cache_reset {
    use super::*;
    use crate::cache::config::CacheSrc;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use opencl::wrapper::system::System;

    #[test]
    fn cache_is_empty() {
        let mut cache_src = CacheSrc::new();
        cache_src.add_lru(16, 32, 8);
        cache_src.add_lru(64, 64, 16);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_lru_by_id(0).unwrap();
        let cache = LRUCacheHandle::new(config, system);

        let keys = vec![vec![0; 16]; 8];
        let values = vec![vec![0; 32]; 8];
        let priorities = vec![0; 8];

        assert_eq!(
            cache.print().unwrap(),
            LRUCacheSnapshot::new(
                1,
                0,
                keys,
                values,
                priorities.clone(),
                ArraySetSnapshot::new(priorities.clone(), priorities.clone())
            )
        );

        let r = cache.reset();
        assert!(r.is_ok());

        assert_eq!(
            cache.print().unwrap(),
            LRUCacheSnapshot::create_empty(16, 32, 8)
        );
    }
}

#[cfg(test)]
mod tests_lru_cache_debug {
    use super::*;
    use crate::cache::config::CacheSrc;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::test_utils::TestMatrix;
    use opencl::wrapper::system::System;

    #[test]
    fn cache_is_empty() {
        let cache_capacity = 32;
        let cache_key_len = 256;
        let cache_value_len = 256;

        let mut cache_src = CacheSrc::new();
        cache_src.add_lru(cache_key_len, cache_value_len, cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_lru_by_id(0).unwrap();
        let cache = LRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let cs = cache.print().unwrap();

        assert_eq!(
            cs,
            LRUCacheSnapshot::create_empty(cache_key_len, cache_value_len, cache_capacity)
        );
    }

    #[test]
    fn cache_is_full() {
        let cache_capacity = 32;
        let cache_key_len = 256;
        let cache_value_len = 256;

        let mut cache_src = CacheSrc::new();
        cache_src.add_lru(cache_key_len, cache_value_len, cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_lru_by_id(0).unwrap();
        let cache = LRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let test_matrix = TestMatrix::new(cache_capacity, cache_key_len, cache_value_len, 1, 10);

        let indices = cache
            .insert(&test_matrix.keys, &test_matrix.values)
            .unwrap();

        let mut cs = cache.print().unwrap();

        #[allow(clippy::needless_range_loop)]
        for i in 0..cache_capacity {
            let c_i = indices[i] as usize;
            assert_eq!(
                cs.keys[c_i],
                ensure_vec_size(&test_matrix.keys[i], cache_key_len)
            );
            assert_eq!(
                cs.values[c_i],
                ensure_vec_size(&test_matrix.values[i], cache_value_len)
            );
        }

        cs.sort();

        assert_eq!(cs.last_priority, 1 + cache_capacity as cl_int);

        let expected_priorities: Vec<cl_int> = (1..=cache_capacity as cl_int).collect();
        assert_eq!(cs.priorities, expected_priorities);
    }

    #[test]
    fn partially_full_cache() {
        let cache_capacity = 32;
        let cache_key_len = 256;
        let cache_value_len = 256;

        let mut cache_src = CacheSrc::new();
        cache_src.add_lru(cache_key_len, cache_value_len, cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_lru_by_id(0).unwrap();
        let cache = LRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let test_matrix = TestMatrix::new(
            cache_capacity / 2,
            cache_key_len / 2,
            cache_value_len / 2,
            1,
            10,
        );

        let indices = cache
            .insert(&test_matrix.keys, &test_matrix.values)
            .unwrap();

        let mut cs = cache.print().unwrap();

        #[allow(clippy::needless_range_loop)]
        for i in 0..cache_capacity {
            if i >= (cache_capacity / 2) {
                assert_eq!(cs.keys[i], vec![i32::cl_default(); cache_key_len]);
                assert_eq!(cs.values[i], vec![i32::cl_default(); cache_value_len]);
                continue;
            }

            let c_i = indices[i] as usize;
            assert_eq!(
                cs.keys[c_i],
                ensure_vec_size(&test_matrix.keys[i], cache_key_len)
            );
            assert_eq!(
                cs.values[c_i],
                ensure_vec_size(&test_matrix.values[i], cache_value_len)
            );
        }

        assert_eq!(cs.last_priority, 1 + (cache_capacity / 2) as cl_int);

        cs.sort();

        let mut expected_priorities: Vec<cl_int> = vec![0; cache_capacity / 2];
        expected_priorities.append(&mut (1..=(cache_capacity / 2) as cl_int).collect());

        assert_eq!(cs.priorities, expected_priorities);
    }
}

#[cfg(test)]
mod tests_lru_cache_insert {
    use super::*;
    use crate::cache::config::CacheSrc;
    use crate::cache::handle::LruSummary;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::test_utils::TestMatrix;
    use crate::utils::{ensure_vec_size, has_unique_elements};
    use opencl::wrapper::system::System;

    #[test]
    fn cache_is_empty() {
        let cache_capacity = 32;
        let cache_key_len = 256;
        let cache_value_len = 256;

        let mut cache_src = CacheSrc::new();
        cache_src.add_lru(cache_key_len, cache_value_len, cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_lru_by_id(0).unwrap();
        let cache = LRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let test_matrix = TestMatrix::new(cache_capacity, cache_key_len, cache_value_len, 1, 10);

        let indices = cache
            .insert(&test_matrix.keys, &test_matrix.values)
            .unwrap();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        let expected_indices: Vec<cl_int> = (0..cache_capacity as cl_int).collect();
        assert_eq!(indices_sorted, expected_indices);

        let mut cs = cache.print().unwrap();
        cs.sort();

        for (i, c_i) in indices.into_iter().enumerate() {
            let ci = c_i as usize;
            assert_eq!(
                cs.keys[ci],
                ensure_vec_size(&test_matrix.keys[i], cache_key_len)
            );
            assert_eq!(
                cs.values[ci],
                ensure_vec_size(&test_matrix.values[i], cache_value_len)
            );
        }

        let expected_priorities: Vec<_> = expected_indices.iter().map(|x| x + 1).collect();
        assert_eq!(cs.priorities, expected_priorities);
    }

    #[test]
    fn cache_is_empty_2() {
        let cache_capacity = 512;
        let cache_key_len = 32;
        let cache_value_len = 32;

        let mut cache_src = CacheSrc::new();
        cache_src.add_lru(cache_key_len, cache_value_len, cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_lru_by_id(0).unwrap();
        let cache = LRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let test_matrix = TestMatrix::new(cache_capacity, cache_key_len, cache_value_len, 1, 10);

        let indices = cache
            .insert(&test_matrix.keys, &test_matrix.values)
            .unwrap();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        let expected_indices: Vec<cl_int> = (0..cache_capacity as cl_int).collect();
        assert_eq!(indices_sorted, expected_indices);

        let mut cs = cache.print().unwrap();
        cs.sort();

        for (i, c_i) in indices.into_iter().enumerate() {
            let ci = c_i as usize;
            assert_eq!(
                cs.keys[ci],
                ensure_vec_size(&test_matrix.keys[i], cache_key_len)
            );
            assert_eq!(
                cs.values[ci],
                ensure_vec_size(&test_matrix.values[i], cache_value_len)
            );
        }

        let expected_priorities: Vec<_> = expected_indices.iter().map(|x| x + 1).collect();
        assert_eq!(cs.priorities, expected_priorities);
    }

    #[test]
    fn cache_is_full() {
        let cache_capacity = 32;
        let cache_key_len = 256;
        let cache_value_len = 256;

        let mut cache_src = CacheSrc::new();
        cache_src.add_lru(cache_key_len, cache_value_len, cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_lru_by_id(0).unwrap();
        let cache = LRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let test_matrix = TestMatrix::new(cache_capacity, cache_key_len, cache_value_len, 1, 10);

        let _ = cache
            .insert(&test_matrix.keys, &test_matrix.values)
            .unwrap();

        let test_matrix_2 = TestMatrix::new(
            cache_capacity,
            cache_key_len,
            cache_value_len,
            10 + cache_capacity as cl_int,
            100,
        );

        let indices = cache
            .insert(&test_matrix_2.keys, &test_matrix_2.values)
            .unwrap();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        let expected: Vec<cl_int> = (0..cache_capacity as cl_int).collect();
        assert_eq!(indices_sorted, expected);

        let mut cs = cache.print().unwrap();

        for (i, c_i) in indices.into_iter().enumerate() {
            let ci = c_i as usize;
            assert_eq!(
                cs.keys[ci],
                ensure_vec_size(&test_matrix_2.keys[i], cache_key_len)
            );
            assert_eq!(
                cs.values[ci],
                ensure_vec_size(&test_matrix_2.values[i], cache_value_len)
            );
        }

        cs.sort();

        let expected_priorities: Vec<cl_int> =
            ((cache_capacity + 1) as cl_int..=(cache_capacity * 2) as cl_int).collect();
        assert_eq!(cs.priorities, expected_priorities);
    }

    #[test]
    fn cache_is_full_2() {
        let cache_capacity = 64;
        let cache_key_len = 256;
        let cache_value_len = 256;

        let mut cache_src = CacheSrc::new();
        cache_src.add_lru(cache_key_len, cache_value_len, cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_lru_by_id(0).unwrap();
        let cache = LRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let test_matrix = TestMatrix::new(cache_capacity, cache_key_len, cache_value_len, 1, 10);

        let _ = cache
            .insert(&test_matrix.keys, &test_matrix.values)
            .unwrap();

        let test_matrix_2 = TestMatrix::new(
            cache_capacity,
            cache_key_len,
            cache_value_len,
            10 + cache_capacity as cl_int,
            100,
        );

        let indices = cache
            .insert(&test_matrix_2.keys, &test_matrix_2.values)
            .unwrap();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        let expected: Vec<cl_int> = (0..cache_capacity as cl_int).collect();
        assert_eq!(indices_sorted, expected);

        let mut cs = cache.print().unwrap();

        for (i, c_i) in indices.into_iter().enumerate() {
            let ci = c_i as usize;
            assert_eq!(
                cs.keys[ci],
                ensure_vec_size(&test_matrix_2.keys[i], cache_key_len)
            );
            assert_eq!(
                cs.values[ci],
                ensure_vec_size(&test_matrix_2.values[i], cache_value_len)
            );
        }

        cs.sort();

        let expected_priorities: Vec<cl_int> =
            ((cache_capacity + 1) as cl_int..=(cache_capacity * 2) as cl_int).collect();
        assert_eq!(cs.priorities, expected_priorities);
    }

    #[test]
    fn cache_is_full_3() {
        let cache_capacity = 256;
        let cache_key_len = 32;
        let cache_value_len = 32;

        let mut cache_src = CacheSrc::new();
        cache_src.add_lru(cache_key_len, cache_value_len, cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_lru_by_id(0).unwrap();
        let cache = LRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let test_matrix = TestMatrix::new(cache_capacity, cache_key_len, cache_value_len, 1, 10);

        let _ = cache
            .insert(&test_matrix.keys, &test_matrix.values)
            .unwrap();

        let test_matrix_2 = TestMatrix::new(
            cache_capacity,
            cache_key_len,
            cache_value_len,
            10 + cache_capacity as cl_int,
            100,
        );

        let indices = cache
            .insert(&test_matrix_2.keys, &test_matrix_2.values)
            .unwrap();

        let mut cs = cache.print().unwrap();
        cs.sort();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        assert!(has_unique_elements(&cs.array_set.items));

        let expected: Vec<cl_int> = (0..cache_capacity as cl_int).collect();
        assert_eq!(indices_sorted, expected);

        for (i, c_i) in indices.into_iter().enumerate() {
            let ci = c_i as usize;
            assert_eq!(
                cs.keys[ci],
                ensure_vec_size(&test_matrix_2.keys[i], cache_key_len)
            );
            assert_eq!(
                cs.values[ci],
                ensure_vec_size(&test_matrix_2.values[i], cache_value_len)
            );
        }

        let expected_priorities: Vec<cl_int> =
            ((cache_capacity + 1) as cl_int..=(cache_capacity * 2) as cl_int).collect();
        assert_eq!(cs.priorities, expected_priorities);
    }

    #[test]
    fn repeated_values() {
        let cache_capacity = 32;
        let cache_key_len = 256;
        let cache_value_len = 256;

        let mut cache_src = CacheSrc::new();
        cache_src.add_lru(cache_key_len, cache_value_len, cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_lru_by_id(0).unwrap();
        let cache = LRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let keys = vec![vec![1; cache_key_len]; cache_capacity];
        let values = vec![vec![10; cache_value_len]; cache_capacity];

        let indices = cache.insert(&keys, &values).unwrap();
        assert_eq!(indices, vec![0; cache_capacity]);

        let cs = cache.print().unwrap();
        assert_eq!(cs.summary(), LruSummary::with(1));
        assert_eq!(cs.last_priority, cache_capacity as cl_int);
        assert!(cs.has_entry(&keys[0], &values[0]));
    }
}

#[cfg(test)]
mod tests_lru_cache_put {
    use super::*;
    use crate::cache::config::CacheSrc;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::test_utils::TestMatrix;
    use crate::utils::ensure_vec_size;
    use opencl::wrapper::system::System;

    #[test]
    fn cache_is_empty() {
        let cache_capacity = 32;
        let cache_key_len = 256;
        let cache_value_len = 256;

        let mut cache_src = CacheSrc::new();
        cache_src.add_lru(cache_key_len, cache_value_len, cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_lru_by_id(0).unwrap();
        let cache = LRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let test_matrix = TestMatrix::new(cache_capacity, cache_key_len, cache_value_len, 1, 10);
        let priorities: Vec<cl_int> = test_matrix
            .indices
            .iter()
            .map(|&x| x as cl_int + 1)
            .collect();

        cache
            .put(&test_matrix.keys, &test_matrix.values, &priorities)
            .unwrap();

        let cs = cache.print().unwrap();

        for (i, key) in test_matrix.keys.into_iter().enumerate() {
            assert_eq!(cs.keys[i], ensure_vec_size(&key, cache_key_len));
            assert_eq!(
                cs.values[i],
                ensure_vec_size(&test_matrix.values[i], cache_value_len)
            );
        }

        assert_eq!(cs.priorities, priorities);
    }
}

#[cfg(test)]
mod tests_lru_cache_get {
    use super::*;
    use crate::cache::config::CacheSrc;
    use crate::config::{ClTypeDefault, DEFAULT_DEVICE_INDEX};
    use crate::test_utils::TestMatrix;
    use opencl::wrapper::system::System;

    #[test]
    fn cache_is_empty() {
        let cache_capacity = 32;
        let cache_key_len = 256;
        let cache_value_len = 256;

        let mut cache_src = CacheSrc::new();
        cache_src.add_lru(cache_key_len, cache_value_len, cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_lru_by_id(0).unwrap();
        let cache = LRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let test_matrix = TestMatrix::new(cache_capacity, cache_key_len, cache_value_len, 1, 10);

        let (values, indices) = cache.get(&test_matrix.keys).unwrap();

        assert_eq!(
            values,
            vec![vec![i32::cl_default(); cache_value_len]; cache_capacity]
        );
        assert_eq!(indices, vec![-1; cache_capacity]);

        let cs = cache.print().unwrap();
        assert_eq!(cs.priorities, vec![0; cache_capacity]);
    }

    #[test]
    fn cache_is_full() {
        let cache_capacity = 32;
        let cache_key_len = 256;
        let cache_value_len = 256;

        let mut cache_src = CacheSrc::new();
        cache_src.add_lru(cache_key_len, cache_value_len, cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_lru_by_id(0).unwrap();
        let cache = LRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let test_matrix = TestMatrix::new(cache_capacity, cache_key_len, cache_value_len, 1, 10);

        let _ = cache
            .insert(&test_matrix.keys, &test_matrix.values)
            .unwrap();
        let (values, indices) = cache.get(&test_matrix.keys).unwrap();

        for (i, value) in test_matrix.values.into_iter().enumerate() {
            assert_eq!(values[i], value);
        }

        // FIXME update indices assert
        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        let indices_expected: Vec<cl_int> = (0..cache_capacity as cl_int).collect();
        assert_eq!(indices_sorted, indices_expected);

        let mut cs = cache.print().unwrap();
        cs.sort();

        let expected_priorities: Vec<cl_int> = indices_expected
            .iter()
            .map(|x| x + 1 + cache_capacity as cl_int)
            .collect();
        assert_eq!(cs.priorities, expected_priorities);
    }

    #[test]
    fn cache_is_full_2() {
        let cache_capacity = 1024;
        let cache_key_len = 32;
        let cache_value_len = 32;

        let mut cache_src = CacheSrc::new();
        cache_src.add_lru(cache_key_len, cache_value_len, cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_lru_by_id(0).unwrap();
        let cache = LRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let test_matrix = TestMatrix::new(cache_capacity, cache_key_len, cache_value_len, 1, 10);

        let _ = cache
            .insert(&test_matrix.keys, &test_matrix.values)
            .unwrap();
        let (values, indices) = cache.get(&test_matrix.keys).unwrap();

        for (i, value) in test_matrix.values.into_iter().enumerate() {
            assert_eq!(values[i], value);
        }

        // FIXME update indices assert
        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        let indices_expected: Vec<cl_int> = (0..cache_capacity as cl_int).collect();
        assert_eq!(indices_sorted, indices_expected);

        let mut cs = cache.print().unwrap();
        cs.sort();

        let expected_priorities: Vec<cl_int> = indices_expected
            .iter()
            .map(|x| x + 1 + cache_capacity as cl_int)
            .collect();
        assert_eq!(cs.priorities, expected_priorities);
    }

    #[test]
    fn partially_full_cache() {
        let cache_capacity = 32;
        let cache_key_len = 256;
        let cache_value_len = 256;

        let mut cache_src = CacheSrc::new();
        cache_src.add_lru(cache_key_len, cache_value_len, cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_lru_by_id(0).unwrap();
        let cache = LRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let test_matrix = TestMatrix::new(cache_capacity, cache_key_len, cache_value_len, 1, 10);

        let _ = cache
            .insert(
                &test_matrix.keys[0..(cache_capacity / 2)].to_vec(),
                &test_matrix.values[0..(cache_capacity / 2)].to_vec(),
            )
            .unwrap();
        let (values, indices) = cache.get(&test_matrix.keys).unwrap();

        for (i, value) in test_matrix.values.into_iter().enumerate() {
            if i >= (cache_capacity / 2) {
                assert_eq!(values[i], vec![i32::cl_default(); cache_value_len]);
                continue;
            }

            assert_eq!(value, values[i]);
        }

        // FIXME update indices assert
        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        let expected_ind: Vec<cl_int> = (0..(cache_capacity / 2) as cl_int).collect();

        let mut expected_indices = vec![i32::cl_default(); cache_capacity / 2];
        expected_indices.append(&mut expected_ind.clone());
        assert_eq!(indices_sorted, expected_indices);

        let mut cs = cache.print().unwrap();
        cs.sort();

        let mut expected_priorities: Vec<cl_int> = vec![0; cache_capacity / 2];
        expected_priorities.append(
            &mut expected_ind
                .iter()
                .map(|x| x + 1 + (cache_capacity / 2) as cl_int)
                .collect(),
        );
        assert_eq!(cs.priorities, expected_priorities);
    }
}

#[cfg(test)]
mod tests_lru_cache_get_keys {
    use super::*;
    use crate::cache::config::CacheSrc;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::test_utils::TestMatrix;
    use opencl::wrapper::system::System;

    #[test]
    fn cache_is_empty() {
        let cache_capacity = 32;
        let cache_key_len = 256;
        let cache_value_len = 256;

        let mut cache_src = CacheSrc::new();
        cache_src.add_lru(cache_key_len, cache_value_len, cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_lru_by_id(0).unwrap();
        let cache = LRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let keys = cache.keys().unwrap();

        assert!(keys.is_empty());
        cache.print().unwrap();
    }

    #[test]
    fn cache_is_full_1() {
        let cache_capacity = 32;
        let cache_key_len = 256;
        let cache_value_len = 256;

        let mut cache_src = CacheSrc::new();
        cache_src.add_lru(cache_key_len, cache_value_len, cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_lru_by_id(0).unwrap();
        let cache = LRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let mut test_matrix =
            TestMatrix::new(cache_capacity, cache_key_len, cache_value_len, 1, 10);
        let priorities: Vec<cl_int> = test_matrix
            .indices
            .iter()
            .map(|&x| x as cl_int + 1)
            .collect();

        cache
            .put(&test_matrix.keys, &test_matrix.values, &priorities)
            .unwrap();

        let keys = cache.keys().unwrap();

        for (i, key_p) in keys.into_iter().enumerate() {
            println!("{key_p:?}");
            assert_eq!(key_p.key, test_matrix.keys[i]);
        }

        println!("sorted");

        let keys = cache.sorted_keys().unwrap();

        test_matrix.keys.reverse();

        for (i, key_p) in keys.into_iter().enumerate() {
            println!("{key_p:?}");
            assert_eq!(key_p.key, test_matrix.keys[i]);
        }

        cache.print().unwrap();
    }

    #[test]
    fn cache_is_full_1_large() {
        let cache_capacity = 1024;
        let cache_key_len = 32;
        let cache_value_len = 32;

        let mut cache_src = CacheSrc::new();
        cache_src.add_lru(cache_key_len, cache_value_len, cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_lru_by_id(0).unwrap();
        let cache = LRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let mut test_matrix =
            TestMatrix::new(cache_capacity, cache_key_len, cache_value_len, 1, 10);
        let priorities: Vec<cl_int> = test_matrix
            .indices
            .iter()
            .map(|&x| x as cl_int + 1)
            .collect();

        cache
            .put(&test_matrix.keys, &test_matrix.values, &priorities)
            .unwrap();

        let keys = cache.keys().unwrap();

        for (i, key_p) in keys.into_iter().enumerate() {
            println!("{key_p:?}");
            assert_eq!(key_p.key, test_matrix.keys[i]);
        }

        println!("sorted");

        let keys = cache.sorted_keys().unwrap();

        test_matrix.keys.reverse();

        for (i, key_p) in keys.into_iter().enumerate() {
            println!("{key_p:?}");
            assert_eq!(key_p.key, test_matrix.keys[i]);
        }

        cache.print().unwrap();
    }

    #[test]
    fn cache_is_full_2() {
        let cache_capacity = 32;
        let cache_key_len = 256;
        let cache_value_len = 256;

        let mut cache_src = CacheSrc::new();
        cache_src.add_lru(cache_key_len, cache_value_len, cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_lru_by_id(0).unwrap();
        let cache = LRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let mut test_matrix =
            TestMatrix::new(cache_capacity, cache_key_len, cache_value_len, 1, 10);
        let priorities: Vec<cl_int> = test_matrix
            .indices
            .iter()
            .map(|&x| x as cl_int + 1)
            .collect();

        cache
            .put(&test_matrix.keys, &test_matrix.values, &priorities)
            .unwrap();
        let _ = cache.get(&test_matrix.keys[0..(cache_capacity / 2)].to_vec());

        let keys = cache.keys().unwrap();

        for (i, key_p) in keys.into_iter().enumerate() {
            println!("{key_p:?}");
            assert_eq!(key_p.key, test_matrix.keys[i]);
        }

        println!("sorted");

        let keys = cache.sorted_keys().unwrap();

        test_matrix.keys.reverse();
        let mut expected_keys = test_matrix.keys[(cache_capacity / 2)..].to_vec();
        expected_keys.append(&mut test_matrix.keys[0..(cache_capacity / 2)].to_vec());

        for (i, key_p) in keys.into_iter().enumerate() {
            println!("{key_p:?}");
            assert_eq!(key_p.key, expected_keys[i]);
        }

        cache.print().unwrap();
    }

    #[test]
    fn cache_is_full_2_large() {
        let cache_capacity = 1024;
        let cache_key_len = 32;
        let cache_value_len = 32;

        let mut cache_src = CacheSrc::new();
        cache_src.add_lru(cache_key_len, cache_value_len, cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_lru_by_id(0).unwrap();
        let cache = LRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let mut test_matrix =
            TestMatrix::new(cache_capacity, cache_key_len, cache_value_len, 1, 10);
        let priorities: Vec<cl_int> = test_matrix
            .indices
            .iter()
            .map(|&x| x as cl_int + 1)
            .collect();

        cache
            .put(&test_matrix.keys, &test_matrix.values, &priorities)
            .unwrap();
        let _ = cache.get(&test_matrix.keys[0..(cache_capacity / 2)].to_vec());

        let keys = cache.keys().unwrap();

        for (i, key_p) in keys.into_iter().enumerate() {
            println!("{key_p:?}");
            assert_eq!(key_p.key, test_matrix.keys[i]);
        }

        println!("sorted");

        let keys = cache.sorted_keys().unwrap();

        test_matrix.keys.reverse();
        let mut expected_keys = test_matrix.keys[(cache_capacity / 2)..].to_vec();
        expected_keys.append(&mut test_matrix.keys[0..(cache_capacity / 2)].to_vec());

        for (i, key_p) in keys.into_iter().enumerate() {
            println!("{key_p:?}");
            assert_eq!(key_p.key, expected_keys[i]);
        }

        cache.print().unwrap();
    }
}

#[cfg(test)]
mod tests_lru_cache_get_sorted_keys {
    use super::*;
    use crate::cache::config::CacheSrc;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::test_utils::TestMatrix;
    use opencl::wrapper::system::System;

    #[test]
    fn cache_is_empty() {
        let cache_capacity = 32;
        let cache_key_len = 256;
        let cache_value_len = 256;

        let mut cache_src = CacheSrc::new();
        cache_src.add_lru(cache_key_len, cache_value_len, cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_lru_by_id(0).unwrap();
        let cache = LRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let keys = cache.get_sorted_keys().unwrap();

        assert!(keys.is_empty());
        cache.print().unwrap();
    }

    #[test]
    fn cache_is_full_1() {
        let cache_capacity = 32;
        let cache_key_len = 256;
        let cache_value_len = 256;

        let mut cache_src = CacheSrc::new();
        cache_src.add_lru(cache_key_len, cache_value_len, cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_lru_by_id(0).unwrap();
        let cache = LRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let mut test_matrix =
            TestMatrix::new(cache_capacity, cache_key_len, cache_value_len, 1, 10);
        let priorities: Vec<cl_int> = test_matrix
            .indices
            .iter()
            .map(|&x| x as cl_int + 1)
            .collect();

        cache
            .put(&test_matrix.keys, &test_matrix.values, &priorities)
            .unwrap();

        let keys = cache.get_sorted_keys().unwrap();

        test_matrix.keys.reverse();

        for (i, key_p) in keys.into_iter().enumerate() {
            println!("{key_p:?}");
            assert_eq!(key_p.key, test_matrix.keys[i]);
        }

        cache.print().unwrap();
    }

    #[test]
    fn cache_is_full_1_large() {
        let cache_capacity = 1024;
        let cache_key_len = 32;
        let cache_value_len = 32;

        let mut cache_src = CacheSrc::new();
        cache_src.add_lru(cache_key_len, cache_value_len, cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_lru_by_id(0).unwrap();
        let cache = LRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let mut test_matrix =
            TestMatrix::new(cache_capacity, cache_key_len, cache_value_len, 1, 10);
        let priorities: Vec<cl_int> = test_matrix
            .indices
            .iter()
            .map(|&x| x as cl_int + 1)
            .collect();

        cache
            .put(&test_matrix.keys, &test_matrix.values, &priorities)
            .unwrap();

        let keys = cache.get_sorted_keys().unwrap();

        test_matrix.keys.reverse();

        for (i, key_p) in keys.into_iter().enumerate() {
            println!("{key_p:?}");
            assert_eq!(key_p.key, test_matrix.keys[i]);
        }

        cache.print().unwrap();
    }

    #[test]
    fn cache_is_full_2() {
        let cache_capacity = 32;
        let cache_key_len = 256;
        let cache_value_len = 256;

        let mut cache_src = CacheSrc::new();
        cache_src.add_lru(cache_key_len, cache_value_len, cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_lru_by_id(0).unwrap();
        let cache = LRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let mut test_matrix =
            TestMatrix::new(cache_capacity, cache_key_len, cache_value_len, 1, 10);
        let priorities: Vec<cl_int> = test_matrix
            .indices
            .iter()
            .map(|&x| x as cl_int + 1)
            .collect();

        cache
            .put(&test_matrix.keys, &test_matrix.values, &priorities)
            .unwrap();
        let _ = cache.get(&test_matrix.keys[0..(cache_capacity / 2)].to_vec());

        let keys = cache.get_sorted_keys().unwrap();

        test_matrix.keys.reverse();
        let mut expected_keys = test_matrix.keys[(cache_capacity / 2)..].to_vec();
        expected_keys.append(&mut test_matrix.keys[..(cache_capacity / 2)].to_vec());

        for (i, key_p) in keys.into_iter().enumerate() {
            println!("{key_p:?}");
            assert_eq!(key_p.key, expected_keys[i]);
        }

        cache.print().unwrap();
        // cache.debug_sort().unwrap();
    }

    #[test]
    fn cache_is_full_2_large() {
        let cache_capacity = 1024;
        let cache_key_len = 32;
        let cache_value_len = 32;

        let mut cache_src = CacheSrc::new();
        cache_src.add_lru(cache_key_len, cache_value_len, cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_lru_by_id(0).unwrap();
        let cache = LRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let mut test_matrix =
            TestMatrix::new(cache_capacity, cache_key_len, cache_value_len, 1, 10);
        let priorities: Vec<cl_int> = test_matrix
            .indices
            .iter()
            .map(|&x| x as cl_int + 1)
            .collect();

        cache
            .put(&test_matrix.keys, &test_matrix.values, &priorities)
            .unwrap();
        let _ = cache.get(&test_matrix.keys[0..(cache_capacity / 2)].to_vec());

        let keys = cache.get_sorted_keys().unwrap();

        test_matrix.keys.reverse();
        let mut expected_keys = test_matrix.keys[(cache_capacity / 2)..].to_vec();
        expected_keys.append(&mut test_matrix.keys[..(cache_capacity / 2)].to_vec());

        for (i, key_p) in keys.into_iter().enumerate() {
            println!("{key_p:?}");
            assert_eq!(key_p.key, expected_keys[i]);
        }

        cache.print().unwrap();
        // cache.debug_sort().unwrap();
    }
}

#[cfg(test)]
mod tests_lru_cache_sort {
    use super::*;
    use crate::cache::config::CacheSrc;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::test_utils::TestMatrix;
    use opencl::wrapper::system::System;

    #[test]
    fn cache_is_empty() {
        let cache_capacity = 32;
        let cache_key_len = 256;
        let cache_value_len = 256;

        let mut cache_src = CacheSrc::new();
        cache_src.add_lru(cache_key_len, cache_value_len, cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_lru_by_id(0).unwrap();
        let cache = LRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let r = cache.sort();
        assert_eq!(r, Ok(()));

        let cs = cache.print().unwrap();
        assert_eq!(cs, LRUCacheSnapshot::create_empty(256, 256, cache_capacity));

        cache.debug_sort().unwrap();
    }

    #[test]
    fn cache_is_full() {
        let cache_capacity = 32;
        let cache_key_len = 256;
        let cache_value_len = 256;

        let mut cache_src = CacheSrc::new();
        cache_src.add_lru(cache_key_len, cache_value_len, cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_lru_by_id(0).unwrap();
        let cache = LRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let test_matrix = TestMatrix::new(cache_capacity, cache_key_len, cache_value_len, 1, 10);

        let _ = cache
            .insert(&test_matrix.keys, &test_matrix.values)
            .unwrap();

        let _cs = cache.print().unwrap();

        let r = cache.sort();
        assert_eq!(r, Ok(()));

        println!("sorted");
        let cs = cache.print().unwrap();

        let mut expected_priorities: Vec<_> = test_matrix
            .indices
            .iter()
            .map(|&x| x as cl_int + 1)
            .collect();
        expected_priorities.reverse();
        assert_eq!(cs.priorities, expected_priorities);

        cache.debug_sort().unwrap();
    }

    #[test]
    fn cache_is_full_2() {
        let cache_capacity = 1024;
        let cache_key_len = 32;
        let cache_value_len = 32;

        let mut cache_src = CacheSrc::new();
        cache_src.add_lru(cache_key_len, cache_value_len, cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_lru_by_id(0).unwrap();
        let cache = LRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let test_matrix = TestMatrix::new(cache_capacity, cache_key_len, cache_value_len, 1, 10);

        let _ = cache
            .insert(&test_matrix.keys, &test_matrix.values)
            .unwrap();

        let _cs = cache.print().unwrap();

        let r = cache.sort();
        assert_eq!(r, Ok(()));

        println!("sorted");
        let cs = cache.print().unwrap();

        let mut expected_priorities: Vec<_> = test_matrix
            .indices
            .iter()
            .map(|&x| x as cl_int + 1)
            .collect();
        expected_priorities.reverse();
        assert_eq!(cs.priorities, expected_priorities);

        cache.debug_sort().unwrap();
    }
}

// TODO explain tests
#[cfg(test)]
mod tests_lru_cache_examples {
    use super::*;
    use crate::cache::config::CacheSrc;
    use crate::config::{ClTypeDefault, DEFAULT_DEVICE_INDEX};
    use crate::test_utils::TestMatrix;
    use crate::utils::has_unique_elements;
    use opencl::wrapper::system::System;

    #[test]
    fn case_1() {
        let cache_capacity = 8;
        let cache_key_len = 32;
        let cache_value_len = 32;

        let mut cache_src = CacheSrc::new();
        cache_src.add_lru(cache_key_len, cache_value_len, cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_lru_by_id(0).unwrap();
        let cache = LRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        println!("put 1");
        let test_matrix = TestMatrix::new(cache_capacity, cache_key_len, cache_value_len, 1, 10);

        let indices = cache.add(&test_matrix.keys, &test_matrix.values).unwrap();
        let _cs = cache.print().unwrap();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        let expected: Vec<cl_int> = (0..cache_capacity as cl_int).collect();
        assert_eq!(indices_sorted, expected);

        println!("put 2");
        let keys_2 = vec![vec![10; cache_key_len], vec![20; cache_key_len]];
        let values_2 = vec![
            vec![100 as cl_int; cache_value_len],
            vec![200 as cl_int; cache_value_len],
        ];

        let indices = cache.add(&keys_2, &values_2).unwrap();
        let _cs = cache.print().unwrap();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        assert_eq!(indices_sorted, vec![0, 1]);

        println!("put 3");
        let keys_3 = vec![vec![100; cache_key_len], vec![200; cache_key_len]];
        let values_3 = vec![
            vec![11 as cl_int; cache_value_len],
            vec![22 as cl_int; cache_value_len],
        ];

        let indices = cache.add(&keys_3, &values_3).unwrap();
        let _cs = cache.print().unwrap();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        assert_eq!(indices_sorted, vec![2, 3]);

        println!("put 4");
        let keys_4 = vec![
            vec![100; cache_key_len],
            vec![200; cache_key_len],
            vec![100; cache_key_len],
            vec![200; cache_key_len],
        ];
        let values_4 = vec![
            vec![11 as cl_int; cache_value_len],
            vec![22 as cl_int; cache_value_len],
            vec![111 as cl_int; cache_value_len],
            vec![222 as cl_int; cache_value_len],
        ];

        let indices = cache.add(&keys_4, &values_4).unwrap();
        let mut cs = cache.print().unwrap();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        assert_eq!(indices_sorted, vec![2, 2, 3, 3]);

        println!("put 5");
        let keys_5 = cs.get_lower_priority_keys(4);
        let values_5 = vec![vec![50; cache_value_len]; 4];

        let indices = cache.add(&keys_5, &values_5).unwrap();
        let _cs = cache.print().unwrap();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        assert_eq!(indices_sorted, vec![4, 5, 6, 7]);
    }

    #[test]
    fn case_2() {
        let cache_capacity = 16;
        let cache_key_len = 256;
        let cache_value_len = 256;

        let mut cache_src = CacheSrc::new();
        cache_src.add_lru(cache_key_len, cache_value_len, cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_lru_by_id(0).unwrap();
        let cache = LRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let test_matrix = TestMatrix::new(cache_capacity, cache_key_len, cache_value_len, 1, 11);

        let _ = cache
            .insert(&test_matrix.keys, &test_matrix.values)
            .unwrap();

        println!("get 1");
        let (values, _indices) = cache.get(&vec![test_matrix.keys[1].to_vec()]).unwrap();
        assert_eq!(values, vec![test_matrix.values[1].to_vec()]);

        println!("get 2");
        let (values, _indices) = cache.get(&test_matrix.keys[3..7].to_vec()).unwrap();
        assert_eq!(values, test_matrix.values[3..7].to_vec());

        println!("get 3");
        let mut keys = vec![test_matrix.keys[1].to_vec(), test_matrix.keys[3].to_vec()];
        keys.push(vec![300; cache_key_len]);
        keys.push(vec![400; cache_key_len]);

        let (values, _indices) = cache.get(&keys).unwrap();
        let mut expected = vec![
            test_matrix.values[1].to_vec(),
            test_matrix.values[3].to_vec(),
        ];
        expected.append(&mut vec![vec![i32::cl_default(); cache_key_len]; 2]);
        assert_eq!(values, expected);

        println!("get 4");
        let key = test_matrix.keys[0].to_vec();
        let (values, _indices) = cache.get(&vec![key; 8]).unwrap();
        let expected_value = test_matrix.values[0].to_vec();
        assert_eq!(values, vec![expected_value; 8]);

        let cs = cache.print().unwrap();
        assert!(has_unique_elements(&cs.priorities));
    }
}
