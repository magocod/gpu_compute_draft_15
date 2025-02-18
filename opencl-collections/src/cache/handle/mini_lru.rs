use crate::cache::config::CacheConfig;
use crate::cache::handle::{CacheIndices, LruSummary};
use crate::cache::kernel::name::{
    get_cache_kernel_name, MINI_LRU_CACHE_ARRAY_SET_RESET, MINI_LRU_CACHE_DEBUG,
    MINI_LRU_CACHE_GET_KEYS, MINI_LRU_CACHE_GET_SORTED_KEYS, MINI_LRU_CACHE_PUT,
    MINI_LRU_CACHE_RESET, MINI_LRU_CACHE_SORT, READ_ON_MINI_LRU_CACHE, WRITE_IN_MINI_LRU_CACHE,
};
use crate::config::{ClTypeDefault, DEBUG_MODE};
use crate::error::OpenClResult;
use crate::set::handle::array_set_v2::ArraySetSnapshot;
use opencl::opencl_sys::bindings::{cl_int, cl_uint};
use opencl::wrapper::system::OpenclCommonOperation;
use std::sync::Arc;

#[derive(Debug, PartialEq)]
pub struct MiniLRUCacheSnapshot {
    pub last_priority: cl_int,
    pub top: cl_int,
    pub keys: Vec<cl_int>,
    pub values: Vec<cl_int>,
    pub priorities: Vec<cl_int>,
    pub array_set: ArraySetSnapshot,
}

impl MiniLRUCacheSnapshot {
    pub fn new(
        last_priority: cl_int,
        top: cl_int,
        keys: Vec<cl_int>,
        values: Vec<cl_int>,
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

    pub fn create_empty(capacity: usize) -> Self {
        Self::new(
            1,
            0,
            vec![i32::cl_default(); capacity],
            vec![i32::cl_default(); capacity],
            vec![0; capacity],
            ArraySetSnapshot::create_empty(capacity),
        )
    }

    pub fn get_lower_priority_keys(&mut self, take: usize) -> Vec<cl_int> {
        let mut keys: Vec<cl_int> = vec![];

        for _ in 0..take {
            let key_index: Option<usize> = self
                .priorities
                .iter()
                .enumerate()
                .filter(|(_i, &x)| x != i32::cl_default())
                .min_by(|(_i, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index);

            if let Some(i) = key_index {
                keys.push(self.keys[i]);
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
                .filter(|&&x| x != i32::cl_default())
                .count(),
            self.values
                .iter()
                .filter(|&&x| x != i32::cl_default())
                .count(),
            self.priorities
                .iter()
                .filter(|&&x| x != i32::cl_default())
                .count(),
        )
    }

    pub fn sort(&mut self) {
        self.keys.sort();
        self.values.sort();
        self.priorities.sort();
    }

    pub fn has_entry(&self, key: cl_int, value: cl_int) -> bool {
        // self.keys.iter().any(|&x| x == key)
        match self.keys.iter().position(|&x| x == key) {
            None => false,
            Some(index) => self.values[index] == value,
        }
    }

    // debug only

    pub fn print_all_key(&self) {
        for (i, key) in self.keys.iter().enumerate() {
            println!(
                "key: {}, value: {}, priority: {}",
                key, self.values[i], self.priorities[i]
            );
        }
    }

    pub fn print_key(&self, key: cl_int) {
        let key_index = self.keys.iter().position(|&x| x == key).unwrap();
        println!(
            "key: {}, value: {}, priority: {}",
            self.keys[key_index], self.values[key_index], self.priorities[key_index]
        );
    }
}

pub type CacheValues = Vec<cl_int>;

#[derive(Debug)]
pub struct MiniLRUCacheHandle<T: OpenclCommonOperation> {
    config: CacheConfig,
    system: Arc<T>,
}

#[derive(Debug)]
pub struct KeyPriority {
    pub priority: cl_int,
    pub key: cl_int,
}

impl<T: OpenclCommonOperation> MiniLRUCacheHandle<T> {
    pub fn new(config: &CacheConfig, system: Arc<T>) -> Self {
        Self {
            config: config.clone(),
            system,
        }
    }

    pub fn get_id(&self) -> usize {
        self.config.id
    }

    pub fn debug(&self) -> OpenClResult<MiniLRUCacheSnapshot> {
        let global_work_size = self.config.capacity;
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let meta_output_capacity = 2;
        let set_items_output_capacity = global_work_size * 2;

        let keys_output_buf = self.system.create_output_buffer(global_work_size)?;
        let values_output_buf = self.system.create_output_buffer(global_work_size)?;
        let priorities_output_buf = self.system.create_output_buffer(global_work_size)?;

        let meta_output_buf = self.system.create_output_buffer(meta_output_capacity)?;
        let set_items_output_buf = self
            .system
            .create_output_buffer(set_items_output_capacity)?;

        let kernel_name = get_cache_kernel_name(MINI_LRU_CACHE_DEBUG, self.get_id());
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

        let keys_output =
            self.system
                .blocking_enqueue_read_buffer(global_work_size, &keys_output_buf, &[])?;

        let values_output =
            self.system
                .blocking_enqueue_read_buffer(global_work_size, &values_output_buf, &[])?;

        let priorities_output = self.system.blocking_enqueue_read_buffer(
            global_work_size,
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

        let items = set_items_output[0..self.config.capacity].to_vec();
        let entries = set_items_output[self.config.capacity..].to_vec();

        let array_set = ArraySetSnapshot::new(items, entries);

        Ok(MiniLRUCacheSnapshot {
            last_priority: meta_output[0],
            top: meta_output[1],
            keys: keys_output,
            values: values_output,
            priorities: priorities_output,
            array_set,
        })
    }

    pub fn print(&self) -> OpenClResult<MiniLRUCacheSnapshot> {
        let cs = self.debug()?;
        // println!("{qs:?}");
        println!(
            "
MiniLRUCacheSnapshot (
   last_priority: {},
   top:           {},
   keys:       {:?},
   values:     {:?},
   priorities: {:?},
   ArraySetSnapshot: (
     items:   {:?},
     entries: {:?}
   )
)
        ",
            cs.last_priority,
            cs.top,
            cs.keys,
            cs.values,
            cs.priorities,
            cs.array_set.items,
            cs.array_set.entries
        );
        Ok(cs)
    }

    pub fn reset(&self) -> OpenClResult<()> {
        let global_work_size = self.config.capacity;
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let kernel_name = get_cache_kernel_name(MINI_LRU_CACHE_RESET, self.get_id());
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

    pub fn insert(&self, keys: &[cl_int], values: &[cl_int]) -> OpenClResult<Vec<cl_int>> {
        if keys.len() != values.len() {
            panic!("error handle keys & values len")
        }

        let global_work_size = keys.len();
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let indices_output_capacity = global_work_size;

        let keys_input_buf = self.system.blocking_prepare_input_buffer(keys)?;
        let values_input_buf = self.system.blocking_prepare_input_buffer(values)?;

        let indices_output_buf = self.system.create_output_buffer(indices_output_capacity)?;

        let kernel_name = get_cache_kernel_name(WRITE_IN_MINI_LRU_CACHE, self.get_id());
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
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

        let kernel_name = get_cache_kernel_name(MINI_LRU_CACHE_ARRAY_SET_RESET, self.get_id());
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

    pub fn add(&self, keys: &[cl_int], values: &[cl_int]) -> OpenClResult<Vec<cl_int>> {
        let r = self.insert(keys, values)?;
        self.reset_array_set()?;
        Ok(r)
    }

    pub fn put(
        &self,
        keys: &[cl_int],
        values: &[cl_int],
        priorities: &[cl_int],
    ) -> OpenClResult<()> {
        if keys.len() != values.len() {
            panic!("error handle keys & values len");
        }

        if keys.len() != priorities.len() {
            panic!("error handle keys & priorities len");
        }

        let global_work_size = keys.len();
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let keys_input_buf = self.system.blocking_prepare_input_buffer(keys)?;
        let values_input_buf = self.system.blocking_prepare_input_buffer(values)?;
        let priorities_input_buf = self.system.blocking_prepare_input_buffer(priorities)?;

        let kernel_name = get_cache_kernel_name(MINI_LRU_CACHE_PUT, self.get_id());
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

    pub fn get(&self, keys: &[cl_int]) -> OpenClResult<(CacheValues, CacheIndices)> {
        let global_work_size = keys.len();
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let values_output_capacity = global_work_size;
        let indices_output_capacity = global_work_size;

        let keys_input_buf = self.system.blocking_prepare_input_buffer(keys)?;
        let values_output_buf = self.system.create_output_buffer(values_output_capacity)?;

        let indices_output_buf = self.system.create_output_buffer(indices_output_capacity)?;

        let kernel_name = get_cache_kernel_name(READ_ON_MINI_LRU_CACHE, self.get_id());
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
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

        if DEBUG_MODE {
            println!("values_output  {values_output:?}");
            println!("indices_output {indices_output:?}");
        }

        Ok((values_output, indices_output))
    }

    pub fn keys(&self) -> OpenClResult<Vec<KeyPriority>> {
        let global_work_size = self.config.capacity;
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let keys_output_buf = self.system.create_output_buffer(global_work_size)?;
        let priorities_output_buf = self.system.create_output_buffer(global_work_size)?;

        let kernel_name = get_cache_kernel_name(MINI_LRU_CACHE_GET_KEYS, self.get_id());
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

        let keys_output =
            self.system
                .blocking_enqueue_read_buffer(global_work_size, &keys_output_buf, &[])?;

        let priorities_output = self.system.blocking_enqueue_read_buffer(
            global_work_size,
            &priorities_output_buf,
            &[],
        )?;

        let keys: Vec<KeyPriority> = keys_output
            .into_iter()
            .enumerate()
            .filter_map(|(i, x)| -> Option<KeyPriority> {
                let priority = priorities_output[i];

                if priority == 0 {
                    return None;
                }

                Some(KeyPriority {
                    key: x,
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

        // CMQ_SORT
        // CMQ_GET_KEYS
        let enqueue_kernel_output_capacity = 2;

        let keys_output_buf = self.system.create_output_buffer(output_capacity)?;
        let priorities_output_buf = self.system.create_output_buffer(output_capacity)?;

        let enqueue_kernel_output_buf = self
            .system
            .create_output_buffer(enqueue_kernel_output_capacity)?;

        let q0 = self
            .system
            .get_device_command_queue_0()
            .get_cl_command_queue();

        let kernel_name = get_cache_kernel_name(MINI_LRU_CACHE_GET_SORTED_KEYS, self.get_id());
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

        let keys_output =
            self.system
                .blocking_enqueue_read_buffer(output_capacity, &keys_output_buf, &[])?;

        let priorities_output: Vec<cl_int> = self.system.blocking_enqueue_read_buffer(
            output_capacity,
            &priorities_output_buf,
            &[],
        )?;

        let keys: Vec<KeyPriority> = keys_output
            .into_iter()
            .enumerate()
            .filter_map(|(i, x)| -> Option<KeyPriority> {
                let priority = priorities_output[i];

                if priority == 0 {
                    return None;
                }

                Some(KeyPriority {
                    key: x,
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

        let kernel_name = get_cache_kernel_name(MINI_LRU_CACHE_SORT, self.get_id());
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
}

#[cfg(test)]
mod tests_mini_lru_cache_reset {
    use super::*;
    use crate::cache::config::CacheSrc;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::set::handle::array_set_v2::ArraySetSnapshot;
    use opencl::wrapper::system::System;

    #[test]
    fn cache_is_empty() {
        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(32);
        cache_src.add_mini_lru(16);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_mini_lru_by_id(0).unwrap();
        let cache = MiniLRUCacheHandle::new(config, system);

        let v = vec![0; 32];

        assert_eq!(
            cache.print().unwrap(),
            MiniLRUCacheSnapshot::new(
                1,
                0,
                v.clone(),
                v.clone(),
                v.clone(),
                ArraySetSnapshot::new(v.clone(), v.clone())
            )
        );

        let r = cache.reset();
        assert!(r.is_ok());

        assert_eq!(
            cache.print().unwrap(),
            MiniLRUCacheSnapshot::create_empty(32)
        );
    }
}

#[cfg(test)]
mod tests_mini_lru_cache_debug {
    use super::*;
    use crate::cache::config::CacheSrc;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use opencl::wrapper::system::System;

    #[test]
    fn cache_is_empty() {
        let cache_capacity = 32;

        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_mini_lru_by_id(0).unwrap();
        let cache = MiniLRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let cs = cache.print().unwrap();

        assert_eq!(cs, MiniLRUCacheSnapshot::create_empty(cache_capacity));
    }

    #[test]
    fn cache_is_full() {
        let cache_capacity = 32;

        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_mini_lru_by_id(0).unwrap();
        let cache = MiniLRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let keys: Vec<cl_int> = (0..cache_capacity as cl_int).collect();
        let values: Vec<cl_int> = keys.iter().map(|x| x + 10).collect();

        let _ = cache.insert(&keys, &values).unwrap();

        let mut cs = cache.print().unwrap();
        cs.sort();

        assert_eq!(cs.keys, keys);
        assert_eq!(cs.values, values);

        let expected_priorities: Vec<_> = keys.iter().map(|x| x + 1).collect();
        assert_eq!(cs.priorities, expected_priorities);
        assert_eq!(cs.last_priority, 1 + cache_capacity as cl_int);
    }

    #[test]
    fn partially_full_cache() {
        let cache_capacity = 32;

        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_mini_lru_by_id(0).unwrap();
        let cache = MiniLRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let keys: Vec<cl_int> = (0..(cache_capacity / 2) as cl_int).collect();
        let values: Vec<cl_int> = keys.iter().map(|x| x + 10).collect();

        let _ = cache.insert(&keys, &values).unwrap();

        let mut cs = cache.print().unwrap();
        cs.sort();

        let mut expected_keys = vec![i32::cl_default(); cache_capacity / 2];
        expected_keys.append(&mut keys.clone());

        let mut expected_values = vec![i32::cl_default(); cache_capacity / 2];
        expected_values.append(&mut values.clone());

        let mut expected_priorities = vec![0; cache_capacity / 2];
        expected_priorities.append(&mut keys.iter().map(|x| x + 1).collect());

        assert_eq!(cs.keys, expected_keys);
        assert_eq!(cs.values, expected_values);
        assert_eq!(cs.priorities, expected_priorities);

        assert_eq!(cs.last_priority, 1 + (cache_capacity / 2) as cl_int);
    }
}

#[cfg(test)]
mod tests_mini_lru_cache_insert {
    use super::*;
    use crate::cache::config::CacheSrc;
    use crate::cache::handle::LruSummary;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::utils::has_unique_elements;
    use opencl::wrapper::system::System;

    #[test]
    fn cache_is_empty() {
        let cache_capacity = 32;

        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_mini_lru_by_id(0).unwrap();
        let cache = MiniLRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let keys: Vec<cl_int> = (0..cache_capacity as cl_int).collect();
        let values: Vec<cl_int> = keys.iter().map(|&x| x + 10).collect();

        let indices = cache.insert(&keys, &values).unwrap();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        let expected_indices = keys.clone();
        assert_eq!(indices_sorted, expected_indices);

        let mut cs = cache.print().unwrap();
        cs.sort();

        let expected_priorities: Vec<_> = keys.clone().iter().map(|x| x + 1).collect();

        assert_eq!(cs.keys, keys);
        assert_eq!(cs.values, values);
        assert_eq!(cs.priorities, expected_priorities);
    }

    #[test]
    fn cache_is_empty_2() {
        let cache_capacity = 256;

        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_mini_lru_by_id(0).unwrap();
        let cache = MiniLRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let keys: Vec<cl_int> = (0..cache_capacity as cl_int).collect();
        let values: Vec<cl_int> = keys.iter().map(|&x| x + 10).collect();

        let indices = cache.insert(&keys, &values).unwrap();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        let expected_indices = keys.clone();
        assert_eq!(indices_sorted, expected_indices);

        let mut cs = cache.print().unwrap();
        cs.sort();

        let expected_priorities: Vec<_> = keys.clone().iter().map(|x| x + 1).collect();

        assert_eq!(cs.keys, keys);
        assert_eq!(cs.values, values);
        assert_eq!(cs.priorities, expected_priorities);
    }

    #[test]
    fn cache_is_empty_3() {
        let cache_capacity = 1024;

        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_mini_lru_by_id(0).unwrap();
        let cache = MiniLRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let keys: Vec<cl_int> = (0..cache_capacity as cl_int).collect();
        let values: Vec<cl_int> = keys.iter().map(|&x| x + 10).collect();

        let indices = cache.insert(&keys, &values).unwrap();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        let mut cs = cache.print().unwrap();
        cs.sort();

        let expected_priorities: Vec<_> = keys.clone().iter().map(|x| x + 1).collect();
        let expected_indices = keys.clone();

        assert_eq!(indices_sorted, expected_indices);

        assert_eq!(cs.keys, keys);
        assert_eq!(cs.values, values);
        assert_eq!(cs.priorities, expected_priorities);
    }

    #[test]
    fn cache_is_full() {
        let cache_capacity = 32;

        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_mini_lru_by_id(0).unwrap();
        let cache = MiniLRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let keys: Vec<cl_int> = (0..cache_capacity as cl_int).collect();
        let values: Vec<cl_int> = keys.iter().map(|&x| x + 10).collect();

        let _ = cache.insert(&keys, &values).unwrap();

        let keys_2: Vec<cl_int> = keys.iter().map(|&x| x + cache_capacity as cl_int).collect();
        let values_2: Vec<cl_int> = values
            .iter()
            .map(|&x| x + cache_capacity as cl_int)
            .collect();

        let indices = cache.insert(&keys_2, &values_2).unwrap();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        let expected_indices: Vec<cl_int> = (0..cache_capacity as cl_int).collect();

        let mut cs = cache.print().unwrap();
        cs.sort();

        assert_eq!(indices_sorted, expected_indices);

        assert_eq!(cs.keys, keys_2);
        assert_eq!(cs.values, values_2);

        let expected_priorities: Vec<cl_int> = (cache_capacity as cl_int
            ..(cache_capacity * 2) as cl_int)
            .map(|x| x + 1)
            .collect();
        assert_eq!(cs.priorities, expected_priorities);
    }

    #[test]
    fn cache_is_full_2() {
        let cache_capacity = 64;

        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_mini_lru_by_id(0).unwrap();
        let cache = MiniLRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let keys: Vec<cl_int> = (0..cache_capacity as cl_int).collect();
        let values: Vec<cl_int> = keys.iter().map(|&x| x + 10).collect();

        let _ = cache.insert(&keys, &values).unwrap();

        let keys_2: Vec<cl_int> = keys.iter().map(|&x| x + cache_capacity as cl_int).collect();
        let values_2: Vec<cl_int> = values
            .iter()
            .map(|&x| x + cache_capacity as cl_int)
            .collect();

        let indices = cache.insert(&keys_2, &values_2).unwrap();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        let expected_indices: Vec<cl_int> = (0..cache_capacity as cl_int).collect();

        let mut cs = cache.print().unwrap();
        cs.sort();

        assert!(has_unique_elements(&cs.array_set.items));

        assert_eq!(indices_sorted, expected_indices);

        assert_eq!(cs.keys, keys_2);
        assert_eq!(cs.values, values_2);

        let expected_priorities: Vec<cl_int> = (cache_capacity as cl_int
            ..(cache_capacity * 2) as cl_int)
            .map(|x| x + 1)
            .collect();
        assert_eq!(cs.priorities, expected_priorities);

        // cache.reset_array_set().unwrap();
        // cache.print().unwrap();
    }

    #[test]
    fn cache_is_full_3() {
        let cache_capacity = 256;

        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_mini_lru_by_id(0).unwrap();
        let cache = MiniLRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let keys: Vec<cl_int> = (0..cache_capacity as cl_int).collect();
        let values: Vec<cl_int> = keys.iter().map(|&x| x + 10).collect();

        let _ = cache.insert(&keys, &values).unwrap();

        let keys_2: Vec<cl_int> = keys.iter().map(|&x| x + cache_capacity as cl_int).collect();
        let values_2: Vec<cl_int> = values
            .iter()
            .map(|&x| x + cache_capacity as cl_int)
            .collect();

        let indices = cache.insert(&keys_2, &values_2).unwrap();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        let expected_indices: Vec<cl_int> = (0..cache_capacity as cl_int).collect();

        let mut cs = cache.print().unwrap();
        cs.sort();

        assert!(has_unique_elements(&cs.priorities));

        // assert!(has_unique_elements(&indices_sorted));
        assert!(has_unique_elements(&cs.array_set.items));

        assert_eq!(indices_sorted, expected_indices);

        assert_eq!(cs.keys, keys_2);
        assert_eq!(cs.values, values_2);

        let expected_priorities: Vec<cl_int> = (cache_capacity as cl_int
            ..(cache_capacity * 2) as cl_int)
            .map(|x| x + 1)
            .collect();
        assert_eq!(cs.priorities, expected_priorities);
    }

    #[test]
    fn repeated_values() {
        let cache_capacity = 32;

        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_mini_lru_by_id(0).unwrap();
        let cache = MiniLRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let keys = vec![1; cache_capacity];
        let values = vec![10; cache_capacity];

        let indices = cache.insert(&keys, &values).unwrap();
        assert_eq!(indices, vec![0; cache_capacity]);

        let cs = cache.print().unwrap();
        assert_eq!(cs.summary(), LruSummary::with(1));
        assert_eq!(cs.last_priority, cache_capacity as cl_int);
        assert!(cs.has_entry(1, 10));
    }
}

#[cfg(test)]
mod tests_mini_lru_cache_put {
    use super::*;
    use crate::cache::config::CacheSrc;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use opencl::wrapper::system::System;

    #[test]
    fn cache_is_empty() {
        let cache_capacity = 32;

        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_mini_lru_by_id(0).unwrap();
        let cache = MiniLRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let keys: Vec<cl_int> = (0..cache_capacity as cl_int).collect();
        let values: Vec<cl_int> = keys.iter().map(|&x| x + 10).collect();
        let priorities: Vec<cl_int> = keys.iter().map(|&x| x + 1).collect();

        cache.put(&keys, &values, &priorities).unwrap();
        // assert!(r.is_ok());

        let cs = cache.print().unwrap();

        assert_eq!(cs.keys, keys);
        assert_eq!(cs.values, values);
        assert_eq!(cs.priorities, priorities);
    }
}

#[cfg(test)]
mod tests_mini_lru_cache_get {
    use super::*;
    use crate::cache::config::CacheSrc;
    use crate::config::{ClTypeDefault, DEFAULT_DEVICE_INDEX};
    use opencl::wrapper::system::System;

    #[test]
    fn cache_is_empty() {
        let cache_capacity = 32;

        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_mini_lru_by_id(0).unwrap();
        let cache = MiniLRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let keys: Vec<cl_int> = (0..cache_capacity as cl_int).collect();

        let (values, indices) = cache.get(&keys).unwrap();

        assert_eq!(values, vec![-1; cache_capacity]);
        assert_eq!(indices, vec![-1; cache_capacity]);

        let cs = cache.print().unwrap();
        assert_eq!(cs.priorities, vec![0; cache_capacity]);
    }

    #[test]
    fn cache_is_full() {
        let cache_capacity = 32;

        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_mini_lru_by_id(0).unwrap();
        let cache = MiniLRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let input_keys: Vec<cl_int> = (0..cache_capacity as cl_int).collect();
        let input_values: Vec<cl_int> = input_keys.iter().map(|x| x + 10).collect();

        let _ = cache.insert(&input_keys, &input_values).unwrap();
        let (values, indices) = cache.get(&input_keys).unwrap();

        let expected_values = input_values.clone();
        assert_eq!(values, expected_values);

        // FIXME update indices assert
        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        let expected_indices = input_keys.clone();
        assert_eq!(indices_sorted, expected_indices);

        let mut cs = cache.print().unwrap();
        cs.sort();

        let expected_priorities: Vec<cl_int> = input_keys
            .iter()
            .map(|x| x + 1 + cache_capacity as cl_int)
            .collect();
        assert_eq!(cs.priorities, expected_priorities);
    }

    #[test]
    fn cache_is_full_2() {
        let cache_capacity = 1024;

        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_mini_lru_by_id(0).unwrap();
        let cache = MiniLRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let input_keys: Vec<cl_int> = (0..cache_capacity as cl_int).collect();
        let input_values: Vec<cl_int> = input_keys.iter().map(|x| x + 10).collect();

        let _ = cache.insert(&input_keys, &input_values).unwrap();
        let (values, indices) = cache.get(&input_keys).unwrap();

        let expected_values = input_values.clone();
        assert_eq!(values, expected_values);

        // FIXME update indices assert
        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        let expected_indices = input_keys.clone();
        assert_eq!(indices_sorted, expected_indices);

        let mut cs = cache.print().unwrap();
        cs.sort();

        let expected_priorities: Vec<cl_int> = input_keys
            .iter()
            .map(|x| x + 1 + cache_capacity as cl_int)
            .collect();
        assert_eq!(cs.priorities, expected_priorities);
    }

    #[test]
    fn partially_full_cache() {
        let cache_capacity = 32;

        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_mini_lru_by_id(0).unwrap();
        let cache = MiniLRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let base_input_keys: Vec<cl_int> = (0..cache_capacity as cl_int).collect();

        let input_keys = base_input_keys[0..(cache_capacity / 2)].to_vec();
        let input_values: Vec<cl_int> = input_keys.iter().map(|x| x + 10).collect();

        let _ = cache.insert(&input_keys, &input_values).unwrap();
        let (values, indices) = cache.get(&base_input_keys).unwrap();

        let mut expected_values = input_values.clone();
        expected_values.append(&mut vec![i32::cl_default(); cache_capacity / 2]);

        assert_eq!(values, expected_values);

        // FIXME update indices assert
        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        let mut expected_indices = vec![i32::cl_default(); cache_capacity / 2];
        expected_indices.append(&mut input_keys.clone());
        assert_eq!(indices_sorted, expected_indices);

        let mut cs = cache.print().unwrap();
        cs.sort();

        let mut expected_priorities: Vec<cl_int> = vec![0; cache_capacity / 2];
        expected_priorities.append(
            &mut input_keys
                .iter()
                .map(|x| x + 1 + (cache_capacity / 2) as cl_int)
                .collect(),
        );
        assert_eq!(cs.priorities, expected_priorities);
    }
}

#[cfg(test)]
mod tests_mini_lru_cache_get_keys {
    use super::*;
    use crate::cache::config::CacheSrc;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use opencl::wrapper::system::System;

    #[test]
    fn cache_is_empty() {
        let cache_capacity = 32;

        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_mini_lru_by_id(0).unwrap();
        let cache = MiniLRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let keys = cache.keys().unwrap();

        assert!(keys.is_empty());
        cache.print().unwrap();
    }

    #[test]
    fn cache_is_full_1() {
        let cache_capacity = 32;

        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_mini_lru_by_id(0).unwrap();
        let cache = MiniLRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let input_keys: Vec<cl_int> = (0..cache_capacity as cl_int).collect();
        let input_values: Vec<cl_int> = input_keys.iter().map(|&x| x + 10).collect();
        let input_priorities: Vec<cl_int> = input_keys.iter().map(|&x| x + 1).collect();

        cache
            .put(&input_keys, &input_values, &input_priorities)
            .unwrap();

        let keys = cache.keys().unwrap();

        for key in keys.iter() {
            println!("{key:?}");
        }

        let keys: Vec<_> = keys.iter().map(|x| x.key).collect();
        assert_eq!(keys, input_keys);

        println!("sorted");

        let keys = cache.sorted_keys().unwrap();

        for key in keys.iter() {
            println!("{key:?}");
        }

        let keys: Vec<_> = keys.iter().map(|x| x.key).collect();
        let mut expected_keys: Vec<_> = input_keys.clone();
        expected_keys.reverse();
        assert_eq!(keys, expected_keys);

        cache.print().unwrap();
    }

    #[test]
    fn cache_is_full_1_large() {
        let cache_capacity = 1024;

        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_mini_lru_by_id(0).unwrap();
        let cache = MiniLRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let input_keys: Vec<cl_int> = (0..cache_capacity as cl_int).collect();
        let input_values: Vec<cl_int> = input_keys.iter().map(|&x| x + 10).collect();
        let input_priorities: Vec<cl_int> = input_keys.iter().map(|&x| x + 1).collect();

        cache
            .put(&input_keys, &input_values, &input_priorities)
            .unwrap();

        let keys = cache.keys().unwrap();

        for key in keys.iter() {
            println!("{key:?}");
        }

        let keys: Vec<_> = keys.iter().map(|x| x.key).collect();
        assert_eq!(keys, input_keys);

        println!("sorted");

        let keys = cache.sorted_keys().unwrap();

        for key in keys.iter() {
            println!("{key:?}");
        }

        let keys: Vec<_> = keys.iter().map(|x| x.key).collect();
        let mut expected_keys: Vec<_> = input_keys.clone();
        expected_keys.reverse();
        assert_eq!(keys, expected_keys);

        cache.print().unwrap();
    }

    #[test]
    fn cache_is_full_2() {
        let cache_capacity = 32;

        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_mini_lru_by_id(0).unwrap();
        let cache = MiniLRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let input_keys: Vec<cl_int> = (0..cache_capacity as cl_int).collect();
        let input_values: Vec<cl_int> = input_keys.iter().map(|&x| x + 10).collect();
        let input_priorities: Vec<cl_int> = input_keys.iter().map(|&x| x + 1).collect();

        cache
            .put(&input_keys, &input_values, &input_priorities)
            .unwrap();
        let _ = cache.get(&input_keys[0..(cache_capacity / 2)]);

        let keys = cache.keys().unwrap();

        for key in keys.iter() {
            println!("{key:?}");
        }

        let keys: Vec<_> = keys.iter().map(|x| x.key).collect();
        assert_eq!(keys, input_keys);

        println!("sorted");

        let keys = cache.sorted_keys().unwrap();

        for key in keys.iter() {
            println!("{key:?}");
        }

        let keys: Vec<_> = keys.iter().map(|x| x.key).collect();

        let mut k: Vec<_> = input_keys.clone();
        k.reverse();
        let mut expected_keys = k[(cache_capacity / 2)..].to_vec();
        expected_keys.append(&mut k[0..(cache_capacity / 2)].to_vec());

        assert_eq!(keys, expected_keys);

        cache.print().unwrap();
    }

    #[test]
    fn cache_is_full_2_large() {
        let cache_capacity = 1024;

        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_mini_lru_by_id(0).unwrap();
        let cache = MiniLRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let input_keys: Vec<cl_int> = (0..cache_capacity as cl_int).collect();
        let input_values: Vec<cl_int> = input_keys.iter().map(|&x| x + 10).collect();
        let input_priorities: Vec<cl_int> = input_keys.iter().map(|&x| x + 1).collect();

        cache
            .put(&input_keys, &input_values, &input_priorities)
            .unwrap();
        let _ = cache.get(&input_keys[0..(cache_capacity / 2)]);

        let keys = cache.keys().unwrap();

        for key in keys.iter() {
            println!("{key:?}");
        }

        let keys: Vec<_> = keys.iter().map(|x| x.key).collect();
        assert_eq!(keys, input_keys);

        println!("sorted");

        let keys = cache.sorted_keys().unwrap();

        for key in keys.iter() {
            println!("{key:?}");
        }

        let keys: Vec<_> = keys.iter().map(|x| x.key).collect();

        let mut k: Vec<_> = input_keys.clone();
        k.reverse();
        let mut expected_keys = k[(cache_capacity / 2)..].to_vec();
        expected_keys.append(&mut k[0..(cache_capacity / 2)].to_vec());

        assert_eq!(keys, expected_keys);

        cache.print().unwrap();
    }
}

#[cfg(test)]
mod tests_mini_lru_cache_get_sorted_keys {
    use super::*;
    use crate::cache::config::CacheSrc;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use opencl::wrapper::system::System;

    #[test]
    fn cache_is_empty() {
        let cache_capacity = 32;

        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_mini_lru_by_id(0).unwrap();
        let cache = MiniLRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let keys = cache.get_sorted_keys().unwrap();

        assert!(keys.is_empty());
        cache.print().unwrap();
    }

    #[test]
    fn cache_is_full_1() {
        let cache_capacity = 32;

        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_mini_lru_by_id(0).unwrap();
        let cache = MiniLRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let input_keys: Vec<cl_int> = (0..cache_capacity as cl_int).collect();
        let input_values: Vec<cl_int> = input_keys.iter().map(|&x| x + 10).collect();
        let input_priorities: Vec<cl_int> = input_keys.iter().map(|&x| x + 1).collect();

        cache
            .put(&input_keys, &input_values, &input_priorities)
            .unwrap();

        let keys = cache.get_sorted_keys().unwrap();

        for key in keys.iter() {
            println!("{key:?}");
        }

        let keys: Vec<_> = keys.iter().map(|x| x.key).collect();
        let mut expected_keys: Vec<_> = input_keys.clone();
        expected_keys.reverse();
        assert_eq!(keys, expected_keys);

        cache.print().unwrap();
    }

    #[test]
    fn cache_is_full_1_large() {
        let cache_capacity = 1024;

        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_mini_lru_by_id(0).unwrap();
        let cache = MiniLRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let input_keys: Vec<cl_int> = (0..cache_capacity as cl_int).collect();
        let input_values: Vec<cl_int> = input_keys.iter().map(|&x| x + 10).collect();
        let input_priorities: Vec<cl_int> = input_keys.iter().map(|&x| x + 1).collect();

        cache
            .put(&input_keys, &input_values, &input_priorities)
            .unwrap();

        let keys = cache.get_sorted_keys().unwrap();

        for key in keys.iter() {
            println!("{key:?}");
        }

        let keys: Vec<_> = keys.iter().map(|x| x.key).collect();
        let mut expected_keys: Vec<_> = input_keys.clone();
        expected_keys.reverse();
        assert_eq!(keys, expected_keys);

        cache.print().unwrap();
    }

    #[test]
    fn cache_is_full_2() {
        let cache_capacity = 32;

        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_mini_lru_by_id(0).unwrap();
        let cache = MiniLRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let input_keys: Vec<cl_int> = (0..cache_capacity as cl_int).collect();
        let input_values: Vec<cl_int> = input_keys.iter().map(|&x| x + 10).collect();
        let input_priorities: Vec<cl_int> = input_keys.iter().map(|&x| x + 1).collect();

        cache
            .put(&input_keys, &input_values, &input_priorities)
            .unwrap();
        let _ = cache.get(&input_keys[0..(cache_capacity / 2)]);

        let keys = cache.get_sorted_keys().unwrap();

        for key in keys.iter() {
            println!("{key:?}");
        }

        let keys: Vec<_> = keys.iter().map(|x| x.key).collect();

        let mut k: Vec<_> = input_keys.clone();
        k.reverse();
        let mut expected_keys = k[(cache_capacity / 2)..].to_vec();
        expected_keys.append(&mut k[0..(cache_capacity / 2)].to_vec());

        assert_eq!(keys, expected_keys);

        cache.print().unwrap();
    }

    #[test]
    fn cache_is_full_2_large() {
        let cache_capacity = 1024;

        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_mini_lru_by_id(0).unwrap();
        let cache = MiniLRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let input_keys: Vec<cl_int> = (0..cache_capacity as cl_int).collect();
        let input_values: Vec<cl_int> = input_keys.iter().map(|&x| x + 10).collect();
        let input_priorities: Vec<cl_int> = input_keys.iter().map(|&x| x + 1).collect();

        cache
            .put(&input_keys, &input_values, &input_priorities)
            .unwrap();
        let _ = cache.get(&input_keys[0..(cache_capacity / 2)]);

        let keys = cache.get_sorted_keys().unwrap();

        for key in keys.iter() {
            println!("{key:?}");
        }

        let keys: Vec<_> = keys.iter().map(|x| x.key).collect();

        let mut k: Vec<_> = input_keys.clone();
        k.reverse();
        let mut expected_keys = k[(cache_capacity / 2)..].to_vec();
        expected_keys.append(&mut k[0..(cache_capacity / 2)].to_vec());

        assert_eq!(keys, expected_keys);

        cache.print().unwrap();
    }
}

#[cfg(test)]
mod tests_mini_lru_cache_sort {
    use super::*;
    use crate::cache::config::CacheSrc;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use opencl::wrapper::system::System;

    #[test]
    fn cache_is_empty() {
        let cache_capacity = 32;

        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_mini_lru_by_id(0).unwrap();
        let cache = MiniLRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let r = cache.sort();
        assert_eq!(r, Ok(()));

        let cs = cache.print().unwrap();
        assert_eq!(cs, MiniLRUCacheSnapshot::create_empty(cache_capacity));
    }

    #[test]
    fn cache_is_full() {
        let cache_capacity = 1024;

        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_mini_lru_by_id(0).unwrap();
        let cache = MiniLRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let keys: Vec<cl_int> = (0..cache_capacity as cl_int).collect();
        let values: Vec<cl_int> = keys.iter().map(|&x| x + 10).collect();

        let _ = cache.insert(&keys, &values).unwrap();

        let _cs = cache.print().unwrap();

        let r = cache.sort();
        assert_eq!(r, Ok(()));

        let cs = cache.print().unwrap();

        let mut expected_priorities: Vec<_> = keys.clone().iter().map(|x| x + 1).collect();
        expected_priorities.reverse();

        // assert_eq!(cs.keys, keys);
        // assert_eq!(cs.values, values);
        assert_eq!(cs.priorities, expected_priorities);
    }
}

// TODO explain tests
#[cfg(test)]
mod tests_mini_lru_cache_examples {
    use super::*;
    use crate::cache::config::CacheSrc;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::utils::has_unique_elements;
    use opencl::wrapper::system::System;

    #[test]
    fn case_1() {
        let cache_capacity = 8;

        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_mini_lru_by_id(0).unwrap();
        let cache = MiniLRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        println!("put 1");
        let keys: Vec<cl_int> = (0..cache_capacity as cl_int).collect();
        let values: Vec<cl_int> = keys.iter().map(|&x| x + 10).collect();

        let indices = cache.add(&keys, &values).unwrap();
        let _qs = cache.print().unwrap();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        let expected: Vec<cl_int> = (0..cache_capacity as cl_int).collect();
        assert_eq!(indices_sorted, expected);

        println!("put 2");
        let keys_2 = vec![10, 20];
        let values_2 = vec![100, 200];

        let indices = cache.add(&keys_2, &values_2).unwrap();
        let _qs = cache.print().unwrap();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        assert_eq!(indices_sorted, vec![0, 1]);

        println!("put 3");
        let keys_3 = vec![100, 200];
        let values_3 = vec![11, 22];

        let indices = cache.add(&keys_3, &values_3).unwrap();
        let _qs = cache.print().unwrap();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        assert_eq!(indices_sorted, vec![2, 3]);

        println!("put 4");
        let keys_4 = vec![100, 200, 100, 200];
        let values_4 = vec![11, 22, 111, 222];

        let indices = cache.add(&keys_4, &values_4).unwrap();
        let mut qs = cache.print().unwrap();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        assert_eq!(indices_sorted, vec![2, 2, 3, 3]);

        println!("put 5");
        let keys_5 = qs.get_lower_priority_keys(4);
        let values_5 = vec![50; 4];

        let indices = cache.add(&keys_5, &values_5).unwrap();
        let _cs = cache.print().unwrap();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();

        assert_eq!(indices_sorted, vec![4, 5, 6, 7]);
    }

    #[test]
    fn case_2() {
        let cache_capacity = 16;

        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(cache_capacity);

        let system = Arc::new(System::new(DEFAULT_DEVICE_INDEX, &cache_src.build()).unwrap());

        let config = cache_src.get_mini_lru_by_id(0).unwrap();
        let cache = MiniLRUCacheHandle::new(config, system);
        cache.initialize().unwrap();

        let input_keys: Vec<cl_int> = (0..cache_capacity as cl_int).collect();
        let input_values: Vec<_> = input_keys.iter().map(|x| x + 10).collect();

        let _ = cache.insert(&input_keys, &input_values).unwrap();

        println!("get 1");
        let (values, _indices) = cache.get(&[1]).unwrap();
        assert_eq!(values, vec![11]);

        println!("get 2");
        let (values, _indices) = cache.get(&[3, 4, 5, 6]).unwrap();
        assert_eq!(values, vec![13, 14, 15, 16]);

        println!("get 3");
        let (values, _indices) = cache.get(&[1, 3, 300, 400]).unwrap();
        assert_eq!(values, vec![11, 13, -1, -1]);

        println!("get 4");
        let (values, _indices) = cache.get(&[1; 6]).unwrap();
        assert_eq!(values, vec![11; 6]);

        let cs = cache.print().unwrap();
        assert!(has_unique_elements(&cs.priorities));
    }
}
