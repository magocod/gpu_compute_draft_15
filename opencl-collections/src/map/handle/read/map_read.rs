use crate::config::{ClTypeTrait, DEBUG_MODE};
use crate::error::OpenClResult;
use crate::map::config::{check_local_work_size, MapConfig, DEFAULT_MAP_KEY_LENGTH};
use crate::map::handle::{MapHandle, MapKeys, Pair};
use crate::map::kernel::name::{
    get_map_kernel_name, MAP_READ, MAP_READ_KEYS, MAP_READ_KEYS_FOR_BLOCK, MAP_READ_WITH_CMQ,
    MAP_READ_WITH_INDEX, MAP_READ_WITH_INDEX_AND_CMQ,
};
use crate::utils::{from_buf_usize_to_vec_i32, KB};
use opencl::opencl_sys::bindings::cl_uint;
use opencl::wrapper::system::OpenclCommonOperation;
use std::iter::zip;

#[derive(Debug, Clone)]
pub struct MapBlockKeys<T: ClTypeTrait> {
    pub config: MapConfig<T>,
    pub keys: Vec<Vec<T>>,
}

impl<T: ClTypeTrait, D: OpenclCommonOperation> MapHandle<T, D> {
    pub fn read_keys_for_block(&self, map_value_len: usize) -> OpenClResult<MapKeys<T>> {
        let config = self.map_src.get_config_by_value_len(map_value_len)?;

        let global_work_size = config.capacity;
        let local_work_size = check_local_work_size(global_work_size);

        let key_output_capacity = DEFAULT_MAP_KEY_LENGTH * global_work_size;

        let keys_output_buf = self.system.create_output_buffer(key_output_capacity)?;

        let kernel_name = get_map_kernel_name(MAP_READ_KEYS_FOR_BLOCK, map_value_len);
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        let map_id = self.map_id as cl_uint;

        unsafe {
            kernel.set_arg(&map_id)?;
            kernel.set_arg(&keys_output_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        let keys_output: Vec<T> =
            self.system
                .blocking_enqueue_read_buffer(key_output_capacity, &keys_output_buf, &[])?;

        let keys: Vec<Vec<_>> = keys_output
            .chunks(DEFAULT_MAP_KEY_LENGTH)
            .map(|x| x.to_vec())
            .collect();

        if DEBUG_MODE {
            for (i, k) in keys.iter().enumerate() {
                println!("key i: {i} {k:?}");
            }
        }

        Ok(keys)
    }

    pub fn read_keys(&self) -> OpenClResult<Vec<MapBlockKeys<T>>> {
        let map_src = &self.map_src;
        let map_blocks = map_src.get_configs();

        let global_work_size = 1;
        let local_work_size = 1;

        let enqueue_kernel_output_capacity = map_blocks.len();

        let key_output_capacity = map_blocks
            .iter()
            .map(|x| DEFAULT_MAP_KEY_LENGTH * x.capacity)
            .sum();

        let keys_output_buf = self.system.create_output_buffer(key_output_capacity)?;
        let enqueue_kernel_output_buf = self
            .system
            .create_output_buffer(enqueue_kernel_output_capacity)?;

        let q0 = self
            .system
            .get_device_command_queue_0()
            .get_cl_command_queue();

        let map_id = self.map_id as cl_uint;

        let mut kernel = self.system.create_kernel(MAP_READ_KEYS)?;

        unsafe {
            kernel.set_arg(&q0)?;
            kernel.set_arg(&map_id)?;
            kernel.set_arg(&keys_output_buf.get_cl_mem())?;
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
                .blocking_enqueue_read_buffer(key_output_capacity, &keys_output_buf, &[])?;

        let mut output: Vec<MapBlockKeys<_>> = Vec::new();

        for (i, config) in map_blocks.iter().enumerate() {
            let block_output_index: usize = map_blocks[0..i]
                .iter()
                .map(|x| x.capacity * DEFAULT_MAP_KEY_LENGTH)
                .sum();

            let keys: Vec<Vec<_>> = keys_output[block_output_index
                ..(block_output_index + (config.capacity * DEFAULT_MAP_KEY_LENGTH))]
                .chunks(DEFAULT_MAP_KEY_LENGTH)
                .map(|x| x.to_vec())
                .collect();

            if DEBUG_MODE && config.capacity < 1024 {
                println!("block: {}, len {}", config.name, keys.len());
                for (sub_i, key) in keys.iter().enumerate() {
                    println!("{} key i: {sub_i}, {key:?}", config.name);
                }
            }

            output.push(MapBlockKeys {
                config: config.clone(),
                keys,
            });
        }

        self.system.assert_device_enqueue_kernel(
            enqueue_kernel_output_capacity,
            enqueue_kernel_output_buf,
            &[],
        )?;

        Ok(output)
    }

    pub fn read(&self, map_value_len: usize) -> OpenClResult<Vec<Pair<T>>> {
        let config = self.map_src.get_config_by_value_len(map_value_len)?;

        let global_work_size = config.capacity;
        let local_work_size = check_local_work_size(global_work_size);

        let key_output_capacity = DEFAULT_MAP_KEY_LENGTH * global_work_size;
        let value_output_capacity = map_value_len * global_work_size;

        let keys_output_buf = self.system.create_output_buffer(key_output_capacity)?;

        let values_output_buf = self.system.create_output_buffer(value_output_capacity)?;

        let map_id = self.map_id as cl_uint;

        let kernel_name = get_map_kernel_name(MAP_READ, map_value_len);
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&map_id)?;
            kernel.set_arg(&keys_output_buf.get_cl_mem())?;
            kernel.set_arg(&values_output_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        let keys_output: Vec<T> =
            self.system
                .blocking_enqueue_read_buffer(key_output_capacity, &keys_output_buf, &[])?;

        let values_output: Vec<T> = self.system.blocking_enqueue_read_buffer(
            value_output_capacity,
            &values_output_buf,
            &[],
        )?;

        // TODO improve result assignment

        let mut pairs: Vec<Pair<T>> = Vec::with_capacity(global_work_size);

        let keys = keys_output.chunks(DEFAULT_MAP_KEY_LENGTH);
        let values = values_output.chunks(map_value_len);

        let iter = zip(keys, values);

        for (i, (key, value)) in iter.enumerate() {
            if DEBUG_MODE && map_value_len < KB {
                println!("i{i} key   {key:?}");
                println!("i{i} value {value:?}");
            }

            pairs.push(Pair::new(key.to_vec(), value.to_vec()));
        }

        Ok(pairs)
    }

    // For now, only searches for a few simultaneous elements
    pub fn read_with_cmq(&self, map_value_len: usize) -> OpenClResult<Vec<Pair<T>>> {
        let config = self.map_src.get_config_by_value_len(map_value_len)?;

        let global_work_size = config.capacity;
        let local_work_size = check_local_work_size(global_work_size);

        let key_output_capacity = DEFAULT_MAP_KEY_LENGTH * global_work_size;
        let value_output_capacity = map_value_len * global_work_size;
        let enqueue_kernel_output_capacity = global_work_size;

        let keys_output_buf = self.system.create_output_buffer(key_output_capacity)?;

        let values_output_buf = self.system.create_output_buffer(value_output_capacity)?;

        let enqueue_kernel_output_buf = self
            .system
            .create_output_buffer(enqueue_kernel_output_capacity)?;

        let q0 = self
            .system
            .get_device_command_queue_0()
            .get_cl_command_queue();

        let map_id = self.map_id as cl_uint;

        let kernel_name = get_map_kernel_name(MAP_READ_WITH_CMQ, map_value_len);
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&q0)?;
            kernel.set_arg(&map_id)?;
            kernel.set_arg(&keys_output_buf.get_cl_mem())?;
            kernel.set_arg(&values_output_buf.get_cl_mem())?;
            kernel.set_arg(&enqueue_kernel_output_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        let keys_output: Vec<T> =
            self.system
                .blocking_enqueue_read_buffer(key_output_capacity, &keys_output_buf, &[])?;

        let values_output: Vec<T> = self.system.blocking_enqueue_read_buffer(
            value_output_capacity,
            &values_output_buf,
            &[],
        )?;

        // TODO improve result assignment

        let mut pairs: Vec<Pair<T>> = Vec::with_capacity(global_work_size);

        let keys = keys_output.chunks(DEFAULT_MAP_KEY_LENGTH);
        let values = values_output.chunks(map_value_len);

        let iter = zip(keys, values);

        for (i, (key, value)) in iter.enumerate() {
            if DEBUG_MODE && map_value_len < KB {
                println!("i: {i}, key   {key:?}");
                println!("i: {i}, value {value:?}");
            }

            pairs.push(Pair::new(key.to_vec(), value.to_vec()));
        }

        self.system.assert_device_enqueue_kernel(
            enqueue_kernel_output_capacity,
            enqueue_kernel_output_buf,
            &[],
        )?;

        Ok(pairs)
    }

    pub fn read_with_index(
        &self,
        map_value_len: usize,
        indices: &[usize],
    ) -> OpenClResult<Vec<Pair<T>>> {
        let _ = self.map_src.get_config_by_value_len(map_value_len)?;

        let global_work_size = indices.len();
        let local_work_size = check_local_work_size(global_work_size);

        let key_output_capacity = DEFAULT_MAP_KEY_LENGTH * global_work_size;
        let value_output_capacity = map_value_len * global_work_size;

        let keys_output_buf = self.system.create_output_buffer(key_output_capacity)?;

        let values_output_buf = self.system.create_output_buffer(value_output_capacity)?;

        let indices_input = from_buf_usize_to_vec_i32(indices);

        let indices_buf = self.system.blocking_prepare_input_buffer(&indices_input)?;

        let map_id = self.map_id as cl_uint;

        let kernel_name = get_map_kernel_name(MAP_READ_WITH_INDEX, map_value_len);
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&map_id)?;
            kernel.set_arg(&keys_output_buf.get_cl_mem())?;
            kernel.set_arg(&values_output_buf.get_cl_mem())?;
            kernel.set_arg(&indices_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        let keys_output: Vec<T> =
            self.system
                .blocking_enqueue_read_buffer(key_output_capacity, &keys_output_buf, &[])?;

        let values_output: Vec<T> = self.system.blocking_enqueue_read_buffer(
            value_output_capacity,
            &values_output_buf,
            &[],
        )?;

        // TODO improve result assignment

        let mut pairs: Vec<Pair<T>> = Vec::with_capacity(global_work_size);

        let keys = keys_output.chunks(DEFAULT_MAP_KEY_LENGTH);
        let values = values_output.chunks(map_value_len);

        let iter = zip(keys, values);

        for (i, (key, value)) in iter.enumerate() {
            if DEBUG_MODE && map_value_len < KB {
                println!("i: {i}, key   {key:?}");
                println!("i: {i}, value {value:?}");
            }

            pairs.push(Pair::new(key.to_vec(), value.to_vec()));
        }

        Ok(pairs)
    }

    // For now, only searches for a few simultaneous elements
    pub fn read_with_index_and_cmq(
        &self,
        map_value_len: usize,
        indices: &[usize],
    ) -> OpenClResult<Vec<Pair<T>>> {
        let _ = self.map_src.get_config_by_value_len(map_value_len)?;

        let global_work_size = indices.len();
        let local_work_size = check_local_work_size(global_work_size);

        let key_output_capacity = DEFAULT_MAP_KEY_LENGTH * global_work_size;
        let value_output_capacity = map_value_len * global_work_size;
        let enqueue_kernel_output_capacity = global_work_size;

        let keys_output_buf = self.system.create_output_buffer(key_output_capacity)?;

        let values_output_buf = self.system.create_output_buffer(value_output_capacity)?;

        let indices_input = from_buf_usize_to_vec_i32(indices);

        let indices_buf = self.system.blocking_prepare_input_buffer(&indices_input)?;

        let enqueue_kernel_output_buf = self
            .system
            .create_output_buffer(enqueue_kernel_output_capacity)?;

        let q0 = self
            .system
            .get_device_command_queue_0()
            .get_cl_command_queue();

        let map_id = self.map_id as cl_uint;

        let kernel_name = get_map_kernel_name(MAP_READ_WITH_INDEX_AND_CMQ, map_value_len);
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&q0)?;
            kernel.set_arg(&map_id)?;
            kernel.set_arg(&keys_output_buf.get_cl_mem())?;
            kernel.set_arg(&values_output_buf.get_cl_mem())?;
            kernel.set_arg(&indices_buf.get_cl_mem())?;
            kernel.set_arg(&enqueue_kernel_output_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        let keys_output: Vec<T> =
            self.system
                .blocking_enqueue_read_buffer(key_output_capacity, &keys_output_buf, &[])?;

        let values_output: Vec<T> = self.system.blocking_enqueue_read_buffer(
            value_output_capacity,
            &values_output_buf,
            &[],
        )?;

        // TODO improve result assignment

        let mut pairs: Vec<Pair<T>> = Vec::with_capacity(global_work_size);

        let keys = keys_output.chunks(DEFAULT_MAP_KEY_LENGTH);
        let values = values_output.chunks(map_value_len);

        let iter = zip(keys, values);

        for (i, (key, value)) in iter.enumerate() {
            if DEBUG_MODE && map_value_len < KB {
                println!("i: {i}, key   {key:?}");
                println!("i: {i}, value {value:?}");
            }

            pairs.push(Pair::new(key.to_vec(), value.to_vec()));
        }

        self.system.assert_device_enqueue_kernel(
            enqueue_kernel_output_capacity,
            enqueue_kernel_output_buf,
            &[],
        )?;

        Ok(pairs)
    }
}

#[cfg(test)]
mod tests_map_read_keys_for_block {
    use crate::config::ClTypeDefault;
    use crate::map::config::{MapSrc, DEFAULT_MAP_KEY_LENGTH};
    use crate::map::handle::test_utils::generate_arc_opencl_block_default;
    use crate::map::handle::MapHandle;
    use crate::test_utils::TestMatrix;
    use crate::utils::{ensure_vec_size, BYTE_256};

    const TOTAL_MAPS: usize = 2;
    const MAP_CAPACITY: usize = 32;
    const MAP_VALUE_LEN: usize = BYTE_256;

    #[test]
    fn map_is_empty() {
        let mut map_src: MapSrc<i16> = MapSrc::new(TOTAL_MAPS);
        map_src.add(MAP_VALUE_LEN, MAP_CAPACITY);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let keys = m.read_keys_for_block(MAP_VALUE_LEN).unwrap();

        assert_eq!(
            keys,
            vec![vec![i16::cl_default(); DEFAULT_MAP_KEY_LENGTH]; MAP_CAPACITY]
        );
    }

    #[test]
    fn map_is_full() {
        let mut map_src: MapSrc<i16> = MapSrc::new(TOTAL_MAPS);
        map_src.add(MAP_VALUE_LEN, MAP_CAPACITY);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix =
            TestMatrix::new(MAP_CAPACITY, DEFAULT_MAP_KEY_LENGTH, MAP_VALUE_LEN, 10, 100);
        test_matrix.put(&m, MAP_VALUE_LEN);

        let keys = m.read_keys_for_block(MAP_VALUE_LEN).unwrap();

        assert_eq!(keys, test_matrix.keys);
    }

    #[test]
    fn map_is_not_full() {
        let mut map_src: MapSrc<i16> = MapSrc::new(TOTAL_MAPS);
        map_src.add(MAP_VALUE_LEN, MAP_CAPACITY);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix = TestMatrix::new(
            MAP_CAPACITY / 2,
            DEFAULT_MAP_KEY_LENGTH / 2,
            MAP_VALUE_LEN / 2,
            40,
            400,
        );
        test_matrix.put(&m, MAP_VALUE_LEN);

        let keys = m.read_keys_for_block(MAP_VALUE_LEN).unwrap();

        for (i, key) in keys[0..(MAP_CAPACITY / 2)].iter().enumerate() {
            assert_eq!(
                key,
                &ensure_vec_size(&test_matrix.keys[i], DEFAULT_MAP_KEY_LENGTH)
            );
        }

        assert_eq!(
            keys[(MAP_CAPACITY / 2)..MAP_CAPACITY],
            vec![vec![i16::cl_default(); DEFAULT_MAP_KEY_LENGTH]; MAP_CAPACITY / 2]
        );
    }
}

#[cfg(test)]
mod tests_map_read {
    use crate::config::ClTypeDefault;
    use crate::map::config::{MapSrc, DEFAULT_MAP_KEY_LENGTH};
    use crate::map::handle::test_utils::generate_arc_opencl_block_default;
    use crate::map::handle::MapHandle;
    use crate::test_utils::TestMatrix;
    use crate::utils::{ensure_vec_size, BYTE_256};

    const TOTAL_MAPS: usize = 2;
    const MAP_CAPACITY: usize = 32;
    const MAP_VALUE_LEN: usize = BYTE_256;

    #[test]
    fn map_is_empty() {
        let mut map_src: MapSrc<i16> = MapSrc::new(TOTAL_MAPS);
        map_src.add(MAP_VALUE_LEN, MAP_CAPACITY);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let pairs = m.read(MAP_VALUE_LEN).unwrap();

        for pair in pairs {
            assert_eq!(pair.key, vec![i16::cl_default(); DEFAULT_MAP_KEY_LENGTH]);
            assert_eq!(pair.value, vec![i16::cl_default(); MAP_VALUE_LEN]);
        }
    }

    #[test]
    fn map_is_full() {
        let mut map_src: MapSrc<i16> = MapSrc::new(TOTAL_MAPS);
        map_src.add(MAP_VALUE_LEN, MAP_CAPACITY);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix =
            TestMatrix::new(MAP_CAPACITY, DEFAULT_MAP_KEY_LENGTH, MAP_VALUE_LEN, 10, 100);
        test_matrix.put(&m, MAP_VALUE_LEN);

        let pairs = m.read(MAP_VALUE_LEN).unwrap();

        for (i, pair) in pairs.into_iter().enumerate() {
            assert_eq!(pair.key, test_matrix.keys[i]);
            assert_eq!(pair.value, test_matrix.values[i]);
        }
    }

    #[test]
    fn map_is_not_full() {
        let mut map_src = MapSrc::new(TOTAL_MAPS);
        map_src.add(MAP_VALUE_LEN, MAP_CAPACITY);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix = TestMatrix::new(
            MAP_CAPACITY / 2,
            DEFAULT_MAP_KEY_LENGTH / 2,
            MAP_VALUE_LEN / 2,
            40,
            400,
        );
        test_matrix.put(&m, MAP_VALUE_LEN);

        let pairs = m.read(MAP_VALUE_LEN).unwrap();

        assert_eq!(pairs.len(), MAP_CAPACITY);

        for (i, pair) in pairs[0..(MAP_CAPACITY / 2)].iter().enumerate() {
            assert_eq!(
                pair.key,
                ensure_vec_size(&test_matrix.keys[i], MAP_VALUE_LEN)
            );
            assert_eq!(
                pair.value,
                ensure_vec_size(&test_matrix.values[i], MAP_VALUE_LEN)
            );
        }

        for pair in pairs[(MAP_CAPACITY / 2)..].iter() {
            assert_eq!(pair.key, vec![i16::cl_default(); DEFAULT_MAP_KEY_LENGTH]);
            assert_eq!(pair.value, vec![i16::cl_default(); MAP_VALUE_LEN]);
        }
    }
}

#[cfg(test)]
mod tests_map_read_with_cmq {
    use crate::config::ClTypeDefault;
    use crate::map::config::{MapSrc, DEFAULT_MAP_KEY_LENGTH};
    use crate::map::handle::test_utils::generate_arc_opencl_block_default;
    use crate::map::handle::MapHandle;
    use crate::test_utils::TestMatrix;
    use crate::utils::{ensure_vec_size, BYTE_256};

    const TOTAL_MAPS: usize = 2;
    const MAP_CAPACITY: usize = 32;
    const MAP_VALUE_LEN: usize = BYTE_256;

    #[test]
    fn map_is_empty() {
        let mut map_src: MapSrc<i16> = MapSrc::new(TOTAL_MAPS);
        map_src.add(MAP_VALUE_LEN, MAP_CAPACITY);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let pairs = m.read_with_cmq(MAP_VALUE_LEN).unwrap();

        for pair in pairs {
            assert_eq!(pair.key, vec![i16::cl_default(); DEFAULT_MAP_KEY_LENGTH]);
            assert_eq!(pair.value, vec![i16::cl_default(); MAP_VALUE_LEN]);
        }
    }

    #[test]
    fn map_is_full() {
        let mut map_src: MapSrc<i16> = MapSrc::new(TOTAL_MAPS);
        map_src.add(MAP_VALUE_LEN, MAP_CAPACITY);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix =
            TestMatrix::new(MAP_CAPACITY, DEFAULT_MAP_KEY_LENGTH, MAP_VALUE_LEN, 10, 100);
        test_matrix.put(&m, MAP_VALUE_LEN);

        let pairs = m.read_with_cmq(MAP_VALUE_LEN).unwrap();

        for (i, pair) in pairs.into_iter().enumerate() {
            assert_eq!(pair.key, test_matrix.keys[i]);
            assert_eq!(pair.value, test_matrix.values[i]);
        }
    }

    #[test]
    fn map_is_not_full() {
        let mut map_src = MapSrc::new(TOTAL_MAPS);
        map_src.add(MAP_VALUE_LEN, MAP_CAPACITY);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix = TestMatrix::new(
            MAP_CAPACITY / 2,
            DEFAULT_MAP_KEY_LENGTH / 2,
            MAP_VALUE_LEN / 2,
            40,
            400,
        );
        test_matrix.put(&m, MAP_VALUE_LEN);

        let pairs = m.read_with_cmq(MAP_VALUE_LEN).unwrap();

        assert_eq!(pairs.len(), MAP_CAPACITY);

        for (i, pair) in pairs[0..(MAP_CAPACITY / 2)].iter().enumerate() {
            assert_eq!(
                pair.key,
                ensure_vec_size(&test_matrix.keys[i], MAP_VALUE_LEN)
            );
            assert_eq!(
                pair.value,
                ensure_vec_size(&test_matrix.values[i], MAP_VALUE_LEN)
            );
        }

        for pair in pairs[(MAP_CAPACITY / 2)..].iter() {
            assert_eq!(pair.key, vec![i16::cl_default(); DEFAULT_MAP_KEY_LENGTH]);
            assert_eq!(pair.value, vec![i16::cl_default(); MAP_VALUE_LEN]);
        }
    }
}

#[cfg(test)]
mod tests_map_read_with_index {
    use crate::config::ClTypeDefault;
    use crate::map::config::{MapSrc, DEFAULT_MAP_KEY_LENGTH};
    use crate::map::handle::test_utils::generate_arc_opencl_block_default;
    use crate::map::handle::MapHandle;
    use crate::test_utils::TestMatrix;
    use crate::utils::BYTE_256;

    const TOTAL_MAPS: usize = 2;
    const MAP_CAPACITY: usize = 32;
    const MAP_VALUE_LEN: usize = BYTE_256;

    #[test]
    fn map_is_empty() {
        let mut map_src: MapSrc<i16> = MapSrc::new(TOTAL_MAPS);
        map_src.add(MAP_VALUE_LEN, MAP_CAPACITY);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let indices: Vec<usize> = (0..MAP_CAPACITY / 2).collect();

        let pairs = m.read_with_index(MAP_VALUE_LEN, &indices).unwrap();

        assert_eq!(pairs.len(), indices.len());

        for pair in pairs {
            assert_eq!(pair.key, vec![i16::cl_default(); DEFAULT_MAP_KEY_LENGTH]);
            assert_eq!(pair.value, vec![i16::cl_default(); MAP_VALUE_LEN]);
        }
    }

    #[test]
    fn all_indices() {
        let mut map_src: MapSrc<i16> = MapSrc::new(TOTAL_MAPS);
        map_src.add(MAP_VALUE_LEN, MAP_CAPACITY);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix =
            TestMatrix::new(MAP_CAPACITY, DEFAULT_MAP_KEY_LENGTH, MAP_VALUE_LEN, 70, 700);
        test_matrix.put(&m, MAP_VALUE_LEN);

        let indices: Vec<usize> = (0..MAP_CAPACITY).collect();

        let pairs = m.read_with_index(MAP_VALUE_LEN, &indices).unwrap();

        assert_eq!(pairs.len(), indices.len());

        for (i, pair) in pairs.into_iter().enumerate() {
            assert_eq!(pair.key, test_matrix.keys[i]);
            assert_eq!(pair.value, test_matrix.values[i]);
        }
    }

    #[test]
    fn some_indices() {
        let mut map_src: MapSrc<i16> = MapSrc::new(TOTAL_MAPS);
        map_src.add(MAP_VALUE_LEN, MAP_CAPACITY);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix =
            TestMatrix::new(MAP_CAPACITY, DEFAULT_MAP_KEY_LENGTH, MAP_VALUE_LEN, 80, 800);
        test_matrix.put(&m, MAP_VALUE_LEN);

        let start_index = MAP_CAPACITY / 2;
        let indices: Vec<usize> = (start_index..MAP_CAPACITY).collect();

        let pairs = m.read_with_index(MAP_VALUE_LEN, &indices).unwrap();

        for (i, m_index) in indices.into_iter().enumerate() {
            assert_eq!(pairs[i].key, test_matrix.keys[m_index]);
            assert_eq!(pairs[i].value, test_matrix.values[m_index]);
        }
    }
}

#[cfg(test)]
mod tests_map_read_with_index_and_cmq {
    use crate::config::ClTypeDefault;
    use crate::map::config::{MapSrc, DEFAULT_MAP_KEY_LENGTH};
    use crate::map::handle::test_utils::generate_arc_opencl_block_default;
    use crate::map::handle::MapHandle;
    use crate::test_utils::TestMatrix;
    use crate::utils::BYTE_256;

    const TOTAL_MAPS: usize = 2;
    const MAP_CAPACITY: usize = 32;
    const MAP_VALUE_LEN: usize = BYTE_256;

    #[test]
    fn map_is_empty() {
        let mut map_src: MapSrc<i16> = MapSrc::new(TOTAL_MAPS);
        map_src.add(MAP_VALUE_LEN, MAP_CAPACITY);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let indices: Vec<usize> = (0..MAP_CAPACITY / 2).collect();

        let pairs = m.read_with_index_and_cmq(MAP_VALUE_LEN, &indices).unwrap();

        assert_eq!(pairs.len(), indices.len());

        for pair in pairs {
            assert_eq!(pair.key, vec![i16::cl_default(); DEFAULT_MAP_KEY_LENGTH]);
            assert_eq!(pair.value, vec![i16::cl_default(); MAP_VALUE_LEN]);
        }
    }

    #[test]
    fn all_indices() {
        let mut map_src: MapSrc<i16> = MapSrc::new(TOTAL_MAPS);
        map_src.add(MAP_VALUE_LEN, MAP_CAPACITY);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix =
            TestMatrix::new(MAP_CAPACITY, DEFAULT_MAP_KEY_LENGTH, MAP_VALUE_LEN, 70, 700);
        test_matrix.put(&m, MAP_VALUE_LEN);

        let indices: Vec<usize> = (0..MAP_CAPACITY).collect();

        let pairs = m.read_with_index_and_cmq(MAP_VALUE_LEN, &indices).unwrap();

        assert_eq!(pairs.len(), indices.len());

        for (i, pair) in pairs.into_iter().enumerate() {
            assert_eq!(pair.key, test_matrix.keys[i]);
            assert_eq!(pair.value, test_matrix.values[i]);
        }
    }

    #[test]
    fn some_indices() {
        let mut map_src: MapSrc<i16> = MapSrc::new(TOTAL_MAPS);
        map_src.add(MAP_VALUE_LEN, MAP_CAPACITY);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix =
            TestMatrix::new(MAP_CAPACITY, DEFAULT_MAP_KEY_LENGTH, MAP_VALUE_LEN, 80, 800);
        test_matrix.put(&m, MAP_VALUE_LEN);

        let start_index = MAP_CAPACITY / 2;
        let indices: Vec<usize> = (start_index..MAP_CAPACITY).collect();

        let pairs = m.read_with_index_and_cmq(MAP_VALUE_LEN, &indices).unwrap();

        for (i, m_index) in indices.into_iter().enumerate() {
            assert_eq!(pairs[i].key, test_matrix.keys[m_index]);
            assert_eq!(pairs[i].value, test_matrix.values[m_index]);
        }
    }
}

#[cfg(test)]
mod tests_map_read_keys {
    use crate::config::ClTypeDefault;
    use crate::map::config::{MapSrc, DEFAULT_MAP_KEY_LENGTH};
    use crate::map::handle::test_utils::generate_arc_opencl_block_default;
    use crate::map::handle::MapHandle;
    use crate::test_utils::TestMatrix;
    use crate::utils::{ensure_vec_size, BYTE_256, BYTE_512, KB};

    const TOTAL_MAPS: usize = 2;
    const MAP_CAPACITY: usize = 32;

    #[test]
    fn map_is_empty() {
        let mut map_src: MapSrc<i32> = MapSrc::new(TOTAL_MAPS);
        map_src.add(BYTE_256, MAP_CAPACITY);
        map_src.add(BYTE_512, MAP_CAPACITY / 2);
        map_src.add(KB, MAP_CAPACITY / 2);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let key_blocks = m.read_keys().unwrap();

        assert_eq!(key_blocks.len(), map_src.get_configs().len());

        let block_256 = &key_blocks[0];

        assert_eq!(block_256.config.value_len, BYTE_256);
        assert_eq!(
            block_256.keys,
            vec![vec![i32::cl_default(); DEFAULT_MAP_KEY_LENGTH]; MAP_CAPACITY]
        );

        let block_512 = &key_blocks[1];

        assert_eq!(block_512.config.value_len, BYTE_512);
        assert_eq!(
            block_512.keys,
            vec![vec![i32::cl_default(); DEFAULT_MAP_KEY_LENGTH]; MAP_CAPACITY / 2]
        );

        let block_1024 = &key_blocks[2];

        assert_eq!(block_1024.config.value_len, KB);
        assert_eq!(
            block_1024.keys,
            vec![vec![i32::cl_default(); DEFAULT_MAP_KEY_LENGTH]; MAP_CAPACITY / 2]
        );
    }

    #[test]
    fn map_is_full() {
        let mut map_src: MapSrc<i32> = MapSrc::new(TOTAL_MAPS);
        map_src.add(BYTE_256, MAP_CAPACITY * 2);
        map_src.add(BYTE_512, MAP_CAPACITY);
        map_src.add(KB, MAP_CAPACITY / 2);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix_256 =
            TestMatrix::new(MAP_CAPACITY * 2, DEFAULT_MAP_KEY_LENGTH, BYTE_256, 10, 100);
        test_matrix_256.put(&m, BYTE_256);

        let test_matrix_512 =
            TestMatrix::new(MAP_CAPACITY, DEFAULT_MAP_KEY_LENGTH, BYTE_512, 300, 100);
        test_matrix_512.put(&m, BYTE_512);

        let test_matrix_1024 =
            TestMatrix::new(MAP_CAPACITY / 2, DEFAULT_MAP_KEY_LENGTH, KB, 500, 100);
        test_matrix_1024.put(&m, KB);

        let key_blocks = m.read_keys().unwrap();

        assert_eq!(key_blocks.len(), map_src.get_configs().len());

        let block_256 = &key_blocks[0];

        assert_eq!(block_256.config.value_len, BYTE_256);
        assert_eq!(block_256.keys, test_matrix_256.keys);

        let block_512 = &key_blocks[1];

        assert_eq!(block_512.config.value_len, BYTE_512);
        assert_eq!(block_512.keys, test_matrix_512.keys);

        let block_1024 = &key_blocks[2];

        assert_eq!(block_1024.config.value_len, KB);
        assert_eq!(block_1024.keys, test_matrix_1024.keys);
    }

    #[test]
    fn map_is_not_full() {
        let mut map_src: MapSrc<i32> = MapSrc::new(TOTAL_MAPS);
        map_src.add(BYTE_256, MAP_CAPACITY);
        map_src.add(BYTE_512, MAP_CAPACITY);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix_256 = TestMatrix::new(
            MAP_CAPACITY / 2,
            DEFAULT_MAP_KEY_LENGTH / 2,
            BYTE_256 / 2,
            40,
            400,
        );
        test_matrix_256.put(&m, BYTE_256);

        let test_matrix_512 = TestMatrix::new(
            MAP_CAPACITY / 4,
            DEFAULT_MAP_KEY_LENGTH / 4,
            BYTE_512 / 4,
            80,
            800,
        );
        test_matrix_512.put(&m, BYTE_512);

        let key_blocks = m.read_keys().unwrap();

        assert_eq!(key_blocks.len(), map_src.get_configs().len());

        let block_256 = &key_blocks[0];

        assert_eq!(block_256.config.value_len, BYTE_256);

        for (i, key) in block_256.keys.iter().enumerate() {
            if i >= (MAP_CAPACITY / 2) {
                assert_eq!(key, &vec![i32::cl_default(); DEFAULT_MAP_KEY_LENGTH]);
                continue;
            }

            assert_eq!(
                key,
                &ensure_vec_size(&test_matrix_256.keys[i], DEFAULT_MAP_KEY_LENGTH)
            );
        }

        let block_512 = &key_blocks[1];

        assert_eq!(block_512.config.value_len, BYTE_512);

        for (i, key) in block_512.keys.iter().enumerate() {
            if i >= (MAP_CAPACITY / 4) {
                assert_eq!(key, &vec![i32::cl_default(); DEFAULT_MAP_KEY_LENGTH]);
                continue;
            }

            assert_eq!(
                key,
                &ensure_vec_size(&test_matrix_512.keys[i], DEFAULT_MAP_KEY_LENGTH)
            );
        }
    }
}
