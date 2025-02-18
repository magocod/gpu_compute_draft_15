use crate::config::ClTypeTrait;
use crate::error::OpenClResult;
use crate::map::config::{MapConfig, DEFAULT_MAP_KEY_LENGTH};
use crate::map::handle::{MapHandle, Pair};
use crate::map::kernel::name::MAP_READ_ASSIGNED_KEYS;
use opencl::opencl_sys::bindings::{cl_int, cl_uint};
use opencl::wrapper::system::OpenclCommonOperation;

#[derive(Debug, Clone)]
pub struct MapEntries<T: ClTypeTrait> {
    pub config: MapConfig<T>,
    pub pairs: Vec<Pair<T>>,
}

impl<T: ClTypeTrait> MapEntries<T> {
    pub fn new(config: &MapConfig<T>, pairs: Vec<Pair<T>>) -> MapEntries<T> {
        Self {
            config: config.clone(),
            pairs,
        }
    }
}

impl<T: ClTypeTrait, D: OpenclCommonOperation> MapHandle<T, D> {
    pub fn read_assigned_keys(&self) -> OpenClResult<Vec<MapEntries<T>>> {
        let map_config = &self.map_src;
        let map_blocks = map_config.get_configs();

        let global_work_size = 1;
        let local_work_size = 1;

        let enqueue_kernel_output_capacity = map_blocks.len();

        let indices_output_capacity = map_blocks.iter().map(|x| x.capacity).sum();
        let key_output_capacity = map_blocks
            .iter()
            .map(|x| x.capacity * DEFAULT_MAP_KEY_LENGTH)
            .sum();
        let value_output_capacity = map_blocks.iter().map(|x| x.capacity * x.value_len).sum();

        let indices_output_buf = self
            .system
            .create_output_buffer::<cl_int>(indices_output_capacity)?;
        let keys_output_buf = self.system.create_output_buffer(key_output_capacity)?;
        let values_output_buf = self.system.create_output_buffer(value_output_capacity)?;

        let enqueue_kernel_output_buf = self
            .system
            .create_output_buffer(enqueue_kernel_output_capacity)?;

        let mut kernel = self.system.create_kernel(MAP_READ_ASSIGNED_KEYS)?;

        let q0 = self
            .system
            .get_device_command_queue_0()
            .get_cl_command_queue();

        let map_id = self.map_id as cl_uint;

        unsafe {
            kernel.set_arg(&q0)?;
            kernel.set_arg(&map_id)?;
            kernel.set_arg(&indices_output_buf.get_cl_mem())?;
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

        let indices_output = self.system.blocking_enqueue_read_buffer(
            indices_output_capacity,
            &indices_output_buf,
            &[],
        )?;

        let keys_output =
            self.system
                .blocking_enqueue_read_buffer(key_output_capacity, &keys_output_buf, &[])?;

        let values_output = self.system.blocking_enqueue_read_buffer(
            value_output_capacity,
            &values_output_buf,
            &[],
        )?;

        let mut output: Vec<MapEntries<T>> = Vec::with_capacity(map_blocks.len());

        for (i, config) in map_blocks.iter().enumerate() {
            let output_map_index: usize = map_blocks[0..i].iter().map(|x| x.capacity).sum();

            let output_key_index: usize = map_blocks[0..i]
                .iter()
                .map(|x| x.capacity * DEFAULT_MAP_KEY_LENGTH)
                .sum();

            let output_value_index: usize = map_blocks[0..i]
                .iter()
                .map(|x| x.capacity * x.value_len)
                .sum();

            let indices = &indices_output[output_map_index..(output_map_index + (config.capacity))];

            let keys: Vec<Vec<_>> = keys_output
                [output_key_index..(output_key_index + (config.capacity * DEFAULT_MAP_KEY_LENGTH))]
                .chunks(DEFAULT_MAP_KEY_LENGTH)
                .map(|x| x.to_vec())
                .collect();

            let values: Vec<Vec<_>> = values_output
                [output_value_index..(output_value_index + (config.capacity * config.value_len))]
                .chunks(config.value_len)
                .map(|x| x.to_vec())
                .collect();

            let pairs = indices
                .iter()
                .enumerate()
                .filter_map(|(i, &key_size)| -> Option<Pair<T>> {
                    if key_size > 0 {
                        return Some(Pair::create_with_index(
                            keys[i].clone(),
                            values[i].clone(),
                            Some(i),
                        ));
                    }
                    None
                })
                .collect();

            let map_entries = MapEntries::new(config, pairs);

            output.push(map_entries);
        }

        self.system.assert_device_enqueue_kernel(
            enqueue_kernel_output_capacity,
            enqueue_kernel_output_buf,
            &[],
        )?;

        Ok(output)
    }
}

#[cfg(test)]
mod tests_map_read_assigned_keys {
    use crate::map::config::{MapSrc, DEFAULT_MAP_KEY_LENGTH};
    use crate::map::handle::test_utils::generate_arc_opencl_block_default;
    use crate::map::handle::MapHandle;
    use crate::test_utils::TestMatrix;
    use crate::utils::{ensure_vec_size, BYTE_256, BYTE_512, KB};

    const TOTAL_MAPS: usize = 2;
    const MAP_CAPACITY: usize = 32;

    #[test]
    fn map_is_empty() {
        let mut map_src: MapSrc<i16> = MapSrc::new(TOTAL_MAPS);
        map_src.add(BYTE_256, MAP_CAPACITY);
        map_src.add(BYTE_512, MAP_CAPACITY);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let map_entries = m.read_assigned_keys().unwrap();

        assert_eq!(map_entries.len(), map_src.get_configs().len());

        let block_256 = &map_entries[0];
        assert_eq!(block_256.config.value_len, BYTE_256);
        assert_eq!(block_256.pairs, vec![]);

        let block_512 = &map_entries[1];
        assert_eq!(block_512.config.value_len, BYTE_512);
        assert_eq!(block_512.pairs, vec![]);
    }

    #[test]
    fn map_is_full() {
        let mut map_src: MapSrc<i16> = MapSrc::new(TOTAL_MAPS);
        map_src.add(BYTE_256, MAP_CAPACITY * 2);
        map_src.add(BYTE_512, MAP_CAPACITY);
        map_src.add(KB, MAP_CAPACITY / 2);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix_256 =
            TestMatrix::new(MAP_CAPACITY * 2, DEFAULT_MAP_KEY_LENGTH, BYTE_256, 10, 100);
        test_matrix_256.put(&m, BYTE_256);

        let test_matrix_512 =
            TestMatrix::new(MAP_CAPACITY, DEFAULT_MAP_KEY_LENGTH, BYTE_512, 300, 200);
        test_matrix_512.put(&m, BYTE_512);

        let test_matrix_1024 =
            TestMatrix::new(MAP_CAPACITY / 2, DEFAULT_MAP_KEY_LENGTH, KB, 500, 300);
        test_matrix_1024.put(&m, KB);

        let map_entries = m.read_assigned_keys().unwrap();

        assert_eq!(map_entries.len(), map_src.get_configs().len());

        // for block in map_entries.iter() {
        //     println!("{:?}", block.config.name);
        //     for pair in block.pairs.iter() {
        //         println!("i {:?}", pair.entry_index);
        //         println!("  key:   {:?}", pair.key);
        //         println!("  value: {:?}", pair.value);
        //     }
        // }

        let block_256 = &map_entries[0];
        assert_eq!(block_256.config.value_len, BYTE_256);

        for (i, pair) in block_256.pairs.iter().enumerate() {
            assert_eq!(pair.entry_index, Some(i));
            assert_eq!(pair.key, test_matrix_256.keys[i]);
            assert_eq!(pair.value, test_matrix_256.values[i]);
        }

        let block_512 = &map_entries[1];
        assert_eq!(block_512.config.value_len, BYTE_512);

        for (i, pair) in block_512.pairs.iter().enumerate() {
            assert_eq!(pair.entry_index, Some(i));
            assert_eq!(pair.key, test_matrix_512.keys[i]);
            assert_eq!(pair.value, test_matrix_512.values[i]);
        }

        let block_1024 = &map_entries[2];
        assert_eq!(block_1024.config.value_len, KB);

        for (i, pair) in block_1024.pairs.iter().enumerate() {
            assert_eq!(pair.entry_index, Some(i));
            assert_eq!(pair.key, test_matrix_1024.keys[i]);
            assert_eq!(pair.value, test_matrix_1024.values[i]);
        }
    }

    #[test]
    fn map_is_not_full() {
        let mut map_src: MapSrc<i16> = MapSrc::new(TOTAL_MAPS);
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

        let test_matrix_512 = TestMatrix::new(MAP_CAPACITY / 4, 1, 1, 50, 500);
        test_matrix_512.put(&m, BYTE_512);

        let map_entries = m.read_assigned_keys().unwrap();

        // for block in map_entries.iter() {
        //     println!("{:?}", block.config.name);
        //     for pair in block.pairs.iter() {
        //         println!("i {:?}", pair.entry_index);
        //         println!("  key:   {:?}", pair.key);
        //         println!("  value: {:?}", pair.value);
        //     }
        // }

        assert_eq!(map_entries.len(), map_src.get_configs().len());

        let block_256 = &map_entries[0];
        assert_eq!(block_256.config.value_len, BYTE_256);

        for (i, pair) in block_256.pairs.iter().enumerate() {
            assert_eq!(pair.entry_index, Some(i));
            assert_eq!(
                pair.key,
                ensure_vec_size(&test_matrix_256.keys[i], DEFAULT_MAP_KEY_LENGTH)
            );
            assert_eq!(
                pair.value,
                ensure_vec_size(&test_matrix_256.values[i], BYTE_256)
            );
        }

        let block_512 = &map_entries[1];
        assert_eq!(block_512.config.value_len, BYTE_512);

        for (i, pair) in block_512.pairs.iter().enumerate() {
            assert_eq!(pair.entry_index, Some(i));
            assert_eq!(
                pair.key,
                ensure_vec_size(&test_matrix_512.keys[i], DEFAULT_MAP_KEY_LENGTH)
            );
            assert_eq!(
                pair.value,
                ensure_vec_size(&test_matrix_512.values[i], BYTE_512)
            );
        }
    }
}
