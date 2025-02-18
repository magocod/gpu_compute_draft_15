use crate::config::{ClTypeTrait, DEBUG_MODE};
use crate::error::OpenClResult;
use crate::map::config::{check_local_work_size, DEFAULT_MAP_KEY_LENGTH};
use crate::map::handle::read::map_get_empty_key::PipeIndices;
use crate::map::handle::{EntryIndices, MapBlockSizes, MapHandle, MapKeys, MapValues};
use crate::map::kernel::name::MAP_ADD;
use crate::utils::ensure_vec_size;
use opencl::opencl_sys::bindings::{cl_int, cl_uint};
use opencl::wrapper::system::OpenclCommonOperation;

impl<T: ClTypeTrait, D: OpenclCommonOperation> MapHandle<T, D> {
    // For now, it is only reliable when storing few items
    // this method only works if map_key_len is equal to or less than map_value_len
    pub fn map_add(
        &self,
        keys: &MapKeys<T>,
        values: &MapValues<T>,
        pipes: &[PipeIndices],
    ) -> OpenClResult<(EntryIndices, MapBlockSizes)> {
        let map_config = &self.map_src;

        let global_work_size = keys.len();

        if global_work_size != values.len() {
            panic!("TODO handle error input keys & values len")
        }

        let local_work_size = check_local_work_size(global_work_size);

        let key_input_capacity = DEFAULT_MAP_KEY_LENGTH * global_work_size;

        // ...
        let max_value_len = map_config.get_max_value_len();
        let value_input_capacity = max_value_len * global_work_size;

        let enqueue_kernel_output_capacity = global_work_size;

        let mut keys_input = Vec::with_capacity(key_input_capacity);
        let mut values_input = Vec::with_capacity(value_input_capacity);
        let mut values_lens_input: Vec<cl_int> = Vec::with_capacity(global_work_size);

        for key in keys {
            let mut v = ensure_vec_size(key, DEFAULT_MAP_KEY_LENGTH);
            keys_input.append(&mut v);
        }

        for value in values {
            values_lens_input.push(value.len() as cl_int);

            let mut v = ensure_vec_size(value, max_value_len);
            values_input.append(&mut v);
        }

        let keys_input_buf = self.system.blocking_prepare_input_buffer(&keys_input)?;
        let values_input_buf = self.system.blocking_prepare_input_buffer(&values_input)?;
        let values_lens_input_buf = self
            .system
            .blocking_prepare_input_buffer(&values_lens_input)?;

        let indices_output_buf = self.system.create_output_buffer(global_work_size)?;
        let block_output_buf = self.system.create_output_buffer(global_work_size)?;
        let enqueue_kernel_output_buf = self
            .system
            .create_output_buffer(enqueue_kernel_output_capacity)?;

        let map_id = self.map_id as cl_uint;

        let q0 = self
            .system
            .get_device_command_queue_0()
            .get_cl_command_queue();

        let mut kernel = self.system.create_kernel(MAP_ADD)?;

        unsafe {
            kernel.set_arg(&q0)?;

            for pipe in pipes.iter() {
                let p = pipe.get_cl_mem();
                kernel.set_arg(&p)?;
            }

            kernel.set_arg(&map_id)?;

            kernel.set_arg(&keys_input_buf.get_cl_mem())?;
            kernel.set_arg(&values_input_buf.get_cl_mem())?;
            kernel.set_arg(&values_lens_input_buf.get_cl_mem())?;

            kernel.set_arg(&indices_output_buf.get_cl_mem())?;
            kernel.set_arg(&block_output_buf.get_cl_mem())?;
            kernel.set_arg(&enqueue_kernel_output_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        let indices_output =
            self.system
                .blocking_enqueue_read_buffer(global_work_size, &indices_output_buf, &[])?;

        let block_output =
            self.system
                .blocking_enqueue_read_buffer(global_work_size, &block_output_buf, &[])?;

        if DEBUG_MODE {
            println!("indices_output {indices_output:?}");
            println!("block_output   {block_output:?}");
        }

        self.system.assert_device_enqueue_kernel(
            enqueue_kernel_output_capacity,
            enqueue_kernel_output_buf,
            &[],
        )?;

        Ok((indices_output, block_output))
    }
}

#[cfg(test)]
mod tests_map_add {
    use super::*;
    use crate::map::config::{MapSrc, DEFAULT_MAP_KEY_LENGTH};
    use crate::map::handle::test_utils::{
        assert_map_block_is_empty, assert_map_block_is_equal_to_test_matrix,
        generate_arc_opencl_block_default, DefaultTypeTrait,
    };
    use crate::map::handle::{MapHandle, KEY_NOT_AVAILABLE_TO_ASSIGN};
    use crate::test_utils::TestMatrix;
    use crate::utils::{ensure_vec_size, BYTE_256, BYTE_512, KB};

    const TOTAL_MAPS: usize = 2;
    const MAP_CAPACITY: usize = 32;

    #[test]
    fn all_index_available() {
        let mut map_src: MapSrc<i32> = MapSrc::new(TOTAL_MAPS);
        map_src.add(BYTE_256, MAP_CAPACITY / 2);
        map_src.add(BYTE_512, MAP_CAPACITY / 2);

        map_src.add_map_add_program_src();

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix = TestMatrix::new(MAP_CAPACITY, DEFAULT_MAP_KEY_LENGTH, BYTE_256, 1, 10);

        let pipes = m.get_empty_keys_pipes().unwrap();

        let (indices, blocks) = m
            .map_add(&test_matrix.keys, &test_matrix.values, &pipes)
            .unwrap();

        assert_eq!(indices.len(), MAP_CAPACITY);
        assert_eq!(
            blocks.iter().filter(|&&x| x == BYTE_256 as cl_int).count(),
            MAP_CAPACITY / 2
        );
        assert_eq!(
            blocks.iter().filter(|&&x| x == BYTE_512 as cl_int).count(),
            MAP_CAPACITY / 2
        );

        let pairs = m.read(BYTE_256).unwrap();

        for (i, pair) in pairs.into_iter().enumerate() {
            let e_i = indices
                .iter()
                .enumerate()
                .position(|(sub_id, &entry_index)| {
                    (entry_index == i as cl_int) && blocks[sub_id] == BYTE_256 as cl_int
                })
                .unwrap();

            assert_eq!(pair.key, test_matrix.keys[e_i]);
            assert_eq!(pair.value, test_matrix.values[e_i]);
        }

        let pairs = m.read(BYTE_512).unwrap();

        for (i, pair) in pairs.into_iter().enumerate() {
            let e_i = indices
                .iter()
                .enumerate()
                .position(|(sub_id, &entry_index)| {
                    (entry_index == i as cl_int) && blocks[sub_id] == BYTE_512 as cl_int
                })
                .unwrap();

            assert_eq!(pair.key, test_matrix.keys[e_i]);
            assert_eq!(
                pair.value,
                ensure_vec_size(&test_matrix.values[e_i], BYTE_512)
            );
        }
    }

    #[test]
    fn no_index_available() {
        let mut map_src: MapSrc<i32> = MapSrc::new(TOTAL_MAPS);
        map_src.add(BYTE_256, MAP_CAPACITY);
        map_src.add(BYTE_512, MAP_CAPACITY);

        map_src.add_map_add_program_src();

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix = TestMatrix::new(MAP_CAPACITY, DEFAULT_MAP_KEY_LENGTH, BYTE_512, 3, 30);
        test_matrix.put(&m, BYTE_512);

        let test_m = TestMatrix::new(MAP_CAPACITY, DEFAULT_MAP_KEY_LENGTH, BYTE_512, 4, 40);

        let pipes = m.get_empty_keys_pipes().unwrap();

        let (indices, blocks) = m.map_add(&test_m.keys, &test_m.values, &pipes).unwrap();

        assert_eq!(indices, vec![KEY_NOT_AVAILABLE_TO_ASSIGN; MAP_CAPACITY]);
        assert_eq!(blocks, vec![0; MAP_CAPACITY]);

        assert_map_block_is_empty(&m, BYTE_256, DefaultTypeTrait::Custom);
        assert_map_block_is_equal_to_test_matrix(&m, BYTE_512, &test_matrix);
    }

    #[test]
    fn no_capacity_to_store() {
        let mut map_src: MapSrc<i32> = MapSrc::new(TOTAL_MAPS);
        map_src.add(BYTE_256, MAP_CAPACITY);
        map_src.add(BYTE_512, MAP_CAPACITY);

        map_src.add_map_add_program_src();

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_m = TestMatrix::new(MAP_CAPACITY, DEFAULT_MAP_KEY_LENGTH, KB, 4, 40);

        let pipes = m.get_empty_keys_pipes().unwrap();

        let (indices, blocks) = m.map_add(&test_m.keys, &test_m.values, &pipes).unwrap();

        assert_eq!(indices, vec![KEY_NOT_AVAILABLE_TO_ASSIGN; MAP_CAPACITY]);
        assert_eq!(blocks, vec![0; MAP_CAPACITY]);

        assert_map_block_is_empty(&m, BYTE_256, DefaultTypeTrait::Custom);
        assert_map_block_is_empty(&m, BYTE_512, DefaultTypeTrait::Custom);
    }

    // TODO some_indices_available
}
