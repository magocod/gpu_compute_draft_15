use crate::config::{ClTypeTrait, DEBUG_MODE};
use crate::error::OpenClResult;
use crate::map::config::{check_local_work_size, check_max_find_work_size, DEFAULT_MAP_KEY_LENGTH};
use crate::map::handle::read::map_get_empty_key::PipeIndices;
use crate::map::handle::tmp::TmpMultiple;
use crate::map::handle::{EntryIndices, MapBlockSizes, MapHandle, MapKeys, MapValues};
use crate::map::kernel::name::{GET_TMP_FOR_MAP_INSERT, MAP_INSERT};
use crate::utils::ensure_vec_size;
use opencl::opencl_sys::bindings::{cl_int, cl_uint};
use opencl::wrapper::system::OpenclCommonOperation;

impl<T: ClTypeTrait, D: OpenclCommonOperation> MapHandle<T, D> {
    // For now, it is only reliable when storing few items
    pub fn map_insert(
        &self,
        keys: &MapKeys<T>,
        values: &MapValues<T>,
        pipes: &[PipeIndices],
    ) -> OpenClResult<(EntryIndices, MapBlockSizes)> {
        check_max_find_work_size(keys.len());

        let map_config = &self.map_src;

        let global_work_size = keys.len();
        if global_work_size != values.len() {
            panic!("TODO handle error input lens")
        }
        let local_work_size = check_local_work_size(global_work_size);

        let max_value_len = map_config.get_max_value_len();

        // CMQ_COMPARE_KEY_IN_BLOCKS
        // CMQ_CONFIRM_MAP_INSERT
        // CMQ_CONFIRM_MAP_PUT
        // CMQ_CONFIRM_MAP_REMOVE
        let total_queues = map_config.get_configs().len() + 4;

        let key_input_capacity = DEFAULT_MAP_KEY_LENGTH * global_work_size;
        let value_input_capacity = max_value_len * global_work_size;
        let enqueue_kernel_output_capacity = total_queues * global_work_size;

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

        let mut kernel = self.system.create_kernel(MAP_INSERT)?;

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

    pub fn get_tmp_match_indices_for_map_insert(
        &self,
        elements: usize,
    ) -> OpenClResult<TmpMultiple<T>> {
        self.get_tmp_multiple(GET_TMP_FOR_MAP_INSERT, elements)
    }
}

#[cfg(test)]
mod tests_map_insert {
    use super::*;
    use crate::config::ClTypeDefault;
    use crate::map::config::{MapSrc, DEFAULT_MAP_KEY_LENGTH, MAX_FIND_WORK_SIZE};
    use crate::map::handle::test_utils::{
        assert_map_block_is_empty, assert_map_block_is_equal_to_test_matrix,
        generate_arc_opencl_block_default, DefaultTypeTrait,
    };
    use crate::map::handle::{MapHandle, KEY_NOT_EXISTS};
    use crate::test_utils::TestMatrix;
    use crate::utils::{ensure_vec_size, BYTE_256, BYTE_512, KB};

    #[test]
    fn key_does_not_exist() {
        let input_len = 8;

        let mut map_src: MapSrc<i16> = MapSrc::new(2);
        map_src.add(BYTE_256, 32);
        map_src.add(BYTE_512, 32);
        map_src.add(KB, 16);

        map_src.add_map_insert_program_src(MAX_FIND_WORK_SIZE);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix = TestMatrix::new(input_len, DEFAULT_MAP_KEY_LENGTH, BYTE_256, 1, 10);

        let pipes = m.get_empty_keys_pipes().unwrap();

        let (indices, blocks) = m
            .map_insert(&test_matrix.keys, &test_matrix.values, &pipes)
            .unwrap();

        let tmp_arr = m.get_tmp_match_indices_for_map_insert(input_len).unwrap();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();
        let expected_indices: Vec<_> = (0..input_len as cl_int).collect();
        assert_eq!(indices_sorted, expected_indices);

        assert_eq!(blocks, vec![BYTE_256 as cl_int; input_len]);

        let pairs = m.read(BYTE_256).unwrap();

        for (i, index) in indices.iter().enumerate() {
            let u_index = *index as usize;

            assert_eq!(
                pairs[u_index].key,
                ensure_vec_size(&test_matrix.keys[i], BYTE_256)
            );
            assert_eq!(
                pairs[u_index].value,
                ensure_vec_size(&test_matrix.values[i], BYTE_256)
            );
        }

        for (i, pair) in pairs.into_iter().enumerate() {
            if !indices.contains(&(i as cl_int)) {
                assert_eq!(pair.key, vec![i16::cl_default(); DEFAULT_MAP_KEY_LENGTH]);
                assert_eq!(pair.value, vec![i16::cl_default(); BYTE_256]);
            }
        }

        assert_map_block_is_empty(&m, BYTE_512, DefaultTypeTrait::Custom);
        assert_map_block_is_empty(&m, KB, DefaultTypeTrait::Custom);

        for tmp_element in tmp_arr {
            // println!("{blocks:#?}");
            for b in tmp_element.blocks {
                assert_eq!(b.values, vec![KEY_NOT_EXISTS; b.config.capacity])
            }
        }
    }

    #[test]
    fn key_does_not_exist_and_save_in_multiple_blocks() {
        let input_len = 16;

        let mut map_src: MapSrc<i16> = MapSrc::new(2);
        map_src.add(BYTE_256, 8);
        map_src.add(BYTE_512, 8);

        map_src.add_map_insert_program_src(MAX_FIND_WORK_SIZE);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix = TestMatrix::new(input_len, DEFAULT_MAP_KEY_LENGTH, BYTE_256, 1, 10);

        let pipes = m.get_empty_keys_pipes().unwrap();

        let (indices, blocks) = m
            .map_insert(&test_matrix.keys, &test_matrix.values, &pipes)
            .unwrap();

        let tmp_arr = m.get_tmp_match_indices_for_map_insert(input_len).unwrap();

        let mut expected_indices: Vec<_> = (0..(input_len / 2) as cl_int).collect();
        expected_indices.append(&mut (0..(input_len / 2) as cl_int).collect());
        expected_indices.sort();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();
        assert_eq!(indices_sorted, expected_indices);

        assert_eq!(
            blocks.iter().filter(|&x| *x == BYTE_256 as cl_int).count(),
            input_len / 2
        );
        assert_eq!(
            blocks.iter().filter(|&x| *x == BYTE_512 as cl_int).count(),
            input_len / 2
        );

        let pairs_256 = m.read(BYTE_256).unwrap();

        for (i, pair) in pairs_256.into_iter().enumerate() {
            let m_i = indices
                .iter()
                .enumerate()
                .position(|(sub_id, &entry_index)| {
                    (entry_index == i as cl_int) && blocks[sub_id] == BYTE_256 as cl_int
                })
                .unwrap();

            assert_eq!(pair.key, test_matrix.keys[m_i]);
            assert_eq!(pair.value, test_matrix.values[m_i]);
        }

        let pairs_512 = m.read(BYTE_512).unwrap();

        for (i, pair) in pairs_512.into_iter().enumerate() {
            let m_i = indices
                .iter()
                .enumerate()
                .position(|(sub_id, &entry_index)| {
                    (entry_index == i as cl_int) && blocks[sub_id] == BYTE_512 as cl_int
                })
                .unwrap();

            assert_eq!(pair.key, test_matrix.keys[m_i]);
            assert_eq!(
                pair.value,
                ensure_vec_size(&test_matrix.values[m_i], BYTE_512)
            );
        }

        for tmp_element in tmp_arr {
            for b in tmp_element.blocks {
                assert_eq!(b.values, vec![KEY_NOT_EXISTS; b.config.capacity])
            }
        }
    }

    #[test]
    fn no_capacity_to_store() {
        let input_len = 8;

        let mut map_src: MapSrc<i16> = MapSrc::new(2);
        map_src.add(BYTE_256, 32);
        map_src.add(BYTE_512, 32);
        map_src.add(KB, 32);

        map_src.add_map_insert_program_src(MAX_FIND_WORK_SIZE);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let mut test_matrix = TestMatrix::new(input_len / 2, DEFAULT_MAP_KEY_LENGTH, KB * 2, 2, 20);
        let mut test_matrix_2 =
            TestMatrix::new(input_len / 2, DEFAULT_MAP_KEY_LENGTH, KB + 1, 2, 20);

        test_matrix.append(&mut test_matrix_2);

        let pipes = m.get_empty_keys_pipes().unwrap();

        let (indices, blocks) = m
            .map_insert(&test_matrix.keys, &test_matrix.values, &pipes)
            .unwrap();

        let tmp_arr = m.get_tmp_match_indices_for_map_insert(input_len).unwrap();

        assert_eq!(indices, vec![-1; input_len]);
        assert_eq!(blocks, vec![0; input_len]);

        assert_map_block_is_empty(&m, BYTE_256, DefaultTypeTrait::Custom);
        assert_map_block_is_empty(&m, BYTE_512, DefaultTypeTrait::Custom);
        assert_map_block_is_empty(&m, KB, DefaultTypeTrait::Custom);

        for tmp_element in tmp_arr {
            for b in tmp_element.blocks {
                assert_eq!(b.values, vec![-3; b.config.capacity])
            }
        }
    }

    #[test]
    fn key_already_exists() {
        let input_len = 16;

        let mut map_src: MapSrc<i16> = MapSrc::new(2);
        map_src.add(BYTE_256, 32);
        map_src.add(BYTE_512, 16);
        map_src.add(KB, 8);

        map_src.add_map_insert_program_src(MAX_FIND_WORK_SIZE);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let fill_key_with = 3;

        let t_m_256 = TestMatrix::new(
            input_len,
            DEFAULT_MAP_KEY_LENGTH,
            BYTE_256,
            fill_key_with,
            30,
        );
        t_m_256.put(&m, BYTE_256);

        let t_m_512 = TestMatrix::new(
            input_len,
            DEFAULT_MAP_KEY_LENGTH,
            BYTE_512,
            fill_key_with,
            40,
        );
        t_m_512.put(&m, BYTE_512);

        let test_matrix = TestMatrix::new(
            input_len,
            DEFAULT_MAP_KEY_LENGTH,
            BYTE_256,
            fill_key_with,
            100,
        );

        let pipes = m.get_empty_keys_pipes().unwrap();

        let (indices, blocks) = m
            .map_insert(&test_matrix.keys, &test_matrix.values, &pipes)
            .unwrap();

        let tmp_arr = m.get_tmp_match_indices_for_map_insert(input_len).unwrap();

        let expected_indices: Vec<_> = (0..input_len as cl_int).collect();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();
        assert_eq!(indices_sorted, expected_indices);

        assert_eq!(blocks, vec![BYTE_256 as cl_int; input_len]);

        let pairs = m.read(BYTE_256).unwrap();

        for (i, pair) in pairs[0..input_len].iter().enumerate() {
            let m_i = indices.iter().position(|&r| r == i as cl_int).unwrap();

            assert_eq!(pair.key, test_matrix.keys[m_i]);
            assert_eq!(pair.value, test_matrix.values[m_i]);
        }

        for pair in pairs[input_len..].iter() {
            assert_eq!(pair.key, vec![i16::cl_default(); DEFAULT_MAP_KEY_LENGTH]);
            assert_eq!(pair.value, vec![i16::cl_default(); BYTE_256]);
        }

        assert_map_block_is_equal_to_test_matrix(&m, BYTE_512, &t_m_512);
        assert_map_block_is_empty(&m, KB, DefaultTypeTrait::Custom);

        for tmp_element in tmp_arr {
            for _b in tmp_element.blocks {
                // TODO assert tmp_array
                // assert_eq!(b.inner, vec![-3; b.config.matrix_column_len])
            }
        }
    }

    #[test]
    fn relocate_blocks() {
        let input_len = 16;

        let mut map_src: MapSrc<i16> = MapSrc::new(2);
        map_src.add(BYTE_256, 16);
        map_src.add(BYTE_512, 32);
        map_src.add(KB, 16);

        map_src.add_map_insert_program_src(MAX_FIND_WORK_SIZE);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let fill_key_with = 3;

        let t_m_256 = TestMatrix::new(
            input_len,
            DEFAULT_MAP_KEY_LENGTH,
            BYTE_256,
            fill_key_with,
            30,
        );
        t_m_256.put(&m, BYTE_256);

        let test_matrix = TestMatrix::new(
            input_len,
            DEFAULT_MAP_KEY_LENGTH,
            BYTE_512,
            fill_key_with,
            100,
        );

        let pipes = m.get_empty_keys_pipes().unwrap();

        let (indices, blocks) = m
            .map_insert(&test_matrix.keys, &test_matrix.values, &pipes)
            .unwrap();

        let tmp_arr = m.get_tmp_match_indices_for_map_insert(input_len).unwrap();

        let expected_indices: Vec<_> = (0..input_len as cl_int).collect();

        let mut indices_sorted = indices.clone();
        indices_sorted.sort();
        assert_eq!(indices_sorted, expected_indices);

        assert_eq!(blocks, vec![BYTE_512 as cl_int; input_len]);

        let pairs = m.read(BYTE_512).unwrap();

        for (i, pair) in pairs[0..input_len].iter().enumerate() {
            let m_i = indices.iter().position(|&r| r == i as cl_int).unwrap();

            assert_eq!(pair.key, test_matrix.keys[m_i]);
            assert_eq!(pair.value, test_matrix.values[m_i]);
        }

        for pair in pairs[input_len..].iter() {
            assert_eq!(pair.key, vec![i16::cl_default(); DEFAULT_MAP_KEY_LENGTH]);
            assert_eq!(pair.value, vec![i16::cl_default(); BYTE_512]);
        }

        assert_map_block_is_empty(&m, BYTE_256, DefaultTypeTrait::Custom);
        assert_map_block_is_empty(&m, KB, DefaultTypeTrait::Custom);

        for tmp_element in tmp_arr {
            for _b in tmp_element.blocks {
                // TODO assert tmp_array
                // assert_eq!(b.inner, vec![-3; b.config.matrix_column_len])
            }
        }
    }

    #[test]
    fn failed_reassignment() {
        let input_len = 16;

        let mut map_src = MapSrc::new(2);
        map_src.add(BYTE_256, 16);
        map_src.add(BYTE_512, 16);
        map_src.add(KB, 16);

        map_src.add_map_insert_program_src(MAX_FIND_WORK_SIZE);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let fill_key_with = 3;

        let t_m_256 = TestMatrix::new(
            input_len,
            DEFAULT_MAP_KEY_LENGTH,
            BYTE_256,
            fill_key_with,
            30,
        );
        t_m_256.put(&m, BYTE_256);

        let test_matrix = TestMatrix::new(
            input_len,
            DEFAULT_MAP_KEY_LENGTH,
            KB + 1,
            fill_key_with,
            100,
        );

        let pipes = m.get_empty_keys_pipes().unwrap();

        let (indices, blocks) = m
            .map_insert(&test_matrix.keys, &test_matrix.values, &pipes)
            .unwrap();

        let tmp_arr = m.get_tmp_match_indices_for_map_insert(input_len).unwrap();

        assert_eq!(indices, vec![-1; input_len]);
        assert_eq!(blocks, vec![0; input_len]);

        assert_map_block_is_equal_to_test_matrix(&m, BYTE_256, &t_m_256);

        assert_map_block_is_empty(&m, BYTE_512, DefaultTypeTrait::Custom);
        assert_map_block_is_empty(&m, KB, DefaultTypeTrait::Custom);

        for tmp_element in tmp_arr {
            for _b in tmp_element.blocks {
                // TODO assert tmp_array
                // assert_eq!(b.inner, vec![-3; b.config.matrix_column_len])
            }
        }
    }

    #[test]
    fn failed_reassignment_2() {
        let input_len = 16;

        let mut map_src = MapSrc::new(2);
        map_src.add(BYTE_256, 16);
        map_src.add(BYTE_512, 16);
        map_src.add(KB, 16);

        map_src.add_map_insert_program_src(MAX_FIND_WORK_SIZE);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let fill_key_with = 3;

        let t_m_512 = TestMatrix::new(
            input_len,
            DEFAULT_MAP_KEY_LENGTH,
            BYTE_512,
            fill_key_with,
            30,
        );
        t_m_512.put(&m, BYTE_512);

        let test_matrix = TestMatrix::new(
            input_len,
            DEFAULT_MAP_KEY_LENGTH,
            KB + 1,
            fill_key_with,
            100,
        );

        let pipes = m.get_empty_keys_pipes().unwrap();

        let (indices, blocks) = m
            .map_insert(&test_matrix.keys, &test_matrix.values, &pipes)
            .unwrap();

        let tmp_arr = m.get_tmp_match_indices_for_map_insert(input_len).unwrap();

        assert_eq!(indices, vec![-1; input_len]);
        assert_eq!(blocks, vec![0; input_len]);

        assert_map_block_is_equal_to_test_matrix(&m, BYTE_512, &t_m_512);

        assert_map_block_is_empty(&m, BYTE_256, DefaultTypeTrait::Custom);
        assert_map_block_is_empty(&m, KB, DefaultTypeTrait::Custom);

        for tmp_element in tmp_arr {
            for _b in tmp_element.blocks {
                // TODO assert tmp_array
                // assert_eq!(b.inner, vec![-3; b.config.matrix_column_len])
            }
        }
    }

    #[test]
    fn no_indices_available() {
        let input_len = 8;

        let mut map_src: MapSrc<i16> = MapSrc::new(2);
        map_src.add(BYTE_256, 32);
        map_src.add(BYTE_512, 32);
        map_src.add(KB, 16);

        map_src.add_map_insert_program_src(MAX_FIND_WORK_SIZE);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix_256 = TestMatrix::new(32, DEFAULT_MAP_KEY_LENGTH, BYTE_256, 1, 10);
        test_matrix_256.put(&m, BYTE_256);

        let test_matrix_512 = TestMatrix::new(32, DEFAULT_MAP_KEY_LENGTH, BYTE_512, 100, 100);
        test_matrix_512.put(&m, BYTE_512);

        let test_matrix_1024 = TestMatrix::new(16, DEFAULT_MAP_KEY_LENGTH, KB, 200, 200);
        test_matrix_1024.put(&m, KB);

        let test_matrix = TestMatrix::new(input_len, DEFAULT_MAP_KEY_LENGTH, BYTE_256, 2000, 20000);

        let pipes = m.get_empty_keys_pipes().unwrap();

        let (indices, blocks) = m
            .map_insert(&test_matrix.keys, &test_matrix.values, &pipes)
            .unwrap();

        let tmp_arr = m.get_tmp_match_indices_for_map_insert(input_len).unwrap();

        assert_eq!(indices, vec![-1; input_len]);
        assert_eq!(blocks, vec![0; input_len]);

        assert_map_block_is_equal_to_test_matrix(&m, BYTE_256, &test_matrix_256);
        assert_map_block_is_equal_to_test_matrix(&m, BYTE_512, &test_matrix_512);
        assert_map_block_is_equal_to_test_matrix(&m, KB, &test_matrix_1024);

        for tmp_element in tmp_arr {
            for b in tmp_element.blocks {
                assert_eq!(b.values, vec![-3; b.config.capacity])
            }
        }
    }

    // error
    #[test]
    fn the_same_simultaneous_key() {
        let input_len = 16;

        let mut map_src: MapSrc<i16> = MapSrc::new(2);
        map_src.add(BYTE_256, 16);
        map_src.add(BYTE_512, 16);

        map_src.add_map_insert_program_src(MAX_FIND_WORK_SIZE);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let keys = vec![vec![1; DEFAULT_MAP_KEY_LENGTH]; input_len];
        let values = vec![vec![2; BYTE_256]; input_len];

        let pipes = m.get_empty_keys_pipes().unwrap();
        let (indices, blocks) = m.map_insert(&keys, &values, &pipes).unwrap();

        let tmp_arr = m.get_tmp_match_indices_for_map_insert(input_len).unwrap();

        // TODO assert_eq!(indices, ??);
        assert_eq!(blocks, vec![BYTE_256 as cl_int; input_len]);

        let pairs = m.read(BYTE_256).unwrap();

        for (i, pair) in pairs.into_iter().enumerate() {
            if indices.contains(&(i as cl_int)) {
                assert_eq!(pair.key, vec![1; DEFAULT_MAP_KEY_LENGTH]);
                assert_eq!(pair.value, vec![2; BYTE_256]);
                continue;
            }

            assert_eq!(pair.key, vec![i16::cl_default(); DEFAULT_MAP_KEY_LENGTH]);
            assert_eq!(pair.value, vec![i16::cl_default(); BYTE_256]);
        }

        assert_map_block_is_empty(&m, BYTE_512, DefaultTypeTrait::Custom);

        for tmp_element in tmp_arr {
            for _b in tmp_element.blocks {
                // TODO assert_eq!(b.inner, vec![-3; b.config.matrix_column_len]);
            }
        }
    }

    #[test]
    fn the_same_simultaneous_key_2() {
        let input_len = 16;

        let mut map_src: MapSrc<i16> = MapSrc::new(2);
        map_src.add(BYTE_256, 16);
        map_src.add(BYTE_512, 16);

        map_src.add_map_insert_program_src(MAX_FIND_WORK_SIZE);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let key = vec![1; DEFAULT_MAP_KEY_LENGTH];
        let value = vec![2; BYTE_256];

        let keys = vec![key.clone(); input_len];
        let values = vec![value.clone(); input_len];

        m.put(BYTE_256, &vec![key], &vec![value]).unwrap();

        let pipes = m.get_empty_keys_pipes().unwrap();
        let (indices, blocks) = m.map_insert(&keys, &values, &pipes).unwrap();

        let tmp_arr = m.get_tmp_match_indices_for_map_insert(input_len).unwrap();

        // TODO assert_eq!(indices, ??);
        assert_eq!(blocks, vec![BYTE_256 as cl_int; input_len]);

        let pairs = m.read(BYTE_256).unwrap();

        for (i, pair) in pairs.into_iter().enumerate() {
            if indices.contains(&(i as cl_int)) {
                assert_eq!(pair.key, vec![1; DEFAULT_MAP_KEY_LENGTH]);
                assert_eq!(pair.value, vec![2; BYTE_256]);
                continue;
            }

            assert_eq!(pair.key, vec![i16::cl_default(); DEFAULT_MAP_KEY_LENGTH]);
            assert_eq!(pair.value, vec![i16::cl_default(); BYTE_256]);
        }

        assert_map_block_is_empty(&m, BYTE_512, DefaultTypeTrait::Custom);

        for tmp_element in tmp_arr {
            for _b in tmp_element.blocks {
                // TODO assert_eq!(b.inner, vec![-3; b.config.matrix_column_len]);
            }
        }
    }
}

// fatal errors
// #[cfg(test)]
// mod tests_issues_map_insert {
//     use super::*;
//     use crate::config::ClTypeDefault;
//     use crate::map::config::{MapSrc, DEFAULT_MAP_KEY_LENGTH, MAX_FIND_WORK_SIZE};
//     use crate::map::handle::test_utils::generate_arc_opencl_block_default;
//     use crate::map::handle::{Map, KEY_NOT_EXISTS};
//     use crate::test_utils::TestMatrix;
//     use crate::utils::{ensure_vec_size, BYTE_256};
//
//     // TODO simultaneous_insertions
//
//     // fatal error
//     #[test]
//     fn too_many_blocks() {
//         let input_len = 8;
//
//         let total_blocks = 16;
//
//         let mut map_src: MapSrc<i16> = MapSrc::default();
//
//         for i in 0..total_blocks {
//             map_src.add(BYTE_256 * (i + 1), 1024);
//         }
//
//         map_src.add_map_insert_program_src(MAX_FIND_WORK_SIZE);
//
//         println!("{map_src:?}");
//
//         let last_config = map_src.get_configs().last().unwrap();
//         let value_len = last_config.value_len;
//
//         let system = generate_arc_opencl_block_default(&map_src);
//         let m = Map::new(0, &map_src, system);
//
//         let test_matrix = TestMatrix::new(input_len, DEFAULT_MAP_KEY_LENGTH, value_len, 1, 10);
//
//         let pipes = m.get_empty_keys_pipes().unwrap();
//
//         let (indices, blocks) = m
//             .map_insert(&test_matrix.keys, &test_matrix.values, &pipes)
//             .unwrap();
//
//         let tmp_arr = m.get_tmp_match_indices_for_map_insert(input_len).unwrap();
//
//         // let mut indices_sorted = indices.clone();
//         // indices_sorted.sort();
//         // let expected_indices: Vec<_> = (0..input_len as cl_int).collect();
//         // assert_eq!(indices_sorted, expected_indices);
//
//         assert_eq!(blocks, vec![value_len as cl_int; input_len]);
//
//         let pairs = m.read(value_len).unwrap();
//
//         for (i, index) in indices.iter().enumerate() {
//             let u_index = *index as usize;
//
//             assert_eq!(
//                 pairs[u_index].key,
//                 ensure_vec_size(&test_matrix.keys[i], DEFAULT_MAP_KEY_LENGTH)
//             );
//             assert_eq!(
//                 pairs[u_index].value,
//                 ensure_vec_size(&test_matrix.values[i], value_len)
//             );
//         }
//
//         for (i, pair) in pairs.into_iter().enumerate() {
//             if !indices.contains(&(i as cl_int)) {
//                 assert_eq!(pair.key, vec![i16::cl_default(); DEFAULT_MAP_KEY_LENGTH]);
//                 assert_eq!(pair.value, vec![i16::cl_default(); value_len]);
//             }
//         }
//
//         // assert_map_block_is_empty(&m, BYTE_512);
//         // assert_map_block_is_empty(&m, KB);
//
//         for tmp_element in tmp_arr {
//             // println!("{blocks:#?}");
//             for b in tmp_element.blocks {
//                 assert_eq!(b.values, vec![KEY_NOT_EXISTS; b.config.capacity])
//             }
//         }
//     }
// }
