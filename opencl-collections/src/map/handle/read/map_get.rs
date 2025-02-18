use crate::config::{ClTypeTrait, DEBUG_MODE};
use crate::error::OpenClResult;
use crate::map::config::{check_local_work_size, check_max_find_work_size, DEFAULT_MAP_KEY_LENGTH};
use crate::map::handle::tmp::TmpMultiple;
use crate::map::handle::{EntryIndices, MapBlockSizes, MapHandle, MapKeys, MapValues};
use crate::map::kernel::name::{GET_TMP_FOR_MAP_GET, MAP_GET};
use crate::utils::{ensure_vec_size, KB};
use opencl::opencl_sys::bindings::cl_uint;
use opencl::wrapper::system::OpenclCommonOperation;

impl<T: ClTypeTrait, D: OpenclCommonOperation> MapHandle<T, D> {
    // For now, only searches for a few simultaneous elements
    pub fn map_get(
        &self,
        keys: &MapKeys<T>,
    ) -> OpenClResult<(EntryIndices, MapBlockSizes, MapValues<T>)> {
        check_max_find_work_size(keys.len());

        let map_config = &self.map_src;

        let global_work_size = keys.len();
        let local_work_size = check_local_work_size(global_work_size);

        let max_value_len = map_config.get_max_value_len();

        let key_input_capacity = DEFAULT_MAP_KEY_LENGTH * global_work_size;
        let value_output_capacity = max_value_len * global_work_size;

        // CMQ_COMPARE_KEY_DEF_2
        // CMQ_CONFIRM_SEARCH_DEF_2
        // CMQ_CONTINUE_SEARCH_DEF_2
        // CMQ_MAP_GET_VALUE
        let enqueue_kernel_output_capacity = global_work_size * 4;

        let mut keys_input = Vec::with_capacity(key_input_capacity);

        for key in keys {
            let mut v = ensure_vec_size(key, DEFAULT_MAP_KEY_LENGTH);
            keys_input.append(&mut v);
        }

        let keys_input_buf = self.system.blocking_prepare_input_buffer(&keys_input)?;

        let values_output_buf = self.system.create_output_buffer(value_output_capacity)?;
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

        let mut kernel = self.system.create_kernel(MAP_GET)?;

        unsafe {
            kernel.set_arg(&q0)?;
            kernel.set_arg(&map_id)?;

            kernel.set_arg(&keys_input_buf.get_cl_mem())?;

            kernel.set_arg(&values_output_buf.get_cl_mem())?;
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

        let values_output = self.system.blocking_enqueue_read_buffer(
            value_output_capacity,
            &values_output_buf,
            &[],
        )?;

        // FIXME map_get - apply the correct block length for each chunk
        let values: Vec<Vec<_>> = values_output
            .chunks(max_value_len)
            .map(|x| x.to_vec())
            .collect();

        if DEBUG_MODE {
            println!("indices {indices_output:?}");
            println!("blocks  {block_output:?}");

            if max_value_len < KB {
                for (i, b) in values.iter().enumerate() {
                    println!("i: {}, len {}, value {:?}", i, b.len(), b);
                }
            }
        }

        // assert enqueue kernels
        self.system.assert_device_enqueue_kernel(
            enqueue_kernel_output_capacity,
            enqueue_kernel_output_buf,
            &[],
        )?;

        Ok((indices_output, block_output, values))
    }

    pub fn get_tmp_for_map_get(&self, elements: usize) -> OpenClResult<TmpMultiple<T>> {
        self.get_tmp_multiple(GET_TMP_FOR_MAP_GET, elements)
    }
}

#[cfg(test)]
mod tests_map_get {
    use crate::config::ClTypeDefault;
    use crate::map::config::{MapSrc, DEFAULT_MAP_KEY_LENGTH, MAX_FIND_WORK_SIZE};
    use crate::map::handle::test_utils::generate_arc_opencl_block_default;
    use crate::map::handle::{MapHandle, KEY_NOT_EXISTS};
    use crate::test_utils::TestMatrix;
    use crate::utils::{ensure_vec_size, BYTE_256, BYTE_512, KB};
    use opencl::opencl_sys::bindings::cl_int;

    const TOTAL_MAPS: usize = 2;
    const MAP_CAPACITY: usize = 16;

    #[test]
    fn no_index_found() {
        let mut map_src: MapSrc<i32> = MapSrc::new(TOTAL_MAPS);
        map_src.add(BYTE_256, MAP_CAPACITY);
        map_src.add(BYTE_512, MAP_CAPACITY);
        map_src.add(KB, MAP_CAPACITY);

        map_src.add_map_get_program_src(MAX_FIND_WORK_SIZE);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix = TestMatrix::new(MAP_CAPACITY, DEFAULT_MAP_KEY_LENGTH, BYTE_256, 70, 80);

        let (indices, blocks, values) = m.map_get(&test_matrix.keys).unwrap();

        let tmp_arr = m.get_tmp_for_map_get(test_matrix.keys.len()).unwrap();

        assert_eq!(
            values,
            vec![vec![i32::cl_default(); KB]; test_matrix.keys.len()]
        );

        assert_eq!(indices, vec![KEY_NOT_EXISTS; MAP_CAPACITY]);
        assert_eq!(blocks, vec![KB as cl_int; MAP_CAPACITY]);

        for tmp_element in tmp_arr {
            for b in tmp_element.blocks {
                assert_eq!(b.values, vec![-3; b.config.capacity]);
            }
        }
    }

    #[test]
    fn index_found() {
        let mut map_src = MapSrc::new(TOTAL_MAPS);
        map_src.add(BYTE_256, MAP_CAPACITY);
        map_src.add(BYTE_512, MAP_CAPACITY);

        map_src.add_map_get_program_src(MAX_FIND_WORK_SIZE);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let mut input_keys: Vec<Vec<cl_int>> = vec![];

        let test_matrix_256 =
            TestMatrix::new(MAP_CAPACITY, DEFAULT_MAP_KEY_LENGTH, BYTE_256, 1, 10);
        test_matrix_256.put(&m, BYTE_256);

        input_keys.append(&mut test_matrix_256.keys.clone());

        let test_matrix_512 = TestMatrix::new(
            MAP_CAPACITY,
            1,
            1,
            1 + MAP_CAPACITY as cl_int,
            10 + MAP_CAPACITY as cl_int,
        );
        test_matrix_512.put(&m, BYTE_512);

        input_keys.append(&mut test_matrix_512.keys.clone());

        let (indices, blocks, values) = m.map_get(&input_keys).unwrap();

        let tmp_arr = m.get_tmp_for_map_get(input_keys.len()).unwrap();

        let mut indices_expected: Vec<cl_int> = (0..MAP_CAPACITY as cl_int).collect();
        indices_expected.append(&mut (0..MAP_CAPACITY as cl_int).collect());

        let mut blocks_expected = vec![BYTE_256 as cl_int; MAP_CAPACITY];
        blocks_expected.append(&mut vec![BYTE_512 as cl_int; MAP_CAPACITY]);

        assert_eq!(indices, indices_expected);
        assert_eq!(blocks, blocks_expected);

        for (i, value) in values[..MAP_CAPACITY].iter().enumerate() {
            assert_eq!(
                value,
                &ensure_vec_size(&test_matrix_256.values[i], BYTE_512)
            );
        }

        for (i, value) in values[MAP_CAPACITY..].iter().enumerate() {
            assert_eq!(
                value,
                &ensure_vec_size(&test_matrix_512.values[i], BYTE_512)
            );
        }

        for tmp_element in tmp_arr {
            for _b in tmp_element.blocks {
                // TODO assert_eq!(b.inner, vec![-3; b.config.MAP_CAPACITY]);
            }
        }
    }

    #[test]
    fn multiple_matches() {
        let mut map_src = MapSrc::new(TOTAL_MAPS);
        map_src.add(BYTE_256, MAP_CAPACITY);
        map_src.add(BYTE_512, MAP_CAPACITY);

        map_src.add_map_get_program_src(MAX_FIND_WORK_SIZE);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let mut input_keys: Vec<Vec<cl_int>> = vec![];

        let test_matrix_256 =
            TestMatrix::new(MAP_CAPACITY, DEFAULT_MAP_KEY_LENGTH, BYTE_256, 1, 10);
        test_matrix_256.put(&m, BYTE_256);
        test_matrix_256.put(&m, BYTE_512);

        input_keys.append(&mut test_matrix_256.keys.clone());

        let test_matrix_512 = TestMatrix::new(
            MAP_CAPACITY,
            1,
            1,
            1 + MAP_CAPACITY as cl_int,
            10 + MAP_CAPACITY as cl_int,
        );

        input_keys.append(&mut test_matrix_512.keys.clone());

        let (indices, blocks, values) = m.map_get(&input_keys).unwrap();

        let tmp_arr = m.get_tmp_for_map_get(input_keys.len()).unwrap();

        let mut indices_expected: Vec<cl_int> = (0..MAP_CAPACITY as cl_int).collect();
        indices_expected.append(&mut vec![-3; MAP_CAPACITY]);

        let mut blocks_expected = vec![BYTE_256 as cl_int; MAP_CAPACITY];
        blocks_expected.append(&mut vec![BYTE_512 as cl_int; MAP_CAPACITY]);

        assert_eq!(indices, indices_expected);
        assert_eq!(blocks, blocks_expected);

        for (i, value) in values[..MAP_CAPACITY].iter().enumerate() {
            assert_eq!(
                value,
                &ensure_vec_size(&test_matrix_256.values[i], BYTE_512)
            );
        }

        assert_eq!(
            values[MAP_CAPACITY..],
            vec![vec![i32::cl_default(); BYTE_512]; MAP_CAPACITY]
        );

        for tmp_element in tmp_arr {
            for _b in tmp_element.blocks {
                // TODO assert_eq!(b.inner, vec![-3; b.config.MAP_CAPACITY]);
            }
        }
    }
}

// same issues -> tests_issues_map_get_index
