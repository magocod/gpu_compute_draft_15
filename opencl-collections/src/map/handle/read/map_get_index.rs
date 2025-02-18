use crate::config::{ClTypeTrait, DEBUG_MODE};
use crate::error::OpenClResult;
use crate::map::config::{check_local_work_size, check_max_find_work_size, DEFAULT_MAP_KEY_LENGTH};
use crate::map::handle::tmp::TmpMultiple;
use crate::map::handle::{EntryIndices, MapBlockSizes, MapHandle, MapKeys};
use crate::map::kernel::name::{GET_TMP_FOR_MAP_GET_INDEX, MAP_GET_INDEX};
use crate::utils::ensure_vec_size;
use opencl::opencl_sys::bindings::cl_uint;
use opencl::wrapper::system::OpenclCommonOperation;

impl<T: ClTypeTrait, D: OpenclCommonOperation> MapHandle<T, D> {
    // For now, only searches for a few simultaneous elements
    pub fn map_get_index(&self, keys: &MapKeys<T>) -> OpenClResult<(EntryIndices, MapBlockSizes)> {
        check_max_find_work_size(keys.len());

        let global_work_size = keys.len();
        let local_work_size = check_local_work_size(global_work_size);

        let key_input_capacity = DEFAULT_MAP_KEY_LENGTH * global_work_size;

        // CMQ_COMPARE_KEY
        // CMQ_CONFIRM_SEARCH
        // CMQ_CONTINUE_SEARCH
        let enqueue_kernel_output_capacity = global_work_size * 3;

        let mut keys_input = Vec::with_capacity(key_input_capacity);

        for key in keys {
            let mut v = ensure_vec_size(key, DEFAULT_MAP_KEY_LENGTH);
            keys_input.append(&mut v);
        }

        let keys_input_buf = self.system.blocking_prepare_input_buffer(&keys_input)?;

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

        let mut kernel = self.system.create_kernel(MAP_GET_INDEX)?;

        unsafe {
            kernel.set_arg(&q0)?;
            kernel.set_arg(&map_id)?;

            kernel.set_arg(&keys_input_buf.get_cl_mem())?;

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
            println!("indices {indices_output:?}");
            println!("blocks  {block_output:?}");
        }

        // assert enqueue kernels
        self.system.assert_device_enqueue_kernel(
            enqueue_kernel_output_capacity,
            enqueue_kernel_output_buf,
            &[],
        )?;

        Ok((indices_output, block_output))
    }

    pub fn get_tmp_match_indices_for_map_get_index(
        &self,
        elements: usize,
    ) -> OpenClResult<TmpMultiple<T>> {
        self.get_tmp_multiple(GET_TMP_FOR_MAP_GET_INDEX, elements)
    }
}

#[cfg(test)]
mod tests_map_get_index {
    use crate::map::config::{MapSrc, DEFAULT_MAP_KEY_LENGTH, MAX_FIND_WORK_SIZE};
    use crate::map::handle::test_utils::generate_arc_opencl_block_default;
    use crate::map::handle::{MapHandle, KEY_NOT_EXISTS};
    use crate::test_utils::TestMatrix;
    use crate::utils::{BYTE_256, BYTE_512, KB};
    use opencl::opencl_sys::bindings::cl_int;

    const TOTAL_MAPS: usize = 2;
    const MAP_CAPACITY: usize = 16;

    #[test]
    fn no_index_found() {
        let mut map_src: MapSrc<i32> = MapSrc::new(TOTAL_MAPS);
        map_src.add(BYTE_256, MAP_CAPACITY);
        map_src.add(BYTE_512, MAP_CAPACITY);
        map_src.add(KB, MAP_CAPACITY);

        map_src.add_map_get_index_program_src(MAX_FIND_WORK_SIZE);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix = TestMatrix::new(MAP_CAPACITY, DEFAULT_MAP_KEY_LENGTH, BYTE_256, 70, 80);

        let test_matrix_256 = TestMatrix::new(
            MAP_CAPACITY,
            DEFAULT_MAP_KEY_LENGTH,
            BYTE_256,
            1 + MAP_CAPACITY as cl_int,
            100,
        );
        test_matrix_256.put(&m, BYTE_256);

        let (read_indices, blocks) = m.map_get_index(&test_matrix.keys).unwrap();

        let tmp_arr = m
            .get_tmp_match_indices_for_map_get_index(test_matrix.keys.len())
            .unwrap();

        assert_eq!(read_indices, vec![KEY_NOT_EXISTS; MAP_CAPACITY]);
        assert_eq!(blocks, vec![KB as cl_int; MAP_CAPACITY]);

        for tmp_element in tmp_arr {
            for b in tmp_element.blocks {
                assert_eq!(b.values, vec![-3; b.config.capacity]);
            }
        }
    }

    #[test]
    fn index_found() {
        let mut map_src: MapSrc<i32> = MapSrc::new(TOTAL_MAPS);
        map_src.add(BYTE_256, MAP_CAPACITY);
        map_src.add(BYTE_512, MAP_CAPACITY);

        map_src.add_map_get_index_program_src(MAX_FIND_WORK_SIZE);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let mut input_keys: Vec<Vec<cl_int>> = vec![];

        let test_matrix_256 =
            TestMatrix::new(MAP_CAPACITY, DEFAULT_MAP_KEY_LENGTH, BYTE_256, 1, 10);
        test_matrix_256.put(&m, BYTE_256);

        input_keys.append(&mut test_matrix_256.keys.clone());

        let test_matrix_512 = TestMatrix::new(
            MAP_CAPACITY,
            DEFAULT_MAP_KEY_LENGTH,
            BYTE_512,
            1 + MAP_CAPACITY as cl_int,
            10 + MAP_CAPACITY as cl_int,
        );
        test_matrix_512.put(&m, BYTE_512);

        input_keys.append(&mut test_matrix_512.keys.clone());

        let (indices, blocks) = m.map_get_index(&input_keys).unwrap();

        let tmp_arr = m
            .get_tmp_match_indices_for_map_get_index(input_keys.len())
            .unwrap();

        let mut indices_expected: Vec<cl_int> = (0..MAP_CAPACITY as cl_int).collect();
        indices_expected.append(&mut (0..MAP_CAPACITY as cl_int).collect());

        let mut blocks_expected = vec![BYTE_256 as cl_int; MAP_CAPACITY];
        blocks_expected.append(&mut vec![BYTE_512 as cl_int; MAP_CAPACITY]);

        assert_eq!(indices, indices_expected);
        assert_eq!(blocks, blocks_expected);

        for tmp_element in tmp_arr {
            for _b in tmp_element.blocks {
                // TODO assert_eq!(b.inner, vec![-3; b.config.MAP_CAPACITY]);
            }
        }
    }

    #[test]
    fn multiple_matches() {
        let mut map_src: MapSrc<i32> = MapSrc::new(TOTAL_MAPS);
        map_src.add(BYTE_256, MAP_CAPACITY);
        map_src.add(BYTE_512, MAP_CAPACITY);
        map_src.add(KB, MAP_CAPACITY);

        map_src.add_map_get_index_program_src(MAX_FIND_WORK_SIZE);

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
            DEFAULT_MAP_KEY_LENGTH,
            BYTE_512,
            1 + MAP_CAPACITY as cl_int,
            10 + MAP_CAPACITY as cl_int,
        );

        input_keys.append(&mut test_matrix_512.keys.clone());

        let (indices, blocks) = m.map_get_index(&input_keys).unwrap();

        let tmp_arr = m
            .get_tmp_match_indices_for_map_get_index(input_keys.len())
            .unwrap();

        let mut indices_expected: Vec<cl_int> = (0..MAP_CAPACITY as cl_int).collect();
        indices_expected.append(&mut vec![-3; MAP_CAPACITY]);

        let mut blocks_expected = vec![BYTE_256 as cl_int; MAP_CAPACITY];
        blocks_expected.append(&mut vec![KB as cl_int; MAP_CAPACITY]);

        assert_eq!(indices, indices_expected);
        assert_eq!(blocks, blocks_expected);

        for tmp_element in tmp_arr {
            for _b in tmp_element.blocks {
                // TODO assert_eq!(b.inner, vec![-3; b.config.MAP_CAPACITY]);
            }
        }
    }

    // slow compilation
    #[test]
    fn big_map() {
        let input_len = 2;
        let total_blocks = 10;

        let mut map_src: MapSrc<i32> = MapSrc::new(TOTAL_MAPS);

        for i in 0..total_blocks {
            map_src.add(BYTE_256 * (i + 1), MAP_CAPACITY);
        }

        map_src.add_map_get_index_program_src(MAX_FIND_WORK_SIZE);

        // println!("{config:#?}");

        let last_config = map_src.get_configs().last().unwrap();

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix = TestMatrix::new(
            input_len,
            DEFAULT_MAP_KEY_LENGTH,
            last_config.value_len,
            1,
            10,
        );
        test_matrix.put(&m, last_config.value_len);

        let (indices, blocks) = m.map_get_index(&test_matrix.keys).unwrap();

        let tmp_arr = m
            .get_tmp_match_indices_for_map_get_index(test_matrix.keys.len())
            .unwrap();

        let expected_indices: Vec<cl_int> = (0..input_len as cl_int).collect();

        assert_eq!(indices, expected_indices);
        assert_eq!(blocks, vec![(BYTE_256 * total_blocks) as cl_int; input_len]);

        for tmp_element in tmp_arr {
            for _b in tmp_element.blocks {
                // TODO assert_eq!(b.inner, vec![-3; b.config.MAP_CAPACITY]);
            }
        }
    }
}

// fatal errors
// #[cfg(test)]
// mod tests_issues_map_get_index {
//     use crate::map::config::{MapSrc, DEFAULT_MAP_KEY_LENGTH, MAX_FIND_WORK_SIZE};
//     use crate::map::handle::test_utils::generate_arc_opencl_block_default;
//     use crate::map::handle::Map;
//     use crate::test_utils::TestMatrix;
//     use crate::utils::BYTE_256;
//     use opencl::opencl_sys::bindings::cl_int;
//
//     const TOTAL_MAPS: usize = 2;
//     const MAP_CAPACITY: usize = 1024;
//
//     fn map_get_index(input_len: usize, total_blocks: usize) {
//         let mut map_src: MapSrc<i32> = MapSrc::new(TOTAL_MAPS);
//
//         for i in 0..total_blocks {
//             map_src.add(BYTE_256 * (i + 1), MAP_CAPACITY);
//         }
//
//         println!("{map_src:#?}");
//
//         map_src.add_map_get_index_program_src(MAX_FIND_WORK_SIZE);
//
//         let last_config = map_src.get_configs().last().unwrap();
//
//         let system = generate_arc_opencl_block_default(&map_src);
//         let m = Map::new(0, &map_src, system);
//
//         let test_matrix = TestMatrix::new(
//             input_len,
//             DEFAULT_MAP_KEY_LENGTH,
//             last_config.value_len,
//             1,
//             10,
//         );
//         test_matrix.put(&m, last_config.value_len);
//
//         let (indices, blocks) = m.map_get_index(&test_matrix.keys).unwrap();
//
//         let tmp_arr = m
//             .get_tmp_match_indices_for_map_get_index(test_matrix.keys.len())
//             .unwrap();
//
//         let expected_indices: Vec<cl_int> = (0..input_len as cl_int).collect();
//
//         assert_eq!(indices, expected_indices);
//         assert_eq!(blocks, vec![(BYTE_256 * total_blocks) as cl_int; input_len]);
//
//         for tmp_element in tmp_arr {
//             for _b in tmp_element.blocks {
//                 // TODO assert_eq!(b.inner, vec![-3; b.config.MAP_CAPACITY]);
//             }
//         }
//     }
//
//     #[test]
//     fn case_1_success() {
//         let input_len = 2;
//         let total_blocks = 10;
//         map_get_index(input_len, total_blocks)
//     }
//
//     #[test]
//     fn case_2_success() {
//         let input_len = 4;
//         let total_blocks = 10;
//         map_get_index(input_len, total_blocks)
//     }
//
//     #[test]
//     fn case_3_success() {
//         let input_len = 8;
//         let total_blocks = 8;
//         map_get_index(input_len, total_blocks)
//     }
//
//     #[test]
//     fn case_4_success() {
//         let input_len = 16;
//         let total_blocks = 4;
//         map_get_index(input_len, total_blocks)
//     }
//
//     // fatal error, rename test to error
//     #[test]
//     fn case_5_success() {
//         let input_len = 2;
//         let total_blocks = 32;
//         map_get_index(input_len, total_blocks)
//     }
//
//     // fatal error or pass
//     #[test]
//     fn case_1_error() {
//         let input_len = 32;
//         let total_blocks = 4;
//         map_get_index(input_len, total_blocks)
//     }
//
//     // fatal error or pass
//     #[test]
//     fn case_2_error() {
//         let input_len = 8;
//         let total_blocks = 16;
//         map_get_index(input_len, total_blocks)
//     }
//
//     // fatal error
//     #[test]
//     fn case_3_error() {
//         let input_len = 4;
//         let total_blocks = 32;
//         map_get_index(input_len, total_blocks)
//     }
// }
