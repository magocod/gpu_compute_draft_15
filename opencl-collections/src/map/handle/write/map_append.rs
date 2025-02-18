use crate::config::{ClTypeTrait, DEBUG_MODE};
use crate::error::OpenClResult;
use crate::map::config::{check_local_work_size, DEFAULT_MAP_KEY_LENGTH};
use crate::map::handle::read::map_get_empty_key::PipeIndices;
use crate::map::handle::tmp::TmpMultiple;
use crate::map::handle::{EntryIndices, MapBlockSizes, MapHandle, MapKeys, MapValues};
use crate::map::kernel::name::{
    get_map_kernel_name, GET_TMP_FOR_MAP_APPEND, MAP_APPEND, MAP_APPEND_FOR_BLOCK,
};
use crate::utils::{ensure_vec_size, from_buf_usize_to_vec_i32};
use opencl::opencl_sys::bindings::{cl_int, cl_uint};
use opencl::wrapper::system::OpenclCommonOperation;

#[derive(Debug, PartialEq)]
pub struct AppendOp {
    pub code: cl_int,
    pub size: cl_int,
}

impl<T: ClTypeTrait, D: OpenclCommonOperation> MapHandle<T, D> {
    pub fn map_append_for_block(
        &self,
        map_value_len: usize,
        values: &MapValues<T>,
        indices: &[usize],
    ) -> OpenClResult<Vec<AppendOp>> {
        let _ = &self.map_src.get_config_by_value_len(map_value_len)?;

        if values.len() != indices.len() {
            panic!("TODO error values & indices len");
        }

        let global_work_size = values.len();
        let local_work_size = check_local_work_size(global_work_size);

        // RESULT_APPEND index 0
        // RESULT_LAST_INDEX index 1
        let result_capacity = 2;

        let result_output_capacity = global_work_size * result_capacity;
        let value_input_capacity = map_value_len * global_work_size;
        let enqueue_kernel_output_capacity = global_work_size;

        let mut values_input = Vec::with_capacity(value_input_capacity);
        let mut values_lens_input = Vec::with_capacity(global_work_size);

        let indices_input = from_buf_usize_to_vec_i32(indices);

        for value in values {
            values_lens_input.push(value.len() as cl_int);

            let mut v = ensure_vec_size(value, map_value_len);
            values_input.append(&mut v);
        }

        let values_input_buf = self.system.blocking_prepare_input_buffer(&values_input)?;
        let values_lens_input_buf = self
            .system
            .blocking_prepare_input_buffer(&values_lens_input)?;
        let indices_input_buf = self.system.blocking_prepare_input_buffer(&indices_input)?;

        let result_output_buf = self.system.create_output_buffer(result_output_capacity)?;
        let enqueue_kernel_output_buf = self
            .system
            .create_output_buffer(enqueue_kernel_output_capacity)?;

        let map_id = self.map_id as cl_uint;

        let q0 = self
            .system
            .get_device_command_queue_0()
            .get_cl_command_queue();

        let kernel_name = get_map_kernel_name(MAP_APPEND_FOR_BLOCK, map_value_len);
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&q0)?;
            kernel.set_arg(&map_id)?;

            kernel.set_arg(&values_input_buf.get_cl_mem())?;
            kernel.set_arg(&values_lens_input_buf.get_cl_mem())?;
            kernel.set_arg(&indices_input_buf.get_cl_mem())?;

            kernel.set_arg(&result_output_buf.get_cl_mem())?;
            kernel.set_arg(&enqueue_kernel_output_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        let result_output = self.system.blocking_enqueue_read_buffer(
            result_output_capacity,
            &result_output_buf,
            &[],
        )?;

        let result_output_chunks: Vec<AppendOp> = result_output
            .chunks(result_capacity)
            .map(|x| AppendOp {
                code: x[0],
                size: x[1],
            })
            .collect();

        if DEBUG_MODE {
            println!("append output");
            for chunk in result_output_chunks.iter() {
                println!(" {:?}", chunk);
            }
        }

        self.system.assert_device_enqueue_kernel(
            enqueue_kernel_output_capacity,
            enqueue_kernel_output_buf,
            &[],
        )?;

        Ok(result_output_chunks)
    }

    // For now, it is only reliable when storing few items
    pub fn map_append(
        &self,
        keys: &MapKeys<T>,
        values: &MapValues<T>,
        pipes: &[PipeIndices],
    ) -> OpenClResult<(EntryIndices, MapBlockSizes)> {
        let map_config = &self.map_src;

        let global_work_size = keys.len();
        if global_work_size != values.len() {
            panic!("TODO handle error input lens")
        }
        let local_work_size = check_local_work_size(global_work_size);

        let max_value_len = map_config.get_max_value_len();

        // CMQ_COMPARE_KEY_IN_BLOCKS
        // CMQ_CONFIRM_MAP_APPEND
        // CMQ_MAP_APPEND
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

        let mut kernel = self.system.create_kernel(MAP_APPEND)?;

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

    pub fn get_tmp_match_indices_for_map_append(
        &self,
        elements: usize,
    ) -> OpenClResult<TmpMultiple<T>> {
        self.get_tmp_multiple(GET_TMP_FOR_MAP_APPEND, elements)
    }
}

#[cfg(test)]
mod tests_map_append_for_block {
    use super::*;
    use crate::config::ClTypeDefault;
    use crate::map::config::MapSrc;
    use crate::map::handle::test_utils::generate_arc_opencl_block_default;
    use crate::map::handle::{CANNOT_APPEND_VALUE, MAP_VALUE_FULL};
    use crate::test_utils::TestMatrix;
    use crate::utils::BYTE_256;

    const TOTAL_MAPS: usize = 2;
    const MAP_CAPACITY: usize = 16;
    const CONFIG_SIZE: usize = BYTE_256;

    #[test]
    fn simple_cases() {
        let mut map_src: MapSrc<i32> = MapSrc::new(TOTAL_MAPS);
        map_src.add(CONFIG_SIZE, MAP_CAPACITY);

        map_src.add_map_append_for_block_program_src();

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix = TestMatrix::new(6, DEFAULT_MAP_KEY_LENGTH, BYTE_256, 1, 2);

        let input_values = vec![
            // empty
            vec![91; BYTE_256],
            // with available space (half)
            vec![92; BYTE_256 / 2],
            // with available space (1)
            vec![100; 1],
            // full
            vec![93; BYTE_256],
            // without enough space
            vec![94; (BYTE_256 / 2) + 1],
            vec![95; 2],
        ];

        let current_values = vec![
            // empty
            vec![],
            // with available space (half)
            vec![22; BYTE_256 / 2],
            // with available space (1)
            vec![33; BYTE_256 - 1],
            // full
            vec![44; BYTE_256],
            // without enough space
            vec![55; BYTE_256 / 2],
            vec![66; BYTE_256 - 1],
        ];

        let indices = [
            0, // empty
            1, // with available space (half)
            5, // with available space (1)
            6, // full
            9, 10, // without enough space
        ];

        m.put_with_index(BYTE_256, &test_matrix.keys, &current_values, &indices)
            .unwrap();

        let results = m
            .map_append_for_block(CONFIG_SIZE, &input_values, &indices)
            .unwrap();

        // empty
        assert_eq!(results[0], AppendOp { code: 0, size: 0 });
        // with available space (half)
        assert_eq!(
            results[1],
            AppendOp {
                code: 0,
                size: (BYTE_256 / 2) as cl_int
            }
        );
        // with available space (1)
        assert_eq!(
            results[2],
            AppendOp {
                code: 0,
                size: (BYTE_256 - 1) as cl_int
            }
        );
        // full
        assert_eq!(
            results[3],
            AppendOp {
                code: MAP_VALUE_FULL,
                size: BYTE_256 as cl_int
            }
        );
        // without enough space
        assert_eq!(
            results[4],
            AppendOp {
                code: CANNOT_APPEND_VALUE,
                size: (BYTE_256 / 2) as cl_int
            }
        );
        assert_eq!(
            results[5],
            AppendOp {
                code: CANNOT_APPEND_VALUE,
                size: (BYTE_256 - 1) as cl_int
            }
        );

        let pairs = m.read(CONFIG_SIZE).unwrap();

        // empty
        assert_eq!(pairs[0].value, input_values[0]);

        // with available space (half)
        let mut expected = current_values[1].clone();
        expected.append(&mut input_values[1].clone());
        assert_eq!(pairs[1].value, expected);

        // with available space (1)
        let mut expected = current_values[2].clone();
        expected.append(&mut input_values[2].clone());
        assert_eq!(pairs[5].value, expected);

        // full
        assert_eq!(pairs[6].value, current_values[3]);

        // without enough space
        assert_eq!(
            pairs[9].value,
            ensure_vec_size(&current_values[4], BYTE_256)
        );
        assert_eq!(
            pairs[10].value,
            ensure_vec_size(&current_values[5], BYTE_256)
        );

        for (i, pair) in pairs.into_iter().enumerate() {
            if !indices.contains(&i) {
                assert_eq!(pair.key, vec![i32::cl_default(); DEFAULT_MAP_KEY_LENGTH]);
                assert_eq!(pair.value, vec![i32::cl_default(); CONFIG_SIZE]);
            }
        }
    }
}

#[cfg(test)]
mod tests_map_append {
    use crate::config::ClTypeDefault;
    use crate::map::config::{MapSrc, DEFAULT_MAP_KEY_LENGTH, MAX_FIND_WORK_SIZE};
    use crate::map::handle::test_utils::generate_arc_opencl_block_default;
    use crate::map::handle::MapHandle;
    use crate::test_utils::TestMatrix;
    use crate::utils::{ensure_vec_size, BYTE_256, BYTE_512, KB};

    const TOTAL_MAPS: usize = 2;

    // FIXME error relocation
    #[test]
    fn with_space_to_append() {
        let mut map_src: MapSrc<i32> = MapSrc::new(TOTAL_MAPS);
        map_src.add(BYTE_256, 16);
        map_src.add(BYTE_512, 16);
        map_src.add(KB, 16);

        map_src.add_map_copy_program_src();
        map_src.add_map_append_program_src(MAX_FIND_WORK_SIZE);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix = TestMatrix::new(6, DEFAULT_MAP_KEY_LENGTH, BYTE_256, 1, 2);

        let input_values = vec![
            // empty
            vec![91; BYTE_256],
            // with available space (half)
            vec![92; BYTE_256 / 2],
            // with available space (1)
            vec![100; 1],
            // relocation // FIXME error relocation
            vec![93; BYTE_256],
            // relocation - large size
            vec![94; BYTE_256 + 1],
            // relocation - empty // FIXME error relocation
            vec![95; BYTE_256 + 1],
        ];

        let current_values = vec![
            // empty
            vec![],
            // with available space (half)
            vec![22; BYTE_256 / 2],
            // with available space (1)
            vec![33; BYTE_256 - 1],
            // relocation
            vec![44; BYTE_256],
            // relocation - large size
            vec![55; BYTE_256],
            // relocation - empty
            vec![],
        ];

        m.put_with_index(
            BYTE_256,
            &test_matrix.keys,
            &current_values,
            &[
                0, 1, 5,  // 256
                6,  // 256 - > 512
                9,  // 256 -> 1024
                10, // 256 -> 512
            ],
        )
        .unwrap();

        let pipes = m.get_empty_keys_pipes().unwrap();

        let (indices, blocks) = m
            .map_append(&test_matrix.keys, &input_values, &pipes)
            .unwrap();

        let tmp_arr = m
            .get_tmp_match_indices_for_map_append(test_matrix.keys.len())
            .unwrap();

        // TODO assert_eq!(indices, ---);
        assert_eq!(blocks, vec![256, 256, 256, 512, 1024, 512]);

        let pairs = m.read(BYTE_256).unwrap();

        assert_eq!(pairs[0].key, test_matrix.keys[0]);
        assert_eq!(pairs[0].value, input_values[0]);

        assert_eq!(pairs[1].key, test_matrix.keys[1]);
        let mut expected_vec = current_values[1].clone();
        expected_vec.append(&mut input_values[1].clone());
        assert_eq!(pairs[1].value, expected_vec);

        assert_eq!(pairs[5].key, test_matrix.keys[2]);
        let mut expected_vec = current_values[2].clone();
        expected_vec.append(&mut input_values[2].clone());
        assert_eq!(pairs[5].value, expected_vec);

        for (i, pair) in pairs.into_iter().enumerate() {
            if ![0, 1, 5].contains(&i) {
                assert_eq!(pair.key, vec![i32::cl_default(); DEFAULT_MAP_KEY_LENGTH]);
                assert_eq!(pair.value, vec![i32::cl_default(); BYTE_256]);
            }
        }

        let pairs = m.read(BYTE_512).unwrap();

        let index = indices[3] as usize;

        assert_eq!(pairs[index].key, test_matrix.keys[3]);
        let mut expected_vec = current_values[3].clone();
        expected_vec.append(&mut input_values[3].clone());
        assert_eq!(pairs[index].value, ensure_vec_size(&expected_vec, BYTE_512));

        let index = indices[5] as usize;
        assert_eq!(pairs[index].key, test_matrix.keys[5]);
        let mut expected_vec = current_values[5].clone();
        expected_vec.append(&mut input_values[5].clone());
        assert_eq!(pairs[index].value, ensure_vec_size(&expected_vec, BYTE_512));

        for (i, pair) in pairs.into_iter().enumerate() {
            if ![0, 1].contains(&i) {
                assert_eq!(pair.key, vec![i32::cl_default(); DEFAULT_MAP_KEY_LENGTH]);
                assert_eq!(pair.value, vec![i32::cl_default(); BYTE_512]);
            }
        }

        let matrix = m.read(KB).unwrap();

        assert_eq!(matrix[0].key, test_matrix.keys[4]);
        let mut expected_vec = current_values[4].clone();
        expected_vec.append(&mut input_values[4].clone());
        assert_eq!(matrix[0].value, ensure_vec_size(&expected_vec, KB));

        for (i, pair) in matrix.into_iter().enumerate() {
            if ![0].contains(&i) {
                assert_eq!(pair.key, vec![i32::cl_default(); DEFAULT_MAP_KEY_LENGTH]);
                assert_eq!(pair.value, vec![i32::cl_default(); KB]);
            }
        }

        for tmp_element in tmp_arr {
            for _b in tmp_element.blocks {
                // TODO assert tmp_array
                // assert_eq!(b.inner, vec![-3; b.config.MAP_CAPACITY])
            }
        }
    }

    #[test]
    fn no_space_to_append() {
        let mut map_src: MapSrc<i32> = MapSrc::new(TOTAL_MAPS);
        map_src.add(BYTE_256, 16);
        map_src.add(BYTE_512, 16);
        map_src.add(KB, 16);

        map_src.add_map_copy_program_src();
        map_src.add_map_append_program_src(MAX_FIND_WORK_SIZE);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix_512 = TestMatrix::new(16, DEFAULT_MAP_KEY_LENGTH, BYTE_512, 1, 2);
        test_matrix_512.put(&m, BYTE_512);

        let test_matrix_1024 = TestMatrix::new(16, DEFAULT_MAP_KEY_LENGTH, KB, 100, 200);
        test_matrix_1024.put(&m, KB);

        let values = vec![
            // not enough space
            vec![96; 2],
            // full buf and no another available block
            vec![97; 1],
            // no capacity to store
            vec![95; KB + 1],
            // blocks without space
            vec![98; BYTE_256 + 1],
        ];

        let current_values = vec![
            // not enough space
            vec![66; BYTE_512 - 1],
            // full buf and no another available block
            vec![77; BYTE_512],
            // no capacity to store
            // ...
            // blocks without space
            // ...
        ];

        m.put_with_index(
            BYTE_512,
            &test_matrix_512.keys[0..2].to_vec(),
            &current_values,
            &[0, 1],
        )
        .unwrap();

        let pipes = m.get_empty_keys_pipes().unwrap();

        let input_keys = test_matrix_512.keys[0..4].to_vec();

        let (indices, blocks) = m.map_append(&input_keys, &values, &pipes).unwrap();

        let tmp_arr = m
            .get_tmp_match_indices_for_map_append(input_keys.len())
            .unwrap();

        assert_eq!(indices, vec![-1, -1, -1, -1]);
        assert_eq!(blocks, vec![0, 0, 0, 0]);

        let matrix = m.read(BYTE_256).unwrap();

        for pair in matrix {
            assert_eq!(pair.key, vec![i32::cl_default(); DEFAULT_MAP_KEY_LENGTH]);
            assert_eq!(pair.value, vec![i32::cl_default(); BYTE_256]);
        }

        let matrix = m.read(BYTE_512).unwrap();

        assert_eq!(matrix[0].key, test_matrix_512.keys[0]);
        assert_eq!(
            matrix[0].value,
            ensure_vec_size(&current_values[0], BYTE_512)
        );

        assert_eq!(matrix[1].key, test_matrix_512.keys[1]);
        assert_eq!(
            matrix[1].value,
            ensure_vec_size(&current_values[1], BYTE_512)
        );

        for (i, pair) in matrix.into_iter().enumerate() {
            if ![0, 1].contains(&i) {
                assert_eq!(pair.key, test_matrix_512.keys[i]);
                assert_eq!(pair.value, test_matrix_512.values[i]);
            }
        }

        let matrix = m.read(KB).unwrap();

        for (i, pair) in matrix.into_iter().enumerate() {
            assert_eq!(pair.key, test_matrix_1024.keys[i]);
            assert_eq!(pair.value, test_matrix_1024.values[i]);
        }

        for tmp_element in tmp_arr {
            for _b in tmp_element.blocks {
                // TODO assert tmp_array
                // assert_eq!(b.inner, vec![-3; b.config.MAP_CAPACITY])
            }
        }
    }
}

// same issues -> tests_issues_map_insert
