use crate::config::{ClTypeTrait, DEBUG_MODE};
use crate::error::OpenClResult;
use crate::map::handle::test_utils::{assert_pair_is_empty, assert_pair_is_not_empty};
use crate::map::handle::MapHandle;
use crate::map::kernel::name::{get_map_kernel_name, MAP_REORDER_FOR_BLOCK};
use opencl::opencl_sys::bindings::{cl_int, cl_uint};
use opencl::wrapper::system::OpenclCommonOperation;

pub const REORDER_KEY_EMPTY: cl_int = -1;
pub const REORDER_KEY_UNMOVED: cl_int = -2;

impl<T: ClTypeTrait, D: OpenclCommonOperation> MapHandle<T, D> {
    pub fn reorder_for_block(
        &self,
        map_value_len: usize,
    ) -> OpenClResult<(Vec<cl_int>, Vec<cl_int>)> {
        let config = self.map_src.get_config_by_value_len(map_value_len)?;

        let global_work_size = 1;
        let local_work_size = 1;
        let indices_output_capacity = config.capacity;
        let result_output_capacity = config.capacity;

        // CMQ_CHECK_MAP_KEYS_TO_REORDER = 0;
        // CMQ_EXECUTE_REORDER_MAP = 1;
        // CMQ_CALCULATE_MAP_REORDERING = 2;
        // CMQ_CONFIRM_MAP_REORDERING = 3;
        let enqueue_kernel_output_capacity = 4;

        let indices_output_buf = self.system.create_output_buffer(indices_output_capacity)?;

        let result_output_buf = self.system.create_output_buffer(indices_output_capacity)?;

        let enqueue_kernel_output_buf = self
            .system
            .create_output_buffer(enqueue_kernel_output_capacity)?;

        let map_id = self.map_id as cl_uint;
        let enqueue_kernel_output_index: cl_int = 0;

        let q0 = self
            .system
            .get_device_command_queue_0()
            .get_cl_command_queue();

        let kernel_name = get_map_kernel_name(MAP_REORDER_FOR_BLOCK, map_value_len);
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&q0)?;
            kernel.set_arg(&map_id)?;
            kernel.set_arg(&enqueue_kernel_output_index)?;

            kernel.set_arg(&indices_output_buf.get_cl_mem())?;
            kernel.set_arg(&result_output_buf.get_cl_mem())?;
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

        let result_output = self.system.blocking_enqueue_read_buffer(
            result_output_capacity,
            &result_output_buf,
            &[],
        )?;

        if DEBUG_MODE {
            println!("indices_output {indices_output:?}");
            println!("result_output  {result_output:?}");
        }

        self.system.assert_device_enqueue_kernel(
            enqueue_kernel_output_capacity,
            enqueue_kernel_output_buf,
            &[],
        )?;

        Ok((result_output, indices_output))
    }

    pub fn reorder(&self) -> OpenClResult<()> {
        todo!()
    }
}

pub fn assert_reorder_result(result_output: &[cl_int], last_index_with_value: usize) {
    // not empty values
    assert!(result_output[0..last_index_with_value]
        .iter()
        .all(|&x| x >= 0 || x == REORDER_KEY_UNMOVED));
    // empty values
    assert!(result_output[last_index_with_value..]
        .iter()
        .all(|&x| x == REORDER_KEY_EMPTY));
}

pub fn assert_map_entries<T: ClTypeTrait, D: OpenclCommonOperation>(
    m: &MapHandle<T, D>,
    map_value_len: usize,
    last_index_with_value: usize,
) {
    let pairs = m.read(map_value_len).unwrap();

    // not empty values
    for pair in pairs[0..last_index_with_value].iter() {
        assert_pair_is_not_empty(pair);
    }

    // empty values
    for pair in pairs[last_index_with_value..].iter() {
        assert_pair_is_empty(pair, map_value_len);
    }
}

#[cfg(test)]
mod tests_reorder_for_block {
    use super::*;
    use crate::map::config::{MapSrc, DEFAULT_MAP_KEY_LENGTH};
    use crate::map::handle::test_utils::{
        assert_map_block_is_empty, assert_map_block_is_equal_to_test_matrix,
        generate_arc_opencl_block_default, DefaultTypeTrait,
    };
    use crate::map::handle::MapHandle;
    use crate::test_utils::TestMatrix;
    use crate::utils::{BYTE_256, BYTE_512, KB};

    #[test]
    fn map_is_empty() {
        let mut map_src: MapSrc<i16> = MapSrc::new(2);
        map_src.add(BYTE_256, 32);
        map_src.add(BYTE_512, 32);
        map_src.add(KB, 16);

        map_src.add_map_reorder_program_src();

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let (result_output, _indices_output) = m.reorder_for_block(BYTE_256).unwrap();

        assert_eq!(result_output, vec![REORDER_KEY_EMPTY; 32]);

        assert_map_block_is_empty(&m, BYTE_256, DefaultTypeTrait::Custom);
        assert_map_block_is_empty(&m, BYTE_512, DefaultTypeTrait::Custom);
        assert_map_block_is_empty(&m, KB, DefaultTypeTrait::Custom);
    }

    #[test]
    fn map_is_full() {
        let map_capacity = 32;

        let mut map_src: MapSrc<i16> = MapSrc::new(2);
        map_src.add(BYTE_256, map_capacity);
        map_src.add(BYTE_512, 32);
        map_src.add(KB, 32);

        map_src.add_map_reorder_program_src();

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix = TestMatrix::new(map_capacity, DEFAULT_MAP_KEY_LENGTH, BYTE_256, 100, 100);
        test_matrix.put(&m, BYTE_256);

        let (result_output, _indices_output) = m.reorder_for_block(BYTE_256).unwrap();

        assert_eq!(result_output, vec![REORDER_KEY_UNMOVED; map_capacity]);

        assert_map_block_is_equal_to_test_matrix(&m, BYTE_256, &test_matrix);
        assert_map_block_is_empty(&m, BYTE_512, DefaultTypeTrait::Custom);
        assert_map_block_is_empty(&m, KB, DefaultTypeTrait::Custom);
    }

    // TODO rename tests

    #[test]
    fn case_1() {
        let map_capacity = 32;

        let mut map_src: MapSrc<i16> = MapSrc::new(2);
        map_src.add(BYTE_256, map_capacity);
        map_src.add(BYTE_512, 32);
        map_src.add(KB, 32);

        map_src.add_map_reorder_program_src();

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let input_len = map_capacity / 2;

        let test_matrix = TestMatrix::new(input_len, DEFAULT_MAP_KEY_LENGTH, BYTE_256, 1, 11);
        let indices: Vec<usize> = (input_len..map_capacity).collect();

        m.put_with_index(BYTE_256, &test_matrix.keys, &test_matrix.values, &indices)
            .unwrap();

        let (result_output, _indices_output) = m.reorder_for_block(BYTE_256).unwrap();

        assert_reorder_result(&result_output, input_len);
        assert_map_entries(&m, BYTE_256, input_len);
    }

    #[test]
    fn case_1_large() {
        let map_capacity = 256;

        let mut map_src: MapSrc<i16> = MapSrc::new(2);
        map_src.add(BYTE_256, map_capacity);
        map_src.add(BYTE_512, 32);
        map_src.add(KB, 32);

        map_src.add_map_reorder_program_src();

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let input_len = map_capacity / 2;

        let test_matrix = TestMatrix::new(input_len, DEFAULT_MAP_KEY_LENGTH, BYTE_256, 1, 11);
        let indices: Vec<usize> = (input_len..map_capacity).collect();

        m.put_with_index(BYTE_256, &test_matrix.keys, &test_matrix.values, &indices)
            .unwrap();

        let (result_output, _indices_output) = m.reorder_for_block(BYTE_256).unwrap();

        assert_reorder_result(&result_output, input_len);
        assert_map_entries(&m, BYTE_256, input_len);
    }

    #[test]
    fn case_2() {
        let map_capacity = 64;

        let mut map_src: MapSrc<i16> = MapSrc::new(2);
        map_src.add(BYTE_256, map_capacity);
        map_src.add(BYTE_512, 32);
        map_src.add(KB, 32);

        map_src.add_map_reorder_program_src();

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let input_len = map_capacity / 2;

        let test_matrix = TestMatrix::new(input_len, DEFAULT_MAP_KEY_LENGTH, BYTE_512, 1, 11);

        let start_index = map_capacity / 4;
        let end_index = map_capacity / 2;
        let mut indices: Vec<usize> = (start_index..end_index).collect();

        let start_index = (map_capacity / 2) + map_capacity / 4;
        let end_index = map_capacity;
        indices.append(&mut (start_index..end_index).collect());

        m.put_with_index(BYTE_256, &test_matrix.keys, &test_matrix.values, &indices)
            .unwrap();

        let (result_output, _indices_output) = m.reorder_for_block(BYTE_256).unwrap();

        assert_reorder_result(&result_output, input_len);
        assert_map_entries(&m, BYTE_256, input_len);
    }

    #[test]
    fn case_2_large() {
        let map_capacity = 256;

        let mut map_src: MapSrc<i16> = MapSrc::new(2);
        map_src.add(BYTE_256, map_capacity);
        map_src.add(BYTE_512, 32);
        map_src.add(KB, 32);

        map_src.add_map_reorder_program_src();

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let input_len = map_capacity / 2;

        let test_matrix = TestMatrix::new(input_len, DEFAULT_MAP_KEY_LENGTH, BYTE_512, 1, 11);

        let start_index = map_capacity / 4;
        let end_index = map_capacity / 2;
        let mut indices: Vec<usize> = (start_index..end_index).collect();

        let start_index = (map_capacity / 2) + map_capacity / 4;
        let end_index = map_capacity;
        indices.append(&mut (start_index..end_index).collect());

        m.put_with_index(BYTE_256, &test_matrix.keys, &test_matrix.values, &indices)
            .unwrap();

        let (result_output, _indices_output) = m.reorder_for_block(BYTE_256).unwrap();

        assert_reorder_result(&result_output, input_len);
        assert_map_entries(&m, BYTE_256, input_len);
    }

    #[test]
    fn case_3() {
        let map_capacity = 32;

        let mut map_src: MapSrc<i16> = MapSrc::new(2);
        map_src.add(BYTE_256, map_capacity);
        map_src.add(BYTE_512, 32);
        map_src.add(KB, 32);

        map_src.add_map_reorder_program_src();

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let key = vec![1; DEFAULT_MAP_KEY_LENGTH];
        let value = vec![2; BYTE_256];
        let indices: Vec<usize> = vec![map_capacity - 1];

        m.put_with_index(BYTE_256, &vec![key.clone()], &vec![value.clone()], &indices)
            .unwrap();

        let (result_output, _indices_output) = m.reorder_for_block(BYTE_256).unwrap();

        assert_reorder_result(&result_output, 1);
        assert_map_entries(&m, BYTE_256, 1);
    }

    #[test]
    fn case_3_large() {
        let map_capacity = 256;

        let mut map_src: MapSrc<i16> = MapSrc::new(2);
        map_src.add(BYTE_256, map_capacity);
        map_src.add(BYTE_512, 32);
        map_src.add(KB, 32);

        map_src.add_map_reorder_program_src();

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let key = vec![1; DEFAULT_MAP_KEY_LENGTH];
        let value = vec![2; BYTE_256];
        let indices: Vec<usize> = vec![map_capacity - 1];

        m.put_with_index(BYTE_256, &vec![key.clone()], &vec![value.clone()], &indices)
            .unwrap();

        let (result_output, _indices_output) = m.reorder_for_block(BYTE_256).unwrap();

        assert_reorder_result(&result_output, 1);
        assert_map_entries(&m, BYTE_256, 1);
    }

    #[test]
    fn case_4() {
        let map_capacity = 32;

        let mut map_src: MapSrc<i16> = MapSrc::new(2);
        map_src.add(BYTE_256, map_capacity);
        map_src.add(BYTE_512, 32);
        map_src.add(KB, 32);

        map_src.add_map_reorder_program_src();

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let key = vec![1; DEFAULT_MAP_KEY_LENGTH];
        let value = vec![2; BYTE_256];
        let indices: Vec<usize> = vec![map_capacity - 2];

        m.put_with_index(BYTE_256, &vec![key.clone()], &vec![value.clone()], &indices)
            .unwrap();

        let (result_output, _indices_output) = m.reorder_for_block(BYTE_256).unwrap();

        assert_reorder_result(&result_output, 1);
        assert_map_entries(&m, BYTE_256, 1);
    }

    #[test]
    fn case_4_large() {
        let map_capacity = 256;

        let mut map_src: MapSrc<i16> = MapSrc::new(2);
        map_src.add(BYTE_256, map_capacity);
        map_src.add(BYTE_512, 32);
        map_src.add(KB, 32);

        map_src.add_map_reorder_program_src();

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let key = vec![1; DEFAULT_MAP_KEY_LENGTH];
        let value = vec![2; BYTE_256];
        let indices: Vec<usize> = vec![map_capacity - 2];

        m.put_with_index(BYTE_256, &vec![key.clone()], &vec![value.clone()], &indices)
            .unwrap();

        let (result_output, _indices_output) = m.reorder_for_block(BYTE_256).unwrap();

        assert_reorder_result(&result_output, 1);
        assert_map_entries(&m, BYTE_256, 1);
    }

    #[test]
    fn case_5() {
        let map_capacity = 32;

        let mut map_src = MapSrc::new(2);
        map_src.add(BYTE_256, 32);
        map_src.add(BYTE_512, 32);
        map_src.add(KB, map_capacity);

        map_src.add_map_reorder_program_src();

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix = TestMatrix::new(map_capacity / 2, DEFAULT_MAP_KEY_LENGTH, KB, 1, 1);
        let indices: Vec<usize> = (0..map_capacity)
            .filter(|x| {
                if x % 2 == 0 {
                    return true;
                }
                false
            })
            .collect();

        m.put_with_index(KB, &test_matrix.keys, &test_matrix.values, &indices)
            .unwrap();

        let (result_output, _indices_output) = m.reorder_for_block(KB).unwrap();

        assert_reorder_result(&result_output, map_capacity / 2);
        assert_map_entries(&m, KB, map_capacity / 2);
    }

    #[test]
    fn case_5_large() {
        let map_capacity = 256;

        let mut map_src = MapSrc::new(2);
        map_src.add(BYTE_256, 32);
        map_src.add(BYTE_512, 32);
        map_src.add(KB, map_capacity);

        map_src.add_map_reorder_program_src();

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix = TestMatrix::new(map_capacity / 2, DEFAULT_MAP_KEY_LENGTH, KB, 1, 1);
        let indices: Vec<usize> = (0..map_capacity)
            .filter(|x| {
                if x % 2 == 0 {
                    return true;
                }
                false
            })
            .collect();

        m.put_with_index(KB, &test_matrix.keys, &test_matrix.values, &indices)
            .unwrap();

        let (result_output, _indices_output) = m.reorder_for_block(KB).unwrap();

        assert_reorder_result(&result_output, map_capacity / 2);
        assert_map_entries(&m, KB, map_capacity / 2);
    }
}
