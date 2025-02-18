use crate::config::ClTypeTrait;
use crate::error::OpenClResult;
use crate::map::config::{check_local_work_size, DEFAULT_MAP_KEY_LENGTH};
use crate::map::handle::{EntryIndices, MapHandle, MapKeys, MapValues};
use crate::map::kernel::name::{
    get_map_kernel_name, MAP_PUT, MAP_PUT_WITH_CMQ, MAP_PUT_WITH_INDEX, MAP_PUT_WITH_PIPE_AND_CMQ,
};
use crate::utils::{ensure_vec_size, from_buf_usize_to_vec_i32};
use opencl::opencl_sys::bindings::cl_uint;
use opencl::wrapper::memory::Pipe;
use opencl::wrapper::system::OpenclCommonOperation;

impl<T: ClTypeTrait, D: OpenclCommonOperation> MapHandle<T, D> {
    pub fn put(
        &self,
        map_value_len: usize,
        keys: &MapKeys<T>,
        values: &MapKeys<T>,
    ) -> OpenClResult<()> {
        let config = self.map_src.get_config_by_value_len(map_value_len)?;
        config.can_hold(keys.len());

        let global_work_size = keys.len();

        let local_work_size = check_local_work_size(global_work_size);

        let key_input_capacity = DEFAULT_MAP_KEY_LENGTH * global_work_size;
        let value_input_capacity = map_value_len * global_work_size;

        let mut keys_input: Vec<_> = Vec::with_capacity(key_input_capacity);

        for k in keys {
            let mut v = ensure_vec_size(k, DEFAULT_MAP_KEY_LENGTH);
            keys_input.append(&mut v);
        }

        let mut values_input: Vec<_> = Vec::with_capacity(value_input_capacity);

        for b in values {
            let mut v = ensure_vec_size(b, map_value_len);
            values_input.append(&mut v);
        }

        let keys_input_buf = self.system.blocking_prepare_input_buffer(&keys_input)?;

        let values_input_buf = self.system.blocking_prepare_input_buffer(&values_input)?;

        let map_id = self.map_id as cl_uint;

        let kernel_name = get_map_kernel_name(MAP_PUT, map_value_len);
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&map_id)?;
            kernel.set_arg(&keys_input_buf.get_cl_mem())?;
            kernel.set_arg(&values_input_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        Ok(())
    }

    // For now only a few simultaneous elements
    pub fn put_with_cmq(
        &self,
        map_value_len: usize,
        keys: &MapKeys<T>,
        values: &MapKeys<T>,
    ) -> OpenClResult<()> {
        let config = self.map_src.get_config_by_value_len(map_value_len)?;
        config.can_hold(keys.len());

        let global_work_size = keys.len();

        let local_work_size = check_local_work_size(global_work_size);

        let key_input_capacity = DEFAULT_MAP_KEY_LENGTH * global_work_size;
        let value_input_capacity = map_value_len * global_work_size;

        // CMQ_PUT_MAP_KEY_INDEX - index 0
        // CMQ_PUT_MAP_VALUE_INDEX - index 1
        let enqueue_kernel_output_capacity = 2 * global_work_size;

        let mut keys_input: Vec<_> = Vec::with_capacity(key_input_capacity);

        for k in keys {
            let mut v = ensure_vec_size(k, DEFAULT_MAP_KEY_LENGTH);
            keys_input.append(&mut v);
        }

        let mut values_input: Vec<_> = Vec::with_capacity(value_input_capacity);

        for b in values {
            let mut v = ensure_vec_size(b, map_value_len);
            values_input.append(&mut v);
        }

        let keys_input_buf = self.system.blocking_prepare_input_buffer(&keys_input)?;

        let values_input_buf = self.system.blocking_prepare_input_buffer(&values_input)?;

        let enqueue_kernel_output_buf = self
            .system
            .create_output_buffer(enqueue_kernel_output_capacity)?;

        let q0 = self
            .system
            .get_device_command_queue_0()
            .get_cl_command_queue();

        let map_id = self.map_id as cl_uint;

        let kernel_name = get_map_kernel_name(MAP_PUT_WITH_CMQ, map_value_len);
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&q0)?;
            kernel.set_arg(&map_id)?;
            kernel.set_arg(&keys_input_buf.get_cl_mem())?;
            kernel.set_arg(&values_input_buf.get_cl_mem())?;
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

    pub fn put_with_index(
        &self,
        map_value_len: usize,
        keys: &MapKeys<T>,
        values: &MapKeys<T>,
        indices: &[usize],
    ) -> OpenClResult<()> {
        let config = self.map_src.get_config_by_value_len(map_value_len)?;

        config.can_hold(keys.len());

        let global_work_size = keys.len();

        if global_work_size != indices.len() || global_work_size != values.len() {
            panic!("TODO message invalid input");
        }

        let local_work_size = check_local_work_size(global_work_size);

        let key_input_capacity = DEFAULT_MAP_KEY_LENGTH * global_work_size;
        let value_input_capacity = map_value_len * global_work_size;

        let mut keys_input: Vec<_> = Vec::with_capacity(key_input_capacity);

        for k in keys {
            let mut v = ensure_vec_size(k, DEFAULT_MAP_KEY_LENGTH);
            keys_input.append(&mut v);
        }

        let mut values_input: Vec<_> = Vec::with_capacity(value_input_capacity);

        for b in values {
            let mut v = ensure_vec_size(b, map_value_len);
            values_input.append(&mut v);
        }

        let indices_input = from_buf_usize_to_vec_i32(indices);

        let keys_input_buf = self.system.blocking_prepare_input_buffer(&keys_input)?;

        let values_input_buf = self.system.blocking_prepare_input_buffer(&values_input)?;

        let indices_buf = self.system.blocking_prepare_input_buffer(&indices_input)?;

        let map_id = self.map_id as cl_uint;

        let kernel_name = get_map_kernel_name(MAP_PUT_WITH_INDEX, map_value_len);
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&map_id)?;
            kernel.set_arg(&keys_input_buf.get_cl_mem())?;
            kernel.set_arg(&values_input_buf.get_cl_mem())?;
            kernel.set_arg(&indices_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        Ok(())
    }

    // For now only a few simultaneous elements
    pub fn put_with_pipe_and_cmq(
        &self,
        map_value_len: usize,
        keys: &MapKeys<T>,
        values: &MapValues<T>,
        pipe: &Pipe<i32>,
    ) -> OpenClResult<EntryIndices> {
        let config = self.map_src.get_config_by_value_len(map_value_len)?;

        let global_work_size = keys.len();

        config.can_hold(global_work_size);

        if global_work_size != values.len() {
            panic!("TODO message invalid input")
        }

        let local_work_size = check_local_work_size(global_work_size);

        let key_input_capacity = DEFAULT_MAP_KEY_LENGTH * global_work_size;
        let value_input_capacity = map_value_len * global_work_size;
        let enqueue_kernel_output_capacity = 2 * global_work_size;

        let mut keys_input = Vec::with_capacity(key_input_capacity);
        let mut values_input = Vec::with_capacity(value_input_capacity);

        for key in keys {
            let mut v = ensure_vec_size(key, DEFAULT_MAP_KEY_LENGTH);
            keys_input.append(&mut v);
        }

        for value in values {
            let mut v = ensure_vec_size(value, map_value_len);
            values_input.append(&mut v);
        }

        let keys_input_buf = self.system.blocking_prepare_input_buffer(&keys_input)?;
        let values_input_buf = self.system.blocking_prepare_input_buffer(&values_input)?;

        let indices_output_buf = self.system.create_output_buffer(global_work_size)?;
        let enqueue_kernel_output_buf = self
            .system
            .create_output_buffer(enqueue_kernel_output_capacity)?;

        let map_id = self.map_id as cl_uint;

        let q0 = self
            .system
            .get_device_command_queue_0()
            .get_cl_command_queue();
        let pipe0 = pipe.get_cl_mem();

        let kernel_name = get_map_kernel_name(MAP_PUT_WITH_PIPE_AND_CMQ, map_value_len);
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&q0)?;
            kernel.set_arg(&pipe0)?;
            kernel.set_arg(&map_id)?;

            kernel.set_arg(&keys_input_buf.get_cl_mem())?;
            kernel.set_arg(&values_input_buf.get_cl_mem())?;
            kernel.set_arg(&indices_output_buf.get_cl_mem())?;
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

        self.system.assert_device_enqueue_kernel(
            enqueue_kernel_output_capacity,
            enqueue_kernel_output_buf,
            &[],
        )?;

        Ok(indices_output)
    }
}

#[cfg(test)]
mod tests_map_put {
    use crate::map::config::{MapSrc, DEFAULT_MAP_KEY_LENGTH};
    use crate::map::handle::test_utils::{
        assert_map_block_is_equal_to_test_matrix, generate_arc_opencl_block_default,
    };
    use crate::map::handle::MapHandle;
    use crate::test_utils::TestMatrix;
    use crate::utils::BYTE_256;

    const TOTAL_MAPS: usize = 2;
    const MAP_CAPACITY: usize = 32;
    const MAP_VALUE_LEN: usize = BYTE_256;

    #[test]
    fn with_exact_size() {
        let mut map_src: MapSrc<i32> = MapSrc::new(TOTAL_MAPS);
        map_src.add(MAP_VALUE_LEN, MAP_CAPACITY);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix =
            TestMatrix::new(MAP_CAPACITY, DEFAULT_MAP_KEY_LENGTH, MAP_VALUE_LEN, 1, 10);

        let r = m.put(MAP_VALUE_LEN, &test_matrix.keys, &test_matrix.values);

        assert!(r.is_ok());
        assert_map_block_is_equal_to_test_matrix(&m, MAP_VALUE_LEN, &test_matrix)
    }

    #[test]
    fn with_size_less_than_maximum() {
        let mut map_src: MapSrc<i32> = MapSrc::new(TOTAL_MAPS);
        map_src.add(MAP_VALUE_LEN, MAP_CAPACITY);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix = TestMatrix::new(
            MAP_CAPACITY,
            DEFAULT_MAP_KEY_LENGTH / 2,
            MAP_VALUE_LEN / 2,
            2,
            20,
        );

        let r = m.put(MAP_VALUE_LEN, &test_matrix.keys, &test_matrix.values);

        assert!(r.is_ok());
        assert_map_block_is_equal_to_test_matrix(&m, MAP_VALUE_LEN, &test_matrix)
    }
}

#[cfg(test)]
mod tests_map_put_with_cmq {
    use crate::map::config::{MapSrc, DEFAULT_MAP_KEY_LENGTH};
    use crate::map::handle::test_utils::{
        assert_map_block_is_equal_to_test_matrix, generate_arc_opencl_block_default,
    };
    use crate::map::handle::MapHandle;
    use crate::test_utils::TestMatrix;
    use crate::utils::BYTE_256;

    const TOTAL_MAPS: usize = 2;
    const MAP_CAPACITY: usize = 32;
    const MAP_VALUE_LEN: usize = BYTE_256;

    #[test]
    fn with_exact_size() {
        let mut map_src: MapSrc<i32> = MapSrc::new(TOTAL_MAPS);
        map_src.add(MAP_VALUE_LEN, MAP_CAPACITY);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix =
            TestMatrix::new(MAP_CAPACITY, DEFAULT_MAP_KEY_LENGTH, MAP_VALUE_LEN, 1, 10);

        let r = m.put_with_cmq(MAP_VALUE_LEN, &test_matrix.keys, &test_matrix.values);

        assert!(r.is_ok());
        assert_map_block_is_equal_to_test_matrix(&m, MAP_VALUE_LEN, &test_matrix)
    }

    #[test]
    fn with_size_less_than_maximum() {
        let mut map_src: MapSrc<i32> = MapSrc::new(TOTAL_MAPS);
        map_src.add(MAP_VALUE_LEN, MAP_CAPACITY);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix = TestMatrix::new(
            MAP_CAPACITY,
            DEFAULT_MAP_KEY_LENGTH / 2,
            MAP_VALUE_LEN / 2,
            2,
            20,
        );

        let r = m.put_with_cmq(MAP_VALUE_LEN, &test_matrix.keys, &test_matrix.values);

        assert!(r.is_ok());
        assert_map_block_is_equal_to_test_matrix(&m, MAP_VALUE_LEN, &test_matrix);
    }
}

#[cfg(test)]
mod tests_map_put_with_index {
    use crate::config::ClTypeDefault;
    use crate::map::config::{MapSrc, DEFAULT_MAP_KEY_LENGTH};
    use crate::map::handle::test_utils::{
        assert_map_block_is_equal_to_test_matrix, generate_arc_opencl_block_default,
    };
    use crate::map::handle::MapHandle;
    use crate::test_utils::TestMatrix;
    use crate::utils::BYTE_256;

    const TOTAL_MAPS: usize = 2;
    const MAP_CAPACITY: usize = 32;
    const MAP_VALUE_LEN: usize = BYTE_256;

    #[test]
    fn all_indices() {
        let mut map_src: MapSrc<i32> = MapSrc::new(TOTAL_MAPS);
        map_src.add(MAP_VALUE_LEN, MAP_CAPACITY);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix =
            TestMatrix::new(MAP_CAPACITY, DEFAULT_MAP_KEY_LENGTH, MAP_VALUE_LEN, 6, 60);

        let r = m.put_with_index(
            MAP_VALUE_LEN,
            &test_matrix.keys,
            &test_matrix.values,
            &test_matrix.indices,
        );

        assert!(r.is_ok());
        assert_map_block_is_equal_to_test_matrix(&m, MAP_VALUE_LEN, &test_matrix)
    }

    #[test]
    fn some_indices() {
        let mut map_src: MapSrc<i32> = MapSrc::new(TOTAL_MAPS);
        map_src.add(MAP_VALUE_LEN, MAP_CAPACITY);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let indices = [0, 5, 6, 7, 10, 15, 16, 30];
        let test_matrix =
            TestMatrix::new(indices.len(), DEFAULT_MAP_KEY_LENGTH, MAP_VALUE_LEN, 7, 70);

        let r = m.put_with_index(
            MAP_VALUE_LEN,
            &test_matrix.keys,
            &test_matrix.values,
            &indices,
        );

        assert!(r.is_ok());

        let pairs = m.read(MAP_VALUE_LEN).unwrap();

        assert_eq!(pairs.len(), MAP_CAPACITY);

        for (i, m_index) in indices.into_iter().enumerate() {
            assert_eq!(pairs[m_index].key, test_matrix.keys[i]);
            assert_eq!(pairs[m_index].value, test_matrix.values[i]);
        }

        for (i, pair) in pairs.into_iter().enumerate() {
            if !indices.contains(&i) {
                assert_eq!(pair.key, vec![i32::cl_default(); DEFAULT_MAP_KEY_LENGTH]);
                assert_eq!(pair.value, vec![i32::cl_default(); MAP_VALUE_LEN]);
            }
        }
    }

    // ...
}

#[cfg(test)]
mod tests_map_put_with_pipe_and_cmq {
    use crate::config::ClTypeDefault;
    use crate::map::config::{MapSrc, DEFAULT_MAP_KEY_LENGTH};
    use crate::map::handle::test_utils::generate_arc_opencl_block_default;
    use crate::map::handle::{MapHandle, KEY_NOT_AVAILABLE_TO_ASSIGN};
    use crate::test_utils::TestMatrix;
    use crate::utils::BYTE_256;
    use opencl::opencl_sys::bindings::cl_int;

    const TOTAL_MAPS: usize = 2;
    const MAP_CAPACITY: usize = 32;
    const CONFIG_SIZE: usize = BYTE_256;

    #[test]
    fn all_index_available() {
        let mut map_src: MapSrc<i16> = MapSrc::new(TOTAL_MAPS);
        map_src.add(CONFIG_SIZE, MAP_CAPACITY);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix = TestMatrix::new(MAP_CAPACITY, DEFAULT_MAP_KEY_LENGTH, CONFIG_SIZE, 1, 10);

        let pipe = m.get_empty_keys_for_block(CONFIG_SIZE).unwrap();

        let indices = m
            .put_with_pipe_and_cmq(CONFIG_SIZE, &test_matrix.keys, &test_matrix.values, &pipe)
            .unwrap();
        assert_eq!(indices.len(), MAP_CAPACITY);

        let pairs = m.read(CONFIG_SIZE).unwrap();

        for (i, pair) in pairs.into_iter().enumerate() {
            let m_i = indices.iter().position(|&r| r == i as cl_int).unwrap();
            assert_eq!(pair.key, test_matrix.keys[m_i]);
            assert_eq!(pair.value, test_matrix.values[m_i]);
        }
    }

    #[test]
    fn no_index_available() {
        let mut map_src: MapSrc<i16> = MapSrc::new(TOTAL_MAPS);
        map_src.add(CONFIG_SIZE, MAP_CAPACITY);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix = TestMatrix::new(MAP_CAPACITY, DEFAULT_MAP_KEY_LENGTH, CONFIG_SIZE, 3, 30);
        test_matrix.put(&m, CONFIG_SIZE);

        let test_m = TestMatrix::new(MAP_CAPACITY / 2, DEFAULT_MAP_KEY_LENGTH, CONFIG_SIZE, 4, 40);

        let pipe = m.get_empty_keys_for_block(CONFIG_SIZE).unwrap();

        let indices = m
            .put_with_pipe_and_cmq(CONFIG_SIZE, &test_m.keys, &test_m.values, &pipe)
            .unwrap();

        assert_eq!(indices, vec![KEY_NOT_AVAILABLE_TO_ASSIGN; MAP_CAPACITY / 2]);

        let pairs = m.read(CONFIG_SIZE).unwrap();

        for (i, pair) in pairs.into_iter().enumerate() {
            assert_eq!(pair.key, test_matrix.keys[i]);
            assert_eq!(pair.value, test_matrix.values[i]);
        }
    }

    #[test]
    fn some_indices_available() {
        let mut map_src: MapSrc<i16> = MapSrc::new(TOTAL_MAPS);
        map_src.add(CONFIG_SIZE, MAP_CAPACITY);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        // group A
        let indices_group_a = [1, 2, 6, 10, 12, 20, 23, 27];
        let test_matrix_a = TestMatrix::new(
            indices_group_a.len(),
            DEFAULT_MAP_KEY_LENGTH,
            CONFIG_SIZE,
            55,
            550,
        );
        m.put_with_index(
            CONFIG_SIZE,
            &test_matrix_a.keys,
            &test_matrix_a.values,
            &indices_group_a,
        )
        .unwrap();

        // group B
        let matrix_len_group_b = MAP_CAPACITY / 2;
        let test_matrix_b = TestMatrix::new(
            matrix_len_group_b,
            DEFAULT_MAP_KEY_LENGTH,
            CONFIG_SIZE,
            66,
            660,
        );

        let pipe = m.get_empty_keys_for_block(CONFIG_SIZE).unwrap();
        let indices_group_b = m
            .put_with_pipe_and_cmq(
                CONFIG_SIZE,
                &test_matrix_b.keys,
                &test_matrix_b.values,
                &pipe,
            )
            .unwrap();

        let matrix = m.read(CONFIG_SIZE).unwrap();

        assert_eq!(indices_group_b.len(), matrix_len_group_b);

        // group A
        for (i, m_index) in indices_group_a.iter().enumerate() {
            assert_eq!(matrix[*m_index].key, test_matrix_a.keys[i]);
            assert_eq!(matrix[*m_index].value, test_matrix_a.values[i]);
        }

        // group B
        for (i, pair) in matrix.into_iter().enumerate() {
            if !indices_group_a.contains(&i) && !indices_group_b.contains(&(i as cl_int)) {
                assert_eq!(pair.key, vec![i16::cl_default(); DEFAULT_MAP_KEY_LENGTH]);
                assert_eq!(pair.value, vec![i16::cl_default(); CONFIG_SIZE]);
            }

            // group B
            if indices_group_b.contains(&(i as cl_int)) {
                let m_i = indices_group_b
                    .iter()
                    .position(|&r| r == i as cl_int)
                    .unwrap();

                assert_eq!(pair.key, test_matrix_b.keys[m_i]);
                assert_eq!(pair.value, test_matrix_b.values[m_i]);
            }
        }
    }
}
