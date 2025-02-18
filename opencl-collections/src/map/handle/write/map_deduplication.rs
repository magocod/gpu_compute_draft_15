use crate::config::ClTypeTrait;
use crate::error::OpenClResult;
use crate::map::config::check_local_work_size;
use crate::map::handle::MapHandle;
use crate::map::kernel::name::{
    get_map_kernel_name, GET_TMP_FOR_MAP_DEDUPLICATION, MAP_DEDUPLICATION,
    MAP_DEDUPLICATION_FOR_BLOCK,
};
use opencl::opencl_sys::bindings::{cl_int, cl_uint};
use opencl::wrapper::system::OpenclCommonOperation;

pub const NO_DUPLICATE_KEY: cl_int = 0;
pub const DUPLICATE_KEY: cl_int = 1;

impl<T: ClTypeTrait, D: OpenclCommonOperation> MapHandle<T, D> {
    pub fn map_deduplication_for_block(&self, map_value_len: usize) -> OpenClResult<()> {
        let _config = self.map_src.get_config_by_value_len(map_value_len)?;

        let global_work_size = 1;
        let local_work_size = 1;
        let enqueue_kernel_output_capacity = 2;

        let enqueue_kernel_output_buf = self
            .system
            .create_output_buffer(enqueue_kernel_output_capacity)?;

        let map_id = self.map_id as cl_uint;
        let enqueue_kernel_output_index: cl_int = 0;

        let q0 = self
            .system
            .get_device_command_queue_0()
            .get_cl_command_queue();

        let kernel_name = get_map_kernel_name(MAP_DEDUPLICATION_FOR_BLOCK, map_value_len);
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&q0)?;
            kernel.set_arg(&map_id)?;
            kernel.set_arg(&enqueue_kernel_output_index)?;
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

    pub fn map_deduplication(&self) -> OpenClResult<()> {
        let global_work_size = self.map_src.get_configs().len();
        let local_work_size = check_local_work_size(global_work_size);

        // CMQ_CHECK_DUPLICATES_MAP_KEYS
        // CMQ_CONFIRM_REMOVE_DUPLICATES_MAP_KEYS
        // CMQ_MAP_DEDUPLICATION_FOR_BLOCK
        let enqueue_kernel_output_capacity = global_work_size * 3;

        let enqueue_kernel_output_buf = self
            .system
            .create_output_buffer(enqueue_kernel_output_capacity)?;

        let map_id = self.map_id as cl_uint;

        let mut kernel = self.system.create_kernel(MAP_DEDUPLICATION)?;

        let q0 = self
            .system
            .get_device_command_queue_0()
            .get_cl_command_queue();

        unsafe {
            kernel.set_arg(&q0)?;
            kernel.set_arg(&map_id)?;
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

    pub fn get_tmp_for_map_deduplication(&self, map_value_len: usize) -> OpenClResult<Vec<cl_int>> {
        self.get_tmp_basic_for_block(map_value_len, GET_TMP_FOR_MAP_DEDUPLICATION)
    }
}

#[cfg(test)]
mod tests_get_tmp_for_map_deduplication {
    use crate::map::config::MapSrc;
    use crate::map::handle::test_utils::generate_arc_opencl_block_default;
    use crate::map::handle::MapHandle;
    use crate::utils::{BYTE_256, BYTE_512, KB};

    #[test]
    fn is_empty() {
        let mut map_src: MapSrc<i16> = MapSrc::new(2);
        map_src.add(BYTE_256, 32);
        map_src.add(BYTE_512, 32);
        map_src.add(KB, 16);

        map_src.add_map_deduplication_program_src();

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let tmp_arr = m.get_tmp_for_map_deduplication(BYTE_256).unwrap();

        assert_eq!(tmp_arr, vec![0; 32])
    }
}

#[cfg(test)]
mod tests_map_deduplication_for_block {
    use super::*;
    use crate::config::ClTypeDefault;
    use crate::map::config::{MapSrc, DEFAULT_MAP_KEY_LENGTH};
    use crate::map::handle::test_utils::{
        assert_map_block_is_empty, generate_arc_opencl_block_default, DefaultTypeTrait,
    };
    use crate::map::handle::MapHandle;
    use crate::test_utils::TestMatrix;
    use crate::utils::{ensure_vec_size, BYTE_256, BYTE_512, KB};

    #[test]
    fn map_is_empty() {
        let mut map_src: MapSrc<i16> = MapSrc::new(2);
        map_src.add(BYTE_256, 32);
        map_src.add(BYTE_512, 32);
        map_src.add(KB, 16);

        map_src.add_map_deduplication_program_src();

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        m.map_deduplication_for_block(BYTE_256).unwrap();

        let tmp_arr = m.get_tmp_for_map_deduplication(BYTE_256).unwrap();
        assert_eq!(tmp_arr, vec![NO_DUPLICATE_KEY; 32]);

        assert_map_block_is_empty(&m, BYTE_256, DefaultTypeTrait::Custom);
    }

    #[test]
    fn no_duplicates() {
        let mut map_src: MapSrc<i16> = MapSrc::new(2);
        map_src.add(BYTE_512, 64);
        map_src.add(KB, 16);

        map_src.add_map_deduplication_program_src();

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix = TestMatrix::new(32, DEFAULT_MAP_KEY_LENGTH, BYTE_512, 1, 11);
        test_matrix.put(&m, BYTE_512);

        m.map_deduplication_for_block(BYTE_512).unwrap();

        let tmp_arr = m.get_tmp_for_map_deduplication(BYTE_512).unwrap();
        assert_eq!(tmp_arr, vec![NO_DUPLICATE_KEY; 64]);

        let matrix = m.read(BYTE_512).unwrap();

        for (i, pair) in matrix.into_iter().enumerate() {
            if i >= 32 {
                assert_eq!(pair.key, vec![i16::cl_default(); DEFAULT_MAP_KEY_LENGTH]);
                assert_eq!(pair.value, vec![i16::cl_default(); BYTE_512]);
                continue;
            }

            assert_eq!(
                pair.key,
                ensure_vec_size(&test_matrix.keys[i], DEFAULT_MAP_KEY_LENGTH)
            );
            assert_eq!(
                pair.value,
                ensure_vec_size(&test_matrix.values[i], BYTE_512)
            );
        }
    }

    #[test]
    fn all_keys_are_the_same() {
        let input_len = 32;

        let mut map_src: MapSrc<i16> = MapSrc::new(2);
        map_src.add(BYTE_256, input_len);
        map_src.add(BYTE_512, 32);
        map_src.add(KB, 16);

        map_src.add_map_deduplication_program_src();

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let key = vec![1; DEFAULT_MAP_KEY_LENGTH];
        let value = vec![2; BYTE_256];

        let keys = vec![key.clone(); input_len];
        let values = vec![value.clone(); input_len];

        m.put(BYTE_256, &keys, &values).unwrap();
        m.put(BYTE_512, &keys, &values).unwrap();

        m.map_deduplication_for_block(BYTE_256).unwrap();

        let tmp_arr = m.get_tmp_for_map_deduplication(BYTE_256).unwrap();
        assert_eq!(
            tmp_arr.iter().filter(|x| **x == NO_DUPLICATE_KEY).count(),
            1
        );
        assert_eq!(
            tmp_arr.iter().filter(|x| **x == DUPLICATE_KEY).count(),
            input_len - 1
        );

        let pairs = m.read(BYTE_256).unwrap();

        for (i, pair) in pairs.into_iter().enumerate() {
            if tmp_arr[i] == DUPLICATE_KEY {
                assert_eq!(pair.key, vec![i16::cl_default(); DEFAULT_MAP_KEY_LENGTH]);
                assert_eq!(pair.value, vec![i16::cl_default(); BYTE_256]);
                continue;
            }

            assert_eq!(pair.key, key);
            assert_eq!(pair.value, value);
        }

        let pairs = m.read(BYTE_512).unwrap();

        for pair in pairs.into_iter() {
            assert_eq!(pair.key, key);
            assert_eq!(pair.value, ensure_vec_size(&value, BYTE_512));
        }
    }

    #[test]
    fn multiple_duplicate_keys() {
        let input_len = 32;

        let mut map_src: MapSrc<i16> = MapSrc::new(2);
        map_src.add(BYTE_512, 64);
        map_src.add(KB, 16);

        map_src.add_map_deduplication_program_src();

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let mut test_matrix = TestMatrix::new(input_len, DEFAULT_MAP_KEY_LENGTH, BYTE_512, 1, 11);

        let mut test_matrix_clone = test_matrix.clone();
        test_matrix.append(&mut test_matrix_clone);
        test_matrix.put(&m, BYTE_512);

        m.map_deduplication_for_block(BYTE_512).unwrap();

        let tmp_arr = m.get_tmp_for_map_deduplication(BYTE_512).unwrap();
        assert_eq!(
            tmp_arr.iter().filter(|x| **x == NO_DUPLICATE_KEY).count(),
            input_len
        );
        assert_eq!(
            tmp_arr.iter().filter(|x| **x == DUPLICATE_KEY).count(),
            input_len
        );

        let pairs = m.read(BYTE_512).unwrap();

        for (i, pair) in pairs.into_iter().enumerate() {
            if tmp_arr[i] == DUPLICATE_KEY {
                assert_eq!(pair.key, vec![i16::cl_default(); DEFAULT_MAP_KEY_LENGTH]);
                assert_eq!(pair.value, vec![i16::cl_default(); BYTE_512]);
                continue;
            }

            assert_eq!(
                pair.key,
                ensure_vec_size(&test_matrix.keys[i], DEFAULT_MAP_KEY_LENGTH)
            );
            assert_eq!(
                pair.value,
                ensure_vec_size(&test_matrix.values[i], BYTE_512)
            );
        }
    }
}

#[cfg(test)]
mod tests_map_deduplication {
    use super::*;
    use crate::config::ClTypeDefault;
    use crate::map::config::{MapSrc, DEFAULT_MAP_KEY_LENGTH};
    use crate::map::handle::test_utils::{
        assert_map_block_is_empty, assert_map_block_is_equal_to_test_matrix,
        generate_arc_opencl_block_default, DefaultTypeTrait,
    };
    use crate::map::handle::MapHandle;
    use crate::test_utils::TestMatrix;
    use crate::utils::{ensure_vec_size, BYTE_256, BYTE_512, KB};

    #[test]
    fn map_is_empty() {
        let mut map_src: MapSrc<i16> = MapSrc::new(2);
        map_src.add(BYTE_256, 32);
        map_src.add(BYTE_512, 32);
        map_src.add(KB, 16);

        map_src.add_map_deduplication_program_src();

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        m.map_deduplication().unwrap();

        let tmp_arr = m.get_tmp_for_map_deduplication(BYTE_256).unwrap();
        assert_eq!(tmp_arr, vec![NO_DUPLICATE_KEY; 32]);

        let tmp_arr = m.get_tmp_for_map_deduplication(BYTE_512).unwrap();
        assert_eq!(tmp_arr, vec![NO_DUPLICATE_KEY; 32]);

        let tmp_arr = m.get_tmp_for_map_deduplication(KB).unwrap();
        assert_eq!(tmp_arr, vec![NO_DUPLICATE_KEY; 16]);

        assert_map_block_is_empty(&m, BYTE_256, DefaultTypeTrait::Custom);
        assert_map_block_is_empty(&m, BYTE_512, DefaultTypeTrait::Custom);
        assert_map_block_is_empty(&m, KB, DefaultTypeTrait::Custom);
    }

    #[test]
    fn all_keys_are_the_same() {
        let mut map_src: MapSrc<i16> = MapSrc::new(2);
        map_src.add(BYTE_256, 32);
        map_src.add(BYTE_512, 64);
        map_src.add(KB, 16);

        map_src.add_map_deduplication_program_src();

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let key = vec![1; DEFAULT_MAP_KEY_LENGTH];

        let keys = vec![key.clone(); 32];
        let values = vec![vec![2; BYTE_256]; 32];
        m.put(BYTE_256, &keys, &values).unwrap();

        let keys = vec![key.clone(); 64];
        let values = vec![vec![2; BYTE_512]; 64];
        m.put(BYTE_512, &keys, &values).unwrap();

        let keys = vec![key.clone(); 16];
        let values = vec![vec![2; KB]; 16];
        m.put(KB, &keys, &values).unwrap();

        m.map_deduplication().unwrap();

        let tmp_arr_256 = m.get_tmp_for_map_deduplication(BYTE_256).unwrap();
        assert_eq!(
            tmp_arr_256
                .iter()
                .filter(|x| **x == NO_DUPLICATE_KEY)
                .count(),
            1
        );
        assert_eq!(
            tmp_arr_256.iter().filter(|x| **x == DUPLICATE_KEY).count(),
            31
        );

        let tmp_arr_512 = m.get_tmp_for_map_deduplication(BYTE_512).unwrap();
        assert_eq!(
            tmp_arr_512
                .iter()
                .filter(|x| **x == NO_DUPLICATE_KEY)
                .count(),
            1
        );
        assert_eq!(
            tmp_arr_512.iter().filter(|x| **x == DUPLICATE_KEY).count(),
            63
        );

        let tmp_arr_1024 = m.get_tmp_for_map_deduplication(KB).unwrap();
        assert_eq!(
            tmp_arr_1024
                .iter()
                .filter(|x| **x == NO_DUPLICATE_KEY)
                .count(),
            1
        );
        assert_eq!(
            tmp_arr_1024.iter().filter(|x| **x == DUPLICATE_KEY).count(),
            15
        );

        let pairs = m.read(BYTE_256).unwrap();

        for (i, pair) in pairs.into_iter().enumerate() {
            if tmp_arr_256[i] == DUPLICATE_KEY {
                assert_eq!(pair.key, vec![i16::cl_default(); DEFAULT_MAP_KEY_LENGTH]);
                assert_eq!(pair.value, vec![i16::cl_default(); BYTE_256]);
                continue;
            }

            assert_eq!(pair.key, key);
            assert_eq!(pair.value, vec![2; BYTE_256]);
        }

        let pairs = m.read(BYTE_512).unwrap();

        for (i, pair) in pairs.into_iter().enumerate() {
            if tmp_arr_512[i] == DUPLICATE_KEY {
                assert_eq!(pair.key, vec![i16::cl_default(); DEFAULT_MAP_KEY_LENGTH]);
                assert_eq!(pair.value, vec![i16::cl_default(); BYTE_512]);
                continue;
            }

            assert_eq!(pair.key, key);
            assert_eq!(pair.value, vec![2; BYTE_512]);
        }

        let pairs = m.read(KB).unwrap();

        for (i, pair) in pairs.into_iter().enumerate() {
            if tmp_arr_1024[i] == DUPLICATE_KEY {
                assert_eq!(pair.key, vec![i16::cl_default(); DEFAULT_MAP_KEY_LENGTH]);
                assert_eq!(pair.value, vec![i16::cl_default(); KB]);
                continue;
            }

            assert_eq!(pair.key, key);
            assert_eq!(pair.value, vec![2; KB]);
        }
    }

    #[test]
    fn multiple_duplicate_keys() {
        let mut map_src: MapSrc<i16> = MapSrc::new(2);

        map_src.add(BYTE_256, 32);
        map_src.add(BYTE_512, 64);
        map_src.add(KB, 16);

        map_src.add_map_deduplication_program_src();

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        // all_keys_are_the_same

        let key = vec![1; DEFAULT_MAP_KEY_LENGTH];
        let value = vec![2; BYTE_256];

        let keys = vec![key.clone(); 32];
        let values = vec![value.clone(); 32];

        m.put(BYTE_256, &keys, &values).unwrap();

        // multiple_duplicate_keys

        let mut test_matrix_512 = TestMatrix::new(32, DEFAULT_MAP_KEY_LENGTH, BYTE_512, 1, 11);
        test_matrix_512.append(&mut test_matrix_512.clone());
        test_matrix_512.put(&m, BYTE_512);

        // no_duplicates

        let test_matrix_1024 = TestMatrix::new(16, DEFAULT_MAP_KEY_LENGTH, KB, 1, 11);
        test_matrix_1024.put(&m, KB);

        m.map_deduplication().unwrap();

        // all_keys_are_the_same
        let tmp_arr = m.get_tmp_for_map_deduplication(BYTE_256).unwrap();
        assert_eq!(
            tmp_arr.iter().filter(|x| **x == NO_DUPLICATE_KEY).count(),
            1
        );
        assert_eq!(tmp_arr.iter().filter(|x| **x == DUPLICATE_KEY).count(), 31);

        let pairs = m.read(BYTE_256).unwrap();

        for (i, pair) in pairs.into_iter().enumerate() {
            if tmp_arr[i] == DUPLICATE_KEY {
                assert_eq!(pair.key, vec![i16::cl_default(); DEFAULT_MAP_KEY_LENGTH]);
                assert_eq!(pair.value, vec![i16::cl_default(); BYTE_256]);
                continue;
            }

            assert_eq!(pair.key, vec![1; DEFAULT_MAP_KEY_LENGTH]);
            assert_eq!(pair.value, vec![2; BYTE_256]);
        }

        // multiple_duplicate_keys
        let tmp_arr = m.get_tmp_for_map_deduplication(BYTE_512).unwrap();
        assert_eq!(
            tmp_arr.iter().filter(|x| **x == NO_DUPLICATE_KEY).count(),
            32
        );
        assert_eq!(tmp_arr.iter().filter(|x| **x == DUPLICATE_KEY).count(), 32);

        let pairs = m.read(BYTE_512).unwrap();

        for (i, pair) in pairs.into_iter().enumerate() {
            if tmp_arr[i] == DUPLICATE_KEY {
                assert_eq!(pair.key, vec![i16::cl_default(); DEFAULT_MAP_KEY_LENGTH]);
                assert_eq!(pair.value, vec![i16::cl_default(); BYTE_512]);
                continue;
            }

            assert_eq!(
                pair.key,
                ensure_vec_size(&test_matrix_512.keys[i], DEFAULT_MAP_KEY_LENGTH)
            );
            assert_eq!(
                pair.value,
                ensure_vec_size(&test_matrix_512.values[i], BYTE_512)
            );
        }

        // no_duplicates
        let tmp_arr = m.get_tmp_for_map_deduplication(KB).unwrap();
        assert_eq!(tmp_arr, vec![NO_DUPLICATE_KEY; 16]);

        assert_map_block_is_equal_to_test_matrix(&m, KB, &test_matrix_1024);
    }
}
