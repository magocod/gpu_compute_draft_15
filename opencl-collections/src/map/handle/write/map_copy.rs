use crate::config::{ClTypeTrait, DEBUG_MODE};
use crate::error::OpenClResult;
use crate::map::config::check_local_work_size;
use crate::map::handle::MapHandle;
use crate::map::kernel::name::{get_map_kernel_name, MAP_COPY_VALUE};
use opencl::opencl_sys::bindings::{cl_int, cl_uint};
use opencl::wrapper::system::OpenclCommonOperation;

// from
// const FROM_ENTRY_INDEX: usize = 0;
// const FROM_LAST_INDEX: usize = 1;

// to
// const TO_VALUE_LEN: usize = 2;
// const TO_ENTRY_INDEX: usize = 3;
// const TO_START_INDEX: usize = 4;

#[derive(Debug, Clone)]
pub struct MapCopyParam {
    from_entry_index: cl_int,
    from_last_index: cl_int,
    to_map_value_len: cl_int,
    to_entry_index: cl_int,
    to_start_index: cl_int,
}

impl MapCopyParam {
    pub fn new(
        from_entry_index: cl_int,
        from_last_index: cl_int,
        to_map_value_len: cl_int,
        to_entry_index: cl_int,
        to_start_index: cl_int,
    ) -> Self {
        Self {
            from_entry_index,
            from_last_index,
            to_map_value_len,
            to_entry_index,
            to_start_index,
        }
    }
}

impl<T: ClTypeTrait, D: OpenclCommonOperation> MapHandle<T, D> {
    pub fn copy_value(
        &self,
        map_value_len: usize,
        params: &[MapCopyParam],
    ) -> OpenClResult<Vec<cl_int>> {
        let _config = self.map_src.get_config_by_value_len(map_value_len)?;

        let global_work_size = params.len();
        let local_work_size = check_local_work_size(global_work_size);

        let mut copy_params_input = Vec::with_capacity(global_work_size * 5);

        for param in params.iter() {
            copy_params_input.append(&mut vec![
                param.from_entry_index, // FROM_ENTRY_INDEX
                param.from_last_index,  // FROM_LAST_INDEX
                param.to_map_value_len, // TO_VALUE_LEN
                param.to_entry_index,   // TO_ENTRY_INDEX
                param.to_start_index,   // TO_START_INDEX
            ]);
        }

        let copy_params_input_buf = self
            .system
            .blocking_prepare_input_buffer(&copy_params_input)?;

        let copy_output_buf = self.system.create_output_buffer(global_work_size)?;

        let map_id = self.map_id as cl_uint;

        let kernel_name = get_map_kernel_name(MAP_COPY_VALUE, map_value_len);
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&map_id)?;
            kernel.set_arg(&copy_params_input_buf.get_cl_mem())?;
            kernel.set_arg(&copy_output_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        let copy_output =
            self.system
                .blocking_enqueue_read_buffer(global_work_size, &copy_output_buf, &[])?;

        if DEBUG_MODE {
            println!("copy_output {copy_output:?}");
        }

        Ok(copy_output)
    }
}

#[cfg(test)]
mod tests_map_copy_value {
    use super::*;
    use crate::config::ClTypeDefault;
    use crate::map::config::{MapSrc, DEFAULT_MAP_KEY_LENGTH};
    use crate::map::handle::test_utils::{
        assert_map_block_is_equal_to_test_matrix, assert_pair_is_empty,
        generate_arc_opencl_block_default,
    };
    use crate::map::handle::write::map_copy::MapCopyParam;
    use crate::map::handle::MapHandle;
    use crate::test_utils::TestMatrix;
    use crate::utils::{ensure_vec_size, BYTE_256, BYTE_512, KB};

    #[test]
    fn simple_cases() {
        let mut map_src: MapSrc<i16> = MapSrc::new(2);
        map_src.add(BYTE_256, 16);
        map_src.add(BYTE_512, 32);
        map_src.add(KB, 32);

        map_src.add_map_copy_program_src();

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix = TestMatrix::new(16, DEFAULT_MAP_KEY_LENGTH, BYTE_256, 1, 10);
        test_matrix.put(&m, BYTE_256);

        let params_512: Vec<MapCopyParam> = (0..4)
            .map(|from_entry_index| -> MapCopyParam {
                let to_entry_index = from_entry_index;
                MapCopyParam::new(from_entry_index, 256, BYTE_512 as cl_int, to_entry_index, 0)
            })
            .collect();

        let params_1024: Vec<_> = (0..4)
            .map(|from_entry_index| -> MapCopyParam {
                let to_entry_index = from_entry_index;
                MapCopyParam::new(from_entry_index, 256, KB as cl_int, to_entry_index, 0)
            })
            .collect();

        // from 256
        let mut params = params_512.clone();
        params.append(&mut params_1024.clone());

        for param in params.iter() {
            println!("p {param:?}");
        }

        let copy_result = m.copy_value(BYTE_256, &params).unwrap();
        assert_eq!(copy_result, vec![0; 8]);

        assert_map_block_is_equal_to_test_matrix(&m, BYTE_256, &test_matrix);

        let pairs = m.read(BYTE_512).unwrap();
        for (i, pair) in pairs.into_iter().enumerate() {
            if i >= 4 {
                assert_pair_is_empty(&pair, BYTE_512);
                continue;
            }
            assert_eq!(pair.key, vec![i16::cl_default(); DEFAULT_MAP_KEY_LENGTH]);
            assert_eq!(
                pair.value,
                ensure_vec_size(&test_matrix.values[i], BYTE_512)
            );
        }

        let pairs = m.read(KB).unwrap();
        for (i, pair) in pairs.into_iter().enumerate() {
            if i >= 4 {
                assert_pair_is_empty(&pair, KB);
                continue;
            }
            assert_eq!(pair.key, vec![i16::cl_default(); DEFAULT_MAP_KEY_LENGTH]);
            assert_eq!(pair.value, ensure_vec_size(&test_matrix.values[i], KB));
        }
    }
}
