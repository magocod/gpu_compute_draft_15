use crate::config::{ClTypeTrait, DEBUG_MODE};
use crate::error::OpenClResult;
use crate::map::config::{check_local_work_size, MapConfig};
use crate::map::handle::{MapBlockSizes, MapHandle};
use crate::map::kernel::name::{get_map_kernel_name, MAP_READ_SIZES, MAP_READ_SIZES_FOR_BLOCK};
use opencl::opencl_sys::bindings::{cl_int, cl_uint};
use opencl::wrapper::system::OpenclCommonOperation;

#[derive(Debug, Clone)]
pub struct MapBlockSizeSummary<T: ClTypeTrait> {
    pub config: MapConfig<T>,
    pub entries_sizes: Vec<cl_int>,
}

impl<T: ClTypeTrait, D: OpenclCommonOperation> MapHandle<T, D> {
    pub fn read_sizes_for_block(&self, map_value_len: usize) -> OpenClResult<MapBlockSizes> {
        let config = self.map_src.get_config_by_value_len(map_value_len)?;

        let global_work_size = config.capacity;
        let local_work_size = check_local_work_size(global_work_size);

        let sizes_output_buf = self.system.create_output_buffer(global_work_size)?;

        let map_id = self.map_id as cl_uint;

        let kernel_name = get_map_kernel_name(MAP_READ_SIZES_FOR_BLOCK, map_value_len);
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&map_id)?;
            kernel.set_arg(&sizes_output_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        let sizes_output: Vec<_> =
            self.system
                .blocking_enqueue_read_buffer(global_work_size, &sizes_output_buf, &[])?;

        if DEBUG_MODE {
            println!("sizes i{sizes_output:?}");
        }

        Ok(sizes_output)
    }

    pub fn read_sizes(&self) -> OpenClResult<Vec<MapBlockSizeSummary<T>>> {
        let map_config = &self.map_src;
        let map_blocks = map_config.get_configs();

        let global_work_size = map_blocks.len();
        let local_work_size = check_local_work_size(global_work_size);

        let sizes_output_capacity = map_blocks.iter().map(|x| x.capacity).sum();

        let enqueue_kernel_output_capacity = global_work_size;

        let sizes_output_buf = self.system.create_output_buffer(sizes_output_capacity)?;

        let enqueue_kernel_output_buf = self
            .system
            .create_output_buffer(enqueue_kernel_output_capacity)?;

        let q0 = self
            .system
            .get_device_command_queue_0()
            .get_cl_command_queue();

        let map_id = self.map_id as cl_uint;

        let mut kernel = self.system.create_kernel(MAP_READ_SIZES)?;

        unsafe {
            kernel.set_arg(&q0)?;
            kernel.set_arg(&map_id)?;

            kernel.set_arg(&sizes_output_buf.get_cl_mem())?;
            kernel.set_arg(&enqueue_kernel_output_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        let sizes_output: Vec<_> = self.system.blocking_enqueue_read_buffer(
            sizes_output_capacity,
            &sizes_output_buf,
            &[],
        )?;

        let mut output: Vec<_> = Vec::new();

        for (i, config) in map_blocks.iter().enumerate() {
            let block_output_index: usize = map_blocks[0..i].iter().map(|x| x.capacity).sum();

            let sizes: Vec<_> =
                sizes_output[block_output_index..(block_output_index + config.capacity)].to_vec();

            if DEBUG_MODE {
                println!("{} len {}, {:?}", config.name, sizes.len(), sizes);
            }

            output.push(MapBlockSizeSummary {
                config: config.clone(),
                entries_sizes: sizes,
            });
        }

        self.system.assert_device_enqueue_kernel(
            global_work_size,
            enqueue_kernel_output_buf,
            &[],
        )?;

        Ok(output)
    }
}

#[cfg(test)]
mod tests_map_read_sizes_for_block {
    use super::*;
    use crate::map::config::{MapSrc, DEFAULT_MAP_KEY_LENGTH};
    use crate::map::handle::test_utils::generate_arc_opencl_block_default;
    use crate::map::handle::MapHandle;
    use crate::test_utils::TestMatrix;
    use crate::utils::{BYTE_256, BYTE_512, KB, MB};

    // FIXME tests names

    #[test]
    fn simple_cases() {
        let mut map_src: MapSrc<i32> = MapSrc::new(2);
        map_src.add(BYTE_256, 16);
        map_src.add(BYTE_512, 64);
        map_src.add(KB, 32);
        map_src.add(MB, 32);

        map_src.add_map_read_sizes_program_src();

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix = TestMatrix::new(16, DEFAULT_MAP_KEY_LENGTH, BYTE_256, 1, 10);
        test_matrix.put(&m, BYTE_256);

        let test_matrix = TestMatrix::new(32, DEFAULT_MAP_KEY_LENGTH, BYTE_512 / 2, 2, 20);
        test_matrix.put(&m, BYTE_512);

        let test_matrix = TestMatrix::new(16, DEFAULT_MAP_KEY_LENGTH, 16, 3, 30);
        test_matrix.put(&m, KB);

        let test_matrix = TestMatrix::new(32, DEFAULT_MAP_KEY_LENGTH, 2, 3, 30);
        test_matrix.put(&m, MB);

        let sizes = m.read_sizes_for_block(BYTE_256).unwrap();

        assert_eq!(sizes, vec![BYTE_256 as cl_int; 16]);

        let sizes = m.read_sizes_for_block(BYTE_512).unwrap();

        assert_eq!(
            sizes
                .iter()
                .filter(|&&x| x == BYTE_512 as cl_int / 2)
                .count(),
            32
        );
        assert_eq!(sizes.iter().filter(|&&x| x == 0).count(), 32);

        let sizes = m.read_sizes_for_block(KB).unwrap();

        assert_eq!(sizes.iter().filter(|&&x| x == 16).count(), 16);
        assert_eq!(sizes.iter().filter(|&&x| x == 0).count(), 16);

        let sizes = m.read_sizes_for_block(MB).unwrap();
        assert_eq!(sizes, vec![2; 32]);
    }
}

#[cfg(test)]
mod tests_map_read_sizes {
    use super::*;
    use crate::map::config::{MapSrc, DEFAULT_MAP_KEY_LENGTH};
    use crate::map::handle::test_utils::generate_arc_opencl_block_default;
    use crate::map::handle::MapHandle;
    use crate::test_utils::TestMatrix;
    use crate::utils::{BYTE_256, BYTE_512, KB, MB};
    // FIXME tests names

    #[test]
    fn simple_cases() {
        let mut map_src: MapSrc<i32> = MapSrc::new(2);
        map_src.add(BYTE_256, 16);
        map_src.add(BYTE_512, 64);
        map_src.add(KB, 32);
        map_src.add(MB, 32);

        map_src.add_map_read_sizes_program_src();

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix = TestMatrix::new(16, DEFAULT_MAP_KEY_LENGTH, BYTE_256, 1, 10);
        test_matrix.put(&m, BYTE_256);

        let test_matrix = TestMatrix::new(32, DEFAULT_MAP_KEY_LENGTH, BYTE_512 / 2, 2, 20);
        test_matrix.put(&m, BYTE_512);

        let test_matrix = TestMatrix::new(16, DEFAULT_MAP_KEY_LENGTH, 16, 3, 30);
        test_matrix.put(&m, KB);

        let test_matrix = TestMatrix::new(32, DEFAULT_MAP_KEY_LENGTH, 2, 3, 30);
        test_matrix.put(&m, MB);

        let map_block_sizes = m.read_sizes().unwrap();

        let m_block_256 = &map_block_sizes[0];
        assert_eq!(m_block_256.entries_sizes, vec![BYTE_256 as cl_int; 16]);

        let m_block_512 = &map_block_sizes[1];
        assert_eq!(
            m_block_512
                .entries_sizes
                .iter()
                .filter(|&&x| x == BYTE_512 as cl_int / 2)
                .count(),
            32
        );
        assert_eq!(
            m_block_512
                .entries_sizes
                .iter()
                .filter(|&&x| x == 0)
                .count(),
            32
        );

        let m_block_1024 = &map_block_sizes[2];
        assert_eq!(m_block_1024.entries_sizes.len(), 32);
        assert_eq!(
            m_block_1024
                .entries_sizes
                .iter()
                .filter(|&&x| x == 16)
                .count(),
            16
        );
        assert_eq!(
            m_block_1024
                .entries_sizes
                .iter()
                .filter(|&&x| x == 0)
                .count(),
            16
        );

        let m_block_mb = &map_block_sizes[3];
        assert_eq!(m_block_mb.entries_sizes, vec![2; 32]);
    }

    #[test]
    fn large_capacity() {
        let mb_8 = MB * 8;
        let mb_20 = MB * 20;

        let mut map_src: MapSrc<i32> = MapSrc::new(1);
        map_src.add(KB, 16);
        map_src.add(MB, 32);
        map_src.add(mb_8, 8);
        map_src.add(mb_20, 8);

        map_src.add_map_read_sizes_program_src();

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix = TestMatrix::new(16, DEFAULT_MAP_KEY_LENGTH, 1, 1, 10);
        test_matrix.put(&m, KB);

        let test_matrix = TestMatrix::new(32, DEFAULT_MAP_KEY_LENGTH, 2, 2, 20);
        test_matrix.put(&m, MB);

        let test_matrix = TestMatrix::new(8, DEFAULT_MAP_KEY_LENGTH, 3, 3, 30);
        test_matrix.put(&m, mb_8);

        let test_matrix = TestMatrix::new(8, DEFAULT_MAP_KEY_LENGTH, 4, 4, 40);
        test_matrix.put(&m, mb_20);

        let map_block_sizes = m.read_sizes().unwrap();

        let m_block_256 = &map_block_sizes[0];
        assert_eq!(m_block_256.entries_sizes, vec![1; 16]);

        let m_block_512 = &map_block_sizes[1];
        assert_eq!(m_block_512.entries_sizes, vec![2; 32]);

        let m_block_1024 = &map_block_sizes[2];
        assert_eq!(m_block_1024.entries_sizes, vec![3; 8]);

        let m_block_mb = &map_block_sizes[3];
        assert_eq!(m_block_mb.entries_sizes, vec![4; 8]);
    }
}
