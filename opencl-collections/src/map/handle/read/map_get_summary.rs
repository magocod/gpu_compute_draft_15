use crate::config::{ClTypeTrait, DEBUG_MODE};
use crate::error::OpenClResult;
use crate::map::config::MapBlockSummary;
use crate::map::handle::MapHandle;
use crate::map::kernel::name::MAP_GET_SUMMARY;
use opencl::opencl_sys::bindings::{cl_int, cl_uint};
use opencl::wrapper::system::OpenclCommonOperation;

#[derive(Debug, PartialEq)]
pub struct MapSummary<T: ClTypeTrait> {
    pub reserved: usize,
    pub capacity: usize,
    pub blocks: Vec<MapBlockSummary<T>>,
}

impl<T: ClTypeTrait> MapSummary<T> {
    pub fn new(reserved: usize, blocks: Vec<MapBlockSummary<T>>) -> Self {
        let capacity = blocks.iter().map(|x| x.block.capacity).sum();
        Self {
            reserved,
            capacity,
            blocks,
        }
    }
}

impl<T: ClTypeTrait, D: OpenclCommonOperation> MapHandle<T, D> {
    pub fn get_summary(&self) -> OpenClResult<MapSummary<T>> {
        let map_config = &self.map_src;

        let map_blocks = map_config.get_configs();

        let global_work_size = 1;
        let local_work_size = 1;

        // SUMMARY_INDEX__GENERAL
        let output_capacity = map_blocks.len() + 1;

        // CMQ_CHECK_MAP_KEYS
        // CMQ_GET_SUMMARY
        let enqueue_kernel_output_capacity = map_blocks.len() + 2;

        let output_buf = self
            .system
            .create_output_buffer::<cl_int>(output_capacity)?;

        let enqueue_kernel_output_buf = self
            .system
            .create_output_buffer(enqueue_kernel_output_capacity)?;

        let q0 = self
            .system
            .get_device_command_queue_0()
            .get_cl_command_queue();

        let map_id = self.map_id as cl_uint;

        let mut kernel = self.system.create_kernel(MAP_GET_SUMMARY)?;

        unsafe {
            kernel.set_arg(&q0)?;
            kernel.set_arg(&map_id)?;
            kernel.set_arg(&output_buf.get_cl_mem())?;
            kernel.set_arg(&enqueue_kernel_output_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        let output = self
            .system
            .blocking_enqueue_read_buffer(output_capacity, &output_buf, &[])?;

        let mut summaries = Vec::with_capacity(map_blocks.len());

        for (i, config) in map_blocks.iter().enumerate() {
            let summary = MapBlockSummary::new(config, output[i] as usize);
            summaries.push(summary)
        }

        self.system.assert_device_enqueue_kernel(
            enqueue_kernel_output_capacity,
            enqueue_kernel_output_buf,
            &[],
        )?;

        let map_summary = MapSummary::new(*output.last().unwrap() as usize, summaries);

        if DEBUG_MODE {
            println!("{:#?}", map_summary);
        }

        Ok(map_summary)
    }
}

// TODO impl Handle get_summary

#[cfg(test)]
mod tests_map_get_summary {
    use super::*;
    use crate::map::config::{MapSrc, DEFAULT_MAP_KEY_LENGTH};
    use crate::map::handle::test_utils::generate_arc_opencl_block_default;
    use crate::map::handle::MapHandle;
    use crate::test_utils::TestMatrix;
    use crate::utils::{BYTE_256, BYTE_512, KB};

    const TOTAL_MAPS: usize = 2;

    #[test]
    fn is_empty() {
        let map_capacity = 32;

        let mut map_src: MapSrc<i32> = MapSrc::new(TOTAL_MAPS);
        map_src.add(BYTE_256, map_capacity * 2);
        map_src.add(BYTE_512, map_capacity);
        map_src.add(KB, map_capacity / 2);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let summary = m.get_summary().unwrap();

        assert_eq!(summary.reserved, 0);
        assert_eq!(summary.capacity, map_src.get_maximum_assignable_keys());

        let summary_256 = &summary.blocks[0];
        assert_eq!(
            summary_256,
            &MapBlockSummary::new(map_src.get_config_by_value_len(BYTE_256).unwrap(), 0)
        );

        let summary_512 = &summary.blocks[1];
        assert_eq!(
            summary_512,
            &MapBlockSummary::new(map_src.get_config_by_value_len(BYTE_512).unwrap(), 0)
        );

        let summary_1024 = &summary.blocks[2];
        assert_eq!(
            summary_1024,
            &MapBlockSummary::new(map_src.get_config_by_value_len(KB).unwrap(), 0)
        );
    }

    #[test]
    fn with_records() {
        let map_capacity = 32;

        let mut map_src: MapSrc<i32> = MapSrc::new(TOTAL_MAPS);
        map_src.add(BYTE_256, map_capacity * 2);
        map_src.add(BYTE_512, map_capacity);
        map_src.add(KB, map_capacity / 2);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix_256 =
            TestMatrix::new(map_capacity / 2, DEFAULT_MAP_KEY_LENGTH, BYTE_256, 40, 400);
        test_matrix_256.put(&m, BYTE_256);

        let test_matrix_512 = TestMatrix::new(map_capacity, 1, 1, 80, 800);
        test_matrix_512.put(&m, BYTE_512);

        let summary = m.get_summary().unwrap();

        assert_eq!(summary.reserved, (map_capacity / 2) + map_capacity);
        assert_eq!(summary.capacity, map_src.get_maximum_assignable_keys());

        let summary_256 = &summary.blocks[0];
        assert_eq!(
            summary_256,
            &MapBlockSummary::new(
                map_src.get_config_by_value_len(BYTE_256).unwrap(),
                map_capacity / 2
            )
        );

        let summary_512 = &summary.blocks[1];
        assert_eq!(
            summary_512,
            &MapBlockSummary::new(
                map_src.get_config_by_value_len(BYTE_512).unwrap(),
                map_capacity
            )
        );

        let summary_1024 = &summary.blocks[2];
        assert_eq!(
            summary_1024,
            &MapBlockSummary::new(map_src.get_config_by_value_len(KB).unwrap(), 0)
        );
    }

    #[test]
    fn with_large_records() {
        let map_capacity = 1024;

        let mut map_src: MapSrc<i32> = MapSrc::new(TOTAL_MAPS);
        map_src.add(BYTE_256, map_capacity * 2);
        map_src.add(BYTE_512, map_capacity);
        map_src.add(KB, map_capacity * 4);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix_256 = TestMatrix::new(map_capacity / 2, 2, BYTE_256, 40, 400);
        test_matrix_256.put(&m, BYTE_256);

        let test_matrix_512 = TestMatrix::new(map_capacity, 3, BYTE_512, 80, 800);
        test_matrix_512.put(&m, BYTE_512);

        let summary = m.get_summary().unwrap();

        assert_eq!(summary.reserved, (map_capacity / 2) + map_capacity);
        assert_eq!(summary.capacity, map_src.get_maximum_assignable_keys());

        let summary_256 = &summary.blocks[0];
        assert_eq!(
            summary_256,
            &MapBlockSummary::new(
                map_src.get_config_by_value_len(BYTE_256).unwrap(),
                map_capacity / 2
            )
        );

        let summary_512 = &summary.blocks[1];
        assert_eq!(
            summary_512,
            &MapBlockSummary::new(
                map_src.get_config_by_value_len(BYTE_512).unwrap(),
                map_capacity
            )
        );

        let summary_1024 = &summary.blocks[2];
        assert_eq!(
            summary_1024,
            &MapBlockSummary::new(map_src.get_config_by_value_len(KB).unwrap(), 0)
        );
    }
}
