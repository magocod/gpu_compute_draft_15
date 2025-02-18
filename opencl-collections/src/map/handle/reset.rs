use crate::config::ClTypeTrait;
use crate::error::OpenClResult;
use crate::map::config::check_local_work_size;
use crate::map::handle::{Handle, MapHandle};
use crate::map::kernel::name::{get_map_kernel_name, MAP_RESET, RESET_ALL_MAPS};
use opencl::opencl_sys::bindings::cl_uint;
use opencl::wrapper::system::OpenclCommonOperation;

impl<T: ClTypeTrait, D: OpenclCommonOperation> MapHandle<T, D> {
    pub fn reset(&self) -> OpenClResult<()> {
        let map_src = &self.map_src;

        let global_work_size = 1;
        let local_work_size = 1;

        let total_queues = map_src.get_configs().len();

        let enqueue_kernel_output_capacity = total_queues;

        let enqueue_kernel_output_buf = self
            .system
            .create_output_buffer(enqueue_kernel_output_capacity)?;

        let kernel_name = get_map_kernel_name(MAP_RESET, self.map_id);
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        let q0 = self
            .system
            .get_device_command_queue_0()
            .get_cl_command_queue();

        let m_id = self.map_id as cl_uint;

        let enqueue_kernel_output_index = 0 as cl_uint;

        unsafe {
            kernel.set_arg(&q0)?;
            kernel.set_arg(&m_id)?;
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
            global_work_size,
            enqueue_kernel_output_buf,
            &[],
        )?;

        Ok(())
    }

    pub fn initialize(&self) -> OpenClResult<()> {
        self.reset()
    }
}

impl<T: ClTypeTrait, D: OpenclCommonOperation> Handle<T, D> {
    pub fn reset_all_maps(&self) -> OpenClResult<()> {
        let map_src = &self.map_src;

        let global_work_size = map_src.get_total_maps();
        let local_work_size = check_local_work_size(global_work_size);

        // + CMQ_RESET_ALL_MAPS
        let total_queues = map_src.get_configs().len() + 1;

        let enqueue_kernel_output_capacity = total_queues * global_work_size;

        let enqueue_kernel_output_buf = self
            .system
            .create_output_buffer(enqueue_kernel_output_capacity)?;

        let q0 = self
            .system
            .get_device_command_queue_0()
            .get_cl_command_queue();

        let mut kernel = self.system.create_kernel(RESET_ALL_MAPS)?;

        unsafe {
            kernel.set_arg(&q0)?;
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

    pub fn initialize_all_maps(&self) -> OpenClResult<()> {
        self.reset_all_maps()
    }
}

#[cfg(test)]
mod tests_map_reset {
    use super::*;
    use crate::map::config::MapSrc;
    use crate::map::handle::test_utils::{
        assert_map_block_is_empty, generate_arc_opencl_block, DefaultTypeTrait,
    };
    use crate::utils::{BYTE_256, BYTE_512, KB};

    const TOTAL_MAPS: usize = 2;
    const MAP_CAPACITY: usize = 32;

    #[test]
    fn test_case_i16() {
        let mut map_src: MapSrc<i16> = MapSrc::new(TOTAL_MAPS);
        map_src.add(BYTE_256, MAP_CAPACITY * 2);
        map_src.add(BYTE_512, MAP_CAPACITY);
        map_src.add(KB, MAP_CAPACITY / 2);

        let system = generate_arc_opencl_block(&map_src, false);
        let map_0 = MapHandle::new(0, &map_src, system.clone());

        let r = map_0.reset();
        assert!(r.is_ok());

        assert_map_block_is_empty(&map_0, BYTE_256, DefaultTypeTrait::Custom);
        assert_map_block_is_empty(&map_0, BYTE_512, DefaultTypeTrait::Custom);
        assert_map_block_is_empty(&map_0, KB, DefaultTypeTrait::Custom);

        let map_1 = MapHandle::new(1, &map_src, system);

        assert_map_block_is_empty(&map_1, BYTE_256, DefaultTypeTrait::Std);
        assert_map_block_is_empty(&map_1, BYTE_512, DefaultTypeTrait::Std);
        assert_map_block_is_empty(&map_1, KB, DefaultTypeTrait::Std);
    }

    #[test]
    fn test_case_u8() {
        let mut map_src: MapSrc<u8> = MapSrc::new(TOTAL_MAPS);
        map_src.add(BYTE_256, MAP_CAPACITY * 2);
        map_src.add(BYTE_512, MAP_CAPACITY);
        map_src.add(KB, MAP_CAPACITY / 2);

        let system = generate_arc_opencl_block(&map_src, false);
        let map_0 = MapHandle::new(0, &map_src, system.clone());

        let r = map_0.reset();
        assert!(r.is_ok());

        assert_map_block_is_empty(&map_0, BYTE_256, DefaultTypeTrait::Custom);
        assert_map_block_is_empty(&map_0, BYTE_512, DefaultTypeTrait::Custom);
        assert_map_block_is_empty(&map_0, KB, DefaultTypeTrait::Custom);

        // map_id 1
        let map_1 = MapHandle::new(1, &map_src, system);

        assert_map_block_is_empty(&map_1, BYTE_256, DefaultTypeTrait::Std);
        assert_map_block_is_empty(&map_1, BYTE_512, DefaultTypeTrait::Std);
        assert_map_block_is_empty(&map_1, KB, DefaultTypeTrait::Std);
    }
}

#[cfg(test)]
mod tests_map_handle_reset_all_maps {
    use crate::map::config::MapSrc;
    use crate::map::handle::test_utils::{
        assert_map_block_is_empty, generate_arc_opencl_block, DefaultTypeTrait,
    };
    use crate::map::handle::{Handle, MapHandle};
    use crate::utils::{BYTE_256, BYTE_512, KB};

    const TOTAL_MAPS: usize = 2;
    const MAP_CAPACITY: usize = 32;

    #[test]
    fn test_case_i16() {
        let mut map_src: MapSrc<i16> = MapSrc::new(TOTAL_MAPS);
        map_src.add(BYTE_256, MAP_CAPACITY * 2);
        map_src.add(BYTE_512, MAP_CAPACITY);
        map_src.add(KB, MAP_CAPACITY / 2);

        let system = generate_arc_opencl_block(&map_src, false);
        let h = Handle::new(&map_src, system.clone());

        let r = h.reset_all_maps();
        assert!(r.is_ok());

        let map_0 = MapHandle::new(0, &map_src, system.clone());

        assert_map_block_is_empty(&map_0, BYTE_256, DefaultTypeTrait::Custom);
        assert_map_block_is_empty(&map_0, BYTE_512, DefaultTypeTrait::Custom);
        assert_map_block_is_empty(&map_0, KB, DefaultTypeTrait::Custom);

        let map_1 = MapHandle::new(1, &map_src, system);

        assert_map_block_is_empty(&map_1, BYTE_256, DefaultTypeTrait::Custom);
        assert_map_block_is_empty(&map_1, BYTE_512, DefaultTypeTrait::Custom);
        assert_map_block_is_empty(&map_1, KB, DefaultTypeTrait::Custom);
    }

    #[test]
    fn test_case_u8() {
        let mut map_src: MapSrc<u8> = MapSrc::new(TOTAL_MAPS);
        map_src.add(BYTE_256, MAP_CAPACITY * 2);
        map_src.add(BYTE_512, MAP_CAPACITY);
        map_src.add(KB, MAP_CAPACITY / 2);

        let system = generate_arc_opencl_block(&map_src, false);
        let h = Handle::new(&map_src, system.clone());

        let r = h.reset_all_maps();
        assert!(r.is_ok());

        // map_id 0
        let map_0 = MapHandle::new(0, &map_src, system.clone());

        assert_map_block_is_empty(&map_0, BYTE_256, DefaultTypeTrait::Custom);
        assert_map_block_is_empty(&map_0, BYTE_512, DefaultTypeTrait::Custom);
        assert_map_block_is_empty(&map_0, KB, DefaultTypeTrait::Custom);

        // map_id 1
        let map_1 = MapHandle::new(1, &map_src, system);

        assert_map_block_is_empty(&map_1, BYTE_256, DefaultTypeTrait::Custom);
        assert_map_block_is_empty(&map_1, BYTE_512, DefaultTypeTrait::Custom);
        assert_map_block_is_empty(&map_1, KB, DefaultTypeTrait::Custom);
    }
}
