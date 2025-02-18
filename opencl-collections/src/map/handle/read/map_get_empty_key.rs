use crate::config::{ClTypeTrait, DEBUG_MODE};
use crate::error::OpenClResult;
use crate::map::config::{check_local_work_size, MapConfig};
use crate::map::handle::MapHandle;
use crate::map::kernel::name::{
    get_map_kernel_name, GET_ORDERED_PIPE_CONTENT, MAP_GET_EMPTY_KEYS, MAP_GET_EMPTY_KEYS_FOR_BLOCK,
};
use opencl::opencl_sys::bindings::{cl_int, cl_uint};
use opencl::wrapper::memory::Pipe;
use opencl::wrapper::system::OpenclCommonOperation;

pub type PipeIndices = Pipe<i32>;

#[derive(Debug)]
pub struct MapBlockPipe<T: ClTypeTrait> {
    pub config: MapConfig<T>,
    pub pipe: PipeIndices,
}

impl<T: ClTypeTrait, D: OpenclCommonOperation> MapHandle<T, D> {
    pub fn get_pipe_content(
        &self,
        map_value_len: usize,
        pipe: &PipeIndices,
    ) -> OpenClResult<Vec<cl_int>> {
        let config = self.map_src.get_config_by_value_len(map_value_len)?;

        let global_work_size = 1;
        let local_work_size = 1;

        let pipe_size = config.capacity;
        let pipe_elements = pipe_size as cl_uint;

        let output_buf = self.system.create_output_buffer(pipe_size)?;

        let mut kernel = self.system.create_kernel(GET_ORDERED_PIPE_CONTENT)?;

        let pipe0 = pipe.get_cl_mem();

        unsafe {
            kernel.set_arg(&pipe0)?;
            kernel.set_arg(&pipe_elements)?;
            kernel.set_arg(&output_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        let output = self
            .system
            .blocking_enqueue_read_buffer(pipe_size, &output_buf, &[])?;

        if DEBUG_MODE {
            println!("pipe output  len: {} - {output:?}", output.len());
        }

        Ok(output)
    }

    pub fn get_empty_keys_for_block(&self, map_value_len: usize) -> OpenClResult<PipeIndices> {
        let config = self.map_src.get_config_by_value_len(map_value_len)?;

        let global_work_size = config.capacity;
        let local_work_size = check_local_work_size(global_work_size);

        let map_id = self.map_id as cl_uint;

        let pipe = Pipe::new(self.system.get_context(), global_work_size as cl_uint)?;
        let pipe0 = pipe.get_cl_mem();

        let kernel_name = get_map_kernel_name(MAP_GET_EMPTY_KEYS_FOR_BLOCK, map_value_len);
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&pipe0)?;
            kernel.set_arg(&map_id)?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        Ok(pipe)
    }

    pub fn get_empty_keys(&self) -> OpenClResult<Vec<MapBlockPipe<T>>> {
        let map_blocks = self.map_src.get_configs();

        let global_work_size = 1;
        let local_work_size = 1;

        let enqueue_kernel_output_capacity = map_blocks.len();

        let enqueue_kernel_output_buf = self
            .system
            .create_output_buffer(enqueue_kernel_output_capacity)?;

        let map_id = self.map_id as cl_uint;

        let q0 = self
            .system
            .get_device_command_queue_0()
            .get_cl_command_queue();

        let mut pipes = Vec::with_capacity(map_blocks.len());

        for block in map_blocks {
            let pipe = Pipe::new(self.system.get_context(), block.capacity as cl_uint)?;

            pipes.push(MapBlockPipe {
                config: block.clone(),
                pipe,
            });
        }

        let mut kernel = self.system.create_kernel(MAP_GET_EMPTY_KEYS)?;

        unsafe {
            kernel.set_arg(&q0)?;

            for block_pipe in pipes.iter() {
                let p = block_pipe.pipe.get_cl_mem();
                kernel.set_arg(&p)?;
            }

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

        Ok(pipes)
    }

    pub fn get_empty_keys_pipes(&self) -> OpenClResult<Vec<PipeIndices>> {
        let block_pipes = self.get_empty_keys()?;
        let pipes: Vec<_> = block_pipes.into_iter().map(|x| x.pipe).collect();
        Ok(pipes)
    }
}

#[cfg(test)]
mod tests_map_get_empty_keys_for_block {
    use super::*;
    use crate::map::config::{MapSrc, DEFAULT_MAP_KEY_LENGTH};
    use crate::map::handle::test_utils::generate_arc_opencl_block_default;
    use crate::map::handle::{MapHandle, KEY_NOT_AVAILABLE_TO_ASSIGN};
    use crate::test_utils::TestMatrix;
    use crate::utils::BYTE_256;

    const TOTAL_MAPS: usize = 2;
    const CAPACITY: usize = 32;
    const CONFIG_SIZE: usize = BYTE_256;

    #[test]
    fn no_indexes_available() {
        let mut map_src: MapSrc<i16> = MapSrc::new(TOTAL_MAPS);
        map_src.add(CONFIG_SIZE, CAPACITY);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix = TestMatrix::new(CAPACITY, DEFAULT_MAP_KEY_LENGTH, CONFIG_SIZE, 10, 100);
        test_matrix.put(&m, CONFIG_SIZE);

        let pipe = m.get_empty_keys_for_block(CONFIG_SIZE).unwrap();
        let pipe_indices = m.get_pipe_content(CONFIG_SIZE, &pipe).unwrap();

        assert_eq!(pipe_indices.len(), CAPACITY);
        assert_eq!(pipe_indices, vec![KEY_NOT_AVAILABLE_TO_ASSIGN; CAPACITY]);

        // let _ = m
        //     .read(CONFIG_SIZE)
        //     .unwrap();
    }

    // FIXME test name
    #[test]
    fn no_indexes_available_2() {
        let mut map_src: MapSrc<i16> = MapSrc::new(TOTAL_MAPS);
        map_src.add(CONFIG_SIZE, CAPACITY);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let test_matrix =
            TestMatrix::new(CAPACITY, DEFAULT_MAP_KEY_LENGTH / 2, CONFIG_SIZE / 2, 1, 10);
        test_matrix.put(&m, CONFIG_SIZE);

        let pipe = m.get_empty_keys_for_block(CONFIG_SIZE).unwrap();
        let pipe_indices = m.get_pipe_content(CONFIG_SIZE, &pipe).unwrap();

        assert_eq!(pipe_indices.len(), CAPACITY);
        assert_eq!(pipe_indices, vec![KEY_NOT_AVAILABLE_TO_ASSIGN; CAPACITY]);

        // let _ = m
        //     .read(CONFIG_SIZE)
        //     .unwrap();
    }

    #[test]
    fn all_indices_available() {
        let mut map_src: MapSrc<i16> = MapSrc::new(TOTAL_MAPS);
        map_src.add(CONFIG_SIZE, CAPACITY);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let pipe = m.get_empty_keys_for_block(CONFIG_SIZE).unwrap();

        let mut pipe_indices = m.get_pipe_content(CONFIG_SIZE, &pipe).unwrap();
        pipe_indices.sort();

        let available_indices: Vec<cl_int> = (0..CAPACITY as cl_int).collect();

        assert_eq!(pipe_indices.len(), CAPACITY);
        assert_eq!(pipe_indices, available_indices);

        // let _ = m
        //     .read(CONFIG_SIZE)
        //     .unwrap();
    }

    #[test]
    fn some_indices_available() {
        let mut map_src: MapSrc<i16> = MapSrc::new(TOTAL_MAPS);
        map_src.add(CONFIG_SIZE, CAPACITY);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let indices = [1, 4, 5, 10, 11, 17, 23, 31];
        let test_matrix =
            TestMatrix::new(indices.len(), DEFAULT_MAP_KEY_LENGTH, CONFIG_SIZE, 10, 100);

        m.put_with_index(
            CONFIG_SIZE,
            &test_matrix.keys,
            &test_matrix.values,
            &indices,
        )
        .unwrap();

        let pipe = m.get_empty_keys_for_block(CONFIG_SIZE).unwrap();
        let pipe_indices = m.get_pipe_content(CONFIG_SIZE, &pipe).unwrap();

        assert_eq!(pipe_indices.len(), CAPACITY);

        // let _ = m
        //     .read(CONFIG_SIZE)
        //     .unwrap();

        // TODO fix assert
        for entry_index in 0..CAPACITY {
            if indices.contains(&entry_index) {
                assert!(!pipe_indices.contains(&(entry_index as cl_int)));
                continue;
            }

            assert!(pipe_indices.contains(&(entry_index as cl_int)));
        }
    }
}

#[cfg(test)]
mod tests_map_get_empty_keys {
    use super::*;
    use crate::map::config::{MapSrc, DEFAULT_MAP_KEY_LENGTH};
    use crate::map::handle::test_utils::generate_arc_opencl_block_default;
    use crate::map::handle::{MapHandle, KEY_NOT_AVAILABLE_TO_ASSIGN};
    use crate::test_utils::TestMatrix;
    use crate::utils::{BYTE_256, BYTE_512, KB};

    const TOTAL_MAPS: usize = 2;
    const CAPACITY: usize = 32;

    #[test]
    fn no_indexes_available() {
        let mut map_src: MapSrc<i16> = MapSrc::new(TOTAL_MAPS);
        map_src.add(BYTE_256, CAPACITY);
        map_src.add(BYTE_512, CAPACITY);
        map_src.add(KB, CAPACITY);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        for c in map_src.get_configs() {
            let test_matrix = TestMatrix::new(CAPACITY, DEFAULT_MAP_KEY_LENGTH, c.value_len, 1, 10);
            test_matrix.put(&m, c.value_len);
        }

        let block_pipes = m.get_empty_keys().unwrap();

        for block_pipe in block_pipes.into_iter() {
            let pipe_indices = m
                .get_pipe_content(block_pipe.config.value_len, &block_pipe.pipe)
                .unwrap();

            assert_eq!(pipe_indices.len(), CAPACITY);
            assert_eq!(pipe_indices, vec![KEY_NOT_AVAILABLE_TO_ASSIGN; CAPACITY]);
        }
    }

    #[test]
    fn all_indices_available() {
        let mut map_src: MapSrc<i16> = MapSrc::new(TOTAL_MAPS);
        map_src.add(BYTE_256, CAPACITY);
        map_src.add(BYTE_512, CAPACITY);
        map_src.add(KB, CAPACITY);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let block_pipes = m.get_empty_keys().unwrap();

        for block_pipe in block_pipes.into_iter() {
            let available_indices: Vec<cl_int> = (0..CAPACITY as cl_int).collect();

            let mut pipe_indices = m
                .get_pipe_content(block_pipe.config.value_len, &block_pipe.pipe)
                .unwrap();
            pipe_indices.sort();

            assert_eq!(pipe_indices.len(), CAPACITY);
            assert_eq!(pipe_indices, available_indices);
        }
    }

    #[test]
    fn some_indices_available() {
        let mut map_src: MapSrc<i16> = MapSrc::new(TOTAL_MAPS);
        map_src.add(BYTE_256, CAPACITY);
        map_src.add(BYTE_512, CAPACITY);
        map_src.add(KB, CAPACITY);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let indices_256 = [1, 4, 5, 10, 11, 17, 23, 31];
        let test_matrix =
            TestMatrix::new(indices_256.len(), DEFAULT_MAP_KEY_LENGTH, BYTE_256, 10, 100);
        m.put_with_index(
            BYTE_256,
            &test_matrix.keys,
            &test_matrix.values,
            &indices_256,
        )
        .unwrap();

        let indices_512 = [5, 3, 7, 1, 3, 16, 20, 25];
        let test_matrix =
            TestMatrix::new(indices_512.len(), DEFAULT_MAP_KEY_LENGTH, BYTE_512, 20, 300);
        m.put_with_index(
            BYTE_512,
            &test_matrix.keys,
            &test_matrix.values,
            &indices_512,
        )
        .unwrap();

        let indices_1024 = [30, 5, 2, 26, 27, 28, 12, 13];
        let test_matrix = TestMatrix::new(
            indices_1024.len(),
            DEFAULT_MAP_KEY_LENGTH,
            BYTE_512,
            30,
            300,
        );
        m.put_with_index(KB, &test_matrix.keys, &test_matrix.values, &indices_1024)
            .unwrap();

        let block_pipes = m.get_empty_keys().unwrap();

        // 256
        let block_pipe = &block_pipes[0];
        let pipe_indices = m
            .get_pipe_content(block_pipe.config.value_len, &block_pipe.pipe)
            .unwrap();

        assert_eq!(pipe_indices.len(), CAPACITY);

        // TODO fix assert
        for entry_index in 0..CAPACITY {
            if indices_256.contains(&entry_index) {
                assert!(!pipe_indices.contains(&(entry_index as cl_int)));
                continue;
            }

            assert!(pipe_indices.contains(&(entry_index as cl_int)));
        }

        // 512
        let block_pipe = &block_pipes[1];
        let pipe_indices = m
            .get_pipe_content(block_pipe.config.value_len, &block_pipe.pipe)
            .unwrap();

        assert_eq!(pipe_indices.len(), CAPACITY);

        // TODO fix assert
        for entry_index in 0..CAPACITY {
            if indices_512.contains(&entry_index) {
                assert!(!pipe_indices.contains(&(entry_index as cl_int)));
                continue;
            }

            assert!(pipe_indices.contains(&(entry_index as cl_int)));
        }

        // 1024
        let block_pipe = &block_pipes[2];
        let pipe_indices = m
            .get_pipe_content(block_pipe.config.value_len, &block_pipe.pipe)
            .unwrap();

        assert_eq!(pipe_indices.len(), CAPACITY);

        // TODO fix assert
        for entry_index in 0..CAPACITY {
            if indices_1024.contains(&entry_index) {
                assert!(!pipe_indices.contains(&(entry_index as cl_int)));
                continue;
            }

            assert!(pipe_indices.contains(&(entry_index as cl_int)));
        }
    }
}
