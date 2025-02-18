use crate::config::{ClTypeTrait, DEBUG_MODE};
use crate::error::OpenClResult;
use crate::map::config::{check_local_work_size, check_max_find_work_size, MapConfig};
use crate::map::handle::MapHandle;
use crate::map::kernel::name::get_map_kernel_name;
use opencl::opencl_sys::bindings::{cl_int, cl_uint};
use opencl::wrapper::system::OpenclCommonOperation;

#[derive(Debug)]
pub struct TmpBlock<T: ClTypeTrait> {
    pub config: MapConfig<T>,
    pub values: Vec<cl_int>,
}

#[derive(Debug)]
pub struct TmpElement<T: ClTypeTrait> {
    pub index: usize,
    pub blocks: Vec<TmpBlock<T>>,
}

// FIXME find better name
pub type TmpMultiple<T> = Vec<TmpElement<T>>;

pub type TmpBasic = Vec<cl_int>;

impl<T: ClTypeTrait, D: OpenclCommonOperation> MapHandle<T, D> {
    pub fn get_tmp_basic_for_block(
        &self,
        map_value_len: usize,
        kernel_name: &str,
    ) -> OpenClResult<TmpBasic> {
        let config = self.map_src.get_config_by_value_len(map_value_len)?;

        let global_work_size = config.capacity;
        let local_work_size = check_local_work_size(global_work_size);

        let output_buf = self.system.create_output_buffer(global_work_size)?;

        let map_id = self.map_id as cl_uint;

        let k_name = get_map_kernel_name(kernel_name, map_value_len);
        let mut kernel = self.system.create_kernel(&k_name)?;

        unsafe {
            kernel.set_arg(&map_id)?;
            kernel.set_arg(&output_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        let output =
            self.system
                .blocking_enqueue_read_buffer(global_work_size, &output_buf, &[])?;

        if DEBUG_MODE {
            println!("{} {:?}", kernel_name, output);
        }

        Ok(output)
    }

    pub fn get_tmp_multiple(
        &self,
        kernel_name: &str,
        elements: usize,
    ) -> OpenClResult<TmpMultiple<T>> {
        check_max_find_work_size(elements);

        let map_config = &self.map_src;
        let total_indices = map_config.get_maximum_assignable_keys();

        let global_work_size = total_indices * elements;
        let local_work_size = check_local_work_size(global_work_size);

        let output_buf = self.system.create_output_buffer(global_work_size)?;

        let map_id = self.map_id as cl_uint;

        let mut kernel = self.system.create_kernel(kernel_name)?;

        unsafe {
            kernel.set_arg(&map_id)?;
            kernel.set_arg(&output_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        let tmp_output =
            self.system
                .blocking_enqueue_read_buffer(global_work_size, &output_buf, &[])?;

        let mut output: Vec<TmpElement<_>> = Vec::with_capacity(elements);

        let map_blocks = map_config.get_configs();

        for (chunk_id, chunk) in tmp_output.chunks(total_indices).enumerate() {
            if DEBUG_MODE {
                println!("element i:  {}", chunk_id);
            }

            let mut tmp_blocks: Vec<TmpBlock<_>> = Vec::with_capacity(map_blocks.len());

            for (i, config) in map_blocks.iter().enumerate() {
                let block_output_index: usize = map_blocks[0..i].iter().map(|x| x.capacity).sum();

                let indices: Vec<_> =
                    chunk[block_output_index..(block_output_index + config.capacity)].to_vec();

                if DEBUG_MODE {
                    println!(
                        "block: {}, len: {}, {:?}",
                        config.name,
                        indices.len(),
                        indices
                    );
                }

                tmp_blocks.push(TmpBlock {
                    config: config.clone(),
                    values: indices,
                });
            }

            output.push(TmpElement {
                index: chunk_id,
                blocks: tmp_blocks,
            });
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests_get_tmp_multiple {
    use crate::map::config::{MapSrc, MAX_FIND_WORK_SIZE};
    use crate::map::handle::test_utils::generate_arc_opencl_block_default;
    use crate::map::handle::MapHandle;
    use crate::map::kernel::name::GET_TMP_FOR_MAP_INSERT;
    use crate::utils::{BYTE_256, BYTE_512, KB};

    #[test]
    fn get_all_tmp() {
        let mut map_src: MapSrc<i16> = MapSrc::new(2);
        map_src.add(BYTE_256, 32);
        map_src.add(BYTE_512, 32);
        map_src.add(KB, 16);

        map_src.add_map_insert_program_src(16);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let tmp_elements = m.get_tmp_multiple(GET_TMP_FOR_MAP_INSERT, 32).unwrap();

        for tmp_element in tmp_elements {
            for b in tmp_element.blocks {
                assert_eq!(b.values, vec![0; b.config.capacity])
            }
        }
    }

    #[test]
    fn get_slice_of_tmp() {
        let mut map_src: MapSrc<i16> = MapSrc::new(2);
        map_src.add(BYTE_256, 32);
        map_src.add(BYTE_512, 64);
        map_src.add(KB, 16);

        map_src.add_map_insert_program_src(MAX_FIND_WORK_SIZE);

        let system = generate_arc_opencl_block_default(&map_src);
        let m = MapHandle::new(0, &map_src, system);

        let tmp_elements = m.get_tmp_multiple(GET_TMP_FOR_MAP_INSERT, 2).unwrap();

        for tmp_element in tmp_elements {
            for b in tmp_element.blocks {
                assert_eq!(b.values, vec![0; b.config.capacity])
            }
        }
    }
}
