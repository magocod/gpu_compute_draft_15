use crate::config::ClTypeTrait;
use crate::map::config::MapSrc;
use crate::map::kernel::common_replace;

const MAP_READ_SIZES_KERNEL: &str = r#"
    kernel void map_read_sizes(
        queue_t q0,
        const uint map_id,
        global int* sizes_output,
        global int* enqueue_kernel_output
        ) {
        int i = get_global_id(0);

        KERNEL_BODY
    }
    "#;

const ENQUEUE_KERNEL: &str = r#"
        // BLOCK_NAME
        enqueue_kernel_output[BLOCK_INDEX] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(MAP_CAPACITY, CAPACITY_DEVICE_LOCAL_WORK_SIZE),
            ^{
               map_read_sizes__BLOCK_NAME(
                    map_id,
                    sizes_output
               );
            }
        );
     "#;

const MAP_READ_SIZES_FOR_BLOCK_KERNEL: &str = r#"
    kernel void map_read_sizes_for_block__BLOCK_NAME(
        const uint map_id,
        global int* sizes_output
        ) {

        int i = get_global_id(0);

        int last_index = get_map_value_size__BLOCK_NAME(
            map_id,
            i // entry_index
        );
        sizes_output[i] = last_index;
    }

    kernel void map_read_sizes__BLOCK_NAME(
        const uint map_id,
        global int* sizes_output
        ) {

        int i = get_global_id(0);

        int last_index = get_map_value_size__BLOCK_NAME(
            map_id,
            i // entry_index
        );
        sizes_output[i + BLOCK_OUTPUT_INDEX] = last_index;
    }
    "#;

impl<T: ClTypeTrait> MapSrc<T> {
    pub fn generate_map_read_sizes_program_src(&self) -> String {
        // utils fn

        let mut enqueue_kernels = String::new();
        let mut map_read_sizes_for_block_kernels = String::new();

        let map_blocks = self.get_configs();

        for (i, config) in map_blocks.iter().enumerate() {
            let block_output_index: usize = map_blocks[0..i].iter().map(|x| x.capacity).sum();

            let template = common_replace(MAP_READ_SIZES_FOR_BLOCK_KERNEL, config)
                .replace("BLOCK_OUTPUT_INDEX", &block_output_index.to_string());
            map_read_sizes_for_block_kernels.push_str(&template);

            let template =
                common_replace(ENQUEUE_KERNEL, config).replace("BLOCK_INDEX", &i.to_string());
            enqueue_kernels.push_str(&template);
        }

        let map_read_sizes_kernel = MAP_READ_SIZES_KERNEL.replace("KERNEL_BODY", &enqueue_kernels);

        format!(
            "
    /// - MAP_READ_SIZES START ///

    /// constants
    // ...

    /// globals
    // ...

    /// kernels
    {map_read_sizes_for_block_kernels}
    {map_read_sizes_kernel}

    /// - MAP_READ_SIZES END ///
        "
        )
    }

    pub fn add_map_read_sizes_program_src(&mut self) -> &mut Self {
        let src = self.generate_map_read_sizes_program_src();
        self.optional_sources.push(src);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::KB;

    #[test]
    fn test_a() {
        let mut map_src: MapSrc<i32> = MapSrc::new(2);
        map_src.add(KB, 16);

        let program_source = map_src.generate_map_read_sizes_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_b() {
        let mut map_src: MapSrc<i32> = MapSrc::new(2);
        map_src.add(256, 8);
        map_src.add(512, 16);

        let program_source = map_src.generate_map_read_sizes_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_c() {
        let mut map_src: MapSrc<i32> = MapSrc::new(2);
        map_src.add(256, 8);
        map_src.add(512, 32);
        map_src.add(1024, 16);

        let program_source = map_src.generate_map_read_sizes_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }
}
