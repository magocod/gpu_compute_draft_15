use crate::config::ClTypeTrait;
use crate::map::config::MapSrc;
use crate::map::kernel::common_replace;

const RESET_ALL_MAPS_KERNEL: &str = r#"
    kernel void reset_all_maps(
        queue_t q0,
        global int* enqueue_kernel_output
        ) {
        uint i = get_global_id(0);
        uint enqueue_kernel_output_index = i * TOTAL_ENQUEUE_KERNELS;

        enqueue_kernel_output[CMQ_RESET_ALL_MAPS + enqueue_kernel_output_index] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(1, 1),
            ^{
               map_reset(
                    q0,
                    i, // map_id
                    enqueue_kernel_output_index,
                    enqueue_kernel_output
               );
            }
        );
    }
    "#;

const MAP_RESET_KERNEL: &str = r#"
    kernel void map_reset(
        queue_t q0,
        const uint map_id,
        const uint enqueue_kernel_output_index,
        global int* enqueue_kernel_output
        ) {

        KERNEL_BODY
    }
    "#;

const MAP_BLOCK_RESET_ENQUEUE_KERNEL: &str = r#"
        // BLOCK_NAME
        enqueue_kernel_output[enqueue_kernel_output_index + CMQ_MAP_BLOCK_RESET_INDEX__BLOCK_NAME] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(MAP_CAPACITY, CAPACITY_DEVICE_LOCAL_WORK_SIZE),
            ^{
               map_block_reset__BLOCK_NAME(
                    map_id
               );
            }
        );
     "#;

const RESET_MAP_BLOCK_KERNEL: &str = r#"
    kernel void map_block_reset__BLOCK_NAME(
        const uint map_id
        ) {
        int entry_index = get_global_id(0);

        for (int index = 0; index < MAP_KEY_LEN; index++) {
            map_keys__BLOCK_NAME[map_id][entry_index][index] = CL_DEFAULT_VALUE;
        }

        for (int index = 0; index < MAP_VALUE_LEN; index++) {
            map_values__BLOCK_NAME[map_id][entry_index][index] = CL_DEFAULT_VALUE;
        }
    }
    "#;

const QUEUE_CONST_DEF: &str = r#"
    const int CMQ_MAP_BLOCK_RESET_INDEX__BLOCK_NAME = BLOCK_INDEX;
    "#;

impl<T: ClTypeTrait> MapSrc<T> {
    pub fn generate_map_reset_program_src(&self) -> String {
        // const definitions
        let mut queue_const_def = String::new();

        let mut map_block_reset_kernels = String::new();
        let mut map_block_reset_enqueue_kernels = String::new();

        let map_blocks = self.get_configs();
        let total_blocks = map_blocks.len();

        // +1 main kernel
        let total_enqueue_kernel = total_blocks + 1;
        let main_enqueue_kernel_result = total_blocks;

        for (i, config) in map_blocks.iter().enumerate() {
            let queue_const =
                common_replace(QUEUE_CONST_DEF, config).replace("BLOCK_INDEX", &i.to_string());
            queue_const_def.push_str(&queue_const);

            let template = common_replace(RESET_MAP_BLOCK_KERNEL, config);
            map_block_reset_kernels.push_str(&template);

            let template = common_replace(MAP_BLOCK_RESET_ENQUEUE_KERNEL, config)
                .replace("BLOCK_INDEX", &i.to_string());
            map_block_reset_enqueue_kernels.push_str(&template);
        }

        let map_reset_kernel =
            MAP_RESET_KERNEL.replace("KERNEL_BODY", &map_block_reset_enqueue_kernels);

        let reset_all_maps_kernel = RESET_ALL_MAPS_KERNEL
            .replace("TOTAL_ENQUEUE_KERNELS", &total_enqueue_kernel.to_string());

        format!(
            "
    /// - MAP_SET_DEFAULT START ///

    /// constants

    {queue_const_def}
    const int CMQ_RESET_ALL_MAPS = {main_enqueue_kernel_result};

    /// globals
    // ...

    /// kernels
    {map_block_reset_kernels}
    {map_reset_kernel}
    {reset_all_maps_kernel}

    /// - MAP_SET_DEFAULT END ///
        "
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a() {
        let mut map_src: MapSrc<i32> = MapSrc::default();
        map_src.add(256, 16);

        let program_source = map_src.generate_map_reset_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_b() {
        let mut map_src: MapSrc<i32> = MapSrc::default();
        map_src.add(256, 8);
        map_src.add(512, 16);

        let program_source = map_src.generate_map_reset_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_c() {
        let mut map_src: MapSrc<i32> = MapSrc::default();
        map_src.add(256, 8);
        map_src.add(512, 32);
        map_src.add(1024, 16);

        let program_source = map_src.generate_map_reset_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }
}
