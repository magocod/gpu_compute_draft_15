use crate::config::ClTypeTrait;
use crate::map::config::MapSrc;
use crate::map::kernel::common_replace;

const MAP_REORDER_FOR_BLOCK_KERNEL: &str = r#"
    kernel void check_map_keys_to_reorder__BLOCK_NAME(
        const uint map_id,
        global int* indices_output,
        global int* result_output
        ) {
        int i = get_global_id(0);

        bool is_empty = is_map_key_empty__BLOCK_NAME(map_id, i);

        if (is_empty == true) {
            indices_output[i] = REORDER_KEY_EMPTY;
            result_output[i] = REORDER_KEY_EMPTY;
        } else {
            indices_output[i] = i;
            result_output[i] = i;
        }
    }


    kernel void map_reorder_for_block__BLOCK_NAME(
        queue_t q0,
        const uint map_id,
        const int enqueue_kernel_output_index,
        global int* indices_output,
        global int* result_output,
        global int* enqueue_kernel_output
        ) {
        int i = get_global_id(0);

        clk_event_t evt0;

        enqueue_kernel_output[CMQ_CHECK_MAP_KEYS_TO_REORDER + enqueue_kernel_output_index] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(MAP_CAPACITY, VALUE_DEVICE_LOCAL_WORK_SIZE),
            0,
            NULL,
            &evt0,
            ^{
               check_map_keys_to_reorder__BLOCK_NAME(
                 map_id,
                 indices_output,
                 result_output
               );
            }
        );

        enqueue_kernel_output[CMQ_EXECUTE_MAP_REORDER + enqueue_kernel_output_index] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(1, 1),
            1,
            &evt0,
            NULL,
            ^{
               execute_map_reorder_for_block__BLOCK_NAME(
                 q0,
                 map_id,
                 enqueue_kernel_output_index,
                 result_output,
                 enqueue_kernel_output
               );
            }
        );

        release_event(evt0);
    }
    "#;

const EXECUTE_MAP_REORDER_FOR_BLOCK_KERNEL: &str = r#"

    kernel void calculate_map_reordering__BLOCK_NAME(
        const uint map_id,
        global int* result_output
        ) {

        for (int index = 0; index < MAP_CAPACITY; index++) {
            int current_key_index = result_output[index];

            if (current_key_index != REORDER_KEY_EMPTY) {
                result_output[index] = REORDER_KEY_UNMOVED;
                continue;
            }

            for (int i = MAP_CAPACITY ; i-- > 0; ) {
                if (i == index) {
                    break;
                }

                int last_key_index = result_output[i];

                if (last_key_index == REORDER_KEY_EMPTY) {
                    continue;
                }

                result_output[index] = result_output[i];
                result_output[i] = -1;

                break;
            }

        }

    }

    kernel void confirm_map_reordering__BLOCK_NAME(
        const uint map_id,
        global int* result_output
        ) {
        int to_entry_index = get_global_id(0);
        int from_entry_index = result_output[to_entry_index];

        if (from_entry_index >= 0) {

            for (int index = 0; index < DEFAULT_MAP_KEY_LENGTH; index++) {
                map_keys__BLOCK_NAME[map_id][to_entry_index][index] = map_keys__BLOCK_NAME[map_id][from_entry_index][index];
                map_keys__BLOCK_NAME[map_id][from_entry_index][index] = CL_DEFAULT_VALUE;
            }

            for (int index = 0; index < MAP_VALUE_LEN; index++) {
                map_values__BLOCK_NAME[map_id][to_entry_index][index] = map_values__BLOCK_NAME[map_id][from_entry_index][index];
                map_values__BLOCK_NAME[map_id][from_entry_index][index] = CL_DEFAULT_VALUE;
            }

        }
    }

    kernel void execute_map_reorder_for_block__BLOCK_NAME(
        queue_t q0,
        const uint map_id,
        const int enqueue_kernel_output_index,
        global int* result_output,
        global int* enqueue_kernel_output
        ) {
        int i = get_global_id(0);

        clk_event_t evt0;

        enqueue_kernel_output[CMQ_CALCULATE_MAP_REORDERING + enqueue_kernel_output_index] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(1, 1),
            0,
            NULL,
            &evt0,
            ^{
               calculate_map_reordering__BLOCK_NAME(
                 map_id,
                 result_output
               );
            }
        );

        enqueue_kernel_output[CMQ_CONFIRM_MAP_REORDERING + enqueue_kernel_output_index] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(MAP_CAPACITY, VALUE_DEVICE_LOCAL_WORK_SIZE),
            1,
            &evt0,
            NULL,
            ^{
               confirm_map_reordering__BLOCK_NAME(
                 map_id,
                 result_output
               );
            }
        );

        release_event(evt0);
    }
    "#;

impl<T: ClTypeTrait> MapSrc<T> {
    pub fn generate_map_reorder_program_src(&self) -> String {
        let mut execute_map_reorder_kernels = String::new();
        let mut reorder_map_kernels = String::new();

        let map_blocks = self.get_configs();

        for config in map_blocks.iter() {
            let template = common_replace(EXECUTE_MAP_REORDER_FOR_BLOCK_KERNEL, config);
            execute_map_reorder_kernels.push_str(&template);

            let template = common_replace(MAP_REORDER_FOR_BLOCK_KERNEL, config);
            reorder_map_kernels.push_str(&template);
        }

        format!(
            "
    /// - MAP_REORDER START ///

    /// constants
    const int CMQ_CHECK_MAP_KEYS_TO_REORDER = 0;
    const int CMQ_EXECUTE_MAP_REORDER = 1;
    const int CMQ_CALCULATE_MAP_REORDERING = 2;
    const int CMQ_CONFIRM_MAP_REORDERING = 3;

    const int REORDER_KEY_EMPTY = -1;
    const int REORDER_KEY_UNMOVED = -2;

    /// globals
    // ...

    /// kernels
    {execute_map_reorder_kernels}
    {reorder_map_kernels}

    /// - MAP_REORDER END ///
        "
        )
    }

    pub fn add_map_reorder_program_src(&mut self) -> &mut Self {
        let src = self.generate_map_reorder_program_src();
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

        let program_source = map_src.generate_map_reorder_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_b() {
        let mut map_src: MapSrc<i32> = MapSrc::new(2);
        map_src.add(256, 8);
        map_src.add(512, 16);

        let program_source = map_src.generate_map_reorder_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_c() {
        let mut map_src: MapSrc<i32> = MapSrc::new(2);
        map_src.add(256, 8);
        map_src.add(512, 32);
        map_src.add(1024, 16);

        let program_source = map_src.generate_map_reorder_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }
}
