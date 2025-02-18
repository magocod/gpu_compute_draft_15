use crate::config::ClTypeTrait;
use crate::map::config::MapSrc;
use crate::map::kernel::common_replace;

const MAP_DEDUPLICATION_KERNEL: &str = r#"
    kernel void map_deduplication(
        queue_t q0,
        const uint map_id,
        global int* enqueue_kernel_output
        ) {

        // KERNEL_BODY

    }
    "#;

const MAP_DEDUPLICATION_FOR_BLOCK_ENQUEUE_KERNEL: &str = r#"
        // BLOCK_NAME

        // int enqueue_kernel_output_index = BLOCK_INDEX * TOTAL_COMMAND_QUEUES;

        enqueue_kernel_output[CMQ_MAP_DEDUPLICATION_FOR_BLOCK + Q_OUTPUT_INDEX] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(1, 1),
            ^{
               map_deduplication_for_block__BLOCK_NAME(
                    q0,
                    map_id,
                    Q_OUTPUT_INDEX, // enqueue_kernel_output_index
                    enqueue_kernel_output
               );
            }
        );
     "#;

const MAP_DEDUPLICATION_FOR_BLOCK_KERNEL: &str = r#"
    kernel void check_duplicate_map_keys__BLOCK_NAME(
        const uint map_id
        ) {
        int i = get_global_id(0);

        bool is_empty = is_map_key_empty__BLOCK_NAME(map_id, i);

        if (is_empty == false) {
            for (int index = i + 1; index < MAP_CAPACITY; index++) {

                bool is_equal = is_map_keys_are_equal__BLOCK_NAME(
                    map_id,
                    i, // first_entry_index
                    index // second_entry_index
                );

                if (is_equal == true) {
                    tmp_for_map_deduplication__BLOCK_NAME[map_id][index] = DUPLICATE_KEY;
                }

            }
        }
    }

    kernel void confirm_remove_map_keys_duplicates__BLOCK_NAME(
        const uint map_id
        ) {
        int i = get_global_id(0);

        if (tmp_for_map_deduplication__BLOCK_NAME[map_id][i] == DUPLICATE_KEY) {

            // remove key
            for (int key_index = 0; key_index < DEFAULT_MAP_KEY_LENGTH; key_index++) {
                map_keys__BLOCK_NAME[map_id][i][key_index] = CL_DEFAULT_VALUE;
            }

            // remove value
            for (int key_index = 0; key_index < MAP_VALUE_LEN; key_index++) {
                map_values__BLOCK_NAME[map_id][i][key_index] = CL_DEFAULT_VALUE;
            }

        }

    }

    kernel void map_deduplication_for_block__BLOCK_NAME(
        queue_t q0,
        const uint map_id,
        const int enqueue_kernel_output_index,
        global int* enqueue_kernel_output
        ) {
        int i = get_global_id(0);

        // reset
        for (int index = 0; index < MAP_CAPACITY; index++) {
            tmp_for_map_deduplication__BLOCK_NAME[map_id][index] = NO_DUPLICATE_KEY;
        }

        clk_event_t evt0;

        enqueue_kernel_output[CMQ_CHECK_DUPLICATES_MAP_KEYS + enqueue_kernel_output_index] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(MAP_CAPACITY - 1, CAPACITY_DEVICE_LOCAL_WORK_SIZE - 1),
            0,
            NULL,
            &evt0,
            ^{
               check_duplicate_map_keys__BLOCK_NAME(
                 map_id
               );
            }
        );

        enqueue_kernel_output[CMQ_CONFIRM_REMOVE_DUPLICATES_MAP_KEYS + enqueue_kernel_output_index] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(MAP_CAPACITY, CAPACITY_DEVICE_LOCAL_WORK_SIZE),
            1,
            &evt0,
            NULL,
            ^{
               confirm_remove_map_keys_duplicates__BLOCK_NAME(
                 map_id
               );
            }
        );

        release_event(evt0);
    }
    "#;

// todo explain
const TMP_DUPLICATES_INDICES: &str = r#"
    // BLOCK_NAME

    __global int tmp_for_map_deduplication__BLOCK_NAME[TOTAL_MAPS][MAP_CAPACITY];

    kernel void get_tmp_for_map_deduplication__BLOCK_NAME(
        const uint map_id,
        global int* output
        ) {
        int i = get_global_id(0);
        output[i] = tmp_for_map_deduplication__BLOCK_NAME[map_id][i];
    }
    "#;

impl<T: ClTypeTrait> MapSrc<T> {
    pub fn generate_map_deduplication_program_src(&self) -> String {
        let mut tmp_array = String::new();

        let mut map_deduplication_for_block_enqueue_kernel = String::new();
        let mut map_deduplication_for_block_kernels = String::new();

        let map_blocks = self.get_configs();

        let total_command_queues = 3;

        for (i, config) in map_blocks.iter().enumerate() {
            let enqueue_kernel_output_index = i * total_command_queues;

            let template = common_replace(TMP_DUPLICATES_INDICES, config)
                .replace("TOTAL_MAPS", &self.get_total_maps().to_string());
            tmp_array.push_str(&template);

            let template = common_replace(MAP_DEDUPLICATION_FOR_BLOCK_KERNEL, config);
            map_deduplication_for_block_kernels.push_str(&template);

            let template = common_replace(MAP_DEDUPLICATION_FOR_BLOCK_ENQUEUE_KERNEL, config)
                .replace("Q_OUTPUT_INDEX", &enqueue_kernel_output_index.to_string())
                .replace("BLOCK_INDEX", &i.to_string());
            map_deduplication_for_block_enqueue_kernel.push_str(&template);
        }

        let map_deduplication_kernel = MAP_DEDUPLICATION_KERNEL
            .replace("KERNEL_BODY", &map_deduplication_for_block_enqueue_kernel)
            .replace("TOTAL_COMMAND_QUEUES", &total_command_queues.to_string());

        format!(
            "
    /// - MAP_DEDUPLICATION START ///

    /// constants
    const int NO_DUPLICATE_KEY = 0;
    const int DUPLICATE_KEY = 1;

    const int CMQ_CHECK_DUPLICATES_MAP_KEYS = 0;
    const int CMQ_CONFIRM_REMOVE_DUPLICATES_MAP_KEYS = 1;
    const int CMQ_MAP_DEDUPLICATION_FOR_BLOCK = 2;

    /// globals
    {tmp_array}

    /// functions
    // ...

    /// kernels
    {map_deduplication_for_block_kernels}

    {map_deduplication_kernel}

    /// - MAP_DEDUPLICATION END ///
        "
        )
    }

    pub fn add_map_deduplication_program_src(&mut self) -> &mut Self {
        let src = self.generate_map_deduplication_program_src();
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

        let program_source = map_src.generate_map_deduplication_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_b() {
        let mut map_src: MapSrc<i32> = MapSrc::new(2);
        map_src.add(256, 8);
        map_src.add(512, 16);

        let program_source = map_src.generate_map_deduplication_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_c() {
        let mut map_src: MapSrc<i32> = MapSrc::new(2);
        map_src.add(256, 8);
        map_src.add(512, 32);
        map_src.add(1024, 16);

        let program_source = map_src.generate_map_deduplication_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }
}
