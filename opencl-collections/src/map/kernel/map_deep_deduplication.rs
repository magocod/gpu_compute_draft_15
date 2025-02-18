use crate::config::ClTypeTrait;
use crate::map::config::{check_local_work_size, MapSrc};
use crate::map::kernel::common_replace;

const TMP_DUPLICATES_INDICES: &str = r#"
    // BLOCK_NAME

    __global int tmp_for_map_deep_deduplication__BLOCK_NAME[TMP_LEN][MAP_CAPACITY];

    kernel void get_tmp_for_map_deep_deduplication__BLOCK_NAME(
        const uint map_id,
        global int* output
        ) {
        int i = get_global_id(0);
        output[i] = tmp_for_map_deep_deduplication__BLOCK_NAME[map_id][i];
    }
    "#;

const RESET_TMP_FN: &str = r#"
    void reset_tmp_for_map_deep_deduplication(const uint map_id) {
        RESET_TMP_FN_BODY
    }
    "#;

const RESET_TMP_FN_BODY: &str = r#"
        // BLOCK_NAME
        for (int index = 0; index < MAP_CAPACITY; index++) {
            tmp_for_map_deep_deduplication__BLOCK_NAME[map_id][index] = NO_DUPLICATE_KEY_DEF_2;
        }
    "#;

const COMPARE_MAP_KEYS_FUNCTION: &str = r#"
    bool is_map_keys_are_equal__from__FIRST_BLOCK__to__SECOND_BLOCK(uint map_id, int first_entry_index, int second_entry_index) {

        for (int key_index = 0; key_index < DEFAULT_MAP_KEY_LENGTH; key_index++) {
            if (map_keys__FIRST_BLOCK[map_id][first_entry_index][key_index] != map_keys__SECOND_BLOCK[map_id][second_entry_index][key_index]) {
                return false;
            }
        }

        return true;
    }
    "#;

const CHECK_KEYS_BODY: &str = r#"
            // BLOCK_NAME

            // last index no check next indices
            if (i != (MAP_CAPACITY - 1)) {

                for (int index = i + 1; index < MAP_CAPACITY; index++) {

                    bool is_equal = is_map_keys_are_equal__from__BLOCK_NAME__to__BLOCK_NAME(
                        map_id,
                        i, // first_entry_index
                        index // second_entry_index
                    );

                    if (is_equal == true) {
                        tmp_for_map_deep_deduplication__BLOCK_NAME[map_id][index] = DUPLICATE_KEY_DEF_2;
                    }

                }

            }
    "#;

const CHECK_KEYS_WITH_OTHER_BLOCK: &str = r#"
            // SECOND_BLOCK
            for (int index = 0; index < SECOND_M_CAPACITY; index++) {

                int current_remove_status = tmp_for_map_deep_deduplication__SECOND_BLOCK[map_id][index];

                if (current_remove_status == DUPLICATE_KEY_DEF_2) {
                    continue;
                }

                bool is_equal = is_map_keys_are_equal__from__FIRST_BLOCK__to__SECOND_BLOCK(
                    map_id,
                    i, // first_entry_index
                    index // second_entry_index
                );

                if (is_equal == true) {
                    tmp_for_map_deep_deduplication__SECOND_BLOCK[map_id][index] = DUPLICATE_KEY_DEF_2;
                }

            }
    "#;

const CHECK_SUB_KERNEL: &str = r#"
    COMPARE_MAP_KEYS_FUNCTIONS

    kernel void deep_check_duplicate_map_keys__BLOCK_NAME(
        const uint map_id
        ) {
        int i = get_global_id(0);

        bool is_empty = is_map_key_empty__BLOCK_NAME(map_id, i);
        int remove_status = tmp_for_map_deep_deduplication__BLOCK_NAME[map_id][i];

        if (is_empty == false && remove_status == NO_DUPLICATE_KEY_DEF_2) {
            CHECK_KEYS_BODY
        }
    }
    "#;

const CONFIRM_REMOVE_BODY: &str = r#"
        // BLOCK_NAME

        if (MAP_CAPACITY > i) {
            if (tmp_for_map_deep_deduplication__BLOCK_NAME[map_id][i] == DUPLICATE_KEY_DEF_2) {

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

    "#;

const CONFIRM_SUB_KERNEL: &str = r#"
    kernel void confirm_deep_remove_map_keys_duplicates(
        const uint map_id
        ) {
        int i = get_global_id(0);

        CONFIRM_REMOVE_BODY

    }
    "#;

const MAP_DEEP_DEDUPLICATION_MAIN_KERNEL: &str = r#"
    kernel void map_deep_deduplication(
        queue_t q0,
        const uint map_id,
        global int* enqueue_kernel_output
        ) {

        // reset
        reset_tmp_for_map_deep_deduplication(map_id);

        ENQUEUE_KERNEL_BODY

    }
    "#;

const MAP_DEEP_DEDUPLICATION_CHILD_KERNEL_DEF: &str = r#"
    kernel void map_deep_deduplication__child__BLOCK_NAME(
        queue_t q0,
        const uint map_id,
        global int* enqueue_kernel_output
    );
    "#;

const MAP_DEEP_DEDUPLICATION_CHILD_KERNEL: &str = r#"
    kernel void map_deep_deduplication__child__BLOCK_NAME(
        queue_t q0,
        const uint map_id,
        global int* enqueue_kernel_output
        ) {

        ENQUEUE_KERNEL_BODY

    }
    "#;

const CONTINUE_ENQUEUE_KERNEL: &str = r#"
        clk_event_t evt0;

        enqueue_kernel_output[CMQ_CHECK_KEYS_DUPLICATES] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(MAP_CAPACITY, CAPACITY_DEVICE_LOCAL_WORK_SIZE),
            0,
            NULL,
            &evt0,
            ^{
               deep_check_duplicate_map_keys__BLOCK_NAME(
                 map_id
               );
            }
        );

        enqueue_kernel_output[CMQ_NEXT_OR_CONFIRM] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(1, 1),
            1,
            &evt0,
            NULL,
            ^{
               map_deep_deduplication__child__NEXT_BLOCK(
                 q0,
                 map_id,
                 enqueue_kernel_output
               );
            }
        );

        release_event(evt0);
    "#;

const CONFIRM_ENQUEUE_KERNEL: &str = r#"
        clk_event_t evt0;

        enqueue_kernel_output[CMQ_CHECK_KEYS_DUPLICATES] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(MAP_CAPACITY, CAPACITY_DEVICE_LOCAL_WORK_SIZE),
            0,
            NULL,
            &evt0,
            ^{
               deep_check_duplicate_map_keys__BLOCK_NAME(
                 map_id
               );
            }
        );

        enqueue_kernel_output[CMQ_CONFIRM_REMOVE] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(CONFIRM_GLOBAL_WORK_SIZE, CONFIRM_LOCAL_WORK_SIZE),
            1,
            &evt0,
            NULL,
            ^{
               confirm_deep_remove_map_keys_duplicates(
                 map_id
               );
            }
        );

        release_event(evt0);
    "#;

const MAP_DEEP_DEDUPLICATION_FOR_BLOCK_KERNEL: &str = r#"
    kernel void map_deep_deduplication_for_block__BLOCK_NAME(
        queue_t q0,
        const uint map_id,
        global int* enqueue_kernel_output
        ) {
        int i = get_global_id(0);

        // reset
        reset_tmp_for_map_deep_deduplication(map_id);

        clk_event_t evt0;

        enqueue_kernel_output[CMQ_CHECK_MAP_KEYS_DUPLICATES] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(MAP_CAPACITY, CAPACITY_DEVICE_LOCAL_WORK_SIZE),
            0,
            NULL,
            &evt0,
            ^{
               deep_check_duplicate_map_keys__BLOCK_NAME(
                 map_id
               );
            }
        );

        enqueue_kernel_output[CMQ_CONFIRM_REMOVE_DUPLICATES] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(CONFIRM_GLOBAL_WORK_SIZE, CONFIRM_LOCAL_WORK_SIZE),
            1,
            &evt0,
            NULL,
            ^{
               confirm_deep_remove_map_keys_duplicates(
                 map_id
               );
            }
        );

        release_event(evt0);
    }
    "#;

impl<T: ClTypeTrait> MapSrc<T> {
    pub fn generate_map_deep_deduplication_program_src(&self) -> String {
        let mut tmp_duplicates_indices_array = String::new();
        let mut reset_tmp_fn_body = String::new();

        let mut deep_check_duplicate_keys_kernel = String::new();
        let mut confirm_remove_duplicate_body = String::new();

        let mut map_deep_deduplication_child_kernels_def = String::new();
        let mut map_deep_deduplication_kernels = String::new();

        let mut map_deep_deduplication_kernels_for_block = String::new();

        let map_blocks = self.get_configs();

        let confirm_global_work_size = self.get_max_capacity();
        let confirm_local_work_size = check_local_work_size(confirm_global_work_size);

        for config in map_blocks.iter() {
            let template = common_replace(TMP_DUPLICATES_INDICES, config)
                .replace("TMP_LEN", &self.get_total_maps().to_string());
            tmp_duplicates_indices_array.push_str(&template);

            let template = common_replace(RESET_TMP_FN_BODY, config);
            reset_tmp_fn_body.push_str(&template);

            let mut check_duplicates_body = String::new();
            let mut compare_map_keys_functions = String::new();

            for block_config in map_blocks.iter() {
                let first_block = &config.name;
                let second_block = &block_config.name;

                let t = common_replace(COMPARE_MAP_KEYS_FUNCTION, config)
                    .replace("FIRST_BLOCK", first_block)
                    .replace("SECOND_BLOCK", second_block);
                compare_map_keys_functions.push_str(&t);

                if block_config.value_len == config.value_len {
                    let t = common_replace(CHECK_KEYS_BODY, config)
                        .replace("FIRST_BLOCK", first_block)
                        .replace("SECOND_BLOCK", second_block);
                    check_duplicates_body.push_str(&t);
                } else {
                    let t = common_replace(CHECK_KEYS_WITH_OTHER_BLOCK, config)
                        .replace("SECOND_M_CAPACITY", &block_config.capacity.to_string())
                        .replace("FIRST_BLOCK", first_block)
                        .replace("SECOND_BLOCK", second_block);
                    check_duplicates_body.push_str(&t);
                }
            }

            let template = common_replace(CHECK_SUB_KERNEL, config)
                .replace("CHECK_KEYS_BODY", &check_duplicates_body)
                .replace("COMPARE_MAP_KEYS_FUNCTIONS", &compare_map_keys_functions);
            deep_check_duplicate_keys_kernel.push_str(&template);

            let template = common_replace(CONFIRM_REMOVE_BODY, config);
            confirm_remove_duplicate_body.push_str(&template);
        }

        for (i, config) in map_blocks.iter().enumerate() {
            let enqueue_section = if i == (map_blocks.len() - 1) {
                common_replace(CONFIRM_ENQUEUE_KERNEL, config)
                    .replace(
                        "CONFIRM_GLOBAL_WORK_SIZE",
                        &confirm_global_work_size.to_string(),
                    )
                    .replace(
                        "CONFIRM_LOCAL_WORK_SIZE",
                        &confirm_local_work_size.to_string(),
                    )
            } else {
                let next_block = map_blocks.get(i + 1).unwrap();
                common_replace(CONTINUE_ENQUEUE_KERNEL, config)
                    .replace("NEXT_BLOCK", &next_block.name)
            };

            let kernel = if i == 0 {
                common_replace(MAP_DEEP_DEDUPLICATION_MAIN_KERNEL, config)
                    .replace("ENQUEUE_KERNEL_BODY", &enqueue_section)
            } else {
                let template = common_replace(MAP_DEEP_DEDUPLICATION_CHILD_KERNEL_DEF, config);
                map_deep_deduplication_child_kernels_def.push_str(&template);

                common_replace(MAP_DEEP_DEDUPLICATION_CHILD_KERNEL, config)
                    .replace("ENQUEUE_KERNEL_BODY", &enqueue_section)
            };

            map_deep_deduplication_kernels.push_str(&kernel);

            let template = common_replace(MAP_DEEP_DEDUPLICATION_FOR_BLOCK_KERNEL, config)
                .replace(
                    "CONFIRM_GLOBAL_WORK_SIZE",
                    &confirm_global_work_size.to_string(),
                )
                .replace(
                    "CONFIRM_LOCAL_WORK_SIZE",
                    &confirm_local_work_size.to_string(),
                );

            map_deep_deduplication_kernels_for_block.push_str(&template);
        }

        let confirm_deep_remove_map_keys_kernels =
            CONFIRM_SUB_KERNEL.replace("CONFIRM_REMOVE_BODY", &confirm_remove_duplicate_body);

        let reset_tmp_fn = RESET_TMP_FN.replace("RESET_TMP_FN_BODY", &reset_tmp_fn_body);

        format!(
            "
    /// - MAP_DEEP_DEDUPLICATION START ///

    /// constants
    const int NO_DUPLICATE_KEY_DEF_2 = 0;
    const int DUPLICATE_KEY_DEF_2 = 1;

    const int CMQ_CHECK_MAP_KEYS_DUPLICATES = 0;
    const int CMQ_CONFIRM_REMOVE_DUPLICATES = 1;

    const int CMQ_CHECK_KEYS_DUPLICATES = 0;
    const int CMQ_NEXT_OR_CONFIRM = 1;
    const int CMQ_CONFIRM_REMOVE = 1;

    /// globals
    {tmp_duplicates_indices_array}

    /// kernels
    {reset_tmp_fn}

    {deep_check_duplicate_keys_kernel}
    {confirm_deep_remove_map_keys_kernels}

    {map_deep_deduplication_child_kernels_def}

    {map_deep_deduplication_kernels}

    {map_deep_deduplication_kernels_for_block}

    /// - MAP_DEEP_DEDUPLICATION END ///
        "
        )
    }

    pub fn add_map_deep_deduplication_program_src(&mut self) -> &mut Self {
        let src = self.generate_map_deep_deduplication_program_src();
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

        let program_source = map_src.generate_map_deep_deduplication_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_b() {
        let mut map_src: MapSrc<i32> = MapSrc::new(2);
        map_src.add(256, 8);
        map_src.add(512, 16);

        let program_source = map_src.generate_map_deep_deduplication_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_c() {
        let mut map_src: MapSrc<i32> = MapSrc::new(2);
        map_src.add(256, 8);
        map_src.add(512, 32);
        map_src.add(1024, 16);

        let program_source = map_src.generate_map_deep_deduplication_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }
}
