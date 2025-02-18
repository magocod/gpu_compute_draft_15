use crate::config::ClTypeTrait;
use crate::map::config::MapSrc;
use crate::map::kernel::common_replace;

const MAP_REMOVE_KERNEL: &str = r#"
    kernel void map_remove(
        queue_t q0,
        const uint map_id,
        global CL_TYPE* keys_input,
        global int* indices_output,
        global int* block_output,
        global int* enqueue_kernel_output
    ) {
        int i = get_global_id(0);

        int key_input_index = i * DEFAULT_MAP_KEY_LENGTH;
        int tmp_index = i * TOTAL_INDICES;
        int enqueue_kernel_output_index = i * TOTAL_QUEUES;
        
        reset_tmp_for_map_remove(map_id, tmp_index);

        clk_event_t evt0;

        enqueue_kernel_output[CMQ_COMPARE_KEY_DEF_3 + enqueue_kernel_output_index] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(MAP_CAPACITY, CAPACITY_DEVICE_LOCAL_WORK_SIZE),
            0,
            NULL,
            &evt0,
            ^{
               compare_map_key_in_block__BLOCK_NAME_def_5(
                  map_id,
                  key_input_index,
                  tmp_index,
                  keys_input
               );
            }
        );

        enqueue_kernel_output[CMQ_CONFIRM_REMOVE_ENTRY + enqueue_kernel_output_index] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(1, 1),
            1,
            &evt0,
            NULL,
            ^{
               confirm_remove__BLOCK_NAME(
                    q0,
                    map_id,
                    i, // parent_global_index
                    key_input_index,
                    tmp_index,
                    enqueue_kernel_output_index,
                    keys_input,
                    indices_output,
                    block_output,
                    enqueue_kernel_output
               );
            }
        );

        release_event(evt0);
    }
    "#;

const COMPARE_MAP_KEY_KERNEL: &str = r#"
    kernel void compare_map_key_in_block__BLOCK_NAME_def_5(
        const uint map_id,
        const int key_input_index,
        const int tmp_index,
        global CL_TYPE* key_input
        ) {
        int i = get_global_id(0);

        bool is_equal = is_map_key_is_equal_to_input__BLOCK_NAME(
            map_id,
            i, // entry_index
            key_input_index,
            key_input
        );

        if (is_equal == true) {
            int item_tmp_index = i + TMP_INDEX__BLOCK_NAME_def_6 + tmp_index;

            tmp_for_map_remove[map_id][item_tmp_index] = i;
        }
    }
    "#;

const CONFIRM_SEARCH_KERNEL: &str = r#"
    int check_matches_in_tmp__BLOCK_NAME_def_5(uint map_id, int tmp_index) {
        for (int index = 0; index < MAP_CAPACITY; index++) {
            int v = tmp_for_map_remove[map_id][index + TMP_INDEX__BLOCK_NAME_def_6 + tmp_index];
            if (v >= 0) {
                return index;
            }
        }

        return -3;
    }

    kernel void map_delete_one__BLOCK_NAME_def_3(
        const uint map_id,
        const int entry_index
        ) {
        int i = get_global_id(0);

        if (i < DEFAULT_MAP_KEY_LENGTH) {
            map_keys__BLOCK_NAME[map_id][entry_index][i] = CL_DEFAULT_VALUE;
        }
        map_values__BLOCK_NAME[map_id][entry_index][i] = CL_DEFAULT_VALUE;
    }

    kernel void confirm_remove__BLOCK_NAME(
        queue_t q0,
        const uint map_id,
        const int parent_global_index,
        const int key_input_index,
        const int tmp_index,
        const int enqueue_kernel_output_index,
        global CL_TYPE* keys_input,
        global int* indices_output,
        global int* block_output,
        global int* enqueue_kernel_output
        ) {

        int entry_index = check_matches_in_tmp__BLOCK_NAME_def_5(
            map_id,
            tmp_index
        );

        indices_output[parent_global_index] = entry_index;
        block_output[parent_global_index] = MAP_VALUE_LEN;

        int enqueue_kernel_continue_search = 0;
        int enqueue_kernel_remove = 0;

        COMMIT_SECTION

        // The remove of the value is done from another conditional,
        // to avoid overcomplicating the code.
        if (entry_index >= 0) {
            enqueue_kernel_remove = enqueue_kernel(
                q0,
                CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
                ndrange_1D(MAP_VALUE_LEN, VALUE_DEVICE_LOCAL_WORK_SIZE),
                ^{
                   map_delete_one__BLOCK_NAME_def_3(
                        map_id,
                        entry_index
                   );
                }
            );
        }

        enqueue_kernel_output[CMQ_CONTINUE_SEARCH_DEF_3 + enqueue_kernel_output_index] = enqueue_kernel_continue_search;
        enqueue_kernel_output[CMQ_REMOVE_ENTRY_3 + enqueue_kernel_output_index] = enqueue_kernel_remove;
    }
    "#;

const COMMIT_SECTION: &str = r#"
        if (entry_index == -3) {
            enqueue_kernel_continue_search = enqueue_kernel(
                q0,
                CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
                ndrange_1D(1, 1),
                ^{
                   map_remove__BLOCK_NAME(
                        q0,
                        map_id,
                        parent_global_index,
                        key_input_index,
                        tmp_index,
                        enqueue_kernel_output_index,
                        keys_input,
                        indices_output,
                        block_output,
                        enqueue_kernel_output
                   );
                }
            );
        }
    "#;

const MAP_REMOVE_BLOCK_KERNEL_DEF: &str = r#"
    kernel void map_remove__BLOCK_NAME(
        queue_t q0,
        const uint map_id,
        const int parent_global_index,
        const int key_input_index,
        const int tmp_index,
        const int enqueue_kernel_output_index,
        global CL_TYPE* keys_input,
        global int* indices_output,
        global int* block_output,
        global int* enqueue_kernel_output
    );
    "#;

const MAP_REMOVE_BLOCK_KERNEL: &str = r#"
    kernel void map_remove__BLOCK_NAME(
        queue_t q0,
        const uint map_id,
        const int parent_global_index,
        const int key_input_index,
        const int tmp_index,
        const int enqueue_kernel_output_index,
        global CL_TYPE* keys_input,
        global int* indices_output,
        global int* block_output,
        global int* enqueue_kernel_output
    ) {
        clk_event_t evt0;

        enqueue_kernel_output[CMQ_COMPARE_KEY_DEF_3 + enqueue_kernel_output_index] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(MAP_CAPACITY, CAPACITY_DEVICE_LOCAL_WORK_SIZE),
            0,
            NULL,
            &evt0,
            ^{
               compare_map_key_in_block__BLOCK_NAME_def_5(
                  map_id,
                  key_input_index,
                  tmp_index,
                  keys_input
               );
            }
        );

        enqueue_kernel_output[CMQ_CONFIRM_REMOVE_ENTRY + enqueue_kernel_output_index] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(1, 1),
            1,
            &evt0,
            NULL,
            ^{
               confirm_remove__BLOCK_NAME(
                    q0,
                    map_id,
                    parent_global_index,
                    key_input_index,
                    tmp_index,
                    enqueue_kernel_output_index,
                    keys_input,
                    indices_output,
                    block_output,
                    enqueue_kernel_output
               );
            }
        );

        release_event(evt0);
    }
    "#;

const GLOBAL_TMP_ARRAY: &str = r#"
    global int tmp_for_map_remove[TOTAL_MAPS][TMP_LEN];

    kernel void get_tmp_for_map_remove(
        const uint map_id,
        global int* output
        ) {
        int i = get_global_id(0);
        output[i] = tmp_for_map_remove[map_id][i];
    }

    void reset_tmp_for_map_remove(uint map_id, int tmp_index) {
        RESET_TMP_ARRAY_BODY
    }
    "#;

const RESET_TMP: &str = r#"
        // reset BLOCK_NAME
        for (int index = 0; index < MAP_CAPACITY; index++) {
            tmp_for_map_remove[map_id][index + TMP_INDEX__BLOCK_NAME_def_6 + tmp_index] = -3;
        }
        "#;

const CONST_DEF: &str = r#"
    const int TMP_INDEX__BLOCK_NAME_def_6 = BLOCK_INDEX;
    "#;

impl<T: ClTypeTrait> MapSrc<T> {
    pub fn generate_map_remove_program_src(&self, max_find_work_size: usize) -> String {
        let map_blocks = self.get_configs();
        let total_blocks = map_blocks.len();

        let total_queues = 4;
        let total_indices = self.get_maximum_assignable_keys();

        // const definitions
        let mut const_def = String::new();

        let mut reset_tmp_array_body = String::new();

        let mut compare_key_kernels = String::new();
        let mut confirm_search_kernels = String::new();

        let mut map_remove_kernels_def = String::new();
        let mut map_remove_kernels = String::new();

        // FIXME improve code iteration and simplicity
        for (i, config) in map_blocks.iter().enumerate() {
            let tmp_index: usize = map_blocks[0..i].iter().map(|x| x.capacity).sum();

            let block_const_def =
                common_replace(CONST_DEF, config).replace("BLOCK_INDEX", &tmp_index.to_string());
            const_def.push_str(&block_const_def);

            let template = common_replace(COMPARE_MAP_KEY_KERNEL, config);
            compare_key_kernels.push_str(&template);

            let commit_section = if (i + 1) == total_blocks {
                String::from("// there are no more blocks ...") // last block
            } else {
                let next_config = &map_blocks[i + 1];
                COMMIT_SECTION.replace("BLOCK_NAME", &next_config.name)
            };

            let template = common_replace(CONFIRM_SEARCH_KERNEL, config)
                .replace("COMMIT_SECTION", &commit_section);
            confirm_search_kernels.push_str(&template);

            if i == 0 {
                // ...
                for c in map_blocks {
                    let template = common_replace(RESET_TMP, c);
                    reset_tmp_array_body.push_str(&template);
                }

                let template = common_replace(MAP_REMOVE_KERNEL, config)
                    .replace("TOTAL_INDICES", &total_indices.to_string())
                    .replace("TOTAL_QUEUES", &total_queues.to_string());

                map_remove_kernels.push_str(&template);
            } else {
                let template = common_replace(MAP_REMOVE_BLOCK_KERNEL, config);
                map_remove_kernels.push_str(&template);

                let template = common_replace(MAP_REMOVE_BLOCK_KERNEL_DEF, config)
                    .replace("TOTAL_INDICES", &total_indices.to_string())
                    .replace("TOTAL_QUEUES", &total_queues.to_string());
                map_remove_kernels_def.push_str(&template);
            }
        }

        let tmp_len = max_find_work_size * total_indices;

        let global_tmp_array = GLOBAL_TMP_ARRAY
            .replace("TMP_LEN", &tmp_len.to_string())
            .replace("TOTAL_MAPS", &self.get_total_maps().to_string())
            .replace("RESET_TMP_ARRAY_BODY", &reset_tmp_array_body);

        format!(
            "
    /// - MAP_REMOVE START ///

    /// constants

    const int CMQ_COMPARE_KEY_DEF_3 = 0;
    const int CMQ_CONFIRM_REMOVE_ENTRY = 1;
    const int CMQ_CONTINUE_SEARCH_DEF_3 = 2;
    const int CMQ_REMOVE_ENTRY_3 = 3;

    {const_def}

    {map_remove_kernels_def}

    /// globals
    {global_tmp_array}

    /// kernels
    {compare_key_kernels}
    {confirm_search_kernels}

    {map_remove_kernels}

    /// - MAP_REMOVE END ///
        "
        )
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

        let program_source = map_src.generate_map_remove_program_src(16);
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_b() {
        let mut map_src: MapSrc<i32> = MapSrc::new(2);
        map_src.add(256, 8);
        map_src.add(512, 16);

        let program_source = map_src.generate_map_remove_program_src(8);
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_c() {
        let mut map_src: MapSrc<i32> = MapSrc::new(2);
        map_src.add(256, 8);
        map_src.add(512, 8);
        map_src.add(1024, 16);

        let program_source = map_src.generate_map_remove_program_src(2);
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }
}
