use crate::config::ClTypeTrait;
use crate::map::config::MapSrc;
use crate::map::kernel::common_replace;

const MAP_INSERT_KERNEL: &str = r#"
    kernel void map_insert(
        queue_t q0,
        PIPES_ARGS
        const uint map_id,
        global CL_TYPE *keys_input,
        global CL_TYPE *values_input,
        global int *values_lens_input,
        global int *indices_output,
        global int *block_output,
        global int *enqueue_kernel_output
    ) {
        int i = get_global_id(0);

        int key_input_index = i * DEFAULT_MAP_KEY_LENGTH;
        int value_input_index = i * MAX_VALUE_LEN;
        int tmp_index = i * TOTAL_INDICES;
        int enqueue_kernel_output_index = i * TOTAL_QUEUES;

        reset_tmp_for_map_insert(map_id, tmp_index);

        clk_event_t evt0;

        enqueue_kernel_output[CMQ_COMPARE_KEY_IN_BLOCKS + enqueue_kernel_output_index] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(1, 1),
            0,
            NULL,
            &evt0,
            ^{
               compare_map_key_in_blocks(
                    q0,
                    map_id,
                    key_input_index,
                    tmp_index,
                    enqueue_kernel_output_index,
                    keys_input,
                    enqueue_kernel_output
               );
            }
        );

        enqueue_kernel_output[CMQ_CONFIRM_MAP_INSERT + enqueue_kernel_output_index] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(1, 1),
            1,
            &evt0,
            NULL,
            ^{
               confirm_map_insert(
                    q0,
                    PIPES_REFS
                    map_id,
                    i, // parent_global_index,
                    key_input_index,
                    value_input_index,
                    tmp_index,
                    enqueue_kernel_output_index,
                    keys_input,
                    values_input,
                    values_lens_input,
                    indices_output,
                    block_output,
                    enqueue_kernel_output
               );
            }
        );

        release_event(evt0);
    }
    "#;

const COMPARE_MAP_KEY_MAIN_KERNEL: &str = r#"
    kernel void compare_map_key_in_blocks(
        queue_t q0,
        const uint map_id,
        const int key_input_index,
        const int tmp_index,
        const int enqueue_kernel_output_index,
        global CL_TYPE *keys_input,
        global int *enqueue_kernel_output
        ) {

        KERNEL_BODY
    }
"#;

const COMPARE_MAP_KEY_ENQUEUE_KERNEL: &str = r#"
        enqueue_kernel_output[CMQ_COMPARE_KEY__BLOCK_NAME + enqueue_kernel_output_index] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(MAP_CAPACITY, CAPACITY_DEVICE_LOCAL_WORK_SIZE),
            ^{
               compare_map_key_in_block__BLOCK_NAME_def_3(
                  map_id,
                  key_input_index,
                  tmp_index,
                  keys_input
               );
            }
        );
"#;

const COMPARE_MAP_KEY_KERNEL: &str = r#"
    kernel void compare_map_key_in_block__BLOCK_NAME_def_3(
        const uint map_id,
        const int key_input_index,
        const int tmp_index,
        global CL_TYPE *key_input
        ) {
        int i = get_global_id(0);

        bool is_equal = is_map_key_is_equal_to_input__BLOCK_NAME(
            map_id,
            i, // entry_index
            key_input_index,
            key_input
        );

        if (is_equal == true) {
            int item_tmp_index = i + TMP_INDEX__BLOCK_NAME_def_4 + tmp_index;

            tmp_for_map_insert[map_id][item_tmp_index] = i;
        }
    }
    "#;

const STRUCT_DEF: &str = r#"
    struct MapInsertResult {
        int entry_index;
        int map_value_len;
        int enqueue_kernel_result;
        // ...
        int previous_entry_index;
        int previous_map_value_len;
    };
    "#;

const CONFIRM_MAP_INSERT_KERNEL: &str = r#"
    kernel void confirm_map_insert(
        queue_t q0,
        PIPES_ARGS
        const uint map_id,
        const int parent_global_index,
        const int key_input_index,
        const int value_input_index,
        const int tmp_index,
        const int enqueue_kernel_output_index,
        global CL_TYPE *keys_input,
        global CL_TYPE *values_input,
        global int *values_lens_input,
        global int *indices_output,
        global int *block_output,
        global int *enqueue_kernel_output
        ) {

        int value_len = values_lens_input[parent_global_index];

        struct MapInsertResult result;
        result.entry_index = -1;
        result.previous_entry_index = -3;
        result.previous_map_value_len = 0;

        map_try_insert_if_exist__BLOCK_NAME(
            q0,
            PIPES_REFS
            map_id,
            key_input_index,
            value_input_index,
            value_len,
            tmp_index,
            keys_input,
            values_input,
            &result
        );

        int remove_enqueue_kernel_result = 0;

        if (result.entry_index == -1) {

            map_try_insert__BLOCK_NAME(
                q0,
                PIPES_REFS
                map_id,
                key_input_index,
                value_input_index,
                value_len,
                tmp_index,
                keys_input,
                values_input,
                &result
            );

            // confirm release block

            switch (result.previous_map_value_len) {
                    case 0:
                              // pass
                              break;

                RELEASE_BLOCK_SWITCH
            }

        }

        indices_output[parent_global_index] = result.entry_index;
        block_output[parent_global_index] = result.map_value_len;

        enqueue_kernel_output[CMQ_CONFIRM_MAP_PUT + enqueue_kernel_output_index] = result.enqueue_kernel_result;
        enqueue_kernel_output[CMQ_CONFIRM_MAP_REMOVE + enqueue_kernel_output_index] = remove_enqueue_kernel_result;

    }
    "#;

const MAP_TRY_INSERT_FUNCTIONS_DEF: &str = r#"
    void map_try_insert_if_exist__BLOCK_NAME(
        queue_t q0,
        PIPES_ARGS
        const uint map_id,
        const int key_input_index,
        const int value_input_index,
        const int value_len,
        const int tmp_index,
        global CL_TYPE *keys_input,
        global CL_TYPE *values_input,
        struct MapInsertResult *result
    );

    void map_try_insert__BLOCK_NAME(
        queue_t q0,
        PIPES_ARGS
        const uint map_id,
        const int key_input_index,
        const int value_input_index,
        const int value_len,
        const int tmp_index,
        global CL_TYPE *keys_input,
        global CL_TYPE *values_input,
        struct MapInsertResult *result
    );
    "#;

const MAP_TRY_INSERT_IF_EXIST_FUNCTION: &str = r#"
    int check_matches_in_tmp__BLOCK_NAME_def_3(uint map_id, int tmp_index) {
        for (int index = 0; index < MAP_CAPACITY; index++) {
            int v = tmp_for_map_insert[map_id][index + TMP_INDEX__BLOCK_NAME_def_4 + tmp_index];
            if (v >= 0) {
                return index;
            }
        }

        return -3;
    }

    void map_try_insert_if_exist__BLOCK_NAME(
        queue_t q0,
        PIPES_ARGS
        const uint map_id,
        const int key_input_index,
        const int value_input_index,
        const int value_len,
        const int tmp_index,
        global CL_TYPE *keys_input,
        global CL_TYPE *values_input,
        struct MapInsertResult *result
        ) {

        // check if exist

        int entry_index = check_matches_in_tmp__BLOCK_NAME_def_3(
            map_id,
            tmp_index
        );

        if (entry_index >= 0) {
            
            // try update
            if (MAP_VALUE_LEN >= value_len) {

                result->entry_index = entry_index;
                result->map_value_len = MAP_VALUE_LEN;

                // result->enqueue_kernel_result = 0;
                result->enqueue_kernel_result = enqueue_kernel(
                    q0,
                    CLK_ENQUEUE_FLAGS_NO_WAIT,
                    ndrange_1D(MAP_VALUE_LEN, VALUE_DEVICE_LOCAL_WORK_SIZE),
                    ^{
                       map_set_one_value__BLOCK_NAME_def_2(
                            map_id,
                            entry_index,
                            value_input_index,
                            values_input
                       );
                    }
                );

                return;
            }

            // index to be released in case of relocation
            result->previous_entry_index = entry_index;
            result->previous_map_value_len = MAP_VALUE_LEN;

            return;
        }

        NEXT_BLOCK

    }
    "#;

const MAP_TRY_INSERT_IF_EXIST_FUNCTION_CALL: &str = r#"
        // next block
        map_try_insert_if_exist__BLOCK_NAME(
            q0,
            PIPES_REFS
            map_id,
            key_input_index,
            value_input_index,
            value_len,
            tmp_index,
            keys_input,
            values_input,
            result
        );
    "#;

const MAP_TRY_INSERT_FUNCTION: &str = r#"
    void map_try_insert__BLOCK_NAME(
        queue_t q0,
        PIPES_ARGS
        const uint map_id,
        const int key_input_index,
        const int value_input_index,
        const int value_len,
        const int tmp_index,
        global CL_TYPE *keys_input,
        global CL_TYPE *values_input,
        struct MapInsertResult *result
        ) {

        int entry_index = -1;

        // try save

        if (MAP_VALUE_LEN >= value_len) {

            read_pipe(pipe_BLOCK_NAME, &entry_index);

            if (entry_index >= 0) {

                result->entry_index = entry_index;
                result->map_value_len = MAP_VALUE_LEN;

                // result->enqueue_kernel_result = 0;
                result->enqueue_kernel_result = enqueue_kernel(
                    q0,
                    CLK_ENQUEUE_FLAGS_NO_WAIT,
                    ndrange_1D(MAP_VALUE_LEN, VALUE_DEVICE_LOCAL_WORK_SIZE),
                    ^{
                       map_put_one__BLOCK_NAME_def_3(
                            map_id,
                            entry_index,
                            key_input_index,
                            value_input_index,
                            keys_input,
                            values_input
                       );
                    }
                );

                return;
            }

        }

        NEXT_BLOCK

    }
    "#;

const MAP_TRY_INSERT_FUNCTION_CALL: &str = r#"
        // next block
        map_try_insert__BLOCK_NAME(
            q0,
            PIPES_REFS
            map_id,
            key_input_index,
            value_input_index,
            value_len,
            tmp_index,
            keys_input,
            values_input,
            result
        );
    "#;

const MAP_TRY_INSERT_RESET_RELEASE_PARAMS: &str = r#"
        // there are no more blocks
        result->entry_index = -1;
        result->map_value_len = 0;
        result->enqueue_kernel_result = 0;

        // Avoid freeing the block if the relocation failed
        result->previous_map_value_len = 0;
    "#;

const RELEASE_BLOCK_SWITCH_CASE: &str = r#"
                    case MAP_VALUE_LEN:
                          remove_enqueue_kernel_result = enqueue_kernel(
                                q0,
                                CLK_ENQUEUE_FLAGS_NO_WAIT,
                                ndrange_1D(MAP_VALUE_LEN, VALUE_DEVICE_LOCAL_WORK_SIZE),
                                ^{
                                   map_delete_one__BLOCK_NAME(
                                        map_id,
                                        result.previous_entry_index // entry_index
                                   );
                                }
                          );
                          break;
"#;

const MAP_PUT_KERNELS: &str = r#"
    kernel void map_put_one__BLOCK_NAME_def_3(
        const uint map_id,
        const int entry_index,
        const int key_input_index,
        const int value_input_index,
        global CL_TYPE *key_input,
        global CL_TYPE *value_input
        ) {
        int i = get_global_id(0);
        if (i < DEFAULT_MAP_KEY_LENGTH) {
            map_keys__BLOCK_NAME[map_id][entry_index][i] = key_input[i + key_input_index];
        }
        map_values__BLOCK_NAME[map_id][entry_index][i] = value_input[i + value_input_index];
    }

    kernel void map_set_one_value__BLOCK_NAME_def_2(
        const uint map_id,
        const int entry_index,
        const int value_input_index,
        global CL_TYPE *value_input
        ) {
        int i = get_global_id(0);
        map_values__BLOCK_NAME[map_id][entry_index][i] = value_input[i + value_input_index];
    }

    kernel void map_delete_one__BLOCK_NAME(
        const uint map_id,
        const int entry_index
        ) {
        int i = get_global_id(0);
        if (i < DEFAULT_MAP_KEY_LENGTH) {
            map_keys__BLOCK_NAME[map_id][entry_index][i] = CL_DEFAULT_VALUE;
        }
        map_values__BLOCK_NAME[map_id][entry_index][i] = CL_DEFAULT_VALUE;
    }
    "#;

const GLOBAL_TMP_ARRAY: &str = r#"
    global int tmp_for_map_insert[TOTAL_MAPS][TMP_LEN];

    kernel void get_tmp_for_map_insert(
        const uint map_id,
        global int *output
        ) {
        int i = get_global_id(0);
        output[i] = tmp_for_map_insert[map_id][i];
    }

    void reset_tmp_for_map_insert(uint map_id, int tmp_index) {
        RESET_TMP_ARRAY_BODY
    }    
    "#;

const RESET_TMP: &str = r#"
        // reset BLOCK_NAME
        for (int index = 0; index < MAP_CAPACITY; index++) {
            tmp_for_map_insert[map_id][index + TMP_INDEX__BLOCK_NAME_def_4 + tmp_index] = -3;
        }
        "#;

const CONST_DEF: &str = r#"
    const int TMP_INDEX__BLOCK_NAME_def_4 = BLOCK_INDEX;
    "#;

const CMQ_CONST_DEF: &str = r#"
    const int CMQ_COMPARE_KEY__BLOCK_NAME = BLOCK_INDEX;
    "#;

const PIPE_ARG: &str = r#"
        __read_only pipe int pipe_BLOCK_NAME,
    "#;

const PIPE_REF: &str = r#"
                    pipe_BLOCK_NAME,
    "#;

impl<T: ClTypeTrait> MapSrc<T> {
    pub fn generate_map_insert_program_src(&self, max_find_work_size: usize) -> String {
        // const definitions
        let mut const_def = String::new();
        let mut cmq_const_def = String::new();

        // kernel
        let mut pipes_args = String::new();
        let mut pipes_refs = String::new();

        let mut reset_tmp_array_body = String::new();

        let mut compare_key_kernels = String::new();
        let mut compare_key_enqueue_kernel = String::new();

        let mut map_put_kernels = String::new();

        let mut map_try_insert_functions_def = String::new();

        let mut map_try_insert_if_exist_functions = String::new();
        let mut map_try_insert_functions = String::new();

        let mut release_switch_cases = String::new();

        let map_blocks = self.get_configs();
        let total_blocks = map_blocks.len();

        let max_value_len = self.get_max_value_len();

        let cmq_compare_key_in_blocks = total_blocks;
        let cmq_confirm_map_insert = total_blocks + 1;
        let cmq_confirm_map_put = total_blocks + 2;
        let cmq_confirm_map_remove = total_blocks + 3;

        let total_queues = total_blocks + 4;

        // FIXME improve code iteration and simplicity

        for config in map_blocks.iter() {
            let p = common_replace(PIPE_ARG, config);
            pipes_args.push_str(&p);

            let p = common_replace(PIPE_REF, config);
            pipes_refs.push_str(&p);

            let p = common_replace(RELEASE_BLOCK_SWITCH_CASE, config);
            release_switch_cases.push_str(&p);
        }

        for (i, config) in map_blocks.iter().enumerate() {
            let tmp_index: usize = map_blocks[0..i].iter().map(|x| x.capacity).sum();

            let template =
                common_replace(CONST_DEF, config).replace("BLOCK_INDEX", &tmp_index.to_string());
            const_def.push_str(&template);

            let template =
                common_replace(CMQ_CONST_DEF, config).replace("BLOCK_INDEX", &i.to_string());
            cmq_const_def.push_str(&template);

            let template = common_replace(RESET_TMP, config);
            reset_tmp_array_body.push_str(&template);

            let template = common_replace(COMPARE_MAP_KEY_KERNEL, config);
            compare_key_kernels.push_str(&template);

            let template = common_replace(COMPARE_MAP_KEY_ENQUEUE_KERNEL, config);
            compare_key_enqueue_kernel.push_str(&template);

            let template = common_replace(MAP_PUT_KERNELS, config);
            map_put_kernels.push_str(&template);

            let next_block = if (i + 1) == total_blocks {
                "// ... ".to_string()
            } else {
                let next_config = &map_blocks[i + 1];

                MAP_TRY_INSERT_IF_EXIST_FUNCTION_CALL
                    .replace("BLOCK_NAME", &next_config.name)
                    .replace("PIPES_ARGS", &pipes_args)
                    .replace("PIPES_REFS", &pipes_refs)
            };

            let template = common_replace(MAP_TRY_INSERT_IF_EXIST_FUNCTION, config)
                .replace("MAX_VALUE_LEN", &max_value_len.to_string())
                .replace("PIPES_ARGS", &pipes_args)
                .replace("PIPES_REFS", &pipes_refs)
                .replace("NEXT_BLOCK", &next_block);
            map_try_insert_if_exist_functions.push_str(&template);

            let next_block = if (i + 1) == total_blocks {
                MAP_TRY_INSERT_RESET_RELEASE_PARAMS.to_string()
            } else {
                let next_config = &map_blocks[i + 1];

                MAP_TRY_INSERT_FUNCTION_CALL
                    .replace("BLOCK_NAME", &next_config.name)
                    .replace("PIPES_ARGS", &pipes_args)
                    .replace("PIPES_REFS", &pipes_refs)
            };

            let template = common_replace(MAP_TRY_INSERT_FUNCTION, config)
                .replace("MAX_VALUE_LEN", &max_value_len.to_string())
                .replace("PIPES_ARGS", &pipes_args)
                .replace("PIPES_REFS", &pipes_refs)
                .replace("NEXT_BLOCK", &next_block);
            map_try_insert_functions.push_str(&template);

            // functions definitions

            let template = common_replace(MAP_TRY_INSERT_FUNCTIONS_DEF, config)
                .replace("PIPES_ARGS", &pipes_args)
                .replace("PIPES_REFS", &pipes_refs);
            map_try_insert_functions_def.push_str(&template);
        }

        let compare_map_key_main_kernel = COMPARE_MAP_KEY_MAIN_KERNEL
            .replace("CL_TYPE", T::cl_enum().to_cl_type_name())
            .replace("KERNEL_BODY", &compare_key_enqueue_kernel);

        let first_config = map_blocks.first().unwrap();

        let confirm_map_insert_kernel = common_replace(CONFIRM_MAP_INSERT_KERNEL, first_config)
            .replace("PIPES_ARGS", &pipes_args)
            .replace("PIPES_REFS", &pipes_refs)
            .replace("RELEASE_BLOCK_SWITCH", &release_switch_cases);

        let total_indices = self.get_maximum_assignable_keys();

        let map_insert_kernel = common_replace(MAP_INSERT_KERNEL, first_config)
            .replace("MAX_VALUE_LEN", &max_value_len.to_string())
            .replace("TOTAL_INDICES", &total_indices.to_string())
            .replace("TOTAL_QUEUES", &total_queues.to_string())
            .replace("RESET_TMP_ARRAY_BODY", &reset_tmp_array_body)
            .replace("PIPES_ARGS", &pipes_args)
            .replace("PIPES_REFS", &pipes_refs);

        let tmp_len = max_find_work_size * total_indices;

        let global_tmp_array = GLOBAL_TMP_ARRAY
            .replace("TMP_LEN", &tmp_len.to_string())
            .replace("TOTAL_MAPS", &self.get_total_maps().to_string())
            .replace("RESET_TMP_ARRAY_BODY", &reset_tmp_array_body);

        format!(
            "
    /// - MAP_INSERT START ///

    /// constants

    {const_def}

    {cmq_const_def}
    const int CMQ_COMPARE_KEY_IN_BLOCKS = {cmq_compare_key_in_blocks};
    const int CMQ_CONFIRM_MAP_INSERT = {cmq_confirm_map_insert};
    const int CMQ_CONFIRM_MAP_PUT = {cmq_confirm_map_put};
    const int CMQ_CONFIRM_MAP_REMOVE = {cmq_confirm_map_remove};

    {STRUCT_DEF}

    {map_try_insert_functions_def}

    /// globals
    {global_tmp_array}

    /// kernels
    {map_put_kernels}

    {compare_key_kernels}
    {compare_map_key_main_kernel}

    {map_try_insert_if_exist_functions}
    {map_try_insert_functions}

    {confirm_map_insert_kernel}
    {map_insert_kernel}

    /// - MAP_INSERT END ///
        "
        )
    }

    pub fn add_map_insert_program_src(&mut self, max_find_work_size: usize) -> &mut Self {
        let src = self.generate_map_insert_program_src(max_find_work_size);
        self.optional_sources.push(src);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a() {
        let mut map_src: MapSrc<i32> = MapSrc::new(2);
        map_src.add(256, 16);

        let program_source = map_src.generate_map_insert_program_src(16);
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_b() {
        let mut map_src: MapSrc<i32> = MapSrc::new(2);
        map_src.add(256, 8);
        map_src.add(512, 16);

        let program_source = map_src.generate_map_insert_program_src(8);
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_c() {
        let mut map_src: MapSrc<i32> = MapSrc::new(2);
        map_src.add(256, 8);
        map_src.add(512, 32);
        map_src.add(1024, 16);

        let program_source = map_src.generate_map_insert_program_src(2);
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }
}
