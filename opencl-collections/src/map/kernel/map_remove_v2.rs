use crate::config::ClTypeTrait;
use crate::map::config::MapSrc;
use crate::map::kernel::common_replace;

const MAP_REMOVE_KERNEL: &str = r#"
    kernel void map_remove(
        queue_t q0,
        const uint map_id,
        global CL_TYPE *keys_input,
        global int *indices_output,
        global int *block_output,
        global int *enqueue_kernel_output
    ) {
        int i = get_global_id(0);

        int key_input_index = i * DEFAULT_MAP_KEY_LENGTH;
        int tmp_index = i * TOTAL_INDICES;
        int enqueue_kernel_output_index = i * TOTAL_QUEUES;

        reset_tmp_for_map_remove(map_id, tmp_index);

        clk_event_t evt0;

        enqueue_kernel_output[CMQ_COMPARE_KEY_IN_BLOCKS_DEF_3 + enqueue_kernel_output_index] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(1, 1),
            0,
            NULL,
            &evt0,
            ^{
               compare_map_key_in_blocks_def_3(
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

        enqueue_kernel_output[CMQ_CONFIRM_MAP_REMOVE_def_3 + enqueue_kernel_output_index] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(1, 1),
            1,
            &evt0,
            NULL,
            ^{
               confirm_map_remove(
                    q0,
                    map_id,
                    i, // parent_global_index,
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

const COMPARE_MAP_KEY_MAIN_KERNEL: &str = r#"
    kernel void compare_map_key_in_blocks_def_3(
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
        enqueue_kernel_output[CMQ_COMPARE_KEY__BLOCK_NAME_DEF_3 + enqueue_kernel_output_index] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(MAP_CAPACITY, CAPACITY_DEVICE_LOCAL_WORK_SIZE),
            ^{
               compare_map_key_in_block__BLOCK_NAME_def_5(
                  map_id,
                  key_input_index,
                  tmp_index,
                  keys_input
               );
            }
        );
"#;

const COMPARE_MAP_KEY_KERNEL: &str = r#"
    kernel void compare_map_key_in_block__BLOCK_NAME_def_5(
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
            int item_tmp_index = i + TMP_INDEX__BLOCK_NAME_def_6 + tmp_index;

            tmp_for_map_remove[map_id][item_tmp_index] = i;
        }
    }
    "#;

const CONFIRM_MAP_REMOVE_KERNEL: &str = r#"

    kernel void confirm_map_remove(
        queue_t q0,
        const uint map_id,
        const int parent_global_index,
        const int key_input_index,
        const int tmp_index,
        const int enqueue_kernel_output_index,
        global CL_TYPE *keys_input,
        global int *indices_output,
        global int *block_output,
        global int *enqueue_kernel_output
        ) {

        int entry_index = -3;
        int block = LAST_VALUE_LEN;

        int remove_enqueue_kernel_result = 0;

        for (int index = 0; index < TOTAL_INDICES; index++) {
            int t_i = tmp_for_map_remove[map_id][index + tmp_index];

            if (t_i >= 0) {

                entry_index = t_i;

                IF_CASES

            }
        }

        indices_output[parent_global_index] = entry_index;
        block_output[parent_global_index] = block;

        enqueue_kernel_output[CMQ_MAP_REMOVE + enqueue_kernel_output_index] = remove_enqueue_kernel_result;

    }
    "#;

const IF_FIRST_CASE: &str = r#"

            // error enqueue
            // remove_enqueue_kernel_result = enqueue_kernel(
            //     q0,
            //     CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
            //     ndrange_1D(MAP_VALUE_LEN, VALUE_DEVICE_LOCAL_WORK_SIZE),
            //     ^{
            //        map_delete_one__BLOCK_NAME_def_3(
            //             map_id,
            //             entry_index
            //        );
            //     }
            // );

            // BLOCK_NAME
            if (index >= TMP_INDEX__BLOCK_NAME_def_6) {

                // remove entry
                remove_enqueue_kernel_result = enqueue_kernel(
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

                block = MAP_VALUE_LEN;
                break;
            }

     "#;

const MAP_PUT_KERNELS: &str = r#"
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
    "#;

const GLOBAL_TMP_ARRAY: &str = r#"
    global int tmp_for_map_remove[TOTAL_MAPS][TMP_LEN];

    kernel void get_tmp_for_map_remove(
        const uint map_id,
        global int *output
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

const CMQ_CONST_DEF: &str = r#"
    const int CMQ_COMPARE_KEY__BLOCK_NAME_DEF_3 = BLOCK_INDEX;
    "#;

const PIPE_ARG: &str = r#"
        __read_only pipe int pipe_BLOCK_NAME,
    "#;

const PIPE_REF: &str = r#"
                    pipe_BLOCK_NAME,
    "#;

impl<T: ClTypeTrait> MapSrc<T> {
    pub fn generate_map_remove_v2_program_src(&self, max_find_work_size: usize) -> String {
        // const definitions
        let mut const_def = String::new();
        let mut cmq_const_def = String::new();

        // kernel
        let mut pipes_args = String::new();
        let mut pipes_refs = String::new();

        let mut if_cases = vec![];

        let mut reset_tmp_array_body = String::new();

        let mut compare_key_kernels = String::new();
        let mut compare_key_enqueue_kernel = String::new();

        let mut map_put_kernels = String::new();

        let map_blocks = self.get_configs();
        let total_blocks = map_blocks.len();

        let max_value_len = self.get_max_value_len();

        let cmq_compare_key_in_blocks = total_blocks;
        let cmq_confirm_map_remove = total_blocks + 1;
        let cmq_map_remove = total_blocks + 2;

        let total_queues = total_blocks + 3;

        // FIXME improve code iteration and simplicity

        for config in map_blocks.iter() {
            let p = common_replace(PIPE_ARG, config);
            pipes_args.push_str(&p);

            let p = common_replace(PIPE_REF, config);
            pipes_refs.push_str(&p);

            let p = common_replace(IF_FIRST_CASE, config);
            if_cases.push(p);
        }

        for (i, config) in map_blocks.iter().enumerate() {
            let tmp_index: usize = map_blocks[0..i].iter().map(|x| x.capacity).sum();

            let template =
                common_replace(CONST_DEF, config).replace("BLOCK_INDEX", &tmp_index.to_string());
            const_def.push_str(&template);

            let template =
                common_replace(CMQ_CONST_DEF, config).replace("BLOCK_INDEX", &i.to_string());
            cmq_const_def.push_str(&template);

            // kernel body
            let template = common_replace(RESET_TMP, config);
            reset_tmp_array_body.push_str(&template);

            let template = common_replace(COMPARE_MAP_KEY_KERNEL, config);
            compare_key_kernels.push_str(&template);

            let template = common_replace(COMPARE_MAP_KEY_ENQUEUE_KERNEL, config);
            compare_key_enqueue_kernel.push_str(&template);

            let template = common_replace(MAP_PUT_KERNELS, config);
            map_put_kernels.push_str(&template);
        }

        let compare_map_key_in_blocks_kernel = COMPARE_MAP_KEY_MAIN_KERNEL
            .replace("CL_TYPE", T::cl_enum().to_cl_type_name())
            .replace("KERNEL_BODY", &compare_key_enqueue_kernel);

        let total_indices = self.get_maximum_assignable_keys();

        let first_config = map_blocks.first().unwrap();
        let last_config = map_blocks.last().unwrap();

        if_cases.reverse();

        let confirm_map_remove_kernel = CONFIRM_MAP_REMOVE_KERNEL
            .replace("CL_TYPE", T::cl_enum().to_cl_type_name())
            .replace("PIPES_ARGS", &pipes_args)
            .replace("PIPES_REFS", &pipes_refs)
            .replace("TOTAL_INDICES", &total_indices.to_string())
            .replace("IF_CASES", &if_cases.join(" "))
            .replace("BLOCK_NAME", &first_config.name)
            .replace("LAST_VALUE_LEN", &last_config.value_len.to_string());

        let map_remove = common_replace(MAP_REMOVE_KERNEL, last_config)
            .replace("MAX_VALUE_LEN", &max_value_len.to_string())
            .replace("TOTAL_INDICES", &total_indices.to_string())
            .replace("TOTAL_QUEUES", &total_queues.to_string())
            .replace("PIPES_ARGS", &pipes_args)
            .replace("PIPES_REFS", &pipes_refs);

        let tmp_len = max_find_work_size * total_indices;

        let global_tmp_array = GLOBAL_TMP_ARRAY
            .replace("TMP_LEN", &tmp_len.to_string())
            .replace("TOTAL_MAPS", &self.get_total_maps().to_string())
            .replace("RESET_TMP_ARRAY_BODY", &reset_tmp_array_body);

        format!(
            "
    /// - MAP_REMOVE START ///

    /// constants

    {const_def}

    {cmq_const_def}
    const int CMQ_COMPARE_KEY_IN_BLOCKS_DEF_3 = {cmq_compare_key_in_blocks};
    const int CMQ_CONFIRM_MAP_REMOVE_def_3 = {cmq_confirm_map_remove};
    const int CMQ_MAP_REMOVE = {cmq_map_remove};

    /// globals
    {global_tmp_array}

    /// kernels
    {map_put_kernels}
    {compare_key_kernels}

    {compare_map_key_in_blocks_kernel}

    {confirm_map_remove_kernel}
    {map_remove}

    /// - MAP_REMOVE END ///
        "
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a() {
        let mut map_src: MapSrc<i32> = MapSrc::new(2);
        map_src.add(256, 16);

        let program_source = map_src.generate_map_remove_v2_program_src(16);
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_b() {
        let mut map_src: MapSrc<i32> = MapSrc::new(2);
        map_src.add(256, 8);
        map_src.add(512, 16);

        let program_source = map_src.generate_map_remove_v2_program_src(8);
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_c() {
        let mut map_src: MapSrc<i32> = MapSrc::new(2);
        map_src.add(256, 8);
        map_src.add(512, 32);
        map_src.add(1024, 16);

        let program_source = map_src.generate_map_remove_v2_program_src(2);
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }
}
