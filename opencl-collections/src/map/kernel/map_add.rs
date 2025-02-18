use crate::config::ClTypeTrait;
use crate::map::config::{MapSrc, DEFAULT_MAP_KEY_LENGTH};
use crate::map::kernel::common_replace;

// this method only works if map_key_len is equal to or less than map_value_len

const STRUCT_DEF: &str = r#"
    struct MapAddResult {
        int entry_index;
        int map_value_len;
        int enqueue_kernel_result;
    };
    "#;

const MAP_ADD_KERNEL: &str = r#"
    kernel void map_add(
        queue_t q0,
        PIPES_ARGS
        const uint map_id,
        global CL_TYPE *keys_input,
        global CL_TYPE *values_input,
        global int* values_len_input,
        global int* indices_output,
        global int* block_output,
        global int* enqueue_kernel_output
        ) {
        int i = get_global_id(0);
        int key_input_index = i * DEFAULT_MAP_KEY_LENGTH;
        int value_input_index = i * MAX_VALUE_LEN;
        int value_len = values_len_input[i];

        struct MapAddResult result;
        
        try_map_add__BLOCK_NAME(
            q0,
            PIPES_REFS
            map_id,
            key_input_index,
            value_input_index,
            value_len,
            keys_input,
            values_input,
            &result
        );

        indices_output[i] = result.entry_index;
        block_output[i] = result.map_value_len;
        enqueue_kernel_output[i] = result.enqueue_kernel_result;
    }
    "#;

const TRY_MAP_ADD_FUNCTION_DEF: &str = r#"
    void try_map_add__BLOCK_NAME(
        queue_t q0,
        PIPES_ARGS
        const uint map_id,
        const int key_input_index,
        const int value_input_index,
        const int value_len,
        global CL_TYPE *keys_input,
        global CL_TYPE *values_input,
        struct MapAddResult *result
    );
    "#;

const TRY_MAP_ADD_FUNCTION: &str = r#"
    void try_map_add__BLOCK_NAME(
        queue_t q0,
        PIPES_ARGS
        const uint map_id,
        const int key_input_index,
        const int value_input_index,
        const int value_len,
        global CL_TYPE *keys_input,
        global CL_TYPE *values_input,
        struct MapAddResult *result
        ) {

        int entry_index = -1;

        if (MAP_VALUE_LEN >= value_len) {

            read_pipe(pipe_BLOCK_NAME, &entry_index);

            if (entry_index >= 0) {

                result->entry_index = entry_index;
                result->map_value_len = MAP_VALUE_LEN;

                result->enqueue_kernel_result = enqueue_kernel(
                    q0,
                    CLK_ENQUEUE_FLAGS_NO_WAIT,
                    ndrange_1D(MAP_VALUE_LEN, VALUE_DEVICE_LOCAL_WORK_SIZE),
                    ^{
                       map_put_one__BLOCK_NAME_def_2(
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

const TRY_MAP_ADD_NEXT_BLOCK: &str = r#"
        // next block
        try_map_add__BLOCK_NAME(
            q0,
            PIPES_REFS
            map_id,
            key_input_index,
            value_input_index,
            value_len,
            keys_input,
            values_input,
            result
        );
    "#;

const TRY_MAP_ADD_NEXT_BLOCK_DEFAULT: &str = r#"
        // there are no more blocks
        result->entry_index = -1;
        result->map_value_len = 0;
        result->enqueue_kernel_result = 0;
    "#;

const MAP_PUT_KERNEL: &str = r#"
    kernel void map_put_one__BLOCK_NAME_def_2(
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
    "#;

const PIPE_ARG: &str = r#"
        __read_only pipe int pipe_BLOCK_NAME,
    "#;

const PIPE_REF: &str = r#"
                    pipe_BLOCK_NAME,
    "#;

impl<T: ClTypeTrait> MapSrc<T> {
    pub fn generate_map_add_program_src(&self) -> String {
        let mut map_put_kernels = String::new();

        let mut try_map_add_functions = String::new();
        let mut try_map_add_functions_def = String::new();

        let mut pipes_args = String::new();
        let mut pipes_refs = String::new();

        let map_blocks = self.get_configs();
        let total_blocks = map_blocks.len();

        let max_value_len = self.get_max_value_len();

        // FIXME improve code iteration and simplicity

        for config in map_blocks.iter() {
            let p = common_replace(PIPE_ARG, config);
            pipes_args.push_str(&p);

            let p = common_replace(PIPE_REF, config);
            pipes_refs.push_str(&p);
        }

        for (i, config) in map_blocks.iter().enumerate() {
            let template = common_replace(MAP_PUT_KERNEL, config);
            map_put_kernels.push_str(&template);

            let next_block = if (i + 1) == total_blocks {
                TRY_MAP_ADD_NEXT_BLOCK_DEFAULT.to_string()
            } else {
                let next_config = &map_blocks[i + 1];
                TRY_MAP_ADD_NEXT_BLOCK
                    .replace("BLOCK_NAME", &next_config.name)
                    .replace("PIPES_ARGS", &pipes_args)
                    .replace("PIPES_REFS", &pipes_refs)
            };

            let template = common_replace(TRY_MAP_ADD_FUNCTION, config)
                .replace("MAX_VALUE_LEN", &max_value_len.to_string())
                .replace("PIPES_ARGS", &pipes_args)
                .replace("PIPES_REFS", &pipes_refs)
                .replace("NEXT_BLOCK", &next_block);
            try_map_add_functions.push_str(&template);

            let template = common_replace(TRY_MAP_ADD_FUNCTION_DEF, config)
                .replace("PIPES_ARGS", &pipes_args)
                .replace("PIPES_REFS", &pipes_refs);
            try_map_add_functions_def.push_str(&template);
        }

        let first_config = map_blocks.first().unwrap();

        let map_add_kernel = MAP_ADD_KERNEL
            .replace("PIPES_ARGS", &pipes_args)
            .replace("PIPES_REFS", &pipes_refs)
            .replace(
                "DEFAULT_MAP_KEY_LENGTH",
                &DEFAULT_MAP_KEY_LENGTH.to_string(),
            )
            .replace("MAX_VALUE_LEN", &max_value_len.to_string())
            .replace("TOTAL_BLOCKS", &total_blocks.to_string())
            .replace("CL_TYPE", T::cl_enum().to_cl_type_name())
            .replace("BLOCK_NAME", &first_config.name);

        format!(
            "
    /// - MAP_ADD START ///

    /// constants
    {STRUCT_DEF}

    {try_map_add_functions_def}

    /// globals
    // ...

    /// kernels
    {map_put_kernels}

    {try_map_add_functions}
    {map_add_kernel}

    /// - MAP_ADD END ///
        "
        )
    }

    pub fn add_map_add_program_src(&mut self) -> &mut Self {
        let src = self.generate_map_add_program_src();
        self.optional_sources.push(src);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a() {
        let mut map_src: MapSrc<i32> = MapSrc::default();
        map_src.add(256, 16);

        let program_source = map_src.generate_map_add_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_b() {
        let mut map_src: MapSrc<i32> = MapSrc::default();
        map_src.add(256, 8);
        map_src.add(512, 16);

        let program_source = map_src.generate_map_add_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_c() {
        let mut map_src: MapSrc<i32> = MapSrc::default();
        map_src.add(256, 8);
        map_src.add(512, 32);
        map_src.add(1024, 16);

        let program_source = map_src.generate_map_add_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn error() {
        let mut map_src: MapSrc<i32> = MapSrc::default();
        map_src.add(32, 16);

        let program_source = map_src.generate_map_add_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }
}
