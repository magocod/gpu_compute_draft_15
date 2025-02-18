use crate::config::ClTypeTrait;
use crate::map::config::MapSrc;
use crate::map::kernel::common_replace;

// FIXME search for a better kernel names

const MAP_GET_EMPTY_KEY_FOR_BLOCK_KERNEL: &str = r#"
    kernel void map_get_empty_keys_for_block__BLOCK_NAME(
        __write_only pipe int pipe0,
        const uint map_id
        ) {

        int i = get_global_id(0);

        bool is_empty = is_map_key_empty__BLOCK_NAME(map_id, i);

        if (is_empty == true) {
            write_pipe(pipe0, &i);
        }
    }
    "#;

const MAP_GET_EMPTY_KEY_KERNEL: &str = r#"
    kernel void map_get_empty_keys(
        queue_t q0,
        PIPES_ARGS
        const uint map_id,
        global int* enqueue_kernel_output
        ) {

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
               map_get_empty_keys_for_block__BLOCK_NAME(
                    pipe_BLOCK_NAME,
                    map_id
               );
            }
        );
     "#;

const PIPE_ARG: &str = r#"
        __write_only pipe int pipe_BLOCK_NAME,
    "#;

impl<T: ClTypeTrait> MapSrc<T> {
    pub fn generate_map_get_empty_key_program_src(&self) -> String {
        let mut map_get_empty_keys_kernels = String::new();

        let mut enqueue_kernels = String::new();
        let mut pipes_args = String::new();

        for (i, config) in self.get_configs().iter().enumerate() {
            let template = common_replace(MAP_GET_EMPTY_KEY_FOR_BLOCK_KERNEL, config);
            map_get_empty_keys_kernels.push_str(&template);

            let p = common_replace(PIPE_ARG, config);
            pipes_args.push_str(&p);

            let template =
                common_replace(ENQUEUE_KERNEL, config).replace("BLOCK_INDEX", &i.to_string());
            enqueue_kernels.push_str(&template);
        }

        let map_get_empty_keys_kernel = MAP_GET_EMPTY_KEY_KERNEL
            .replace("PIPES_ARGS", &pipes_args)
            .replace("KERNEL_BODY", &enqueue_kernels);

        format!(
            "
    /// - MAP_GET_EMPTY_KEY START ///

    /// constants
    // ...

    /// globals
    // ...


    /// kernels
    {map_get_empty_keys_kernels}
    {map_get_empty_keys_kernel}

    /// - MAP_GET_EMPTY_KEY END ///
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
        let mut map_src: MapSrc<i16> = MapSrc::new(2);
        map_src.add(KB, 16);

        let program_source = map_src.generate_map_get_empty_key_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_b() {
        let mut map_src: MapSrc<i32> = MapSrc::new(2);
        map_src.add(256, 8);
        map_src.add(512, 16);

        let program_source = map_src.generate_map_get_empty_key_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_c() {
        let mut map_src: MapSrc<i32> = MapSrc::new(2);
        map_src.add(256, 8);
        map_src.add(512, 32);
        map_src.add(1024, 16);

        let program_source = map_src.generate_map_get_empty_key_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }
}
