use crate::config::ClTypeTrait;
use crate::map::config::MapSrc;
use crate::map::kernel::common_replace;

const MAP_APPEND_KERNEL: &str = r#"
    kernel void map_append_one_value__BLOCK_NAME(
        const uint map_id,
        const int entry_index,
        const int value_input_index,
        const int offset_index,
        global CL_TYPE* value_input
        ) {
        int i = get_global_id(0);
        map_values__BLOCK_NAME[map_id][entry_index][i + offset_index] = value_input[i + value_input_index];
    }

    kernel void map_append_for_block__BLOCK_NAME(
        queue_t q0,
        const uint map_id,
        global CL_TYPE* values_input,
        global int* values_lens_input,
        global int* indices_input,
        global int* result_output,
        global int* enqueue_kernel_output
        ) {
        int i = get_global_id(0);

        int result_index = i * 2;
        int value_input_index = i * MAP_VALUE_LEN;

        int value_len = values_lens_input[i];
        int entry_index = indices_input[i];

        int result_append = MAP_VALUE_FULL;
        int enqueue_kernel_result = 0;

        int offset_index = get_map_value_size__BLOCK_NAME(map_id, entry_index);

        if (offset_index != MAP_VALUE_LEN) {
            if (MAP_VALUE_LEN >= (offset_index + value_len)) {
                result_append = 0;

                int local_work_size = check_device_local_work_size(value_len);

                enqueue_kernel_result = enqueue_kernel(
                    q0,
                    CLK_ENQUEUE_FLAGS_NO_WAIT,
                    ndrange_1D(value_len, local_work_size),
                    ^{
                       map_append_one_value__BLOCK_NAME(
                           map_id,
                           entry_index,
                           value_input_index,
                           offset_index,
                           values_input
                       );
                    }
                );
            } else {
                result_append = CANNOT_APPEND_VALUE;
            }
        }

        result_output[RESULT_APPEND + result_index] = result_append;
        result_output[RESULT_LAST_INDEX + result_index] = offset_index;

        enqueue_kernel_output[i] = enqueue_kernel_result;
    }
    "#;

impl<T: ClTypeTrait> MapSrc<T> {
    pub fn generate_map_append_for_block_program_src(&self) -> String {
        let mut map_append_kernels = String::new();

        let map_blocks = self.get_configs();

        for config in map_blocks {
            let template = common_replace(MAP_APPEND_KERNEL, config);
            map_append_kernels.push_str(&template);
        }

        format!(
            "
    /// - MAP_APPEND_FOR_BLOCK START ///

    /// constants

    const int CANNOT_APPEND_VALUE = -4;
    const int MAP_VALUE_FULL = -5;

    const int RESULT_APPEND = 0;
    const int RESULT_LAST_INDEX = 1;

    /// globals
    // ...

    /// kernels
    {map_append_kernels}

    /// - MAP_APPEND_FOR_BLOCK END ///
        "
        )
    }

    pub fn add_map_append_for_block_program_src(&mut self) -> &mut Self {
        let src = self.generate_map_append_for_block_program_src();
        self.optional_sources.push(src);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{BYTE_256, KB};

    #[test]
    fn test_a() {
        let mut map_src: MapSrc<i32> = MapSrc::new(2);
        map_src.add(KB, 16);

        let program_source = map_src.generate_map_append_for_block_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_b() {
        let mut map_src: MapSrc<i16> = MapSrc::new(2);
        map_src.add(BYTE_256, 32);
        map_src.add(KB, 16);

        let program_source = map_src.generate_map_append_for_block_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }
}
