use crate::config::ClTypeTrait;
use crate::map::config::{check_local_work_size, MapSrc};
use crate::map::kernel::common_replace;

const MAP_PUT_KERNELS: &str = r#"
    kernel void map_put_one__BLOCK_NAME(
        const uint map_id,
        const int entry_index,
        const int key_input_index,
        const int value_input_index,
        global CL_TYPE* key_input,
        global CL_TYPE* value_input
        ) {
        int i = get_global_id(0);
        if (i < MAP_KEY_LEN) {
            map_keys__BLOCK_NAME[map_id][entry_index][i] = key_input[i + key_input_index];
        }
        map_values__BLOCK_NAME[map_id][entry_index][i] = value_input[i + value_input_index];
    }

    kernel void map_set_one_key__BLOCK_NAME(
        const uint map_id,
        const int entry_index,
        const int key_input_index,
        global CL_TYPE* key_input
        ) {
        int i = get_global_id(0);
        map_keys__BLOCK_NAME[map_id][entry_index][i] = key_input[i + key_input_index];
    }

    kernel void map_set_one_value__BLOCK_NAME(
        const uint map_id,
        const int entry_index,
        const int value_input_index,
        global CL_TYPE* value_input
        ) {
        int i = get_global_id(0);
        map_values__BLOCK_NAME[map_id][entry_index][i] = value_input[i + value_input_index];
    }

    kernel void map_put__BLOCK_NAME(
        const uint map_id,
        global CL_TYPE* keys_input,
        global CL_TYPE* values_input
        ) {
        int i = get_global_id(0);
        int key_input_index = i * MAP_KEY_LEN;
        int value_input_index = i * MAP_VALUE_LEN;

        for (int index = 0; index < MAP_KEY_LEN; index++) {
            map_keys__BLOCK_NAME[map_id][i][index] = keys_input[index + key_input_index];
        }

        for (int index = 0; index < MAP_VALUE_LEN; index++) {
            map_values__BLOCK_NAME[map_id][i][index] = values_input[index + value_input_index];
        }
    }

    kernel void map_put_with_cmq__BLOCK_NAME(
        queue_t q0,
        const uint map_id,
        global CL_TYPE* keys_input,
        global CL_TYPE* values_input,
        global int* enqueue_kernel_output
        ) {
        int i = get_global_id(0);
        int key_input_index = i * MAP_KEY_LEN;
        int value_input_index = i * MAP_VALUE_LEN;
        int enqueue_kernel_output_index = i * 2;

        enqueue_kernel_output[CMQ_PUT_MAP_KEY_INDEX + enqueue_kernel_output_index] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(MAP_KEY_LEN, KEY_DEVICE_LOCAL_WORK_SIZE),
            ^{
               map_set_one_key__BLOCK_NAME(
                 map_id,
                 i, // entry_index
                 key_input_index,
                 keys_input
               );
            }
        );

        enqueue_kernel_output[CMQ_PUT_MAP_VALUE_INDEX + enqueue_kernel_output_index] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(MAP_VALUE_LEN, VALUE_DEVICE_LOCAL_WORK_SIZE),
            ^{
               map_set_one_value__BLOCK_NAME(
                 map_id,
                 i, // entry_index
                 value_input_index,
                 values_input
               );
            }
        );
    }

    kernel void map_put_with_index__BLOCK_NAME(
        const uint map_id,
        global CL_TYPE* keys_input,
        global CL_TYPE* values_input,
        global int* indices_input
        ) {
        int i = get_global_id(0);
        int key_input_index = i * MAP_KEY_LEN;
        int value_input_index = i * MAP_VALUE_LEN;

        int entry_index = indices_input[i];

        for (int index = 0; index < MAP_KEY_LEN; index++) {
            map_keys__BLOCK_NAME[map_id][entry_index][index] = keys_input[index + key_input_index];
        }

        for (int index = 0; index < MAP_VALUE_LEN; index++) {
            map_values__BLOCK_NAME[map_id][entry_index][index] = values_input[index + value_input_index];
        }
    }

    kernel void map_put_with_pipe_and_cmq__BLOCK_NAME(
        queue_t q0,
        __read_only pipe int pipe0,
        const uint map_id,
        global CL_TYPE* keys_input,
        global CL_TYPE* values_input,
        global int* indices_output,
        global int* enqueue_kernel_output
        ) {
        int i = get_global_id(0);
        int key_input_index = i * MAP_KEY_LEN;
        int value_input_index = i * MAP_VALUE_LEN;
        int enqueue_kernel_output_index = i * 2;

        int pi = -1;
        read_pipe(pipe0, &pi);

        if (pi != -1) {

            int r_q_0 = enqueue_kernel(
                q0,
                CLK_ENQUEUE_FLAGS_NO_WAIT,
                ndrange_1D(MAP_KEY_LEN, KEY_DEVICE_LOCAL_WORK_SIZE),
                ^{
                   map_set_one_key__BLOCK_NAME(
                     map_id,
                     pi, // entry_index
                     key_input_index,
                     keys_input
                   );
                }
            );
            enqueue_kernel_output[CMQ_PUT_MAP_KEY_INDEX + enqueue_kernel_output_index] = r_q_0;

            int r_q_1 = enqueue_kernel(
                q0,
                CLK_ENQUEUE_FLAGS_NO_WAIT,
                ndrange_1D(MAP_VALUE_LEN, VALUE_DEVICE_LOCAL_WORK_SIZE),
                ^{
                   map_set_one_value__BLOCK_NAME(
                     map_id,
                     pi, // entry_index
                     value_input_index,
                     values_input
                   );
                }
            );
            enqueue_kernel_output[CMQ_PUT_MAP_VALUE_INDEX + enqueue_kernel_output_index] = r_q_1;
            
        } else {
            enqueue_kernel_output[CMQ_PUT_MAP_KEY_INDEX + enqueue_kernel_output_index] = 0;
            enqueue_kernel_output[CMQ_PUT_MAP_VALUE_INDEX + enqueue_kernel_output_index] = 0;
        }

        indices_output[i] = pi;
    }
    "#;

impl<T: ClTypeTrait> MapSrc<T> {
    pub fn generate_map_put_program_src(&self) -> String {
        let mut map_put_kernels = String::new();

        for config in self.get_configs().iter() {
            let key_device_local_work_size = check_local_work_size(config.key_len).to_string();
            let value_device_local_work_size = check_local_work_size(config.value_len).to_string();

            let template = common_replace(MAP_PUT_KERNELS, config)
                .replace("KEY_DEVICE_LOCAL_WORK_SIZE", &key_device_local_work_size)
                .replace(
                    "VALUE_DEVICE_LOCAL_WORK_SIZE",
                    &value_device_local_work_size,
                );
            map_put_kernels.push_str(&template);
        }

        format!(
            "
    /// - MAP_PUT START ///
    
    /// constants

    const int CMQ_PUT_MAP_KEY_INDEX = 0;
    const int CMQ_PUT_MAP_VALUE_INDEX = 1;

    /// globals
    // ...

    
    /// kernels
    {map_put_kernels}

    /// - MAP_PUT END ///
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
        let mut map_src: MapSrc<i32> = MapSrc::default();
        map_src.add(KB, 16);

        let program_source = map_src.generate_map_put_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_b() {
        let mut map_src: MapSrc<i32> = MapSrc::default();
        map_src.add(256, 8);
        map_src.add(512, 16);

        let program_source = map_src.generate_map_put_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_c() {
        let mut map_src: MapSrc<i32> = MapSrc::default();
        map_src.add(256, 8);
        map_src.add(512, 32);
        map_src.add(1024, 16);

        let program_source = map_src.generate_map_put_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }
}
