use crate::config::ClTypeTrait;
use crate::map::config::MapSrc;
use crate::map::kernel::common_replace;

const MAP_READ_KERNELS: &str = r#"
    kernel void map_read_one__BLOCK_NAME(
        const uint map_id,
        const int entry_index,
        const int key_output_index,
        const int value_output_index,
        global CL_TYPE* key_output,
        global CL_TYPE* value_output
        ) {
        int i = get_global_id(0);

        if (i < MAP_KEY_LEN) {
            key_output[i + key_output_index] = map_keys__BLOCK_NAME[map_id][entry_index][i];
        }

        value_output[i + value_output_index] = map_values__BLOCK_NAME[map_id][entry_index][i];
    }

    kernel void map_read_one_key__BLOCK_NAME(
        const uint map_id,
        const int entry_index,
        const int key_output_index,
        global CL_TYPE* key_output
        ) {
        int i = get_global_id(0);
        key_output[i + key_output_index] = map_keys__BLOCK_NAME[map_id][entry_index][i];
    }

    kernel void map_read_one_value__BLOCK_NAME(
        const uint map_id,
        const int entry_index,
        const int value_output_index,
        global CL_TYPE* value_output
        ) {
        int i = get_global_id(0);
        value_output[i + value_output_index] = map_values__BLOCK_NAME[map_id][entry_index][i];
    }

    kernel void map_read_keys__BLOCK_NAME(
        const uint map_id,
        global CL_TYPE* keys_output
        ) {

        int i = get_global_id(0);
        int key_output_index = i * MAP_KEY_LEN;

        for (int index = 0; index < MAP_KEY_LEN; index++) {
            keys_output[index + key_output_index] = map_keys__BLOCK_NAME[map_id][i][index];
        }
    }

    kernel void map_read__BLOCK_NAME(
        const uint map_id,
        global CL_TYPE* keys_output,
        global CL_TYPE* values_output
        ) {

        int i = get_global_id(0);
        int key_output_index = i * MAP_KEY_LEN;
        int value_output_index = i * MAP_VALUE_LEN;

        for (int index = 0; index < MAP_KEY_LEN; index++) {
            keys_output[index + key_output_index] = map_keys__BLOCK_NAME[map_id][i][index];
        }

        for (int index = 0; index < MAP_VALUE_LEN; index++) {
            values_output[index + value_output_index] = map_values__BLOCK_NAME[map_id][i][index];
        }
    }

    kernel void map_read_with_cmq__BLOCK_NAME(
        queue_t q0,
        const uint map_id,
        global CL_TYPE* keys_output,
        global CL_TYPE* values_output,
        global int* enqueue_kernel_output
        ) {

        int i = get_global_id(0);
        int key_output_index = i * MAP_KEY_LEN;
        int value_output_index = i * MAP_VALUE_LEN;
        int enqueue_kernel_output_index = i * 2;

        enqueue_kernel_output[CMQ_GET_MAP_KEY_INDEX + enqueue_kernel_output_index] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(MAP_KEY_LEN, KEY_DEVICE_LOCAL_WORK_SIZE),
            ^{
               map_read_one_key__BLOCK_NAME(
                    map_id,
                    i, // entry_index
                    key_output_index,
                    keys_output
               );
            }
        );

        enqueue_kernel_output[CMQ_GET_MAP_VALUE_INDEX + enqueue_kernel_output_index] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(MAP_VALUE_LEN, VALUE_DEVICE_LOCAL_WORK_SIZE),
            ^{
               map_read_one_value__BLOCK_NAME(
                    map_id,
                    i, // entry_index
                    value_output_index,
                    values_output
               );
            }
        );
    }

    kernel void map_read_with_index__BLOCK_NAME(
        const uint map_id,
        global CL_TYPE* keys_output,
        global CL_TYPE* values_output,
        global int* indices_input
        ) {

        int i = get_global_id(0);
        int key_output_index = i * MAP_KEY_LEN;
        int value_output_index = i * MAP_VALUE_LEN;

        int entry_index = indices_input[i];

        for (int index = 0; index < MAP_KEY_LEN; index++) {
            keys_output[index + key_output_index] = map_keys__BLOCK_NAME[map_id][entry_index][index];
        }

        for (int index = 0; index < MAP_VALUE_LEN; index++) {
            values_output[index + value_output_index] = map_values__BLOCK_NAME[map_id][entry_index][index];
        }
    }

    kernel void map_read_with_index_and_cmq__BLOCK_NAME(
        queue_t q0,
        const uint map_id,
        global CL_TYPE* keys_output,
        global CL_TYPE* values_output,
        global int* indices_input,
        global int* enqueue_kernel_output
        ) {

        int i = get_global_id(0);
        int key_output_index = i * MAP_KEY_LEN;
        int value_output_index = i * MAP_VALUE_LEN;
        int enqueue_kernel_output_index = i * 2;

        int entry_index = indices_input[i];

        enqueue_kernel_output[CMQ_GET_MAP_KEY_INDEX + enqueue_kernel_output_index] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(MAP_KEY_LEN, KEY_DEVICE_LOCAL_WORK_SIZE),
            ^{
               map_read_one_key__BLOCK_NAME(
                    map_id,
                    entry_index,
                    key_output_index,
                    keys_output
               );
            }
        );

        enqueue_kernel_output[CMQ_GET_MAP_VALUE_INDEX + enqueue_kernel_output_index] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(MAP_VALUE_LEN, VALUE_DEVICE_LOCAL_WORK_SIZE),
            ^{
               map_read_one_value__BLOCK_NAME(
                    map_id,
                    entry_index,
                    value_output_index,
                    values_output
               );
            }
        );
    }
    "#;

const MAP_READ_KEYS_KERNEL: &str = r#"
    kernel void map_read_keys(
        queue_t q0,
        const uint map_id,
        global CL_TYPE* keys_output,
        global int* enqueue_kernel_output
        ) {

        KERNEL_BODY

    }
    "#;

const MAP_READ_KEYS_ENQUEUE_KERNEL: &str = r#"
        // BLOCK_NAME

        enqueue_kernel_output[BLOCK_INDEX] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(MAP_CAPACITY, CAPACITY_DEVICE_LOCAL_WORK_SIZE),
            ^{
               map_read_keys_for__BLOCK_NAME(
                    map_id,
                    keys_output
               );
            }
        );
     "#;

const KERNEL_GET_MATRIX_FOR_BLOCK: &str = r#"
    kernel void map_read_keys_for__BLOCK_NAME(
        const uint map_id,
        global CL_TYPE* keys_output
        ) {

        int i = get_global_id(0);
        int key_output_index = i * MAP_KEY_LEN;

        // BLOCK_NAME (block output index) = BLOCK_OUTPUT_INDEX

        for (int index = 0; index < MAP_KEY_LEN; index++) {
            keys_output[index + key_output_index + BLOCK_OUTPUT_INDEX] = map_keys__BLOCK_NAME[map_id][i][index];
        }
    }
    "#;

impl<T: ClTypeTrait> MapSrc<T> {
    pub fn generate_map_read_program_src(&self) -> String {
        let map_blocks = self.get_configs();

        let mut map_read_kernels = String::new();

        let mut map_read_keys_enqueue_kernel = String::new();
        let mut read_sub_kernels = String::new();

        for (i, config) in map_blocks.iter().enumerate() {
            let template = common_replace(MAP_READ_KERNELS, config);
            map_read_kernels.push_str(&template);

            // map_get_keys kernels
            let block_output_index: usize = map_blocks[0..i]
                .iter()
                .map(|x| x.capacity * x.key_len)
                .sum();

            let template = common_replace(KERNEL_GET_MATRIX_FOR_BLOCK, config)
                .replace("BLOCK_OUTPUT_INDEX", &block_output_index.to_string());
            read_sub_kernels.push_str(&template);

            let template = common_replace(MAP_READ_KEYS_ENQUEUE_KERNEL, config)
                .replace("BLOCK_INDEX", &i.to_string());
            map_read_keys_enqueue_kernel.push_str(&template);
        }

        let map_read_keys_kernel = MAP_READ_KEYS_KERNEL
            .replace("KERNEL_BODY", &map_read_keys_enqueue_kernel)
            .replace("CL_TYPE", T::cl_enum().to_cl_type_name());

        format!(
            "
    /// - MAP_READ START ///

    /// constants
    const int CMQ_GET_MAP_KEY_INDEX = 0;
    const int CMQ_GET_MAP_VALUE_INDEX = 1;

    /// globals
    // ...
    
    /// kernels
    {map_read_kernels}

    {read_sub_kernels}
    {map_read_keys_kernel}

    /// - MAP_READ END ///
        "
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a() {
        let mut map_src: MapSrc<i32> = MapSrc::default();
        map_src.add(256, 16);

        let program_source = map_src.generate_map_read_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_b() {
        let mut map_src: MapSrc<i32> = MapSrc::default();
        map_src.add(256, 8);
        map_src.add(512, 16);

        let program_source = map_src.generate_map_read_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_c() {
        let mut map_src: MapSrc<i32> = MapSrc::default();
        map_src.add(256, 8);
        map_src.add(512, 32);
        map_src.add(1024, 16);

        let program_source = map_src.generate_map_read_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }
}
