use crate::config::ClTypeTrait;
use crate::map::config::{MapSrc, DEFAULT_MAP_KEY_LENGTH};
use crate::map::kernel::common_replace;

const MAP_READ_ASSIGNED_KEYS_KERNEL: &str = r#"
    kernel void map_read_assigned_keys(
        queue_t q0,
        const uint map_id,
        global int* sizes_output,
        global CL_TYPE* keys_output,
        global CL_TYPE* values_output,
        global int* enqueue_kernel_output
        ) {
        
        KERNEL_BODY

    }
    "#;

const MAP_READ_ENTRIES_ENQUEUE_KERNEL: &str = r#"
        // BLOCK_NAME

        enqueue_kernel_output[BLOCK_INDEX] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(MAP_CAPACITY, CAPACITY_DEVICE_LOCAL_WORK_SIZE),
            ^{
               map_read_entries__BLOCK_NAME(
                    map_id,
                    sizes_output,
                    keys_output,
                    values_output
               );
            }
        );
     "#;

const MAP_READ_ENTRIES: &str = r#"
    kernel void map_read_entries__BLOCK_NAME(
        const uint map_id,
        global int* sizes_output,
        global CL_TYPE* keys_output,
        global CL_TYPE* values_output
        ) {

        int i = get_global_id(0);
        int key_output_index = i * DEFAULT_MAP_KEY_LENGTH;
        int value_output_index = i * MAP_VALUE_LEN;

        int key_size = get_map_key_size__BLOCK_NAME(
            map_id,
            i // entry_index
        );
        sizes_output[i + SIZE_OUTPUT_INDEX__BLOCK_NAME] = key_size;

        if (key_size > 0) {

            for (int index = 0; index < DEFAULT_MAP_KEY_LENGTH; index++) {
                keys_output[index + key_output_index + KEY_OUTPUT_INDEX__BLOCK_NAME] = map_keys__BLOCK_NAME[map_id][i][index];
            }

            for (int index = 0; index < MAP_VALUE_LEN; index++) {
                values_output[index + value_output_index + VALUE_OUTPUT_INDEX__BLOCK_NAME] = map_values__BLOCK_NAME[map_id][i][index];
            }

        } else {

            for (int index = 0; index < DEFAULT_MAP_KEY_LENGTH; index++) {
                keys_output[index + key_output_index + KEY_OUTPUT_INDEX__BLOCK_NAME] = CL_DEFAULT_VALUE;
            }

            for (int index = 0; index < MAP_VALUE_LEN; index++) {
                values_output[index + value_output_index + VALUE_OUTPUT_INDEX__BLOCK_NAME] = CL_DEFAULT_VALUE;
            }

        }
    }
    "#;

const TMP_CONST_DEF: &str = r#"
    const int SIZE_OUTPUT_INDEX__BLOCK_NAME = SIZE_INDEX;
    const int KEY_OUTPUT_INDEX__BLOCK_NAME = KEY_INDEX;
    const int VALUE_OUTPUT_INDEX__BLOCK_NAME = VALUE_INDEX;
    "#;

impl<T: ClTypeTrait> MapSrc<T> {
    pub fn generate_map_read_assigned_keys_program_src(&self) -> String {
        // const definitions
        let mut output_const_def = String::new();

        let mut map_read_entries_enqueue_kernel = String::new();
        let mut map_read_entries_kernels = String::new();

        let map_blocks = self.get_configs();

        for (i, config) in map_blocks.iter().enumerate() {
            let size_output_index: usize = map_blocks[0..i].iter().map(|x| x.capacity).sum();

            let key_output_index: usize = map_blocks[0..i]
                .iter()
                .map(|x| x.capacity * DEFAULT_MAP_KEY_LENGTH)
                .sum();

            let value_output_index: usize = map_blocks[0..i]
                .iter()
                .map(|x| x.capacity * x.value_len)
                .sum();

            let template = common_replace(TMP_CONST_DEF, config)
                .replace("SIZE_INDEX", &size_output_index.to_string())
                .replace("KEY_INDEX", &key_output_index.to_string())
                .replace("VALUE_INDEX", &value_output_index.to_string());

            output_const_def.push_str(&template);

            let template =
                common_replace(MAP_READ_ENTRIES, config).replace("BLOCK_INDEX", &i.to_string());
            map_read_entries_kernels.push_str(&template);

            let template = common_replace(MAP_READ_ENTRIES_ENQUEUE_KERNEL, config)
                .replace("BLOCK_INDEX", &i.to_string());
            map_read_entries_enqueue_kernel.push_str(&template);
        }

        let map_read_assigned_kernel = MAP_READ_ASSIGNED_KEYS_KERNEL
            .replace("KERNEL_BODY", &map_read_entries_enqueue_kernel)
            .replace("CL_TYPE", T::cl_enum().to_cl_type_name());

        format!(
            "
    /// - MAP_READ_ASSIGNED_KEYS START ///

    /// constants
    {output_const_def}

    /// globals
    // ...

    /// kernels
    {map_read_entries_kernels}
    {map_read_assigned_kernel}

    /// - MAP_READ_ASSIGNED_KEYS END ///
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

        let program_source = map_src.generate_map_read_assigned_keys_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_b() {
        let mut map_src: MapSrc<i32> = MapSrc::new(2);
        map_src.add(256, 8);
        map_src.add(512, 16);

        let program_source = map_src.generate_map_read_assigned_keys_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_c() {
        let mut map_src: MapSrc<i32> = MapSrc::new(2);
        map_src.add(256, 8);
        map_src.add(512, 32);
        map_src.add(1024, 16);

        let program_source = map_src.generate_map_read_assigned_keys_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }
}
