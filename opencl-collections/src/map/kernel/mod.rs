use crate::config::ClTypeTrait;
use crate::map::config::{
    check_local_work_size, MapConfig, MapSrc, DEFAULT_DEVICE_LOCAL_WORK_SIZE,
    DEFAULT_MAP_KEY_LENGTH, MAX_FIND_WORK_SIZE, MAX_LOCAL_WORK_SIZE,
};

pub mod name;

pub mod map_add;
pub mod map_append;
pub mod map_append_for_block;
pub mod map_copy;
pub mod map_deduplication;
pub mod map_deep_deduplication;
pub mod map_get;
pub mod map_get_empty_key;
pub mod map_get_index;
pub mod map_get_summary;
pub mod map_insert;
pub mod map_put;
pub mod map_read;
pub mod map_read_assigned_keys;
pub mod map_read_sizes;
pub mod map_remove;
pub mod map_remove_v2;
pub mod map_reorder;
pub mod map_reset;

#[allow(dead_code)]
pub enum RemoveVersion {
    V1,
    V2,
}

pub const REMOVE_VERSION: RemoveVersion = RemoveVersion::V1;

// v1
// error while assigning by using struct.
// const GLOBAL_ARRAY: &str = r#"
//     struct map_entry_BLOCK_NAME {
//         CL_TYPE key[MAP_KEY_LEN];
//         CL_TYPE value[MAP_VALUE_LEN];
//     };
//
//     __global struct map_entry_BLOCK_NAME map_BLOCK_NAME[TOTAL_MAPS][MAP_CAPACITY];
//     "#;

const GLOBAL_ARRAY: &str = r#"
    // BLOCK_NAME

    __global CL_TYPE map_keys__BLOCK_NAME[TOTAL_MAPS][MAP_CAPACITY][MAP_KEY_LEN];
    __global CL_TYPE map_values__BLOCK_NAME[TOTAL_MAPS][MAP_CAPACITY][MAP_VALUE_LEN];
    "#;

const FN_UTILS: &str = r#"
    bool is_map_key_is_equal_to_input__BLOCK_NAME(
        uint map_id,
        int entry_index,
        int key_input_index,
        CL_TYPE* key_input
        ) {
        for (int key_index = 0; key_index < MAP_KEY_LEN; key_index++) {
            if (map_keys__BLOCK_NAME[map_id][entry_index][key_index] != key_input[key_index + key_input_index]) {
                return false;
            }
        }
        return true;
    }

    bool is_map_keys_are_equal__BLOCK_NAME(
        uint map_id,
        int first_entry_index,
        int second_entry_index
        ) {
        for (int key_index = 0; key_index < MAP_KEY_LEN; key_index++) {
            if (map_keys__BLOCK_NAME[map_id][first_entry_index][key_index] != map_keys__BLOCK_NAME[map_id][second_entry_index][key_index]) {
                return false;
            }
        }
        return true;
    }

    bool is_map_key_empty__BLOCK_NAME(uint map_id, int entry_index) {
        for (int key_index = 0; key_index < MAP_KEY_LEN; key_index++) {
            if (map_keys__BLOCK_NAME[map_id][entry_index][key_index] != CL_DEFAULT_VALUE) {
                return false;
            }
        }
        return true;
    }
    
    int get_last_index_with_value_in_map_key__BLOCK_NAME(uint map_id, int entry_index) {

        for (int index = (MAP_KEY_LEN - 1); index >= 0; index--) {
            if (map_keys__BLOCK_NAME[map_id][entry_index][index] != CL_DEFAULT_VALUE) {
                return index;
            }
        }

        return -1;
    }
    
    int get_map_key_size__BLOCK_NAME(uint map_id, int entry_index) {
        int last_index = get_last_index_with_value_in_map_key__BLOCK_NAME(map_id, entry_index);
        return last_index + 1;
    }

    int get_last_index_with_value_in_map_value__BLOCK_NAME(uint map_id, int entry_index) {

        for (int index = (MAP_VALUE_LEN - 1); index >= 0; index--) {
            if (map_values__BLOCK_NAME[map_id][entry_index][index] != CL_DEFAULT_VALUE) {
                return index;
            }
        }

        return -1;
    }

    int get_map_value_size__BLOCK_NAME(uint map_id, int entry_index) {
        int last_index = get_last_index_with_value_in_map_value__BLOCK_NAME(map_id, entry_index);
        return last_index + 1;
    }
    "#;

// FIXME replace with opencl functions and a correct assignment of local_work_size
const TEMPORAL_FN_UTILS: &str = r#"
    int check_device_local_work_size(int global_work_size) {
        if (global_work_size > MAX_LOCAL_WORK_SIZE) {
            return DEFAULT_DEVICE_LOCAL_WORK_SIZE;
        }

        return global_work_size;
    }
    "#;

const KERNEL_UTILS: &str = r#"
    kernel void get_ordered_pipe_content(
        __read_only pipe int pipe_0,
        const uint pipe_elements,
        global int* output
        ) {
    
        for (int i = 0; i < pipe_elements; i++) {
           int pi = -1;
    
           // TODO check read_pipe return
           read_pipe(pipe_0, &pi);
    
           output[i] = pi;
        }
    }
    "#;

pub fn common_replace<T: ClTypeTrait>(src: &str, config: &MapConfig<T>) -> String {
    let key_device_local_work_size = check_local_work_size(config.key_len).to_string();
    let value_device_local_work_size = check_local_work_size(config.value_len).to_string();
    let capacity_device_local_work_size = check_local_work_size(config.capacity).to_string();

    src.replace("BLOCK_NAME", &config.name)
        .replace(
            "DEFAULT_MAP_KEY_LENGTH",
            &DEFAULT_MAP_KEY_LENGTH.to_string(),
        )
        .replace("MAP_KEY_LEN", &config.key_len.to_string())
        .replace("MAP_VALUE_LEN", &config.value_len.to_string())
        .replace("MAP_CAPACITY", &config.capacity.to_string())
        .replace("CL_TYPE", T::cl_enum().to_cl_type_name())
        .replace("CL_DEFAULT_VALUE", &T::cl_enum().cl_default().to_string())
        .replace("KEY_DEVICE_LOCAL_WORK_SIZE", &key_device_local_work_size)
        .replace(
            "VALUE_DEVICE_LOCAL_WORK_SIZE",
            &value_device_local_work_size,
        )
        .replace(
            "CAPACITY_DEVICE_LOCAL_WORK_SIZE",
            &capacity_device_local_work_size,
        )
}

impl<T: ClTypeTrait> MapSrc<T> {
    // Delete src is left here, to facilitate the reading of the global variable
    pub fn add_map_remove_program_src(&mut self) -> &mut Self {
        let src = match REMOVE_VERSION {
            RemoveVersion::V1 => self.generate_map_remove_program_src(MAX_FIND_WORK_SIZE),
            RemoveVersion::V2 => self.generate_map_remove_v2_program_src(MAX_FIND_WORK_SIZE),
        };
        self.optional_sources.push(src);
        self
    }

    pub fn build(&self) -> String {
        let blocks = self.get_configs();

        if blocks.is_empty() {
            return String::new();
        }

        let mut global_arrays = String::from("");
        let mut fn_utils = String::from("");

        let total_maps = self.get_total_maps().to_string();

        for config in blocks {
            let template = common_replace(GLOBAL_ARRAY, config).replace("TOTAL_MAPS", &total_maps);
            global_arrays.push_str(&template);

            let template = common_replace(FN_UTILS, config);
            fn_utils.push_str(&template);
        }

        let map_get_empty_key_program_src = self.generate_map_get_empty_key_program_src();

        let map_get_summary_program_src = self.generate_map_get_summary_program_src();

        let map_put_program_src = self.generate_map_put_program_src();

        let map_read_assigned_keys_program_src = self.generate_map_read_assigned_keys_program_src();

        let map_read_program_src = self.generate_map_read_program_src();

        let map_reset_program_src = self.generate_map_reset_program_src();

        // The code generation using strings got a little out of hand,
        // compilation becomes very slow if all kernels written in strings are used.
        let mut optional_src = String::new();

        // TODO improve this iteration
        for source in self.optional_sources.iter() {
            optional_src = format!(
                "
                {optional_src}
                {source}
                "
            );
        }

        let kernel_utils = KERNEL_UTILS;

        let temporal_fn_utils = TEMPORAL_FN_UTILS
            .replace("MAX_LOCAL_WORK_SIZE", &MAX_LOCAL_WORK_SIZE.to_string())
            .replace(
                "DEFAULT_DEVICE_LOCAL_WORK_SIZE",
                &DEFAULT_DEVICE_LOCAL_WORK_SIZE.to_string(),
            );

        format!(
            "
    /// *** MAP SRC START *** ///

    /// constants
    // ...

    /// globals
    {global_arrays}

    /// kernels
    {fn_utils}

    {temporal_fn_utils}
    {kernel_utils}

    {map_get_empty_key_program_src}

    {map_put_program_src}
    {map_read_program_src}
    {map_read_assigned_keys_program_src}

    {map_reset_program_src}
    {map_get_summary_program_src}

    /// optional src
    {optional_src}

    /// *** MAP SRC END *** ///
    "
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{BYTE_256, BYTE_512, KB};

    #[test]
    fn test_a() {
        let mut map_src: MapSrc<i32> = MapSrc::default();
        map_src.add(BYTE_512, 8);

        let program_source = map_src.build();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_b() {
        let mut map_src: MapSrc<i16> = MapSrc::new(8);
        map_src.add(BYTE_256, 16);
        map_src.add(BYTE_512, 8);
        map_src.add(KB, 32);

        let program_source = map_src.build();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_with_optional_source() {
        let mut map_src: MapSrc<i32> = MapSrc::default();
        map_src.add(BYTE_512, 8);

        let program_source = map_src
            .add_map_add_program_src()
            .add_map_append_program_src(16)
            .build();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_config_is_empty() {
        let map_src: MapSrc<i16> = MapSrc::new(8);

        let program_source = map_src.build();
        println!("{program_source}");
        assert!(program_source.is_empty());
    }
}
