pub fn get_dict_kernel_name(kernel_name: &str, id: usize) -> String {
    kernel_name.replace("DICT_ID", &id.to_string())
}

pub const DICT_DEBUG: &str = "dict_debug__DICT_ID";
pub const DICT_RESET: &str = "dict_reset__DICT_ID";

pub const DICT_GET_KEYS: &str = "dict_get_keys__DICT_ID";
pub const DICT_GET_SUMMARY: &str = "dict_get_summary__DICT_ID";

pub const READ_ON_DICT: &str = "read_on_dict__DICT_ID";
pub const READ_VALUE_SIZE_ON_DICT: &str = "read_value_size_on_dict__DICT_ID";

pub const WRITE_TO_DICT: &str = "write_to_dict__DICT_ID";
pub const REMOVE_FROM_DICT: &str = "remove_from_dict__DICT_ID";

pub const VERIFY_AND_WRITE_IN_DICT: &str = "verify_and_write_in_dict__DICT_ID";
pub const VERIFY_AND_REMOVE_IN_DICT: &str = "verify_and_remove_in_dict__DICT_ID";
