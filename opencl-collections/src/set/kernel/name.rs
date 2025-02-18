pub fn get_set_kernel_name(kernel_name: &str, id: usize) -> String {
    kernel_name.replace("SET_ID", &id.to_string())
}

// array set

pub const ARRAY_SET_DEBUG: &str = "array_set_debug__SET_ID";
pub const ARRAY_SET_RESET: &str = "array_set_reset__SET_ID";

pub const WRITE_IN_ARRAY_SET: &str = "write_in_array_set__SET_ID";
pub const REMOVE_IN_ARRAY_SET: &str = "remove_in_array_set__SET_ID";

// array set v1
pub const WRITE_WITH_CMQ_IN_ARRAY_SET: &str = "write_with_cmq_in_array_set__SET_ID";
pub const WRITE_WITH_SINGLE_THREAD_IN_ARRAY_SET: &str =
    "write_with_single_thread_in_array_set__SET_ID";

// set
// ...
