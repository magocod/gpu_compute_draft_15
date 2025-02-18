pub fn get_stack_kernel_name(kernel_name: &str, id: usize) -> String {
    kernel_name.replace("STACK_ID", &id.to_string())
}

// stack v1
pub const STACK_DEBUG: &str = "stack_debug__STACK_ID";
pub const STACK_RESET: &str = "stack_reset__STACK_ID";

pub const WRITE_TO_STACK: &str = "write_to_stack";
pub const READ_ON_STACK: &str = "read_on_stack";
