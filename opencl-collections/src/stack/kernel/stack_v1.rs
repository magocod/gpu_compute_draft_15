use crate::stack::config::StackSrc;
use crate::stack::kernel::common_replace;

const GLOBALS: &str = r#"
    __global int stack__STACK_ID[STACK_CAPACITY];
    __global int stack_top__STACK_ID = -1;
    "#;

// To write and read from the stack, functions similar to opencl pipes are used.
// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_C.html#pipe-functions
const PIPE_FUNCTIONS: &str = r#"
    int read_pipe_st(uint stack_id, int* v) {
        int front_i = -1;

        switch (stack_id) {
            READ_BODY_CASE
        }

        return front_i;
    }

    int write_pipe_st(uint stack_id, int* v) {
        int rear_i = -1;

        switch (stack_id) {
            WRITE_BODY_CASE
        }

        return rear_i;
    }
    "#;

const READ_PIPE_BODY_CASE: &str = r#"
            case STACK_ID:
                if (stack_top__STACK_ID >= 0) {
                    front_i = atomic_fetch_sub(&stack_top__STACK_ID, 1);

                    if (front_i >= 0) {
                        *v = stack__STACK_ID[front_i];
                    }
                }

                // FIXME stack overflow
                if (stack_top__STACK_ID < -1) {
                    atomic_store(&stack_top__STACK_ID, -1);
                }
                break;
    "#;

const WRITE_PIPE_BODY_CASE: &str = r#"
            case STACK_ID:
                if (stack_top__STACK_ID >= -1 && stack_top__STACK_ID <= STACK_MAX_CAPACITY) {
                    rear_i = (atomic_fetch_add(&stack_top__STACK_ID, 1) + 1);

                    if (STACK_MAX_CAPACITY >= rear_i) {

                        stack__STACK_ID[rear_i] = *v;

                    } else {

                        rear_i = - 1;
                        // FIXME stack overflow
                        atomic_store(&stack_top__STACK_ID, STACK_MAX_CAPACITY);

                    }
                }
                break;
    "#;

const BASE_KERNELS: &str = r#"
    kernel void stack_reset__STACK_ID() {
        int i = get_global_id(0);

        stack__STACK_ID[i] = 0;

        if (i == 0) {
            stack_top__STACK_ID = -1;
        }
    }

    kernel void stack_debug__STACK_ID(
        global int* items_output,
        global int* meta_output
        ) {
        int i = get_global_id(0);

        items_output[i] = stack__STACK_ID[i];

        if (i == 0) {
            meta_output[0] = stack_top__STACK_ID;
        }
    }
    "#;

const BASIC_KERNELS: &str = r#"
    kernel void write_to_stack(
        const uint stack_id,
        global int* input,
        global int* output
        ) {
        int i = get_global_id(0);
        output[i] = write_pipe_st(stack_id, &input[i]);
    }

    kernel void read_on_stack(
        const uint stack_id,
        global int* output
        ) {
        int i = get_global_id(0);

        int pi = -1;
        read_pipe_st(stack_id, &pi);

        output[i] = pi;
    }
    "#;

impl StackSrc {
    pub(crate) fn generate_stack_program_source_v1(&self) -> String {
        if self.is_empty() {
            return String::new();
        }

        let mut globals = String::new();

        let mut fn_write_cases = String::new();
        let mut fn_read_cases = String::new();

        let mut base_kernels = String::new();

        for st_config in self.get_configs() {
            let template = common_replace(GLOBALS, st_config);
            globals.push_str(&template);

            let template = common_replace(WRITE_PIPE_BODY_CASE, st_config);
            fn_write_cases.push_str(&template);

            let template = common_replace(READ_PIPE_BODY_CASE, st_config);
            fn_read_cases.push_str(&template);

            let template = common_replace(BASE_KERNELS, st_config);
            base_kernels.push_str(&template);
        }

        let pipe_functions = PIPE_FUNCTIONS
            .replace("WRITE_BODY_CASE", &fn_write_cases)
            .replace("READ_BODY_CASE", &fn_read_cases);

        format!(
            " 
    /// *** STACK V1 SRC *** ///

    // st = stack

    /// constants
    // ...

    /// globals
    {globals}

    /// kernels

    {pipe_functions}

    {base_kernels}

    {BASIC_KERNELS}

    /// *** STACK V1 SRC *** ///
    "
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a() {
        let mut stack_src = StackSrc::new();
        stack_src.add(8);

        let program_source = stack_src.generate_stack_program_source_v1();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_b() {
        let mut stack_src = StackSrc::new();
        stack_src.add(8);
        stack_src.add(32);
        stack_src.add(16);

        let program_source = stack_src.generate_stack_program_source_v1();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_is_empty() {
        let stack_src = StackSrc::new();

        let program_source = stack_src.generate_stack_program_source_v1();
        println!("{program_source}");
        assert!(program_source.is_empty());
    }
}
