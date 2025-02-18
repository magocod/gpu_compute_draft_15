//! # utils
//!
//! This module contains temporary functions that will be modified
//! in future versions (better implementations).
//!

pub const EXAMPLE_KERNEL_NAME: &str = "vecAdd";

pub const EXAMPLE_PROGRAM_SOURCE: &str = r#"
        __kernel void vecAdd(
            __global int *input_a,
            __global int *input_b,
            __global int *output_c
        ) {
            int id = get_global_id(0);

            output_c[id] = input_a[id] + input_b[id];
        }
    "#;

pub fn get_board_name_amd(name: &str) -> &'static str {
    match name {
        "gfx1032" => "AMD Radeon RX 6600",
        "gfx1034" => "AMD Radeon RX 6500 XT",
        "gfx90c:xnack-" => "5700G AMD Radeon Graphics",
        _ => "...",
    }
}
