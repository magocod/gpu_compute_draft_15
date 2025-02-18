#[allow(unknown_lints)]
#[allow(clippy::manual_div_ceil)]
pub fn ceiling_div(dividend: u32, divisor: u32) -> u32 {
    (dividend + divisor - 1) / divisor
}
