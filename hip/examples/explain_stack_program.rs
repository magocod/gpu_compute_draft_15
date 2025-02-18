use hip::collections::stack::kernel::generate_stack_program_source;
use std::fs;

fn main() {
    let stack_src = generate_stack_program_source(32);
    // println!("{stack_src}");
    fs::write("./tmp/stack_src.cpp", &stack_src).unwrap();

    // ...
}
