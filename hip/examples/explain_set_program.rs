use hip::collections::set::kernel::generate_array_set_program_source;
use std::fs;

fn main() {
    let set_src = generate_array_set_program_source(32);
    // println!("{set_src}");
    fs::write("./tmp/set_src.cpp", &set_src).unwrap();

    // ...
}
