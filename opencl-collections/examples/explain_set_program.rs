use opencl::wrapper::system::System;
use opencl_collections::config::DEFAULT_DEVICE_INDEX;
use opencl_collections::set::config::{ArraySetVersion, SetSrc};
use std::path::Path;
use std::time::{Duration, Instant};
use std::{fs, thread};
// TODO ...

const SECOND_SLEEP: u64 = 5;

fn create_program<P: AsRef<Path>>(src: &str, path: P) {
    // println!("{program_source}");
    fs::write(path, src).unwrap();

    println!("start compile cl");
    let now = Instant::now();
    let system = System::new(DEFAULT_DEVICE_INDEX, src).unwrap();
    println!("system {}", system.get_id());
    println!("{} seg compile cl", now.elapsed().as_secs());

    thread::sleep(Duration::from_secs(SECOND_SLEEP));
}

fn main() {
    let mut set_src = SetSrc::new();
    set_src.add(256);
    set_src.add(512);

    create_program(
        &set_src.build(ArraySetVersion::V1),
        "./tmp/array_set_v1_src.cl",
    );
    create_program(
        &set_src.build(ArraySetVersion::V2),
        "./tmp/array_set_v2_src.cl",
    );
}
