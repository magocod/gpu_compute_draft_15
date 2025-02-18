use opencl::wrapper::system::{OpenclCommonOperation, System};
use opencl_collections::cache::config::CacheSrc;
use opencl_collections::config::DEFAULT_DEVICE_INDEX;
use std::time::{Duration, Instant};
use std::{fs, thread};

const SECOND_SLEEP: u64 = 5;

fn main() {
    let mut cache_src = CacheSrc::new();
    cache_src.add_mini_lru(128);
    cache_src.add_mini_lru(256);
    cache_src.add_lru(128, 256, 256);
    cache_src.add_lru(256, 256, 512);

    let program_source = cache_src.build();
    // println!("{program_source}");
    fs::write("./tmp/cache_src.cl", &program_source).unwrap();

    println!("start compile cl");
    let now = Instant::now();
    let system = System::new(DEFAULT_DEVICE_INDEX, &program_source).unwrap();
    println!("system {}", system.get_id());
    system.initialize_memory().unwrap();
    println!("{} seg compile cl", now.elapsed().as_secs());

    thread::sleep(Duration::from_secs(SECOND_SLEEP));
}
