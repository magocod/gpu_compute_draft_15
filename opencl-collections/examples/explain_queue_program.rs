use opencl::wrapper::system::{OpenclCommonOperation, System};
use opencl_collections::config::DEFAULT_DEVICE_INDEX;
use opencl_collections::queue::config::QueueSrc;
use std::time::{Duration, Instant};
use std::{fs, thread};
// TODO ...

const SECOND_SLEEP: u64 = 5;

fn main() {
    let mut queue_src = QueueSrc::new();
    // queue_src.add_lq(256);
    // queue_src.add_lq(256);

    queue_src.add_pq(256);
    queue_src.add_pq(256);

    // queue_src.add_cq(256);
    // queue_src.add_cq(256);

    // queue_src.add_lq(MB * 50);
    // queue_src.add_pq(MB * 50);
    // queue_src.add_cq(MB * 50);

    let program_source = queue_src.build();
    // println!("{program_source}");
    fs::write("./tmp/queue_src.cl", &program_source).unwrap();

    println!("start compile cl");
    let now = Instant::now();
    let system = System::new(DEFAULT_DEVICE_INDEX, &program_source).unwrap();
    println!("system {}", system.get_id());
    system.initialize_memory().unwrap();
    println!("{} seg compile cl", now.elapsed().as_secs());

    thread::sleep(Duration::from_secs(SECOND_SLEEP));
}
