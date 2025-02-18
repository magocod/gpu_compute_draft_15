use opencl_collections::config::ClTypeTrait;
use opencl_collections::map::config::{MapSrc, MAX_FIND_WORK_SIZE};
use opencl_collections::map::handle::Handle;
use opencl_collections::opencl::wrapper::system::{OpenclCommonOperation, System};
use opencl_collections::utils::{KB, MB};
use std::sync::Arc;
use std::time::Instant;

// REQUIRED 2 GPUS

const GPU_8GB_DEVICE_INDEX: usize = 0;
const GPU_4GB_DEVICE_INDEX: usize = 1;

pub type ArcOpenclBlock = Arc<System>;

/// big files - gpu 8gb
pub fn create_system_8gb<T: ClTypeTrait>(total_maps: usize) -> (ArcOpenclBlock, MapSrc<T>) {
    let mut map_src: MapSrc<T> = MapSrc::new(total_maps);

    map_src.add(KB * 512, 512);
    map_src.add(MB, 16);
    map_src.add(MB * 8, 16);
    map_src.add(MB * 16, 8);

    map_src.add_map_insert_program_src(MAX_FIND_WORK_SIZE);

    let summary = map_src.summary();
    println!(
        "!!!! map device 8gb - memory_required {}",
        summary.total_memory_required_text
    );

    println!("**** map device 8gb - start compile cl");
    let now = Instant::now();

    let system = System::new(GPU_8GB_DEVICE_INDEX, &map_src.build()).unwrap();
    system.initialize_memory().unwrap();

    println!(
        "**** map device 8gb compile end: {} seg",
        now.elapsed().as_secs()
    );

    let arc_system = Arc::new(system);

    let h = Handle::new(&map_src, arc_system.clone());
    h.initialize_all_maps().unwrap();

    (arc_system, map_src)
}

// small files - gpu 4gb
pub fn create_system_4gb<T: ClTypeTrait>(total_maps: usize) -> (ArcOpenclBlock, MapSrc<T>) {
    let mut map_src: MapSrc<T> = MapSrc::new(total_maps);

    let memory_blocks = 2;

    for i in 0..memory_blocks {
        map_src.add(KB * (i + 1), 2048);
    }

    map_src.add(KB * 8, 1024);
    map_src.add(KB * 16, 1024);
    map_src.add(KB * 32, 1024);
    // config.add(MB * 16, 8);

    map_src.add_map_insert_program_src(MAX_FIND_WORK_SIZE);

    let summary = map_src.summary();
    println!(
        "!!!! map device 4gb - memory_required {}",
        summary.total_memory_required_text
    );

    println!("**** map device 4gb - start compile cl");
    let now = Instant::now();

    let system = System::new(GPU_4GB_DEVICE_INDEX, &map_src.build()).unwrap();
    system.initialize_memory().unwrap();

    println!(
        "**** map device 4gb compile end: {} seg",
        now.elapsed().as_secs()
    );

    let arc_system = Arc::new(system);

    let h = Handle::new(&map_src, arc_system.clone());
    h.initialize_all_maps().unwrap();

    (arc_system, map_src)
}
