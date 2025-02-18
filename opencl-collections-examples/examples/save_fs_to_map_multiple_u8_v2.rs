use opencl_collections_examples::example_save_fs_to_map::system::{
    create_system_4gb, create_system_8gb,
};
use opencl_collections_examples::example_save_fs_to_map::utils::{
    copy_directory_to_map_v2, get_map_summary, read_from_fs_and_multiple_write_to_map,
    wrapper_read_buf_u8, wrapper_read_vec_u8,
};
use std::path::Path;
use std::thread::available_parallelism;
use std::time::Instant;

pub type BaseType = u8;

fn main() {
    let total_time = Instant::now();

    let default_parallelism_approx = available_parallelism().unwrap().get();
    println!("available_parallelism {}", default_parallelism_approx);

    let total_threads = 4;
    println!("total_threads {}", total_threads);

    println!("**** start compile cl");
    let now = Instant::now();

    let (system_8gb, map_8gb_src) = create_system_8gb(total_threads);
    let (system_4gb, map_4gb_src) = create_system_4gb(total_threads);

    println!("**** end compile cl: {} seg", now.elapsed().as_secs());

    // let base_path = Path::new("./resources/ts_example");
    //
    // let cs = copy_directory_to_map_v2::<BaseType>(
    //     total_threads,
    //     &base_path,
    //     system_8gb.clone(),
    //     system_4gb.clone(),
    //     &map_8gb_src,
    //     &map_4gb_src,
    //     read_from_fs_and_multiple_write_to_map,
    //     wrapper_read_buf_u8,
    //     wrapper_read_vec_u8,
    // );
    // println!("{:#?}", cs);

    let base_path = Path::new("./resources/nest_example");

    let cs = copy_directory_to_map_v2::<BaseType>(
        total_threads,
        &base_path,
        system_8gb.clone(),
        system_4gb.clone(),
        &map_8gb_src,
        &map_4gb_src,
        read_from_fs_and_multiple_write_to_map,
        wrapper_read_buf_u8,
        wrapper_read_vec_u8,
    );
    println!("{:#?}", cs);

    println!("**** device block 4gb - summary");
    let mg1 = get_map_summary(total_threads, system_4gb, &map_4gb_src);

    println!("**** device block 8gb - summary");
    let mg2 = get_map_summary(total_threads, system_8gb, &map_8gb_src);

    let mg = mg1 + mg2;
    println!("{:#?}", mg);

    println!();
    println!("{:#?}", cs.global_summary);

    println!("**** completed: {} seg", total_time.elapsed().as_secs());

    // thread::sleep(Duration::from_millis(0));
    println!("Thread #main finished!");
}
