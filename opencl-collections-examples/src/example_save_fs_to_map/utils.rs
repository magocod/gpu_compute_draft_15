use crate::common::get_files_from_path;
use crate::example_save_fs_to_map::system::ArcOpenclBlock;
use opencl_collections::config::ClTypeTrait;
use opencl_collections::map::config::MapSrc;
use opencl_collections::map::handle::MapHandle;
use opencl_collections::opencl::opencl_sys::bindings::{cl_int, cl_short};
use opencl_collections::utils::{KB, MB};
use std::ops::Add;
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::{fs, thread};

// type conversions

// FIXME type conversions
pub fn from_vec_i16_to_vec_u8(buf: Vec<cl_short>) -> Vec<u8> {
    buf.iter().map(|x| *x as u8).collect()
}

pub fn wrapper_read_buf_u8(buf: &[u8]) -> Vec<u8> {
    buf.to_vec()
}

pub fn wrapper_read_vec_u8(v: Vec<u8>) -> Vec<u8> {
    v
}

#[derive(Debug)]
pub struct ThreadSummary {
    pub id: usize,
    pub success: usize,
    pub errors: usize,
}

impl ThreadSummary {
    pub fn new(id: usize, success: usize, errors: usize) -> Self {
        Self {
            id,
            success,
            errors,
        }
    }
}

#[derive(Debug)]
pub struct MainThreadSummary {
    pub success: usize,
    pub errors: usize,
    pub child_threads: Vec<ThreadSummary>,
}

impl MainThreadSummary {
    pub fn new(summaries: Vec<ThreadSummary>) -> Self {
        Self {
            success: summaries.iter().map(|x| x.success).sum(),
            errors: summaries.iter().map(|x| x.errors).sum(),
            child_threads: summaries,
        }
    }
}

// read from fs

pub fn read_from_fs_and_single_write_to_map<T: ClTypeTrait + 'static>(
    ocl_block: ArcOpenclBlock,
    map_src: &MapSrc<T>,
    total_threads: usize,
    files: &[PathBuf],
    process_fn: fn(&[u8]) -> Vec<T>,
) -> MainThreadSummary {
    let work_by_group = files.len() as f64 / total_threads as f64;

    println!("files - total {}", files.len());
    println!("files - work group {}", work_by_group.ceil());

    let chunks: Vec<_> = files
        .chunks(work_by_group.ceil() as usize)
        .map(|x| x.to_vec())
        .collect();

    println!("files - chunks {}", chunks.len());

    let threads: Vec<_> = chunks
        .into_iter()
        .enumerate()
        .map(|(i, paths)| {
            let cl_block = ocl_block.clone();
            let m_src = map_src.clone();

            thread::spawn(move || {
                println!("  Thread #{i} started!");

                let map = MapHandle::new(i, &m_src, cl_block);
                let pipes = map.get_empty_keys_pipes().unwrap();

                let results: Vec<cl_int> = paths
                    .into_iter()
                    .map(|path_buf| -> cl_int {
                        // println!("{:?}", path_buf);

                        let p_buf = path_buf.to_string_lossy();
                        let entry_key = process_fn(p_buf.as_bytes());

                        // read from disk
                        let fs_read_value = fs::read(path_buf.as_path()).unwrap();
                        let entry_value = process_fn(&fs_read_value);

                        match map.insert_one(&entry_key, &entry_value, &pipes) {
                            Ok((map_entry, _block_size)) => {
                                // println!("block_index {:?} - {:?}", block_index, block_size);
                                if map_entry < 0 {
                                    println!(
                                        "      Thread #{i} - e: {:?} _ path: {:?} ",
                                        map_entry, path_buf
                                    );
                                }

                                map_entry
                            }
                            Err(e) => {
                                println!("     Thread #{i} - e: {} _ path: {:?}", e, path_buf);
                                -1
                            }
                        }
                        // thread::sleep(Duration::from_secs(5));
                    })
                    .collect();

                let t_summary = ThreadSummary::new(
                    i,
                    results.iter().filter(|&&x| x >= 0).count(),
                    results.iter().filter(|&&x| x < 0).count(),
                );

                println!("  Thread #{i} finished! {t_summary:#?}");

                t_summary
            })
        })
        .collect();

    let mut summaries = Vec::with_capacity(total_threads);

    for j_handle in threads.into_iter() {
        let t_summary = j_handle.join().unwrap();
        summaries.push(t_summary);
    }

    MainThreadSummary::new(summaries)
}

pub fn read_from_fs_and_multiple_write_to_map<T: ClTypeTrait + 'static>(
    ocl_block: ArcOpenclBlock,
    map_src: &MapSrc<T>,
    total_threads: usize,
    files: &[PathBuf],
    process_fn: fn(&[u8]) -> Vec<T>,
) -> MainThreadSummary {
    let work_by_group = files.len() as f64 / total_threads as f64;

    println!("files - total vec {}", files.len());
    println!("files - work group {}", work_by_group.ceil());

    let chunks: Vec<_> = files
        .chunks(work_by_group.ceil() as usize)
        .map(|x| x.to_vec())
        .collect();
    println!("files chunks {}", chunks.len());

    let threads: Vec<_> = chunks
        .into_iter()
        .enumerate()
        .map(|(i, paths)| {

            let cl_block = ocl_block.clone();
            let m_src = map_src.clone();

            thread::spawn(move || {
                println!("  Thread #{i} started!");

                let map = MapHandle::new(i, &m_src, cl_block);
                let pipes = map.get_empty_keys_pipes().unwrap();

                // let save_chunk_size = 8; // stable
                let save_chunk_size = 16; // stable
                // let save_chunk_size = 32; // gpu deadlock
                // let save_chunk_size = 64; // fatal error

                // (success, errors)
                let results: Vec<(usize, usize)> = paths.chunks(save_chunk_size).enumerate().map(|(save_chunk_id, paths_group)| {

                    let entry_keys: Vec<Vec<T>> = paths_group
                        .iter()
                        .map(|path_buf| {

                            let p_buf = path_buf.to_string_lossy();
                            process_fn(p_buf.as_bytes())

                        })
                        .collect();

                    // get file buffers
                    let entry_values: Vec<Vec<T>> = paths_group
                        .iter()
                        .map(|path_buf| {

                            // read from disk
                            let fs_read_value = fs::read(path_buf.as_path()).unwrap();
                            process_fn(&fs_read_value)

                        })
                        .collect();

                    match map.map_insert(
                        &entry_keys,
                        &entry_values,
                        &pipes,
                    ) {
                        Ok((indices, blocks)) => {
                            // println!("Thread #{i} indices - {:?}", indices);
                            // println!("Thread #{i} blocks - {:?}", blocks);

                            let success = indices.iter().filter(|&&x| x >= 0).count();
                            let failed = indices.iter().filter(|&&x| x < 0).count();

                            if failed > 0 {
                                println!("      Thread #{i} save_chunk_id - {save_chunk_id} - failed: {failed}");

                                for (result_index, &m_index) in indices.iter().enumerate() {
                                    if m_index < 0 {
                                        println!("      Thread #{i} - save_chunk_id - {} - e: {} _ {} - path {:?}",
                                                 save_chunk_id,
                                                 m_index,
                                                 blocks[result_index],
                                                 paths_group[result_index]
                                        );
                                    }
                                }

                            }

                            (success, failed)
                        }
                        Err(e) => {
                            println!("      Thread #{i} save_chunk_id - {save_chunk_id} - error {}", e);
                            (0, paths_group.len())
                        }
                    }

                }).collect();

                let t_summary = ThreadSummary::new(
                    i,
                    results.iter().map(|(success, _)| success).sum(),
                    results.iter().map(|(_, errors)| errors).sum(),
                );

                println!("  Thread #{i} finished!");
                t_summary
            })
        })
        .collect();

    let mut summaries = Vec::with_capacity(total_threads);

    for j_handle in threads.into_iter() {
        let t_summary = j_handle.join().unwrap();
        summaries.push(t_summary);
    }

    MainThreadSummary::new(summaries)
}

// write to fs

pub fn write_map_fields_to_fs<T: ClTypeTrait + 'static>(
    ocl_block: ArcOpenclBlock,
    map_src: &MapSrc<T>,
    total_threads: usize,
    process_fn: fn(Vec<T>) -> Vec<u8>,
) -> MainThreadSummary {
    let threads: Vec<_> = (0..total_threads)
        .map(|i| {
            let cl_block = ocl_block.clone();
            let m_src = map_src.clone();

            thread::spawn(move || {
                let map = MapHandle::new(i, &m_src, cl_block);
                println!("  Thread #{i} started!");

                let t_summary = match map.read_assigned_keys() {
                    Ok(entries) => {
                        let results: Vec<usize> = entries
                            .into_iter()
                            .map(|map_block_entries| {
                                // println!(
                                //     "   Thread #{i}  block {} capacity {}/{}",
                                //     map_block_entries.config.name,
                                //     map_block_entries.pairs.len(),
                                //     map_block_entries.config.capacity
                                // );

                                let block_total = map_block_entries.pairs.len();

                                for pair in map_block_entries.pairs {
                                    let key_u8 = process_fn(pair.get_key());
                                    let value_u8 = process_fn(pair.get_value());

                                    let base_path = String::from_utf8(key_u8).unwrap();
                                    // println!("base_path {:?}", p);

                                    // update base path
                                    let tmp_path = base_path.replace("./", "./tmp/");
                                    // println!("tmp_path {}", tmp_path);

                                    // remove file from path
                                    let mut tmp_path_dir = PathBuf::from(tmp_path.as_str());
                                    tmp_path_dir.pop();
                                    // println!("tmp_path_dir {:?}", tmp_path_dir);

                                    // TODO handle errors
                                    fs::create_dir_all(tmp_path_dir).unwrap();
                                    fs::write(tmp_path, value_u8).unwrap();
                                }

                                block_total
                            })
                            .collect();

                        ThreadSummary::new(i, results.iter().sum(), 0)
                    }
                    Err(e) => {
                        println!("     Thread #{i} - e: {}", e);
                        ThreadSummary::new(i, 0, 0)
                    }
                };

                println!("  Thread #{i} - {:?} finished!", t_summary);

                t_summary
            })
        })
        .collect();

    let mut summaries = Vec::with_capacity(total_threads);

    for j_handle in threads.into_iter() {
        let t_summary = j_handle.join().unwrap();
        summaries.push(t_summary);
    }

    MainThreadSummary::new(summaries)
}

#[derive(Debug)]
pub struct DirSummary {
    paths: Vec<PathBuf>,
    total: usize,
    // device with 8gb
    large_files: usize,
    // device with 4gb
    small_files: usize,
}

impl DirSummary {
    pub fn new(paths: Vec<PathBuf>, large_files: usize, small_files: usize) -> Self {
        Self {
            paths,
            total: small_files + large_files,
            large_files,
            small_files,
        }
    }
}

impl Add for DirSummary {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut other_paths = self.paths.clone();
        let mut paths = other.paths.clone();
        paths.append(&mut other_paths);

        Self {
            paths,
            total: self.total + other.total,
            small_files: self.small_files + other.small_files,
            large_files: self.large_files + other.large_files,
        }
    }
}

// 32kb
const MAX_SMALL_FILE_SIZE: usize = KB * 32;

// 16mb
const MAX_LARGE_FILE_SIZE: usize = MB * 16;

#[derive(Debug)]
pub struct GlobalSummary {
    pub total: usize,
    pub success: usize,
    pub errors: usize,
    pub total_threads: usize,
}

#[derive(Debug)]
pub struct CopySummary {
    pub global_summary: GlobalSummary,
    pub dir_summary: DirSummary,
    pub save_small_files_to_map_4gb: MainThreadSummary,
    pub save_large_files_to_map_8gb: MainThreadSummary,
    pub write_map_4gb_in_fs: MainThreadSummary,
    pub write_map_8gb_in_fs: MainThreadSummary,
}

impl CopySummary {
    pub fn new(
        total_threads: usize,
        dir_summary: DirSummary,
        save_small_files_to_map_4gb: MainThreadSummary,
        save_large_files_to_map_8gb: MainThreadSummary,
        write_map_4gb_in_fs: MainThreadSummary,
        write_map_8gb_in_fs: MainThreadSummary,
    ) -> Self {
        Self {
            global_summary: GlobalSummary {
                total: dir_summary.total,
                success: save_small_files_to_map_4gb.success + save_large_files_to_map_8gb.success,
                errors: save_small_files_to_map_4gb.errors + save_large_files_to_map_8gb.errors,
                total_threads,
            },
            dir_summary,
            save_small_files_to_map_4gb,
            save_large_files_to_map_8gb,
            write_map_4gb_in_fs,
            write_map_8gb_in_fs,
        }
    }
}

pub type CopyFromFsToMap<T> = fn(
    ocl_block: ArcOpenclBlock,
    map_src: &MapSrc<T>,
    total_threads: usize,
    files: &[PathBuf],
    process_fn: fn(&[u8]) -> Vec<T>,
) -> MainThreadSummary;

#[allow(clippy::too_many_arguments)]
pub fn copy_directory_to_map<T: ClTypeTrait + 'static>(
    total_threads: usize,
    path: &Path,
    ocl_block_8gb: ArcOpenclBlock,
    ocl_block_4gb: ArcOpenclBlock,
    map_8gb_src: &MapSrc<T>,
    map_4gb_src: &MapSrc<T>,
    fn_save_to_map: CopyFromFsToMap<T>,
    process_fs: fn(&[u8]) -> Vec<T>,
    process_map_entry: fn(Vec<T>) -> Vec<u8>,
) -> CopySummary {
    let mut large_files: Vec<PathBuf> = vec![];
    let mut small_files: Vec<PathBuf> = vec![];

    get_files_from_path(
        path,
        &mut large_files,
        &mut small_files,
        MAX_SMALL_FILE_SIZE as u64,
        MAX_LARGE_FILE_SIZE as u64,
    )
    .unwrap();

    let dir_summary = DirSummary::new(vec![path.into()], large_files.len(), small_files.len());
    println!("{dir_summary:#?}");

    // fs -> map

    // small files
    let now = Instant::now();
    println!("start: write small files, fs -> map");
    let save_small_files_to_map_4gb = fn_save_to_map(
        ocl_block_4gb.clone(),
        map_4gb_src,
        total_threads,
        &small_files,
        process_fs,
    );
    println!(
        "end: write small files, fs -> map, seg: {}",
        now.elapsed().as_secs()
    );

    // large files
    let now = Instant::now();
    println!("start: write large files, fs -> map");
    let save_large_files_to_map_8gb = fn_save_to_map(
        ocl_block_8gb.clone(),
        map_8gb_src,
        total_threads,
        &large_files,
        process_fs,
    );
    println!(
        "end: write large files, fs -> map, seg: {}",
        now.elapsed().as_secs()
    );

    // map -> fs

    // small files
    let now = Instant::now();
    println!("start: write small files, map -> fs");
    let write_map_4gb_in_fs =
        write_map_fields_to_fs(ocl_block_4gb, map_4gb_src, total_threads, process_map_entry);
    println!(
        "end: write small files, map -> fs, seg: {}",
        now.elapsed().as_secs()
    );

    // large files
    let now = Instant::now();
    println!("start: write large files, map -> fs");
    let write_map_8gb_in_fs =
        write_map_fields_to_fs(ocl_block_8gb, map_8gb_src, total_threads, process_map_entry);
    println!(
        "end: write large files, map -> fs, seg: {}",
        now.elapsed().as_secs()
    );

    CopySummary::new(
        total_threads,
        dir_summary,
        save_small_files_to_map_4gb,
        save_large_files_to_map_8gb,
        write_map_4gb_in_fs,
        write_map_8gb_in_fs,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn copy_directory_to_map_v2<T: ClTypeTrait + 'static>(
    total_threads: usize,
    path: &Path,
    ocl_block_8gb: ArcOpenclBlock,
    ocl_block_4gb: ArcOpenclBlock,
    map_8gb_src: &MapSrc<T>,
    map_4gb_src: &MapSrc<T>,
    fn_save_to_map: CopyFromFsToMap<T>,
    process_fs: fn(&[u8]) -> Vec<T>,
    process_map_entry: fn(Vec<T>) -> Vec<u8>,
) -> CopySummary {
    let mut large_files: Vec<PathBuf> = vec![];
    let mut small_files: Vec<PathBuf> = vec![];

    get_files_from_path(
        path,
        &mut large_files,
        &mut small_files,
        (KB * 32) as u64,
        (MB * 16) as u64,
    )
    .unwrap();

    let dir_summary = DirSummary::new(vec![path.into()], large_files.len(), small_files.len());
    println!("{dir_summary:#?}");

    let m_src = map_4gb_src.clone();

    let thread_small_files = thread::spawn(move || {
        // fs -> map

        let now = Instant::now();
        println!("start: write small files, fs -> map");
        let fs_to_map_result = fn_save_to_map(
            ocl_block_4gb.clone(),
            &m_src,
            total_threads,
            &small_files,
            process_fs,
        );
        println!(
            "end: write small files, fs -> map, seg: {}",
            now.elapsed().as_secs()
        );

        // map -> fs

        let now = Instant::now();
        println!("start: write small files, map -> fs");
        let map_to_fs_result =
            write_map_fields_to_fs(ocl_block_4gb, &m_src, total_threads, process_map_entry);
        println!(
            "end: write small files, map -> fs, seg: {}",
            now.elapsed().as_secs()
        );

        (fs_to_map_result, map_to_fs_result)
    });

    let m_src = map_8gb_src.clone();

    let thread_large_files = thread::spawn(move || {
        // fs -> map

        let now = Instant::now();
        println!("start: write large files, map -> fs");
        let fs_to_map_result = fn_save_to_map(
            ocl_block_8gb.clone(),
            &m_src,
            total_threads,
            &large_files,
            process_fs,
        );
        println!(
            "end: write large files, fs -> map, seg: {}",
            now.elapsed().as_secs()
        );

        // map -> fs

        let now = Instant::now();
        println!("start: write large files, map -> fs");
        let map_to_fs_result =
            write_map_fields_to_fs(ocl_block_8gb, &m_src, total_threads, process_map_entry);
        println!(
            "end: write large files, map -> fs, seg: {}",
            now.elapsed().as_secs()
        );

        (fs_to_map_result, map_to_fs_result)
    });

    let (save_small_files_to_map_4gb, write_map_4gb_in_fs) = thread_small_files.join().unwrap();
    let (save_large_files_to_map_8gb, write_map_8gb_in_fs) = thread_large_files.join().unwrap();

    CopySummary::new(
        total_threads,
        dir_summary,
        save_small_files_to_map_4gb,
        save_large_files_to_map_8gb,
        write_map_4gb_in_fs,
        write_map_8gb_in_fs,
    )
}

#[derive(Debug)]
pub struct MapsSummary {
    total_reserved: usize,
    total_capacity: usize,
}

impl MapsSummary {
    pub fn new(total_reserved: usize, total_capacity: usize) -> Self {
        Self {
            total_reserved,
            total_capacity,
        }
    }
}

impl Add for MapsSummary {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            total_reserved: self.total_reserved + other.total_reserved,
            total_capacity: self.total_capacity + other.total_capacity,
        }
    }
}

pub fn get_map_summary<T: ClTypeTrait>(
    total_threads: usize,
    ocl_block: ArcOpenclBlock,
    map_src: &MapSrc<T>,
) -> MapsSummary {
    println!("start: map -> summary");

    let map_summaries: Vec<_> = (0..total_threads)
        .map(|i| {
            let map = MapHandle::new(i, map_src, ocl_block.clone());
            let summary = map.get_summary().unwrap();

            println!(
                "  Map #i {i} - reserved: {} / {}",
                summary.reserved, summary.capacity
            );
            // println!("  Map #i {i} - {:#?}", summaries);

            summary
        })
        .collect();

    let total_reserved: usize = map_summaries.iter().map(|s| s.reserved).sum();
    let total_capacity: usize = map_summaries.iter().map(|s| s.capacity).sum();

    println!();
    println!("total_reserved {total_reserved}");
    println!("total_capacity {total_capacity}");
    println!();

    println!("end: map -> summary");
    println!();

    MapsSummary::new(total_reserved, total_capacity)
}
