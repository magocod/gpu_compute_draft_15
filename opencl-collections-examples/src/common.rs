use std::path::{Path, PathBuf};
use std::{fs, io};

pub const DEFAULT_DEVICE_INDEX: usize = 1;

pub fn get_files_from_path(
    dir: &Path,
    big_files_vec: &mut Vec<PathBuf>,
    small_files_vec: &mut Vec<PathBuf>,
    max_small_file_size: u64,
    max_large_file_size: u64,
) -> io::Result<()> {
    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                get_files_from_path(
                    &path,
                    big_files_vec,
                    small_files_vec,
                    max_small_file_size,
                    max_large_file_size,
                )?;
            } else {
                let size = entry.metadata()?.len();
                // println!("size {}", size);

                if size <= max_small_file_size {
                    small_files_vec.push(entry.path());
                } else if size <= max_large_file_size {
                    big_files_vec.push(entry.path());
                } else {
                    println!("file weight out of range {:?}", entry.path());
                }
            }
        }
    }
    Ok(())
}

#[derive(Debug)]
pub struct Counter {
    pub id: usize,
    pub count: u64,
}

impl Counter {
    pub fn new(id: usize) -> Self {
        Self { id, count: 0 }
    }

    pub fn inc(&mut self) {
        self.count += 1;
    }

    pub fn inc_with(&mut self, n: u64) {
        self.count += n;
    }
}
