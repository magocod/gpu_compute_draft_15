use crate::config::ClTypeTrait;
use crate::dictionary::config::DictConfig;
use opencl::opencl_sys::bindings::{cl_int, cl_uint};

pub mod dict_v1;

pub const KEYS_NOT_AVAILABLE: cl_int = -1;
pub const KEY_NOT_EXIST: cl_int = -2;
pub const DUPLICATE_KEY: cl_int = -3;

#[derive(Debug, PartialEq)]
pub struct DictSnapshotSummary {
    pub keys: usize,
    pub values: usize,
}

impl DictSnapshotSummary {
    pub fn new(keys: usize, values: usize) -> Self {
        Self { keys, values }
    }

    pub fn with(entries: usize) -> Self {
        Self::new(entries, entries)
    }
}

#[derive(Debug, PartialEq)]
pub struct DictSummary<T: ClTypeTrait> {
    pub config: DictConfig<T>,
    pub keys_sizes: Vec<cl_uint>,
    pub values_sizes: Vec<cl_uint>,
    pub available: usize,
}

impl<T: ClTypeTrait> DictSummary<T> {
    pub fn new(
        config: &DictConfig<T>,
        keys_sizes: &[cl_uint],
        values_sizes: &[cl_uint],
        available: usize,
    ) -> Self {
        Self {
            config: config.clone(),
            keys_sizes: keys_sizes.to_vec(),
            values_sizes: values_sizes.to_vec(),
            available,
        }
    }

    pub fn create_empty(config: &DictConfig<T>) -> Self {
        Self::new(
            config,
            &vec![0; config.capacity],
            &vec![0; config.capacity],
            config.capacity,
        )
    }
}
