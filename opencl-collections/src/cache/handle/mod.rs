use opencl::opencl_sys::bindings::cl_int;

pub mod lru;
pub mod mini_lru;

#[derive(Debug, PartialEq)]
pub struct LruSummary {
    pub keys: usize,
    pub values: usize,
    pub priorities: usize,
}

impl LruSummary {
    pub fn new(keys: usize, values: usize, priorities: usize) -> Self {
        Self {
            keys,
            values,
            priorities,
        }
    }

    pub fn with(values: usize) -> Self {
        Self::new(values, values, values)
    }

    pub fn empty() -> Self {
        Self::new(0, 0, 0)
    }
}

pub type CacheIndices = Vec<cl_int>;

pub type CachePriorities = Vec<cl_int>;
