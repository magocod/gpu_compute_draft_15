use crate::config::ClTypeTrait;
use crate::map::config::MapSrc;
use crate::utils::remove_padding_cl_default;
use opencl::opencl_sys::bindings::cl_int;
use opencl::wrapper::system::OpenclCommonOperation;
use std::marker::PhantomData;
use std::sync::Arc;

// pub mod loader;
pub mod reset;
pub mod tmp;

pub mod read;
pub mod write;

pub mod test_utils;

pub const KEY_NOT_AVAILABLE_TO_ASSIGN: cl_int = -1;
pub const KEY_EXISTS: cl_int = -2;
pub const KEY_NOT_EXISTS: cl_int = -3;
pub const CANNOT_APPEND_VALUE: cl_int = -4;
pub const MAP_VALUE_FULL: cl_int = -5;
pub const MAP_VALUE_NOT_ENOUGH_SPACE: cl_int = -6;

pub type MapKey<T> = Vec<T>;
pub type MapValue<T> = Vec<T>;

pub type MapKeys<T> = Vec<Vec<T>>;
pub type MapValues<T> = Vec<Vec<T>>;

#[derive(Debug, Clone, PartialEq)]
pub struct Pair<T: ClTypeTrait> {
    pub key: Vec<T>,
    pub value: Vec<T>,
    pub entry_index: Option<usize>,
}

impl<T: ClTypeTrait> Pair<T> {
    pub fn new(key: Vec<T>, value: Vec<T>) -> Self {
        Self::create_with_index(key, value, None)
    }

    pub fn create_with_index(key: Vec<T>, value: Vec<T>, entry_index: Option<usize>) -> Self {
        Self {
            key,
            value,
            entry_index,
        }
    }

    pub fn get_key(&self) -> Vec<T> {
        remove_padding_cl_default(&self.key)
    }

    pub fn get_value(&self) -> Vec<T> {
        remove_padding_cl_default(&self.value)
    }
}

pub type EntryIndex = cl_int;
pub type EntryIndices = Vec<cl_int>;

pub type MapBlockSize = cl_int;
pub type MapBlockSizes = Vec<cl_int>;

#[derive(Debug)]
pub struct MapHandle<T: ClTypeTrait, D: OpenclCommonOperation> {
    map_id: usize,
    map_src: MapSrc<T>,
    system: Arc<D>,
}

impl<T: ClTypeTrait, D: OpenclCommonOperation> MapHandle<T, D> {
    pub fn new(map_id: usize, map_src: &MapSrc<T>, system: Arc<D>) -> MapHandle<T, D> {
        // fatal error map_id invalid

        Self {
            map_id,
            map_src: map_src.clone(),
            system,
        }
    }

    pub fn get_map_id(&self) -> usize {
        self.map_id
    }

    pub fn set_map_id(&mut self, id: usize) {
        self.map_id = id;
    }
}

#[derive(Debug)]
pub struct Handle<T: ClTypeTrait, D: OpenclCommonOperation> {
    map_src: MapSrc<T>,
    system: Arc<D>,
    phantom: PhantomData<T>,
}

impl<T: ClTypeTrait, D: OpenclCommonOperation> Handle<T, D> {
    pub fn new(map_src: &MapSrc<T>, system: Arc<D>) -> Handle<T, D> {
        Self {
            map_src: map_src.clone(),
            system,
            phantom: Default::default(),
        }
    }
}
