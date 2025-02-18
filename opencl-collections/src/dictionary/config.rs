use crate::config::ClTypeTrait;
use crate::error::{OpenClResult, OpenclError, CL_COLLECTION_INVALID_DICT_ID};
use std::marker::PhantomData;

/// ...
///
/// ```c
/// __global CL_TYPE dict_keys__DICT_ID[DICT_CAPACITY][DICT_KEY_LEN];
/// __global CL_TYPE dict_values__DICT_ID[DICT_CAPACITY][DICT_VALUE_LEN];
///
/// __global int dict_entries__DICT_ID[DICT_CAPACITY];
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct DictConfig<T: ClTypeTrait> {
    cl_type: PhantomData<T>,
    pub id: usize,
    pub key_len: usize,
    pub value_len: usize,
    pub capacity: usize,
}

impl<T: ClTypeTrait> DictConfig<T> {
    pub fn new(id: usize, key_len: usize, value_len: usize, capacity: usize) -> Self {
        Self {
            cl_type: Default::default(),
            id,
            key_len,
            value_len,
            capacity,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DictSrc<T: ClTypeTrait> {
    cl_type: PhantomData<T>,
    blocks: Vec<DictConfig<T>>,
}

impl<T: ClTypeTrait> DictSrc<T> {
    pub fn new() -> Self {
        Self {
            cl_type: Default::default(),
            blocks: Vec::new(),
        }
    }

    pub fn get_configs(&self) -> &Vec<DictConfig<T>> {
        &self.blocks
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    pub fn add(&mut self, key_len: usize, value_len: usize, capacity: usize) {
        let id = self.blocks.len();

        self.blocks
            .push(DictConfig::new(id, key_len, value_len, capacity));
    }

    pub fn add_many(&mut self, key_len: usize, value_len: usize, capacity: usize, quantity: usize) {
        for _ in 0..quantity {
            self.add(key_len, value_len, capacity);
        }
    }

    pub fn get_config_by_id(&self, id: usize) -> OpenClResult<&DictConfig<T>> {
        match self.blocks.iter().find(|x| x.id == id) {
            None => Err(OpenclError::OpenclCollection(CL_COLLECTION_INVALID_DICT_ID)),
            Some(c) => Ok(c),
        }
    }
}

impl<T: ClTypeTrait> Default for DictSrc<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dict_src_add() {
        let mut dict_src: DictSrc<i32> = DictSrc::new();
        dict_src.add(32, 32, 8);
        dict_src.add(256, 256, 32);
        dict_src.add(8, 8, 8);

        let ids: Vec<usize> = dict_src.get_configs().iter().map(|x| x.id).collect();
        assert_eq!(ids, vec![0, 1, 2]);
    }

    #[test]
    fn test_dict_src_add_many() {
        let mut dict_src: DictSrc<i32> = DictSrc::new();
        dict_src.add_many(256, 256, 8, 3);

        let ids: Vec<usize> = dict_src.get_configs().iter().map(|x| x.id).collect();
        assert_eq!(ids, vec![0, 1, 2]);
    }

    #[test]
    fn test_dict_src_get_config_by_id() {
        let mut dict_src: DictSrc<i32> = DictSrc::new();
        dict_src.add(8, 8, 8);
        dict_src.add(16, 16, 16);

        assert_eq!(
            dict_src.get_config_by_id(0),
            Ok(&DictConfig::new(0, 8, 8, 8))
        );
        assert_eq!(
            dict_src.get_config_by_id(1),
            Ok(&DictConfig::new(1, 16, 16, 16))
        );

        assert_eq!(
            dict_src.get_config_by_id(5),
            Err(OpenclError::OpenclCollection(CL_COLLECTION_INVALID_DICT_ID))
        );
    }
}
