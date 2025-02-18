use crate::error::{
    OpenClResult, OpenclError, CL_COLLECTION_INVALID_LRU_ID, CL_COLLECTION_INVALID_MINI_LRU_ID,
};

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum CacheType {
    MiniLRU,
    LRU,
    // TTL
}

/// ...
///
/// mini lru cache
/// ```c
/// __global int mini_lru_last_priority__CACHE_ID = 1;
/// __global int mini_lru_top__CACHE_ID = 0;
///
/// __global int mini_lru_keys__CACHE_ID[CACHE_CAPACITY];
/// __global int mini_lru_values__CACHE_ID[CACHE_CAPACITY];
/// __global int mini_lru_priorities__CACHE_ID[CACHE_CAPACITY];
///
/// // array set ...
/// ```
///
/// lru cache
/// ```c
/// __global int lru_last_priority__CACHE_ID = 0;
/// __global int lru_top__CACHE_ID = 0;
///
/// __global int lru_keys__CACHE_ID[CACHE_CAPACITY][KEY_LEN];
/// __global int lru_values__CACHE_ID[CACHE_CAPACITY][VALUE_LEN];
/// __global int lru_priorities__CACHE_ID[CACHE_CAPACITY];
///
/// // array set ...
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CacheConfig {
    pub id: usize,
    pub key_len: usize,
    pub value_len: usize,
    pub capacity: usize,
    pub cache_type: CacheType,
}

impl CacheConfig {
    pub fn new(
        id: usize,
        key_len: usize,
        value_len: usize,
        capacity: usize,
        cache_type: CacheType,
    ) -> Self {
        Self {
            id,
            key_len,
            value_len,
            capacity,
            cache_type,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CacheSrc {
    blocks: Vec<CacheConfig>,
}

impl CacheSrc {
    pub fn new() -> Self {
        Self { blocks: Vec::new() }
    }

    pub fn get_configs_by_type(&self, cache_type: CacheType) -> Vec<&CacheConfig> {
        self.blocks
            .iter()
            .filter(|&x| x.cache_type == cache_type)
            .collect()
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    pub fn add(
        &mut self,
        key_len: usize,
        value_len: usize,
        capacity: usize,
        cache_type: CacheType,
    ) {
        // get_cache_id
        let id = self
            .blocks
            .iter()
            .filter(|x| x.cache_type == cache_type)
            .count();

        self.blocks.push(CacheConfig::new(
            id, key_len, value_len, capacity, cache_type,
        ));
    }

    pub fn add_mini_lru(&mut self, capacity: usize) {
        self.add(1, 1, capacity, CacheType::MiniLRU)
    }

    pub fn add_lru(&mut self, key_len: usize, value_len: usize, capacity: usize) {
        self.add(key_len, value_len, capacity, CacheType::LRU)
    }

    pub fn get_config_by_id(&self, id: usize, cache_type: CacheType) -> Option<&CacheConfig> {
        self.blocks
            .iter()
            .find(|x| x.id == id && x.cache_type == cache_type)
    }

    pub fn get_mini_lru_by_id(&self, id: usize) -> OpenClResult<&CacheConfig> {
        match self.get_config_by_id(id, CacheType::MiniLRU) {
            None => Err(OpenclError::OpenclCollection(
                CL_COLLECTION_INVALID_MINI_LRU_ID,
            )),
            Some(c) => Ok(c),
        }
    }

    pub fn get_lru_by_id(&self, id: usize) -> OpenClResult<&CacheConfig> {
        match self.get_config_by_id(id, CacheType::LRU) {
            None => Err(OpenclError::OpenclCollection(CL_COLLECTION_INVALID_LRU_ID)),
            Some(c) => Ok(c),
        }
    }
}

impl Default for CacheSrc {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_src_add() {
        let mut cache_src = CacheSrc::new();
        cache_src.add(8, 256, 256, CacheType::LRU);
        cache_src.add(32, 1, 1, CacheType::LRU);
        cache_src.add_lru(8, 16, 16);

        println!("{:#?}", cache_src);

        assert_eq!(cache_src.len(), 3);
        let ids: Vec<usize> = cache_src
            .get_configs_by_type(CacheType::LRU)
            .iter()
            .map(|x| x.id)
            .collect();
        assert_eq!(ids, vec![0, 1, 2]);
    }

    #[test]
    fn test_cache_src_get_config_by_id() {
        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(8);
        cache_src.add_lru(16, 16, 32);
        cache_src.add_lru(32, 64, 32);

        assert_eq!(
            cache_src.get_mini_lru_by_id(0),
            Ok(&CacheConfig::new(0, 1, 1, 8, CacheType::MiniLRU))
        );
        assert_eq!(
            cache_src.get_lru_by_id(1),
            Ok(&CacheConfig::new(1, 32, 64, 32, CacheType::LRU))
        );

        assert_eq!(
            cache_src.get_lru_by_id(5),
            Err(OpenclError::OpenclCollection(CL_COLLECTION_INVALID_LRU_ID))
        );
        assert_eq!(
            cache_src.get_mini_lru_by_id(1),
            Err(OpenclError::OpenclCollection(
                CL_COLLECTION_INVALID_MINI_LRU_ID
            ))
        );
    }
}
