use crate::config::{ClType, ClTypeTrait};
use crate::error::{OpenClResult, OpenclError, CL_COLLECTION_INVALID_MAP_VALUE_LEN};
use humansize::{format_size, DECIMAL};
use std::marker::PhantomData;

pub const DEFAULT_MAP_KEY_LENGTH: usize = 256;

pub fn get_map_block_name(value_len: usize) -> String {
    format!("{value_len}_byte")
}

// ...
pub const MAX_FIND_WORK_SIZE: usize = 32;
// pub const MAX_FIND_WORK_SIZE: usize = 64; // fatal error

pub fn check_max_find_work_size(len: usize) {
    if len > MAX_FIND_WORK_SIZE {
        panic!("TODO handle MAX_FIND_WORK_SIZE")
    }
}

pub const MAX_LOCAL_WORK_SIZE: usize = 256;
pub const DEFAULT_DEVICE_LOCAL_WORK_SIZE: usize = 256;

// TODO replace this method for kernel clGetKernelWorkGroupInfo - CL_KERNEL_WORK_GROUP_SIZE
pub fn check_local_work_size(global_work_size: usize) -> usize {
    if global_work_size > MAX_LOCAL_WORK_SIZE {
        return DEFAULT_DEVICE_LOCAL_WORK_SIZE;
    }
    global_work_size
}

/// ...
///
/// ```c
/// __global CL_TYPE map_keys__BLOCK_NAME[MAP_ID][MAP_CAPACITY][MAP_KEY_LEN];
/// __global CL_TYPE map_values__BLOCK_NAME[MAP_ID][MAP_CAPACITY][MAP_VALUE_LEN];
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct MapConfig<T: ClTypeTrait> {
    /// ...
    ///
    /// ```c
    /// __global CL_TYPE map_keys__256_byte[64][64][256];
    /// __global CL_TYPE map_values__256_byte[64][64][256];
    /// ```
    cl_type: PhantomData<T>,

    /// ...
    ///
    /// ```c
    /// __global int map_keys__BLOCK_NAME[64][64][256];
    /// __global int map_values__BLOCK_NAME[64][64][256];
    /// ```
    pub name: String,

    /// ...
    ///
    /// ```c
    /// __global int map_keys__256_byte[64][64][MAP_KEY_LEN];
    /// __global int map_values__256_byte[64][64][256];
    /// ```
    pub key_len: usize,

    /// ...
    ///
    /// ```c
    /// __global int map_keys__256_byte[64][64][256];
    /// __global int map_values__256_byte[64][64][MAP_VALUE_LEN];
    /// ```
    pub value_len: usize,

    /// ...
    ///
    /// ```c
    /// __global int map_keys__256_byte[64][MAP_CAPACITY][256];
    /// __global int map_values__256_byte[64][MAP_CAPACITY][256];
    /// ```
    pub capacity: usize,
}

impl<T: ClTypeTrait> MapConfig<T> {
    pub fn new(value_len: usize, capacity: usize) -> Self {
        Self {
            cl_type: Default::default(),
            name: get_map_block_name(value_len),
            key_len: DEFAULT_MAP_KEY_LENGTH,
            value_len,
            capacity,
        }
    }

    pub fn can_hold(&self, len: usize) {
        if len > self.capacity {
            panic!("TODO complete input len error")
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct MapBlockSummary<T: ClTypeTrait> {
    pub block: MapConfig<T>,
    /// memory required in bytes
    pub memory_required: usize,
    pub memory_required_text: String,
    // ...
    pub reserved: usize,
}

impl<T: ClTypeTrait> MapBlockSummary<T> {
    pub fn new(map_block_config: &MapConfig<T>, reserved: usize) -> Self {
        let type_mul = match T::cl_enum() {
            ClType::U8 => 1,
            ClType::U16 => 2,
            ClType::U32 => 4,
            ClType::U64 => 8,
            ClType::I16 => 2,
            ClType::I32 => 4,
            ClType::I64 => 8,
        };

        let bytes = (map_block_config.capacity * map_block_config.value_len) * type_mul;

        Self {
            block: map_block_config.clone(),
            memory_required: bytes,
            memory_required_text: format_size(bytes, DECIMAL),
            reserved,
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct MapSummary<T: ClTypeTrait> {
    pub reserved: usize,
    pub capacity: usize,
    /// memory required in bytes
    pub map_memory_required: usize,
    pub map_memory_required_text: String,
    /// memory required in bytes
    pub total_memory_required: usize,
    pub total_memory_required_text: String,
    pub blocks: Vec<MapBlockSummary<T>>,
}

impl<T: ClTypeTrait> MapSummary<T> {
    pub fn new(total_maps: usize, blocks: Vec<MapBlockSummary<T>>) -> Self {
        let map_memory_required: usize = blocks.iter().map(|x| x.memory_required).sum();
        let reserved = blocks.iter().map(|x| x.reserved).sum();
        let capacity = blocks.iter().map(|x| x.block.capacity).sum();

        Self {
            reserved,
            capacity,
            map_memory_required,
            map_memory_required_text: format_size(map_memory_required, DECIMAL),
            total_memory_required: map_memory_required * total_maps,
            total_memory_required_text: format_size(map_memory_required * total_maps, DECIMAL),
            blocks,
        }
    }

    pub fn create(reserved: usize, total_maps: usize, blocks: Vec<MapBlockSummary<T>>) -> Self {
        let map_memory_required: usize = blocks.iter().map(|x| x.memory_required).sum();
        let capacity = blocks.iter().map(|x| x.block.capacity).sum();

        Self {
            reserved,
            capacity,
            map_memory_required,
            map_memory_required_text: format_size(map_memory_required, DECIMAL),
            total_memory_required: map_memory_required * total_maps,
            total_memory_required_text: format_size(map_memory_required * total_maps, DECIMAL),
            blocks,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MapSrc<T: ClTypeTrait> {
    cl_type: PhantomData<T>,
    total_maps: usize,
    blocks: Vec<MapConfig<T>>,
    // The code generation using strings got a little out of hand,
    // compilation becomes very slow if all kernels written in strings are used.
    pub optional_sources: Vec<String>,
}

impl<T: ClTypeTrait> MapSrc<T> {
    pub fn new(total_maps: usize) -> Self {
        if total_maps == 0 {
            // TODO update to result
            panic!("Handle error map_src total_maps = 0")
        }

        Self {
            cl_type: Default::default(),
            total_maps,
            blocks: Vec::new(),
            optional_sources: Vec::new(),
        }
    }

    pub fn get_total_maps(&self) -> usize {
        self.total_maps
    }

    pub fn set_total_maps(&mut self, total_maps: usize) {
        self.total_maps = total_maps;
    }

    pub fn get_configs(&self) -> &Vec<MapConfig<T>> {
        &self.blocks
    }

    pub fn add(&mut self, value_len: usize, capacity: usize) {
        self.blocks.push(MapConfig::new(value_len, capacity));
    }

    pub fn reorder_by_capacity(&mut self) {
        self.blocks.sort_by(|a, b| a.value_len.cmp(&b.value_len));
    }

    pub fn get_config_by_value_len(&self, value_len: usize) -> OpenClResult<&MapConfig<T>> {
        for c in &self.blocks {
            if c.value_len == value_len {
                return Ok(c);
            }
        }
        Err(OpenclError::OpenclCollection(
            CL_COLLECTION_INVALID_MAP_VALUE_LEN,
        ))
    }

    pub fn get_config_by_gte_value_len(&self, value_len: usize) -> OpenClResult<&MapConfig<T>> {
        for c in &self.blocks {
            if c.value_len >= value_len {
                return Ok(c);
            }
        }
        Err(OpenclError::OpenclCollection(
            CL_COLLECTION_INVALID_MAP_VALUE_LEN,
        ))
    }

    /// The highest value (capacity) of all map settings.
    pub fn get_max_capacity(&self) -> usize {
        let values = self.blocks.iter().map(|x| x.capacity).max().unwrap();
        values
    }

    // The highest value (value_len) of all map settings.
    pub fn get_max_value_len(&self) -> usize {
        let values = self.blocks.iter().map(|x| x.value_len).max().unwrap();
        values
    }

    // the total items that can be saved on all maps.
    pub fn get_maximum_assignable_keys(&self) -> usize {
        let value = self.blocks.iter().map(|x| x.capacity).sum();
        value
    }

    // FIXME configuration verification
    pub fn check(&self) -> bool {
        let block_sizes: Vec<_> = self.blocks.iter().map(|x| x.value_len).collect();

        for config in self.blocks.iter() {
            let c = block_sizes
                .iter()
                .filter(|&x| x.eq(&config.value_len))
                .count();
            if c > 1 {
                return false;
            }
        }

        true
    }

    /// Approximate memory required for the program
    /// FIXME correct sum of required memory, add value of keys in the results.
    /// Not all values are taken into account when calculating the required memory.
    pub fn summary(&self) -> MapSummary<T> {
        let summaries: Vec<MapBlockSummary<T>> = self
            .blocks
            .iter()
            .map(|x| -> MapBlockSummary<T> { MapBlockSummary::new(x, 0) })
            .collect();

        MapSummary::new(self.total_maps, summaries)
    }
}

impl<T: ClTypeTrait> Default for MapSrc<T> {
    fn default() -> Self {
        Self::new(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{BYTE_256, BYTE_512, KB, MB};

    #[test]
    fn test_map_src_add() {
        let mut map_src: MapSrc<i32> = MapSrc::new(2);
        map_src.add(BYTE_256, 8);
        map_src.add(BYTE_512, 16);

        let configs = map_src.get_configs();

        println!("{map_src:#?}");

        assert_eq!(configs.len(), 2);
        assert_eq!(configs[0], MapConfig::new(BYTE_256, 8));
        assert_eq!(configs[1], MapConfig::new(BYTE_512, 16));
    }

    #[test]
    #[should_panic]
    fn test_map_src_invalid_total_maps() {
        let _map_src: MapSrc<i32> = MapSrc::new(0);
    }

    #[test]
    fn test_map_src_check_config_without_repeated_map_value_len() {
        let mut map_src: MapSrc<i32> = MapSrc::default();
        map_src.add(BYTE_256, 8);
        map_src.add(BYTE_512, 8);
        map_src.add(MB, 8);

        let result = map_src.check();
        assert!(result);
    }

    #[test]
    fn test_map_src_check_config_with_repeated_map_value_len() {
        let mut map_src: MapSrc<i32> = MapSrc::default();
        map_src.add(KB, 8);
        map_src.add(KB, 16);

        let result = map_src.check();
        assert!(!result);
    }

    #[test]
    fn test_map_src_reorder_config_by_capacity() {
        let mut map_src: MapSrc<i32> = MapSrc::default();
        map_src.add(MB, 8);
        map_src.add(KB, 8);
        map_src.add(BYTE_256, 8);
        map_src.add(BYTE_512, 8);

        map_src.reorder_by_capacity();
        let block_sizes: Vec<usize> = map_src.get_configs().iter().map(|x| x.value_len).collect();
        assert_eq!(block_sizes, vec![BYTE_256, BYTE_512, KB, MB]);
    }

    #[test]
    fn test_map_src_get_max_capacity() {
        let mut map_src: MapSrc<i32> = MapSrc::default();
        map_src.add(BYTE_256, 8);
        map_src.add(KB, 32);
        map_src.add(MB, 16);

        let result = map_src.get_max_capacity();
        assert_eq!(result, 32);
    }

    #[test]
    fn test_map_src_get_max_value_len() {
        let mut map_src: MapSrc<i32> = MapSrc::default();
        map_src.add(BYTE_256, 8);
        map_src.add(KB, 32);
        map_src.add(MB, 16);

        let result = map_src.get_max_value_len();
        assert_eq!(result, MB);
    }

    #[test]
    fn test_map_src_get_config_by_value_len() {
        let mut map_src: MapSrc<i32> = MapSrc::default();
        map_src.add(BYTE_256, 8);
        map_src.add(KB, 32);
        map_src.add(MB, 16);

        let result = map_src.get_config_by_value_len(KB).unwrap();
        assert_eq!(result, &MapConfig::new(KB, 32));
    }

    #[test]
    fn test_map_src_get_config_by_gte_value_len() {
        let mut map_src: MapSrc<i32> = MapSrc::default();
        map_src.add(BYTE_256, 8);
        map_src.add(KB, 32);
        map_src.add(MB, 16);

        let result = map_src.get_config_by_gte_value_len(KB + 1).unwrap();
        assert_eq!(result, &MapConfig::new(MB, 16));
    }

    #[test]
    fn test_map_src_get_summary() {
        let mut map_src: MapSrc<i32> = MapSrc::new(2);
        map_src.add(BYTE_256, 8);
        map_src.add(KB, 32);
        map_src.add(MB, 16);

        let summary = map_src.summary();
        println!("{:#?}", summary);

        // TODO assert test
        assert_eq!(summary.reserved, 0);
        assert_eq!(summary.capacity, map_src.get_maximum_assignable_keys());
    }
}
