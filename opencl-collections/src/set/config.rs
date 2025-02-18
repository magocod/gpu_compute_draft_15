use crate::error::{OpenClResult, OpenclError, CL_COLLECTION_INVALID_ARRAY_SET_ID};

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ArraySetVersion {
    V1,
    V2,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum SetType {
    ArraySet,
    // Set,
    // HashSet,
}

/// ...
///
/// ```c
///
/// // v1
/// __global int array_set__SET_ID[SET_CAPACITY];
///
/// // v2
/// __global int array_set__SET_ID[SET_CAPACITY];
/// __global int array_set_entries__SET_ID[SET_CAPACITY];
///
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SetConfig {
    pub id: usize,
    pub capacity: usize,
    pub set_type: SetType,
}

impl SetConfig {
    pub fn new(id: usize, capacity: usize, set_type: SetType) -> Self {
        Self {
            id,
            capacity,
            set_type,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SetSrc {
    blocks: Vec<SetConfig>,
}

impl SetSrc {
    pub fn new() -> Self {
        Self { blocks: Vec::new() }
    }

    pub fn get_configs(&self) -> &Vec<SetConfig> {
        &self.blocks
    }

    pub fn get_configs_by_type(&self, set_type: SetType) -> Vec<&SetConfig> {
        self.blocks
            .iter()
            .filter(|&x| x.set_type == set_type)
            .collect()
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    pub fn add(&mut self, capacity: usize) {
        let id = self.blocks.len();

        self.blocks
            .push(SetConfig::new(id, capacity, SetType::ArraySet));
    }

    pub fn add_many(&mut self, capacity: usize, quantity: usize) {
        for _ in 0..quantity {
            self.add(capacity);
        }
    }

    pub fn get_config_by_id(&self, id: usize) -> OpenClResult<&SetConfig> {
        match self.blocks.iter().find(|x| x.id == id) {
            None => Err(OpenclError::OpenclCollection(
                CL_COLLECTION_INVALID_ARRAY_SET_ID,
            )),
            Some(c) => Ok(c),
        }
    }
}

impl Default for SetSrc {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_src_add() {
        let mut set_src = SetSrc::new();
        set_src.add(8);
        set_src.add(32);
        set_src.add(8);

        assert_eq!(set_src.len(), 3);
        let ids: Vec<usize> = set_src
            .get_configs_by_type(SetType::ArraySet)
            .iter()
            .map(|x| x.id)
            .collect();
        assert_eq!(ids, vec![0, 1, 2]);
    }

    #[test]
    fn test_set_src_config_get_by_id() {
        let mut set_src = SetSrc::new();
        set_src.add(8);
        set_src.add(16);

        assert_eq!(
            set_src.get_config_by_id(0),
            Ok(&SetConfig::new(0, 8, SetType::ArraySet))
        );
        assert_eq!(
            set_src.get_config_by_id(1),
            Ok(&SetConfig::new(1, 16, SetType::ArraySet))
        );

        assert_eq!(
            set_src.get_config_by_id(5),
            Err(OpenclError::OpenclCollection(
                CL_COLLECTION_INVALID_ARRAY_SET_ID
            ))
        );
    }
}
