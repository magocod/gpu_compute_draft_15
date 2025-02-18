use crate::error::{OpenClResult, OpenclError, CL_COLLECTION_INVALID_STACK_ID};

/// ...
///
/// ```c
/// __global int stack__STACK_ID[STACK_CAPACITY];
/// __global int st_top__STACK_ID = -1;
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct StackConfig {
    pub id: usize,
    pub capacity: usize,
}

#[derive(Debug, Clone)]
pub struct StackSrc {
    blocks: Vec<StackConfig>,
}

impl StackSrc {
    pub fn new() -> Self {
        Self { blocks: Vec::new() }
    }

    pub fn get_configs(&self) -> &Vec<StackConfig> {
        &self.blocks
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    pub fn add(&mut self, capacity: usize) {
        let id = self.blocks.len();

        self.blocks.push(StackConfig { id, capacity });
    }

    pub fn add_many(&mut self, capacity: usize, quantity: usize) {
        for _ in 0..quantity {
            self.add(capacity);
        }
    }

    pub fn get_config_by_id(&self, id: usize) -> OpenClResult<&StackConfig> {
        match self.blocks.iter().find(|x| x.id == id) {
            None => Err(OpenclError::OpenclCollection(
                CL_COLLECTION_INVALID_STACK_ID,
            )),
            Some(c) => Ok(c),
        }
    }
}

impl Default for StackSrc {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack_src_add() {
        let mut stack_src = StackSrc::new();
        stack_src.add(8);
        stack_src.add(32);
        stack_src.add(8);

        assert_eq!(stack_src.len(), 3);
        let ids: Vec<usize> = stack_src.get_configs().iter().map(|x| x.id).collect();
        assert_eq!(ids, vec![0, 1, 2]);
    }

    #[test]
    fn test_stack_src_get_config_by_id() {
        let mut stack_src = StackSrc::new();
        stack_src.add(8);
        stack_src.add(16);

        assert_eq!(
            stack_src.get_config_by_id(0),
            Ok(&StackConfig { id: 0, capacity: 8 })
        );
        assert_eq!(
            stack_src.get_config_by_id(1),
            Ok(&StackConfig {
                id: 1,
                capacity: 16,
            })
        );

        assert_eq!(
            stack_src.get_config_by_id(5),
            Err(OpenclError::OpenclCollection(
                CL_COLLECTION_INVALID_STACK_ID
            ))
        );
    }
}
