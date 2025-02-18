// ...

use crate::error::{OpenClResult, OpenclError, CL_COLLECTION_INVALID_QUEUE_ID};

#[derive(Debug, Copy, Clone)]
pub enum PriorityQueueType {
    Ordered,
    // UnOrdered,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum QueueType {
    Lineal,
    Priority,
    Circular,
}

/// ...
///
/// linear queue
/// ```c
/// __global int lq__QUEUE_ID[QUEUE_CAPACITY];
/// __global int lq_front__QUEUE_ID = -1;
/// __global int lq_rear__QUEUE_ID = -1;
/// ```
///
/// priority queue
/// ```c
/// __global int pq_tmp_index__QUEUE_ID = -1;
/// __global int pq_last_index__QUEUE_ID = -1;
/// __global int pq_value__QUEUE_ID[QUEUE_CAPACITY];
/// __global int pq_priority__QUEUE_ID[QUEUE_CAPACITY];
///
/// __global struct PqValue tmp_pq_indices__QUEUE_ID[QUEUE_CAPACITY];
/// ```
///
/// circular queue
/// ```c
/// __global int cq__QUEUE_ID[QUEUE_CAPACITY];
/// __global int cq_front__QUEUE_ID = -1;
/// __global int cq_rear__QUEUE_ID = -1;
///
/// __global int cq_operation_index__QUEUE_ID = -1;
/// __global int cq_max_operations__QUEUE_ID = -1;
/// __global int cq_operations__QUEUE_ID[QUEUE_CAPACITY];
/// ```
///
#[derive(Debug, Clone, PartialEq)]
pub struct QueueConfig {
    pub id: usize,
    pub capacity: usize,
    pub queue_type: QueueType,
}

impl QueueConfig {
    pub fn new(id: usize, capacity: usize, queue_type: QueueType) -> Self {
        Self {
            id,
            capacity,
            queue_type,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QueueSrc {
    blocks: Vec<QueueConfig>,
}

impl QueueSrc {
    pub fn new() -> Self {
        Self { blocks: Vec::new() }
    }

    pub fn get_configs(&self) -> &Vec<QueueConfig> {
        &self.blocks
    }

    pub fn get_configs_by_type(&self, queue_type: QueueType) -> Vec<&QueueConfig> {
        self.blocks
            .iter()
            .filter(|&x| x.queue_type == queue_type)
            .collect()
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    fn get_queue_id(&mut self, queue_type: QueueType) -> usize {
        self.get_configs_by_type(queue_type).len()
    }

    pub fn add(&mut self, capacity: usize, queue_type: QueueType) {
        let id = self.get_queue_id(queue_type);

        self.blocks.push(QueueConfig::new(id, capacity, queue_type));
    }

    // lq = linear queue
    pub fn add_lq(&mut self, capacity: usize) {
        self.add(capacity, QueueType::Lineal)
    }

    // pq = priority queue
    pub fn add_pq(&mut self, capacity: usize) {
        self.add(capacity, QueueType::Priority)
    }

    // cq = circular queue
    pub fn add_cq(&mut self, capacity: usize) {
        self.add(capacity, QueueType::Circular)
    }

    pub fn add_many(&mut self, capacity: usize, quantity: usize, queue_type: QueueType) {
        for _ in 0..quantity {
            self.add(capacity, queue_type);
        }
    }

    // lq = linear queue
    pub fn add_many_lq(&mut self, capacity: usize, quantity: usize) {
        self.add_many(capacity, quantity, QueueType::Lineal)
    }

    // pq = priority queue
    pub fn add_many_pq(&mut self, capacity: usize, quantity: usize) {
        self.add_many(capacity, quantity, QueueType::Priority)
    }

    // cq = circular queue
    pub fn add_many_cq(&mut self, capacity: usize, quantity: usize) {
        self.add_many(capacity, quantity, QueueType::Circular)
    }

    pub fn get_config_by_id(&self, id: usize, queue_type: QueueType) -> OpenClResult<&QueueConfig> {
        match self
            .blocks
            .iter()
            .filter(|x| x.queue_type == queue_type)
            .find(|x| x.id == id)
        {
            None => Err(OpenclError::OpenclCollection(
                CL_COLLECTION_INVALID_QUEUE_ID,
            )),
            Some(c) => Ok(c),
        }
    }

    pub fn get_lq_config_by_id(&self, id: usize) -> OpenClResult<&QueueConfig> {
        self.get_config_by_id(id, QueueType::Lineal)
    }

    pub fn get_pq_config_by_id(&self, id: usize) -> OpenClResult<&QueueConfig> {
        self.get_config_by_id(id, QueueType::Priority)
    }

    pub fn get_cq_config_by_id(&self, id: usize) -> OpenClResult<&QueueConfig> {
        self.get_config_by_id(id, QueueType::Circular)
    }

    // TODO get_config_by_capacity
}

impl Default for QueueSrc {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_src_add() {
        let queue_type = QueueType::Lineal;

        let mut queue_src = QueueSrc::new();
        queue_src.add(8, queue_type);
        queue_src.add(32, queue_type);
        queue_src.add(8, queue_type);

        assert_eq!(queue_src.len(), 3);
        let ids: Vec<usize> = queue_src.get_configs().iter().map(|x| x.id).collect();
        assert_eq!(ids, vec![0, 1, 2]);
    }

    #[test]
    fn test_queue_src_add_many() {
        let queue_type = QueueType::Lineal;

        let mut queue_src = QueueSrc::new();
        queue_src.add_many(8, 3, queue_type);

        assert_eq!(queue_src.len(), 3);
        let ids: Vec<usize> = queue_src.get_configs().iter().map(|x| x.id).collect();
        assert_eq!(ids, vec![0, 1, 2]);
    }

    #[test]
    fn test_queue_src_get_config_by_id() {
        let mut queue_src = QueueSrc::new();
        queue_src.add(8, QueueType::Lineal);
        queue_src.add(16, QueueType::Priority);

        assert_eq!(
            queue_src.get_lq_config_by_id(0),
            Ok(&QueueConfig::new(0, 8, QueueType::Lineal))
        );
        assert_eq!(
            queue_src.get_pq_config_by_id(0),
            Ok(&QueueConfig::new(0, 16, QueueType::Priority))
        );

        assert_eq!(
            queue_src.get_lq_config_by_id(5),
            Err(OpenclError::OpenclCollection(
                CL_COLLECTION_INVALID_QUEUE_ID
            ))
        );
        assert_eq!(
            queue_src.get_cq_config_by_id(0),
            Err(OpenclError::OpenclCollection(
                CL_COLLECTION_INVALID_QUEUE_ID
            ))
        );
    }
}
