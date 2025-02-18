use crate::config::ClTypeTrait;
use crate::error::OpenClResult;
use crate::map::handle::{EntryIndex, MapBlockSize, MapHandle};
use opencl::wrapper::system::OpenclCommonOperation;

impl<T: ClTypeTrait, D: OpenclCommonOperation> MapHandle<T, D> {
    pub fn get_one(&self, key: &[T]) -> OpenClResult<(EntryIndex, MapBlockSize, Vec<T>)> {
        let (mut indices, mut blocks, mut values) = self.map_get(&vec![key.to_vec()])?;
        Ok((
            indices.pop().unwrap(),
            blocks.pop().unwrap(),
            values.pop().unwrap(),
        ))
    }
}
