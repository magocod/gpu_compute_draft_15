use crate::config::ClTypeTrait;
use crate::error::OpenClResult;
use crate::map::handle::read::map_get_empty_key::PipeIndices;
use crate::map::handle::{EntryIndex, MapBlockSize, MapHandle};
use opencl::wrapper::system::OpenclCommonOperation;

impl<T: ClTypeTrait, D: OpenclCommonOperation> MapHandle<T, D> {
    pub fn add_one(
        &self,
        key: &[T],
        value: &[T],
        pipes: &[PipeIndices],
    ) -> OpenClResult<(EntryIndex, MapBlockSize)> {
        let (mut indices, mut blocks) =
            self.map_add(&vec![key.to_vec()], &vec![value.to_vec()], pipes)?;
        Ok((indices.pop().unwrap(), blocks.pop().unwrap()))
    }

    pub fn insert_one(
        &self,
        key: &[T],
        value: &[T],
        pipes: &[PipeIndices],
    ) -> OpenClResult<(EntryIndex, MapBlockSize)> {
        let (mut indices, mut blocks) =
            self.map_insert(&vec![key.to_vec()], &vec![value.to_vec()], pipes)?;
        Ok((indices.pop().unwrap(), blocks.pop().unwrap()))
    }
}
