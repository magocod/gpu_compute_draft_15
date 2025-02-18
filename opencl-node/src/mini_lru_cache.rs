use opencl_collections::cache::handle::mini_lru::MiniLRUCacheHandle;
use opencl_collections::opencl::wrapper::system::System;
use std::io;

#[napi]
pub struct MiniLruCache {
  inner: MiniLRUCacheHandle<System>,
}

impl MiniLruCache {
  pub fn new(inner: MiniLRUCacheHandle<System>) -> Self {
    Self { inner }
  }

  pub fn _set(&self, key: i32, value: i32) -> io::Result<i32> {
    let mut r = self.inner.insert(&[key], &[value])?;
    Ok(r.pop().unwrap())
  }

  pub fn _get(&self, key: i32) -> io::Result<i32> {
    let (mut r, _) = self.inner.get(&[key])?;
    Ok(r.pop().unwrap())
  }

  pub fn _debug(&self) -> io::Result<()> {
    let cs = self.inner.print()?;
    cs.print_all_key();
    Ok(())
  }

  pub fn _keys(&self) -> io::Result<Vec<i32>> {
    // let keys_raw = self.inner.sorted_keys()?;
    let keys_raw = self.inner.get_sorted_keys()?;
    let keys: Vec<_> = keys_raw.into_iter().map(|x| x.key).collect();
    Ok(keys)
  }
}

#[napi]
impl MiniLruCache {
  #[napi]
  pub fn set(&self, key: i32, value: i32) -> napi::Result<i32> {
    let r = self._set(key, value)?;
    Ok(r)
  }

  #[napi]
  pub fn get(&self, key: i32) -> napi::Result<i32> {
    let r = self._get(key)?;
    Ok(r)
  }

  #[napi]
  pub fn keys(&self) -> napi::Result<Vec<i32>> {
    let r = self._keys()?;
    Ok(r)
  }

  #[napi]
  pub fn debug(&self) -> napi::Result<()> {
    self._debug()?;
    Ok(())
  }
}
