use crate::serialize::{from_buf_i32_to_vec_u8, serialize_map_entry_i32, MapEntry};
use bincode::config;
use opencl_collections::cache::handle::lru::LRUCacheHandle;
use opencl_collections::opencl::wrapper::system::System;
use opencl_collections::utils::{from_buf_u8_to_vec_i32, remove_padding_cl_default};
use serde_json::Value;
use std::io;

#[derive(Debug)]
#[napi(object)]
pub struct Payload {
  pub key: String,
  pub value: Value,
}

#[napi]
pub struct LruCache {
  inner: LRUCacheHandle<System>,
}

impl LruCache {
  pub fn new(inner: LRUCacheHandle<System>) -> Self {
    Self { inner }
  }

  pub fn _set(&self, key: String, value: Value) -> io::Result<i32> {
    let key_buf = from_buf_u8_to_vec_i32(key.as_bytes());

    let map_entry: MapEntry = value.into();
    let value_buf = serialize_map_entry_i32(map_entry);

    let mut indices = self.inner.insert(&vec![key_buf], &vec![value_buf])?;

    Ok(indices.pop().unwrap())
  }

  pub fn _set_many(&self, input: Vec<Payload>) -> io::Result<Vec<i32>> {
    let mut keys = Vec::with_capacity(input.len());
    let mut values = Vec::with_capacity(input.len());

    for payload in input {
      let key_buf = from_buf_u8_to_vec_i32(payload.key.as_bytes());
      let map_entry: MapEntry = payload.value.into();
      let value_buf = serialize_map_entry_i32(map_entry);

      keys.push(key_buf);
      values.push(value_buf);
    }

    let indices = self.inner.insert(&keys, &values)?;
    Ok(indices)
  }

  pub fn _get(&self, key: String) -> io::Result<Value> {
    let key_buf = from_buf_u8_to_vec_i32(key.as_bytes());
    let conf = config::standard();

    let (mut values, mut indices) = self.inner.get(&vec![key_buf])?;
    let entry_index = indices.pop().unwrap();

    if entry_index < 0 {
      return Ok(Value::Null);
    }

    let value_buf = values.pop().unwrap();

    let value_u8 = from_buf_i32_to_vec_u8(&value_buf);

    let (map_entry, _): (MapEntry, usize) = bincode::decode_from_slice(&value_u8, conf).unwrap();
    // println!("{:?}", map_entry);

    let value: Value = map_entry.into();
    Ok(value)
  }

  pub fn _debug(&self) -> io::Result<()> {
    let _cs = self.inner.print()?;
    Ok(())
  }

  pub fn _keys(&self) -> io::Result<Vec<String>> {
    // let keys_raw = self.inner.sorted_keys()?;
    let keys_raw = self.inner.get_sorted_keys()?;

    let keys: Vec<_> = keys_raw
      .into_iter()
      .filter_map(|x| -> Option<String> {
        let v = remove_padding_cl_default(&x.key);

        if v.is_empty() {
          return None;
        }

        let vec_u8 = from_buf_i32_to_vec_u8(&v);
        Some(String::from_utf8(vec_u8).unwrap())
      })
      .collect();

    Ok(keys)
  }
}

#[napi]
impl LruCache {
  #[napi]
  pub fn set(&self, key: String, value: Value) -> napi::Result<i32> {
    let r = self._set(key, value)?;
    Ok(r)
  }

  #[napi]
  pub fn set_many(&self, input: Vec<Payload>) -> napi::Result<Vec<i32>> {
    let r = self._set_many(input)?;
    Ok(r)
  }

  #[napi]
  pub fn get(&self, key: String) -> napi::Result<Value> {
    let r = self._get(key)?;
    Ok(r)
  }

  #[napi]
  pub fn keys(&self) -> napi::Result<Vec<String>> {
    let r = self._keys()?;
    Ok(r)
  }

  #[napi]
  pub fn debug(&self) -> napi::Result<()> {
    self._debug()?;
    Ok(())
  }
}
