// https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Map

use crate::serialize::{from_buf_i32_to_vec_u8, serialize_map_entry_i32, MapEntry};
use bincode::config;
use opencl_collections::map::handle::MapHandle;
use opencl_collections::opencl::wrapper::system::System;
use opencl_collections::utils::{from_buf_u8_to_vec_i32, remove_padding_cl_default};
use serde_json::Value;
use std::io;

#[derive(Debug)]
#[napi(object)]
pub struct MetaEntry {
  pub value_len: i32,
  pub map_id: i32,
  pub index: i32,
}

impl MetaEntry {
  pub fn new(value_len: i32, map_id: i32, index: i32) -> Self {
    Self {
      value_len,
      map_id,
      index,
    }
  }

  pub fn create_empty(map_id: i32) -> Self {
    Self::new(0, map_id, 0)
  }
}

#[napi]
pub struct MapCl {
  inner: MapHandle<i32, System>,
}

impl MapCl {
  pub fn new(inner: MapHandle<i32, System>) -> Self {
    Self { inner }
  }

  pub fn _set(&self, key: String, value: Value) -> io::Result<MetaEntry> {
    let key_buf = from_buf_u8_to_vec_i32(key.as_bytes());

    let map_entry: MapEntry = value.into();
    let value_buf = serialize_map_entry_i32(map_entry);

    let pipes = self.inner.get_empty_keys_pipes()?;

    let (entry_index, map_value_len) = self.inner.insert_one(&key_buf, &value_buf, &pipes)?;

    if entry_index < 0 {
      return Ok(MetaEntry::create_empty(self.inner.get_map_id() as i32));
    }

    let meta_entry = MetaEntry::new(map_value_len, self.inner.get_map_id() as i32, entry_index);
    Ok(meta_entry)
  }

  pub fn _get(&self, key: String) -> io::Result<Value> {
    let key_buf = from_buf_u8_to_vec_i32(key.as_bytes());
    let conf = config::standard();

    let (entry_index, map_value_len, value) = self.inner.get_one(&key_buf)?;

    if entry_index < 0 {
      return Ok(Value::Null);
    }

    let meta_entry = MetaEntry::new(map_value_len, self.inner.get_map_id() as i32, entry_index);
    println!("{:?}", meta_entry);

    let value_u8 = from_buf_i32_to_vec_u8(&value);

    let (map_entry, _): (MapEntry, usize) = bincode::decode_from_slice(&value_u8, conf).unwrap();

    let value: Value = map_entry.into();
    Ok(value)
  }

  pub fn _keys(&self) -> io::Result<Vec<String>> {
    let map_entries = self.inner.read_assigned_keys()?;

    let raw_keys: Vec<Vec<i32>> = map_entries
      .into_iter()
      .flat_map(|block| -> Vec<Vec<i32>> {
        block
          .pairs
          .into_iter()
          .map(|pair| remove_padding_cl_default(&pair.key))
          .collect()
      })
      .collect();

    let keys: Vec<String> = raw_keys
      .into_iter()
      .map(|x| String::from_utf8(from_buf_i32_to_vec_u8(&x)).unwrap())
      .collect();

    Ok(keys)
  }
}

#[napi]
impl MapCl {
  #[napi]
  pub fn set_map_id(&mut self, map_id: i32) {
    self.inner.set_map_id(map_id as usize)
  }

  #[napi]
  pub fn set(&self, key: String, value: Value) -> napi::Result<MetaEntry> {
    let r = self._set(key, value)?;
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

  // TODO clear

  // TODO delete

  // TODO entries

  // TODO forEach

  // TODO has

  // TODO values
}
