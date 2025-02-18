use bincode::{config, Decode, Encode};
use opencl_collections::utils::{from_buf_u8_to_vec_i16, from_buf_u8_to_vec_i32};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Encode, Decode, Serialize, Deserialize, PartialEq, Debug)]
pub enum EntryType {
  Null,
  Bool,
  Number,
  String,
}

impl From<&Value> for EntryType {
  fn from(value: &Value) -> Self {
    match value {
      Value::Null => Self::Null,
      Value::Bool(_) => Self::Bool,
      Value::Number(_) => Self::Number,
      Value::String(_) => Self::String,
      // TODO complete
      Value::Array(_) => todo!(),
      Value::Object(_) => todo!(),
    }
  }
}

#[derive(Encode, Decode, Serialize, Deserialize, PartialEq, Debug)]
pub enum MapEntryType {
  Null,
  Bool,
  Number,
  String,
  Array(EntryType),
  Object(EntryType),
}

#[derive(Encode, Decode, Serialize, Deserialize, PartialEq, Debug)]
pub struct MapEntry {
  pub entry_type: MapEntryType,
  pub value: Vec<u8>,
}

impl MapEntry {
  pub fn new(entry_type: MapEntryType, value: Vec<u8>) -> Self {
    Self { entry_type, value }
  }

  pub fn set_type(&mut self, entry_type: MapEntryType) {
    self.entry_type = entry_type;
  }

  pub fn set_value(&mut self, value: Vec<u8>) {
    self.value = value;
  }
}

impl From<Value> for MapEntry {
  fn from(value: Value) -> Self {
    let conf = config::standard();

    match value {
      Value::Null => Self::new(MapEntryType::Null, vec![]),
      Value::Bool(v) => Self::new(
        MapEntryType::Bool,
        bincode::serde::encode_to_vec(v, conf).unwrap(),
      ),
      Value::Number(v) => {
        let v_i64 = v.as_i64().unwrap();
        Self::new(
          MapEntryType::Number,
          bincode::serde::encode_to_vec(v_i64, conf).unwrap(),
        )
      }
      Value::String(v) => Self::new(
        MapEntryType::String,
        bincode::serde::encode_to_vec(v, conf).unwrap(),
      ),
      Value::Array(v) => {
        let vec_type: EntryType = match v.first() {
          None => {
            panic!("todo handle error Value::Array empty")
          }
          Some(val) => val.into(),
        };

        match vec_type {
          EntryType::Bool => {
            let arr: Vec<bool> = v.iter().filter_map(|x| x.as_bool()).collect();

            Self::new(
              MapEntryType::Array(vec_type),
              bincode::serde::encode_to_vec(arr, conf).unwrap(),
            )
          }
          EntryType::Number => {
            let arr: Vec<i64> = v.iter().filter_map(|x| x.as_i64()).collect();

            Self::new(
              MapEntryType::Array(vec_type),
              bincode::serde::encode_to_vec(arr, conf).unwrap(),
            )
          }
          EntryType::String => {
            let arr: Vec<String> = v
              .iter()
              .filter_map(|x| -> Option<String> { x.as_str().map(|s| s.to_string()) })
              .collect();

            Self::new(
              MapEntryType::Array(vec_type),
              bincode::serde::encode_to_vec(arr, conf).unwrap(),
            )
          }
          _ => {
            panic!("todo handle error Value::Array - type")
          }
        }
      }
      Value::Object(_) => panic!("todo handle error Value::Object"),
    }
  }
}

impl From<MapEntry> for Value {
  fn from(map_entry: MapEntry) -> Self {
    let conf = config::standard();

    match map_entry.entry_type {
      MapEntryType::Null => Value::Null,
      MapEntryType::Bool => {
        let (decoded, _): (bool, usize) =
          bincode::decode_from_slice(&map_entry.value, conf).unwrap();
        Value::Bool(decoded)
      }
      MapEntryType::Number => {
        let (decoded, _): (i64, usize) =
          bincode::decode_from_slice(&map_entry.value, conf).unwrap();
        Value::Number(decoded.into())
      }
      MapEntryType::String => {
        let (decoded, _): (String, usize) =
          bincode::decode_from_slice(&map_entry.value, conf).unwrap();
        Value::String(decoded)
      }
      MapEntryType::Array(vec_type) => {
        let v = match vec_type {
          EntryType::Bool => {
            let (decoded, _): (Vec<bool>, usize) =
              bincode::decode_from_slice(&map_entry.value, conf).unwrap();
            let values: Vec<Value> = decoded.into_iter().map(|x| x.into()).collect();
            values
          }
          EntryType::Number => {
            let (decoded, _): (Vec<i64>, usize) =
              bincode::decode_from_slice(&map_entry.value, conf).unwrap();
            let values: Vec<Value> = decoded.into_iter().map(|x| x.into()).collect();
            values
          }
          EntryType::String => {
            let (decoded, _): (Vec<String>, usize) =
              bincode::decode_from_slice(&map_entry.value, conf).unwrap();
            let values: Vec<Value> = decoded.into_iter().map(|x| x.into()).collect();
            values
          }
          _ => {
            panic!("invalid get array value");
          }
        };

        Value::Array(v)
      }
      MapEntryType::Object(_) => {
        panic!("todo handle error Value::Object")
      }
    }
  }
}

pub fn serialize_map_entry_i16(value: MapEntry) -> Vec<i16> {
  let encoded = bincode::serde::encode_to_vec(value, config::standard()).unwrap();
  from_buf_u8_to_vec_i16(&encoded)
}

pub fn serialize_map_entry_i32(value: MapEntry) -> Vec<i32> {
  let encoded = bincode::serde::encode_to_vec(value, config::standard()).unwrap();
  from_buf_u8_to_vec_i32(&encoded)
}

// FIXME type conversions

pub fn from_buf_i16_to_vec_u8(buf: &[i16]) -> Vec<u8> {
  buf.iter().map(|x| *x as u8).collect()
}

pub fn from_buf_i32_to_vec_u8(buf: &[i32]) -> Vec<u8> {
  buf.iter().map(|x| *x as u8).collect()
}
