pub mod ocl;

pub mod lru_cache;
pub mod map;
pub mod mini_lru_cache;

pub mod serialize;

#[macro_use]
extern crate napi_derive;

#[napi]
pub fn sum(a: i32, b: i32) -> i32 {
  a + b
}
