use std::collections::HashSet;
use std::fmt::Display;
use std::hash::Hash;
use std::string::FromUtf8Error;

/// simple conversion of i8 buffer to string (rust), removing 0 and negative values
pub fn buf_i8_to_string(buf: &[i8]) -> Result<String, FromUtf8Error> {
    let vec_u8: Vec<u8> = buf
        .iter()
        .filter_map(|&x| {
            if x <= 0 {
                return None;
            }
            Some(x as u8)
        })
        .collect();

    String::from_utf8(vec_u8)
}

pub fn buf_u8_remove_zero_to_string(buf: &[u8]) -> Result<String, FromUtf8Error> {
    let vec = buf
        .iter()
        .filter_map(|&x| {
            if x == 0 {
                return None;
            }
            Some(x)
        })
        .collect();

    String::from_utf8(vec)
}

// FIXME improve function
// pub fn has_unique_elements<T: Eq + Hash + Copy + Display>(buf: &[T]) -> bool {
//     let mut set: HashSet<T> = HashSet::new();
//     for x in buf.iter() {
//         if !set.insert(*x) {
//             println!("v: {}", x);
//             return false;
//         }
//     }
//     true
// }

pub fn has_unique_elements<T: Eq + Hash + Copy + Display>(buf: &[T]) -> bool {
    let mut set: HashSet<T> = HashSet::new();

    let mut unique = true;

    for x in buf.iter() {
        if !set.insert(*x) {
            println!("v: {}", x);
            unique = false;
        }
    }

    unique
}

// TODO test buf_i8_to_string

// TODO test buf_u8_remove_zero_to_string

// TODO test has_unique_elements
