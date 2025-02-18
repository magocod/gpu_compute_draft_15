use crate::config::ClTypeDefault;
use opencl::opencl_sys::bindings::{cl_int, cl_short, cl_uint};
use std::collections::HashSet;
use std::fmt::Display;
use std::hash::Hash;

pub const BYTE_256: usize = 256;
pub const BYTE_512: usize = 512;

pub const KB: usize = 1024;
pub const KB_8: usize = KB * 8;
pub const MB: usize = KB * KB;

// FIXME conversion functions

pub fn from_buf_u8_to_vec_i16(buf: &[u8]) -> Vec<cl_short> {
    buf.iter().map(|x| *x as cl_short).collect()
}

pub fn from_buf_u8_to_vec_i32(buf: &[u8]) -> Vec<cl_int> {
    buf.iter().map(|x| *x as cl_int).collect()
}

pub fn from_buf_usize_to_vec_i32(buf: &[usize]) -> Vec<cl_int> {
    buf.iter().map(|x| *x as cl_int).collect()
}

pub fn from_buf_usize_to_vec_u32(buf: &[usize]) -> Vec<cl_uint> {
    buf.iter().map(|x| *x as cl_uint).collect()
}

pub fn ensure_vec_size<T: Copy + Default + ClTypeDefault>(buf: &[T], vector_size: usize) -> Vec<T> {
    if buf.len() >= vector_size {
        return buf[0..vector_size].to_vec();
    }

    let mut v: Vec<_> = buf.to_vec();
    v.append(&mut vec![T::cl_default(); vector_size - buf.len()]);
    v
}

pub fn limit_vec_size<T: Copy + Default + ClTypeDefault>(buf: &[T], vector_size: usize) -> &[T] {
    if buf.len() >= vector_size {
        return &buf[0..vector_size];
    }

    buf
}

pub fn remove_padding_cl_default<T: Copy + Default + ClTypeDefault + PartialEq>(
    buf: &[T],
) -> Vec<T> {
    for (i, v) in buf.iter().enumerate().rev() {
        if *v != T::cl_default() {
            return buf[0..=i].to_vec();
        }
    }

    vec![]
}

pub fn is_all_same<T: Copy + PartialEq>(arr: &[T]) -> bool {
    if arr.is_empty() {
        return true;
    }
    let first = arr[0];
    arr.iter().all(|&item| item == first)
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

#[cfg(test)]
mod tests_remove_padding_cl_default_i32 {
    use super::*;

    #[test]
    fn remove_padding_1() {
        let vector = vec![1, 1, 1, 1, -1, -1, -1, -1, -1];
        let v = remove_padding_cl_default(&vector);
        println!("{v:?}");

        assert_eq!(v, vec![1, 1, 1, 1]);
    }

    #[test]
    fn remove_padding_2() {
        let vector = vec![2, 2, 2, 2, 0, 0, 0, 0, 0, 2, -1, -1];
        let v = remove_padding_cl_default(&vector);
        println!("{v:?}");

        assert_eq!(v, vec![2, 2, 2, 2, 0, 0, 0, 0, 0, 2]);
    }

    #[test]
    fn remove_padding_3() {
        let vector = vec![3; 16];
        let v = remove_padding_cl_default(&vector);
        println!("{v:?}");

        assert_eq!(v, vec![3; 16]);
    }

    #[test]
    fn remove_padding_4() {
        let vector = vec![0, 0, 0, 4, 4, 4, 4, -1, -1, -1];
        let v = remove_padding_cl_default(&vector);
        println!("{v:?}");

        assert_eq!(v, vec![0, 0, 0, 4, 4, 4, 4]);
    }

    #[test]
    fn remove_padding_5() {
        let vector = vec![-1, -1, -1, -1, -1, -1, -1, -1, -1];
        let v = remove_padding_cl_default(&vector);
        let empty: Vec<i32> = vec![];
        println!("{v:?}");

        assert_eq!(v, empty);
    }

    #[test]
    fn remove_padding_6() {
        let vector = vec![-1, -1, -1, -1, 0, -1, -1, -1, -1];
        let v = remove_padding_cl_default(&vector);
        println!("{v:?}");

        assert_eq!(v, vec![-1, -1, -1, -1, 0]);
    }

    #[test]
    fn remove_padding_7() {
        let vector = vec![-1, 1, -1, -1, -1, -1, 2, -1];
        let v = remove_padding_cl_default(&vector);
        println!("{v:?}");

        assert_eq!(v, vec![-1, 1, -1, -1, -1, -1, 2]);
    }
}

#[cfg(test)]
mod tests_remove_padding_cl_default_u8 {
    use super::*;

    #[test]
    fn remove_padding_1() {
        let vector: Vec<u8> = vec![1, 1, 1, 1, 0, 0, 0, 0, 0];
        let v = remove_padding_cl_default(&vector);
        println!("{v:?}");

        assert_eq!(v, vec![1, 1, 1, 1]);
    }

    #[test]
    fn remove_padding_2() {
        let vector: Vec<u8> = vec![2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 0, 0];
        let v = remove_padding_cl_default(&vector);
        println!("{v:?}");

        assert_eq!(v, vec![2, 2, 2, 2, 0, 0, 0, 0, 0, 2]);
    }

    #[test]
    fn remove_padding_3() {
        let vector: Vec<u8> = vec![3; 16];
        let v = remove_padding_cl_default(&vector);
        println!("{v:?}");

        assert_eq!(v, vec![3; 16]);
    }

    #[test]
    fn remove_padding_4() {
        let vector: Vec<u8> = vec![0, 0, 0, 4, 4, 4, 4, 0, 0, 0];
        let v = remove_padding_cl_default(&vector);
        println!("{v:?}");

        assert_eq!(v, vec![0, 0, 0, 4, 4, 4, 4]);
    }

    #[test]
    fn remove_padding_5() {
        let vector: Vec<u8> = vec![0, 0, 0, 0, 0, 0, 0, 0, 0];
        let v = remove_padding_cl_default(&vector);
        let empty: Vec<u8> = vec![];
        println!("{v:?}");

        assert_eq!(v, empty);
    }
}

#[cfg(test)]
mod tests_is_all_same {
    use super::*;

    #[test]
    fn vec_is_empty() {
        let vector: Vec<i32> = vec![];
        let b = is_all_same(&vector);
        assert!(b);
    }

    #[test]
    fn vec_all_elements_are_equal() {
        let vector = vec![1, 1, 1, 1];
        let b = is_all_same(&vector);
        assert!(b);
    }

    #[test]
    fn vector_all_elements_are_different() {
        let vector = vec![1, 2, 3, 4];
        let b = !is_all_same(&vector);
        assert!(b);
    }
}
