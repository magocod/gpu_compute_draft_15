use crate::config::ClTypeTrait;
use num::{FromPrimitive, Zero};
use std::ops::{Add, Div};

#[derive(Debug, PartialEq, Clone)]
pub struct TestMatrix<T: Clone> {
    pub keys: Vec<Vec<T>>,
    pub values: Vec<Vec<T>>,
    pub indices: Vec<usize>,
}

impl<T> TestMatrix<T>
where
    T: Add<Output = T>
        + ClTypeTrait
        + Zero
        + Add<T, Output = T>
        + Div<T, Output = T>
        + FromPrimitive,
{
    pub fn new(
        matrix_len: usize,
        key_len: usize,
        value_len: usize,
        fill_key_with: T,
        fill_vec_with: T,
    ) -> Self {
        let mut keys: Vec<Vec<T>> = Vec::with_capacity(key_len * matrix_len);
        let mut values: Vec<Vec<T>> = Vec::with_capacity(value_len * matrix_len);
        let indices: Vec<usize> = (0..matrix_len).collect();

        for i in 0..matrix_len {
            let v: T = FromPrimitive::from_usize(i).unwrap();
            keys.push(vec![v + fill_key_with; key_len]);
            values.push(vec![v + fill_vec_with; value_len]);
        }

        TestMatrix {
            keys,
            values,
            indices,
        }
    }

    pub fn append(&mut self, other: &mut TestMatrix<T>) {
        self.keys.append(&mut other.keys);
        self.values.append(&mut other.values);

        self.indices = (0..self.keys.len()).collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencl::opencl_sys::bindings::{cl_int, cl_uchar};

    #[test]
    fn case_u8() {
        let mut keys: Vec<Vec<cl_uchar>> = Vec::with_capacity(32 * 32);
        let mut values: Vec<Vec<cl_uchar>> = Vec::with_capacity(32 * 32);
        let indices: Vec<usize> = (0..32).collect();

        for i in 0..32 {
            keys.push(vec![i as cl_uchar + 1; 32]);
            values.push(vec![i as cl_uchar + 10; 32]);
        }

        let test_matrix: TestMatrix<u8> = TestMatrix::new(32, 32, 32, 1, 10);

        println!("key gen u8");
        for v in test_matrix.keys.iter().enumerate() {
            println!("{v:?}");
        }

        println!("value gen u8");
        for v in test_matrix.values.iter().enumerate() {
            println!("{v:?}");
        }

        assert_eq!(test_matrix.keys, keys);
        assert_eq!(test_matrix.values, values);
        assert_eq!(test_matrix.indices, indices);
    }

    #[test]
    fn case_i32() {
        let mut keys: Vec<Vec<cl_int>> = Vec::with_capacity(32 * 32);
        let mut values: Vec<Vec<cl_int>> = Vec::with_capacity(32 * 32);
        let indices: Vec<usize> = (0..32).collect();

        for i in 0..32 {
            keys.push(vec![i as cl_int + 1; 32]);
            values.push(vec![i as cl_int + 1000; 32]);
        }

        let test_matrix: TestMatrix<i32> = TestMatrix::new(32, 32, 32, 1, 1000);

        println!("key gen i32");
        for v in test_matrix.keys.iter().enumerate() {
            println!("{v:?}");
        }

        println!("value gen i32");
        for v in test_matrix.values.iter().enumerate() {
            println!("{v:?}");
        }

        assert_eq!(test_matrix.keys, keys);
        assert_eq!(test_matrix.values, values);
        assert_eq!(test_matrix.indices, indices);
    }
}
