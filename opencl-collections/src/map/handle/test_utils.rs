use crate::config::{ClTypeTrait, DEFAULT_DEVICE_INDEX};
use crate::map::config::{MapSrc, DEFAULT_MAP_KEY_LENGTH};
use crate::map::handle::{Handle, MapHandle, Pair};
use crate::test_utils::TestMatrix;
use crate::utils::ensure_vec_size;
use opencl::wrapper::system::{OpenclCommonOperation, System};
use std::sync::Arc;

pub fn generate_arc_opencl_block<T: ClTypeTrait>(
    map_src: &MapSrc<T>,
    initialize_all_maps: bool,
) -> Arc<System> {
    let system = System::new(DEFAULT_DEVICE_INDEX, &map_src.build()).unwrap();
    let arc_system = Arc::new(system);

    if initialize_all_maps {
        let h = Handle::new(map_src, arc_system.clone());
        h.initialize_all_maps().unwrap();
    }

    arc_system
}

pub fn generate_arc_opencl_block_default<T: ClTypeTrait>(map_src: &MapSrc<T>) -> Arc<System> {
    generate_arc_opencl_block(map_src, true)
}

pub enum DefaultTypeTrait {
    Std,    // Default
    Custom, //  ClTypeDefault
}

pub fn assert_map_block_is_empty<T: ClTypeTrait, D: OpenclCommonOperation>(
    map: &MapHandle<T, D>,
    map_value_len: usize,
    default_type_trait: DefaultTypeTrait,
) {
    let pairs = map.read(map_value_len).unwrap();

    match default_type_trait {
        DefaultTypeTrait::Std => {
            for pair in pairs.into_iter() {
                assert_eq!(pair.key, vec![T::default(); DEFAULT_MAP_KEY_LENGTH]);
                assert_eq!(pair.value, vec![T::default(); map_value_len]);
            }
        }
        DefaultTypeTrait::Custom => {
            for pair in pairs.into_iter() {
                assert_eq!(pair.key, vec![T::cl_default(); DEFAULT_MAP_KEY_LENGTH]);
                assert_eq!(pair.value, vec![T::cl_default(); map_value_len]);
            }
        }
    }
}

pub fn assert_map_block_is_equal_to_test_matrix<T: ClTypeTrait, D: OpenclCommonOperation>(
    map: &MapHandle<T, D>,
    map_value_len: usize,
    test_matrix: &TestMatrix<T>,
) {
    let pairs = map.read(map_value_len).unwrap();

    for (i, pair) in pairs.into_iter().enumerate() {
        assert_eq!(
            pair.key,
            ensure_vec_size(&test_matrix.keys[i], DEFAULT_MAP_KEY_LENGTH)
        );
        assert_eq!(
            pair.value,
            ensure_vec_size(&test_matrix.values[i], map_value_len)
        );
    }
}

pub fn assert_pair_is_empty<T: ClTypeTrait>(pair: &Pair<T>, map_value_len: usize) {
    assert_eq!(pair.key, vec![T::cl_default(); DEFAULT_MAP_KEY_LENGTH]);
    assert_eq!(pair.value, vec![T::cl_default(); map_value_len]);
}

pub fn assert_pair_is_not_empty<T: ClTypeTrait>(pair: &Pair<T>) {
    assert!(pair.key.iter().any(|&x| x != T::cl_default()));
    assert!(pair.value.iter().any(|&x| x != T::cl_default()));
}

impl<T: ClTypeTrait> TestMatrix<T> {
    pub fn put<D: OpenclCommonOperation>(&self, map: &MapHandle<T, D>, map_value_len: usize) {
        map.put(map_value_len, &self.keys, &self.values).unwrap()
    }
}
