// kernel names

use crate::map::config::get_map_block_name;

pub fn get_map_kernel_name(kernel_name: &str, value_len: usize) -> String {
    kernel_name.replace("BLOCK_NAME", &get_map_block_name(value_len))
}

// write

// src/kernel/map_put.rs
pub const MAP_PUT_ONE: &str = "map_put_one__BLOCK_NAME";
pub const MAP_PUT: &str = "map_put__BLOCK_NAME";
pub const MAP_PUT_WITH_CMQ: &str = "map_put_with_cmq__BLOCK_NAME";
pub const MAP_PUT_WITH_INDEX: &str = "map_put_with_index__BLOCK_NAME";
pub const MAP_PUT_WITH_PIPE_AND_CMQ: &str = "map_put_with_pipe_and_cmq__BLOCK_NAME";

// src/kernel/map_copy.rs
pub const MAP_COPY_VALUE: &str = "map_copy_value_for__BLOCK_NAME";

// src/kernel/map_add.rs
pub const MAP_ADD: &str = "map_add";

// src/kernel/map_insert.rs
pub const MAP_INSERT: &str = "map_insert";
pub const GET_TMP_FOR_MAP_INSERT: &str = "get_tmp_for_map_insert";

// src/kernel/map_append_for_block.rs
pub const MAP_APPEND_FOR_BLOCK: &str = "map_append_for_block__BLOCK_NAME";

// src/kernel/map_append.rs
pub const MAP_APPEND: &str = "map_append";
pub const GET_TMP_FOR_MAP_APPEND: &str = "get_tmp_for_map_append";

// src/kernel/map_remove.rs
// src/kernel/map_remove_v2.rs
pub const MAP_REMOVE: &str = "map_remove";
pub const GET_TMP_FOR_MAP_REMOVE: &str = "get_tmp_for_map_remove";

// read

// src/kernel/map_read.rs
pub const MAP_READ_ONE: &str = "map_read_one__BLOCK_NAME";
pub const MAP_READ_KEYS_FOR_BLOCK: &str = "map_read_keys__BLOCK_NAME";
pub const MAP_READ: &str = "map_read__BLOCK_NAME";
pub const MAP_READ_WITH_CMQ: &str = "map_read_with_cmq__BLOCK_NAME";
pub const MAP_READ_WITH_INDEX: &str = "map_read_with_index__BLOCK_NAME";
pub const MAP_READ_WITH_INDEX_AND_CMQ: &str = "map_read_with_index_and_cmq__BLOCK_NAME";
pub const MAP_READ_KEYS: &str = "map_read_keys";

// src/kernel/map_read_assigned_keys.rs
pub const MAP_READ_ASSIGNED_KEYS: &str = "map_read_assigned_keys";

// src/kernel/map_read_sizes.rs
pub const MAP_READ_SIZES_FOR_BLOCK: &str = "map_read_sizes_for_block__BLOCK_NAME";
pub const MAP_READ_SIZES: &str = "map_read_sizes";

// src/kernel/map_get_index.rs
pub const MAP_GET_INDEX: &str = "map_get_index";
pub const GET_TMP_FOR_MAP_GET_INDEX: &str = "get_tmp_for_map_get_index";

// src/kernel/map_get.rs
pub const MAP_GET: &str = "map_get";
pub const GET_TMP_FOR_MAP_GET: &str = "get_tmp_for_map_get";

// src/kernel/map_get_empty_keys.rs
pub const MAP_GET_EMPTY_KEYS_FOR_BLOCK: &str = "map_get_empty_keys_for_block__BLOCK_NAME";
pub const MAP_GET_EMPTY_KEYS: &str = "map_get_empty_keys";

// other

// src/kernel/map_deduplication.rs
pub const MAP_DEDUPLICATION_FOR_BLOCK: &str = "map_deduplication_for_block__BLOCK_NAME";
pub const MAP_DEDUPLICATION: &str = "map_deduplication";
pub const GET_TMP_FOR_MAP_DEDUPLICATION: &str = "get_tmp_for_map_deduplication__BLOCK_NAME";

// src/kernel/map_deep_deduplication.rs
pub const MAP_DEEP_DEDUPLICATION_FOR_BLOCK: &str = "map_deep_deduplication_for_block__BLOCK_NAME";
pub const MAP_DEEP_DEDUPLICATION: &str = "map_deep_deduplication";
pub const GET_TMP_FOR_MAP_DEEP_DEDUPLICATION: &str =
    "get_tmp_for_map_deep_deduplication__BLOCK_NAME";

// src/kernel/map_reorder.rs
pub const MAP_REORDER_FOR_BLOCK: &str = "map_reorder_for_block__BLOCK_NAME";
// pub const MAP_REORDER: &str = "map_reorder";

// src/kernel/map_get_summary.rs
pub const MAP_GET_SUMMARY: &str = "map_get_summary";
pub const GET_TMP_FOR_MAP_SUMMARY: &str = "get_tmp_for_map_get_summary";

// src/kernel/map_reset.rs
pub const MAP_RESET: &str = "map_reset";
pub const RESET_ALL_MAPS: &str = "reset_all_maps";

// util kernels

// src/kernel/mod.rs - KERNEL_UTILS
pub const GET_ORDERED_PIPE_CONTENT: &str = "get_ordered_pipe_content";
