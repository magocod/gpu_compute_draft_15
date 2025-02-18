pub fn get_cache_kernel_name(kernel_name: &str, id: usize) -> String {
    kernel_name.replace("CACHE_ID", &id.to_string())
}

// MiniLRU

pub const MINI_LRU_CACHE_DEBUG: &str = "mini_lru_debug__CACHE_ID";
pub const MINI_LRU_CACHE_RESET: &str = "mini_lru_reset__CACHE_ID";
pub const MINI_LRU_CACHE_ARRAY_SET_RESET: &str = "mini_lru_array_set_reset__CACHE_ID";

pub const MINI_LRU_CACHE_GET_KEYS: &str = "mini_lru_get_keys__CACHE_ID";
pub const MINI_LRU_CACHE_GET_SORTED_KEYS: &str = "mini_lru_get_sorted_keys__CACHE_ID";

pub const MINI_LRU_CACHE_SORT: &str = "mini_lru_sort__CACHE_ID";

pub const MINI_LRU_CACHE_PUT: &str = "mini_lru_put__CACHE_ID";

pub const WRITE_IN_MINI_LRU_CACHE: &str = "write_in_mini_lru__CACHE_ID";
pub const READ_ON_MINI_LRU_CACHE: &str = "read_on_mini_lru__CACHE_ID";

// LRU

pub const LRU_CACHE_DEBUG: &str = "lru_debug__CACHE_ID";
pub const LRU_CACHE_RESET: &str = "lru_reset__CACHE_ID";
pub const LRU_CACHE_ARRAY_SET_RESET: &str = "lru_array_set_reset__CACHE_ID";

pub const LRU_CACHE_GET_KEYS: &str = "lru_get_keys__CACHE_ID";
pub const LRU_CACHE_GET_SORTED_KEYS: &str = "lru_get_sorted_keys__CACHE_ID";

pub const LRU_CACHE_SORT: &str = "lru_sort__CACHE_ID";
pub const LRU_CACHE_DEBUG_SORT: &str = "lru_debug_sort__CACHE_ID";

pub const LRU_CACHE_PUT: &str = "lru_put__CACHE_ID";

pub const WRITE_IN_LRU_CACHE: &str = "write_in_lru__CACHE_ID";
pub const READ_ON_LRU_CACHE: &str = "read_on_lru__CACHE_ID";
