use crate::cache::config::{CacheConfig, CacheSrc};

mod lru;
mod mini_lru;

pub mod name;

fn common_replace(src: &str, config: &CacheConfig) -> String {
    let cache_max_capacity = (config.capacity - 1).to_string();

    src.replace("CACHE_MAX_CAPACITY", &cache_max_capacity)
        .replace("CACHE_CAPACITY", &config.capacity.to_string())
        .replace("CACHE_ID", &config.id.to_string())
        .replace("KEY_LEN", &config.key_len.to_string())
        .replace("VALUE_LEN", &config.value_len.to_string())
}

impl CacheSrc {
    pub fn build(&self) -> String {
        // ...
        let mini_lru_cache_src = self.generate_cache_mini_lru_program_source();

        let lru_cache_src = self.generate_cache_lru_program_source();

        format!(
            "
    /// *** CACHE SRC *** ///

    {mini_lru_cache_src}

    {lru_cache_src}

    /// *** CACHE SRC *** ///
    "
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_only_mini_lru() {
        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(8);

        let program_source = cache_src.build();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_only_lru() {
        let mut cache_src = CacheSrc::new();
        cache_src.add_lru(8, 8, 16);

        let program_source = cache_src.build();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_all_types() {
        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(8);
        cache_src.add_mini_lru(32);
        cache_src.add_lru(16, 16, 16);
        cache_src.add_lru(64, 32, 64);

        let program_source = cache_src.build();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_is_empty() {
        let cache_src = CacheSrc::new();

        let program_source = cache_src.build();
        println!("{program_source}");
        // assert!(program_source.is_empty());
    }
}
