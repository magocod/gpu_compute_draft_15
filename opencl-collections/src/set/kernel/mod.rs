use crate::set::config::{ArraySetVersion, SetConfig, SetSrc};

pub mod array_set_v1;
pub mod array_set_v2;
pub mod name;

fn common_replace(src: &str, config: &SetConfig) -> String {
    let cache_max_capacity = (config.capacity - 1).to_string();

    src.replace("SET_MAX_CAPACITY", &cache_max_capacity)
        .replace("SET_CAPACITY", &config.capacity.to_string())
        .replace("SET_ID", &config.id.to_string())
}

impl SetSrc {
    pub fn build(&self, version: ArraySetVersion) -> String {
        match version {
            ArraySetVersion::V1 => self.generate_array_set_program_source_v1(),
            ArraySetVersion::V2 => self.generate_array_set_program_source_v2(),
        }

        // set_src
        // ...

        // ...
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a() {
        let mut set_src = SetSrc::new();
        set_src.add(8);

        let program_source = set_src.build(ArraySetVersion::V2);
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_b() {
        let mut set_src = SetSrc::new();
        set_src.add(8);
        set_src.add(32);
        set_src.add(16);
        set_src.add(64);

        let program_source = set_src.build(ArraySetVersion::V2);
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_c() {
        let mut set_src = SetSrc::new();
        set_src.add(8);

        let program_source = set_src.build(ArraySetVersion::V1);
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_is_empty() {
        let set_src = SetSrc::new();

        let program_source = set_src.build(ArraySetVersion::V2);
        println!("{program_source}");
        assert!(program_source.is_empty());
    }
}
