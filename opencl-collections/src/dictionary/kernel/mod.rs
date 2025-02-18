use crate::config::ClTypeTrait;
use crate::dictionary::config::{DictConfig, DictSrc};

pub mod name;

pub mod dict_v1;

fn common_replace<T: ClTypeTrait>(src: &str, config: &DictConfig<T>) -> String {
    let dict_max_capacity = (config.capacity - 1).to_string();

    src.replace("DICT_CAPACITY", &config.capacity.to_string())
        .replace("DICT_MAX_CAPACITY", &dict_max_capacity)
        .replace("DICT_ID", &config.id.to_string())
        .replace("DICT_KEY_LEN", &config.key_len.to_string())
        .replace("DICT_VALUE_LEN", &config.value_len.to_string())
        .replace("CL_DEFAULT_VALUE", &T::cl_enum().cl_default().to_string())
        .replace("CL_TYPE", T::cl_enum().to_cl_type_name())
}

impl<T: ClTypeTrait> DictSrc<T> {
    pub fn build(&self) -> String {
        self.generate_dict_program_source_v1()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a() {
        let mut dict_src: DictSrc<i32> = DictSrc::new();
        dict_src.add(8, 8, 8);

        let program_source = dict_src.build();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_b() {
        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(8, 16, 32);
        dict_src.add(32, 64, 64);
        dict_src.add(16, 32, 32);

        let program_source = dict_src.build();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_is_empty() {
        let dict_src: DictSrc<i32> = DictSrc::new();

        let program_source = dict_src.build();
        println!("{program_source}");
        assert!(program_source.is_empty());
    }
}
