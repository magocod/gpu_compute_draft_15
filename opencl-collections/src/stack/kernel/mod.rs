use crate::stack::config::{StackConfig, StackSrc};

pub mod name;
pub mod stack_v1;

fn common_replace(src: &str, config: &StackConfig) -> String {
    let stack_max_capacity = (config.capacity - 1).to_string();

    src.replace("STACK_CAPACITY", &config.capacity.to_string())
        .replace("STACK_ID", &config.id.to_string())
        .replace("STACK_MAX_CAPACITY", &stack_max_capacity)
}

impl StackSrc {
    pub fn build(&self) -> String {
        self.generate_stack_program_source_v1()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a() {
        let mut stack_src = StackSrc::new();
        stack_src.add(8);

        let program_source = stack_src.build();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_b() {
        let mut stack_src = StackSrc::new();
        stack_src.add(8);
        stack_src.add(32);
        stack_src.add(16);

        let program_source = stack_src.build();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_is_empty() {
        let stack_src = StackSrc::new();

        let program_source = stack_src.build();
        println!("{}", program_source);
        assert!(program_source.is_empty());
    }
}
