pub mod name;

mod circular;
mod linear;
mod priority;

use crate::queue::config::{QueueConfig, QueueSrc};

fn common_replace(src: &str, config: &QueueConfig) -> String {
    let queue_max_capacity = (config.capacity - 1).to_string();

    src.replace("QUEUE_MAX_CAPACITY", &queue_max_capacity)
        .replace("QUEUE_CAPACITY", &config.capacity.to_string())
        .replace("QUEUE_ID", &config.id.to_string())
}

impl QueueSrc {
    pub fn build(&self) -> String {
        // ...
        let linear_queue_src = self.generate_linear_queue_program_source();

        let priority_queue_src = self.generate_priority_queue_program_source();

        let circular_queue_src = self.generate_circular_queue_program_source();

        format!(
            "
    /// *** QUEUE SRC START *** ///

    {linear_queue_src}

    {priority_queue_src}

    {circular_queue_src}

    /// *** QUEUE SRC END *** ///
    "
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_types() {
        let mut queue_src = QueueSrc::new();
        queue_src.add_lq(8);
        queue_src.add_pq(8);
        queue_src.add_cq(8);

        let program_source = queue_src.build();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_only_lq() {
        let mut queue_src = QueueSrc::new();
        queue_src.add_lq(8);
        queue_src.add_lq(32);
        queue_src.add_lq(16);

        let program_source = queue_src.build();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_only_pq() {
        let mut queue_src = QueueSrc::new();
        queue_src.add_pq(8);
        queue_src.add_pq(32);
        queue_src.add_pq(16);

        let program_source = queue_src.build();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_is_empty() {
        let queue_src = QueueSrc::new();

        let program_source = queue_src.build();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }
}
