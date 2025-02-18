// implementation of a queue (FIFO) similar to opencl pipe (FIFO)

// SOURCE from resources/linear_queue.c

use crate::queue::config::{QueueSrc, QueueType};
use crate::queue::kernel::common_replace;

const GLOBALS: &str = r#"
    __global int lq__QUEUE_ID[QUEUE_CAPACITY];
    __global int lq_front__QUEUE_ID = -1;
    __global int lq_rear__QUEUE_ID = -1;
    "#;

const BASE_KERNELS: &str = r#"
    int lq_push__QUEUE_ID(int* v) {
        int rear_i = -1;

        if (lq_rear__QUEUE_ID <= QUEUE_MAX_CAPACITY) {

            rear_i = (atomic_fetch_add(&lq_rear__QUEUE_ID, 1) + 1);

            if (QUEUE_MAX_CAPACITY >= rear_i) {
               lq__QUEUE_ID[rear_i] = *v;
            } else {
               rear_i = - 1;
            }

        }

        return rear_i;
    }

    int lq_pop__QUEUE_ID(int* v) {
        int front_i = -1;

        if (lq_rear__QUEUE_ID != -1 && QUEUE_MAX_CAPACITY >= lq_front__QUEUE_ID) {

            front_i = (atomic_fetch_add(&lq_front__QUEUE_ID, 1) + 1);

            if (lq_rear__QUEUE_ID >= front_i && QUEUE_MAX_CAPACITY >= front_i) {
                *v = lq__QUEUE_ID[front_i];
            } else {
            
                // FIXME linear queue overflow
                if (front_i > lq_rear__QUEUE_ID) {
                    atomic_store(&lq_front__QUEUE_ID, lq_rear__QUEUE_ID);
                }
            
                front_i = -1;
            }

        }

        return front_i;
    }

    kernel void lq_reset__QUEUE_ID() {
        int i = get_global_id(0);

        lq__QUEUE_ID[i] = 0;

        if (i == 0) {
            lq_front__QUEUE_ID = -1, lq_rear__QUEUE_ID = -1;
        }
    }

    kernel void lq_debug__QUEUE_ID(
        global int* items_output,
        global int* meta_output
        ) {
        int i = get_global_id(0);

        items_output[i] = lq__QUEUE_ID[i];
        
        if (i == 0) {
            meta_output[0] = lq_front__QUEUE_ID;
            meta_output[1] = lq_rear__QUEUE_ID;
        }
    }
    
    kernel void write_to_lq__QUEUE_ID(
        global int* input,
        global int* output
        ) {
        int i = get_global_id(0);
        output[i] = lq_push__QUEUE_ID(&input[i]);
    }

    kernel void read_on_lq__QUEUE_ID(
        global int* output
        ) {
        int i = get_global_id(0);

        int pi = -1;
        lq_pop__QUEUE_ID(&pi);

        output[i] = pi;
    }
    "#;

impl QueueSrc {
    pub fn generate_linear_queue_program_source(&self) -> String {
        let blocks = self.get_configs_by_type(QueueType::Lineal);

        if blocks.is_empty() {
            return String::new();
        }

        let mut globals = String::new();

        let mut base_kernels = String::new();

        for config in blocks {
            let template = common_replace(GLOBALS, config);
            globals.push_str(&template);

            let template = common_replace(BASE_KERNELS, config);
            base_kernels.push_str(&template);
        }

        format!(
            "
    /// *** LINEAR QUEUE SRC *** ///

    // lq = linear_queue

    /// constants
    // ...

    /// globals
    {globals}

    /// kernels
    {base_kernels}

    /// *** LINEAR QUEUE SRC *** ///
    "
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a() {
        let mut queue_src = QueueSrc::new();
        queue_src.add_lq(8);

        let program_source = queue_src.generate_linear_queue_program_source();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_b() {
        let mut queue_src = QueueSrc::new();
        queue_src.add_lq(8);
        queue_src.add_lq(32);
        queue_src.add_lq(16);

        let program_source = queue_src.generate_linear_queue_program_source();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_is_empty() {
        let queue_src = QueueSrc::new();

        let program_source = queue_src.generate_linear_queue_program_source();
        println!("{program_source}");
        assert!(program_source.is_empty());
    }
}
