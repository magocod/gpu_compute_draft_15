// SOURCE CODE:
// resources/priority_queue_ordered_array.c
// resources/selectionSort.c

use crate::queue::config::{QueueSrc, QueueType};
use crate::queue::kernel::common_replace;

// FIXME pq_tmp_rear__QUEUE_ID

const GLOBALS: &str = r#"
    // ...
    __global int pq_tmp_rear__QUEUE_ID = -1;
    __global int pq_rear__QUEUE_ID = -1;
    
    __global int pq_value__QUEUE_ID[QUEUE_CAPACITY];
    __global int pq_priority__QUEUE_ID[QUEUE_CAPACITY];

    "#;

const BASE_FUNCTIONS: &str = r#"
    void pq_selection_sort__QUEUE_ID() {
    
        if (pq_rear__QUEUE_ID > 0) {

            for (int i = 0; i < QUEUE_CAPACITY - 1; i++) {
                int min_idx = i;

                // Find the minimum element in the remaining unsorted array
                for (int j = i + 1; j <= pq_rear__QUEUE_ID; j++) {
                    if (pq_priority__QUEUE_ID[j] < pq_priority__QUEUE_ID[min_idx]) {
                        min_idx = j;
                    }
                }

                // Swap the found minimum element with the first element

                int temp_value = pq_value__QUEUE_ID[min_idx];

                pq_value__QUEUE_ID[min_idx] = pq_value__QUEUE_ID[i];
                pq_value__QUEUE_ID[i] = temp_value;

                int temp_priority = pq_priority__QUEUE_ID[min_idx];

                pq_priority__QUEUE_ID[min_idx] = pq_priority__QUEUE_ID[i];
                pq_priority__QUEUE_ID[i] = temp_priority;

            }

        } else {

            // reset required
            if (pq_rear__QUEUE_ID < -1 || (pq_rear__QUEUE_ID <= -1 && pq_tmp_rear__QUEUE_ID >= 0)) {

                // use enqueue_kernel here?, any advantage?

                for (int index = 0; index < QUEUE_CAPACITY; index++) {
                     pq_value__QUEUE_ID[index] = 0;
                     pq_priority__QUEUE_ID[index] = 0;
                }

                pq_tmp_rear__QUEUE_ID = -1, pq_rear__QUEUE_ID = -1;

            }

        }
    
    }
    
    int pq_push__QUEUE_ID(int* value, int* priority) {
        int rear_i = -1;

        if (pq_rear__QUEUE_ID <= QUEUE_MAX_CAPACITY) {
    
            rear_i = (atomic_fetch_add(&pq_tmp_rear__QUEUE_ID, 1) + 1);

            if (QUEUE_MAX_CAPACITY >= rear_i) {
                pq_value__QUEUE_ID[rear_i] = *value;
                pq_priority__QUEUE_ID[rear_i] = *priority;
                
                atomic_fetch_add(&pq_rear__QUEUE_ID, 1);
            } else {
                rear_i = - 1;
            }
        
        }

        return rear_i;
    }

    int pq_pop__QUEUE_ID(int* value) {
        int front_i = -1;

        if (pq_rear__QUEUE_ID > -1) {
        
            front_i = atomic_fetch_sub(&pq_rear__QUEUE_ID, 1);

            if (front_i >= 0) {
            
                *value = pq_value__QUEUE_ID[front_i];

            } else {
            
                front_i = -1;
                
            }
        }

        return front_i;
    }
    "#;

const BASE_KERNELS: &str = r#"
    kernel void pq_reset__QUEUE_ID() {
        int i = get_global_id(0);

        pq_value__QUEUE_ID[i] = 0;
        pq_priority__QUEUE_ID[i] = 0;

        if (i == 0) {
            pq_tmp_rear__QUEUE_ID = -1, pq_rear__QUEUE_ID = -1;
        }
    }

    kernel void pq_debug__QUEUE_ID(
        global int* items_output,
        global int* meta_output
        ) {
        int i = get_global_id(0);

        items_output[i] = pq_value__QUEUE_ID[i];
        items_output[i + QUEUE_CAPACITY] = pq_priority__QUEUE_ID[i];

        if (i == 0) {
            meta_output[0] = pq_tmp_rear__QUEUE_ID;
            meta_output[1] = pq_rear__QUEUE_ID;
        }
    }

    kernel void priority_queue_sort__QUEUE_ID() {
        pq_selection_sort__QUEUE_ID();
    }
    
    kernel void write_to_pq__QUEUE_ID(
        const uint input_global_work_size,
        global int* input,
        global int* output
        ) {
        int i = get_global_id(0);

        output[i] = pq_push__QUEUE_ID(
            &input[i], // value
            &input[i + input_global_work_size] // priority
        );
    }

    kernel void read_on_pq__QUEUE_ID(
        global int* output
        ) {
        int i = get_global_id(0);

        int pi = -1;
        pq_pop__QUEUE_ID(&pi);

        output[i] = pi;
    }

    kernel void write_to_pq_and_sort__QUEUE_ID(
        queue_t q0,
        const uint input_global_work_size,
        const uint input_local_work_size,
        global int* input,
        global int* output,
        global int* enqueue_kernel_output
        ) {

        clk_event_t evt0;

        enqueue_kernel_output[0] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(input_global_work_size, input_local_work_size),
            0,
            NULL,
            &evt0,
            ^{
                int i = get_global_id(0);
                
                output[i] = pq_push__QUEUE_ID(
                    &input[i], // value
                    &input[i + input_global_work_size] // priority
                );
            }
        );

        enqueue_kernel_output[1] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(1, 1),
            1,
            &evt0,
            NULL,
            ^{
               pq_selection_sort__QUEUE_ID();
            }
        );

        release_event(evt0);
    }

    kernel void read_on_pq_and_sort__QUEUE_ID(
        queue_t q0,
        const uint output_global_work_size,
        const uint output_local_work_size,
        global int* output,
        global int* enqueue_kernel_output
        ) {
        clk_event_t evt0;

        enqueue_kernel_output[0] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(output_global_work_size, output_local_work_size),
            0,
            NULL,
            &evt0,
            ^{
                int i = get_global_id(0);
        
                int pi = -1;
                pq_pop__QUEUE_ID(&pi);
        
                output[i] = pi;
            }
        );

        enqueue_kernel_output[1] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(1, 1),
            1,
            &evt0,
            NULL,
            ^{
               pq_selection_sort__QUEUE_ID();
            }
        );

        release_event(evt0);
    }
    "#;

impl QueueSrc {
    pub fn generate_priority_queue_program_source(&self) -> String {
        let blocks = self.get_configs_by_type(QueueType::Priority);

        if blocks.is_empty() {
            return String::new();
        }

        let mut globals = String::new();

        let mut base_kernels = String::new();
        let mut base_functions = String::new();

        for config in blocks {
            let template = common_replace(GLOBALS, config);
            globals.push_str(&template);

            let template = common_replace(BASE_FUNCTIONS, config);
            base_functions.push_str(&template);

            let template = common_replace(BASE_KERNELS, config);
            base_kernels.push_str(&template);
        }

        format!(
            "
    /// *** PRIORITY QUEUE SRC *** ///

    // pq = priority_queue

    /// constants
    // ..

    /// globals
    {globals}

    /// kernels
    {base_functions}

    {base_kernels}

    /// *** PRIORITY QUEUE SRC *** ///
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
        queue_src.add_pq(8);

        let program_source = queue_src.generate_priority_queue_program_source();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_b() {
        let mut queue_src = QueueSrc::new();
        queue_src.add_pq(8);
        queue_src.add_pq(32);
        queue_src.add_pq(16);

        let program_source = queue_src.generate_priority_queue_program_source();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_is_empty() {
        let queue_src = QueueSrc::new();

        let program_source = queue_src.generate_priority_queue_program_source();
        println!("{program_source}");
        assert!(program_source.is_empty());
    }
}
