use crate::queue::config::{QueueSrc, QueueType};
use crate::queue::kernel::common_replace;

const GLOBALS: &str = r#"
    __global int cq__QUEUE_ID[QUEUE_CAPACITY];
    __global int cq_front__QUEUE_ID = -1;
    __global int cq_rear__QUEUE_ID = -1;

    __global int cq_entry_index__QUEUE_ID = -1;
    __global int cq_max_entries__QUEUE_ID = -1;
    __global int cq_entries__QUEUE_ID[QUEUE_CAPACITY];
    "#;

const BASE_FUNCTIONS: &str = r#"
    void cq_reset_entries__QUEUE_ID() {
        for (int i = 0; i < QUEUE_CAPACITY; i++) {
            cq_entries__QUEUE_ID[i] = -1;
        }
    }

    void cq_prepare_write__QUEUE_ID() {
        int front = cq_front__QUEUE_ID;
        int rear = cq_rear__QUEUE_ID;

        int max_entry_index = -1;

        cq_reset_entries__QUEUE_ID();

        for (int i = 0; i < QUEUE_CAPACITY; i++) {

            // isFull
            if ((rear + 1) % QUEUE_CAPACITY == front) {
                break;
            }

            if (front == -1) {
                front = 0;
            }

            rear = (rear + 1) % QUEUE_CAPACITY;
            cq_entries__QUEUE_ID[i] = rear;

            max_entry_index++;
        }

        cq_entry_index__QUEUE_ID = -1;
        cq_max_entries__QUEUE_ID = max_entry_index;
    }

    void cq_confirm_write__QUEUE_ID() {
        if (cq_entry_index__QUEUE_ID == -1) {
            return;
        }

        int limit = cq_entry_index__QUEUE_ID;

        if (limit > cq_max_entries__QUEUE_ID) {
            limit = cq_max_entries__QUEUE_ID;
        }

        cq_reset_entries__QUEUE_ID();

        for (int i = 0; i <= limit; i++) {

            // isFull
            if ((cq_rear__QUEUE_ID + 1) % QUEUE_CAPACITY == cq_front__QUEUE_ID) {
                break;
            }

            if (cq_front__QUEUE_ID == -1) {
                cq_front__QUEUE_ID = 0;
            }

            cq_rear__QUEUE_ID = (cq_rear__QUEUE_ID + 1) % QUEUE_CAPACITY;
        }

        cq_entry_index__QUEUE_ID = -1;
        cq_max_entries__QUEUE_ID = -1;
    }

    void cq_prepare_read__QUEUE_ID() {
        int front = cq_front__QUEUE_ID;
        int rear = cq_rear__QUEUE_ID;

        int max_entry_index = -1;

        cq_reset_entries__QUEUE_ID();

        for (int i = 0; i < QUEUE_CAPACITY; i++) {

            // isEmpty
            if (front == -1) {
                break;
            }

            cq_entries__QUEUE_ID[i] = front;

            if (front == rear) {
                front = rear = -1;
            } else {
                front = (front + 1) % QUEUE_CAPACITY;
            }

            max_entry_index++;
        }

        cq_entry_index__QUEUE_ID = -1;
        cq_max_entries__QUEUE_ID = max_entry_index;
    }

    void cq_confirm_read__QUEUE_ID() {
        if (cq_entry_index__QUEUE_ID == -1) {
            return;
        }

        int limit = cq_entry_index__QUEUE_ID;

        if (limit > cq_max_entries__QUEUE_ID) {
            limit = cq_max_entries__QUEUE_ID;
        }

        cq_reset_entries__QUEUE_ID();

        for (int i = 0; i <= limit; i++) {

            // isEmpty
            if (cq_front__QUEUE_ID == -1) {
                break;
            }

            if (cq_front__QUEUE_ID == cq_rear__QUEUE_ID) {
                cq_front__QUEUE_ID = cq_rear__QUEUE_ID = -1;
            } else {
                cq_front__QUEUE_ID = (cq_front__QUEUE_ID + 1) % QUEUE_CAPACITY;
            }

        }

        cq_entry_index__QUEUE_ID = -1;
        cq_max_entries__QUEUE_ID = -1;
    }
    
    int cq_push__QUEUE_ID(int* value) {
        int rear_i = -1;

        if (cq_max_entries__QUEUE_ID >= cq_entry_index__QUEUE_ID) {

            int entry_index = (atomic_fetch_add(&cq_entry_index__QUEUE_ID, 1) + 1);

            if (cq_max_entries__QUEUE_ID >= entry_index) {

               rear_i = cq_entries__QUEUE_ID[entry_index];
               cq__QUEUE_ID[rear_i] = *value;

            } else {

               rear_i = - 1;

            }

        }

        return rear_i;
    }

    int cq_pop__QUEUE_ID(int* value) {
        int front_i = -1;

        if (cq_max_entries__QUEUE_ID >= cq_entry_index__QUEUE_ID) {

            int entry_index = (atomic_fetch_add(&cq_entry_index__QUEUE_ID, 1) + 1);;

            if (cq_max_entries__QUEUE_ID >= entry_index) {

                front_i = cq_entries__QUEUE_ID[entry_index];
                *value = cq__QUEUE_ID[front_i];

            } else {

                front_i = -1;

            }

        }

        return front_i;
    }
    "#;

const BASE_KERNELS: &str = r#"
    kernel void circular_queue_prepare_write__QUEUE_ID() {
        cq_prepare_write__QUEUE_ID();
    }

    kernel void circular_queue_confirm_write__QUEUE_ID() {
        cq_confirm_write__QUEUE_ID();
    }

    kernel void circular_queue_prepare_read__QUEUE_ID() {
        cq_prepare_read__QUEUE_ID();
    }

    kernel void circular_queue_confirm_read__QUEUE_ID() {
        cq_confirm_read__QUEUE_ID();
    }

    kernel void cq_reset__QUEUE_ID() {
        int i = get_global_id(0);

        cq__QUEUE_ID[i] = 0;

        cq_entries__QUEUE_ID[i] = 0;
        // cq_entries__QUEUE_ID[i] = i;

        if (i == 0) {
            cq_front__QUEUE_ID = -1, cq_rear__QUEUE_ID = -1;
            cq_entry_index__QUEUE_ID = -1;

            // cq_max_entries__QUEUE_ID = QUEUE_MAX_CAPACITY;
            cq_max_entries__QUEUE_ID = -1;
        }
    }

    kernel void cq_debug__QUEUE_ID(
        global int* items_output,
        global int* meta_output
        ) {

        int i = get_global_id(0);

        items_output[i] = cq__QUEUE_ID[i];
        items_output[i + QUEUE_CAPACITY] = cq_entries__QUEUE_ID[i];

        if (i == 0) {
            meta_output[0] = cq_front__QUEUE_ID;
            meta_output[1] = cq_rear__QUEUE_ID;
            meta_output[2] = cq_entry_index__QUEUE_ID;
            meta_output[3] = cq_max_entries__QUEUE_ID;
        }
    }
    
    kernel void write_to_cq__QUEUE_ID(
        global int* input,
        global int* output
        ) {
        int i = get_global_id(0);
        output[i] = cq_push__QUEUE_ID(&input[i]);
    }
    
    kernel void read_on_cq__QUEUE_ID(
        global int* output
        ) {
        int i = get_global_id(0);

        int pi = -1;
        cq_pop__QUEUE_ID(&pi);

        output[i] = pi;
    }

    kernel void prepare_and_write_to_cq__QUEUE_ID(
        queue_t q0,
        const uint input_global_work_size,
        const uint input_local_work_size,
        global int* input,
        global int* output,
        global int* enqueue_kernel_output
        ) {

        cq_prepare_write__QUEUE_ID();

        clk_event_t evt0;

        enqueue_kernel_output[0] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
            ndrange_1D(input_global_work_size, input_local_work_size),
            0,
            NULL,
            &evt0,
            ^{
                int i = get_global_id(0);
                output[i] = cq_push__QUEUE_ID(&input[i]);
            }
        );

        enqueue_kernel_output[1] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
            ndrange_1D(1, 1),
            1,
            &evt0,
            NULL,
            ^{
               cq_confirm_write__QUEUE_ID();
            }
        );

        release_event(evt0);

    }

    kernel void prepare_and_read_on_cq__QUEUE_ID(
        queue_t q0,
        const uint output_global_work_size,
        const uint output_local_work_size,
        global int* output,
        global int* enqueue_kernel_output
        ) {

        cq_prepare_read__QUEUE_ID();

        clk_event_t evt0;

        enqueue_kernel_output[0] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
            ndrange_1D(output_global_work_size, output_local_work_size),
            0,
            NULL,
            &evt0,
            ^{
                int i = get_global_id(0);
        
                int pi = -1;
                cq_pop__QUEUE_ID(&pi);
        
                output[i] = pi;
            }
        );

        enqueue_kernel_output[1] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
            ndrange_1D(1, 1),
            1,
            &evt0,
            NULL,
            ^{
               cq_confirm_read__QUEUE_ID();
            }
        );

        release_event(evt0);

    }
    "#;

impl QueueSrc {
    pub fn generate_circular_queue_program_source(&self) -> String {
        let blocks = self.get_configs_by_type(QueueType::Circular);

        if blocks.is_empty() {
            return String::new();
        }

        // ...
        let mut globals = String::new();

        let mut base_functions = String::new();

        let mut base_kernels = String::new();

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
    /// *** CIRCULAR QUEUE SRC *** ///

    // ciq = circular_queue

    /// constants

    /// globals
    {globals}

    /// kernels
    {base_functions}
    {base_kernels}

    /// *** CIRCULAR QUEUE SRC *** ///
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
        queue_src.add_cq(8);

        let program_source = queue_src.generate_circular_queue_program_source();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_b() {
        let mut queue_src = QueueSrc::new();
        queue_src.add_cq(8);
        queue_src.add_cq(32);
        queue_src.add_cq(16);

        let program_source = queue_src.generate_circular_queue_program_source();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_is_empty() {
        let queue_src = QueueSrc::new();

        let program_source = queue_src.generate_circular_queue_program_source();
        println!("{program_source}");
        assert_eq!(program_source, String::new());
    }
}
