pub fn get_queue_kernel_name(kernel_name: &str, id: usize) -> String {
    kernel_name.replace("QUEUE_ID", &id.to_string())
}

// linear queue (lq)

pub const LINEAR_QUEUE_DEBUG: &str = "lq_debug__QUEUE_ID";
pub const LINEAR_QUEUE_RESET: &str = "lq_reset__QUEUE_ID";

pub const WRITE_TO_LINEAR_QUEUE: &str = "write_to_lq__QUEUE_ID";
pub const READ_ON_LINEAR_QUEUE: &str = "read_on_lq__QUEUE_ID";

// priority queue (pq)

pub const PRIORITY_QUEUE_DEBUG: &str = "pq_debug__QUEUE_ID";
pub const PRIORITY_QUEUE_RESET: &str = "pq_reset__QUEUE_ID";

pub const PRIORITY_QUEUE_SORT: &str = "priority_queue_sort__QUEUE_ID";

pub const WRITE_TO_PRIORITY_QUEUE: &str = "write_to_pq__QUEUE_ID";
pub const READ_ON_PRIORITY_QUEUE: &str = "read_on_pq__QUEUE_ID";

pub const WRITE_TO_PRIORITY_QUEUE_AND_SORT: &str = "write_to_pq_and_sort__QUEUE_ID";
pub const READ_ON_PRIORITY_QUEUE_AND_SORT: &str = "read_on_pq_and_sort__QUEUE_ID";

// circular queue (cq)

pub const CIRCULAR_QUEUE_DEBUG: &str = "cq_debug__QUEUE_ID";
pub const CIRCULAR_QUEUE_RESET: &str = "cq_reset__QUEUE_ID";

pub const WRITE_TO_CIRCULAR_QUEUE: &str = "write_to_cq__QUEUE_ID";
pub const READ_ON_CIRCULAR_QUEUE: &str = "read_on_cq__QUEUE_ID";

pub const CIRCULAR_QUEUE_PREPARE_WRITE: &str = "circular_queue_prepare_write__QUEUE_ID";
pub const CIRCULAR_QUEUE_CONFIRM_WRITE: &str = "circular_queue_confirm_write__QUEUE_ID";

pub const CIRCULAR_QUEUE_PREPARE_READ: &str = "circular_queue_prepare_read__QUEUE_ID";
pub const CIRCULAR_QUEUE_CONFIRM_READ: &str = "circular_queue_confirm_read__QUEUE_ID";

pub const PREPARE_AND_WRITE_TO_CIRCULAR_QUEUE: &str = "prepare_and_write_to_cq__QUEUE_ID";
pub const PREPARE_AND_READ_ON_CIRCULAR_QUEUE: &str = "prepare_and_read_on_cq__QUEUE_ID";
