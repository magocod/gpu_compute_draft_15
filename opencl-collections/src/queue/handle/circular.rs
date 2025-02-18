use crate::config::DEBUG_MODE;
use crate::error::OpenClResult;
use crate::queue::config::QueueConfig;
use crate::queue::kernel::name::{
    get_queue_kernel_name, CIRCULAR_QUEUE_CONFIRM_READ, CIRCULAR_QUEUE_CONFIRM_WRITE,
    CIRCULAR_QUEUE_DEBUG, CIRCULAR_QUEUE_PREPARE_READ, CIRCULAR_QUEUE_PREPARE_WRITE,
    CIRCULAR_QUEUE_RESET, PREPARE_AND_READ_ON_CIRCULAR_QUEUE, PREPARE_AND_WRITE_TO_CIRCULAR_QUEUE,
};
use opencl::opencl_sys::bindings::{cl_int, cl_uint};
use opencl::wrapper::system::OpenclCommonOperation;

#[derive(Debug, PartialEq)]
pub struct CircularQueueSnapshot {
    pub front: cl_int,
    pub rear: cl_int,
    pub items: Vec<cl_int>,
    pub entry_index: cl_int,
    pub max_entries: cl_int,
    pub entries: Vec<cl_int>,
}

impl CircularQueueSnapshot {
    pub fn new(
        front: cl_int,
        rear: cl_int,
        items: Vec<cl_int>,
        entry_index: cl_int,
        max_entries: cl_int,
        entries: Vec<cl_int>,
    ) -> Self {
        Self {
            front,
            rear,
            items,
            entry_index,
            max_entries,
            entries,
        }
    }

    pub fn create_empty(capacity: usize) -> Self {
        // let indices: Vec<_> = (0..capacity as cl_int).collect();

        Self::new(
            -1,
            -1,
            vec![0; capacity],
            -1,
            // (capacity as cl_int) - 1,
            // indices
            -1,
            vec![0; capacity],
        )
    }
}

#[derive(Debug)]
pub struct CircularQueueHandle<T: OpenclCommonOperation> {
    config: QueueConfig,
    system: T,
}

impl<T: OpenclCommonOperation> CircularQueueHandle<T> {
    pub fn new(config: &QueueConfig, system: T) -> Self {
        Self {
            config: config.clone(),
            system,
        }
    }

    pub fn get_id(&self) -> usize {
        self.config.id
    }

    pub fn debug(&self) -> OpenClResult<CircularQueueSnapshot> {
        let global_work_size = self.config.capacity;
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let output_capacity = global_work_size * 2;
        let meta_output_capacity = 4;

        let output_buf = self.system.create_output_buffer(output_capacity)?;
        let meta_buf = self.system.create_output_buffer(meta_output_capacity)?;

        let kernel_name = get_queue_kernel_name(CIRCULAR_QUEUE_DEBUG, self.get_id());
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&output_buf.get_cl_mem())?;
            kernel.set_arg(&meta_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        let output = self
            .system
            .blocking_enqueue_read_buffer(output_capacity, &output_buf, &[])?;

        let items = output[0..self.config.capacity].to_vec();
        let entries = output[self.config.capacity..].to_vec();

        let meta_output =
            self.system
                .blocking_enqueue_read_buffer(meta_output_capacity, &meta_buf, &[])?;

        Ok(CircularQueueSnapshot {
            front: meta_output[0],
            rear: meta_output[1],
            items,
            entry_index: meta_output[2],
            max_entries: meta_output[3],
            entries,
        })
    }

    pub fn print(&self) -> OpenClResult<CircularQueueSnapshot> {
        let qs = self.debug()?;
        // println!("{q_s:?}");
        println!(
            "
CircularQueueSnapshot (
   front: {},
   rear:  {},
   items:       {:?},
   entry_index: {},
   max_entries: {},
   entries:     {:?}
)
        ",
            qs.front, qs.rear, qs.items, qs.entry_index, qs.max_entries, qs.entries
        );
        Ok(qs)
    }

    pub fn reset(&self) -> OpenClResult<()> {
        let global_work_size = self.config.capacity;
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let kernel_name = get_queue_kernel_name(CIRCULAR_QUEUE_RESET, self.get_id());
        let kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        Ok(())
    }

    /// TODO explain
    /// single thread kernel without args
    fn common_kernel(&self, kernel_name: &str) -> OpenClResult<()> {
        let global_work_size = 1;
        let local_work_size = 1;

        let kernel_name = get_queue_kernel_name(kernel_name, self.get_id());
        let kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        Ok(())
    }

    pub fn prepare_write(&self) -> OpenClResult<()> {
        self.common_kernel(CIRCULAR_QUEUE_PREPARE_WRITE)
    }

    pub fn confirm_write(&self) -> OpenClResult<()> {
        self.common_kernel(CIRCULAR_QUEUE_CONFIRM_WRITE)
    }

    pub fn prepare_read(&self) -> OpenClResult<()> {
        self.common_kernel(CIRCULAR_QUEUE_PREPARE_READ)
    }

    pub fn confirm_read(&self) -> OpenClResult<()> {
        self.common_kernel(CIRCULAR_QUEUE_CONFIRM_READ)
    }

    pub fn enqueue(&self, values: &[cl_int]) -> OpenClResult<Vec<cl_int>> {
        let global_work_size = 1;
        let local_work_size = 1;

        let input_len = values.len();

        let output_capacity = input_len;

        // CMQ_WRITE = 0
        // CMQ_CONFIRM = 1
        let enqueue_kernel_output_capacity = 2;

        let input_buf = self.system.blocking_prepare_input_buffer(values)?;

        let output_buf = self.system.create_output_buffer(output_capacity)?;

        let enqueue_kernel_output_buf = self
            .system
            .create_output_buffer(enqueue_kernel_output_capacity)?;

        let kernel_name = get_queue_kernel_name(PREPARE_AND_WRITE_TO_CIRCULAR_QUEUE, self.get_id());
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        let input_global_work_size = input_len as cl_uint;
        let input_local_work_size =
            self.system.first_device_check_local_work_size(input_len) as cl_uint;

        let q0 = self
            .system
            .get_device_command_queue_0()
            .get_cl_command_queue();

        unsafe {
            kernel.set_arg(&q0)?;
            kernel.set_arg(&input_global_work_size)?;
            kernel.set_arg(&input_local_work_size)?;
            kernel.set_arg(&input_buf.get_cl_mem())?;
            kernel.set_arg(&output_buf.get_cl_mem())?;
            kernel.set_arg(&enqueue_kernel_output_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        let indices_output =
            self.system
                .blocking_enqueue_read_buffer(output_capacity, &output_buf, &[])?;

        if DEBUG_MODE {
            println!("enqueue output {indices_output:?}");
        }

        self.system.assert_device_enqueue_kernel(
            enqueue_kernel_output_capacity,
            enqueue_kernel_output_buf,
            &[],
        )?;

        Ok(indices_output)
    }

    pub fn dequeue(&self, take: usize) -> OpenClResult<Vec<cl_int>> {
        let global_work_size = 1;
        let local_work_size = 1;

        let output_len = take;

        let output_capacity = output_len;

        // CMQ_READ = 0
        // CMQ_CONFIRM = 1
        let enqueue_kernel_output_capacity = 2;

        let output_buf = self.system.create_output_buffer(output_capacity)?;

        let enqueue_kernel_output_buf = self
            .system
            .create_output_buffer(enqueue_kernel_output_capacity)?;

        let kernel_name = get_queue_kernel_name(PREPARE_AND_READ_ON_CIRCULAR_QUEUE, self.get_id());
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        let output_global_work_size = output_len as cl_uint;
        let output_local_work_size =
            self.system.first_device_check_local_work_size(output_len) as cl_uint;

        let q0 = self
            .system
            .get_device_command_queue_0()
            .get_cl_command_queue();

        unsafe {
            kernel.set_arg(&q0)?;
            kernel.set_arg(&output_global_work_size)?;
            kernel.set_arg(&output_local_work_size)?;
            kernel.set_arg(&output_buf.get_cl_mem())?;
            kernel.set_arg(&enqueue_kernel_output_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        let values = self
            .system
            .blocking_enqueue_read_buffer(output_capacity, &output_buf, &[])?;

        if DEBUG_MODE {
            println!("dequeue values: {values:?}");
        }

        self.system.assert_device_enqueue_kernel(
            enqueue_kernel_output_capacity,
            enqueue_kernel_output_buf,
            &[],
        )?;

        Ok(values)
    }
}

#[cfg(test)]
mod tests_cq_debug {
    use super::*;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::queue::config::QueueSrc;
    use opencl::wrapper::system::System;

    #[test]
    fn queue_is_empty() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_cq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_cq_config_by_id(0).unwrap();
        let cq = CircularQueueHandle::new(config, system);

        let qs = cq.print().unwrap();
        assert_eq!(qs, CircularQueueSnapshot::create_empty(queue_capacity));

        // write

        cq.prepare_write().unwrap();
        let qs = cq.print().unwrap();

        let items: Vec<_> = vec![0; queue_capacity];
        let entries: Vec<_> = (0..queue_capacity as cl_int).collect();
        assert_eq!(
            qs,
            CircularQueueSnapshot::new(-1, -1, items, -1, (queue_capacity as cl_int) - 1, entries)
        );

        // read

        cq.prepare_read().unwrap();
        let qs = cq.print().unwrap();

        let items: Vec<_> = vec![0; queue_capacity];
        let entries: Vec<_> = vec![-1; queue_capacity];
        assert_eq!(
            qs,
            CircularQueueSnapshot::new(-1, -1, items, -1, -1, entries)
        );
    }

    #[test]
    fn queue_is_full() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_cq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_cq_config_by_id(0).unwrap();
        let cq = CircularQueueHandle::new(config, system);

        let input: Vec<cl_int> = (0..queue_capacity as cl_int).collect();
        let _ = cq.enqueue(&input).unwrap();

        let qs = cq.print().unwrap();
        let items = input.clone();
        let entries = vec![-1; queue_capacity];

        assert_eq!(
            qs,
            CircularQueueSnapshot::new(0, (queue_capacity as cl_int) - 1, items, -1, -1, entries)
        );

        // write

        cq.prepare_write().unwrap();
        let qs = cq.print().unwrap();

        let items = input.clone();
        let entries = vec![-1; queue_capacity];

        assert_eq!(
            qs,
            CircularQueueSnapshot::new(0, (queue_capacity as cl_int) - 1, items, -1, -1, entries)
        );

        // read

        cq.prepare_read().unwrap();
        let qs = cq.print().unwrap();

        let items = input.clone();
        let entries: Vec<_> = input.clone();

        assert_eq!(
            qs,
            CircularQueueSnapshot::new(
                0,
                (queue_capacity as cl_int) - 1,
                items,
                -1,
                (queue_capacity as cl_int) - 1,
                entries
            )
        );
    }

    #[test]
    fn partially_full_queue() {
        let queue_capacity = 32;
        let half_capacity = queue_capacity / 2;

        let mut queue_src = QueueSrc::new();
        queue_src.add_cq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_cq_config_by_id(0).unwrap();
        let cq = CircularQueueHandle::new(config, system);

        let input: Vec<cl_int> = (0..half_capacity as cl_int).collect();
        let _ = cq.enqueue(&input).unwrap();

        let qs = cq.print().unwrap();
        let mut items = input.clone();
        items.append(&mut vec![0; half_capacity]);
        let entries = vec![-1; queue_capacity];

        assert_eq!(
            qs,
            CircularQueueSnapshot::new(0, (half_capacity as cl_int) - 1, items, -1, -1, entries)
        );

        // write

        cq.prepare_write().unwrap();
        let qs = cq.print().unwrap();
        let mut items = input.clone();

        items.append(&mut vec![0; half_capacity]);
        let mut entries: Vec<_> = ((half_capacity as cl_int)..queue_capacity as cl_int).collect();
        entries.append(&mut vec![-1; half_capacity]);

        assert_eq!(
            qs,
            CircularQueueSnapshot::new(
                0,
                (half_capacity as cl_int) - 1,
                items,
                -1,
                (half_capacity as cl_int) - 1,
                entries
            )
        );

        // read

        cq.prepare_read().unwrap();
        let qs = cq.print().unwrap();

        let mut items = input.clone();
        items.append(&mut vec![0; half_capacity]);
        let mut entries: Vec<_> = input.clone();
        entries.append(&mut vec![-1; half_capacity]);

        assert_eq!(
            qs,
            CircularQueueSnapshot::new(
                0,
                (half_capacity as cl_int) - 1,
                items,
                -1,
                (half_capacity as cl_int) - 1,
                entries
            )
        );
    }
}

#[cfg(test)]
mod tests_cq_reset {
    use super::*;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::queue::config::QueueSrc;
    use opencl::wrapper::system::System;

    #[test]
    fn queue_is_empty() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_cq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_cq_config_by_id(0).unwrap();
        let cq = CircularQueueHandle::new(config, system);

        cq.reset().unwrap();
        let qs = cq.print().unwrap();

        assert_eq!(qs, CircularQueueSnapshot::create_empty(queue_capacity));
    }

    #[test]
    fn queue_is_full() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_cq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_cq_config_by_id(0).unwrap();
        let cq = CircularQueueHandle::new(config, system);

        let input = vec![10; queue_capacity];
        let _ = cq.enqueue(&input).unwrap();

        cq.reset().unwrap();
        let qs = cq.print().unwrap();

        assert_eq!(qs, CircularQueueSnapshot::create_empty(queue_capacity));
    }
}

#[cfg(test)]
mod tests_cq_enqueue {
    use super::*;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::queue::config::QueueSrc;
    use opencl::wrapper::system::System;

    #[test]
    fn queue_is_empty() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_cq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_cq_config_by_id(0).unwrap();
        let cq = CircularQueueHandle::new(config, system);

        let input: Vec<cl_int> = (0..queue_capacity as cl_int).collect();

        let result = cq.enqueue(&input).unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        assert_eq!(result_sorted, input);

        cq.print().unwrap();
    }

    #[test]
    fn queue_is_full() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_cq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_cq_config_by_id(0).unwrap();
        let cq = CircularQueueHandle::new(config, system);

        let input: Vec<cl_int> = (0..queue_capacity as cl_int).collect();
        let _ = cq.enqueue(&input).unwrap();

        let input_b = vec![20; queue_capacity];
        let result = cq.enqueue(&input_b).unwrap();

        assert_eq!(result, vec![-1; queue_capacity]);

        cq.print().unwrap();
    }

    #[test]
    fn queue_is_full_2() {
        let queue_capacity = 1024;

        let mut queue_src = QueueSrc::new();
        queue_src.add_cq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_cq_config_by_id(0).unwrap();
        let cq = CircularQueueHandle::new(config, system);

        let input: Vec<cl_int> = (0..queue_capacity as cl_int).collect();

        let result = cq.enqueue(&input).unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        assert_eq!(result_sorted, input);

        cq.print().unwrap();
    }

    #[test]
    fn partially_full_queue() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_cq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_cq_config_by_id(0).unwrap();
        let cq = CircularQueueHandle::new(config, system);

        let input: Vec<cl_int> = (0..(queue_capacity / 2) as cl_int).collect();
        let input_b: Vec<cl_int> = (0..queue_capacity as cl_int)
            .map(|x| x + queue_capacity as cl_int)
            .collect();

        let _ = cq.enqueue(&input).unwrap();
        let result = cq.enqueue(&input_b).unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        let mut expected: Vec<i32> = vec![-1; queue_capacity / 2];
        expected
            .append(&mut (((queue_capacity / 2) as cl_int)..queue_capacity as cl_int).collect());

        assert_eq!(result_sorted, expected);

        cq.print().unwrap();
    }

    #[test]
    fn queue_without_enough_space() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_cq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_cq_config_by_id(0).unwrap();
        let cq = CircularQueueHandle::new(config, system);

        let input: Vec<cl_int> = (0..(queue_capacity * 2) as cl_int).collect();
        let result = cq.enqueue(&input).unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        let mut indices: Vec<i32> = vec![-1; queue_capacity];
        indices.append(&mut (0..queue_capacity as cl_int).collect());
        assert_eq!(result_sorted, indices);

        cq.print().unwrap();
    }
}

#[cfg(test)]
mod tests_cq_dequeue {
    use super::*;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::queue::config::QueueSrc;
    use opencl::wrapper::system::System;

    #[test]
    fn queue_is_empty() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_cq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_cq_config_by_id(0).unwrap();
        let cq = CircularQueueHandle::new(config, system);

        let result = cq.dequeue(queue_capacity).unwrap();

        assert_eq!(result, vec![-1; queue_capacity]);

        cq.print().unwrap();
    }

    #[test]
    fn queue_is_full() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_cq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_cq_config_by_id(0).unwrap();
        let cq = CircularQueueHandle::new(config, system);

        let input: Vec<cl_int> = (0..queue_capacity as cl_int).collect();
        let _ = cq.enqueue(&input).unwrap();
        let result = cq.dequeue(queue_capacity).unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        assert_eq!(result_sorted, input);

        cq.print().unwrap();
    }

    #[test]
    fn queue_is_full_2() {
        let queue_capacity = 1024;

        let mut queue_src = QueueSrc::new();
        queue_src.add_cq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_cq_config_by_id(0).unwrap();
        let cq = CircularQueueHandle::new(config, system);

        let input: Vec<cl_int> = (0..queue_capacity as cl_int).collect();
        let _ = cq.enqueue(&input).unwrap();
        let result = cq.dequeue(queue_capacity).unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        assert_eq!(result_sorted, input);

        cq.print().unwrap();
    }

    #[test]
    fn queue_is_full_3() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_cq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_cq_config_by_id(0).unwrap();
        let cq = CircularQueueHandle::new(config, system);

        let input: Vec<cl_int> = (0..queue_capacity as cl_int).collect();
        let _ = cq.enqueue(&input).unwrap();
        let result = cq.dequeue(queue_capacity * 2).unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        let mut expected = vec![-1; queue_capacity];
        expected.append(&mut input.clone());

        assert_eq!(result_sorted, expected);

        cq.print().unwrap();
    }

    #[test]
    fn partially_full_queue() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_cq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_cq_config_by_id(0).unwrap();
        let cq = CircularQueueHandle::new(config, system);

        let input: Vec<cl_int> = (0..(queue_capacity / 2) as cl_int).collect();
        let _ = cq.enqueue(&input).unwrap();
        let result = cq.dequeue(queue_capacity).unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        let mut expected = vec![-1; queue_capacity / 2];
        expected.append(&mut input.clone());

        assert_eq!(result_sorted, expected);

        cq.print().unwrap();
    }

    #[test]
    fn queue_emptied() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_cq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_cq_config_by_id(0).unwrap();
        let cq = CircularQueueHandle::new(config, system);

        let input: Vec<cl_int> = (0..queue_capacity as cl_int).collect();
        let _ = cq.enqueue(&input).unwrap();
        let _ = cq.dequeue(queue_capacity).unwrap();
        let result = cq.dequeue(queue_capacity).unwrap();

        assert_eq!(result, vec![-1; 32]);

        cq.print().unwrap();
    }
}

// TODO explain tests
#[cfg(test)]
mod tests_cq_examples {
    use super::*;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::queue::config::QueueSrc;
    use opencl::wrapper::system::System;

    #[test]
    fn case_1() {
        let queue_capacity = 8;

        let mut queue_src = QueueSrc::new();
        queue_src.add_cq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_cq_config_by_id(0).unwrap();
        let cq = CircularQueueHandle::new(config, system);

        let input = vec![10, 20, 30, 40, 50, 60];
        let result = cq.enqueue(&input).unwrap();

        let expected: Vec<_> = (0..6).collect();
        let mut result_sorted = result.clone();
        result_sorted.sort();
        assert_eq!(result_sorted, expected);

        let result = cq.dequeue(1).unwrap();
        assert_eq!(result, vec![10]);

        let result = cq.dequeue(4).unwrap();
        assert_eq!(result, vec![20, 30, 40, 50]);

        let result = cq.dequeue(4).unwrap();
        assert_eq!(result, vec![60, -1, -1, -1]);

        let input = vec![70, 80];
        let _ = cq.enqueue(&input).unwrap();

        let result = cq.dequeue(4).unwrap();
        assert_eq!(result, vec![70, 80, -1, -1]);

        let result = cq.dequeue(4).unwrap();
        assert_eq!(result, vec![-1; 4]);

        cq.print().unwrap();
    }

    #[test]
    fn case_2() {
        let queue_capacity = 8;

        let mut queue_src = QueueSrc::new();
        queue_src.add_cq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_cq_config_by_id(0).unwrap();
        let cq = CircularQueueHandle::new(config, system);

        let input = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let result = cq.enqueue(&input).unwrap();

        let expected: Vec<_> = (0..8).collect();
        let mut result_sorted = result.clone();
        result_sorted.sort();
        assert_eq!(result_sorted, expected);

        let result = cq.dequeue(1).unwrap();
        assert_eq!(result, vec![10]);

        let result = cq.dequeue(4).unwrap();
        assert_eq!(result, vec![20, 30, 40, 50]);

        let input = vec![100, 200, 300, 400, 500, 600];
        let result = cq.enqueue(&input).unwrap();

        let mut expected: Vec<_> = vec![-1];
        expected.append(&mut (0..5).collect());
        let mut result_sorted = result.clone();
        result_sorted.sort();
        assert_eq!(result_sorted, expected);

        let result = cq.dequeue(2).unwrap();
        assert_eq!(result, vec![60, 70]);

        let input = vec![1000, 2000, 3000, 4000];
        let result = cq.enqueue(&input).unwrap();
        assert_eq!(result, vec![5, 6, -1, -1]);

        let result = cq.dequeue(4).unwrap();
        assert_eq!(result, vec![80, 100, 200, 300]);

        cq.print().unwrap();
    }
}
