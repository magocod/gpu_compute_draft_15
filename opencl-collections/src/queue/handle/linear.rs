use crate::config::DEBUG_MODE;
use crate::error::OpenClResult;
use crate::queue::config::QueueConfig;
use crate::queue::kernel::name::{
    get_queue_kernel_name, LINEAR_QUEUE_DEBUG, LINEAR_QUEUE_RESET, READ_ON_LINEAR_QUEUE,
    WRITE_TO_LINEAR_QUEUE,
};
use opencl::opencl_sys::bindings::cl_int;
use opencl::wrapper::system::OpenclCommonOperation;

#[derive(Debug, PartialEq)]
pub struct LinearQueueSnapshot {
    pub front: cl_int,
    pub rear: cl_int,
    pub items: Vec<cl_int>,
}

impl LinearQueueSnapshot {
    pub fn new(front: cl_int, rear: cl_int, items: Vec<cl_int>) -> Self {
        Self { front, rear, items }
    }

    pub fn create_empty(capacity: usize) -> Self {
        Self::new(-1, -1, vec![0; capacity])
    }
}

#[derive(Debug)]
pub struct LinearQueueHandle<T: OpenclCommonOperation> {
    config: QueueConfig,
    system: T,
}

impl<T: OpenclCommonOperation> LinearQueueHandle<T> {
    pub fn new(config: &QueueConfig, system: T) -> Self {
        Self {
            config: config.clone(),
            system,
        }
    }

    pub fn get_id(&self) -> usize {
        self.config.id
    }

    pub fn debug(&self) -> OpenClResult<LinearQueueSnapshot> {
        let global_work_size = self.config.capacity;
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let meta_output_capacity = 2;

        let output_buf = self.system.create_output_buffer(global_work_size)?;
        let meta_buf = self.system.create_output_buffer(global_work_size)?;

        let kernel_name = get_queue_kernel_name(LINEAR_QUEUE_DEBUG, self.get_id());
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

        let output =
            self.system
                .blocking_enqueue_read_buffer(global_work_size, &output_buf, &[])?;

        let meta_output =
            self.system
                .blocking_enqueue_read_buffer(meta_output_capacity, &meta_buf, &[])?;

        Ok(LinearQueueSnapshot {
            front: meta_output[0],
            rear: meta_output[1],
            items: output,
        })
    }

    pub fn print(&self) -> OpenClResult<LinearQueueSnapshot> {
        let qs = self.debug()?;
        println!("{qs:?}");
        Ok(qs)
    }

    pub fn reset(&self) -> OpenClResult<()> {
        let global_work_size = self.config.capacity;
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let kernel_name = get_queue_kernel_name(LINEAR_QUEUE_RESET, self.get_id());
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

    pub fn enqueue(&self, input: &[cl_int]) -> OpenClResult<Vec<cl_int>> {
        let global_work_size = input.len();
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let input_buf = self.system.blocking_prepare_input_buffer(input)?;

        let output_buf = self.system.create_output_buffer(global_work_size)?;

        let kernel_name = get_queue_kernel_name(WRITE_TO_LINEAR_QUEUE, self.get_id());
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&input_buf.get_cl_mem())?;
            kernel.set_arg(&output_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        let output =
            self.system
                .blocking_enqueue_read_buffer(global_work_size, &output_buf, &[])?;

        if DEBUG_MODE {
            println!("enqueue output {output:?}");
        }

        Ok(output)
    }

    pub fn dequeue(&self, take: usize) -> OpenClResult<Vec<cl_int>> {
        let global_work_size = take;
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let output_buf = self.system.create_output_buffer(global_work_size)?;

        let kernel_name = get_queue_kernel_name(READ_ON_LINEAR_QUEUE, self.get_id());
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&output_buf.get_cl_mem())?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        let output =
            self.system
                .blocking_enqueue_read_buffer(global_work_size, &output_buf, &[])?;

        if DEBUG_MODE {
            println!("dequeue output {output:?}");
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests_lq_debug {
    use super::*;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::queue::config::QueueSrc;
    use opencl::wrapper::system::System;

    #[test]
    fn queue_is_empty() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_lq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_lq_config_by_id(0).unwrap();
        let lq = LinearQueueHandle::new(config, system);

        let qs = lq.print().unwrap();

        assert_eq!(qs, LinearQueueSnapshot::create_empty(queue_capacity));
    }

    #[test]
    fn queue_is_full() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_lq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_lq_config_by_id(0).unwrap();
        let lq = LinearQueueHandle::new(config, system);

        let input: Vec<cl_int> = (0..queue_capacity as cl_int).collect();
        let _ = lq.enqueue(&input).unwrap();
        let qs = lq.print().unwrap();

        let mut sorted_items = qs.items.clone();
        sorted_items.sort();

        assert_eq!(qs.front, -1);
        assert_eq!(qs.rear, (queue_capacity - 1) as cl_int);
        assert_eq!(sorted_items, input);
    }

    #[test]
    fn partially_full_queue() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_lq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_lq_config_by_id(0).unwrap();
        let lq = LinearQueueHandle::new(config, system);

        let input: Vec<cl_int> = (0..(queue_capacity / 2) as cl_int).collect();
        let _ = lq.enqueue(&input).unwrap();

        let qs = lq.print().unwrap();

        let mut sorted_items = qs.items.clone();
        sorted_items.sort();

        let mut expected = vec![0; queue_capacity / 2];
        expected.append(&mut input.clone());

        assert_eq!(qs.front, -1);
        assert_eq!(qs.rear, ((queue_capacity / 2) - 1) as cl_int);
        assert_eq!(sorted_items, expected);
    }

    #[test]
    fn queue_emptied() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_lq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_lq_config_by_id(0).unwrap();
        let lq = LinearQueueHandle::new(config, system);

        let input: Vec<cl_int> = (0..queue_capacity as cl_int).collect();
        let _ = lq.enqueue(&input).unwrap();
        let _ = lq.dequeue(queue_capacity).unwrap();

        let qs = lq.print().unwrap();

        let mut sorted_items = qs.items.clone();
        sorted_items.sort();

        assert_eq!(qs.front, (queue_capacity - 1) as cl_int);
        assert_eq!(qs.rear, (queue_capacity - 1) as cl_int);
        assert_eq!(sorted_items, input);
    }

    // FIXME linear queue overflow
    #[test]
    fn overflowing_queue() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_lq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_lq_config_by_id(0).unwrap();
        let lq = LinearQueueHandle::new(config, system);

        let input: Vec<cl_int> = (0..(queue_capacity * 2) as cl_int).collect();
        let _ = lq.enqueue(&input).unwrap();
        let _ = lq.dequeue(queue_capacity * 2).unwrap();

        let qs = lq.print().unwrap();

        let mut sorted_items = qs.items.clone();
        sorted_items.sort();

        assert_eq!(qs.front, ((queue_capacity * 2) - 1) as cl_int);
        assert_eq!(qs.rear, ((queue_capacity * 2) - 1) as cl_int);
        assert_eq!(sorted_items, input[..queue_capacity]);
    }
}

#[cfg(test)]
mod tests_lq_reset {
    use super::*;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::queue::config::QueueSrc;
    use opencl::wrapper::system::System;

    #[test]
    fn queue_is_empty() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_lq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_lq_config_by_id(0).unwrap();
        let lq = LinearQueueHandle::new(config, system);

        lq.reset().unwrap();
        let qs = lq.print().unwrap();

        assert_eq!(qs, LinearQueueSnapshot::create_empty(queue_capacity));
    }

    #[test]
    fn queue_is_full() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_lq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_lq_config_by_id(0).unwrap();
        let lq = LinearQueueHandle::new(config, system);

        let input = vec![10; queue_capacity];
        let _ = lq.enqueue(&input).unwrap();

        lq.reset().unwrap();
        let qs = lq.print().unwrap();

        assert_eq!(qs, LinearQueueSnapshot::create_empty(queue_capacity));
    }
}

#[cfg(test)]
mod tests_lq_enqueue {
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::queue::config::QueueSrc;
    use crate::queue::handle::linear::LinearQueueHandle;
    use opencl::opencl_sys::bindings::cl_int;
    use opencl::wrapper::system::System;

    #[test]
    fn queue_is_empty() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_lq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_lq_config_by_id(0).unwrap();
        let lq = LinearQueueHandle::new(config, system);

        let input: Vec<cl_int> = (0..queue_capacity as cl_int).collect();

        let result = lq.enqueue(&input).unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        assert_eq!(result_sorted, input);

        lq.print().unwrap();
    }

    #[test]
    fn queue_is_empty_2() {
        let queue_capacity = 1024;

        let mut queue_src = QueueSrc::new();
        queue_src.add_lq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_lq_config_by_id(0).unwrap();
        let lq = LinearQueueHandle::new(config, system);

        let input: Vec<cl_int> = (0..queue_capacity as cl_int).collect();

        let result = lq.enqueue(&input).unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        assert_eq!(result_sorted, input);

        lq.print().unwrap();
    }

    #[test]
    fn queue_is_full() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_lq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_lq_config_by_id(0).unwrap();
        let lq = LinearQueueHandle::new(config, system);

        let input_1: Vec<cl_int> = (0..queue_capacity as cl_int).collect();
        let _ = lq.enqueue(&input_1).unwrap();

        let input_2 = vec![20; queue_capacity];
        let result = lq.enqueue(&input_2).unwrap();

        assert_eq!(result, vec![-1; queue_capacity]);

        lq.print().unwrap();
    }

    #[test]
    fn partially_full_queue() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_lq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_lq_config_by_id(0).unwrap();
        let lq = LinearQueueHandle::new(config, system);

        let input: Vec<cl_int> = (0..(queue_capacity / 2) as cl_int).collect();
        let input_b: Vec<cl_int> = (0..queue_capacity as cl_int)
            .map(|x| x + queue_capacity as cl_int)
            .collect();

        let _ = lq.enqueue(&input).unwrap();
        let result = lq.enqueue(&input_b).unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        let mut expected: Vec<i32> = vec![-1; queue_capacity / 2];
        expected
            .append(&mut (((queue_capacity / 2) as cl_int)..queue_capacity as cl_int).collect());

        assert_eq!(result_sorted, expected);

        lq.print().unwrap();
    }

    // FIXME linear queue overflow
    #[test]
    fn queue_without_enough_space() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_lq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_lq_config_by_id(0).unwrap();
        let lq = LinearQueueHandle::new(config, system);

        let input: Vec<cl_int> = (0..(queue_capacity * 2) as cl_int).collect();
        let result = lq.enqueue(&input).unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        let mut indices: Vec<i32> = vec![-1; queue_capacity];
        indices.append(&mut (0..queue_capacity as cl_int).collect());
        assert_eq!(result_sorted, indices);

        lq.print().unwrap();
    }
}

#[cfg(test)]
mod tests_lq_dequeue {
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::queue::config::QueueSrc;
    use crate::queue::handle::linear::LinearQueueHandle;
    use opencl::opencl_sys::bindings::cl_int;
    use opencl::wrapper::system::System;

    #[test]
    fn queue_is_empty() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_lq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_lq_config_by_id(0).unwrap();
        let lq = LinearQueueHandle::new(config, system);

        let result = lq.dequeue(queue_capacity).unwrap();

        assert_eq!(result, vec![-1; queue_capacity]);

        lq.print().unwrap();
    }

    #[test]
    fn queue_is_full() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_lq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_lq_config_by_id(0).unwrap();
        let lq = LinearQueueHandle::new(config, system);

        let input: Vec<cl_int> = (0..queue_capacity as cl_int).collect();
        let _ = lq.enqueue(&input).unwrap();
        let result = lq.dequeue(queue_capacity).unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        assert_eq!(result_sorted, input);

        lq.print().unwrap();
    }

    #[test]
    fn queue_is_full_2() {
        let queue_capacity = 1024;

        let mut queue_src = QueueSrc::new();
        queue_src.add_lq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_lq_config_by_id(0).unwrap();
        let lq = LinearQueueHandle::new(config, system);

        let input: Vec<cl_int> = (0..queue_capacity as cl_int).collect();
        let _ = lq.enqueue(&input).unwrap();
        let result = lq.dequeue(queue_capacity).unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        assert_eq!(result_sorted, input);

        lq.print().unwrap();
    }

    #[test]
    fn queue_is_full_3() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_lq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_lq_config_by_id(0).unwrap();
        let lq = LinearQueueHandle::new(config, system);

        let input: Vec<cl_int> = (0..queue_capacity as cl_int).collect();
        let _ = lq.enqueue(&input).unwrap();
        let result = lq.dequeue(queue_capacity * 2).unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        let mut expected = vec![-1; queue_capacity];
        expected.append(&mut input.clone());

        assert_eq!(result_sorted, expected);

        lq.print().unwrap();
    }

    #[test]
    fn partially_full_queue() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_lq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_lq_config_by_id(0).unwrap();
        let lq = LinearQueueHandle::new(config, system);

        let input: Vec<cl_int> = (0..(queue_capacity / 2) as cl_int).collect();
        let _ = lq.enqueue(&input).unwrap();
        let result = lq.dequeue(queue_capacity).unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        let mut expected = vec![-1; queue_capacity / 2];
        expected.append(&mut input.clone());

        assert_eq!(result_sorted, expected);

        lq.print().unwrap();
    }

    #[test]
    fn queue_emptied() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_lq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_lq_config_by_id(0).unwrap();
        let lq = LinearQueueHandle::new(config, system);

        let input: Vec<cl_int> = (0..queue_capacity as cl_int).collect();
        let _ = lq.enqueue(&input).unwrap();
        let _ = lq.dequeue(queue_capacity).unwrap();
        let result = lq.dequeue(queue_capacity).unwrap();

        assert_eq!(result, vec![-1; 32]);

        lq.print().unwrap();
    }

    // FIXME linear queue overflow
    #[test]
    fn queue_is_full_with_overloaded_rear_tail() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_lq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_lq_config_by_id(0).unwrap();
        let lq = LinearQueueHandle::new(config, system);

        let input: Vec<cl_int> = (0..(queue_capacity * 2) as cl_int).collect();
        let _ = lq.enqueue(&input).unwrap();

        let result = lq.dequeue(queue_capacity).unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        assert_eq!(result_sorted, input[..queue_capacity]);

        lq.print().unwrap();
    }
}

// TODO explain tests
#[cfg(test)]
mod tests_lq_examples {
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::queue::config::QueueSrc;
    use crate::queue::handle::linear::LinearQueueHandle;
    use opencl::wrapper::system::System;

    #[test]
    fn case_1() {
        let queue_capacity = 8;

        let mut queue_src = QueueSrc::new();
        queue_src.add_lq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_lq_config_by_id(0).unwrap();
        let lq = LinearQueueHandle::new(config, system);

        let input = vec![10, 20, 30, 40, 50, 60];
        let _ = lq.enqueue(&input).unwrap();

        let result = lq.dequeue(1).unwrap();
        assert_eq!(result, vec![10]);

        let result = lq.dequeue(4).unwrap();
        assert_eq!(result, vec![20, 30, 40, 50]);

        let result = lq.dequeue(4).unwrap();
        assert_eq!(result, vec![60, -1, -1, -1]);

        let input = vec![70, 80];
        let _ = lq.enqueue(&input).unwrap();

        let result = lq.dequeue(4).unwrap();
        assert_eq!(result, vec![70, 80, -1, -1]);

        let result = lq.dequeue(4).unwrap();
        assert_eq!(result, vec![-1; 4]);

        lq.print().unwrap();
    }
}
