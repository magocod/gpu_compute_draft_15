use crate::config::DEBUG_MODE;
use crate::error::OpenClResult;
use crate::queue::config::QueueConfig;
use crate::queue::kernel::name::{
    get_queue_kernel_name, PRIORITY_QUEUE_DEBUG, PRIORITY_QUEUE_RESET, PRIORITY_QUEUE_SORT,
    READ_ON_PRIORITY_QUEUE_AND_SORT, WRITE_TO_PRIORITY_QUEUE_AND_SORT,
};
use opencl::opencl_sys::bindings::{cl_int, cl_uint};
use opencl::wrapper::system::OpenclCommonOperation;

#[derive(Debug, PartialEq)]
pub struct PriorityQueueSnapshot {
    pub tmp_rear: cl_int,
    pub rear: cl_int,
    pub values: Vec<cl_int>,
    pub priorities: Vec<cl_int>,
}

impl PriorityQueueSnapshot {
    pub fn new(
        tmp_rear: cl_int,
        rear: cl_int,
        values: Vec<cl_int>,
        priorities: Vec<cl_int>,
    ) -> Self {
        Self {
            tmp_rear,
            rear,
            values,
            priorities,
        }
    }

    pub fn create_empty(capacity: usize) -> Self {
        Self::new(-1, -1, vec![0; capacity], vec![0; capacity])
    }
}

#[derive(Debug)]
pub struct PriorityQueueHandle<T: OpenclCommonOperation> {
    config: QueueConfig,
    system: T,
}

impl<T: OpenclCommonOperation> PriorityQueueHandle<T> {
    pub fn new(config: &QueueConfig, system: T) -> Self {
        Self {
            config: config.clone(),
            system,
        }
    }

    pub fn get_id(&self) -> usize {
        self.config.id
    }

    pub fn debug(&self) -> OpenClResult<PriorityQueueSnapshot> {
        let global_work_size = self.config.capacity;
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let output_capacity = global_work_size * 2;
        let meta_output_capacity = 2;

        let output_buf = self.system.create_output_buffer(output_capacity)?;
        let meta_buf = self.system.create_output_buffer(meta_output_capacity)?;

        let kernel_name = get_queue_kernel_name(PRIORITY_QUEUE_DEBUG, self.get_id());
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
        };

        let output = self
            .system
            .blocking_enqueue_read_buffer(output_capacity, &output_buf, &[])?;

        let values = output[0..self.config.capacity].to_vec();
        let priorities = output[self.config.capacity..].to_vec();

        let meta_output =
            self.system
                .blocking_enqueue_read_buffer(meta_output_capacity, &meta_buf, &[])?;

        Ok(PriorityQueueSnapshot {
            tmp_rear: meta_output[0],
            rear: meta_output[1],
            values,
            priorities,
        })
    }

    pub fn print(&self) -> OpenClResult<PriorityQueueSnapshot> {
        let qs = self.debug()?;
        // println!("{q_s:?}");
        println!(
            "
PriorityQueueSnapshot (
   tmp_rear: {},
   rear: {},
   values:     {:?},
   priorities: {:?}
)
        ",
            qs.tmp_rear, qs.rear, qs.values, qs.priorities
        );
        Ok(qs)
    }

    pub fn reset(&self) -> OpenClResult<()> {
        let global_work_size = self.config.capacity;
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let kernel_name = get_queue_kernel_name(PRIORITY_QUEUE_RESET, self.get_id());
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

    pub fn sort(&self) -> OpenClResult<()> {
        let global_work_size = 1;
        let local_work_size = 1;

        let kernel_name = get_queue_kernel_name(PRIORITY_QUEUE_SORT, self.get_id());
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

    pub fn enqueue(&self, values: &[cl_int], priorities: &[cl_int]) -> OpenClResult<Vec<cl_int>> {
        if values.len() != priorities.len() {
            panic!("error handle values & priorities len");
        }

        let global_work_size = 1;
        let local_work_size = 1;

        let output_capacity = values.len();

        // CMQ_INSERT = 0
        // CMQ_SORT = 1
        let enqueue_kernel_output_capacity = 2;

        let mut input = values.to_vec();
        input.append(&mut priorities.to_vec());

        let input_buf = self.system.blocking_prepare_input_buffer(&input)?;

        let output_buf = self.system.create_output_buffer(output_capacity)?;

        let enqueue_kernel_output_buf = self
            .system
            .create_output_buffer(enqueue_kernel_output_capacity)?;

        let input_global_work_size = values.len() as cl_uint;
        let input_local_work_size =
            self.system.first_device_check_local_work_size(values.len()) as cl_uint;

        let q0 = self
            .system
            .get_device_command_queue_0()
            .get_cl_command_queue();

        let kernel_name = get_queue_kernel_name(WRITE_TO_PRIORITY_QUEUE_AND_SORT, self.get_id());
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&q0)?;
            kernel.set_arg(&input_global_work_size)?;
            kernel.set_arg(&input_local_work_size)?;
            kernel.set_arg(&input_buf)?;
            kernel.set_arg(&output_buf)?;
            kernel.set_arg(&enqueue_kernel_output_buf)?;

            kernel.enqueue_nd_range_kernel_dim_1(
                self.system.get_host_command_queue(),
                global_work_size,
                local_work_size,
                &[],
            )?;
        }

        let index_output =
            self.system
                .blocking_enqueue_read_buffer(output_capacity, &output_buf, &[])?;

        if DEBUG_MODE {
            println!("enqueue output {index_output:?}");
        }

        self.system.assert_device_enqueue_kernel(
            enqueue_kernel_output_capacity,
            enqueue_kernel_output_buf,
            &[],
        )?;

        Ok(index_output)
    }

    pub fn dequeue(&self, take: usize) -> OpenClResult<Vec<cl_int>> {
        let global_work_size = 1;
        let local_work_size = 1;

        let output_capacity = take;

        // CMQ_GET = 0
        // CMQ_SORT = 1
        let enqueue_kernel_output_capacity = 2;

        let output_global_work_size = take as cl_uint;
        let output_local_work_size =
            self.system.first_device_check_local_work_size(take) as cl_uint;

        let output_buf = self.system.create_output_buffer(output_capacity)?;

        let enqueue_kernel_output_buf = self
            .system
            .create_output_buffer(enqueue_kernel_output_capacity)?;

        let q0 = self
            .system
            .get_device_command_queue_0()
            .get_cl_command_queue();

        let kernel_name = get_queue_kernel_name(READ_ON_PRIORITY_QUEUE_AND_SORT, self.get_id());
        let mut kernel = self.system.create_kernel(&kernel_name)?;

        unsafe {
            kernel.set_arg(&q0)?;
            kernel.set_arg(&output_global_work_size)?;
            kernel.set_arg(&output_local_work_size)?;
            kernel.set_arg(&output_buf)?;
            kernel.set_arg(&enqueue_kernel_output_buf)?;

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

        if DEBUG_MODE {
            println!("dequeue values: {output:?}");
        }

        self.system.assert_device_enqueue_kernel(
            enqueue_kernel_output_capacity,
            enqueue_kernel_output_buf,
            &[],
        )?;

        Ok(output)
    }
}

#[cfg(test)]
mod tests_pq_debug {
    use super::*;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::queue::config::QueueSrc;
    use opencl::wrapper::system::System;

    #[test]
    fn queue_is_empty() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_pq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_pq_config_by_id(0).unwrap();
        let pq = PriorityQueueHandle::new(config, system);

        let qs = pq.print().unwrap();

        assert_eq!(qs, PriorityQueueSnapshot::create_empty(queue_capacity));
    }

    #[test]
    fn queue_is_full() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_pq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_pq_config_by_id(0).unwrap();
        let pq = PriorityQueueHandle::new(config, system);

        let vec_a = vec![8; queue_capacity / 4];
        let vec_b = vec![1; queue_capacity / 4];
        let vec_c = vec![3; queue_capacity / 4];
        let vec_d = vec![2; queue_capacity / 4];

        let mut values = vec_a.clone();
        values.append(&mut vec_b.clone());
        values.append(&mut vec_c.clone());
        values.append(&mut vec_d.clone());

        let priorities = &values;

        let _ = pq.enqueue(&values, priorities).unwrap();

        let qs = pq.print().unwrap();

        let mut expected = vec_b.clone();
        expected.append(&mut vec_d.clone());
        expected.append(&mut vec_c.clone());
        expected.append(&mut vec_a.clone());

        assert_eq!(qs.tmp_rear, (queue_capacity - 1) as cl_int);
        assert_eq!(qs.rear, (queue_capacity - 1) as cl_int);
        assert_eq!(qs.values, expected);
        assert_eq!(qs.priorities, expected);
    }

    #[test]
    fn partially_full_queue() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_pq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_pq_config_by_id(0).unwrap();
        let pq = PriorityQueueHandle::new(config, system);

        let vec_a = vec![7; queue_capacity / 4];
        let vec_b = vec![2; queue_capacity / 4];

        let mut values = vec_a.clone();
        values.append(&mut vec_b.clone());

        let priorities = &values;

        let _ = pq.enqueue(&values, priorities).unwrap();
        let qs = pq.print().unwrap();

        let mut expected = vec_b.clone();
        expected.append(&mut vec_a.clone());
        expected.append(&mut vec![0; queue_capacity / 2]);

        assert_eq!(qs.tmp_rear, ((queue_capacity / 2) - 1) as cl_int);
        assert_eq!(qs.rear, ((queue_capacity / 2) - 1) as cl_int);
        assert_eq!(qs.values, expected);
        assert_eq!(qs.priorities, expected);
    }

    #[test]
    fn queue_emptied() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_pq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_pq_config_by_id(0).unwrap();
        let pq = PriorityQueueHandle::new(config, system);

        let values = vec![1; queue_capacity];
        let priorities = vec![10; queue_capacity];

        pq.enqueue(&values, &priorities).unwrap();
        pq.dequeue(queue_capacity).unwrap();

        let qs = pq.print().unwrap();
        assert_eq!(qs, PriorityQueueSnapshot::create_empty(queue_capacity));
    }
}

#[cfg(test)]
mod tests_pq_reset {
    use super::*;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::queue::config::QueueSrc;
    use opencl::wrapper::system::System;

    #[test]
    fn queue_is_empty() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_pq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_pq_config_by_id(0).unwrap();
        let pq = PriorityQueueHandle::new(config, system);

        let r = pq.reset();
        assert!(r.is_ok());

        let qs = pq.print().unwrap();

        assert_eq!(qs, PriorityQueueSnapshot::create_empty(queue_capacity));
    }

    #[test]
    fn queue_is_full() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_pq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_pq_config_by_id(0).unwrap();
        let pq = PriorityQueueHandle::new(config, system);

        let input_values = vec![1; queue_capacity];
        let input_priorities = vec![10; queue_capacity];

        let _ = pq.enqueue(&input_values, &input_priorities).unwrap();

        let r = pq.reset();
        assert!(r.is_ok());

        let qs = pq.print().unwrap();
        assert_eq!(qs, PriorityQueueSnapshot::create_empty(queue_capacity));
    }
}

#[cfg(test)]
mod tests_pq_enqueue {
    use super::*;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::queue::config::QueueSrc;
    use opencl::wrapper::system::System;

    #[test]
    fn queue_is_empty() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_pq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_pq_config_by_id(0).unwrap();
        let pq = PriorityQueueHandle::new(config, system);

        let values: Vec<_> = (0..queue_capacity as cl_int).collect();
        let mut priorities = values.clone();
        priorities.reverse();

        let result = pq.enqueue(&values, &priorities).unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        assert_eq!(result, values);

        pq.print().unwrap();
    }

    #[test]
    fn queue_is_empty_2() {
        let queue_capacity = 1024;

        let mut queue_src = QueueSrc::new();
        queue_src.add_pq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_pq_config_by_id(0).unwrap();
        let pq = PriorityQueueHandle::new(config, system);

        let values: Vec<_> = (0..queue_capacity as cl_int).collect();
        let mut priorities = values.clone();
        priorities.reverse();

        let result = pq.enqueue(&values, &priorities).unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        assert_eq!(result_sorted, values);

        pq.print().unwrap();
    }

    #[test]
    fn queue_is_full() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_pq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_pq_config_by_id(0).unwrap();
        let pq = PriorityQueueHandle::new(config, system);

        let values = vec![1; queue_capacity];
        let priorities = vec![10; queue_capacity];

        let _result = pq.enqueue(&values, &priorities).unwrap();
        let result = pq.enqueue(&values, &priorities).unwrap();

        assert_eq!(result, vec![-1; queue_capacity]);

        pq.print().unwrap();
    }

    #[test]
    fn partially_full_queue() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_pq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_pq_config_by_id(0).unwrap();
        let pq = PriorityQueueHandle::new(config, system);

        let vec_a = vec![8; queue_capacity / 4];
        let vec_b = vec![3; queue_capacity / 4];

        let mut values = vec_a.clone();
        values.append(&mut vec_b.clone());

        let priorities = &values;

        let _ = pq.enqueue(&values, priorities).unwrap();

        let vec_c = vec![1; queue_capacity / 2];
        let vec_d = vec![2; queue_capacity / 2];

        let mut values = vec_c.clone();
        values.append(&mut vec_d.clone());

        let priorities = &values;

        let result = pq.enqueue(&values, priorities).unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        let mut expected_indices: Vec<cl_int> = vec![-1; queue_capacity / 2];
        expected_indices
            .append(&mut (((queue_capacity / 2) as cl_int)..queue_capacity as cl_int).collect());

        assert_eq!(result_sorted, expected_indices);

        let mut expected = vec_c.clone();
        expected.append(&mut vec_b.clone());
        expected.append(&mut vec_a.clone());

        let qs = pq.print().unwrap();
        assert_eq!(qs.values, expected);
        assert_eq!(qs.priorities, expected);
    }

    #[test]
    fn queue_without_enough_space() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_pq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_pq_config_by_id(0).unwrap();
        let pq = PriorityQueueHandle::new(config, system);

        let values: Vec<cl_int> = (0..(queue_capacity * 2) as cl_int).collect();
        let mut priorities = values.clone();
        priorities.reverse();

        let result = pq.enqueue(&values, &priorities).unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        let mut expected_indices: Vec<cl_int> = vec![-1; queue_capacity];
        expected_indices.append(&mut (0..queue_capacity as cl_int).collect());

        assert_eq!(result_sorted, expected_indices);

        pq.print().unwrap();
    }
}

#[cfg(test)]
mod tests_pq_dequeue {
    use super::*;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::queue::config::QueueSrc;
    use opencl::wrapper::system::System;

    #[test]
    fn queue_is_empty() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_pq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_pq_config_by_id(0).unwrap();
        let pq = PriorityQueueHandle::new(config, system);

        let values = pq.dequeue(queue_capacity).unwrap();

        assert_eq!(values, vec![-1; queue_capacity]);

        pq.print().unwrap();
    }

    #[test]
    fn queue_is_full() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_pq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_pq_config_by_id(0).unwrap();
        let pq = PriorityQueueHandle::new(config, system);

        let input_values: Vec<_> = (0..queue_capacity as cl_int).collect();
        let mut input_priorities = input_values.clone();
        input_priorities.reverse();

        let _ = pq.enqueue(&input_values, &input_priorities).unwrap();

        let values = pq.dequeue(queue_capacity).unwrap();

        let mut values_ordered = values.clone();
        values_ordered.sort();

        assert_eq!(values_ordered, input_values);

        let _ = pq.print().unwrap();
    }

    #[test]
    fn queue_is_full_2() {
        let queue_capacity = 1024;

        let mut queue_src = QueueSrc::new();
        queue_src.add_pq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_pq_config_by_id(0).unwrap();
        let pq = PriorityQueueHandle::new(config, system);

        let input_values: Vec<_> = (0..queue_capacity as cl_int).collect();
        let mut input_priorities = input_values.clone();
        input_priorities.reverse();

        let _ = pq.enqueue(&input_values, &input_priorities).unwrap();

        let values = pq.dequeue(queue_capacity).unwrap();

        let mut values_ordered = values.clone();
        values_ordered.sort();

        assert_eq!(values_ordered, input_values);

        let _ = pq.print().unwrap();
    }

    #[test]
    fn queue_is_full_3() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_pq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_pq_config_by_id(0).unwrap();
        let pq = PriorityQueueHandle::new(config, system);

        let input_values: Vec<_> = (0..queue_capacity as cl_int).collect();
        let mut input_priorities = input_values.clone();
        input_priorities.reverse();

        let _ = pq.enqueue(&input_values, &input_priorities).unwrap();

        let values = pq.dequeue(queue_capacity * 2).unwrap();

        let mut values_ordered = values.clone();
        values_ordered.sort();

        let mut expected_values = vec![-1; queue_capacity];
        expected_values.append(&mut input_values.clone());

        assert_eq!(values_ordered, expected_values);

        let _ = pq.print().unwrap();
    }

    #[test]
    fn partially_full_queue() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_pq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_pq_config_by_id(0).unwrap();
        let pq = PriorityQueueHandle::new(config, system);

        let input_values: Vec<_> = (0..(queue_capacity / 2) as cl_int).collect();
        let mut input_priorities = input_values.clone();
        input_priorities.reverse();

        let _ = pq.enqueue(&input_values, &input_priorities).unwrap();

        let values = pq.dequeue(queue_capacity).unwrap();

        let mut values_ordered = values.clone();
        values_ordered.sort();

        let mut expected_values = vec![-1; queue_capacity / 2];
        expected_values.append(&mut input_values.clone());

        assert_eq!(values_ordered, expected_values);

        let _ = pq.print().unwrap();
    }

    #[test]
    fn queue_emptied() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_pq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_pq_config_by_id(0).unwrap();
        let pq = PriorityQueueHandle::new(config, system);

        let input_values = vec![1; queue_capacity];
        let input_priorities = vec![10; queue_capacity];

        let _ = pq.enqueue(&input_values, &input_priorities).unwrap();
        pq.dequeue(queue_capacity).unwrap();

        let values = pq.dequeue(queue_capacity).unwrap();

        assert_eq!(values, vec![-1; queue_capacity]);

        let _ = pq.print().unwrap();
    }
}

// TODO explain tests
#[cfg(test)]
mod tests_pq_examples {
    use super::*;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::queue::config::QueueSrc;
    use opencl::wrapper::system::System;

    // TODO rename tests

    #[test]
    fn case_1() {
        let queue_capacity = 32;

        let mut queue_src = QueueSrc::new();
        queue_src.add_pq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_pq_config_by_id(0).unwrap();
        let pq = PriorityQueueHandle::new(config, system);

        println!("enq 1");
        let input_values = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let input_priorities = vec![7, 7, 2, 3, 5, 8, 1, 1];
        let _ = pq.enqueue(&input_values, &input_priorities).unwrap();
        pq.print().unwrap();

        println!("deq 1");
        let values = pq.dequeue(4).unwrap();
        pq.print().unwrap();

        let mut values_ordered = values.clone();
        values_ordered.sort();

        assert_eq!(values_ordered, vec![1, 2, 5, 6]);

        println!("deq 2");
        let values = pq.dequeue(4).unwrap();
        pq.print().unwrap();

        let mut values_ordered = values.clone();
        values_ordered.sort();

        assert_eq!(values_ordered, vec![3, 4, 7, 8]);

        println!("enq 2");

        let input_values = vec![9, 10, 11, 12, 13, 14];
        let input_priorities = vec![1, 1, 6, 6, 4, 3];

        let _ = pq.enqueue(&input_values, &input_priorities).unwrap();
        pq.print().unwrap();

        println!("deq 3");
        let values = pq.dequeue(1).unwrap();
        pq.print().unwrap();

        assert_eq!(values, vec![11]);

        println!("deq 4");
        let values = pq.dequeue(1).unwrap();
        pq.print().unwrap();

        assert_eq!(values, vec![12]);

        println!("deq 5");
        let values = pq.dequeue(1).unwrap();
        pq.print().unwrap();

        assert_eq!(values, vec![13]);

        println!("deq 6");
        let values = pq.dequeue(1).unwrap();
        pq.print().unwrap();

        assert_eq!(values, vec![14]);

        println!("deq 7");
        let _values = pq.dequeue(3).unwrap();

        // TODO update assert
        // assert_eq!(values_ordered, vec![9, 10, -1]);

        let _ = pq.print().unwrap();
    }

    #[test]
    fn case_2() {
        let queue_capacity = 8;

        let mut queue_src = QueueSrc::new();
        queue_src.add_pq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_pq_config_by_id(0).unwrap();
        let pq = PriorityQueueHandle::new(config, system);

        let values = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let priorities = vec![1, 2, 3, 4, 5, 6, 7, 8];

        let _result = pq.enqueue(&values, &priorities).unwrap();
        // TODO assert result

        let qs = pq.print().unwrap();
        assert_eq!(qs.values, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(qs.priorities, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn case_3() {
        let queue_capacity = 8;

        let mut queue_src = QueueSrc::new();
        queue_src.add_pq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_pq_config_by_id(0).unwrap();
        let pq = PriorityQueueHandle::new(config, system);

        let values = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let priorities = vec![8, 7, 6, 5, 4, 3, 2, 1];

        let _result = pq.enqueue(&values, &priorities).unwrap();
        // TODO assert result

        let qs = pq.print().unwrap();
        assert_eq!(qs.values, vec![8, 7, 6, 5, 4, 3, 2, 1]);
        assert_eq!(qs.priorities, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn case_4() {
        let queue_capacity = 8;

        let mut queue_src = QueueSrc::new();
        queue_src.add_pq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_pq_config_by_id(0).unwrap();
        let pq = PriorityQueueHandle::new(config, system);

        let values = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let priorities = vec![1, 8, 2, 7, 3, 6, 4, 5];

        let _result = pq.enqueue(&values, &priorities).unwrap();
        // TODO assert result

        let qs = pq.print().unwrap();
        assert_eq!(qs.values, vec![1, 3, 5, 7, 8, 6, 4, 2]);
        assert_eq!(qs.priorities, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn case_5() {
        let queue_capacity = 8;

        let mut queue_src = QueueSrc::new();
        queue_src.add_pq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_pq_config_by_id(0).unwrap();
        let pq = PriorityQueueHandle::new(config, system);

        let values = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let priorities = vec![1, 8, 1, 2, 1, 5, 5, 5];

        let _result = pq.enqueue(&values, &priorities).unwrap();
        // TODO assert result

        let qs = pq.print().unwrap();
        // assert_eq!(qs.values, vec![5, 3, 1, 4, 8, 7, 6, 2]);
        assert_eq!(qs.priorities, vec![1, 1, 1, 2, 5, 5, 5, 8]);
    }

    #[test]
    fn case_6() {
        let queue_capacity = 8;

        let mut queue_src = QueueSrc::new();
        queue_src.add_pq(queue_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &queue_src.build()).unwrap();

        let config = queue_src.get_pq_config_by_id(0).unwrap();
        let pq = PriorityQueueHandle::new(config, system);

        let values = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let priorities = vec![1; 8];

        let _result = pq.enqueue(&values, &priorities).unwrap();
        // TODO assert result

        let qs = pq.print().unwrap();
        // assert_eq!(qs.values, vec![8, 7, 6, 5, 4, 3, 2, 1]);
        assert_eq!(qs.priorities, vec![1; 8]);
    }
}
