use crate::config::DEBUG_MODE;
use crate::error::OpenClResult;
use crate::stack::config::StackConfig;
use crate::stack::kernel::name::{
    get_stack_kernel_name, READ_ON_STACK, STACK_DEBUG, STACK_RESET, WRITE_TO_STACK,
};
use opencl::opencl_sys::bindings::{cl_int, cl_uint};
use opencl::wrapper::system::OpenclCommonOperation;

#[derive(Debug, PartialEq)]
pub struct StackSnapshot {
    pub top: cl_int,
    pub items: Vec<cl_int>,
}

impl StackSnapshot {
    pub fn new(top: cl_int, items: Vec<cl_int>) -> Self {
        Self { top, items }
    }

    pub fn create_empty(capacity: usize) -> Self {
        Self::new(-1, vec![0; capacity])
    }
}

#[derive(Debug)]
pub struct StackHandle<T: OpenclCommonOperation> {
    config: StackConfig,
    system: T,
}

impl<T: OpenclCommonOperation> StackHandle<T> {
    pub fn new(config: &StackConfig, system: T) -> Self {
        Self {
            config: config.clone(),
            system,
        }
    }

    pub fn get_id(&self) -> usize {
        self.config.id
    }

    pub fn debug(&self) -> OpenClResult<StackSnapshot> {
        let global_work_size = self.config.capacity;
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let meta_output_capacity = 1;

        let output_buf = self.system.create_output_buffer(global_work_size)?;
        let meta_buf = self.system.create_output_buffer(meta_output_capacity)?;

        let kernel_name = get_stack_kernel_name(STACK_DEBUG, self.get_id());
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

        Ok(StackSnapshot {
            top: meta_output[0],
            items: output,
        })
    }

    pub fn print(&self) -> OpenClResult<StackSnapshot> {
        let sn = self.debug()?;
        println!("{sn:?}");
        Ok(sn)
    }

    pub fn reset(&self) -> OpenClResult<()> {
        let global_work_size = self.config.capacity;
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let kernel_name = get_stack_kernel_name(STACK_RESET, self.get_id());
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

    pub fn push(&self, input: &[cl_int]) -> OpenClResult<Vec<cl_int>> {
        let global_work_size = input.len();
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let input_buf = self.system.blocking_prepare_input_buffer(input)?;
        let output_buf = self.system.create_output_buffer(global_work_size)?;

        let mut kernel = self.system.create_kernel(WRITE_TO_STACK)?;

        let stack_id = self.get_id() as cl_uint;

        unsafe {
            kernel.set_arg(&stack_id)?;
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
            println!("push output {output:?}");
        }

        Ok(output)
    }

    pub fn pop(&self, take: usize) -> OpenClResult<Vec<cl_int>> {
        let global_work_size = take;
        let local_work_size = self
            .system
            .first_device_check_local_work_size(global_work_size);

        let output_buf = self.system.create_output_buffer(global_work_size)?;

        let mut kernel = self.system.create_kernel(READ_ON_STACK)?;

        let stack_id = self.get_id() as cl_uint;

        unsafe {
            kernel.set_arg(&stack_id)?;
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
            println!("pop output {output:?}");
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests_stack_debug {
    use super::*;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::stack::config::StackSrc;
    use opencl::wrapper::system::System;

    #[test]
    fn stack_is_empty() {
        let stack_capacity = 32;

        let mut stack_src = StackSrc::new();
        stack_src.add(stack_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &stack_src.build()).unwrap();

        let config = stack_src.get_config_by_id(0).unwrap();
        let st = StackHandle::new(config, system);

        let st_sn = st.print().unwrap();

        assert_eq!(st_sn, StackSnapshot::create_empty(stack_capacity));
    }

    #[test]
    fn stack_is_full() {
        let stack_capacity = 32;

        let mut stack_src = StackSrc::new();
        stack_src.add(stack_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &stack_src.build()).unwrap();

        let config = stack_src.get_config_by_id(0).unwrap();
        let st = StackHandle::new(config, system);

        let input: Vec<cl_int> = (0..stack_capacity as cl_int).collect();
        let _ = st.push(&input).unwrap();
        let st_sn = st.print().unwrap();

        assert_eq!(st_sn.top, (stack_capacity as cl_int) - 1);
        assert_eq!(st_sn.items, input);
    }

    #[test]
    fn partially_full_stack() {
        let stack_capacity = 32;

        let mut stack_src = StackSrc::new();
        stack_src.add(stack_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &stack_src.build()).unwrap();

        let config = stack_src.get_config_by_id(0).unwrap();
        let st = StackHandle::new(config, system);

        let input: Vec<cl_int> = (0..(stack_capacity / 2) as cl_int).collect();
        let _ = st.push(&input).unwrap();

        let st_sn = st.print().unwrap();

        let mut expected = input.clone();
        expected.append(&mut vec![0; stack_capacity / 2]);

        assert_eq!(st_sn.top, (stack_capacity as cl_int / 2) - 1);
        assert_eq!(st_sn.items, expected);
    }

    #[test]
    fn overflowing_stack() {
        let stack_capacity = 32;

        let mut stack_src = StackSrc::new();
        stack_src.add(stack_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &stack_src.build()).unwrap();

        let config = stack_src.get_config_by_id(0).unwrap();
        let st = StackHandle::new(config, system);

        let input: Vec<cl_int> = (0..(stack_capacity * 2) as cl_int).collect();
        let _st_in = st.push(&input).unwrap();
        let _st_out = st.pop(stack_capacity * 2).unwrap();

        let q_s = st.print().unwrap();

        assert_eq!(q_s.top, -1);
        assert_eq!(q_s.items, &input[0..stack_capacity]);
    }
}

#[cfg(test)]
mod tests_stack_reset {
    use super::*;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::stack::config::StackSrc;
    use opencl::wrapper::system::System;

    #[test]
    fn stack_is_empty() {
        let stack_capacity = 32;

        let mut stack_src = StackSrc::new();
        stack_src.add(stack_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &stack_src.build()).unwrap();

        let config = stack_src.get_config_by_id(0).unwrap();
        let st = StackHandle::new(config, system);

        st.reset().unwrap();
        let snapshot = st.print().unwrap();

        assert_eq!(snapshot, StackSnapshot::create_empty(stack_capacity));
    }

    #[test]
    fn stack_is_full() {
        let stack_capacity = 32;

        let mut stack_src = StackSrc::new();
        stack_src.add(stack_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &stack_src.build()).unwrap();

        let config = stack_src.get_config_by_id(0).unwrap();
        let st = StackHandle::new(config, system);

        let input = vec![10; stack_capacity];
        let _ = st.push(&input).unwrap();

        st.reset().unwrap();
        let snapshot = st.print().unwrap();

        assert_eq!(snapshot, StackSnapshot::create_empty(stack_capacity));
    }
}

#[cfg(test)]
mod tests_stack_push {
    use super::*;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::stack::config::StackSrc;
    use opencl::wrapper::system::System;

    #[test]
    fn stack_is_empty() {
        let stack_capacity = 32;

        let mut stack_src = StackSrc::new();
        stack_src.add(stack_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &stack_src.build()).unwrap();

        let config = stack_src.get_config_by_id(0).unwrap();
        let st = StackHandle::new(config, system);

        let input: Vec<cl_int> = (0..stack_capacity as cl_int).collect();
        let result = st.push(&input).unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        assert_eq!(result_sorted, input);

        st.print().unwrap();
    }

    // FIXME rename test
    #[test]
    fn stack_is_empty_2() {
        let stack_capacity = 1024;

        let mut stack_src = StackSrc::new();
        stack_src.add(stack_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &stack_src.build()).unwrap();

        let config = stack_src.get_config_by_id(0).unwrap();
        let st = StackHandle::new(config, system);

        let input: Vec<cl_int> = (0..stack_capacity as cl_int).collect();
        let result = st.push(&input).unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        let expected_indices: Vec<i32> = (0..stack_capacity).map(|x| x as cl_int).collect();
        assert_eq!(result_sorted, expected_indices);

        st.print().unwrap();
    }

    // FIXME rename test, the test consists of inserting data into the stack and there is space left
    #[test]
    fn stack_is_empty_3() {
        let stack_capacity = 32;

        let mut stack_src = StackSrc::new();
        stack_src.add(stack_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &stack_src.build()).unwrap();

        let config = stack_src.get_config_by_id(0).unwrap();
        let st = StackHandle::new(config, system);

        let input: Vec<cl_int> = (0..(stack_capacity / 2) as cl_int).collect();

        let result = st.push(&input).unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        let expected_indices: Vec<i32> = (0..(stack_capacity / 2) as cl_int).collect();
        assert_eq!(result_sorted, expected_indices);

        st.print().unwrap();
    }

    #[test]
    fn stack_is_full() {
        let stack_capacity = 32;

        let mut stack_src = StackSrc::new();
        stack_src.add(stack_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &stack_src.build()).unwrap();

        let config = stack_src.get_config_by_id(0).unwrap();
        let st = StackHandle::new(config, system);

        let input: Vec<cl_int> = (0..stack_capacity as cl_int).collect();
        let input_b: Vec<cl_int> = input.iter().map(|x| x + stack_capacity as cl_int).collect();

        let _ = st.push(&input).unwrap();
        let result = st.push(&input_b).unwrap();

        assert_eq!(result, vec![-1; stack_capacity]);

        st.print().unwrap();
    }

    #[test]
    fn stack_without_enough_space() {
        let stack_capacity = 512;

        let mut stack_src = StackSrc::new();
        stack_src.add(stack_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &stack_src.build()).unwrap();

        let config = stack_src.get_config_by_id(0).unwrap();
        let st = StackHandle::new(config, system);

        let input: Vec<cl_int> = (0..(stack_capacity * 4) as cl_int).collect();

        let result = st.push(&input).unwrap();

        assert_eq!(
            result.iter().filter(|&&x| x < 0).count(),
            stack_capacity * 3
        );
        assert_eq!(result.iter().filter(|&&x| x >= 0).count(), stack_capacity);

        st.print().unwrap();
    }
}

#[cfg(test)]
mod tests_stack_pop {
    use super::*;
    use crate::config::DEFAULT_DEVICE_INDEX;
    use crate::stack::config::StackSrc;
    use opencl::wrapper::system::System;

    #[test]
    fn stack_is_empty() {
        let stack_capacity = 32;

        let mut stack_src = StackSrc::new();
        stack_src.add(stack_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &stack_src.build()).unwrap();

        let config = stack_src.get_config_by_id(0).unwrap();
        let st = StackHandle::new(config, system);

        let result = st.pop(stack_capacity).unwrap();

        assert_eq!(result, vec![-1; stack_capacity]);

        st.print().unwrap();
    }

    #[test]
    fn stack_is_full() {
        let stack_capacity = 32;

        let mut stack_src = StackSrc::new();
        stack_src.add(stack_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &stack_src.build()).unwrap();

        let config = stack_src.get_config_by_id(0).unwrap();
        let st = StackHandle::new(config, system);

        let input: Vec<cl_int> = (0..stack_capacity as cl_int).collect();
        let _ = st.push(&input).unwrap();

        let result = st.pop(stack_capacity).unwrap();
        let mut result_sorted = result.clone();
        result_sorted.sort();

        assert_eq!(result_sorted, input);

        st.print().unwrap();
    }

    #[test]
    fn stack_is_full_2() {
        let stack_capacity = 1024;

        let mut stack_src = StackSrc::new();
        stack_src.add(stack_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &stack_src.build()).unwrap();

        let config = stack_src.get_config_by_id(0).unwrap();
        let st = StackHandle::new(config, system);

        let input: Vec<i32> = (0..stack_capacity as cl_int).collect();
        let _ = st.push(&input).unwrap();
        let result = st.pop(stack_capacity).unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();
        assert_eq!(result_sorted, input);

        st.print().unwrap();
    }

    #[test]
    fn partially_full_stack() {
        let stack_capacity = 64;

        let mut stack_src = StackSrc::new();
        stack_src.add(stack_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &stack_src.build()).unwrap();

        let config = stack_src.get_config_by_id(0).unwrap();
        let st = StackHandle::new(config, system);

        let input: Vec<cl_int> = (0..(stack_capacity / 2) as cl_int).collect();
        let _ = st.push(&input).unwrap();

        let result = st.pop(stack_capacity).unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        let mut expected = vec![-1; stack_capacity / 2];
        expected.append(&mut input.clone());

        assert_eq!(result_sorted, expected);

        st.print().unwrap();
    }

    // FIXME rename tests - The test consists of removing data from the stack and not leaving it empty
    #[test]
    fn simple_case() {
        let stack_capacity = 128;

        let mut stack_src = StackSrc::new();
        stack_src.add(stack_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &stack_src.build()).unwrap();

        let config = stack_src.get_config_by_id(0).unwrap();
        let st = StackHandle::new(config, system);

        let input: Vec<i32> = (0..stack_capacity as cl_int).collect();
        let _ = st.push(&input).unwrap();

        let _result = st.pop(stack_capacity / 2).unwrap();

        // FIXME assert result
        st.print().unwrap();
    }

    // FIXME rename tests - The test consists of multiple calls to the stack until it is empty
    #[test]
    fn stack_emptied() {
        let stack_capacity = 32;

        let mut stack_src = StackSrc::new();
        stack_src.add(stack_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &stack_src.build()).unwrap();

        let config = stack_src.get_config_by_id(0).unwrap();
        let st = StackHandle::new(config, system);

        let input: Vec<cl_int> = (0..stack_capacity as cl_int).collect();
        let _ = st.push(&input).unwrap();
        let _ = st.pop(stack_capacity).unwrap();

        let result = st.pop(stack_capacity).unwrap();
        assert_eq!(result, vec![-1; 32]);

        let result = st.pop(stack_capacity).unwrap();
        assert_eq!(result, vec![-1; 32]);

        st.print().unwrap();
    }

    // FIXME rename tests - The test consists of multiple calls to the stack until it is empty
    #[test]
    fn stack_emptied_2() {
        let stack_capacity = 64;

        let mut stack_src = StackSrc::new();
        stack_src.add(stack_capacity);

        let system = System::new(DEFAULT_DEVICE_INDEX, &stack_src.build()).unwrap();

        let config = stack_src.get_config_by_id(0).unwrap();
        let st = StackHandle::new(config, system);

        let input: Vec<i32> = (0..stack_capacity as cl_int).collect();
        let _ = st.push(&input).unwrap();

        let result = st.pop(stack_capacity * 2).unwrap();

        let mut result_sorted = result.clone();
        result_sorted.sort();

        let mut expected = vec![-1; stack_capacity];
        expected.append(&mut input.clone());

        assert_eq!(result_sorted, expected);

        let result = st.pop(stack_capacity).unwrap();
        assert_eq!(result, vec![-1; stack_capacity]);

        st.print().unwrap();
    }
}
