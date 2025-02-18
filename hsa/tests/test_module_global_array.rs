use hsa::code_object::{CodeObject, DispatchPacketOptions};
use hsa::memory::{memory_copy_async, HsaBuffer};
use hsa::queue::{atomic_set_packet_header, Queue};
use hsa::signal::Signal;
use hsa::system::System;
use hsa_sys::bindings::{
    hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM,
    hsa_packet_header_t_HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE,
    hsa_packet_header_t_HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE,
    hsa_packet_type_t_HSA_PACKET_TYPE_KERNEL_DISPATCH,
};
use std::env;
use std::path::Path;

const WORK_GROUP_SIZE_X: usize = 32;

#[derive(Debug)]
struct HsaModule<'a> {
    system: &'a System,
    signal: Signal,
    queue: Queue,
    code_object: CodeObject,
}

impl<'a> HsaModule<'a> {
    pub fn new<P: AsRef<Path>>(system: &'a System, file_path: P) -> Self {
        let gpu_agent = system.get_first_gpu().unwrap();

        let signal = Signal::new(1, 0).unwrap();

        let code_object = CodeObject::new(file_path, gpu_agent).unwrap();

        let queue = Queue::new(gpu_agent).unwrap();

        Self {
            system,
            signal,
            queue,
            code_object,
        }
    }

    pub fn global_array_put(&self, input: &[i32]) {
        // allocate_and_init_buffers
        let cpu_agent = self.system.get_first_cpu().unwrap();
        let gpu_agent = self.system.get_first_gpu().unwrap();

        let cpu_mem_pool = cpu_agent.get_standard_pool().unwrap();
        // let gpu_mem_pool = gpu_agent.get_standard_pool().unwrap();

        let agents = &[cpu_agent, gpu_agent];

        let input_buf: HsaBuffer<i32> =
            HsaBuffer::new(WORK_GROUP_SIZE_X, cpu_mem_pool, agents).unwrap();

        // Setup the kernel args
        #[repr(C)]
        #[derive(Debug)]
        struct LocalArgsT {
            input: *mut std::os::raw::c_void,
        }

        let mut local_args = LocalArgsT {
            input: input_buf.get_mem_ptr(),
        };

        let kernel = self
            .code_object
            .get_kernel("_Z16global_array_putPi.kd")
            .unwrap();

        let mut aql_packet = kernel.populate_aql_packet(
            DispatchPacketOptions {
                workgroup_size_x: WORK_GROUP_SIZE_X as u16,
                workgroup_size_y: 0,
                workgroup_size_z: 0,
                grid_size_x: WORK_GROUP_SIZE_X as u32,
                grid_size_y: 0,
                grid_size_z: 0,
            },
            self.signal.get_hsa_signal_t(),
        );

        let kern_arg_pool = self.system.get_kern_arg_pool();

        unsafe {
            let src_ptr = input.as_ptr() as *mut std::os::raw::c_void;
            let input_size = input.len() * std::mem::size_of::<u32>();

            memory_copy_async(
                input_buf.get_mem_ptr(),
                cpu_agent.get_hsa_agent_t(),
                src_ptr,
                cpu_agent.get_hsa_agent_t(),
                input_size,
            )
            .expect("memory_copy_async error");

            aql_packet
                .alloc_and_set_kern_args(
                    &mut local_args as *mut _ as *mut std::os::raw::c_void,
                    std::mem::size_of_val(&local_args),
                    kern_arg_pool,
                    agents,
                )
                .unwrap();

            let write_aql_packet = self
                .queue
                .write_aql_packet(aql_packet.get_hsa_kernel_dispatch_packet_t());

            let mut aql_header = hsa_packet_type_t_HSA_PACKET_TYPE_KERNEL_DISPATCH;
            aql_header |= hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM
                << hsa_packet_header_t_HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
            aql_header |= hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM
                << hsa_packet_header_t_HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;

            atomic_set_packet_header(
                aql_header as u16,
                aql_packet.get_hsa_kernel_dispatch_packet_t().setup,
                write_aql_packet.packet_ptr,
            );

            self.queue
                .dispatch(write_aql_packet, self.signal.get_hsa_signal_t());

            let slice = std::slice::from_raw_parts(input_buf.get_mem_ptr() as *mut i32, 32);
            println!("input {:?}", slice);
        }
    }

    pub fn global_array_get(&self) -> Vec<i32> {
        // allocate_and_init_buffers
        let cpu_agent = self.system.get_first_cpu().unwrap();
        let gpu_agent = self.system.get_first_gpu().unwrap();

        let cpu_mem_pool = cpu_agent.get_standard_pool().unwrap();

        let agents = &[cpu_agent, gpu_agent];

        let output_buf: HsaBuffer<i32> =
            HsaBuffer::new(WORK_GROUP_SIZE_X, cpu_mem_pool, agents).unwrap();

        // Setup the kernel args
        #[repr(C)]
        #[derive(Debug)]
        struct LocalArgsT {
            output: *mut std::os::raw::c_void,
        }

        let mut local_args = LocalArgsT {
            output: output_buf.get_mem_ptr(),
        };

        let kernel = self
            .code_object
            .get_kernel("_Z16global_array_getPi.kd")
            .unwrap();

        let mut aql_packet = kernel.populate_aql_packet(
            DispatchPacketOptions {
                workgroup_size_x: WORK_GROUP_SIZE_X as u16,
                workgroup_size_y: 0,
                workgroup_size_z: 0,
                grid_size_x: WORK_GROUP_SIZE_X as u32,
                grid_size_y: 0,
                grid_size_z: 0,
            },
            self.signal.get_hsa_signal_t(),
        );

        let kern_arg_pool = self.system.get_kern_arg_pool();

        let mut output = vec![0; WORK_GROUP_SIZE_X];
        unsafe {
            aql_packet
                .alloc_and_set_kern_args(
                    &mut local_args as *mut _ as *mut std::os::raw::c_void,
                    std::mem::size_of_val(&local_args),
                    kern_arg_pool,
                    agents,
                )
                .unwrap();

            let write_aql_packet = self
                .queue
                .write_aql_packet(aql_packet.get_hsa_kernel_dispatch_packet_t());

            let mut aql_header = hsa_packet_type_t_HSA_PACKET_TYPE_KERNEL_DISPATCH;
            aql_header |= hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM
                << hsa_packet_header_t_HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
            aql_header |= hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM
                << hsa_packet_header_t_HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;

            atomic_set_packet_header(
                aql_header as u16,
                aql_packet.get_hsa_kernel_dispatch_packet_t().setup,
                write_aql_packet.packet_ptr,
            );

            self.queue
                .dispatch(write_aql_packet, self.signal.get_hsa_signal_t());

            let slice = std::slice::from_raw_parts(output_buf.get_mem_ptr() as *mut i32, 32);
            println!("output {:?}", slice);

            output.copy_from_slice(slice);
        }

        output
    }

    pub fn global_array_increase(&self) {
        let kernel = self
            .code_object
            .get_kernel("_Z21global_array_increasev.kd")
            .unwrap();

        let aql_packet = kernel.populate_aql_packet(
            DispatchPacketOptions {
                workgroup_size_x: WORK_GROUP_SIZE_X as u16,
                workgroup_size_y: 0,
                workgroup_size_z: 0,
                grid_size_x: WORK_GROUP_SIZE_X as u32,
                grid_size_y: 0,
                grid_size_z: 0,
            },
            self.signal.get_hsa_signal_t(),
        );

        unsafe {
            let write_aql_packet = self
                .queue
                .write_aql_packet(aql_packet.get_hsa_kernel_dispatch_packet_t());

            let mut aql_header = hsa_packet_type_t_HSA_PACKET_TYPE_KERNEL_DISPATCH;
            aql_header |= hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM
                << hsa_packet_header_t_HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
            aql_header |= hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM
                << hsa_packet_header_t_HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;

            atomic_set_packet_header(
                aql_header as u16,
                aql_packet.get_hsa_kernel_dispatch_packet_t().setup,
                write_aql_packet.packet_ptr,
            );

            self.queue
                .dispatch(write_aql_packet, self.signal.get_hsa_signal_t());
        }
    }

    pub fn global_array_insert(&self, input: &[i32]) -> Vec<i32> {
        // allocate_and_init_buffers
        let cpu_agent = self.system.get_first_cpu().unwrap();
        let gpu_agent = self.system.get_first_gpu().unwrap();

        let cpu_mem_pool = cpu_agent.get_standard_pool().unwrap();

        let agents = &[cpu_agent, gpu_agent];

        let input_buf: HsaBuffer<i32> =
            HsaBuffer::new(WORK_GROUP_SIZE_X, cpu_mem_pool, agents).unwrap();
        let output_buf: HsaBuffer<i32> =
            HsaBuffer::new(WORK_GROUP_SIZE_X, cpu_mem_pool, agents).unwrap();

        // Setup the kernel args
        #[repr(C)]
        #[derive(Debug)]
        struct LocalArgsT {
            input: *mut std::os::raw::c_void,
            output: *mut std::os::raw::c_void,
        }

        let mut local_args = LocalArgsT {
            input: input_buf.get_mem_ptr(),
            output: output_buf.get_mem_ptr(),
        };

        let kernel = self
            .code_object
            .get_kernel("_Z19global_array_insertPiS_.kd")
            .unwrap();

        let mut aql_packet = kernel.populate_aql_packet(
            DispatchPacketOptions {
                workgroup_size_x: WORK_GROUP_SIZE_X as u16,
                workgroup_size_y: 0,
                workgroup_size_z: 0,
                grid_size_x: WORK_GROUP_SIZE_X as u32,
                grid_size_y: 0,
                grid_size_z: 0,
            },
            self.signal.get_hsa_signal_t(),
        );

        let kern_arg_pool = self.system.get_kern_arg_pool();

        let mut output = vec![0; WORK_GROUP_SIZE_X];

        unsafe {
            let src_ptr = input.as_ptr() as *mut std::os::raw::c_void;
            let input_size = input.len() * std::mem::size_of::<u32>();

            memory_copy_async(
                input_buf.get_mem_ptr(),
                cpu_agent.get_hsa_agent_t(),
                src_ptr,
                cpu_agent.get_hsa_agent_t(),
                input_size,
            )
            .expect("memory_copy_async error");

            aql_packet
                .alloc_and_set_kern_args(
                    &mut local_args as *mut _ as *mut std::os::raw::c_void,
                    std::mem::size_of_val(&local_args),
                    kern_arg_pool,
                    agents,
                )
                .unwrap();

            let write_aql_packet = self
                .queue
                .write_aql_packet(aql_packet.get_hsa_kernel_dispatch_packet_t());

            let mut aql_header = hsa_packet_type_t_HSA_PACKET_TYPE_KERNEL_DISPATCH;
            aql_header |= hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM
                << hsa_packet_header_t_HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
            aql_header |= hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM
                << hsa_packet_header_t_HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;

            atomic_set_packet_header(
                aql_header as u16,
                aql_packet.get_hsa_kernel_dispatch_packet_t().setup,
                write_aql_packet.packet_ptr,
            );

            self.queue
                .dispatch(write_aql_packet, self.signal.get_hsa_signal_t());

            let slice = std::slice::from_raw_parts(input_buf.get_mem_ptr() as *mut i32, 32);
            println!("input {:?}", slice);

            let slice = std::slice::from_raw_parts(output_buf.get_mem_ptr() as *mut i32, 32);
            println!("output {:?}", slice);

            output.copy_from_slice(slice);
        }

        output
    }
}

#[test]
fn test_module_global_array() {
    let module_file_name = "global_array-hip-amdgcn-amd-amdhsa_gfx1032.o";

    let mut module_path = env::current_dir().unwrap();
    module_path.push(module_file_name);
    println!("module_path: {:?}", module_path);

    let system = System::new().unwrap();

    let module_1 = HsaModule::new(&system, &module_path);

    let input_1 = vec![2; WORK_GROUP_SIZE_X];
    module_1.global_array_put(&input_1);

    let input_2: Vec<i32> = (0..WORK_GROUP_SIZE_X as i32).collect();
    let output = module_1.global_array_insert(&input_2);
    assert_eq!(output, input_1);

    let input_3 = vec![3; WORK_GROUP_SIZE_X];
    let output = module_1.global_array_insert(&input_3);
    assert_eq!(output, input_2);

    let output = module_1.global_array_get();
    assert_eq!(output, input_3);

    module_1.global_array_increase();
    module_1.global_array_increase();

    let output = module_1.global_array_get();
    assert_eq!(output, vec![5; WORK_GROUP_SIZE_X]);

    let module_2 = HsaModule::new(&system, module_path);

    let output = module_2.global_array_get();
    assert_eq!(output, vec![0; WORK_GROUP_SIZE_X]);
}
