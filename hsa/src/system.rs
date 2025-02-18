use crate::error::{hsa_check, HsaResult};
use hsa_sys::bindings::{
    hsa_agent_get_info, hsa_agent_info_t_HSA_AGENT_INFO_DEVICE,
    hsa_agent_info_t_HSA_AGENT_INFO_NAME, hsa_agent_t, hsa_amd_agent_iterate_memory_pools,
    hsa_amd_memory_pool_get_info,
    hsa_amd_memory_pool_global_flag_s_HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED,
    hsa_amd_memory_pool_global_flag_s_HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT,
    hsa_amd_memory_pool_info_t_HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
    hsa_amd_memory_pool_info_t_HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
    hsa_amd_memory_pool_info_t_HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
    hsa_amd_memory_pool_info_t_HSA_AMD_MEMORY_POOL_INFO_SIZE, hsa_amd_memory_pool_t,
    hsa_amd_segment_t, hsa_amd_segment_t_HSA_AMD_SEGMENT_GLOBAL, hsa_device_type_t,
    hsa_device_type_t_HSA_DEVICE_TYPE_CPU, hsa_device_type_t_HSA_DEVICE_TYPE_GPU, hsa_init,
    hsa_iterate_agents, hsa_shut_down, hsa_status_t, hsa_status_t_HSA_STATUS_SUCCESS,
};
use std::ffi::CString;
use utilities::helper_functions::buf_u8_remove_zero_to_string;

#[derive(Debug, Clone)]
pub struct MemoryPool {
    pool: hsa_amd_memory_pool_t,
    fine: bool,
    kern_arg: bool,
    size: usize,
    granule: usize,
}

impl MemoryPool {
    pub fn get_hsa_amd_memory_pool_t(&self) -> hsa_amd_memory_pool_t {
        self.pool
    }
}

impl Default for MemoryPool {
    fn default() -> Self {
        Self {
            pool: hsa_amd_memory_pool_t { handle: 0 },
            fine: false,
            kern_arg: false,
            size: 0,
            granule: 0,
        }
    }
}

impl PartialEq for MemoryPool {
    fn eq(&self, other: &Self) -> bool {
        self.pool.handle.eq(&other.pool.handle)
    }
}

#[derive(Debug, Clone)]
pub struct Agent {
    pub device_type: hsa_device_type_t,
    pub name: String,
    agent: hsa_agent_t,
    pools: Vec<MemoryPool>,
    fine: u32,
    coarse: u32,
}

impl Agent {
    pub fn get_hsa_agent_t(&self) -> hsa_agent_t {
        self.agent
    }

    pub fn get_standard_pool(&self) -> Option<&MemoryPool> {
        self.pools.iter().find(|&x| !x.kern_arg)
    }

    pub fn get_kern_arg_pool(&self) -> Option<&MemoryPool> {
        self.pools.iter().find(|&x| x.kern_arg)
    }
}

impl PartialEq for Agent {
    fn eq(&self, other: &Self) -> bool {
        self.agent.handle.eq(&other.agent.handle)
    }
}

impl Default for Agent {
    fn default() -> Self {
        Self {
            device_type: 0,
            name: "".to_string(),
            agent: hsa_agent_t { handle: 0 },
            pools: vec![],
            fine: 0,
            coarse: 0,
        }
    }
}

unsafe extern "C" fn get_memory_pools(
    pool: hsa_amd_memory_pool_t,
    data: *mut std::os::raw::c_void,
) -> hsa_status_t {
    let agent = &mut *(data as *mut Agent);

    let mut segment: hsa_amd_segment_t = 0;

    let err = hsa_amd_memory_pool_get_info(
        pool,
        hsa_amd_memory_pool_info_t_HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
        &mut segment as *mut _ as *mut std::os::raw::c_void,
    );

    if err != hsa_status_t_HSA_STATUS_SUCCESS {
        return err;
    }

    if hsa_amd_segment_t_HSA_AMD_SEGMENT_GLOBAL != segment {
        return hsa_status_t_HSA_STATUS_SUCCESS;
    }

    let mut flags: u32 = 0;

    let err = hsa_amd_memory_pool_get_info(
        pool,
        hsa_amd_memory_pool_info_t_HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
        &mut flags as *mut _ as *mut std::os::raw::c_void,
    );

    if err != hsa_status_t_HSA_STATUS_SUCCESS {
        return err;
    }

    let mut mem_pool = MemoryPool {
        pool,
        ..Default::default()
    };

    mem_pool.fine = (flags
        & hsa_amd_memory_pool_global_flag_s_HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED)
        == 2;
    mem_pool.kern_arg = (flags
        & hsa_amd_memory_pool_global_flag_s_HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT)
        == 1;

    let err = hsa_amd_memory_pool_get_info(
        pool,
        hsa_amd_memory_pool_info_t_HSA_AMD_MEMORY_POOL_INFO_SIZE,
        &mut mem_pool.size as *mut _ as *mut std::os::raw::c_void,
    );
    if err != hsa_status_t_HSA_STATUS_SUCCESS {
        return err;
    }

    let err = hsa_amd_memory_pool_get_info(
        pool,
        hsa_amd_memory_pool_info_t_HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
        &mut mem_pool.granule as *mut _ as *mut std::os::raw::c_void,
    );
    if err != hsa_status_t_HSA_STATUS_SUCCESS {
        return err;
    }

    agent.pools.push(mem_pool);

    hsa_status_t_HSA_STATUS_SUCCESS
}

unsafe extern "C" fn get_agents(
    agent: hsa_agent_t,
    data: *mut std::os::raw::c_void,
) -> hsa_status_t {
    let system = &mut *(data as *mut System);

    let mut dev = Agent {
        agent,
        ..Default::default()
    };

    let err = hsa_agent_get_info(
        agent,
        hsa_agent_info_t_HSA_AGENT_INFO_DEVICE,
        &mut dev.device_type as *mut _ as *mut std::os::raw::c_void,
    );
    if err != hsa_status_t_HSA_STATUS_SUCCESS {
        return err;
    }

    let name = CString::new(vec![32; 63]).unwrap();

    let err = hsa_agent_get_info(
        agent,
        hsa_agent_info_t_HSA_AGENT_INFO_NAME,
        name.as_ptr() as *mut std::os::raw::c_void,
    );

    if err != hsa_status_t_HSA_STATUS_SUCCESS {
        return err;
    }

    dev.name = buf_u8_remove_zero_to_string(name.as_bytes()).unwrap();

    let err = hsa_amd_agent_iterate_memory_pools(
        agent,
        Some(get_memory_pools),
        &mut dev as *mut _ as *mut std::os::raw::c_void,
    );

    if err != hsa_status_t_HSA_STATUS_SUCCESS {
        return err;
    }

    if !dev.pools.is_empty() {
        for (i, pool) in dev.pools.iter().enumerate() {
            if pool.fine && pool.kern_arg && dev.fine == 0 {
                dev.fine = i as u32;
            }

            if pool.fine && !pool.kern_arg {
                dev.fine = i as u32;
            }

            if !pool.fine {
                dev.coarse = i as u32;
            }
        }

        system.agents.push(dev);
    }

    hsa_status_t_HSA_STATUS_SUCCESS
}

#[derive(Debug, PartialEq)]
pub struct System {
    agents: Vec<Agent>,
    kern_arg_pool: MemoryPool,
}

impl System {
    pub fn new() -> HsaResult<System> {
        let mut system = Self {
            agents: vec![],
            kern_arg_pool: MemoryPool::default(),
        };

        unsafe {
            let ret = hsa_init();
            hsa_check(ret)?;

            let ret = hsa_iterate_agents(
                Some(get_agents),
                &mut system as *mut _ as *mut std::os::raw::c_void,
            );
            hsa_check(ret)?;
        }

        let mut kern_arg_pool = MemoryPool::default();

        for agent in system.cpus() {
            for mem_pool in agent.pools.iter() {
                if mem_pool.fine && mem_pool.kern_arg {
                    kern_arg_pool = mem_pool.clone();
                    break;
                }
            }
        }

        system.kern_arg_pool = kern_arg_pool;

        Ok(system)
    }

    pub fn cpus(&self) -> Vec<&Agent> {
        self.agents
            .iter()
            .filter(|&x| x.device_type == hsa_device_type_t_HSA_DEVICE_TYPE_CPU)
            .collect()
    }

    pub fn gpus(&self) -> Vec<&Agent> {
        self.agents
            .iter()
            .filter(|&x| x.device_type == hsa_device_type_t_HSA_DEVICE_TYPE_GPU)
            .collect()
    }

    pub fn get_agents(&self) -> Vec<&Agent> {
        self.agents.iter().collect()
    }

    pub fn get_first_cpu(&self) -> Option<&Agent> {
        self.agents
            .iter()
            .find(|&x| x.device_type == hsa_device_type_t_HSA_DEVICE_TYPE_CPU)
    }

    pub fn get_first_gpu(&self) -> Option<&Agent> {
        self.agents
            .iter()
            .find(|&x| x.device_type == hsa_device_type_t_HSA_DEVICE_TYPE_GPU)
    }

    pub fn get_kern_arg_pool(&self) -> &MemoryPool {
        &self.kern_arg_pool
    }
}

impl Drop for System {
    fn drop(&mut self) {
        unsafe {
            let ret = hsa_shut_down();
            if ret != hsa_status_t_HSA_STATUS_SUCCESS {
                panic!("hsa_shut_down error: {:?}", ret);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hsa_system_create() {
        let system = System::new().unwrap();

        println!("{:#?}", system);
        // TODO assert
    }
}
