#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(clippy::useless_transmute)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::ptr_offset_with_cast)]
#![allow(clippy::missing_safety_doc)]
pub mod bindings;
pub mod utils;

#[cfg(test)]
mod tests {
    use crate::bindings::{
        hsa_agent_get_info, hsa_agent_info_t_HSA_AGENT_INFO_DEVICE,
        hsa_agent_info_t_HSA_AGENT_INFO_NAME, hsa_agent_t, hsa_device_type_t, hsa_init,
        hsa_iterate_agents, hsa_shut_down, hsa_status_t, hsa_status_t_HSA_STATUS_SUCCESS,
    };
    use crate::utils::get_device_type_str;
    use std::ffi::CString;
    use utilities::helper_functions::buf_u8_remove_zero_to_string;

    #[repr(C)]
    #[derive(Debug)]
    pub struct HsaAgents {
        agents: Vec<hsa_agent_t>,
    }

    unsafe extern "C" fn get_kernel_agents(
        agent: hsa_agent_t,
        data: *mut std::os::raw::c_void,
    ) -> hsa_status_t {
        let payload = &mut *(data as *mut HsaAgents);
        payload.agents.push(agent);

        hsa_status_t_HSA_STATUS_SUCCESS
    }

    unsafe fn agent_get_info(agent: hsa_agent_t) -> (String, hsa_device_type_t) {
        // TODO update parameter name
        let name = CString::new(vec![32; 63]).unwrap();
        let mut device_type: hsa_device_type_t = hsa_device_type_t::default();

        let ret = hsa_agent_get_info(
            agent,
            hsa_agent_info_t_HSA_AGENT_INFO_NAME,
            name.as_ptr() as *mut std::os::raw::c_void,
        );
        if ret != hsa_status_t_HSA_STATUS_SUCCESS {
            panic!("hsa_agent_get_info failed with {}", ret);
        }

        let ret = hsa_agent_get_info(
            agent,
            hsa_agent_info_t_HSA_AGENT_INFO_DEVICE,
            &mut device_type as *mut _ as *mut std::os::raw::c_void,
        );
        if ret != hsa_status_t_HSA_STATUS_SUCCESS {
            panic!("hsa_agent_get_info failed with {}", ret);
        }

        (
            buf_u8_remove_zero_to_string(name.as_bytes()).unwrap(),
            device_type,
        )
    }

    #[test]
    fn test_example() {
        let ret = unsafe { hsa_init() };
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        let mut hsa_agents = HsaAgents { agents: Vec::new() };

        //  iterate agents
        let ret = unsafe {
            hsa_iterate_agents(
                Some(get_kernel_agents),
                &mut hsa_agents as *mut _ as *mut std::os::raw::c_void,
            )
        };
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);

        for agent in hsa_agents.agents.into_iter() {
            let (name, device_type) = unsafe { agent_get_info(agent) };

            println!(
                "Agent - Name: {:#?}, DEVICE_TYPE: {}",
                name,
                get_device_type_str(device_type)
            );
        }

        let ret = unsafe { hsa_shut_down() };
        assert_eq!(ret, hsa_status_t_HSA_STATUS_SUCCESS);
    }
}
