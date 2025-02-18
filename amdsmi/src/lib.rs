// FIXME remove not_unsafe_ptr_arg_deref
#![allow(clippy::not_unsafe_ptr_arg_deref)]

//! # amdsmi_sys
//!
//! ...
//!
//! https://github.com/ROCm/amdsmi
//!

use crate::error::{chk_amd_smi_ret, AmdSmiResult};
use amdsmi_sys::bindings::{
    amdsmi_asic_info_t, amdsmi_bdf_t, amdsmi_board_info_t, amdsmi_clk_info_t, amdsmi_clk_type_t,
    amdsmi_dpm_policy_t, amdsmi_driver_info_t, amdsmi_engine_usage_t, amdsmi_error_count_t,
    amdsmi_fw_info_t, amdsmi_get_clock_info, amdsmi_get_fw_info, amdsmi_get_gpu_activity,
    amdsmi_get_gpu_asic_info, amdsmi_get_gpu_bad_page_info, amdsmi_get_gpu_board_info,
    amdsmi_get_gpu_cache_info, amdsmi_get_gpu_device_bdf, amdsmi_get_gpu_device_uuid,
    amdsmi_get_gpu_driver_info, amdsmi_get_gpu_metrics_info, amdsmi_get_gpu_process_list,
    amdsmi_get_gpu_ras_block_features_enabled, amdsmi_get_gpu_ras_feature_info,
    amdsmi_get_gpu_total_ecc_count, amdsmi_get_gpu_vbios_info, amdsmi_get_gpu_vram_info,
    amdsmi_get_gpu_vram_usage, amdsmi_get_lib_version, amdsmi_get_link_topology_nearest,
    amdsmi_get_pcie_info, amdsmi_get_power_cap_info, amdsmi_get_power_info,
    amdsmi_get_processor_handle_from_bdf, amdsmi_get_processor_handles, amdsmi_get_processor_type,
    amdsmi_get_soc_pstate, amdsmi_get_socket_handles, amdsmi_get_socket_info,
    amdsmi_get_temp_metric, amdsmi_gpu_block_t, amdsmi_gpu_cache_info_t, amdsmi_gpu_metrics_t,
    amdsmi_init, amdsmi_init_flags_t_AMDSMI_INIT_AMD_GPUS, amdsmi_link_type_t, amdsmi_pcie_info_t,
    amdsmi_power_cap_info_t, amdsmi_power_info_t, amdsmi_proc_info_t,
    amdsmi_proc_info_t_engine_usage_, amdsmi_proc_info_t_memory_usage_, amdsmi_process_handle_t,
    amdsmi_processor_handle, amdsmi_ras_err_state_t,
    amdsmi_ras_err_state_t_AMDSMI_RAS_ERR_STATE_NONE, amdsmi_ras_feature_t,
    amdsmi_retired_page_record_t, amdsmi_shut_down, amdsmi_socket_handle,
    amdsmi_status_t_AMDSMI_STATUS_SUCCESS, amdsmi_temperature_metric_t, amdsmi_temperature_type_t,
    amdsmi_topology_nearest_t, amdsmi_vbios_info_t, amdsmi_version_t, amdsmi_vram_info_t,
    amdsmi_vram_usage_t, processor_type_t, processor_type_t_AMDSMI_PROCESSOR_TYPE_UNKNOWN,
    AMDSMI_GPU_UUID_SIZE,
};

pub mod error;
pub mod utils;

///
/// * only gpu device types
///
///
/// ```rust
///
/// use amdsmi::AmdSmi;
///
/// let amd_smi = AmdSmi::new(); // amdsmi_init(AMDSMI_INIT_AMD_GPUS)
///
/// drop(amd_smi); // amdsmi_shut_down()
///
/// ```
#[derive(Debug)]
pub struct AmdSmi;

impl AmdSmi {
    pub fn new() -> AmdSmiResult<Self> {
        unsafe {
            amdsmi_init(amdsmi_init_flags_t_AMDSMI_INIT_AMD_GPUS as u64);
        }
        Ok(Self)
    }

    pub fn get_version(&self) -> AmdSmiResult<amdsmi_version_t> {
        let mut version = std::mem::MaybeUninit::<amdsmi_version_t>::uninit();

        unsafe {
            let ret = amdsmi_get_lib_version(version.as_mut_ptr());
            chk_amd_smi_ret(ret)?;
        };

        let version = unsafe { version.assume_init() };

        Ok(version)
    }

    pub fn get_socket_handles_count(&self) -> AmdSmiResult<u32> {
        let mut socket_count = 0;

        unsafe {
            let ret = amdsmi_get_socket_handles(&mut socket_count, std::ptr::null_mut());
            chk_amd_smi_ret(ret)?;
        }

        Ok(socket_count)
    }

    pub fn get_socket_handles(&self) -> AmdSmiResult<Vec<amdsmi_socket_handle>> {
        let mut socket_count = self.get_socket_handles_count()?;

        // Allocate the memory for the sockets
        let mut sockets: Vec<amdsmi_socket_handle> =
            vec![std::ptr::null_mut(); socket_count as usize];

        unsafe {
            let ret = amdsmi_get_socket_handles(&mut socket_count, sockets.as_mut_ptr());
            chk_amd_smi_ret(ret)?;
        }

        Ok(sockets)
    }

    pub fn get_socket_info(&self, socket_handle: amdsmi_socket_handle) -> AmdSmiResult<Vec<i8>> {
        // Get Socket info
        let mut socket_info: Vec<i8> = vec![i8::default(); 128];

        unsafe {
            let ret = amdsmi_get_socket_info(socket_handle, 128, socket_info.as_mut_ptr());
            chk_amd_smi_ret(ret)?;
        }

        Ok(socket_info)
    }

    pub fn get_processor_handles_count(
        &self,
        socket_handle: amdsmi_socket_handle,
    ) -> AmdSmiResult<u32> {
        // Get the device count for the socket.
        let mut processor_count = 0;

        unsafe {
            let ret = amdsmi_get_processor_handles(
                socket_handle,
                &mut processor_count,
                std::ptr::null_mut(),
            );
            chk_amd_smi_ret(ret)?;
        }

        Ok(processor_count)
    }

    pub fn get_processor_handles(
        &self,
        socket_handle: amdsmi_socket_handle,
    ) -> AmdSmiResult<Vec<amdsmi_processor_handle>> {
        let mut processor_count = self.get_processor_handles_count(socket_handle)?;

        // Allocate the memory for the device handlers on the socket
        let mut processor_handles: Vec<amdsmi_processor_handle> =
            vec![std::ptr::null_mut(); processor_count as usize];

        unsafe {
            let ret = amdsmi_get_processor_handles(
                socket_handle,
                &mut processor_count,
                processor_handles.as_mut_ptr(),
            );
            chk_amd_smi_ret(ret)?;
        }

        Ok(processor_handles)
    }

    pub fn get_processor_type(
        &self,
        processor_handle: amdsmi_processor_handle,
    ) -> AmdSmiResult<processor_type_t> {
        let mut processor_type: processor_type_t = processor_type_t_AMDSMI_PROCESSOR_TYPE_UNKNOWN;

        unsafe {
            let ret = amdsmi_get_processor_type(processor_handle, &mut processor_type);
            chk_amd_smi_ret(ret)?;
        }

        Ok(processor_type)
    }

    pub fn get_gpu_board_info(
        &self,
        processor_handle: amdsmi_processor_handle,
    ) -> AmdSmiResult<amdsmi_board_info_t> {
        let mut board_info = std::mem::MaybeUninit::<amdsmi_board_info_t>::uninit();

        unsafe {
            let ret = amdsmi_get_gpu_board_info(processor_handle, board_info.as_mut_ptr());
            chk_amd_smi_ret(ret)?;
        }

        let board_info = unsafe { board_info.assume_init() };

        Ok(board_info)
    }

    pub fn get_temp_metric(
        &self,
        processor_handle: amdsmi_processor_handle,
        sensor_type: amdsmi_temperature_type_t,
        metric: amdsmi_temperature_metric_t,
    ) -> AmdSmiResult<i64> {
        let mut temperature = 0;

        unsafe {
            let ret =
                amdsmi_get_temp_metric(processor_handle, sensor_type, metric, &mut temperature);
            chk_amd_smi_ret(ret)?;
        }

        Ok(temperature)
    }

    pub fn get_gpu_ras_feature_info(
        &self,
        processor_handle: amdsmi_processor_handle,
    ) -> AmdSmiResult<amdsmi_ras_feature_t> {
        let mut ras_feature = std::mem::MaybeUninit::<amdsmi_ras_feature_t>::uninit();

        unsafe {
            let ret = amdsmi_get_gpu_ras_feature_info(processor_handle, ras_feature.as_mut_ptr());
            chk_amd_smi_ret(ret)?;
        }

        let ras_feature = unsafe { ras_feature.assume_init() };
        Ok(ras_feature)
    }

    pub fn get_gpu_device_bdf(
        &self,
        processor_handle: amdsmi_processor_handle,
    ) -> AmdSmiResult<amdsmi_bdf_t> {
        let mut bdf = std::mem::MaybeUninit::<amdsmi_bdf_t>::uninit();

        unsafe {
            let ret = amdsmi_get_gpu_device_bdf(processor_handle, bdf.as_mut_ptr());
            chk_amd_smi_ret(ret)?;
        }

        let bdf = unsafe { bdf.assume_init() };
        Ok(bdf)
    }

    pub fn get_processor_handle_from_bdf(
        &self,
        bdf: amdsmi_bdf_t,
    ) -> AmdSmiResult<amdsmi_processor_handle> {
        let mut processor_handle: amdsmi_processor_handle = std::ptr::null_mut();

        unsafe {
            let ret = amdsmi_get_processor_handle_from_bdf(bdf, &mut processor_handle);
            chk_amd_smi_ret(ret)?;
        }

        Ok(processor_handle)
    }

    pub fn get_gpu_asic_info(
        &self,
        processor_handle: amdsmi_processor_handle,
    ) -> AmdSmiResult<amdsmi_asic_info_t> {
        let mut asic_info = std::mem::MaybeUninit::<amdsmi_asic_info_t>::uninit();

        unsafe {
            let ret = amdsmi_get_gpu_asic_info(processor_handle, asic_info.as_mut_ptr());
            chk_amd_smi_ret(ret)?;
        }

        let asic_info = unsafe { asic_info.assume_init() };
        Ok(asic_info)
    }

    pub fn get_gpu_vbios_info(
        &self,
        processor_handle: amdsmi_processor_handle,
    ) -> AmdSmiResult<amdsmi_vbios_info_t> {
        let mut vbios_info = std::mem::MaybeUninit::<amdsmi_vbios_info_t>::uninit();

        unsafe {
            let ret = amdsmi_get_gpu_vbios_info(processor_handle, vbios_info.as_mut_ptr());
            chk_amd_smi_ret(ret)?;
        }

        let vbios_info = unsafe { vbios_info.assume_init() };
        Ok(vbios_info)
    }

    pub fn get_gpu_activity(
        &self,
        processor_handle: amdsmi_processor_handle,
    ) -> AmdSmiResult<amdsmi_engine_usage_t> {
        let mut engine_usage = std::mem::MaybeUninit::<amdsmi_engine_usage_t>::uninit();

        unsafe {
            let ret = amdsmi_get_gpu_activity(processor_handle, engine_usage.as_mut_ptr());
            chk_amd_smi_ret(ret)?;
        }

        let engine_usage = unsafe { engine_usage.assume_init() };
        Ok(engine_usage)
    }

    pub fn get_fw_info(
        &self,
        processor_handle: amdsmi_processor_handle,
    ) -> AmdSmiResult<amdsmi_fw_info_t> {
        let mut fw_information = std::mem::MaybeUninit::<amdsmi_fw_info_t>::uninit();

        unsafe {
            let ret = amdsmi_get_fw_info(processor_handle, fw_information.as_mut_ptr());
            chk_amd_smi_ret(ret)?;
        }

        let fw_information = unsafe { fw_information.assume_init() };
        Ok(fw_information)
    }

    pub fn get_gpu_bad_page_info_count(
        &self,
        processor_handle: amdsmi_processor_handle,
    ) -> AmdSmiResult<u32> {
        let mut num_pages = 0;

        unsafe {
            let ret = amdsmi_get_gpu_bad_page_info(
                processor_handle,
                &mut num_pages,
                std::ptr::null_mut(),
            );
            chk_amd_smi_ret(ret)?;
        }

        Ok(num_pages)
    }

    pub fn get_gpu_bad_page_info(
        &self,
        processor_handle: amdsmi_processor_handle,
    ) -> AmdSmiResult<Vec<amdsmi_retired_page_record_t>> {
        let mut num_pages = self.get_gpu_bad_page_info_count(processor_handle)?;

        let info = amdsmi_retired_page_record_t {
            page_address: u64::default(),
            page_size: u64::default(),
            status: 0,
        };

        let mut bad_page_info: Vec<amdsmi_retired_page_record_t> = vec![info; num_pages as usize];

        unsafe {
            let ret = amdsmi_get_gpu_bad_page_info(
                processor_handle,
                &mut num_pages,
                bad_page_info.as_mut_ptr(),
            );
            chk_amd_smi_ret(ret)?;
        }

        Ok(bad_page_info)
    }

    pub fn get_gpu_total_ecc_count(
        &self,
        processor_handle: amdsmi_processor_handle,
    ) -> AmdSmiResult<amdsmi_error_count_t> {
        let mut err_cnt_info = std::mem::MaybeUninit::<amdsmi_error_count_t>::uninit();

        unsafe {
            let ret = amdsmi_get_gpu_total_ecc_count(processor_handle, err_cnt_info.as_mut_ptr());
            chk_amd_smi_ret(ret)?;
        }

        let err_cnt_info = unsafe { err_cnt_info.assume_init() };
        Ok(err_cnt_info)
    }

    pub fn get_gpu_vram_usage(
        &self,
        processor_handle: amdsmi_processor_handle,
    ) -> AmdSmiResult<amdsmi_vram_usage_t> {
        let mut vram_usage = std::mem::MaybeUninit::<amdsmi_vram_usage_t>::uninit();

        unsafe {
            let ret = amdsmi_get_gpu_vram_usage(processor_handle, vram_usage.as_mut_ptr());
            chk_amd_smi_ret(ret)?;
        }

        let vram_usage = unsafe { vram_usage.assume_init() };
        Ok(vram_usage)
    }

    pub fn get_power_cap_info(
        &self,
        processor_handle: amdsmi_processor_handle,
        sensor_ind: u32,
    ) -> AmdSmiResult<amdsmi_power_cap_info_t> {
        let mut cap_info = std::mem::MaybeUninit::<amdsmi_power_cap_info_t>::uninit();

        unsafe {
            let ret =
                amdsmi_get_power_cap_info(processor_handle, sensor_ind, cap_info.as_mut_ptr());
            chk_amd_smi_ret(ret)?;
        }

        let cap_info = unsafe { cap_info.assume_init() };
        Ok(cap_info)
    }

    pub fn get_soc_pstate(
        &self,
        processor_handle: amdsmi_processor_handle,
    ) -> AmdSmiResult<amdsmi_dpm_policy_t> {
        let mut policy = std::mem::MaybeUninit::<amdsmi_dpm_policy_t>::uninit();

        unsafe {
            let ret = amdsmi_get_soc_pstate(processor_handle, policy.as_mut_ptr());
            chk_amd_smi_ret(ret)?;
        }

        let policy = unsafe { policy.assume_init() };
        Ok(policy)
    }

    pub fn get_link_topology_nearest(
        &self,
        processor_handle: amdsmi_processor_handle,
        link_type: amdsmi_link_type_t,
    ) -> AmdSmiResult<amdsmi_topology_nearest_t> {
        let mut topology_nearest_info =
            std::mem::MaybeUninit::<amdsmi_topology_nearest_t>::uninit();

        unsafe {
            let ret = amdsmi_get_link_topology_nearest(
                processor_handle,
                link_type,
                topology_nearest_info.as_mut_ptr(),
            );
            chk_amd_smi_ret(ret)?;
        }

        let topology_nearest_info = unsafe { topology_nearest_info.assume_init() };
        Ok(topology_nearest_info)
    }

    pub fn get_gpu_vram_info(
        &self,
        processor_handle: amdsmi_processor_handle,
    ) -> AmdSmiResult<amdsmi_vram_info_t> {
        let mut vram_info = std::mem::MaybeUninit::<amdsmi_vram_info_t>::uninit();

        unsafe {
            let ret = amdsmi_get_gpu_vram_info(processor_handle, vram_info.as_mut_ptr());
            chk_amd_smi_ret(ret)?;
        }

        let vram_info = unsafe { vram_info.assume_init() };
        Ok(vram_info)
    }

    pub fn get_gpu_cache_info(
        &self,
        processor_handle: amdsmi_processor_handle,
    ) -> AmdSmiResult<amdsmi_gpu_cache_info_t> {
        let mut cache_info = std::mem::MaybeUninit::<amdsmi_gpu_cache_info_t>::uninit();

        unsafe {
            let ret = amdsmi_get_gpu_cache_info(processor_handle, cache_info.as_mut_ptr());
            chk_amd_smi_ret(ret)?;
        }

        let cache_info = unsafe { cache_info.assume_init() };
        Ok(cache_info)
    }

    pub fn get_power_info(
        &self,
        processor_handle: amdsmi_processor_handle,
    ) -> AmdSmiResult<amdsmi_power_info_t> {
        let mut power_measure = std::mem::MaybeUninit::<amdsmi_power_info_t>::uninit();

        unsafe {
            let ret = amdsmi_get_power_info(processor_handle, power_measure.as_mut_ptr());
            chk_amd_smi_ret(ret)?;
        }

        let power_measure = unsafe { power_measure.assume_init() };
        Ok(power_measure)
    }

    pub fn get_gpu_driver_info(
        &self,
        processor_handle: amdsmi_processor_handle,
    ) -> AmdSmiResult<amdsmi_driver_info_t> {
        let mut driver_info = std::mem::MaybeUninit::<amdsmi_driver_info_t>::uninit();

        unsafe {
            let ret = amdsmi_get_gpu_driver_info(processor_handle, driver_info.as_mut_ptr());
            chk_amd_smi_ret(ret)?;
        }

        let driver_info = unsafe { driver_info.assume_init() };
        Ok(driver_info)
    }

    pub fn get_gpu_device_uuid(
        &self,
        processor_handle: amdsmi_processor_handle,
    ) -> AmdSmiResult<Vec<i8>> {
        let mut uuid_length = AMDSMI_GPU_UUID_SIZE;
        let mut uuid = vec![i8::default(); uuid_length as usize];

        unsafe {
            let ret =
                amdsmi_get_gpu_device_uuid(processor_handle, &mut uuid_length, uuid.as_mut_ptr());
            chk_amd_smi_ret(ret)?;
        }

        Ok(uuid)
    }

    pub fn get_clock_info(
        &self,
        processor_handle: amdsmi_processor_handle,
        clk_type: amdsmi_clk_type_t,
    ) -> AmdSmiResult<amdsmi_clk_info_t> {
        let mut gfx_clk_values = std::mem::MaybeUninit::<amdsmi_clk_info_t>::uninit();

        unsafe {
            let ret =
                amdsmi_get_clock_info(processor_handle, clk_type, gfx_clk_values.as_mut_ptr());
            chk_amd_smi_ret(ret)?;
        }

        let gfx_clk_values = unsafe { gfx_clk_values.assume_init() };
        Ok(gfx_clk_values)
    }

    pub fn get_pcie_info(
        &self,
        processor_handle: amdsmi_processor_handle,
    ) -> AmdSmiResult<amdsmi_pcie_info_t> {
        let mut pcie_info = std::mem::MaybeUninit::<amdsmi_pcie_info_t>::uninit();

        unsafe {
            let ret = amdsmi_get_pcie_info(processor_handle, pcie_info.as_mut_ptr());
            chk_amd_smi_ret(ret)?;
        }

        let pcie_info = unsafe { pcie_info.assume_init() };
        Ok(pcie_info)
    }

    pub fn get_gpu_ras_block_features_enabled(
        &self,
        processor_handle: amdsmi_processor_handle,
        block: amdsmi_gpu_block_t,
    ) -> AmdSmiResult<amdsmi_ras_err_state_t> {
        let mut state = amdsmi_ras_err_state_t_AMDSMI_RAS_ERR_STATE_NONE;

        unsafe {
            let ret =
                amdsmi_get_gpu_ras_block_features_enabled(processor_handle, block, &mut state);
            chk_amd_smi_ret(ret)?;
        }

        Ok(state)
    }

    pub fn get_gpu_process_list_count(
        &self,
        processor_handle: amdsmi_processor_handle,
    ) -> AmdSmiResult<u32> {
        let mut num_process = 0;

        unsafe {
            let ret = amdsmi_get_gpu_process_list(
                processor_handle,
                &mut num_process,
                std::ptr::null_mut(),
            );
            chk_amd_smi_ret(ret)?;
        }

        Ok(num_process)
    }

    pub fn get_gpu_process_list(
        &self,
        processor_handle: amdsmi_processor_handle,
    ) -> AmdSmiResult<Vec<amdsmi_proc_info_t>> {
        let mut num_process = self.get_gpu_process_list_count(processor_handle)?;

        let process = amdsmi_proc_info_t {
            name: [i8::default(); 32usize],
            pid: amdsmi_process_handle_t::default(),
            mem: u64::default(),
            engine_usage: amdsmi_proc_info_t_engine_usage_ {
                gfx: u64::default(),
                enc: u64::default(),
                reserved: [u32::default(); 12usize],
            },
            memory_usage: amdsmi_proc_info_t_memory_usage_ {
                gtt_mem: u64::default(),
                cpu_mem: u64::default(),
                vram_mem: u64::default(),
                reserved: [u32::default(); 10usize],
            },
            container_name: [i8::default(); 32usize],
            reserved: [u32::default(); 4usize],
        };

        let mut process_info_list: Vec<amdsmi_proc_info_t> = vec![process; num_process as usize];

        unsafe {
            let ret = amdsmi_get_gpu_process_list(
                processor_handle,
                &mut num_process,
                process_info_list.as_mut_ptr(),
            );
            chk_amd_smi_ret(ret)?;
        }

        Ok(process_info_list)
    }

    pub fn get_gpu_metrics_info(
        &self,
        processor_handle: amdsmi_processor_handle,
    ) -> AmdSmiResult<amdsmi_gpu_metrics_t> {
        let mut smu = std::mem::MaybeUninit::<amdsmi_gpu_metrics_t>::uninit();

        unsafe {
            let ret = amdsmi_get_gpu_metrics_info(processor_handle, smu.as_mut_ptr());
            chk_amd_smi_ret(ret)?;
        }

        let smu = unsafe { smu.assume_init() };
        Ok(smu)
    }
}

impl Drop for AmdSmi {
    fn drop(&mut self) {
        unsafe {
            let ret = amdsmi_shut_down();
            if ret != amdsmi_status_t_AMDSMI_STATUS_SUCCESS {
                panic!("amdsmi_shut_down error: {:?}", ret);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use amdsmi_sys::bindings::{
        amdsmi_status_t_AMDSMI_STATUS_NOT_INIT, amdsmi_temperature_metric_t_AMDSMI_TEMP_CURRENT,
        amdsmi_temperature_type_t_AMDSMI_TEMPERATURE_TYPE_EDGE,
        processor_type_t_AMDSMI_PROCESSOR_TYPE_AMD_GPU,
    };
    use utilities::helper_functions::buf_i8_to_string;

    #[test]
    fn test_check_version() {
        let amd_smi = AmdSmi::new().unwrap();

        let version = amd_smi.get_version().unwrap();
        println!("{:?}", version);

        assert!(version.year > 0);
    }

    #[test]
    fn test_amdsmi_drop() {
        let amd_smi = AmdSmi::new().unwrap();

        amd_smi.get_version().unwrap();
        drop(amd_smi);

        let mut version = std::mem::MaybeUninit::<amdsmi_version_t>::uninit();
        let ret = unsafe { amdsmi_get_lib_version(version.as_mut_ptr()) };

        assert_eq!(ret, amdsmi_status_t_AMDSMI_STATUS_NOT_INIT);
    }

    // https://rocm.docs.amd.com/projects/amdsmi/en/latest/how-to/amdsmi-cpp-lib.html#hello-amd-smi
    // resources/hello_amdsmi.cpp
    // example Hello AMD SMI
    #[test]
    fn test_example_hello_amd_smi() {
        let amd_smi = AmdSmi::new().unwrap();

        // Init amdsmi for sockets and devices. Here we are only interested in AMD_GPUS.
        // * is done inside the AmdSmi::new()

        // Get all sockets

        // Get the socket handles in the system
        let sockets = amd_smi.get_socket_handles().unwrap();
        let socket_count = sockets.len();

        println!("Total Socket: {}", socket_count);
        println!("sockets: {:?}", sockets);
        println!();

        // For each socket, get identifier and devices
        for socket_handle in sockets {
            // Get Socket info
            let socket_info = amd_smi.get_socket_info(socket_handle).unwrap();

            println!("Socket: {}", buf_i8_to_string(&socket_info).unwrap());

            // Get all devices of the socket
            let processor_handles = amd_smi.get_processor_handles(socket_handle).unwrap();
            let device_count = processor_handles.len();

            println!("device_count: {}", device_count);

            // For each device of the socket, get name and temperature.
            for processor_handle in processor_handles {
                // Get device type. Since the amdsmi is initialized with
                // AMD_SMI_INIT_AMD_GPUS, the processor_type must be AMDSMI_PROCESSOR_TYPE_AMD_GPU.
                let processor_type = amd_smi.get_processor_type(processor_handle).unwrap();

                if processor_type != processor_type_t_AMDSMI_PROCESSOR_TYPE_AMD_GPU {
                    panic!("Expect AMDSMI_PROCESSOR_TYPE_AMD_GPU device type!")
                }

                // Get device name
                let board_info = amd_smi.get_gpu_board_info(processor_handle).unwrap();

                println!(
                    "  Device Name: {:?}",
                    buf_i8_to_string(&board_info.product_name)
                );

                // Get temperature
                // let mut val_i64 = 0;
                let val_i64 = amd_smi
                    .get_temp_metric(
                        processor_handle,
                        amdsmi_temperature_type_t_AMDSMI_TEMPERATURE_TYPE_EDGE,
                        amdsmi_temperature_metric_t_AMDSMI_TEMP_CURRENT,
                    )
                    .unwrap();

                println!("  Temperature: {:?} C", val_i64);

                println!();
            }
        }

        // Clean up resources allocated at amdsmi_init. It will invalidate sockets and devices pointers
        // * is done inside the AmdSmi drop trait
    }
}
