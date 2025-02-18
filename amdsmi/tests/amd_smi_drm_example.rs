// https://github.com/ROCm/amdsmi/blob/amd-staging/example/amd_smi_nodrm_example.cc
// resources/amd_smi_nodrm_example.cc

use amdsmi::utils::get_amd_smi_vram_type;
use amdsmi::AmdSmi;
use amdsmi_sys::bindings::{
    amdsmi_clk_type_t_AMDSMI_CLK_TYPE_GFX, amdsmi_clk_type_t_AMDSMI_CLK_TYPE_MEM,
    amdsmi_fw_block_t_AMDSMI_FW_ID_ASD, amdsmi_fw_block_t_AMDSMI_FW_ID_CP_CE,
    amdsmi_fw_block_t_AMDSMI_FW_ID_CP_ME, amdsmi_fw_block_t_AMDSMI_FW_ID_CP_MEC1,
    amdsmi_fw_block_t_AMDSMI_FW_ID_CP_MEC2, amdsmi_fw_block_t_AMDSMI_FW_ID_CP_MEC_JT1,
    amdsmi_fw_block_t_AMDSMI_FW_ID_CP_MEC_JT2, amdsmi_fw_block_t_AMDSMI_FW_ID_CP_PFP,
    amdsmi_fw_block_t_AMDSMI_FW_ID_CP_PM4, amdsmi_fw_block_t_AMDSMI_FW_ID_DFC,
    amdsmi_fw_block_t_AMDSMI_FW_ID_DMCU, amdsmi_fw_block_t_AMDSMI_FW_ID_DMCU_ERAM,
    amdsmi_fw_block_t_AMDSMI_FW_ID_DMCU_ISR, amdsmi_fw_block_t_AMDSMI_FW_ID_DRV_CAP,
    amdsmi_fw_block_t_AMDSMI_FW_ID_ISP, amdsmi_fw_block_t_AMDSMI_FW_ID_MC,
    amdsmi_fw_block_t_AMDSMI_FW_ID_MMSCH, amdsmi_fw_block_t_AMDSMI_FW_ID_PM,
    amdsmi_fw_block_t_AMDSMI_FW_ID_PSP_BL, amdsmi_fw_block_t_AMDSMI_FW_ID_PSP_KEYDB,
    amdsmi_fw_block_t_AMDSMI_FW_ID_PSP_SOSDRV, amdsmi_fw_block_t_AMDSMI_FW_ID_PSP_SPL,
    amdsmi_fw_block_t_AMDSMI_FW_ID_PSP_SYSDRV, amdsmi_fw_block_t_AMDSMI_FW_ID_PSP_TOC,
    amdsmi_fw_block_t_AMDSMI_FW_ID_RLC, amdsmi_fw_block_t_AMDSMI_FW_ID_RLC_RESTORE_LIST_CNTL,
    amdsmi_fw_block_t_AMDSMI_FW_ID_RLC_RESTORE_LIST_GPM_MEM,
    amdsmi_fw_block_t_AMDSMI_FW_ID_RLC_RESTORE_LIST_SRM_MEM,
    amdsmi_fw_block_t_AMDSMI_FW_ID_RLC_SRLG, amdsmi_fw_block_t_AMDSMI_FW_ID_RLC_SRLS,
    amdsmi_fw_block_t_AMDSMI_FW_ID_RLC_V, amdsmi_fw_block_t_AMDSMI_FW_ID_SDMA0,
    amdsmi_fw_block_t_AMDSMI_FW_ID_SDMA1, amdsmi_fw_block_t_AMDSMI_FW_ID_SDMA2,
    amdsmi_fw_block_t_AMDSMI_FW_ID_SDMA3, amdsmi_fw_block_t_AMDSMI_FW_ID_SDMA4,
    amdsmi_fw_block_t_AMDSMI_FW_ID_SDMA5, amdsmi_fw_block_t_AMDSMI_FW_ID_SDMA6,
    amdsmi_fw_block_t_AMDSMI_FW_ID_SDMA7, amdsmi_fw_block_t_AMDSMI_FW_ID_SMU,
    amdsmi_fw_block_t_AMDSMI_FW_ID_TA_RAS, amdsmi_fw_block_t_AMDSMI_FW_ID_TA_XGMI,
    amdsmi_fw_block_t_AMDSMI_FW_ID_UVD, amdsmi_fw_block_t_AMDSMI_FW_ID_VCE,
    amdsmi_fw_block_t_AMDSMI_FW_ID_VCN, amdsmi_gpu_block_t_AMDSMI_GPU_BLOCK_FIRST,
    amdsmi_gpu_block_t_AMDSMI_GPU_BLOCK_LAST, amdsmi_temperature_metric_t_AMDSMI_TEMP_CRITICAL,
    amdsmi_temperature_metric_t_AMDSMI_TEMP_CURRENT,
    amdsmi_temperature_type_t_AMDSMI_TEMPERATURE_TYPE_EDGE,
    amdsmi_temperature_type_t_AMDSMI_TEMPERATURE_TYPE_HOTSPOT,
    amdsmi_temperature_type_t_AMDSMI_TEMPERATURE_TYPE_PLX,
    amdsmi_temperature_type_t_AMDSMI_TEMPERATURE_TYPE_VRAM,
    amdsmi_temperature_type_t_AMDSMI_TEMPERATURE_TYPE__MAX,
    processor_type_t_AMDSMI_PROCESSOR_TYPE_AMD_GPU,
};
use utilities::helper_functions::buf_i8_to_string;

// getFWNameFromId
#[allow(non_upper_case_globals)]
fn get_fwname_from_id(id: u32) -> &'static str {
    match id {
        amdsmi_fw_block_t_AMDSMI_FW_ID_SMU => "SMU",
        amdsmi_fw_block_t_AMDSMI_FW_ID_CP_CE => "CP_CE",
        amdsmi_fw_block_t_AMDSMI_FW_ID_CP_PFP => "CP_PFP",
        amdsmi_fw_block_t_AMDSMI_FW_ID_CP_ME => "CP_ME",
        amdsmi_fw_block_t_AMDSMI_FW_ID_CP_MEC_JT1 => "CP_MEC_JT1",
        amdsmi_fw_block_t_AMDSMI_FW_ID_CP_MEC_JT2 => "CP_MEC_JT2",
        amdsmi_fw_block_t_AMDSMI_FW_ID_CP_MEC1 => "CP_MEC1",
        amdsmi_fw_block_t_AMDSMI_FW_ID_CP_MEC2 => "CP_MEC2",
        amdsmi_fw_block_t_AMDSMI_FW_ID_RLC => "RLC",
        amdsmi_fw_block_t_AMDSMI_FW_ID_SDMA0 => "SDMA0",
        amdsmi_fw_block_t_AMDSMI_FW_ID_SDMA1 => "SDMA1",
        amdsmi_fw_block_t_AMDSMI_FW_ID_SDMA2 => "SDMA2",
        amdsmi_fw_block_t_AMDSMI_FW_ID_SDMA3 => "SDMA3",
        amdsmi_fw_block_t_AMDSMI_FW_ID_SDMA4 => "SDMA4",
        amdsmi_fw_block_t_AMDSMI_FW_ID_SDMA5 => "SDMA5",
        amdsmi_fw_block_t_AMDSMI_FW_ID_SDMA6 => "SDMA6",
        amdsmi_fw_block_t_AMDSMI_FW_ID_SDMA7 => "SDMA7",
        amdsmi_fw_block_t_AMDSMI_FW_ID_VCN => "VCN",
        amdsmi_fw_block_t_AMDSMI_FW_ID_UVD => "UVD",
        amdsmi_fw_block_t_AMDSMI_FW_ID_VCE => "VCE",
        amdsmi_fw_block_t_AMDSMI_FW_ID_ISP => "ISP",
        amdsmi_fw_block_t_AMDSMI_FW_ID_DMCU_ERAM => "DMCU_ERAM",
        amdsmi_fw_block_t_AMDSMI_FW_ID_DMCU_ISR => "DMCU_ISR",
        amdsmi_fw_block_t_AMDSMI_FW_ID_RLC_RESTORE_LIST_GPM_MEM => "RLC_RESTORE_LIST_GPM_MEM",
        amdsmi_fw_block_t_AMDSMI_FW_ID_RLC_RESTORE_LIST_SRM_MEM => "RLC_RESTORE_LIST_SRM_MEM",
        amdsmi_fw_block_t_AMDSMI_FW_ID_RLC_RESTORE_LIST_CNTL => "RLC_RESTORE_LIST_CNTL",
        amdsmi_fw_block_t_AMDSMI_FW_ID_RLC_V => "RLC_V",
        amdsmi_fw_block_t_AMDSMI_FW_ID_MMSCH => "MMSCH",
        amdsmi_fw_block_t_AMDSMI_FW_ID_PSP_SYSDRV => "PSP_SYSDRV",
        amdsmi_fw_block_t_AMDSMI_FW_ID_PSP_SOSDRV => "PSP_SOSDRV",
        amdsmi_fw_block_t_AMDSMI_FW_ID_PSP_TOC => "PSP_TOC",
        amdsmi_fw_block_t_AMDSMI_FW_ID_PSP_KEYDB => "PSP_KEYDB",
        amdsmi_fw_block_t_AMDSMI_FW_ID_DFC => "DFC",
        amdsmi_fw_block_t_AMDSMI_FW_ID_PSP_SPL => "PSP_SPL",
        amdsmi_fw_block_t_AMDSMI_FW_ID_DRV_CAP => "DRV_CAP",
        amdsmi_fw_block_t_AMDSMI_FW_ID_MC => "MC",
        amdsmi_fw_block_t_AMDSMI_FW_ID_PSP_BL => "PSP_BL",
        amdsmi_fw_block_t_AMDSMI_FW_ID_CP_PM4 => "CP_PM4",
        amdsmi_fw_block_t_AMDSMI_FW_ID_ASD => "ID_ASD",
        amdsmi_fw_block_t_AMDSMI_FW_ID_TA_RAS => "ID_TA_RAS",
        amdsmi_fw_block_t_AMDSMI_FW_ID_TA_XGMI => "ID_TA_XGMI",
        amdsmi_fw_block_t_AMDSMI_FW_ID_RLC_SRLG => "ID_RLC_SRLG",
        amdsmi_fw_block_t_AMDSMI_FW_ID_RLC_SRLS => "ID_RLC_SRLS",
        amdsmi_fw_block_t_AMDSMI_FW_ID_PM => "ID_PM",
        amdsmi_fw_block_t_AMDSMI_FW_ID_DMCU => "ID_DMCU",
        _ => "",
    }
}

#[test]
fn test_amd_smi_drm_example() {
    let amd_smi = AmdSmi::new().unwrap();

    // Init amdsmi for sockets and devices.
    // Here we are only interested in AMD_GPUS.
    // * is done inside the AmdSmi::new()

    // Get all sockets

    // Get the socket handles in the system
    let sockets = amd_smi.get_socket_handles().unwrap();

    println!("Total Socket: {}", sockets.len());
    println!("sockets: {:?}", sockets);
    println!();

    // For each socket, get identifier and devices
    for (i, socket_handle) in sockets.into_iter().enumerate() {
        // Get Socket info
        let socket_info: Vec<i8> = amd_smi.get_socket_info(socket_handle).unwrap();

        println!("Socket: {:?}", buf_i8_to_string(&socket_info));

        // Get all devices of the socket
        let processor_handles = amd_smi.get_processor_handles(socket_handle).unwrap();

        println!("device_count: {}", processor_handles.len());
        println!("processor_handles: {:?}", processor_handles);

        // For each device of the socket, get name and temperature.
        for (j, processor_handle) in processor_handles.into_iter().enumerate() {
            // Get device type. Since the amdsmi is initialized with
            // AMD_SMI_INIT_AMD_GPUS, the processor_type must be AMDSMI_PROCESSOR_TYPE_AMD_GPU.
            let processor_type = amd_smi.get_processor_type(processor_handle).unwrap();

            println!("  processor_type: {:?}", processor_type);

            if processor_type != processor_type_t_AMDSMI_PROCESSOR_TYPE_AMD_GPU {
                panic!("Expect AMDSMI_PROCESSOR_TYPE_AMD_GPU device type!")
            }

            let bdf = amd_smi.get_gpu_device_bdf(processor_handle).unwrap();

            println!("  Output of amdsmi_get_gpu_device_bdf:");

            unsafe {
                println!(
                    "   Device [{}] BDF {:04}:{:02}:{:02}.{}",
                    i,
                    bdf.__bindgen_anon_1.domain_number(),
                    bdf.__bindgen_anon_1.bus_number(),
                    bdf.__bindgen_anon_1.device_number(),
                    bdf.__bindgen_anon_1.function_number(),
                );
            }

            // Get handle from BDF
            let dev_handle = amd_smi.get_processor_handle_from_bdf(bdf).unwrap();
            println!("  processor_handle: {:?}", processor_handle);
            println!("  dev_handle: {:?}", dev_handle);

            // Get ASIC info
            let asic_info = amd_smi.get_gpu_asic_info(processor_handle).unwrap();

            println!("  Output of amdsmi_get_gpu_asic_info:");

            println!(
                "      Market Name: {}",
                buf_i8_to_string(&asic_info.market_name).unwrap()
            );
            println!("      DeviceID: {}", asic_info.device_id);
            println!("      RevisionID: {}", asic_info.rev_id);
            println!(
                "      Asic serial: {}",
                buf_i8_to_string(&asic_info.asic_serial).unwrap()
            );
            println!("      OAM id: {}", asic_info.oam_id);
            println!("      Num of Computes: {}", asic_info.num_of_compute_units);

            // Get VRAM info
            let r = amd_smi.get_gpu_vram_info(processor_handle);

            println!("  Output of amdsmi_get_gpu_vram_info:");
            match r {
                Ok(vram_info) => {
                    println!("      VRAM Size: {}", vram_info.vram_size);
                    println!("      BIT Width: {}", vram_info.vram_bit_width);
                    println!(
                        "      Type: {:?}",
                        get_amd_smi_vram_type(vram_info.vram_type)
                    );
                    println!("      Vendor: {}", vram_info.vram_vendor);
                }
                Err(error) => {
                    println!("  Error: {:?}", error);
                }
            }

            // Get VBIOS info
            let vbios_info = amd_smi.get_gpu_vbios_info(processor_handle).unwrap();

            println!("  Output of amdsmi_get_gpu_vbios_info:");

            println!(
                "      VBios Name: {}",
                buf_i8_to_string(&vbios_info.name).unwrap()
            );
            println!(
                "      Build Date: {}",
                buf_i8_to_string(&vbios_info.build_date).unwrap()
            );
            println!(
                "      Part Number: {}",
                buf_i8_to_string(&vbios_info.part_number).unwrap()
            );
            println!(
                "      VBios Version String: {}",
                buf_i8_to_string(&vbios_info.version).unwrap()
            );

            // Get Cache info
            let cache_info = amd_smi.get_gpu_cache_info(processor_handle).unwrap();

            println!("  Output of amdsmi_get_gpu_cache_info:");

            for cache_index in 0..cache_info.num_cache_types {
                let c_info = &cache_info.cache[cache_index as usize];

                println!(
                    "      Cache Level: {}, Cache Size: {} KB, Cache type: {}",
                    c_info.cache_level, c_info.cache_size, c_info.cache_properties
                );
                println!(
                    "      Max number CU shared: {}, Number of instances: {}",
                    c_info.max_num_cu_shared, c_info.num_cache_instance
                );
                println!();
            }

            // Get power measure
            println!("  Output of amdsmi_get_power_info:");

            match amd_smi.get_power_info(processor_handle) {
                Ok(power_measure) => {
                    println!("      Current GFX Voltage: {}", power_measure.gfx_voltage);
                    println!(
                        "      Average socket power: {}",
                        power_measure.average_socket_power
                    );
                    println!("      GPU Power limit: {}", power_measure.power_limit);

                    println!(
                        "      Current socket power: {}",
                        power_measure.current_socket_power
                    );
                    println!("      Soc voltage: {}", power_measure.soc_voltage);
                    println!("      Mem voltage: {}", power_measure.mem_voltage);
                }
                Err(e) => {
                    println!("      {:?}", e);
                }
            };

            // Get driver version
            let driver_info = amd_smi.get_gpu_driver_info(processor_handle).unwrap();

            println!("  Output of amdsmi_get_gpu_driver_info:");

            println!(
                "      Driver name: {}",
                buf_i8_to_string(&driver_info.driver_name).unwrap()
            );
            println!(
                "      Driver version: {}",
                buf_i8_to_string(&driver_info.driver_version).unwrap()
            );
            println!(
                "      Driver date: {}",
                buf_i8_to_string(&driver_info.driver_date).unwrap()
            );

            // Get device uuid
            let uuid = amd_smi.get_gpu_device_uuid(processor_handle).unwrap();

            println!("  Output of amdsmi_get_gpu_device_uuid:");

            println!("      Driver name: {}", buf_i8_to_string(&uuid).unwrap());

            // Get engine usage info
            println!("  Output of amdsmi_get_gpu_activity:");
            // error with 5700G gpu
            match amd_smi.get_gpu_activity(processor_handle) {
                Ok(engine_usage) => {
                    println!("      Average GFX Activity: {}", engine_usage.gfx_activity);
                    println!("      Average MM Activity: {}", engine_usage.mm_activity);
                    println!("      Average UMC Activity: {}", engine_usage.umc_activity);
                }
                Err(e) => {
                    println!("      {:?}", e);
                }
            };

            // Get firmware info
            let fw_information = amd_smi.get_fw_info(processor_handle).unwrap();

            println!("  Output of amdsmi_get_fw_info:");

            println!("      Number of Microcodes: {}", fw_information.num_fw_info);

            for ucode_index in 0..fw_information.num_fw_info {
                let fw_info = fw_information.fw_info_list[ucode_index as usize];
                let ucode_name = get_fwname_from_id(fw_info.fw_id);
                println!(
                    "          id: {} - {}: {}",
                    fw_info.fw_id, ucode_name, fw_info.fw_version
                );
            }

            // Get GFX clock measurements
            println!("  Output of amdsmi_get_clock_info:");

            match amd_smi.get_clock_info(processor_handle, amdsmi_clk_type_t_AMDSMI_CLK_TYPE_GFX) {
                Ok(gfx_clk_values) => {
                    println!("      GPU GFX Max Clock: {}", gfx_clk_values.max_clk);
                    println!("      GPU GFX Current Clock: {}", gfx_clk_values.clk);

                    println!("      GPU GFX Min Clock: {}", gfx_clk_values.min_clk);
                    println!("      GPU GFX Clk locked: {}", gfx_clk_values.clk_locked);
                    println!(
                        "      GPU GFX Clk deep sleep: {}",
                        gfx_clk_values.clk_deep_sleep
                    );
                }
                Err(e) => {
                    println!("      {:?}", e);
                }
            };

            println!();

            // Get MEM clock measurements
            match amd_smi.get_clock_info(processor_handle, amdsmi_clk_type_t_AMDSMI_CLK_TYPE_MEM) {
                Ok(mem_clk_values) => {
                    println!("      GPU MEM Max Clock: {}", mem_clk_values.max_clk);
                    println!("      GPU MEM Current Clock: {}", mem_clk_values.clk);

                    println!("      GPU MEM Min Clock: {}", mem_clk_values.min_clk);
                    println!("      GPU MEM Clk locked: {}", mem_clk_values.clk_locked);
                    println!(
                        "      GPU MEM Clk deep sleep: {}",
                        mem_clk_values.clk_deep_sleep
                    );
                }
                Err(e) => {
                    println!("      {:?}", e);
                }
            };

            // Get PCIe status
            println!("  Output of amdsmi_get_pcie_info:");

            match amd_smi.get_pcie_info(processor_handle) {
                Ok(pcie_info) => {
                    println!(
                        "      Current PCIe lanes: {}",
                        pcie_info.pcie_metric.pcie_width
                    );
                    println!(
                        "      Current PCIe speed: {}",
                        pcie_info.pcie_metric.pcie_speed
                    );

                    println!(
                        "      Current PCIe Interface Version: {}",
                        pcie_info.pcie_static.pcie_interface_version
                    );
                    println!("      PCIe slot type: {}", pcie_info.pcie_static.slot_type);
                    println!(
                        "      PCIe max lanes: {}",
                        pcie_info.pcie_static.max_pcie_width
                    );
                    println!(
                        "      PCIe max speed: {}",
                        pcie_info.pcie_static.max_pcie_speed
                    );

                    // additional pcie related metrics
                    println!(
                        "      PCIe bandwidth: {}",
                        pcie_info.pcie_metric.pcie_bandwidth
                    );
                    println!(
                        "      PCIe replay count: {}",
                        pcie_info.pcie_metric.pcie_replay_count
                    );
                    println!(
                        "      PCIe L0 recovery count: {}",
                        pcie_info.pcie_metric.pcie_l0_to_recovery_count
                    );
                    println!(
                        "      PCIe rollover count: {}",
                        pcie_info.pcie_metric.pcie_replay_roll_over_count
                    );
                    println!(
                        "      PCIe nak received count: {}",
                        pcie_info.pcie_metric.pcie_nak_received_count
                    );
                    println!(
                        "      PCIe nak sent count: {}",
                        pcie_info.pcie_metric.pcie_nak_sent_count
                    );
                }
                Err(e) => {
                    println!("      {:?}", e);
                }
            }

            // Get VRAM temperature limit
            println!("  Output of amdsmi_get_temp_metric:");

            match amd_smi.get_temp_metric(
                processor_handle,
                amdsmi_temperature_type_t_AMDSMI_TEMPERATURE_TYPE_VRAM,
                amdsmi_temperature_metric_t_AMDSMI_TEMP_CRITICAL,
            ) {
                Ok(temperature) => {
                    println!("      GPU VRAM temp limit: {}", temperature);
                }
                Err(e) => {
                    println!("      {:?}", e);
                }
            }

            // Get GFX temperature limit
            match amd_smi.get_temp_metric(
                processor_handle,
                amdsmi_temperature_type_t_AMDSMI_TEMPERATURE_TYPE_EDGE,
                amdsmi_temperature_metric_t_AMDSMI_TEMP_CRITICAL,
            ) {
                Ok(temperature) => {
                    println!("      GPU GFX temp limit: {}", temperature);
                }
                Err(e) => {
                    println!("      {:?}", e);
                }
            }

            // Get temperature measurements
            // amdsmi_temperature_t edge_temp, hotspot_temp, vram_temp, plx_temp;
            println!("  Output of amdsmi_get_temp_metric:");

            let mut temp_measurements = vec![
                i64::default();
                (amdsmi_temperature_type_t_AMDSMI_TEMPERATURE_TYPE__MAX + 1)
                    as usize
            ];
            let temp_types: Vec<_> = vec![
                amdsmi_temperature_type_t_AMDSMI_TEMPERATURE_TYPE_EDGE,
                amdsmi_temperature_type_t_AMDSMI_TEMPERATURE_TYPE_HOTSPOT,
                amdsmi_temperature_type_t_AMDSMI_TEMPERATURE_TYPE_VRAM,
                amdsmi_temperature_type_t_AMDSMI_TEMPERATURE_TYPE_PLX,
            ];

            for temp_type in temp_types {
                match amd_smi.get_temp_metric(
                    processor_handle,
                    temp_type,
                    amdsmi_temperature_metric_t_AMDSMI_TEMP_CURRENT,
                ) {
                    Ok(v) => {
                        temp_measurements[temp_type as usize] = v;
                    }
                    Err(e) => {
                        println!("      {:?}", e)
                    }
                };
            }

            println!("  Output of amdsmi_get_temp_metric:");

            println!(
                "      GPU Edge temp measurement: {}",
                temp_measurements[amdsmi_temperature_type_t_AMDSMI_TEMPERATURE_TYPE_EDGE as usize]
            );
            println!(
                "      GPU Hotspot temp measurement: {}",
                temp_measurements
                    [amdsmi_temperature_type_t_AMDSMI_TEMPERATURE_TYPE_HOTSPOT as usize]
            );
            println!(
                "      GPU VRAM temp measurement: {}",
                temp_measurements[amdsmi_temperature_type_t_AMDSMI_TEMPERATURE_TYPE_VRAM as usize]
            );
            println!(
                "      GPU PLX temp measurement: {}",
                temp_measurements[amdsmi_temperature_type_t_AMDSMI_TEMPERATURE_TYPE_PLX as usize]
            );

            // Get RAS features enabled
            let block_names = [
                "UMC",
                "SDMA",
                "GFX",
                "MMHUB",
                "ATHUB",
                "PCIE_BIF",
                "HDP",
                "XGMI_WAFL",
                "DF",
                "SMN",
                "SEM",
                "MP0",
                "MP1",
                "FUSE",
                // error missing in example
                "MCA",
                "VCN",
                "JPEG",
                "IH",
                "MPIO",
            ];
            let status_names = [
                "NONE", "DISABLED", "PARITY", "SING_C", "MULT_UC", "POISON", "ENABLED",
            ];

            println!("  Output of amdsmi_get_gpu_ras_block_features_enabled:");

            let mut index = 0;
            let mut block = amdsmi_gpu_block_t_AMDSMI_GPU_BLOCK_FIRST;

            loop {
                println!("      Block ({}): {}", block, block_names[index]);

                match amd_smi.get_gpu_ras_block_features_enabled(processor_handle, block) {
                    Ok(state) => {
                        println!("      Status ({}): {}", block, status_names[state as usize]);
                    }
                    Err(e) => {
                        println!("      {:?}", e);
                    }
                };

                if block >= amdsmi_gpu_block_t_AMDSMI_GPU_BLOCK_LAST {
                    break;
                }

                block *= 2;
                index += 1;
            }

            // Get bad pages
            // let bad_page_status_names = vec![
            //     "RESERVED",
            //     "PENDING",
            //     "UNRESERVABLE"
            // ];

            // for RADEON RX 6500XT / 6600
            // ret = amdsmi_status_t_AMDSMI_STATUS_NOT_SUPPORTED

            println!("  Output of amdsmi_get_gpu_bad_page_info:",);

            match amd_smi.get_gpu_bad_page_info(processor_handle) {
                Ok(_bad_page_info) => {
                    // ...
                }
                Err(e) => {
                    println!("      {:?}", e);
                }
            };

            // Get ECC error counts
            let err_cnt_info = amd_smi.get_gpu_total_ecc_count(processor_handle).unwrap();

            println!("  Output of get_gpu_total_ecc_count:");

            println!(
                "      Correctable errors: {}",
                err_cnt_info.correctable_count
            );
            println!(
                "      Uncorrectable errors: {}",
                err_cnt_info.uncorrectable_count
            );

            // Get process list

            let process_info_list = amd_smi.get_gpu_process_list(processor_handle).unwrap();
            let num_process = process_info_list.len();

            println!(
                "  Output of amdsmi_get_gpu_process_list: (num_process = {})",
                num_process
            );

            if num_process == 0 {
                println!("       No processes found.");
            } else {
                println!("       Processes found: {}", num_process);

                let mut mem = 0;
                let mut gtt_mem = 0;
                let mut cpu_mem = 0;
                let mut vram_mem = 0;
                let mut gfx = 0;
                let mut enc = 0;

                let bdf_str = unsafe {
                    format!(
                        "{:04}:{:02}:{:02}.{}",
                        bdf.__bindgen_anon_1.domain_number(),
                        bdf.__bindgen_anon_1.bus_number(),
                        bdf.__bindgen_anon_1.device_number(),
                        bdf.__bindgen_anon_1.function_number(),
                    )
                };

                // println!("bdf_str: {}", bdf_str);

                println!("       Allocation size for process list: {}", num_process);
                println!();

                for proc in process_info_list.iter() {
                    println!(
                        "       *Process id: {} / Name: {} / VRAM: {:?}",
                        proc.pid,
                        buf_i8_to_string(&proc.name).unwrap(),
                        proc.memory_usage.vram_mem
                    );
                }

                println!();

                // FIXME print table
                println!("+=======+==================+============+==============+=============+=============+=============+==============+=========================================+");
                println!("| pid   | name             | user       | gpu bdf      | fb usage    | gtt memory  | cpu memory  | vram memory  | engine usage (ns)                       |");
                println!("|       |                  |            |              |             |             |             |              | gfx     enc                             |");
                println!("+=======+==================+============+==============+=============+=============+=============+==============+=================+=======================+");

                for proc in process_info_list {
                    println!(
                        "| {} | {} | {} | {} | {} KiB | {} KiB | {} KiB | {} KiB  | {}  {} |",
                        proc.pid,
                        buf_i8_to_string(&proc.name).unwrap(),
                        0,
                        bdf_str,
                        proc.mem / 1024,
                        proc.memory_usage.gtt_mem / 1024,
                        proc.memory_usage.cpu_mem / 1024,
                        proc.memory_usage.vram_mem / 1024,
                        proc.engine_usage.gfx,
                        proc.engine_usage.enc
                    );

                    // println!(
                    //     "| {} | {} | {} | {} | {} KiB | {} KiB | {} KiB | {} KiB  | {}  {} |",
                    //     proc.pid,
                    //     buf_i8_to_string(&proc.name),
                    //     " ",
                    //     bdf_str,
                    //     proc.mem / 1024,
                    //     proc.memory_usage.gtt_mem / 1024,
                    //     proc.memory_usage.cpu_mem / 1024,
                    //     proc.memory_usage.vram_mem / 1024,
                    //     proc.engine_usage.gfx,
                    //     proc.engine_usage.enc
                    // );

                    mem += proc.mem / 1024;
                    gtt_mem += proc.memory_usage.gtt_mem / 1024;
                    cpu_mem += proc.memory_usage.cpu_mem / 1024;
                    vram_mem += proc.memory_usage.vram_mem / 1024;
                    gfx = proc.engine_usage.gfx;
                    enc = proc.engine_usage.enc;
                }

                println!("+=======+==================+============+==============+=============+=============+=============+==============+=================+=======================+");
                println!(
                    "|                                 TOTAL:| {} | {} KiB | {} KiB | {} KiB | {} KiB | {}  {}  |",
                    bdf_str,
                    mem,
                    gtt_mem,
                    cpu_mem,
                    vram_mem,
                    gfx,
                    enc
                );
                println!("+=======+==================+============+==============+=============+=============+=============+==============+=================+=======================+");
            }

            // Get device name
            let board_info = amd_smi.get_gpu_board_info(processor_handle).unwrap();

            println!("  Output of amdsmi_get_gpu_board_info:");

            println!("      device[{}]", j);
            println!(
                "          Product name: {}",
                buf_i8_to_string(&board_info.product_name).unwrap()
            );
            println!(
                "          Model Number: {}",
                buf_i8_to_string(&board_info.model_number).unwrap()
            );
            println!(
                "          Board Serial: {}",
                buf_i8_to_string(&board_info.product_serial).unwrap()
            );
            println!(
                "          Manufacturer Name: {}",
                buf_i8_to_string(&board_info.manufacturer_name).unwrap()
            );

            // Get temperature
            let val_i64 = amd_smi
                .get_temp_metric(
                    processor_handle,
                    amdsmi_temperature_type_t_AMDSMI_TEMPERATURE_TYPE_EDGE,
                    amdsmi_temperature_metric_t_AMDSMI_TEMP_CURRENT,
                )
                .unwrap();

            println!("  Output of amdsmi_get_temp_metric:");

            println!("      Temperature: {:?} C", val_i64);

            // Get frame buffer
            let vram_usage = amd_smi.get_gpu_vram_usage(processor_handle).unwrap();

            println!("  Output of amdsmi_get_gpu_vram_usage:");

            println!("      Frame buffer usage (MB): {:?}", vram_usage.vram_used);
            println!("      VRAM TOTAL (MB): {:?}", vram_usage.vram_total);

            let cap_info = amd_smi.get_power_cap_info(processor_handle, 0).unwrap();

            println!("  Output of amdsmi_get_power_cap_info:");

            println!("      Power Cap: {:?} W", cap_info.power_cap / 1000000);
            println!(
                "      Default Power Cap: {:?} W",
                cap_info.default_power_cap / 1000000
            );
            println!("      Dpm Cap: {:?} W", cap_info.dpm_cap / 1000000);
            println!(
                "      Min Power Cap: {:?} W",
                cap_info.min_power_cap / 1000000
            );
            println!(
                "      Max Power Cap: {:?} W",
                cap_info.max_power_cap / 1000000
            );

            // Get GPU Metrics info
            println!("  Output of amdsmi_get_gpu_metrics_info:");

            match amd_smi.get_gpu_metrics_info(processor_handle) {
                Ok(smu) => {
                    unsafe {
                        println!(
                            "   Device [{}] BDF {:04}:{:02}:{:02}.{}",
                            i,
                            bdf.__bindgen_anon_1.domain_number(),
                            bdf.__bindgen_anon_1.bus_number(),
                            bdf.__bindgen_anon_1.device_number(),
                            bdf.__bindgen_anon_1.function_number(),
                        );
                    }

                    println!("      METRIC TABLE HEADER:");

                    println!("      structure_size: {}", smu.common_header.structure_size);
                    println!(
                        "      format_revision: {}",
                        smu.common_header.format_revision
                    );
                    println!(
                        "      content_revision: {}",
                        smu.common_header.content_revision
                    );
                    println!();

                    println!("      TIME STAMPS (ns):");

                    println!("      system_clock_counter: {}", smu.system_clock_counter);
                    println!(
                        "      firmware_timestamp (10ns resolution)=: {}",
                        smu.firmware_timestamp
                    );
                    println!();

                    println!("      TEMPERATURES (C):");

                    println!("      temperature_edge: {}", smu.temperature_edge);
                    println!("      temperature_hotspot: {}", smu.temperature_hotspot);
                    println!("      temperature_mem: {}", smu.temperature_mem);
                    println!("      temperature_vrgfx: {}", smu.temperature_vrgfx);
                    println!("      temperature_vrsoc: {}", smu.temperature_vrsoc);
                    println!("      temperature_vrmem: {}", smu.temperature_vrmem);
                    println!("      temperature_hbm: {:?}", smu.temperature_hbm);
                    println!();

                    println!("      UTILIZATION (%):");

                    println!("      average_gfx_activity: {}", smu.average_gfx_activity);
                    println!("      average_umc_activity: {}", smu.average_umc_activity);
                    println!("      average_mm_activity: {}", smu.average_mm_activity);
                    println!("      vcn_activity: {:?}", smu.vcn_activity);
                    println!("      jpeg_activity: {:?}", smu.jpeg_activity);
                    println!();

                    println!("      POWER (W)/ENERGY (15.259uJ per 1ns):");

                    println!("      average_socket_power: {}", smu.average_socket_power);
                    println!("      current_socket_power: {}", smu.current_socket_power);
                    println!("      energy_accumulator: {}", smu.energy_accumulator);
                    println!();

                    println!("      AVG CLOCKS (MHz):");

                    println!(
                        "      average_gfxclk_frequency: {}",
                        smu.average_gfxclk_frequency
                    );
                    println!(
                        "      average_uclk_frequency: {}",
                        smu.average_uclk_frequency
                    );
                    println!(
                        "      average_vclk0_frequency: {}",
                        smu.average_vclk0_frequency
                    );
                    println!(
                        "      average_vclk1_frequency: {}",
                        smu.average_vclk1_frequency
                    );
                    println!();

                    println!("      CURRENT CLOCKS (MHz):");

                    println!("      current_gfxclk: {}", smu.current_gfxclk);
                    println!("      current_gfxclks: {:?}", smu.current_gfxclks);

                    println!("      current_socclk: {}", smu.current_socclk);
                    println!("      current_socclks: {:?}", smu.current_socclks);

                    println!("      current_uclk: {}", smu.current_uclk);

                    println!("      current_vclk0: {}", smu.current_vclk0);
                    println!("      current_vclk0s: {:?}", smu.current_vclk0s);

                    println!("      current_dclk0: {}", smu.current_dclk0);
                    println!("      current_dclk0s: {:?}", smu.current_dclk0s);

                    println!("      current_vclk1: {}", smu.current_vclk1);
                    println!("      current_dclk1: {}", smu.current_dclk1);

                    println!();

                    println!("      TROTTLE STATUS:");

                    println!("      throttle_status: {}", smu.throttle_status);
                    println!();

                    println!("      FAN SPEED:");

                    println!("      current_fan_speed: {}", smu.current_fan_speed);
                    println!();

                    println!("      LINK WIDTH (number of lanes) /SPEED (0.1 GT/s):");

                    println!("      pcie_link_width: {}", smu.pcie_link_width);
                    println!("      pcie_link_speed: {}", smu.pcie_link_speed);
                    println!("      xgmi_link_width: {}", smu.xgmi_link_width);
                    println!("      xgmi_link_speed: {}", smu.xgmi_link_speed);
                    println!();

                    println!("      Utilization Accumulated(%):");

                    println!("      gfx_activity_acc: {}", smu.gfx_activity_acc);
                    println!("      mem_activity_acc: {}", smu.mem_activity_acc);
                    println!();

                    println!("      XGMI ACCUMULATED DATA TRANSFER SIZE (KB):");

                    println!("      xgmi_read_data_acc: {:?}", smu.xgmi_read_data_acc);
                    println!("      xgmi_write_data_acc: {:?}", smu.xgmi_write_data_acc);

                    // Voltage (mV)
                    println!("      Voltage (mV):");

                    println!("      voltage_soc: {}", smu.voltage_soc);
                    println!("      voltage_gfx: {}", smu.voltage_gfx);
                    println!("      voltage_mem: {}", smu.voltage_mem);

                    println!("      indep_throttle_status: {}", smu.indep_throttle_status);

                    // Clock Lock Status. Each bit corresponds to clock instance
                    println!(
                        "      gfxclk_lock_status (in hex): {}",
                        smu.gfxclk_lock_status
                    );

                    // Bandwidth (GB/sec)
                    println!("      pcie_bandwidth_acc: {}", smu.pcie_bandwidth_acc);
                    println!("      pcie_bandwidth_inst: {}", smu.pcie_bandwidth_inst);

                    // Counts
                    println!(
                        "      pcie_l0_to_recov_count_acc: {}",
                        smu.pcie_l0_to_recov_count_acc
                    );
                    println!("      pcie_replay_count_acc: {}", smu.pcie_replay_count_acc);
                    println!(
                        "      pcie_replay_rover_count_acc: {}",
                        smu.pcie_replay_rover_count_acc
                    );
                    println!(
                        "      pcie_nak_sent_count_acc: {}",
                        smu.pcie_nak_sent_count_acc
                    );
                    println!(
                        "      pcie_nak_rcvd_count_acc: {}",
                        smu.pcie_nak_rcvd_count_acc
                    );

                    println!();

                    // Accumulation cycle counter
                    // Accumulated throttler residencies
                    println!("      RESIDENCY ACCUMULATION / COUNTER:");

                    println!("      accumulation_counter: {}", smu.accumulation_counter);
                    println!("      prochot_residency_acc: {}", smu.prochot_residency_acc);
                    println!("      ppt_residency_acc: {}", smu.ppt_residency_acc);
                    println!(
                        "      socket_thm_residency_acc: {}",
                        smu.socket_thm_residency_acc
                    );
                    println!("      vr_thm_residency_acc: {}", smu.vr_thm_residency_acc);
                    println!("      hbm_thm_residency_acc: {}", smu.hbm_thm_residency_acc);

                    // Number of current partitions
                    println!("      num_partition: {}", smu.num_partition);
                    // PCIE other end recovery counter
                    println!(
                        "      pcie_lc_perf_other_end_recovery: {}",
                        smu.pcie_lc_perf_other_end_recovery
                    );

                    println!("      xcp_stats: {:#?}", smu.xcp_stats);

                    println!();

                    println!("      ** -> Checking metrics with constant changes ** :");

                    let k_max_iter_test = 10;

                    for idx in 0..k_max_iter_test {
                        let gpu_metrics_check =
                            amd_smi.get_gpu_metrics_info(processor_handle).unwrap();
                        println!(
                            "      -> firmware_timestamp [{} / {}]: {}",
                            idx, k_max_iter_test, gpu_metrics_check.firmware_timestamp
                        );
                    }

                    println!();

                    for idx in 0..k_max_iter_test {
                        let gpu_metrics_check =
                            amd_smi.get_gpu_metrics_info(processor_handle).unwrap();
                        println!(
                            "      -> system_clock_counter [{} / {}]: {}",
                            idx, k_max_iter_test, gpu_metrics_check.system_clock_counter
                        );
                    }

                    println!("  ** Note: Values MAX'ed out (UINTX MAX are unsupported for the version in question) **");
                    println!();
                }
                Err(e) => {
                    println!("      {:?}", e);
                }
            }

            // error
            // Get nearest GPUs
            // let topology_link_type_str = [
            //     "AMDSMI_LINK_TYPE_INTERNAL",
            //     "AMDSMI_LINK_TYPE_XGMI",
            //     "AMDSMI_LINK_TYPE_PCIE",
            //     "AMDSMI_LINK_TYPE_NOT_APPLICABLE",
            //     "AMDSMI_LINK_TYPE_UNKNOWN",
            // ];
            //
            // println!("  Output of amdsmi_get_link_topology_nearest:");
            //
            // for topo_link_type in amdsmi_link_type_t_AMDSMI_LINK_TYPE_INTERNAL
            //     ..amdsmi_link_type_t_AMDSMI_LINK_TYPE_UNKNOWN
            // {
            //     let topology_nearest_info = amd_smi
            //         .get_link_topology_nearest(processor_handle, topo_link_type)
            //         .unwrap();
            //
            //     println!(
            //         "      Nearest GPUs found at {}",
            //         topology_link_type_str[topo_link_type as usize],
            //     );
            //     println!("      Nearest Count: {}", topology_nearest_info.count);
            // }

            println!();
        }
    }

    // Clean up resources allocated at amdsmi_init. It will invalidate sockets and devices pointers
    // * is done inside the AmdSmi drop trait
}
