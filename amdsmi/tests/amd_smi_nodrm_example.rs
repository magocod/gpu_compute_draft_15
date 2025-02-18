// https://github.com/ROCm/amdsmi/blob/amd-staging/example/amd_smi_nodrm_example.cc
// resources/amd_smi_nodrm_example.cc

use amdsmi::AmdSmi;
use amdsmi_sys::bindings::{
    amdsmi_fw_block_t_AMDSMI_FW_ID_CP_CE, amdsmi_fw_block_t_AMDSMI_FW_ID_CP_ME,
    amdsmi_fw_block_t_AMDSMI_FW_ID_CP_MEC1, amdsmi_fw_block_t_AMDSMI_FW_ID_CP_MEC2,
    amdsmi_fw_block_t_AMDSMI_FW_ID_CP_PFP, amdsmi_fw_block_t_AMDSMI_FW_ID_MC,
    amdsmi_fw_block_t_AMDSMI_FW_ID_PM, amdsmi_fw_block_t_AMDSMI_FW_ID_PSP_SOSDRV,
    amdsmi_fw_block_t_AMDSMI_FW_ID_RLC, amdsmi_fw_block_t_AMDSMI_FW_ID_RLC_RESTORE_LIST_CNTL,
    amdsmi_fw_block_t_AMDSMI_FW_ID_RLC_RESTORE_LIST_GPM_MEM,
    amdsmi_fw_block_t_AMDSMI_FW_ID_RLC_RESTORE_LIST_SRM_MEM, amdsmi_fw_block_t_AMDSMI_FW_ID_SDMA0,
    amdsmi_fw_block_t_AMDSMI_FW_ID_SMU, amdsmi_fw_block_t_AMDSMI_FW_ID_VCN,
    amdsmi_temperature_metric_t_AMDSMI_TEMP_CURRENT,
    amdsmi_temperature_type_t_AMDSMI_TEMPERATURE_TYPE_EDGE,
    amdsmi_temperature_type_t_AMDSMI_TEMPERATURE_TYPE_HOTSPOT,
    amdsmi_temperature_type_t_AMDSMI_TEMPERATURE_TYPE_PLX,
    amdsmi_temperature_type_t_AMDSMI_TEMPERATURE_TYPE_VRAM,
    amdsmi_temperature_type_t_AMDSMI_TEMPERATURE_TYPE__MAX,
    processor_type_t_AMDSMI_PROCESSOR_TYPE_AMD_GPU,
};
use utilities::helper_functions::buf_i8_to_string;

#[test]
fn test_amd_smi_nodrm_example() {
    let amd_smi = AmdSmi::new().unwrap();

    // Init amdsmi for sockets and devices.
    // Here we are only interested in AMD_GPUS.
    // * is done inside the AmdSmi::new()

    // Get all sockets

    // Get the socket handles in the system
    let sockets = amd_smi.get_socket_handles().unwrap();

    println!("Total Socket: {}", sockets.len());
    println!();

    // For each socket, get identifier and devices
    for (i, socket_handle) in sockets.into_iter().enumerate() {
        // Get Socket info
        let socket_info = amd_smi.get_socket_info(socket_handle).unwrap();

        println!("Socket: {:?}", buf_i8_to_string(&socket_info));

        // Get all devices of the socket
        let processor_handles = amd_smi.get_processor_handles(socket_handle).unwrap();

        println!("device_count: {}", processor_handles.len());

        // For each device of the socket, get name and temperature.
        for (j, processor_handle) in processor_handles.into_iter().enumerate() {
            // Get device type. Since the amdsmi is initialized with
            // AMD_SMI_INIT_AMD_GPUS, the processor_type must be AMDSMI_PROCESSOR_TYPE_AMD_GPU.
            let processor_type = amd_smi.get_processor_type(processor_handle).unwrap();

            println!("  processor_type: {:?}", processor_type);

            if processor_type != processor_type_t_AMDSMI_PROCESSOR_TYPE_AMD_GPU {
                panic!("Expect AMDSMI_PROCESSOR_TYPE_AMD_GPU device type!")
            }

            let r = amd_smi.get_gpu_ras_feature_info(processor_handle);

            match r {
                Ok(ras_feature) => {
                    println!(
                        "  ras_feature: version: {}, schema: {}",
                        ras_feature.ras_eeprom_version, ras_feature.ecc_correction_schema_flag
                    );
                }
                Err(e) => {
                    println!("  {:?}", e)
                }
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

            println!("      Firmware version: {}", fw_information.num_fw_info);
            println!(
                "      SMU: {:?}",
                fw_information.fw_info_list[amdsmi_fw_block_t_AMDSMI_FW_ID_SMU as usize].fw_version
            );
            println!(
                "      PM: {:?}",
                fw_information.fw_info_list[amdsmi_fw_block_t_AMDSMI_FW_ID_PM as usize].fw_version
            );
            println!(
                "      VCN: {:?}",
                fw_information.fw_info_list[amdsmi_fw_block_t_AMDSMI_FW_ID_VCN as usize].fw_version
            );
            println!(
                "      CP_ME: {:?}",
                fw_information.fw_info_list[amdsmi_fw_block_t_AMDSMI_FW_ID_CP_ME as usize]
                    .fw_version
            );
            println!(
                "      CP_PFP: {:?}",
                fw_information.fw_info_list[amdsmi_fw_block_t_AMDSMI_FW_ID_CP_PFP as usize]
                    .fw_version
            );
            println!(
                "      CP_CE: {:?}",
                fw_information.fw_info_list[amdsmi_fw_block_t_AMDSMI_FW_ID_CP_CE as usize]
                    .fw_version
            );
            println!(
                "      RLC: {:?}",
                fw_information.fw_info_list[amdsmi_fw_block_t_AMDSMI_FW_ID_RLC as usize].fw_version
            );
            println!(
                "      CP_MEC1: {:?}",
                fw_information.fw_info_list[amdsmi_fw_block_t_AMDSMI_FW_ID_CP_MEC1 as usize]
                    .fw_version
            );
            println!(
                "      CP_MEC2: {:?}",
                fw_information.fw_info_list[amdsmi_fw_block_t_AMDSMI_FW_ID_CP_MEC2 as usize]
                    .fw_version
            );
            println!(
                "      SDMA0: {:?}",
                fw_information.fw_info_list[amdsmi_fw_block_t_AMDSMI_FW_ID_SDMA0 as usize]
                    .fw_version
            );
            println!(
                "      MC: {:?}",
                fw_information.fw_info_list[amdsmi_fw_block_t_AMDSMI_FW_ID_MC as usize].fw_version
            );
            println!(
                "      RLC RESTORE LIST CNTL: {:?}",
                fw_information.fw_info_list
                    [amdsmi_fw_block_t_AMDSMI_FW_ID_RLC_RESTORE_LIST_CNTL as usize]
                    .fw_version
            );
            println!(
                "      RLC RESTORE LIST GPM MEM: {:?}",
                fw_information.fw_info_list
                    [amdsmi_fw_block_t_AMDSMI_FW_ID_RLC_RESTORE_LIST_GPM_MEM as usize]
                    .fw_version
            );
            println!(
                "      RLC RESTORE LIST SRM MEM: {:?}",
                fw_information.fw_info_list
                    [amdsmi_fw_block_t_AMDSMI_FW_ID_RLC_RESTORE_LIST_SRM_MEM as usize]
                    .fw_version
            );
            println!(
                "      PSP SOSDRV: {:?}",
                fw_information.fw_info_list[amdsmi_fw_block_t_AMDSMI_FW_ID_PSP_SOSDRV as usize]
                    .fw_version
            );

            // Get temperature measurements
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

            // Get bad pages
            // let bad_page_status_names = vec![
            //     "RESERVED",
            //     "PENDING",
            //     "UNRESERVABLE"
            // ];

            // for RADEON RX 6500XT / 6600
            // ret = amdsmi_status_t_AMDSMI_STATUS_NOT_SUPPORTED

            println!("  Output of amdsmi_get_gpu_bad_page_info:");

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
            println!(
                "      Min Power Cap: {:?} W",
                cap_info.min_power_cap / 1000000
            );
            println!(
                "      Max Power Cap: {:?} W",
                cap_info.max_power_cap / 1000000
            );
            println!("      Dpm Cap: {:?} W", cap_info.dpm_cap / 1000000);

            println!("  Output of amdsmi_get_soc_pstate:");
            match amd_smi.get_soc_pstate(processor_handle) {
                Ok(_policy) => {
                    // ...
                }
                Err(e) => {
                    println!("      {:?}", e);
                }
            };

            // for RADEON RX 6500XT / 6600
            // amdsmi_status_t_AMDSMI_STATUS_NOT_SUPPORTED

            // Get nearest GPUs
            let _topology_link_type_str = [
                "AMDSMI_LINK_TYPE_INTERNAL",
                "AMDSMI_LINK_TYPE_XGMI",
                "AMDSMI_LINK_TYPE_PCIE",
                "AMDSMI_LINK_TYPE_NOT_APPLICABLE",
                "AMDSMI_LINK_TYPE_UNKNOWN",
            ];

            println!("  Output of amdsmi_get_link_topology_nearest:");

            // error
            // for topo_link_type in amdsmi_link_type_t_AMDSMI_LINK_TYPE_INTERNAL
            //     ..amdsmi_link_type_t_AMDSMI_LINK_TYPE_UNKNOWN
            // {
            //     let topology_nearest_info = amd_smi
            //         .get_link_topology_nearest(processor_handle, topo_link_type)
            //         .unwrap();
            //
            //     println!(
            //         "      Nearest GPUs found at {} - {:#?}",
            //         topology_link_type_str[topo_link_type as usize], topology_nearest_info.count
            //     );
            // }

            println!();
        }
    }

    // Clean up resources allocated at amdsmi_init. It will invalidate sockets and devices pointers
    // * is done inside the AmdSmi drop trait
}
