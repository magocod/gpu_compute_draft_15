#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(improper_ctypes)]
#![allow(clippy::approx_constant)]
#![allow(clippy::useless_transmute)]
#![allow(clippy::unnecessary_cast)]
#[cfg(feature = "rocm_6_2_2")]
pub mod bindings;

#[cfg(test)]
mod tests {
    use crate::bindings::{
        amdsmi_board_info_t, amdsmi_get_gpu_board_info, amdsmi_get_lib_version,
        amdsmi_get_processor_handles, amdsmi_get_processor_type, amdsmi_get_socket_handles,
        amdsmi_get_socket_info, amdsmi_get_temp_metric, amdsmi_init,
        amdsmi_init_flags_t_AMDSMI_INIT_AMD_GPUS, amdsmi_processor_handle, amdsmi_shut_down,
        amdsmi_socket_handle, amdsmi_status_t_AMDSMI_STATUS_SUCCESS,
        amdsmi_temperature_metric_t_AMDSMI_TEMP_CURRENT,
        amdsmi_temperature_type_t_AMDSMI_TEMPERATURE_TYPE_EDGE, amdsmi_version_t, processor_type_t,
        processor_type_t_AMDSMI_PROCESSOR_TYPE_AMD_GPU,
        processor_type_t_AMDSMI_PROCESSOR_TYPE_UNKNOWN,
    };
    use utilities::helper_functions::buf_i8_to_string;

    #[test]
    fn test_check_version() {
        unsafe {
            let ret = amdsmi_init(amdsmi_init_flags_t_AMDSMI_INIT_AMD_GPUS as u64);
            assert_eq!(ret, amdsmi_status_t_AMDSMI_STATUS_SUCCESS);

            let mut version = std::mem::MaybeUninit::<amdsmi_version_t>::uninit();
            let ret = amdsmi_get_lib_version(version.as_mut_ptr());
            assert_eq!(ret, amdsmi_status_t_AMDSMI_STATUS_SUCCESS);

            let version = version.assume_init();
            println!("{:?}", version);

            let ret = amdsmi_shut_down();
            assert_eq!(ret, amdsmi_status_t_AMDSMI_STATUS_SUCCESS);
        }
    }

    // https://rocm.docs.amd.com/projects/amdsmi/en/latest/how-to/amdsmi-cpp-lib.html#hello-amd-smi
    // resources/hello_amdsmi.cpp
    // example Hello AMD SMI
    #[test]
    fn test_example_hello_amd_smi() {
        // Init amdsmi for sockets and devices. Here we are only interested in AMD_GPUS.
        let ret = unsafe { amdsmi_init(amdsmi_init_flags_t_AMDSMI_INIT_AMD_GPUS as u64) };
        assert_eq!(ret, amdsmi_status_t_AMDSMI_STATUS_SUCCESS);

        // Get all sockets
        let mut socket_count = 0;
        // Get the socket count available in the system.
        let ret = unsafe { amdsmi_get_socket_handles(&mut socket_count, std::ptr::null_mut()) };
        assert_eq!(ret, amdsmi_status_t_AMDSMI_STATUS_SUCCESS);

        // Allocate the memory for the sockets
        let mut sockets: Vec<amdsmi_socket_handle> =
            vec![std::ptr::null_mut(); socket_count as usize];

        // Get the socket handles in the system
        let ret = unsafe { amdsmi_get_socket_handles(&mut socket_count, sockets.as_mut_ptr()) };
        assert_eq!(ret, amdsmi_status_t_AMDSMI_STATUS_SUCCESS);

        println!("Total Socket: {}", socket_count);

        // For each socket, get identifier and devices
        for socket_handle in sockets {
            let mut socket_info: Vec<i8> = vec![i8::default(); 128];

            let ret =
                unsafe { amdsmi_get_socket_info(socket_handle, 128, socket_info.as_mut_ptr()) };

            assert_eq!(ret, amdsmi_status_t_AMDSMI_STATUS_SUCCESS);

            println!("Socket: {}", buf_i8_to_string(&socket_info).unwrap());

            // Get the device count for the socket.
            let mut device_count = 0;
            let ret = unsafe {
                amdsmi_get_processor_handles(socket_handle, &mut device_count, std::ptr::null_mut())
            };
            assert_eq!(ret, amdsmi_status_t_AMDSMI_STATUS_SUCCESS);

            // Allocate the memory for the device handlers on the socket
            let mut processor_handles: Vec<amdsmi_processor_handle> =
                vec![std::ptr::null_mut(); device_count as usize];

            // Get all devices of the socket
            let ret = unsafe {
                amdsmi_get_processor_handles(
                    socket_handle,
                    &mut device_count,
                    processor_handles.as_mut_ptr(),
                )
            };
            assert_eq!(ret, amdsmi_status_t_AMDSMI_STATUS_SUCCESS);

            // For each device of the socket, get name and temperature.
            for processor_handle in processor_handles {
                // Get device type. Since the amdsmi is initialized with
                // AMD_SMI_INIT_AMD_GPUS, the processor_type must be AMDSMI_PROCESSOR_TYPE_AMD_GPU.
                let mut processor_type: processor_type_t =
                    processor_type_t_AMDSMI_PROCESSOR_TYPE_UNKNOWN;

                let ret =
                    unsafe { amdsmi_get_processor_type(processor_handle, &mut processor_type) };
                assert_eq!(ret, amdsmi_status_t_AMDSMI_STATUS_SUCCESS);

                if processor_type != processor_type_t_AMDSMI_PROCESSOR_TYPE_AMD_GPU {
                    panic!("Expect AMDSMI_PROCESSOR_TYPE_AMD_GPU device type!")
                }

                // Get device name
                let mut board_info = std::mem::MaybeUninit::<amdsmi_board_info_t>::uninit();

                let ret =
                    unsafe { amdsmi_get_gpu_board_info(processor_handle, board_info.as_mut_ptr()) };
                assert_eq!(ret, amdsmi_status_t_AMDSMI_STATUS_SUCCESS);

                let board_info = unsafe { board_info.assume_init() };
                // println!("{:?}", board_info);

                println!(
                    "  device Name: {:?}",
                    buf_i8_to_string(&board_info.product_name)
                );

                // Get temperature
                let mut val_i64 = 0;
                let ret = unsafe {
                    amdsmi_get_temp_metric(
                        processor_handle,
                        amdsmi_temperature_type_t_AMDSMI_TEMPERATURE_TYPE_EDGE,
                        amdsmi_temperature_metric_t_AMDSMI_TEMP_CURRENT,
                        &mut val_i64,
                    )
                };
                assert_eq!(ret, amdsmi_status_t_AMDSMI_STATUS_SUCCESS);

                println!("  Temperature: {:?} C", val_i64);
            }
        }

        // Clean up resources allocated at amdsmi_init. It will invalidate sockets
        // and devices pointers
        let ret = unsafe { amdsmi_shut_down() };
        assert_eq!(ret, amdsmi_status_t_AMDSMI_STATUS_SUCCESS);
    }
}
