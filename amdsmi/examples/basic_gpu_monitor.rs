use amdsmi::utils::get_amd_smi_vram_type;
use amdsmi::AmdSmi;
use amdsmi_sys::bindings::{amdsmi_processor_handle, amdsmi_socket_handle};
use std::thread;
use std::time::Duration;
use utilities::helper_functions::buf_i8_to_string;
// simple example of a basic gpu monitor, to keep it as simple as possible, all errors are directly panic

// indicate the gpu index here, check this section of the amdsmi lib documentation (there is always one distinct GPU for a socket)
// https://rocm.docs.amd.com/projects/amdsmi/en/latest/how-to/using-amdsmi-for-C%2B%2B.html#device-socket-handles
pub const SOCKET_HANDLE_INDEX: usize = 0;
// always first device (GPU)
pub const DEVICE_HANDLE_INDEX: usize = 0;

struct GpuMonitor {
    amd_smi: AmdSmi,
    socket_handle: amdsmi_socket_handle,
    device_handle: amdsmi_processor_handle,
}

impl GpuMonitor {
    pub fn new() -> Self {
        let amd_smi = AmdSmi::new().unwrap();

        // Get all sockets
        let sockets = amd_smi.get_socket_handles().unwrap();

        if sockets.is_empty() {
            panic!("no gpu socket found");
        }

        let socket_handle = sockets[SOCKET_HANDLE_INDEX];

        // Get all devices of the socket
        let device_handles = amd_smi.get_processor_handles(socket_handle).unwrap();

        if device_handles.is_empty() {
            panic!("no gpu device found");
        }

        let device_handle = device_handles[DEVICE_HANDLE_INDEX];

        Self {
            amd_smi,
            socket_handle,
            device_handle,
        }
    }

    pub fn info(&self) {
        println!("GPU INFO: ");

        let socket_info = self.amd_smi.get_socket_info(self.socket_handle).unwrap();
        println!("  Socket: {}", buf_i8_to_string(&socket_info).unwrap());

        let board_info = self.amd_smi.get_gpu_board_info(self.device_handle).unwrap();
        println!(
            "  Name: {}",
            buf_i8_to_string(&board_info.product_name).unwrap()
        );

        let vram_info = self.amd_smi.get_gpu_vram_info(self.device_handle).unwrap();
        println!("  Memory Size: {} MB", vram_info.vram_size);
        println!(
            "  Memory Type: {:?}",
            get_amd_smi_vram_type(vram_info.vram_type)
        );
        println!("  Memory Bus:  {} bit", vram_info.vram_bit_width);

        println!();
    }

    pub fn processes(&self) {
        let process_info_list = self
            .amd_smi
            .get_gpu_process_list(self.device_handle)
            .unwrap();

        println!("PROCESSES:");
        println!();

        println!(
            "| {0: <20} | {1: <20} | {2: <20} | {3: <20} | {4: <20} | {5: <20} | {6: <20} |",
            "Name", "ID", "GFX ", "Mem (MB)", "GTT Mem (MB)", "Cpu Mem (MB)", "Vram (MB)"
        );
        // println!(
        //     "| {0: <20} | {1: <20} | {2: <20} | {3: <20} | {4: <20} | {5: <20} | {6: <20} |",
        //     "Name", "0", "0 ", "0", "0", "0", "0"
        // );

        let mb = 1024 * 1024;

        for process_info in process_info_list.iter() {
            println!(
                "| {0: <20} | {1: <20} | {2: <20} | {3: <20} | {4: <20} | {5: <20} | {6: <20} |",
                buf_i8_to_string(&process_info.name).unwrap(),
                process_info.pid,
                process_info.engine_usage.gfx,
                process_info.mem / mb,
                process_info.memory_usage.gtt_mem / mb,
                process_info.memory_usage.cpu_mem / mb,
                process_info.memory_usage.vram_mem / mb
            );
        }

        println!();
    }

    pub fn metrics(&self) {
        println!("SENSORS:");
        println!();

        match self.amd_smi.get_gpu_metrics_info(self.device_handle) {
            Ok(smu) => {
                let vram_usage = self.amd_smi.get_gpu_vram_usage(self.device_handle).unwrap();

                println!("  Memory Usage (MB):");
                println!(
                    "   VRAM: [ {0: <20} / {1: <20} ]",
                    vram_usage.vram_used, vram_usage.vram_total
                );
                println!();

                println!("  TEMPERATURES (C):");
                println!("      temperature_edge: {}", smu.temperature_edge);
                println!("      temperature_hotspot: {}", smu.temperature_hotspot);
                println!("      temperature_mem: {}", smu.temperature_mem);
                println!();

                println!("  UTILIZATION (%):");
                println!("      average_gfx_activity: {}", smu.average_gfx_activity);
                println!("      average_umc_activity: {}", smu.average_umc_activity);
                println!("      average_mm_activity: {}", smu.average_mm_activity);
                println!();

                println!("  POWER (W):");
                println!("      average_socket_power: {}", smu.average_socket_power);
                println!("      current_socket_power: {}", smu.current_socket_power);
                println!();

                println!("system_clock_counter [{}]", smu.system_clock_counter);
                println!();
            }
            Err(e) => {
                println!("Error getting GPU metrics: {:?}", e);
                println!();
            }
        }
    }
}

pub fn main() {
    let gpu_monitor = GpuMonitor::new();
    gpu_monitor.info();

    loop {
        gpu_monitor.processes();
        gpu_monitor.metrics();

        println!();
        thread::sleep(Duration::from_secs(1));
    }
}
