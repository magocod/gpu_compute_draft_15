use opencl::wrapper::platform::Platform;

fn main() {
    let platform = Platform::first().unwrap();

    let devices = platform.get_gpu_devices().unwrap();

    for device in devices.iter() {
        println!("{:?}", device.info().unwrap());
    }
}
