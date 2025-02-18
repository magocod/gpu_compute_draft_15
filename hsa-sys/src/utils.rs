use crate::bindings::{
    hsa_device_type_t, hsa_device_type_t_HSA_DEVICE_TYPE_CPU,
    hsa_device_type_t_HSA_DEVICE_TYPE_DSP, hsa_device_type_t_HSA_DEVICE_TYPE_GPU,
};

pub fn get_device_type_str(device_type: hsa_device_type_t) -> &'static str {
    match device_type {
        hsa_device_type_t_HSA_DEVICE_TYPE_CPU => "CPU",
        hsa_device_type_t_HSA_DEVICE_TYPE_GPU => "GPU",
        hsa_device_type_t_HSA_DEVICE_TYPE_DSP => "DSP",
        _ => "...",
    }
}
