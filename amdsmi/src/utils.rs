#![allow(non_upper_case_globals)]

use amdsmi_sys::bindings::{
    amdsmi_vram_type_t, amdsmi_vram_type_t_AMDSMI_VRAM_TYPE_DDR4,
    amdsmi_vram_type_t_AMDSMI_VRAM_TYPE_GDDR6,
};

#[derive(Debug, Copy, Clone)]
pub enum AmdSmiVramType {
    Unknown,
    DDR4,
    GDDR6,
}

pub fn get_amd_smi_vram_type(vram_type: amdsmi_vram_type_t) -> AmdSmiVramType {
    match vram_type {
        amdsmi_vram_type_t_AMDSMI_VRAM_TYPE_DDR4 => AmdSmiVramType::DDR4,
        amdsmi_vram_type_t_AMDSMI_VRAM_TYPE_GDDR6 => AmdSmiVramType::GDDR6,
        _ => AmdSmiVramType::Unknown,
    }
}
