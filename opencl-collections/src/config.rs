use std::fmt::Debug;

pub const DEFAULT_DEVICE_INDEX: usize = 0;

// TODO configure rust log
pub const DEBUG_MODE: bool = false;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClType {
    U8,
    U16,
    U32,
    U64,
    I16,
    I32,
    I64,
}

impl ClType {
    pub fn to_cl_type_name(&self) -> &str {
        match self {
            ClType::U8 => "uchar",
            ClType::U16 => "ushort",
            ClType::U32 => "uint",
            ClType::U64 => "ulong",
            ClType::I16 => "short",
            ClType::I32 => "int",
            ClType::I64 => "long",
        }
    }

    /// TODO explain ...
    pub fn cl_default(&self) -> i32 {
        match self {
            ClType::U8 => 0,
            ClType::U16 => 0,
            ClType::U32 => 0,
            ClType::U64 => 0,
            ClType::I16 => -1,
            ClType::I32 => -1,
            ClType::I64 => -1,
        }
    }
}

pub trait ClTypeDefault {
    fn cl_default() -> Self;
    fn cl_enum() -> ClType;
}

impl ClTypeDefault for u8 {
    fn cl_default() -> u8 {
        ClType::U8.cl_default() as u8
    }

    fn cl_enum() -> ClType {
        ClType::U8
    }
}

impl ClTypeDefault for u16 {
    fn cl_default() -> u16 {
        ClType::U16.cl_default() as u16
    }

    fn cl_enum() -> ClType {
        ClType::U16
    }
}

impl ClTypeDefault for u32 {
    fn cl_default() -> u32 {
        ClType::U32.cl_default() as u32
    }

    fn cl_enum() -> ClType {
        ClType::U32
    }
}

impl ClTypeDefault for u64 {
    fn cl_default() -> u64 {
        ClType::U64.cl_default() as u64
    }

    fn cl_enum() -> ClType {
        ClType::U64
    }
}

impl ClTypeDefault for i16 {
    fn cl_default() -> i16 {
        ClType::I16.cl_default() as i16
    }

    fn cl_enum() -> ClType {
        ClType::I16
    }
}

impl ClTypeDefault for i32 {
    fn cl_default() -> i32 {
        ClType::I32.cl_default()
    }

    fn cl_enum() -> ClType {
        ClType::I32
    }
}

impl ClTypeDefault for i64 {
    fn cl_default() -> i64 {
        ClType::I64.cl_default() as i64
    }

    fn cl_enum() -> ClType {
        ClType::I64
    }
}

// + PartialOrd ???
pub trait ClTypeTrait:
    Copy + Clone + Default + ClTypeDefault + Debug + PartialEq + Send + Sync
{
}

impl<T: Copy + Clone + Default + ClTypeDefault + Debug + PartialEq + Send + Sync> ClTypeTrait
    for T
{
}

#[cfg(test)]
mod cl_default_tests {
    use crate::config::ClTypeDefault;

    #[test]
    fn cl_default_u8() {
        let d_1 = u8::default();
        let d_2 = u8::cl_default();

        assert_eq!(d_1, 0);
        assert_eq!(d_2, 0);
    }

    #[test]
    fn cl_default_i16() {
        let d_1 = i16::default();
        let d_2 = i16::cl_default();

        assert_eq!(d_1, 0);
        assert_eq!(d_2, -1);
    }
}
