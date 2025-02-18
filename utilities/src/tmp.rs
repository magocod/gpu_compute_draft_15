#[repr(C)]
#[derive(Debug)]
pub struct SharedValues {
    pub arr: [i32; 8],
    pub index: i32,
}
